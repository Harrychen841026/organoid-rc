"""
ENCODING:  input vector  ->  electrical stimulation pattern on one well.

Design
------
* The reservoir input for a well is a (q + 2l)-dim vector (its core sites plus
  the halo borrowed from neighbours). Each component is assigned to its own
  spatially separated stimulation cluster on the HD-MEA (so the spatial layout
  of the data is preserved -- analogous to the random input matrix Win in a
  classic reservoir, here a fixed spatial projection).
* Each scalar is normalised to [0, 1] (statistics estimated from the training
  trajectory) and mapped to a stimulus parameter:
      - "rate"      : number of biphasic pulses in the stim window  (robust,
                      recommended -- neurons rate-code well, amplitude is touchy)
      - "amplitude" : amplitude of a single biphasic pulse
* All pulses are CHARGE-BALANCED biphasic (cathodic-first) to protect the
  electrodes and the tissue. Keep charge/phase within your safety budget.

This file isolates every MaxLab Live API call so the rest of the codebase
stays hardware-agnostic. With no rig/API present it still imports, so the ESN
dry run works.
"""
import numpy as np

try:
    import maxlab
    import maxlab.chip
    import maxlab.system
    _HAVE_MAXLAB = True
except Exception:          # noqa
    _HAVE_MAXLAB = False


class InputNormalizer:
    """Per-dimension min/max from the training trajectory -> [0,1]."""
    def __init__(self):
        self.lo = None
        self.hi = None

    def fit(self, inputs: np.ndarray):          # (n_samples, in_dim)
        self.lo = inputs.min(0)
        self.hi = inputs.max(0)
        return self

    def to_unit(self, u: np.ndarray) -> np.ndarray:
        z = (np.asarray(u, float) - self.lo) / (self.hi - self.lo + 1e-9)
        return np.clip(z, 0.0, 1.0)


class StimEncoder:
    """
    Builds a MaxLab stimulation Sequence for one well from a unit-scaled input.

    well_stim_map[well_id] = list of stimulation-unit handles, one per input dim
    (set up once during hardware configuration; see hardware.py).
    """
    def __init__(self, cfg, normalizer: InputNormalizer, well_stim_map: dict):
        self.cfg = cfg
        self.norm = normalizer
        self.stim_map = well_stim_map
        hw = cfg.hw
        self.phase_samples = int(hw.pulse_phase_us * 1e-6 * hw.fs_hz)
        self.win_samples = int(cfg.timing.stim_window_ms * 1e-3 * hw.fs_hz)

    # ---- value -> stimulus parameter -------------------------------------
    def _n_pulses(self, unit_val: float) -> int:
        lo, hi = self.cfg.hw.pulse_rate_hz_range
        rate = lo + unit_val * (hi - lo)
        return max(1, int(rate * self.cfg.timing.stim_window_ms * 1e-3))

    def _amp_dac(self, unit_val: float) -> int:
        lo, hi = self.cfg.hw.pulse_amp_mV_range
        mv = lo + unit_val * (hi - lo)
        return int(round(mv / 2.9))          # ~2.9 mV per DAC LSB (verify!)

    # ---- main entry -------------------------------------------------------
    def encode(self, well_id: int, u: np.ndarray):
        """Return a maxlab.Sequence (hardware) or a plain spec dict (dry run)."""
        z = self.norm.to_unit(u)
        if not _HAVE_MAXLAB:
            # dry-run: return a human-readable description of the stimulus
            return {"well": well_id,
                    "mode": self.cfg.hw.encoding_mode,
                    "per_dim": [
                        {"dim": i,
                         "n_pulses": self._n_pulses(v),
                         "amp_dac": self._amp_dac(v)}
                        for i, v in enumerate(z)]}

        seq = maxlab.Sequence()
        stim_units = self.stim_map[well_id]
        if self.cfg.hw.encoding_mode == "rate":
            # interleave pulses across the window; per-dim pulse counts
            counts = [self._n_pulses(v) for v in z]
            slots = self.win_samples // (max(counts) + 1)
            for slot in range(max(counts)):
                for i, stim in enumerate(stim_units):
                    if slot < counts[i]:
                        self._append_biphasic(seq, stim,
                                              self._amp_dac(0.7))  # fixed amp
                seq.append(maxlab.system.DelaySamples(slots))
        else:  # amplitude coding: one pulse per dim, amplitude carries value
            for i, stim in enumerate(stim_units):
                self._append_biphasic(seq, stim, self._amp_dac(z[i]))
            seq.append(maxlab.system.DelaySamples(self.win_samples))
        return seq

    def _append_biphasic(self, seq, stim, amp_dac):
        """Charge-balanced cathodic-first biphasic pulse on one stim unit."""
        ps = self.phase_samples
        seq.append(stim.power_up(True))
        seq.append(maxlab.chip.DAC(0, 512 - amp_dac))   # cathodic phase
        seq.append(maxlab.system.DelaySamples(ps))
        seq.append(maxlab.chip.DAC(0, 512 + amp_dac))   # anodic phase
        seq.append(maxlab.system.DelaySamples(ps))
        seq.append(maxlab.chip.DAC(0, 512))             # return to baseline
        seq.append(stim.power_up(False))
