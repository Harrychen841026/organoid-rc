"""
DECODING:  evoked spikes  ->  reservoir state vector.

The reservoir state for a well is built from the post-stimulus spiking of its
readout electrodes. For each of N readout channels we compute a firing feature
in each of n_state_bins time bins across the response window:

    state = [ rate(ch, bin) for ch in channels for bin in bins ]   length N*B

Optionally smooth spike trains first. This binned firing-rate vector is the
high-dimensional, fixed nonlinear projection of the input -- the analogue of
the ESN's r(t). It is then fed to the trained ridge readout.

Spike detection itself (threshold / template matching) is assumed to come from
MaxLab Live's online spike detector; here we just consume spike events
(channel, time). Swap in your own detector if needed.
"""
import numpy as np


class SpikeDecoder:
    def __init__(self, cfg, channel_ids):
        self.cfg = cfg
        self.channels = list(channel_ids)            # readout channel order
        self.ch_index = {c: i for i, c in enumerate(self.channels)}
        self.n_ch = len(self.channels)
        self.n_bins = cfg.decode.n_state_bins
        self.win_s = cfg.timing.response_window_ms * 1e-3

    def decode(self, spikes) -> np.ndarray:
        """
        spikes: iterable of (channel_id, t_seconds_relative_to_window_start).
        Returns state vector of length n_ch * n_bins.
        """
        mat = np.zeros((self.n_ch, self.n_bins))
        bin_w = self.win_s / self.n_bins
        for ch, t in spikes:
            if ch not in self.ch_index:
                continue
            b = min(int(t / bin_w), self.n_bins - 1)
            mat[self.ch_index[ch], b] += 1.0
        if self.cfg.decode.feature == "rate":
            mat /= bin_w                              # spikes/s
        return mat.reshape(-1)

    @property
    def state_dim(self) -> int:
        return self.n_ch * self.n_bins


def state_from_esn(r: np.ndarray) -> np.ndarray:
    """Identity passthrough for the ESN surrogate (state is already a vector)."""
    return r
