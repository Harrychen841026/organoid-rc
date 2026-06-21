"""
Confirm a stimulation pulse is actually delivered -- by recording while you stim.

Two parts:
  1. record_during_stim(...)  -- LIVE on the rig: opens a recording, fires the
     pulse train, stops. Records the stim onset times so you can align to them.
     (MaxLab Saving API calls are marked [VERIFY] -- check your install.)
  2. peristim_check(...)       -- OFFLINE: load the recorded spikes, align them to
     the stim times, and show a peri-stimulus raster + PSTH with a PASS/FAIL
     verdict. This part is hardware-free and is what the --demo below tests.

WHAT A DELIVERED PULSE LOOKS LIKE
  * Stimulation ARTIFACT: a large, near-instantaneous deflection time-locked to
    stim onset, on MANY recording electrodes at once (strongest near the stim
    site). Online spike detection registers it as a synchronous "spike flood" at
    t=0. Seeing that synchronous flood is the simplest proof the pulse went out.
  * EVOKED response: a few ms later, more spatially restricted spikes on
    connected electrodes -- evidence the tissue actually responded, not just that
    charge was injected.
  A flat peri-stimulus histogram = nothing was delivered (or electrode not
  coupled / wrong unit / amplitude too low).

OTHER SANITY CHECKS (do these too):
  * MaxLab Live Scope GUI: watch the live trace and fire -- you should SEE the
    artifact. Fastest first check, no code.
  * Electrode impedance before stim: a broken / high-impedance electrode won't
    pass charge. Check it's well coupled.
  * API return values: query_stimulation_at_electrode must return a real unit;
    route()/download() must not error. (Necessary, not sufficient -- software
    "ok" is not proof of physical delivery; the artifact is.)
  * You cannot cleanly record the STIM electrode itself (it saturates/blanks);
    read its NEIGHBOURS.

Demo (no hardware):  python stim_and_record.py --demo
"""
import argparse
import os
import time
import numpy as np

import connectivity as cx          # reuse NetworkAssayData + load_network_assay

try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    _HAVE_MAXLAB = True
except Exception:                  # noqa
    _HAVE_MAXLAB = False

FS_HZ = 20_000.0


# ======================================================================
# 1. LIVE: record while stimulating  (rig only; calls marked [VERIFY])
# ======================================================================
def record_during_stim(stim_electrodes, record_seconds=12.0, amp_dac=80,
                        phase_us=200, n_pulses=10, rate_hz=10.0,
                        out_dir=".", filename="stim_check", well=None):
    """Route stim + recording, start a file, fire pulses, stop. Returns
    (h5_path, stim_times_s) so you can run peristim_check on it.

    The recording captures ALL routed electrodes, so neighbours of the stim site
    are recorded and will show the artifact. Marked [VERIFY] calls vary across
    MaxLab versions -- cross-check the api-docs Stimulation + Saving examples.
    """
    if not _HAVE_MAXLAB:
        raise RuntimeError("maxlab not importable -- run this on the MaxLab host.")

    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Amplifier().set_gain(512))
    # [MAXTWO] activate the target well here before routing (install-specific).

    array = maxlab.chip.Array("stimulation")
    array.reset()
    array.select_stimulation_electrodes(stim_electrodes)        # [VERIFY]
    array.route()
    units = []
    for e in stim_electrodes:
        array.connect_electrode_to_stimulation(e)               # [VERIFY]
        u = array.query_stimulation_at_electrode(e)             # [VERIFY]
        if not u:
            raise RuntimeError(f"electrode {e}: no stim unit assigned.")
        units.append(u)
    array.download()
    maxlab.util.offset()

    stim_units = []
    for u in units:
        s = (maxlab.chip.StimulationUnit(u)
             .power_up(True).connect(True).set_voltage_mode().dac_source(0))
        maxlab.send(s)
        stim_units.append(s)

    # ---- start recording -------------------------------------------------
    saving = maxlab.Saving()                                     # [VERIFY]
    saving.open_directory(out_dir)
    saving.start_file(filename)
    saving.start_recording([0])
    t_rec0 = time.time()

    phase_samples = max(1, round(phase_us * 1e-6 * FS_HZ))
    ipi = max(1, round(FS_HZ / rate_hz) - 2 * phase_samples)

    time.sleep(2.0)                       # baseline before stim
    seq = maxlab.Sequence()
    for _ in range(n_pulses):
        seq.append(maxlab.chip.DAC(0, 512 - amp_dac))
        seq.append(maxlab.system.DelaySamples(phase_samples))
        seq.append(maxlab.chip.DAC(0, 512 + amp_dac))
        seq.append(maxlab.system.DelaySamples(phase_samples))
        seq.append(maxlab.chip.DAC(0, 512))
        seq.append(maxlab.system.DelaySamples(ipi))
    t_stim_start = time.time() - t_rec0   # stim onset relative to recording start
    seq.send()
    stim_times = [t_stim_start + k / rate_hz for k in range(n_pulses)]

    time.sleep(max(0.0, record_seconds - (time.time() - t_rec0)))
    saving.stop_recording()
    saving.stop_file()
    for s in stim_units:
        maxlab.send(s.power_up(False))

    h5_path = os.path.join(out_dir, filename + ".raw.h5")
    print(f"Recorded {h5_path}; {n_pulses} pulses at t={stim_times[0]:.2f}s ...")
    return h5_path, stim_times


# ======================================================================
# 2. OFFLINE: peri-stimulus sanity check (hardware-free, testable)
# ======================================================================
def peristim_check(data, stim_times_s, pre_ms=20.0, post_ms=40.0,
                   bin_ms=1.0, sync_ms=2.0, plot_path=None):
    """Align recorded spikes to stim onsets; quantify + (optionally) plot.

    Returns a dict with the verdict. The two signatures of a delivered pulse:
      peak_ratio   : population PSTH peak just after stim / baseline before stim.
                     >> 1 means activity is time-locked to stim.
      sync_fraction: fraction of channels firing within +/- sync_ms of stim onset
                     (the synchronous artifact flood).
    """
    chans = data.channels
    n_ch = len(chans)
    pre, post = pre_ms * 1e-3, post_ms * 1e-3
    edges = np.arange(-pre, post + 1e-9, bin_ms * 1e-3)
    psth = np.zeros(len(edges) - 1)
    raster = []                       # (rel_time, channel_index) for plotting
    sync_hits = 0
    sync_possible = 0
    for ci, c in enumerate(chans):
        st = np.asarray(data.spikes.get(c, ()))
        if st.size == 0:
            continue
        for ts in stim_times_s:
            rel = st - ts
            m = (rel >= -pre) & (rel < post)
            if m.any():
                psth += np.histogram(rel[m], edges)[0]
                for r in rel[m]:
                    raster.append((r, ci))
            # synchronous-artifact test, per (channel, stim)
            sync_possible += 1
            if np.any(np.abs(rel) <= sync_ms * 1e-3):
                sync_hits += 1

    n_stim = max(len(stim_times_s), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    base = psth[centers < 0]
    # only count an evoked peak in a physiologically plausible early window
    early = (centers >= 0) & (centers <= 0.015)
    base_mean = base.mean() if base.size else 0.0
    base_std = base.std() if base.size else 0.0
    peak = psth[early].max() if early.any() else 0.0
    peak_ratio = float(peak / base_mean) if base_mean > 0 else (np.inf if peak > 0 else 0.0)
    # significance of the peak vs baseline (Poisson-ish z); robust to sparse data
    peak_z = float((peak - base_mean) / (base_std + np.sqrt(base_mean) + 1e-9))
    sync_fraction = float(sync_hits / sync_possible) if sync_possible else 0.0
    # primary signal = synchronous artifact flood; PSTH peak must be SIGNIFICANT
    delivered = (sync_fraction >= 0.2) or (peak_ratio >= 5.0 and peak_z >= 4.0)

    result = {"peak_ratio": peak_ratio, "peak_z": peak_z,
              "sync_fraction": sync_fraction, "n_stim": n_stim,
              "delivered": bool(delivered),
              "peak_latency_ms": float(centers[np.argmax(psth)] * 1e3) if psth.any() else None}

    print("=== PERI-STIM SANITY CHECK ===")
    print(f"  stim events            : {n_stim}")
    print(f"  synchronous channels   : {sync_fraction*100:.0f}% fire within "
          f"+/-{sync_ms:.0f} ms of stim  (artifact flood -- primary signal)")
    print(f"  PSTH early peak/baseline: {peak_ratio:.1f}  (z={peak_z:.1f}; "
          "need ratio>=5 AND z>=4)")
    print(f"  VERDICT                : "
          + ("STIM DELIVERED (response time-locked to pulse)" if delivered
             else "NO time-locked response -- check coupling / unit / amplitude / GUI"))

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        if raster:
            rt, rc = zip(*raster)
            a0.scatter(np.array(rt) * 1e3, rc, s=4, color="k", alpha=0.5)
        a0.axvline(0, color="red", lw=1.5, label="stim onset")
        a0.set_ylabel("recording channel"); a0.set_ylim(-1, n_ch)
        a0.set_title(f"peri-stimulus raster (all {n_stim} pulses overlaid)  "
                     f"verdict: {'DELIVERED' if delivered else 'NOT DETECTED'}")
        a0.legend(loc="upper right")
        a1.bar(centers * 1e3, psth / n_stim, width=bin_ms, color="tab:blue",
               align="center")
        a1.axvline(0, color="red", lw=1.5)
        a1.set_xlabel("time relative to stim (ms)")
        a1.set_ylabel("spikes / bin / stim")
        a1.set_title(f"population PSTH  (peak/baseline = {peak_ratio:.1f})")
        fig.tight_layout(); fig.savefig(plot_path, dpi=130); plt.close(fig)
        print(f"  wrote {plot_path}")
    return result


# ======================================================================
# 3. Synthetic demo: prove the offline check flags a delivered pulse
# ======================================================================
def _synth_stim_recording(seed=0, duration=40.0, n_ch=40, stim_times=(10, 20, 30),
                          deliver=True):
    """Spontaneous spikes + (if deliver) an artifact flood at each stim onset and
    a short-latency evoked bump on a subset of channels."""
    rng = np.random.default_rng(seed)
    spikes = {c: list(np.where(rng.random(int(duration * 1000)) < 0.004)[0] / 1000.0)
              for c in range(n_ch)}
    if deliver:
        for ts in stim_times:
            for c in range(n_ch):                 # artifact: ~all channels at t=0
                if rng.random() < 0.9:
                    spikes[c].append(ts + rng.normal(0, 0.0005))
            for c in rng.choice(n_ch, size=12, replace=False):   # evoked +3-6 ms
                if rng.random() < 0.7:
                    spikes[c].append(ts + 0.003 + rng.random() * 0.003)
    pos = {c: (float((c % 8) * 60), float((c // 8) * 60)) for c in range(n_ch)}
    return cx.NetworkAssayData(
        spikes={c: np.array(sorted(v)) for c, v in spikes.items()},
        duration=duration, positions=pos, electrode={c: 1000 + c for c in range(n_ch)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stimulation sanity check")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic delivered-vs-not test (no hardware)")
    ap.add_argument("--file", help="recorded .raw.h5 to check")
    ap.add_argument("--well", default=None)
    ap.add_argument("--stim-times", default=None,
                    help="comma-separated stim onset times in seconds")
    ap.add_argument("--out", default="peristim_check.png")
    args = ap.parse_args()

    if args.demo:
        here = os.path.dirname(os.path.abspath(__file__))
        print(">> case A: pulse DELIVERED")
        d1 = _synth_stim_recording(seed=1, deliver=True)
        peristim_check(d1, [10, 20, 30],
                       plot_path=os.path.join(here, "fig_conn5_peristim_demo.png"))
        print("\n>> case B: pulse NOT delivered (control)")
        d0 = _synth_stim_recording(seed=2, deliver=False)
        peristim_check(d0, [10, 20, 30])
    elif args.file:
        stim_times = [float(x) for x in args.stim_times.split(",")] if args.stim_times else []
        if not stim_times:
            raise SystemExit("provide --stim-times (seconds) from your stim run")
        data = cx.load_network_assay(args.file, well=args.well)
        peristim_check(data, stim_times, plot_path=args.out)
    else:
        ap.print_help()
