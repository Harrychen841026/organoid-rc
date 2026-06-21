"""
Quantify the delay between issuing a stimulation command in Python and the
electrode actually firing -- and, just as important, its trial-to-trial JITTER.

Two latencies exist (see stim_and_record.py): the pulse SHAPE is hardware-exact
once a maxlab.Sequence is downloaded (sample clock, no jitter); the DISPATCH
from seq.send() to first execution crosses Python -> mxwserver -> MaxHub and IS
variable. This script measures the dispatch end-to-end by co-recording.

METHOD
  live  : route + activate well, start a recording, then fire N single pulses,
          each bracketed with time.perf_counter() (host clock). Also times how
          long the send() call itself takes. Logs host send times + the recording
          start time so the offline step can align them.
  offline: find each pulse's ARTIFACT onset in the recording (synchronous
          multi-channel flood), convert to host time, and compute
              latency_i = artifact_host_i - send_host_i
          Reports mean +/- std (jitter) and a histogram.

CLOCK CAVEAT: absolute latency depends on aligning the host clock to the
recording's frame clock (we use the host time at start_recording as t=0 of the
recording). That alignment carries a constant unknown offset, so trust the
JITTER (std, alignment-free) most; treat the absolute mean as approximate unless
you have a hardware sync. The pulse-to-pulse interval jitter is also reported and
needs no alignment at all.

Demo (no hardware):  python measure_stim_latency.py --demo
"""
import argparse
import json
import os
import time
import numpy as np

import connectivity as cx

try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    _HAVE_MAXLAB = True
except Exception:                       # noqa
    _HAVE_MAXLAB = False

FS_HZ = 20_000.0


# ======================================================================
# LIVE probe (rig only; [VERIFY] calls as in stim_test.py)
# ======================================================================
def run_latency_probe(stim_electrodes, well=None, n_pulses=30, isi_s=0.5,
                      amp_dac=80, phase_us=200, gain=512,
                      out_dir=".", filename="latency_probe"):
    """Fire N single pulses while recording; log host send times. Returns a dict
    (also written to <filename>_timing.json) for latency_stats()."""
    if not _HAVE_MAXLAB:
        raise RuntimeError("maxlab not importable -- run on the MaxLab host.")

    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Amplifier().set_gain(gain))
    if well is not None:
        wi = int(str(well)[4:]) if str(well).lower().startswith("well") else int(well)
        maxlab.activate([wi])                       # [VERIFY]
        try:
            maxlab.set_primary_well(wi)
        except Exception:                           # noqa
            pass

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

    phase_samples = max(1, round(phase_us * 1e-6 * FS_HZ))

    def one_pulse():
        seq = maxlab.Sequence()
        seq.append(maxlab.chip.DAC(0, 512 - amp_dac))
        seq.append(maxlab.system.DelaySamples(phase_samples))
        seq.append(maxlab.chip.DAC(0, 512 + amp_dac))
        seq.append(maxlab.system.DelaySamples(phase_samples))
        seq.append(maxlab.chip.DAC(0, 512))
        return seq

    saving = maxlab.Saving()                                     # [VERIFY]
    saving.open_directory(out_dir)
    saving.start_file(filename)
    saving.start_recording([0])
    t_rec_start = time.perf_counter()                           # recording t=0 (host)

    time.sleep(1.0)                                             # baseline
    send_times, send_durations = [], []
    for _ in range(n_pulses):
        seq = one_pulse()
        t0 = time.perf_counter()
        seq.send()
        t1 = time.perf_counter()
        send_times.append(t0 - t_rec_start)     # send time in recording seconds
        send_durations.append(t1 - t0)          # how long send() blocked
        time.sleep(isi_s)
    time.sleep(1.0)

    saving.stop_recording()
    saving.stop_file()
    for s in stim_units:
        maxlab.send(s.power_up(False))

    h5_path = os.path.join(out_dir, filename + ".raw.h5")
    log = {"h5": h5_path, "well": well, "send_times_s": send_times,
           "send_durations_s": send_durations, "isi_s": isi_s,
           "n_pulses": n_pulses, "t_rec_start_host": 0.0}   # send_times already rec-relative
    with open(os.path.join(out_dir, filename + "_timing.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"Recorded {h5_path}; send() median = "
          f"{np.median(send_durations)*1e3:.2f} ms")
    return log


# ======================================================================
# OFFLINE analysis (hardware-free, tested)
# ======================================================================
def detect_artifact_onsets(data, bin_ms=0.5, frac_channels=0.3, min_sep_ms=50.0):
    """Onset times (s) of synchronous multi-channel events (stim artifacts).

    An artifact fires most electrodes within one bin, unlike spontaneous
    activity. We threshold the number of channels active per bin.
    """
    M, chans = cx.binned_matrix(data, bin_ms)
    B = (M > 0).astype(np.int8)
    pop = B.sum(axis=0)                       # channels active per bin
    thr = max(2.0, frac_channels * len(chans))
    bw = bin_ms * 1e-3
    onsets = []
    in_evt = False
    last = -1e9
    for k, v in enumerate(pop):
        t = k * bw
        if v >= thr and not in_evt and (t - last) >= min_sep_ms * 1e-3:
            onsets.append(t)
            last = t
            in_evt = True
        elif v < thr:
            in_evt = False
    return np.array(onsets)


def latency_stats(data, send_times_s, t_rec_start_host=0.0,
                  match_window_ms=60.0, plot_path=None, send_durations_s=None):
    """Match stim send times to artifact onsets; report latency + jitter."""
    onsets = detect_artifact_onsets(data)
    send = np.asarray(send_times_s, float)
    lat = []
    matched_onsets = []
    for ts in send:
        if onsets.size == 0:
            continue
        j = int(np.argmin(np.abs(onsets - ts)))
        dt = onsets[j] - ts
        if 0 <= dt <= match_window_ms * 1e-3:     # artifact must follow the send
            lat.append(dt)
            matched_onsets.append(onsets[j])
    lat = np.array(lat)

    res = {"n_sent": len(send), "n_matched": int(lat.size)}
    if lat.size:
        res.update(mean_latency_ms=float(lat.mean() * 1e3),
                   jitter_ms=float(lat.std() * 1e3),
                   min_ms=float(lat.min() * 1e3), max_ms=float(lat.max() * 1e3))
    # alignment-free: interval jitter (artifact intervals vs scheduled sends)
    if len(matched_onsets) > 2:
        oi = np.diff(matched_onsets)
        si = np.diff(send[:len(matched_onsets)])
        res["interval_jitter_ms"] = float(np.std(oi - si) * 1e3)

    print("=== STIM LATENCY ===")
    print(f"  matched {res['n_matched']}/{res['n_sent']} pulses to artifacts")
    if lat.size:
        print(f"  send -> stim latency : {res['mean_latency_ms']:.2f} +/- "
              f"{res['jitter_ms']:.2f} ms  (mean +/- jitter)")
        print(f"  range                : {res['min_ms']:.2f} - {res['max_ms']:.2f} ms")
        print("  NOTE: absolute mean carries a constant clock-alignment offset; "
              "trust the jitter most.")
    if send_durations_s is not None and len(send_durations_s):
        print(f"  send() call duration : {np.median(send_durations_s)*1e3:.2f} ms "
              "median (Python-side dispatch)")

    if plot_path and lat.size:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
        a0.hist(lat * 1e3, bins=20, color="tab:blue", edgecolor="k")
        a0.axvline(lat.mean() * 1e3, color="red", ls="--",
                   label=f"mean {lat.mean()*1e3:.2f} ms")
        a0.set_xlabel("send -> artifact latency (ms)"); a0.set_ylabel("count")
        a0.set_title(f"latency  (jitter sigma = {lat.std()*1e3:.2f} ms)")
        a0.legend()
        a1.plot(np.arange(lat.size), lat * 1e3, "o-", ms=3)
        a1.set_xlabel("pulse #"); a1.set_ylabel("latency (ms)")
        a1.set_title("per-pulse latency (drift check)")
        fig.tight_layout(); fig.savefig(plot_path, dpi=130); plt.close(fig)
        print(f"  wrote {plot_path}")
    return res


# ======================================================================
# Synthetic demo: inject a known latency + jitter and recover it
# ======================================================================
def _synth_latency_recording(send_times, true_latency_ms=4.0, jitter_ms=0.8,
                             n_ch=40, seed=0):
    rng = np.random.default_rng(seed)
    duration = max(send_times) + 2.0
    spikes = {c: list(np.where(rng.random(int(duration * 1000)) < 0.003)[0] / 1000.0)
              for c in range(n_ch)}
    for ts in send_times:                       # artifact = send + latency + jitter
        t_art = ts + true_latency_ms * 1e-3 + rng.normal(0, jitter_ms * 1e-3)
        for c in range(n_ch):
            if rng.random() < 0.9:
                spikes[c].append(t_art + rng.normal(0, 0.0003))
    pos = {c: (float((c % 8) * 60), float((c // 8) * 60)) for c in range(n_ch)}
    return cx.NetworkAssayData(
        spikes={c: np.array(sorted(v)) for c, v in spikes.items()},
        duration=duration, positions=pos, electrode={c: 1000 + c for c in range(n_ch)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure send->stimulation latency")
    ap.add_argument("--demo", action="store_true", help="synthetic recovery test")
    ap.add_argument("--file", help="recorded .raw.h5 from run_latency_probe")
    ap.add_argument("--timing", help="<filename>_timing.json from run_latency_probe")
    ap.add_argument("--well", default=None)
    ap.add_argument("--out", default="stim_latency.png")
    args = ap.parse_args()

    if args.demo:
        here = os.path.dirname(os.path.abspath(__file__))
        send_times = [1.0 + 0.5 * i for i in range(30)]      # 30 pulses, 0.5 s apart
        print(">> injecting true latency = 4.0 ms, jitter = 0.8 ms")
        data = _synth_latency_recording(send_times, true_latency_ms=4.0, jitter_ms=0.8)
        latency_stats(data, send_times,
                      plot_path=os.path.join(here, "fig_conn6_latency_demo.png"))
    elif args.file and args.timing:
        with open(args.timing) as fh:
            log = json.load(fh)
        data = cx.load_network_assay(args.file, well=args.well)
        latency_stats(data, log["send_times_s"],
                      t_rec_start_host=log.get("t_rec_start_host", 0.0),
                      send_durations_s=log.get("send_durations_s"),
                      plot_path=args.out)
    else:
        ap.print_help()
