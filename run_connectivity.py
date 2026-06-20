"""
Run the functional-connectivity electrode selection on a real MaxWell Network
assay file.  THIS is the script to run on your data (connectivity.py is the
library behind it; running connectivity.py only runs its synthetic self-test).

Examples
--------
# 1) First, confirm the file layout / find well names:
python run_connectivity.py /path/to/network.raw.h5 --inspect

# 2) Run selection on one well (MaxTwo) and write a QC plot + JSON:
python run_connectivity.py /path/to/network.raw.h5 --well well000 \
       --n-inputs 6 --n-outputs 256 --bin-ms 2 --delays 2,4,6,8,10,14,18

# 3) Restrict inputs to electrodes you can route to stim units (recommended):
python run_connectivity.py network.raw.h5 --well well000 \
       --stimulable 12044,9810,15522,3001,20418,7777

Output (next to the .h5, or --out-dir):
  <stem>_<well>_selection.json   input/output channels + electrodes + peak delays
  <stem>_<well>_qc.png           delay-vs-distance QC + spatial map of picks
"""
import argparse
import json
import os
import numpy as np

import connectivity as cx


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""] if s else None


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def process_well(args, well, out_dir, stem):
    """Load one well, run selection, write JSON + QC + FC npz. Returns summary."""
    tag = f"{stem}_{well or 'well'}"
    qc_path = os.path.join(out_dir, f"{tag}_qc.png")
    json_path = os.path.join(out_dir, f"{tag}_selection.json")
    npz_path = os.path.join(out_dir, f"{tag}_fc.npz")

    print(f"\n--- {well or 'single well'} ---")
    data = cx.load_network_assay(args.h5, well=(well or None), rec=args.rec)
    rates = data.firing_rates()
    print(f"  {len(data.channels)} channels, duration {data.duration:.1f}s, "
          f"fs {data.fs:.0f} Hz, median rate "
          f"{np.median(list(rates.values())):.2f} Hz")
    if not data.positions:
        print("  WARNING: no positions -> spacing + delay-vs-distance QC skipped.")

    sel = cx.select_io_electrodes(
        data,
        n_inputs=args.n_inputs, n_outputs=args.n_outputs,
        bin_ms=args.bin_ms, delays_ms=parse_float_list(args.delays),
        min_rate_hz=args.min_rate_hz, max_channels=args.max_channels,
        input_min_spacing_um=args.input_spacing_um,
        stimulable_channels=parse_int_list(args.stimulable),
        n_surrogates=args.n_surrogates, rng_seed=args.seed,
        qc_plot_path=qc_path if data.positions else None)

    chans = sel.channels
    elecs = np.array([data.electrode.get(c, c) for c in chans])
    fr = np.array([rates.get(c, 0.0) for c in chans])
    pos = np.array([data.positions.get(c, (np.nan, np.nan)) for c in chans], float)
    # save FC in ELECTRODE space (physical ids are stable across days; channels are not)
    np.savez(npz_path,
             electrodes=elecs, positions=pos, W=sel.W, delay_ms=sel.delay_ms,
             directedness=sel.scores["directedness"],
             in_strength=sel.scores["in_strength"],
             out_strength=sel.scores["out_strength"], firing_rate=fr,
             input_electrodes=np.array(sel.input_electrodes),
             output_electrodes=np.array(sel.output_electrodes),
             bin_ms=args.bin_ms, delays_ms=parse_float_list(args.delays),
             duration=data.duration, fs=data.fs, well=str(well))

    result = {
        "file": os.path.abspath(args.h5), "well": well, "rec": args.rec,
        "bin_ms": args.bin_ms, "delays_ms": parse_float_list(args.delays),
        "input_channels": [int(c) for c in sel.input_channels],
        "output_channels": [int(c) for c in sel.output_channels],
        "input_electrodes": [int(e) for e in sel.input_electrodes],
        "output_electrodes": [int(e) for e in sel.output_electrodes],
    }
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  INPUT electrodes : {result['input_electrodes']}")
    print(f"  wrote {os.path.basename(json_path)}, "
          f"{os.path.basename(npz_path)}"
          + (f", {os.path.basename(qc_path)}" if data.positions else ""))
    return result


def main():
    ap = argparse.ArgumentParser(description="MaxWell Network-assay -> input/output electrodes")
    ap.add_argument("h5", help="path to the Network-assay .raw.h5 file")
    ap.add_argument("--inspect", action="store_true",
                    help="just print the file tree + detected format, then exit")
    ap.add_argument("--well", default=None, help="MaxTwo well, e.g. well000 (new format)")
    ap.add_argument("--all-wells", action="store_true",
                    help="process every well in the file (one selection + QC + FC each)")
    ap.add_argument("--rec", default=None, help="recording id, e.g. rec0000")
    ap.add_argument("--n-inputs", type=int, default=6, help="# stim/input electrodes")
    ap.add_argument("--n-outputs", type=int, default=256, help="# readout/output electrodes")
    ap.add_argument("--bin-ms", type=float, default=2.0)
    ap.add_argument("--delays", default="2,4,6,8,10,14,18",
                    help="comma-separated TE delays in ms (peak-delay sweep)")
    ap.add_argument("--min-rate-hz", type=float, default=0.1,
                    help="drop electrodes below this firing rate before TE")
    ap.add_argument("--max-channels", type=int, default=256,
                    help="prefilter to this many most-active channels (TE is O(N^2))")
    ap.add_argument("--input-spacing-um", type=float, default=100.0)
    ap.add_argument("--stimulable", default=None,
                    help="comma-separated channels routable to stim units "
                         "(strongly recommended; only 32 stim units/well)")
    ap.add_argument("--n-surrogates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.inspect:
        cx.inspect_h5(args.h5)
        return

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.h5))
    stem = os.path.splitext(os.path.splitext(os.path.basename(args.h5))[0])[0]

    if args.all_wells:
        wells = cx.list_wells(args.h5)
        print(f"Processing {len(wells)} wells: {wells}")
        for w in wells:
            process_well(args, w, out_dir, stem)
    else:
        process_well(args, args.well, out_dir, stem)

    print("\nNext: validate each well's INPUT electrodes with a stim scan before "
          "committing. To compare across days, run compare_days.py on the *_fc.npz.")


if __name__ == "__main__":
    main()
