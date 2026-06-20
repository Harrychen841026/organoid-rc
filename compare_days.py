"""
Compare functional connectivity of ONE well across several days.

Feed it the *_fc.npz files produced by run_connectivity.py (same well, different
recording days, in chronological order). It reports how the network develops and
how stable it is -- both of which matter for the reservoir: development tells you
the organoid is maturing, drift tells you how often you must re-select electrodes
and recalibrate readouts.

WHAT IT COMPARES
  Per-day trajectory (is the network developing?):
    - active electrodes, edge density, mean TE (overall coupling strength)
    - mean firing rate, # source/sink hubs, mean conduction (peak) delay
  Pairwise stability across days (is it drifting?), all in ELECTRODE space on the
  electrodes COMMON to both days (channel ids are not stable across days; physical
  electrode ids are):
    - directedness correlation  -> do source/sink ROLES persist (key for I/O picks)
    - edge Jaccard              -> turnover of significant connections
    - input/output selection overlap -> can you reuse the same electrodes

IMPORTANT for a fair comparison: use the SAME bin_ms, delays, sig_z and (crucially)
RECORDING DURATION across days -- TE estimates and edge counts depend on data
length and firing rate, so unequal recordings create artefactual "development".
This script warns if bin or duration differ.

Run:
  python compare_days.py day1_well000_fc.npz day2_well000_fc.npz day3_well000_fc.npz \
         --labels D7,D14,D21 --out well000_across_days
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def day_metrics(d):
    W = d["W"]
    n = W.shape[0]
    sig = W > 0
    dirn = d["directedness"]
    delay = d["delay_ms"]
    return {
        "n_active": int(n),
        "edge_density": float(sig.sum() / max(n * (n - 1), 1)),
        "mean_TE": float(W[sig].mean()) if sig.any() else 0.0,
        "mean_firing_hz": float(np.mean(d["firing_rate"])),
        "n_source_hubs": int((dirn > 0.5).sum()),
        "n_sink_hubs": int((dirn < -0.5).sum()),
        "mean_peak_delay_ms": float(np.nanmean(delay[sig])) if sig.any() else float("nan"),
    }


def common_submatrix(da, db):
    """Indices into each day's arrays for their common electrodes (sorted)."""
    ea = {int(e): i for i, e in enumerate(da["electrodes"])}
    eb = {int(e): i for i, e in enumerate(db["electrodes"])}
    common = sorted(set(ea) & set(eb))
    ia = np.array([ea[e] for e in common], int)
    ib = np.array([eb[e] for e in common], int)
    return common, ia, ib


def pairwise(da, db):
    common, ia, ib = common_submatrix(da, db)
    out = {"n_common": len(common)}
    if len(common) < 3:
        out.update(directedness_r=float("nan"), edge_jaccard=float("nan"))
    else:
        # role stability: correlation of directedness on common electrodes
        Aa = da["directedness"][ia]
        Bb = db["directedness"][ib]
        out["directedness_r"] = (float(np.corrcoef(Aa, Bb)[0, 1])
                                 if Aa.std() > 0 and Bb.std() > 0 else float("nan"))
        # edge turnover: Jaccard of significant edges on the common submatrix
        Wa = da["W"][np.ix_(ia, ia)] > 0
        Wb = db["W"][np.ix_(ib, ib)] > 0
        off = ~np.eye(len(common), dtype=bool)
        A, B = Wa & off, Wb & off
        inter = (A & B).sum()
        union = (A | B).sum()
        out["edge_jaccard"] = float(inter / union) if union else float("nan")
    # selection reuse: overlap of chosen input / output electrodes
    def jac(x, y):
        x, y = set(map(int, x)), set(map(int, y))
        return float(len(x & y) / len(x | y)) if (x | y) else float("nan")
    out["input_overlap"] = jac(da["input_electrodes"], db["input_electrodes"])
    out["output_overlap"] = jac(da["output_electrodes"], db["output_electrodes"])
    return out


def heat(ax, M, labels, title, cmap, fmt="{:.2f}", vmin=None, vmax=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            ax.text(j, i, "" if np.isnan(v) else fmt.format(v),
                    ha="center", va="center",
                    color="white" if (vmin is None or v < (vmin + vmax) / 2) else "black",
                    fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser(description="Compare FC across days (one well)")
    ap.add_argument("npz", nargs="+", help="*_fc.npz files, chronological order")
    ap.add_argument("--labels", default=None, help="comma labels, e.g. D7,D14,D21")
    ap.add_argument("--out", default="fc_across_days", help="output prefix")
    args = ap.parse_args()

    days = [load(p) for p in args.npz]
    labels = (args.labels.split(",") if args.labels
              else [f"d{i+1}" for i in range(len(days))])
    assert len(labels) == len(days), "labels count must match number of files"

    # ---- methodology guard ------------------------------------------------
    bins = {float(d["bin_ms"]) for d in days}
    durs = [float(d["duration"]) for d in days]
    if len(bins) > 1:
        print(f"WARNING: bin_ms differs across files ({bins}) -- not comparable.")
    if max(durs) / max(min(durs), 1e-9) > 1.2:
        print(f"WARNING: recording durations differ a lot ({[round(x) for x in durs]} s). "
              "TE/edge counts scale with duration -- truncate to equal length for a "
              "fair comparison.")

    # ---- per-day metrics --------------------------------------------------
    mets = [day_metrics(d) for d in days]
    keys = ["n_active", "edge_density", "mean_TE", "mean_firing_hz",
            "n_source_hubs", "n_sink_hubs", "mean_peak_delay_ms"]
    print("\nPer-day metrics:")
    print("  " + "  ".join(f"{'day':>6}") + "  " + "  ".join(f"{k:>16}" for k in keys))
    for lab, m in zip(labels, mets):
        print(f"  {lab:>6}  " + "  ".join(f"{m[k]:16.3f}" for k in keys))

    # ---- pairwise matrices ------------------------------------------------
    n = len(days)
    R = np.full((n, n), np.nan)      # directedness correlation
    J = np.full((n, n), np.nan)      # edge Jaccard
    IN = np.full((n, n), np.nan)     # input-selection overlap
    pair_json = {}
    for i in range(n):
        R[i, i] = 1.0; J[i, i] = 1.0; IN[i, i] = 1.0
        for j in range(i + 1, n):
            pr = pairwise(days[i], days[j])
            R[i, j] = R[j, i] = pr["directedness_r"]
            J[i, j] = J[j, i] = pr["edge_jaccard"]
            IN[i, j] = IN[j, i] = pr["input_overlap"]
            pair_json[f"{labels[i]}_vs_{labels[j]}"] = pr

    # ---- figure -----------------------------------------------------------
    x = range(n)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax[0, 0].plot(x, [m["edge_density"] for m in mets], "o-")
    ax[0, 0].set_title("edge density (network integration)"); ax[0, 0].set_ylabel("density")
    ax[0, 1].plot(x, [m["mean_TE"] for m in mets], "o-", color="tab:red")
    ax[0, 1].set_title("mean TE (coupling strength)"); ax[0, 1].set_ylabel("bits")
    ax[0, 2].plot(x, [m["mean_firing_hz"] for m in mets], "o-", color="tab:green")
    ax2b = ax[0, 2].twinx()
    ax2b.plot(x, [m["mean_peak_delay_ms"] for m in mets], "s--", color="tab:purple")
    ax[0, 2].set_title("firing rate (green) & mean delay (purple)")
    ax[0, 2].set_ylabel("Hz", color="tab:green"); ax2b.set_ylabel("ms", color="tab:purple")
    for a in ax[0]:
        a.set_xticks(list(x)); a.set_xticklabels(labels)

    heat(ax[1, 0], R, labels, "directedness correlation\n(source/sink ROLE stability)",
         "viridis", vmin=-1, vmax=1)
    heat(ax[1, 1], J, labels, "edge Jaccard\n(connection turnover)",
         "magma", vmin=0, vmax=1)
    heat(ax[1, 2], IN, labels, "input-electrode overlap\n(can you reuse picks?)",
         "cividis", vmin=0, vmax=1)
    fig.suptitle("Functional connectivity across days (one well)", fontsize=13)
    fig.tight_layout()
    fig_path = f"{args.out}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    summary = {"labels": labels, "files": [os.path.abspath(p) for p in args.npz],
               "per_day": dict(zip(labels, mets)), "pairwise": pair_json}
    with open(f"{args.out}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote {fig_path} and {args.out}.json")
    print("Read it as: top row = development trajectory; bottom row = stability/drift "
          "(high directedness-r + high overlap = reuse electrodes; low = re-select).")


if __name__ == "__main__":
    main()
