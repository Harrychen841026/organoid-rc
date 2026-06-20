"""
Sweep the transfer-entropy bin size and delay, evaluate them, and visualise.

Run:  python make_connectivity_figures.py

Produces three figures next to this file:
  fig_conn1_param_sweep.png   -- bin x delay grid of metrics (+ F1 vs truth)
  fig_conn2_best_network.png  -- connectivity matrix & spatial map at the pick
  fig_conn3_delay_distance.png-- why a single delay band-passes by distance

It runs on a synthetic grid network whose edges have DISTANCE-DEPENDENT delays
(connectivity._synthesize_spatial_demo), so we know the ground truth and can
score each (bin, delay) by F1. On real MaxWell data you have no ground truth,
so use the unsupervised criteria the script also prints (occupancy in range,
effect size, stability) and the max-over-delays matrix.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import connectivity as cx

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def prf1(recovered, true_set):
    """precision, recall, F1 of a recovered edge set vs ground-truth pairs."""
    rec = set(recovered)
    inter = len(rec & true_set)
    p = inter / len(rec) if rec else 0.0
    r = inter / len(true_set) if true_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def annotate(ax, grid, fmt="{:.0f}"):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                    color="white", fontsize=9)


# ----------------------------------------------------------------------
# build the test network
# ----------------------------------------------------------------------
print("Building spatial multi-delay test network...")
data, true_edges = cx._synthesize_spatial_demo(seed=SEED)
true_pairs = {(s, d) for s, d, _ in true_edges}
true_delay = {(s, d): dl for s, d, dl in true_edges}
print(f"  {len(data.channels)} electrodes, {len(true_pairs)} true edges, "
      f"delays {sorted(set(true_delay.values()))} ms")

bins_ms = [2, 5, 10]
delays_ms = [2, 4, 8, 16]


# ======================================================================
# FIGURE 1 -- parameter sweep grid
# ======================================================================
print("Sweeping bin x delay ...")
sw = cx.sweep_bin_delay(data, bins_ms, delays_ms, n_surrogates=6, max_channels=64,
                        rng_seed=SEED)

f1 = np.zeros((len(bins_ms), len(delays_ms)))
for bi in range(len(bins_ms)):
    for di in range(len(delays_ms)):
        _, _, f1[bi, di] = prf1(sw["edges"][bi][di].keys(), true_pairs)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
panels = [("# significant edges", sw["n_edges"], "{:.0f}", "viridis"),
          ("F1 vs ground truth", f1, "{:.2f}", "magma"),
          ("bin occupancy (want 0.01-0.3)", sw["occupancy"], "{:.3f}", "cividis"),
          ("edge effect size (SNR vs null)", sw["effect_size"], "{:.1f}", "plasma")]
for ax, (title, grid, fmt, cmap) in zip(axes.ravel(), panels):
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    annotate(ax, grid, fmt)
    ax.set_xticks(range(len(delays_ms)), [f"{d}" for d in delays_ms])
    ax.set_yticks(range(len(bins_ms)), [f"{b}" for b in bins_ms])
    ax.set_xlabel("TE delay (ms)")
    ax.set_ylabel("bin size (ms)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# mark the unresolvable cells (delay < bin) on the F1 panel
for bi, b in enumerate(bins_ms):
    for di, d in enumerate(delays_ms):
        if not sw["resolvable"][bi, di]:
            axes[0, 1].add_patch(plt.Rectangle((di - .5, bi - .5), 1, 1, fill=False,
                                               edgecolor="red", lw=2, ls=":"))
fig.suptitle("TE parameter sweep  (red dotted = delay < bin, cannot resolve direction)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_conn1_param_sweep.png"), dpi=130)
plt.close(fig)

# best pick by F1
bbi, bdi = np.unravel_index(np.argmax(f1), f1.shape)
best_bin, best_delay = bins_ms[bbi], delays_ms[bdi]
p, r, fbest = prf1(sw["edges"][bbi][bdi].keys(), true_pairs)
print(f"  BEST single (bin, delay) by F1 = ({best_bin} ms, {best_delay} ms): "
      f"P={p:.2f} R={r:.2f} F1={fbest:.2f}")


# ======================================================================
# FIGURE 2 -- best-parameter connectivity matrix + spatial map
# ======================================================================
print("Rendering best-parameter network ...")
W = sw["W"][bbi][bdi]
chans = sw["chans"]
sc = cx.node_scores(W)
pos = data.positions

fig, (axm, axs) = plt.subplots(1, 2, figsize=(13, 5.5))
im = axm.imshow(W, cmap="inferno", aspect="auto")
axm.set_title(f"TE matrix  W[i,j]=TE(i->j)\nbin={best_bin} ms, delay={best_delay} ms")
axm.set_xlabel("target j"); axm.set_ylabel("source i")
fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04, label="TE (bits)")

xs = np.array([pos[c][0] for c in chans]); ys = np.array([pos[c][1] for c in chans])
dvals = sc["directedness"]
edges = cx.significant_edges(W, chans)
ci = {c: k for k, c in enumerate(chans)}
top = sorted(edges.items(), key=lambda kv: -kv[1])[:25]
for (s, d), w in top:
    axs.annotate("", xy=(pos[d][0], pos[d][1]), xytext=(pos[s][0], pos[s][1]),
                 arrowprops=dict(arrowstyle="-|>", color="0.4", lw=0.8, alpha=0.6))
scat = axs.scatter(xs, ys, c=dvals, cmap="coolwarm", s=260, vmin=-1, vmax=1,
                   edgecolor="k", zorder=3)
for c in chans:
    axs.text(pos[c][0], pos[c][1], str(c), ha="center", va="center", fontsize=6, zorder=4)
axs.set_title("electrode map: colour = directedness\n(red = source/input, blue = sink/output)")
axs.set_xlabel("x (um)"); axs.set_ylabel("y (um)"); axs.set_aspect("equal")
axs.invert_yaxis()
fig.colorbar(scat, ax=axs, fraction=0.046, pad=0.04, label="(out-in)/(out+in)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_conn2_best_network.png"), dpi=130)
plt.close(fig)


# ======================================================================
# FIGURE 3 -- the delay story: a single delay band-passes by distance
# ======================================================================
print("Building delay-vs-distance analysis ...")
fine_delays = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18]
bin_for_delay = 2.0

# raw TE for each true edge across fine delays  ->  argmax delay per edge
raw_by_delay = {}
thr_by_delay = {}
for dm in fine_delays:
    db = max(1, int(round(dm / bin_for_delay)))
    res = cx.te_and_null(data, bin_ms=bin_for_delay, delay_bins=db,
                         n_surrogates=5, rng_seed=SEED)
    raw_by_delay[dm] = (res.W_raw, res.chans)
    thr_by_delay[dm] = res.threshold(3.0)

chf = raw_by_delay[fine_delays[0]][1]
cif = {c: k for k, c in enumerate(chf)}
dist_list, argmax_delay_list = [], []
for (s, d, dl) in true_edges:
    te_curve = [raw_by_delay[dm][0][cif[s], cif[d]] for dm in fine_delays]
    best_dm = fine_delays[int(np.argmax(te_curve))]
    dist = float(np.hypot(pos[s][0] - pos[d][0], pos[s][1] - pos[d][1]))
    dist_list.append(dist); argmax_delay_list.append(best_dm)

# recall vs single delay
recall_curve = []
for dm in fine_delays:
    rec = cx.significant_edges(thr_by_delay[dm], chf).keys()
    _, rr, _ = prf1(rec, true_pairs)
    recall_curve.append(rr)

# single-best vs max-over-delays
best_single_recall = max(recall_curve)
Wmax, chm = cx.connectivity_max_over_delays(data, bin_for_delay, fine_delays,
                                            n_surrogates=6, rng_seed=SEED)
_, recall_max, f1_max = prf1(cx.significant_edges(Wmax, chm).keys(), true_pairs)

fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.5))
a0.scatter(dist_list, argmax_delay_list, s=60, c="tab:blue", edgecolor="k")
zz = np.polyfit(dist_list, argmax_delay_list, 1)
xv = np.linspace(min(dist_list), max(dist_list), 50)
a0.plot(xv, np.polyval(zz, xv), "r--", label=f"slope={zz[0]*1000:.1f} ms/mm")
a0.set_xlabel("inter-electrode distance (um)")
a0.set_ylabel("delay that maximises TE (ms)")
a0.set_title("delay grows with distance\n(so one delay = one distance band)")
a0.legend()

a1.plot(fine_delays, recall_curve, "o-", color="tab:green")
a1.set_xlabel("single TE delay (ms)")
a1.set_ylabel("recall of true edges")
a1.set_title("a single delay recovers only\nedges near that latency")
a1.set_ylim(0, 1.05)

a2.bar(["best single\ndelay", "max over\ndelays"],
       [best_single_recall, recall_max],
       color=["tab:orange", "tab:purple"])
a2.set_ylabel("recall of true edges")
a2.set_title("max-over-delays recovers\nshort + long edges together")
a2.set_ylim(0, 1.05)
for i, v in enumerate([best_single_recall, recall_max]):
    a2.text(i, v + 0.02, f"{v:.2f}", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_conn3_delay_distance.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
print("\n================ EVALUATION SUMMARY ================")
print(f"Best single (bin, delay) by F1     : {best_bin} ms, {best_delay} ms  "
      f"(P={p:.2f} R={r:.2f} F1={fbest:.2f})")
print(f"Occupancy at that bin              : {sw['occupancy'][bbi, bdi]:.3f}  "
      "(target 0.01-0.30)")
print(f"[delay study @ {bin_for_delay:.0f} ms bin] best single-delay recall : "
      f"{best_single_recall:.2f}")
print(f"[delay study @ {bin_for_delay:.0f} ms bin] max-over-delays recall/F1 : "
      f"{recall_max:.2f} / {f1_max:.2f}")
print("  -> max-over-delays catches ALL latencies (recall up) but adds false")
print("     positives (precision/F1 down); tighten sig_z if you use it.")
print("\nFigures written to:", HERE)
for f in ("fig_conn1_param_sweep.png", "fig_conn2_best_network.png",
          "fig_conn3_delay_distance.png"):
    print("  ", f)
