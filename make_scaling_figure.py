"""
Scaling figure: fix the Lorenz-96 system (K=12) and vary the number of wells
g (= number of reservoirs). Each well is an identical 600-node reservoir, so
more wells = finer spatial decomposition = more total compute, exactly the
Pathak et al. Fig 5(b) experiment. ACTUALLY runs the model.

Outputs: fig4_scaling_wells.png   (run: python make_scaling_figure.py)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from lorenz96 import make_dataset, largest_lyapunov
from decomposition import RingDecomposition
from reservoir import ESNReservoir
from readout import RidgeReadout
from closed_loop import ParallelReservoirRC, normalized_rmse

cfg = CFG
lam = largest_lyapunov(cfg)
dt = cfg.l96.dt_pred
K = cfg.l96.K
L = cfg.decomp.l                     # buffer width, fixed
NODES = cfg.esn.n_nodes             # per well, identical across g

# divisors of K -> valid (g, q) with g*q = K
G_LIST = [g for g in range(1, K + 1) if K % g == 0]   # 1,2,3,4,6,12
WARM, HORIZON, NWIN, TRAIN = 60, 150, 10, 5000

train_raw = make_dataset(cfg, n_pred_steps=TRAIN, seed=1)
mean, std = train_raw.mean(), train_raw.std()
train = (train_raw - mean) / std


def build_train(g):
    q = K // g
    d = RingDecomposition(K, g, q, L)
    res = [ESNReservoir(d.in_dim, NODES, cfg.esn.spectral_radius,
                        cfg.esn.input_scaling, cfg.esn.leak, seed=i)
           for i in range(g)]
    ro = [RidgeReadout(cfg.readout.ridge, cfg.readout.quadratic) for _ in range(g)]
    model = ParallelReservoirRC(cfg, d, res, ro).train(train, washout=150)
    return model


# evaluate every g over the same forecast windows
windows = []
for s in range(NWIN):
    sg = (make_dataset(cfg, n_pred_steps=WARM + HORIZON, seed=300 + s) - mean) / std
    windows.append((sg[:WARM], sg[WARM - 1: WARM - 1 + HORIZON + 1]))

lead = np.arange(HORIZON + 1) * dt * lam
rmse_by_g, vt_by_g = {}, {}
for g in G_LIST:
    model = build_train(g)
    runs = [normalized_rmse(tf, model.forecast(w, HORIZON)) for w, tf in windows]
    arr = np.array(runs)
    rmse_by_g[g] = arr.mean(0)
    vts = []
    for r in arr:
        over = np.where(r > 0.3)[0]
        vts.append((over[0] if len(over) else len(r)) * dt * lam)
    vt_by_g[g] = (np.mean(vts), np.std(vts))
    print(f"g={g:2d} wells (q={K//g}): valid time = "
          f"{vt_by_g[g][0]:.2f} +/- {vt_by_g[g][1]:.2f} Lyapunov times")

# -------------------- plot --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(G_LIST)))
for c, g in zip(cmap, G_LIST):
    ax1.plot(lead, rmse_by_g[g], color=c, lw=2,
             label=f"g={g} well{'s' if g>1 else ''} (q={K//g})")
ax1.axhline(0.3, color="grey", ls=":", lw=1)
ax1.text(lead[-1], 0.31, "valid threshold", ha="right", va="bottom",
         fontsize=8, color="grey")
ax1.set_xlabel("forecast lead time (Lyapunov times)")
ax1.set_ylabel("normalised RMSE")
ax1.set_title("(a) Error growth vs number of wells")
ax1.set_ylim(0, 1.4); ax1.legend(frameon=False, fontsize=8)

gs = np.array(G_LIST)
means = np.array([vt_by_g[g][0] for g in G_LIST])
sds = np.array([vt_by_g[g][1] for g in G_LIST])
ax2.errorbar(gs, means, yerr=sds, marker="o", color="#185FA5", lw=2, capsize=3)
ax2.axvline(6, color="#993C1D", ls="--", lw=1)
ax2.text(6, ax2.get_ylim()[0], " MaxTwo = 6 wells", color="#993C1D",
         fontsize=8, va="bottom", ha="left", rotation=90)
ax2.set_xlabel("number of wells / reservoirs  g")
ax2.set_ylabel("valid prediction time (Lyapunov times)")
ax2.set_title("(b) Scaling: more wells -> longer valid forecast")
ax2.set_xticks(gs)

fig.suptitle("Fig 4. Parallel scaling for fixed Lorenz-96 (K=12), "
             f"identical {NODES}-node reservoirs, mean over {NWIN} windows",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig4_scaling_wells.png", dpi=140)
print("saved: fig4_scaling_wells.png")
