"""
Grid sweep over q (sites per well) and l (halo overlap), on the FULL 6-well
plate (g=6 fixed, so the system size K = 6*q). Every combo is checked against
the MaxTwo spec: <= 6 wells and input dim (q+2l) <= 32 stimulation sites.

Outputs:
  fig5_ql_grid_curves.png  -- a Fig-3-style RMSE-vs-leadtime plot per (q,l)
  fig6_ql_heatmap.png      -- valid prediction time over the (q,l) grid
Run: python make_ql_grid.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from lorenz96 import simulate, rk4_step
from decomposition import RingDecomposition
from reservoir import ESNReservoir
from readout import RidgeReadout
from closed_loop import ParallelReservoirRC, normalized_rmse
from baselines import persistence_forecast

# ---- fixed hardware / sweep ranges ----
G = 6                      # all 6 wells
Q_LIST = [1, 2, 3, 4]      # sites per well  -> K = 6*q in {6,12,18,24}
L_LIST = [0, 1, 2, 3]      # halo width each side
MAX_STIM = 32              # MaxTwo stim sites per well
F = CFG.l96.F
DT_INT, DT_PRED, SPIN = CFG.l96.dt_int, CFG.l96.dt_pred, CFG.l96.spinup
NODES, TRAIN, NWIN, HORIZON, WARM = 200, 2000, 5, 110, 50


def gen(K, n_pred, seed):
    sub = max(1, round(DT_PRED / DT_INT))
    spin = round(SPIN / DT_INT)
    full = simulate(K, F, DT_INT, spin + n_pred * sub, seed=seed)
    return full[spin::sub][: n_pred + 1]


def lyap(K, t_total=150.0, d0=1e-8):
    dt = DT_INT
    n = round(t_total / dt)
    x = simulate(K, F, dt, round(SPIN / dt), seed=1)[-1]
    xp = x + d0 * np.ones(K) / np.sqrt(K)
    s = 0.0
    for _ in range(n):
        x = rk4_step(x, F, dt); xp = rk4_step(xp, F, dt)
        dd = np.linalg.norm(xp - x); s += np.log(dd / d0)
        xp = x + (d0 / dd) * (xp - x)
    return s / (n * dt)


# cache per-q data (depends only on K=6q, not l)
cache = {}
for q in Q_LIST:
    K = G * q
    lam = lyap(K)
    train_raw = gen(K, TRAIN, seed=1)
    m, s = train_raw.mean(), train_raw.std()
    train = (train_raw - m) / s
    wins = []
    for sd in range(NWIN):
        sg = (gen(K, WARM + HORIZON, seed=300 + sd) - m) / s
        wins.append((sg[:WARM], sg[WARM - 1: WARM - 1 + HORIZON + 1]))
    cache[q] = dict(K=K, lam=lam, train=train, wins=wins)

lead_by_q = {q: np.arange(HORIZON + 1) * DT_PRED * cache[q]["lam"] for q in Q_LIST}

rmse_grid = {}     # (q,l) -> (mean parallel curve, mean persistence curve)
vt_grid = np.full((len(Q_LIST), len(L_LIST)), np.nan)

for qi, q in enumerate(Q_LIST):
    K, lam, train, wins = (cache[q][k] for k in ("K", "lam", "train", "wins"))
    for li, l in enumerate(L_LIST):
        in_dim = q + 2 * l
        if in_dim > MAX_STIM:                 # spec check
            continue
        d = RingDecomposition(K, G, q, l)
        res = [ESNReservoir(d.in_dim, NODES, CFG.esn.spectral_radius,
                            CFG.esn.input_scaling, CFG.esn.leak, seed=i)
               for i in range(G)]
        ro = [RidgeReadout(CFG.readout.ridge, CFG.readout.quadratic)
              for _ in range(G)]
        model = ParallelReservoirRC(CFG, d, res, ro).train(train, washout=120)

        par, per = [], []
        for w, tf in wins:
            par.append(normalized_rmse(tf, model.forecast(w, HORIZON)))
            per.append(normalized_rmse(tf, persistence_forecast(w, HORIZON)))
        par_m = np.array(par).mean(0)
        rmse_grid[(q, l)] = (par_m, np.array(per).mean(0))
        over = np.where(par_m > 0.3)[0]
        vt = (over[0] if len(over) else len(par_m)) * DT_PRED * lam
        vt_grid[qi, li] = vt
        print(f"q={q} (K={K}) l={l}  in_dim={in_dim:2d}  vt={vt:.2f} Lyap")

# ---------------- FIG 5: grid of RMSE curves ----------------
fig, axs = plt.subplots(len(Q_LIST), len(L_LIST), figsize=(12, 10),
                        sharex=False, sharey=True)
for qi, q in enumerate(Q_LIST):
    lead = lead_by_q[q]
    for li, l in enumerate(L_LIST):
        ax = axs[qi, li]
        if (q, l) in rmse_grid:
            par_m, per_m = rmse_grid[(q, l)]
            ax.plot(lead, par_m, color="#185FA5", lw=2, label="parallel")
            ax.plot(lead, per_m, color="#993C1D", lw=1.2, ls="--", label="persistence")
            ax.axhline(0.3, color="grey", ls=":", lw=0.8)
            ax.set_title(f"q={q} (K={6*q}), l={l}  ·  vt={vt_grid[qi,li]:.2f}",
                         fontsize=9)
        else:
            ax.text(0.5, 0.5, f"q+2l={q+2*l}\n> 32 sites", ha="center",
                    va="center", fontsize=9, color="#993C1D")
        ax.set_ylim(0, 1.4)
        if qi == len(Q_LIST) - 1:
            ax.set_xlabel("lead (Lyap times)", fontsize=8)
        if li == 0:
            ax.set_ylabel("norm RMSE", fontsize=8)
axs[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Fig 5. Forecast error vs lead time across q (sites/well) and "
             "l (overlap), g=6 wells fixed", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("fig5_ql_grid_curves.png", dpi=130)
plt.close(fig)

# ---------------- FIG 6: valid-time heatmap ----------------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(vt_grid, cmap="viridis", origin="lower", aspect="auto")
ax.set_xticks(range(len(L_LIST))); ax.set_xticklabels(L_LIST)
ax.set_yticks(range(len(Q_LIST))); ax.set_yticklabels([f"{q} (K={6*q})" for q in Q_LIST])
ax.set_xlabel("overlap  l  (halo sites each side)")
ax.set_ylabel("sites per well  q")
ax.set_title("Fig 6. Valid prediction time (Lyapunov times)\ng=6 wells, "
             "color & label = vt")
for qi in range(len(Q_LIST)):
    for li in range(len(L_LIST)):
        v = vt_grid[qi, li]
        txt = "n/a" if np.isnan(v) else f"{v:.2f}"
        ax.text(li, qi, txt, ha="center", va="center",
                color="white" if (np.isnan(v) or v < np.nanmax(vt_grid) * 0.6) else "black",
                fontsize=10)
fig.colorbar(im, ax=ax, label="valid time (Lyapunov times)")
fig.tight_layout()
fig.savefig("fig6_ql_heatmap.png", dpi=130)
plt.close(fig)
print("DONE: fig5_ql_grid_curves.png, fig6_ql_heatmap.png")
