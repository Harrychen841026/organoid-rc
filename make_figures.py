"""
Generate the key figures for this task by ACTUALLY running the model.
Outputs PNGs next to this file. Run: python make_figures.py
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
from closed_loop import ParallelReservoirRC, normalized_rmse, valid_prediction_time
from baselines import persistence_forecast, MonolithicESNForecaster

cfg = CFG
lam = largest_lyapunov(cfg)
LT = 1.0 / lam
dt = cfg.l96.dt_pred

# ---- data (standardised for the reservoir; de-standardised for plots) ----
train_raw = make_dataset(cfg, n_pred_steps=8000, seed=1)
mean, std = train_raw.mean(), train_raw.std()
train = (train_raw - mean) / std

d = RingDecomposition(cfg.l96.K, cfg.decomp.g, cfg.decomp.q, cfg.decomp.l)
reservoirs = [ESNReservoir(d.in_dim, cfg.esn.n_nodes, cfg.esn.spectral_radius,
                           cfg.esn.input_scaling, cfg.esn.leak, seed=cfg.esn.seed + i)
              for i in range(cfg.decomp.g)]
readouts = [RidgeReadout(cfg.readout.ridge, cfg.readout.quadratic)
            for _ in range(cfg.decomp.g)]
model = ParallelReservoirRC(cfg, d, reservoirs, readouts).train(train, washout=150)
mono = MonolithicESNForecaster(cfg, n_nodes=1200).fit(train, washout=150)

warm_len, horizon = 60, 200


def de(x):           # back to physical Lorenz-96 units
    return x * std + mean


# ============================================================
# FIG 1 -- Lorenz-96 space-time (Hovmoller) : the target system
# ============================================================
long = make_dataset(cfg, n_pred_steps=400, seed=3)
fig, ax = plt.subplots(figsize=(9, 3.2))
im = ax.imshow(long.T, aspect="auto", cmap="RdBu_r", origin="lower",
               extent=[0, 400 * dt * lam, 0, cfg.l96.K])
ax.set_xlabel("time (Lyapunov times)")
ax.set_ylabel("site k")
ax.set_title("Fig 1. Lorenz-96 target (K=12, F=8): chaotic space-time evolution")
fig.colorbar(im, ax=ax, label="X_k", pad=0.01)
fig.tight_layout(); fig.savefig("fig1_lorenz96_spacetime.png", dpi=140)
plt.close(fig)

# ============================================================
# FIG 2 -- truth / prediction / error heatmaps (Pathak Fig 4 style)
# ============================================================
seg = (make_dataset(cfg, n_pred_steps=warm_len + horizon, seed=107) - mean) / std
warm_seg = seg[:warm_len]
true_future = seg[warm_len - 1: warm_len - 1 + horizon + 1]
pred = model.forecast(warm_seg, horizon)

T, P = de(true_future).T, de(pred).T
E = P - T
xext = [0, (horizon) * dt * lam, 0, cfg.l96.K]
vmax = np.abs(T).max()

fig, axs = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
for ax, dat, ttl, cm, vm in [
        (axs[0], T, "(a) Truth", "RdBu_r", vmax),
        (axs[1], P, "(b) Parallel organoid-reservoir forecast (6 wells)", "RdBu_r", vmax),
        (axs[2], E, "(c) Error (forecast - truth)", "PuOr_r", vmax)]:
    im = ax.imshow(dat, aspect="auto", cmap=cm, origin="lower", extent=xext,
                   vmin=-vm, vmax=vm)
    ax.set_ylabel("site k"); ax.set_title(ttl, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01)
vt, _ = valid_prediction_time(true_future, pred, dt, lam)
axs[2].axvline(vt, color="k", ls="--", lw=1)
axs[2].set_xlabel("forecast lead time (Lyapunov times)")
fig.suptitle(f"Fig 2. Forecast vs truth (valid to ~{vt:.1f} Lyapunov times, "
             "dashed line)", fontsize=11)
fig.tight_layout(); fig.savefig("fig2_forecast_truth_error.png", dpi=140)
plt.close(fig)

# ============================================================
# FIG 3 -- normalised RMSE growth vs lead time (parallel vs baselines)
# ============================================================
nseeds = 12
lead = np.arange(horizon + 1) * dt * lam
curves = {"parallel (6 wells)": [], "monolithic ESN": [], "persistence": []}
for s in range(nseeds):
    sg = (make_dataset(cfg, n_pred_steps=warm_len + horizon, seed=200 + s) - mean) / std
    w = sg[:warm_len]; tf = sg[warm_len - 1: warm_len - 1 + horizon + 1]
    curves["parallel (6 wells)"].append(normalized_rmse(tf, model.forecast(w, horizon)))
    curves["monolithic ESN"].append(normalized_rmse(tf, mono.forecast(w, horizon)))
    curves["persistence"].append(normalized_rmse(tf, persistence_forecast(w, horizon)))

fig, ax = plt.subplots(figsize=(8, 4.6))
colors = {"parallel (6 wells)": "#185FA5", "monolithic ESN": "#1D9E75",
          "persistence": "#993C1D"}
for k, runs in curves.items():
    arr = np.array(runs)
    m = arr.mean(0); sd = arr.std(0)
    ax.plot(lead, m, color=colors[k], lw=2, label=k)
    ax.fill_between(lead, m - sd, m + sd, color=colors[k], alpha=0.15)
ax.axhline(0.3, color="grey", ls=":", lw=1)
ax.text(lead[-1], 0.31, "valid-time threshold (0.3)", ha="right", va="bottom",
        fontsize=8, color="grey")
ax.set_xlabel("forecast lead time (Lyapunov times)")
ax.set_ylabel("normalised RMSE")
ax.set_title(f"Fig 3. Forecast error growth (mean +/- sd over {nseeds} windows)")
ax.set_ylim(0, 1.4); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("fig3_rmse_vs_leadtime.png", dpi=140)
plt.close(fig)

# print the numbers behind the figures
print(f"Lyapunov time = {LT:.3f} MTU  ({dt*lam:.3f} Lyap-times per step)")
print("valid prediction time (Lyapunov times), mean over windows:")
for k, runs in curves.items():
    arr = np.array(runs)
    vts = []
    for r in arr:
        over = np.where(r > 0.3)[0]
        vts.append((over[0] if len(over) else len(r)) * dt * lam)
    print(f"  {k:20s}: {np.mean(vts):.2f} +/- {np.std(vts):.2f}")
print("saved: fig1_lorenz96_spacetime.png, fig2_forecast_truth_error.png, "
      "fig3_rmse_vs_leadtime.png")
