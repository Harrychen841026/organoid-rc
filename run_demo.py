"""
End-to-end IN-SILICO dry run of the full experiment.

Each of the 6 wells is an ESN "organoid surrogate". This validates the entire
architecture -- decomposition, halo exchange, per-well training, autoregressive
forecasting, metrics -- on a laptop, with NO hardware. When the rig is ready,
swap ESNReservoir for OrganoidReservoir in build_reservoirs() and the rest is
unchanged.

Run:  python run_demo.py
"""
import numpy as np

from config import CFG
from lorenz96 import make_dataset, largest_lyapunov
from decomposition import RingDecomposition
from reservoir import ESNReservoir
from readout import RidgeReadout
from closed_loop import ParallelReservoirRC, valid_prediction_time
from baselines import persistence_forecast, MonolithicESNForecaster


def build_reservoirs(cfg, in_dim):
    return [ESNReservoir(in_dim=in_dim,
                         n_nodes=cfg.esn.n_nodes,
                         spectral_radius=cfg.esn.spectral_radius,
                         input_scaling=cfg.esn.input_scaling,
                         leak=cfg.esn.leak,
                         seed=cfg.esn.seed + i)
            for i in range(cfg.decomp.g)]


def main():
    cfg = CFG
    rng = np.random.default_rng(42)

    print("== Lorenz-96 target ==")
    lam = largest_lyapunov(cfg)
    print(f"  lambda_max ~ {lam:.3f}/MTU  ->  Lyapunov time ~ {1/lam:.3f} MTU")

    # data (standardised: raw L96 values ~ +-10 would saturate tanh / the
    # neuronal dynamic range; the rig encoder does the same job via to_unit()).
    train_raw = make_dataset(cfg, n_pred_steps=8000, seed=1)
    mean, std = train_raw.mean(), train_raw.std()
    train = (train_raw - mean) / std
    horizon = 200
    warm_len = 60

    # model
    d = RingDecomposition(cfg.l96.K, cfg.decomp.g, cfg.decomp.q, cfg.decomp.l)
    reservoirs = build_reservoirs(cfg, d.in_dim)
    readouts = [RidgeReadout(cfg.readout.ridge, cfg.readout.quadratic)
                for _ in range(cfg.decomp.g)]
    model = ParallelReservoirRC(cfg, d, reservoirs, readouts)

    print("\n== Training 6 per-well readouts (teacher forced) ==")
    model.train(train, washout=150)
    mono = MonolithicESNForecaster(cfg, n_nodes=1200).fit(train, washout=150)

    # evaluate over several independent forecast windows for a stable number
    print("\n== Valid prediction time (Lyapunov times, mean over windows) ==")
    vts, vts_p, vts_b = [], [], []
    for seed in range(8):
        seg = make_dataset(cfg, n_pred_steps=warm_len + horizon, seed=100 + seed)
        seg = (seg - mean) / std
        warm_seg = seg[:warm_len]
        true_future = seg[warm_len - 1: warm_len - 1 + horizon + 1]
        vts.append(valid_prediction_time(
            true_future, model.forecast(warm_seg, horizon),
            cfg.l96.dt_pred, lam)[0])
        vts_p.append(valid_prediction_time(
            true_future, persistence_forecast(warm_seg, horizon),
            cfg.l96.dt_pred, lam)[0])
        vts_b.append(valid_prediction_time(
            true_future, mono.forecast(warm_seg, horizon),
            cfg.l96.dt_pred, lam)[0])

    print(f"  parallel reservoir (6 wells) : {np.mean(vts):5.2f}  +/- {np.std(vts):.2f}")
    print(f"  monolithic single ESN        : {np.mean(vts_b):5.2f}  +/- {np.std(vts_b):.2f}")
    print(f"  persistence baseline         : {np.mean(vts_p):5.2f}  +/- {np.std(vts_p):.2f}")
    print("\n(The parallel system beating persistence and tracking the "
          "monolithic ESN confirms the decomposition + halo exchange works.)")


if __name__ == "__main__":
    main()
