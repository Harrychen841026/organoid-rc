"""
Baselines + controls. A demo only means something against these.

  persistence_forecast : trivial "tomorrow = today" baseline. The organoid
                         system must beat this to claim it predicts anything.
  single_esn_forecast  : one monolithic software ESN over the whole ring (no
                         decomposition). Shows what an idealised silicon
                         reservoir achieves -> upper reference for the wetware.

Suggested additional controls (run with the same pipeline):
  * shuffled-input control : permute the input->electrode map. Encoding now
    carries no usable spatial info; valid prediction time should collapse.
  * spontaneous control    : skip stimulation, decode spontaneous spikes; the
    readout should fail (reservoir is not being driven by the data).
"""
import numpy as np
from reservoir import ESNReservoir
from readout import RidgeReadout


def persistence_forecast(warmup_traj, horizon):
    x = warmup_traj[-1]
    return np.repeat(x[None, :], horizon + 1, axis=0)


class MonolithicESNForecaster:
    """
    One ESN over the whole K-dim ring (no decomposition). Build/train once,
    forecast many windows. Reference for what an idealised silicon reservoir
    achieves vs the parallel wetware scheme.
    """
    def __init__(self, cfg, n_nodes=1200, seed=0):
        self.cfg = cfg
        self.res = ESNReservoir(in_dim=cfg.l96.K, n_nodes=n_nodes,
                                spectral_radius=cfg.esn.spectral_radius,
                                input_scaling=cfg.esn.input_scaling,
                                leak=cfg.esn.leak, seed=seed)
        self.ro = RidgeReadout(cfg.readout.ridge, cfg.readout.quadratic)

    def fit(self, train_traj, washout=150):
        self.res.reset()
        R, Y = [], []
        for t in range(len(train_traj) - 1):
            s = self.res.step(train_traj[t])
            if t >= washout:
                R.append(s); Y.append(train_traj[t + 1])
        self.ro.fit(np.array(R), np.array(Y))
        return self

    def forecast(self, warmup_traj, horizon):
        self.res.reset()
        for t in range(len(warmup_traj) - 1):
            self.res.step(warmup_traj[t])
        x = warmup_traj[-1].copy()
        pred = [x.copy()]
        for _ in range(horizon):
            x = self.ro.predict(self.res.step(x))[0]
            pred.append(x.copy())
        return np.array(pred)
