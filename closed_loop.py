"""
Orchestrates the parallel reservoir-computing experiment.

Two phases, exactly as in the theory:
  TRAIN    : teacher-forced. Drive every well with the TRUE local input,
             record its reservoir state, fit a per-well ridge readout so the
             state predicts the well's core one step ahead.
  FORECAST : autonomous / free-running. Each step, assemble every well's input
             from the current PREDICTED full state (the halo exchange), step
             each reservoir, decode -> predict each core, reassemble the full
             state, and feed it back in. This is the autoregressive loop.

Works identically with ESNReservoir (dry run) or OrganoidReservoir (hardware),
because both expose reset()/step(input)->state.
"""
import numpy as np


class ParallelReservoirRC:
    def __init__(self, cfg, decomp, reservoirs, readouts):
        self.cfg = cfg
        self.d = decomp
        self.res = reservoirs            # list length g
        self.ro = readouts               # list length g (RidgeReadout)
        assert len(reservoirs) == decomp.g == len(readouts)

    # ------------------------------------------------------------------
    def _drive(self, traj, collect=True, washout=20):
        """
        Teacher-force every reservoir along traj (n+1, K).
        Returns per-well (states, targets) lists if collect else None.
        """
        n = len(traj) - 1
        inputs = [self.d.assemble_inputs(traj[t]) for t in range(n)]
        targets = [self.d.split_cores(traj[t + 1]) for t in range(n)]
        for r in self.res:
            r.reset()
        states = [[] for _ in range(self.d.g)]
        ys = [[] for _ in range(self.d.g)]
        for t in range(n):
            for i in range(self.d.g):
                s = self.res[i].step(inputs[t][i])
                if collect and t >= washout:
                    states[i].append(s)
                    ys[i].append(targets[t][i])
        if not collect:
            return None
        return ([np.array(states[i]) for i in range(self.d.g)],
                [np.array(ys[i]) for i in range(self.d.g)])

    def train(self, train_traj, washout=20):
        states, ys = self._drive(train_traj, collect=True, washout=washout)
        for i in range(self.d.g):
            self.ro[i].fit(states[i], ys[i])
        return self

    # ------------------------------------------------------------------
    def forecast(self, warmup_traj, horizon):
        """
        warmup_traj (w+1, K): true states used to set the reservoir memory.
        Returns predicted trajectory (horizon+1, K), starting at warmup_traj[-1].
        """
        for r in self.res:
            r.reset()
        # teacher-force the warmup to load reservoir memory (echo-state)
        for t in range(len(warmup_traj) - 1):
            inp = self.d.assemble_inputs(warmup_traj[t])
            for i in range(self.d.g):
                self.res[i].step(inp[i])

        x = warmup_traj[-1].copy()
        pred = [x.copy()]
        for _ in range(horizon):
            inp = self.d.assemble_inputs(x)          # <-- halo exchange
            cores = []
            for i in range(self.d.g):
                s = self.res[i].step(inp[i])
                cores.append(self.ro[i].predict(s)[0])
            x = self.d.gather_cores(cores)
            pred.append(x.copy())
        return np.array(pred)


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def normalized_rmse(true, pred):
    """RMSE(t) normalised by the climatological std of the system."""
    sigma = true.std()
    err = np.sqrt(((pred - true) ** 2).mean(axis=1))
    return err / sigma


def valid_prediction_time(true, pred, dt_pred, lyap, threshold=0.3):
    """First lead time (in Lyapunov times) where normalised RMSE exceeds thr."""
    nr = normalized_rmse(true, pred)
    over = np.where(nr > threshold)[0]
    steps = over[0] if len(over) else len(nr)
    return steps * dt_pred * lyap, nr
