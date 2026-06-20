"""
Linear / quadratic ridge readout -- the ONLY trained part of the system
(the reservoir, organoid or ESN, is fixed).

Mirrors Pathak et al.:  output = P1 . r + P2 . r^2   (quadratic breaks the
spurious tanh sign-symmetry; for organoids it also helps because firing-rate
features are non-negative and benefit from a second-order term).

One readout is trained PER WELL, independently (the inhomogeneous / mu != 0
case in the paper: organoids differ, so weights are not shared).
"""
import numpy as np


class RidgeReadout:
    def __init__(self, ridge: float = 1e-4, quadratic: bool = True):
        self.ridge = ridge
        self.quadratic = quadratic
        self.W = None       # (out_dim, feat_dim)
        self.mu = None
        self.sd = None

    def _features(self, R: np.ndarray) -> np.ndarray:
        """R: (n_samples, state_dim) -> design matrix with bias (+ r^2)."""
        if self.mu is None:
            self.mu = R.mean(0)
            self.sd = R.std(0) + 1e-8
        Rn = (R - self.mu) / self.sd
        feats = [np.ones((R.shape[0], 1)), Rn]
        if self.quadratic:
            feats.append(Rn ** 2)
        return np.hstack(feats)

    def fit(self, R: np.ndarray, Y: np.ndarray):
        """R: (n, state_dim) reservoir states ; Y: (n, out_dim) targets."""
        X = self._features(R)
        n_feat = X.shape[1]
        A = X.T @ X + self.ridge * np.eye(n_feat)
        B = X.T @ Y
        self.W = np.linalg.solve(A, B).T            # (out_dim, feat_dim)
        return self

    def predict(self, R: np.ndarray) -> np.ndarray:
        # reuse stored normalisation (do not recompute mu/sd at test time)
        Rn = (np.atleast_2d(R) - self.mu) / self.sd
        feats = [np.ones((Rn.shape[0], 1)), Rn]
        if self.quadratic:
            feats.append(Rn ** 2)
        X = np.hstack(feats)
        return (X @ self.W.T)
