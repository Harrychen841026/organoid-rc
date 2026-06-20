"""
Lorenz-96 simulator + utilities.

    dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F      (indices mod K)

This is the synthetic "ground truth" the organoids learn to forecast.
It is a ring with purely local coupling -> ideal for spatial decomposition.
"""
import numpy as np


def l96_rhs(x: np.ndarray, F: float) -> np.ndarray:
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def rk4_step(x: np.ndarray, F: float, dt: float) -> np.ndarray:
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(K: int, F: float, dt_int: float, n_steps: int,
             x0: np.ndarray = None, seed: int = 0) -> np.ndarray:
    """Return trajectory of shape (n_steps+1, K) at the integration step."""
    rng = np.random.default_rng(seed)
    x = (F + 0.01 * rng.standard_normal(K)) if x0 is None else x0.copy()
    traj = np.empty((n_steps + 1, K))
    traj[0] = x
    for t in range(n_steps):
        x = rk4_step(x, F, dt_int)
        traj[t + 1] = x
    return traj


def make_dataset(cfg, n_pred_steps: int, seed: int = 0):
    """
    Generate a trajectory sampled at the RC prediction step dt_pred.
    Returns array (n_pred_steps+1, K). Spin-up is discarded.
    """
    l96 = cfg.l96
    sub = max(1, round(l96.dt_pred / l96.dt_int))   # integration steps per RC step
    spin_steps = round(l96.spinup / l96.dt_int)
    total_int = spin_steps + n_pred_steps * sub
    full = simulate(l96.K, l96.F, l96.dt_int, total_int, seed=seed)
    sampled = full[spin_steps::sub][: n_pred_steps + 1]
    return sampled


def largest_lyapunov(cfg, t_total: float = 200.0, d0: float = 1e-8) -> float:
    """
    Benettin algorithm: estimate the largest Lyapunov exponent (1/MTU).
    Lyapunov time = 1 / lambda_max ; valid-forecast horizons are reported
    in Lyapunov times so results are comparable to the Pathak paper.
    """
    l96 = cfg.l96
    dt = l96.dt_int
    n = round(t_total / dt)
    x = simulate(l96.K, l96.F, dt, round(l96.spinup / dt), seed=1)[-1]
    xp = x + d0 * np.ones(l96.K) / np.sqrt(l96.K)
    s = 0.0
    for _ in range(n):
        x = rk4_step(x, l96.F, dt)
        xp = rk4_step(xp, l96.F, dt)
        d = np.linalg.norm(xp - x)
        s += np.log(d / d0)
        xp = x + (d0 / d) * (xp - x)          # renormalise
    lam = s / (n * dt)
    return lam


if __name__ == "__main__":
    from config import CFG
    lam = largest_lyapunov(CFG)
    print(f"K={CFG.l96.K} F={CFG.l96.F}")
    print(f"largest Lyapunov exponent  ~ {lam:.3f} / MTU")
    print(f"Lyapunov time              ~ {1/lam:.3f} MTU")
    print(f"RC prediction step dt_pred = {CFG.l96.dt_pred} MTU "
          f"= {CFG.l96.dt_pred*lam:.3f} Lyapunov times/step")
