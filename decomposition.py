"""
Spatial decomposition + halo exchange.

Each well i owns q contiguous sites (its "core"). Its reservoir is fed those
q sites PLUS l buffer sites on each side, borrowed from the neighbouring wells
(indices wrap around the ring). The well outputs predictions for its core only.

The halo exchange is the assemble_inputs() step: each well's next input is
built from the full predicted state, i.e. its own predicted core plus the
neighbours' predicted boundary values. On the rig this assembly happens on the
host PC between frames -- the "digital relay" -- because the organoids are not
physically wired to each other.
"""
import numpy as np


class RingDecomposition:
    def __init__(self, K: int, g: int, q: int, l: int):
        assert g * q == K
        self.K, self.g, self.q, self.l = K, g, q, l
        self.in_dim = q + 2 * l
        self.out_dim = q
        # precompute index sets (mod K)
        self.core_idx = [np.arange(i * q, i * q + q) % K for i in range(g)]
        self.input_idx = [
            (np.arange(i * q - l, i * q + q + l)) % K for i in range(g)
        ]

    def core(self, i: int) -> np.ndarray:
        return self.core_idx[i]

    def split_cores(self, full_state: np.ndarray) -> list:
        """full_state (K,) -> list of g core targets (q,)."""
        return [full_state[self.core_idx[i]] for i in range(self.g)]

    def assemble_inputs(self, full_state: np.ndarray) -> list:
        """
        full_state (K,) -> list of g reservoir inputs (q+2l,).
        This IS the halo exchange: each well's input reads the boundary
        values that physically belong to its neighbours.
        """
        return [full_state[self.input_idx[i]] for i in range(self.g)]

    def gather_cores(self, core_preds: list) -> np.ndarray:
        """list of g predicted cores (q,) -> reassembled full state (K,)."""
        full = np.empty(self.K)
        for i in range(self.g):
            full[self.core_idx[i]] = core_preds[i]
        return full


if __name__ == "__main__":
    from config import CFG
    d = RingDecomposition(CFG.l96.K, CFG.decomp.g, CFG.decomp.q, CFG.decomp.l)
    print("input dim per well :", d.in_dim, " output dim per well:", d.out_dim)
    for i in range(d.g):
        print(f"well {i}: core={d.core_idx[i]}  input(window)={d.input_idx[i]}")
