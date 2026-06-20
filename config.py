"""
Central configuration for the parallel organoid reservoir-computing demo.

Target system : Lorenz-96 (toy weather model, local coupling, periodic ring).
Hardware      : MaxWell MaxTwo, 6-well HD-MEA. One organoid per well = one reservoir.
Scheme        : spatial decomposition + halo (boundary) exchange, after
                Pathak, Hunt, Girvan, Lu & Ott, PRL 120, 024102 (2018).

All timings/electrode counts are starting points -- tune them against the
characterisation you run in Phase 0 (see the protocol document).
"""
from dataclasses import dataclass, field
from typing import List


# ----------------------------------------------------------------------
# 1. Target dynamical system: Lorenz-96
# ----------------------------------------------------------------------
@dataclass
class Lorenz96Config:
    K: int = 96          # number of sites on the ring (== g * q)
    F: float = 8.0       # forcing; F=8 is firmly chaotic
    dt_int: float = 0.01 # RK4 integration step (model time units, MTU)
    # One RC prediction step = how far ahead each reservoir forecasts.
    # Chosen so a step is a meaningful fraction of a Lyapunov time.
    dt_pred: float = 0.05
    spinup: float = 20.0 # MTU discarded so trajectories sit on the attractor


# ----------------------------------------------------------------------
# 2. Spatial decomposition across the 6 wells
# ----------------------------------------------------------------------
@dataclass
class DecompositionConfig:
    g: int = 6           # number of reservoirs == number of wells
    q: int = 16           # core sites each well predicts  (g*q must equal K)
    l: int = 4           # buffer (halo) width on EACH side; covers L96 stencil

    @property
    def in_dim(self) -> int:
        """reservoir input dimension per well = q + 2l (output dim = q)."""
        return self.q + 2 * self.l


# ----------------------------------------------------------------------
# 3. Hardware (MaxTwo / MaxLab Live) -- per well
# ----------------------------------------------------------------------
@dataclass
class HardwareConfig:
    n_wells: int = 6
    fs_hz: int = 20_000          # acquisition sampling rate
    max_record_channels: int = 1024
    max_stim_units: int = 32

    # Electrodes used to INJECT the encoded input (one cluster per input dim).
    # input dim = q + 2l = 6 by default -> 6 stim clusters per well.
    stim_clusters_per_well: int = 24
    electrodes_per_stim_cluster: int = 1     # raise for redundancy

    # Biphasic, charge-balanced pulse (safety: keep charge/phase modest).
    pulse_phase_us: int = 200                # per phase
    pulse_amp_mV_range: tuple = (0.0, 400.0) # amplitude-coded range (cathodic-first)
    # rate coding alternative: pulses delivered within the stim window
    pulse_rate_hz_range: tuple = (5.0, 100.0)
    encoding_mode: str = "rate"              # "rate" | "amplitude"

    # Readout electrodes: the reservoir state vector is built from these.
    readout_channels_per_well: int = 256     # <= max_record_channels


# ----------------------------------------------------------------------
# 4. Frame timing  (maps one RC step onto biological time)
# ----------------------------------------------------------------------
@dataclass
class FrameTimingConfig:
    stim_window_ms: float = 50.0     # deliver the encoded stimulus
    response_window_ms: float = 200.0# integrate the evoked spiking response
    washout_ms: float = 50.0         # quiet gap so state reflects current input
    # one frame = stim + response + washout (+ host compute/relay latency)
    @property
    def frame_ms(self) -> float:
        return self.stim_window_ms + self.response_window_ms + self.washout_ms


# ----------------------------------------------------------------------
# 5. Decoding (spikes -> reservoir state vector)
# ----------------------------------------------------------------------
@dataclass
class DecodingConfig:
    n_state_bins: int = 4            # time bins inside the response window
    feature: str = "rate"           # "rate" (spikes/bin) | "count"
    smooth_ms: float = 5.0          # Gaussian smoothing of spike trains
    # state dim per well = readout_channels_per_well * n_state_bins


# ----------------------------------------------------------------------
# 6. Readout (linear / quadratic ridge regression)
# ----------------------------------------------------------------------
@dataclass
class ReadoutConfig:
    ridge: float = 1e-4             # Tikhonov regularisation
    quadratic: bool = True         # use [r, r^2] features (Pathak P1 r + P2 r^2)


# ----------------------------------------------------------------------
# 7. ESN surrogate (in-silico stand-in for an organoid, for dry runs)
# ----------------------------------------------------------------------
@dataclass
class ESNSurrogateConfig:
    n_nodes: int = 600             # per well
    spectral_radius: float = 0.5
    input_scaling: float = 0.5
    leak: float = 1.0              # no leak: respond fully to current input
    seed: int = 0


@dataclass
class ExperimentConfig:
    l96: Lorenz96Config = field(default_factory=Lorenz96Config)
    decomp: DecompositionConfig = field(default_factory=DecompositionConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    timing: FrameTimingConfig = field(default_factory=FrameTimingConfig)
    decode: DecodingConfig = field(default_factory=DecodingConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    esn: ESNSurrogateConfig = field(default_factory=ESNSurrogateConfig)

    def validate(self):
        assert self.decomp.g * self.decomp.q == self.l96.K, \
            "g*q must equal K (every site owned by exactly one well)"
        assert self.decomp.g == self.hw.n_wells, "one reservoir per well"
        in_dim = self.decomp.q + 2 * self.decomp.l
        assert in_dim == self.hw.stim_clusters_per_well, \
            "stim clusters per well must equal input dim (q+2l)"
        return self


CFG = ExperimentConfig().validate()
