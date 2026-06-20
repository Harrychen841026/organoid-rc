"""
Reservoir backends behind a common interface.

    Reservoir.reset()
    state = Reservoir.step(input_vec)     # advance ONE rc frame, return state

Two implementations:
  * ESNReservoir          -- in-silico echo-state network. The "organoid
                             surrogate": run the entire pipeline end-to-end on
                             a laptop before touching wetware.
  * OrganoidReservoir     -- hardware backend. step() = encode input as stimulus
                             -> stimulate a well -> record evoked spikes ->
                             decode to a state vector. Wraps encoding/decoding
                             and the MaxLab Live API (see encoding.py/decoding.py).

Swapping ESN <-> Organoid changes nothing in closed_loop.py: that is the whole
point of treating the reservoir as a fixed nonlinear feature map.
"""
import numpy as np


class ESNReservoir:
    """Standard leaky-integrator ESN. Stands in for one organoid/well."""
    def __init__(self, in_dim, n_nodes=300, spectral_radius=0.9,
                 input_scaling=0.5, leak=0.3, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_nodes, n_nodes))
        # sparse-ish + scale to desired spectral radius
        W[rng.random((n_nodes, n_nodes)) > 0.1] = 0.0
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / eig)
        self.Win = input_scaling * rng.uniform(-1, 1, (n_nodes, in_dim))
        self.leak = leak
        self.n_nodes = n_nodes
        self.state_dim = n_nodes
        self.reset()

    def reset(self):
        self.r = np.zeros(self.n_nodes)

    def step(self, u: np.ndarray) -> np.ndarray:
        pre = self.W @ self.r + self.Win @ np.asarray(u, float)
        self.r = (1 - self.leak) * self.r + self.leak * np.tanh(pre)
        return self.r.copy()


class OrganoidReservoir:
    """
    Hardware backend for ONE well. Fixed nonlinear reservoir = living organoid.

    NOTE: requires the MaxLab Live API + a connected MaxTwo. The encode/record/
    decode calls are isolated in encoding.py and decoding.py; fill the TODOs
    there for your rig. Until then, use ESNReservoir for dry runs.
    """
    def __init__(self, well_id, cfg, encoder, decoder, session):
        self.well_id = well_id
        self.cfg = cfg
        self.encoder = encoder      # encoding.StimEncoder
        self.decoder = decoder      # decoding.SpikeDecoder
        self.session = session      # hardware.MaxLabSession (open connection)
        self.state_dim = (cfg.hw.readout_channels_per_well *
                          cfg.decode.n_state_bins)

    def reset(self):
        # optional: deliver a quiet/washout period to settle the network
        self.session.quiet(self.well_id, self.cfg.timing.washout_ms)

    def step(self, u: np.ndarray) -> np.ndarray:
        seq = self.encoder.encode(self.well_id, u)          # input -> stimulus
        spikes = self.session.stimulate_and_record(         # apply + record
            self.well_id, seq,
            record_ms=self.cfg.timing.response_window_ms)
        state = self.decoder.decode(spikes)                 # spikes -> state vec
        self.session.quiet(self.well_id, self.cfg.timing.washout_ms)
        return state


# ---------------------------------------------------------------------------
# Wiring: connectivity.IOSelection  ->  per-well organoid reservoirs
# ---------------------------------------------------------------------------
# `selections` is {well_id: connectivity.IOSelection} produced by
# connectivity.select_for_well(network_assay_data, CFG). Its fields plug in
# directly: .input_electrodes -> stimulation, .output_channels -> readout order.
from encoding import StimEncoder                      # noqa: E402
from decoding import SpikeDecoder                     # noqa: E402


def build_encoder(cfg, selections, session=None, normalizer=None):
    """One StimEncoder shared across wells, using the routed stim handles."""
    stim_map = getattr(session, "well_stim_map", None) or {}
    if not stim_map:        # no live session: use the selected electrode ids
        stim_map = {w: list(sel.input_electrodes) for w, sel in selections.items()}
    return StimEncoder(cfg, normalizer, stim_map)


def build_decoders(cfg, selections):
    """One SpikeDecoder per well, keyed on that well's chosen readout channels."""
    return {w: SpikeDecoder(cfg, sel.output_channels)
            for w, sel in selections.items()}


def build_organoid_reservoirs(cfg, selections, session, normalizer):
    """
    Turn the connectivity output into a list of OrganoidReservoir (one per well),
    ready to hand to closed_loop.ParallelReservoirRC -- exactly where the ESN
    surrogates go in run_demo.py.

      selections : {well_id: connectivity.IOSelection}
      session    : an opened + configure_from_selection()'d MaxLabSession
      normalizer : encoding.InputNormalizer fit on the assembled inputs
    """
    enc = build_encoder(cfg, selections, session, normalizer)
    decs = build_decoders(cfg, selections)
    return [OrganoidReservoir(w, cfg, enc, decs[w], session)
            for w in range(cfg.decomp.g)]
