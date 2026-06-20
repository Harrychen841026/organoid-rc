"""
HARDWARE entry point: forecast Lorenz-96 on real organoids, using electrodes
chosen by connectivity.py. This mirrors run_demo.py but swaps the ESN surrogates
for OrganoidReservoir backends. Must run on the rig PC (MaxLab Live + MaxTwo).

Pipeline (the connectivity output flows straight through):
  1. Record a Network assay per well -> connectivity.load_network_assay(...)
  2. connectivity.select_for_well(data, CFG) -> IOSelection per well
     (.input_electrodes = stim sites, .output_channels = readout order)
  3. MaxLabSession.configure_from_selection({well: sel})  <-- direct hand-off
  4. reservoir.build_organoid_reservoirs(CFG, selections, session, normalizer)
  5. closed_loop.ParallelReservoirRC(...).train(...).forecast(...)

Nothing below knows it's talking to wetware rather than an ESN.
"""
import numpy as np

from config import CFG
from lorenz96 import make_dataset
from decomposition import RingDecomposition
from readout import RidgeReadout
from encoding import InputNormalizer
from closed_loop import ParallelReservoirRC, valid_prediction_time
import connectivity
import hardware
import reservoir


def select_electrodes_per_well(cfg):
    """
    Returns {well_id: connectivity.IOSelection}. Replace the loader with your
    real per-well Network-assay .h5 files. select_for_well sizes the picks to
    CFG (n_inputs = q+2l, n_outputs = readout_channels_per_well).
    """
    selections = {}
    for well in range(cfg.decomp.g):
        # --- REAL RIG: load this well's Network-assay recording ---
        # data = connectivity.load_network_assay(
        #     f"/path/well{well}_network.raw.h5", well=f"well{well:03d}")
        # --- SCAFFOLD: synthetic stand-in so the flow is runnable offline ---
        data = connectivity._synthesize_demo(n_ch=400, duration=120.0, seed=well)
        sel = connectivity.select_for_well(
            data, cfg,
            bin_ms=5.0, min_rate_hz=0.0, input_min_spacing_um=100.0,
            n_surrogates=10, rng_seed=well)
        selections[well] = sel
        print(f"well {well}: {len(sel.input_electrodes)} stim electrodes, "
              f"{len(sel.output_channels)} readout channels")
    return selections


def main():
    cfg = CFG
    d = RingDecomposition(cfg.l96.K, cfg.decomp.g, cfg.decomp.q, cfg.decomp.l)

    # data
    train = make_dataset(cfg, n_pred_steps=2000, seed=1)
    m, s = train.mean(), train.std()
    train = (train - m) / s

    # 1-2. choose electrodes from functional connectivity (one IOSelection/well)
    selections = select_electrodes_per_well(cfg)

    # fit the input normaliser on the assembled per-well inputs
    all_inputs = np.vstack([np.array(d.assemble_inputs(train[t]))
                            for t in range(len(train) - 1)])      # (n*g, in_dim)
    normalizer = InputNormalizer().fit(all_inputs)

    # 3. open the rig and configure every well straight from the selection
    session = hardware.MaxLabSession(cfg)        # needs MaxLab Live on the rig
    session.open()
    session.configure_from_selection(selections)

    # 4. build organoid reservoirs (drop-in for the ESNs in run_demo.py)
    reservoirs = reservoir.build_organoid_reservoirs(
        cfg, selections, session, normalizer)
    readouts = [RidgeReadout(cfg.readout.ridge, cfg.readout.quadratic)
                for _ in range(cfg.decomp.g)]

    # 5. identical to the simulation from here on
    model = ParallelReservoirRC(cfg, d, reservoirs, readouts)
    model.train(train, washout=50)

    test = (make_dataset(cfg, n_pred_steps=260, seed=7) - m) / s
    warm, horizon = test[:60], 200
    pred = model.forecast(warm, horizon)
    true_future = test[59:59 + horizon + 1]
    vt, _ = valid_prediction_time(true_future, pred, cfg.l96.dt_pred, 1.41)
    print(f"valid prediction time: {vt:.2f} Lyapunov times")

    session.close()


if __name__ == "__main__":
    main()
