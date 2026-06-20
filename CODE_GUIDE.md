# Code Guide — Parallel Organoid Reservoir Computing

A plain-language walkthrough of every file in `organoid_rc_demo/`: what each does,
what its parameters mean, and how the files connect. Pair this with
`Organoid_RC_Experiment_Protocol.docx` (the science/bench protocol) one folder up.

---

## 1. The big picture

Goal: forecast a chaotic "toy weather" system (Lorenz-96) using **6 reservoirs in
parallel**, one per well of a MaxWell MaxTwo plate. Each reservoir is a fixed
nonlinear system (an ESN in simulation, an organoid on hardware); the only thing
*trained* is a small linear readout per well.

The whole thing runs **end to end in simulation today** (no hardware). When the
rig is ready, you swap the simulated reservoir for the organoid backend and
nothing else changes.

### Data flow (how one forecast step works)

```
   lorenz96.py            decomposition.py        reservoir.py        readout.py
 ┌────────────┐  state   ┌───────────────┐ input ┌───────────┐ state ┌──────────┐
 │ true /     │────────► │ split into 6   │──────►│ reservoir │──────►│ ridge    │─┐
 │ predicted  │          │ wells + halo   │  (per │ (ESN or   │       │ readout  │ │
 │ state u(t) │ ◄──────┐ │ exchange       │ well) │ organoid) │       │ per well │ │
 └────────────┘        │ └───────────────┘       └───────────┘       └──────────┘ │
        ▲              │        ▲  reassemble 6 predicted cores ────────────────────┘
        │              └────────┘
        └──────────── feed prediction back in (autoregressive loop) ───────────────
```

`closed_loop.py` runs this loop (train, then forecast). `baselines.py` gives you
something to compare against. The `run_*.py` / `make_*.py` scripts are entry points.

---

## 2. File-by-file

### `config.py` — all settings in one place
Defines dataclasses holding every tunable parameter, bundled into `ExperimentConfig`
and exposed as the singleton **`CFG`**. Every other module imports `CFG`.
`CFG.validate()` runs on import and enforces the consistency rules (see §3). **This
is the file you edit to change an experiment.** Full parameter reference in §3.

### `lorenz96.py` — the synthetic target ("ground truth")
The system the reservoirs learn to predict.
- `l96_rhs(x, F)` — the Lorenz-96 right-hand side (the differential equations).
- `rk4_step(x, F, dt)` — one 4th-order Runge-Kutta integration step.
- `simulate(K, F, dt_int, n_steps, ...)` — produce a trajectory.
- `make_dataset(cfg, n_pred_steps, seed)` — trajectory sampled at the **prediction
  step** `dt_pred`, with spin-up discarded. This is what training/forecasting use.
- `largest_lyapunov(cfg)` — estimates the largest Lyapunov exponent (Benettin
  method). Its inverse is the **Lyapunov time**, the unit all forecast horizons
  are reported in. Run `python lorenz96.py` to print these numbers.

### `decomposition.py` — splitting the system across wells + the halo exchange
`RingDecomposition(K, g, q, l)` does all the spatial bookkeeping:
- `core(i)` / `split_cores(state)` — which sites well *i* owns (and predicts).
- `assemble_inputs(state)` — **the halo exchange**: builds each well's input from
  its own sites plus `l` boundary sites borrowed from each neighbour (with
  ring wrap-around). This is the only coupling between wells.
- `gather_cores(core_preds)` — stitch the 6 wells' predicted cores back into one
  full state vector.
- `in_dim = q + 2l` (reservoir input size), `out_dim = q` (what it predicts).
Run `python decomposition.py` to print the well → site map.

### `reservoir.py` — the fixed nonlinear reservoir (two backends)
Common interface: `reset()` and `step(input) -> state`.
- `ESNReservoir` — an echo-state network, the **in-silico stand-in for one
  organoid**. Used in every simulation run.
- `OrganoidReservoir` — the **hardware backend**: its `step()` encodes the input as
  stimulation, stimulates a well, records spikes, and decodes them to a state.
  Needs MaxLab Live + a connected MaxTwo.
Because both expose the same interface, the rest of the code never knows which is
in use — that is what makes the swap to wetware trivial.

### `readout.py` — the only trained component
`RidgeReadout(ridge, quadratic)` is a per-well linear regression mapping a reservoir
state to the predicted next-step values.
- `fit(states, targets)` — closed-form ridge regression (no gradient training).
- `predict(states)` — apply it.
- Uses features `[1, r, r²]` when `quadratic=True` (the `r²` term mirrors the
  Pathak paper and suits non-negative firing-rate features).
One readout is trained **independently per well** (organoids differ).

### `connectivity.py` — choosing which electrodes are input vs readout
From a MaxWell **Network-assay** recording it infers directed functional
connectivity (delayed transfer entropy vs jittered-spike surrogates, peak-delay
per pair) and picks, per well: **input electrodes** = network *sources* (high
directedness, stimulable, spatially spread) and **readout electrodes** = *sinks*
chosen for decorrelated coverage (high state dimensionality). Output is an
`IOSelection` (`.input_electrodes`, `.output_channels`, `.output_electrodes`, plus
scores/`W`/`delay_ms`). `select_for_well(data, CFG)` sizes the picks to the config
(`stim_clusters_per_well` inputs = q+2l, `readout_channels_per_well` outputs).
This is the upstream of the hardware path — its output feeds straight into
`MaxLabSession.configure_from_selection(...)`.

### `encoding.py` — data → stimulation (hardware side)
- `InputNormalizer` — scales each input value to `[0, 1]` using training-set
  statistics (keeps stimulation in a safe, non-saturating range).
- `StimEncoder` — turns a well's input vector into a MaxLab stimulation sequence:
  one stimulation cluster per input value, charge-balanced biphasic pulses, either
  **rate-coded** or **amplitude-coded**. Falls back to a readable dict if MaxLab
  isn't installed (so simulation still imports).

### `decoding.py` — spikes → reservoir state (hardware side)
- `SpikeDecoder` — bins post-stimulus spikes per readout channel into time bins
  and converts to firing rates, producing the state vector
  (`readout_channels_per_well × n_state_bins`).
- `state_from_esn(r)` — identity pass-through for the ESN path.

### `hardware.py` — the MaxLab Live session wrapper
`MaxLabSession` opens the device, configures each well's electrodes, delivers a
stimulation sequence and returns spikes, and does washout. It has clearly marked
`TODO`s — **this is the only file you complete to go live.** Everything else is
hardware-agnostic.

### `closed_loop.py` — the orchestrator + metrics
`ParallelReservoirRC(cfg, decomp, reservoirs, readouts)`:
- `train(train_traj)` — teacher-forced: drive each well with the *true* local
  input, record states, fit each readout.
- `forecast(warmup_traj, horizon)` — autonomous loop: assemble inputs (halo
  exchange), step every reservoir, predict each core, reassemble, feed back.
- `_drive(...)` — internal helper shared by both.
Metrics:
- `normalized_rmse(true, pred)` — RMSE divided by the system's spread.
- `valid_prediction_time(...)` — lead time (in Lyapunov times) until normalized
  RMSE first crosses the threshold (default 0.3). The headline number.

### `baselines.py` — what to compare against
- `persistence_forecast` — "the future equals the present" (the must-beat floor).
- `MonolithicESNForecaster` — one big ESN over the whole ring (no decomposition);
  an idealised silicon reference. Build/train once, forecast many windows.

### Entry-point scripts
- `run_demo.py` — end-to-end simulation; prints valid prediction time vs baselines.
- `make_figures.py` — Figs 1–3 (target space-time; forecast vs truth vs error;
  error growth vs lead time).
- `make_scaling_figure.py` — Fig 4: fix the system, vary the number of wells.
- `make_ql_grid.py` — Figs 5–6: sweep `q` (sites/well) and `l` (overlap) on the
  6-well plate; grid of curves + a valid-time heatmap.

### Support files
- `README.md` — quick orientation and run commands.
- `requirements.txt` — `numpy` (core), `matplotlib` (figures); `maxlab` only on
  the rig.

---

## 3. Parameter reference (everything in `config.py`)

### `Lorenz96Config` — the target system
| Param | Meaning | Effect / typical |
|---|---|---|
| `K` | number of sites on the ring (system dimension) | must equal `g × q`. Bigger = harder. Default 12 |
| `F` | forcing strength | F=8 strongly chaotic; ~5 mild; higher = harder, changes Lyapunov time |
| `dt_int` | RK4 integration step (model time units) | numerical accuracy; default 0.01 |
| `dt_pred` | one reservoir prediction step | smaller = easier per step (more "persistence-like"); larger = harder map. Default 0.05 |
| `spinup` | model time discarded so trajectories sit on the attractor | default 20 |

### `DecompositionConfig` — splitting across wells
| Param | Meaning | Effect / typical |
|---|---|---|
| `g` | number of reservoirs = number of wells | must divide `K`; ≤ 6 on MaxTwo. Default 6 |
| `q` | core sites each well owns/predicts (= K/g) | larger q = each well handles more, needs a richer reservoir |
| `l` | halo/overlap width on **each** side | 0 = isolated wells (worse); raise until it covers the coupling range (~2), then diminishing returns |
| `in_dim` (derived) | reservoir input size = `q + 2l` | maps to stimulation clusters per well |

### `HardwareConfig` — MaxTwo / MaxLab (hardware path only*)
| Param | Meaning | Notes |
|---|---|---|
| `n_wells` | wells in use | must equal `g`. Default 6 |
| `fs_hz` | acquisition sampling rate | 20 kHz |
| `max_record_channels` | simultaneous recording channels per well | ~1024 |
| `max_stim_units` | stimulation units per well | up to 32 |
| `stim_clusters_per_well` | electrode clusters used for input | must equal `q + 2l` (see validate) |
| `electrodes_per_stim_cluster` | electrodes per input value (redundancy) | hardware robustness only; no effect in simulation |
| `pulse_phase_us` | duration of each biphasic phase | 200 µs |
| `pulse_amp_mV_range` | amplitude range for amplitude coding | safety-bounded |
| `pulse_rate_hz_range` | pulse-rate range for rate coding | e.g. 5–100 Hz |
| `encoding_mode` | `"rate"` or `"amplitude"` | rate is the recommended default |
| `readout_channels_per_well` | electrodes feeding the state vector | 256; sets hardware state dim with `n_state_bins` |

\* These do **not** change the simulation results — they configure the wetware path
(and the `validate()` consistency check).

### `FrameTimingConfig` — mapping one step to biological time
| Param | Meaning |
|---|---|
| `stim_window_ms` | how long the encoded stimulus is delivered (50) |
| `response_window_ms` | window over which evoked spikes form the state (200) |
| `washout_ms` | quiet gap so the state reflects the current input (50) |
| `frame_ms` (derived) | total per-frame time = stim + response + washout |

### `DecodingConfig` — spikes → state (hardware path)
| Param | Meaning |
|---|---|
| `n_state_bins` | time bins across the response window (state dim = channels × bins) |
| `feature` | `"rate"` (spikes/s) or `"count"` |
| `smooth_ms` | optional spike-train smoothing |

### `ReadoutConfig` — the trained readout
| Param | Meaning | Effect |
|---|---|---|
| `ridge` | Tikhonov regularisation strength | higher = smoother, more noise-robust, less overfit |
| `quadratic` | include `r²` features | `False` to test the linear-only readout (usually worse) |

### `ESNSurrogateConfig` — the in-silico "organoid" (drives simulation results)
| Param | Meaning | Effect |
|---|---|---|
| `n_nodes` | reservoir size per well | bigger = more capacity (analog of a richer organoid / more readout electrodes) |
| `spectral_radius` | scaling of the recurrent weights | memory depth/stability; keep < ~1 |
| `input_scaling` | how strongly input drives the reservoir | analog of stimulation amplitude; too high saturates `tanh` |
| `leak` | leaky-integration rate | 1.0 = react fully to current input; lower = slower / more memory |
| `seed` | random seed | a different seed = "a different organoid" |

### Consistency rules (`ExperimentConfig.validate()`)
- `g × q == K` (every site owned by exactly one well)
- `g == n_wells` (one reservoir per well)
- `q + 2l == stim_clusters_per_well` (input dim matches the stim clusters)

If you change `q` or `l`, update `stim_clusters_per_well` to match, or the import
will fail. (Ask if you'd like this auto-derived so it never blocks experiments.)

---

## 4. How the files link

```
config.py  (CFG)  ─────────────── imported by EVERYTHING
     │
lorenz96.py ──────────── makes data ───────────► run_demo / make_* scripts
     │                                                   │
decomposition.py ── slices data & halo exchange ─► closed_loop.py
reservoir.py ───── ESN or organoid backend ──────► closed_loop.py
readout.py ─────── trained per-well mapping ─────► closed_loop.py
     │                                                   │
encoding.py + decoding.py + hardware.py ── used ONLY by OrganoidReservoir
     │                                                   │
closed_loop.py ── train()/forecast()/metrics ───► run_demo / make_* scripts
baselines.py ──── persistence + monolithic ESN ─► run_demo / make_* scripts
```

In words: **`config` feeds everything**. The scripts ask `lorenz96` for data, build a
`RingDecomposition` plus six reservoirs (`reservoir`) and six readouts (`readout`),
hand them to `ParallelReservoirRC` (`closed_loop`) to train and forecast, and score
the result against `baselines`. On hardware, only the reservoir backend changes —
`encoding`, `decoding`, and `hardware` come into play behind `OrganoidReservoir`,
and the loop above is untouched.

---

## 5. Simulation vs hardware — what runs where

| Runs anywhere (laptop/cloud) | Must run on the rig PC (connected MaxTwo) |
|---|---|
| `config`, `lorenz96`, `decomposition`, `reservoir.ESNReservoir`, `readout`, `closed_loop`, `baselines`, all `make_*`/`run_demo` | `reservoir.OrganoidReservoir`, `encoding`, `decoding`, `hardware`, and the live closed loop |

Readout training and all plotting are just math on recorded data, so they can run
off the rig (record on the rig → train/analyse elsewhere → push weights back).

---

## 6. Glossary
- **Reservoir** — a fixed, high-dimensional nonlinear system that expands the input;
  here an ESN or an organoid. Not trained.
- **Readout** — the trained linear map from reservoir state to output.
- **Halo / overlap (`l`)** — boundary sites a well borrows from its neighbours so it
  can see the local coupling.
- **Teacher forcing** — driving the reservoir with true data during training.
- **Autoregressive forecast** — feeding predictions back in as the next input.
- **Washout** — discarding/settling early steps so the state reflects current input.
- **Echo-state property** — the reservoir forgets its initial condition; its state
  depends only on recent input history.
- **Lyapunov time** — the e-folding time of errors in the chaotic system; the unit
  forecast horizons are reported in.
- **Valid prediction time (vts)** — how long (in Lyapunov times) the forecast stays
  under the error threshold; the main success metric.
- **Persistence** — the naive "future = present" baseline.
