# Parallel organoid reservoir computing — demo scaffold

Forecast a **Lorenz-96** "toy weather" system with **6 organoids** on a
**MaxWell MaxTwo** 6-well HD-MEA, using the parallel reservoir scheme of
Pathak, Hunt, Girvan, Lu & Ott (*PRL* **120**, 024102, 2018): the ring is split
into patches, each well predicts its own patch, and neighbours exchange only
their boundary ("halo") values each step.

## The idea in one line
Each organoid is a **fixed nonlinear reservoir**; only a small linear **readout**
is trained per well. Input = local state encoded as stimulation; output = decoded
spikes → predicted next state; feed predictions back → autoregressive forecast.

## Files
| file | role |
|------|------|
| `config.py` | all parameters (system, decomposition, hardware, timing) |
| `lorenz96.py` | synthetic target: simulate + Lyapunov-time estimate |
| `decomposition.py` | patch/core/buffer indexing + **halo exchange** |
| `encoding.py` | input vector → charge-balanced stimulation (MaxLab) |
| `decoding.py` | evoked spikes → binned firing-rate state vector |
| `readout.py` | per-well ridge readout (linear + quadratic) |
| `reservoir.py` | `ESNReservoir` (surrogate) and `OrganoidReservoir` (hardware) |
| `hardware.py` | MaxLab Live session wrapper (TODOs to fill on the rig) |
| `closed_loop.py` | train (teacher-forced) + forecast (autoregressive) + metrics |
| `baselines.py` | persistence + monolithic-ESN reference + controls |
| `run_demo.py` | **end-to-end in-silico dry run, no hardware needed** |

## Run the dry run (no rig)
```bash
pip install numpy
python run_demo.py
```
Expected: the 6-well parallel system reaches ~1–2 Lyapunov times of valid
forecast, clearly beating persistence (~0.14). That confirms the decomposition,
halo exchange, per-well training, and autoregressive loop are correct **before**
you touch wetware.

## Going to hardware
1. Fill the TODOs in `hardware.py` against your MaxLab Live install
   (api-docs.mxwbio.com): well power-up, electrode routing, stim delivery,
   spike streaming.
2. In `run_demo.py`'s `build_reservoirs`, swap `ESNReservoir` for
   `OrganoidReservoir(well_id, cfg, encoder, decoder, session)`.
3. Nothing else changes — the closed loop is backend-agnostic.

See `Organoid_RC_Experiment_Protocol.docx` (one folder up) for the full
experimental design: phased plan, electrode budget, timing, safety, controls,
and metrics.

## Default mapping (config.py)
- Lorenz-96: K=12 sites, F=8 (chaotic), prediction step dt=0.05 MTU
- 6 wells → g=6 reservoirs, q=2 core sites each, l=2 buffer each side
- input dim per well = q+2l = 6 (→ 6 stim clusters); output dim = 2
