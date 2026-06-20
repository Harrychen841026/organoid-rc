"""
CONNECTIVITY:  MaxWell Network-assay spikes  ->  input / output electrode choice.

Why this file exists
--------------------
The protocol (Section 5) currently just says inputs go on "spatially separated
clusters" with no rule for *which* electrodes. This module picks them from data,
following the design we agreed on:

  * INPUT  (stimulation) electrodes  = network SOURCES / drivers.
        Stimulating a driver injects the encoded signal and lets it propagate
        widely -- the physical analogue of a strong input projection Win.
  * OUTPUT (readout) electrodes      = SINKS that also span the network.
        We want the readout to integrate the response, but for reservoir
        computing the bigger win is a HIGH-DIMENSIONAL, DECORRELATED state, so
        outputs are chosen for coverage/decorrelation (in-strength only breaks
        ties). This fights the global synchronised bursting that otherwise
        collapses an organoid's effective dimensionality.

Three things this implements that the naive "sum |FC|" idea misses
------------------------------------------------------------------
1. Connectivity is computed from a SIMULTANEOUS recording (the Network assay),
   never the Activity Scan -- scan blocks are recorded at different times, so
   cross-electrode timing is meaningless there.
2. Direction needs a DIRECTED measure. Symmetric measures (cross-correlation,
   coherence, STTC) cannot define "outward vs inward". We use delayed
   transfer entropy (TE) on binned spike trains, thresholded against
   jittered-spike surrogates.
3. A hub scores high on BOTH in- and out-strength, so we rank inputs on a
   directedness index  (out - in) / (out + in)  to isolate true sources, not
   plain out-strength.

IMPORTANT: passive TE is only a *proxy* for how a site responds to
stimulation. Use the returned input list as a SHORTLIST, then validate it with
an actual stim scan on the rig (reliable, wide, low-threshold, stim-routable,
spatially separated) before committing -- see `shortlist_only=True`.

Dependencies: numpy only. h5py is imported lazily and only needed to read a
real MaxWell .h5 file; everything else (and the __main__ demo) runs without it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
    _HAVE_H5PY = True
except Exception:  # noqa
    _HAVE_H5PY = False


# ======================================================================
# 1. Container for a loaded Network-assay recording
# ======================================================================
@dataclass
class NetworkAssayData:
    """Spikes + geometry from one well's Network-assay recording.

    spikes    : {channel_id: np.ndarray of spike times in SECONDS}
    electrode : {channel_id: electrode_id}            (routing, optional)
    positions : {channel_id: (x_um, y_um)}            (optional, for spacing)
    duration  : recording length in seconds
    fs        : sampling rate (Hz)
    """
    spikes: Dict[int, np.ndarray]
    duration: float
    fs: float = 20_000.0
    electrode: Dict[int, int] = field(default_factory=dict)
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    @property
    def channels(self) -> List[int]:
        return sorted(self.spikes.keys())

    def firing_rates(self) -> Dict[int, float]:
        return {c: len(t) / max(self.duration, 1e-9) for c, t in self.spikes.items()}


# ----------------------------------------------------------------------
# 1a. Loaders
# ----------------------------------------------------------------------
def from_spike_lists(spikes: Dict[int, Sequence[float]],
                     duration: float,
                     fs: float = 20_000.0,
                     positions: Optional[Dict[int, Tuple[float, float]]] = None,
                     electrode: Optional[Dict[int, int]] = None
                     ) -> NetworkAssayData:
    """Build a NetworkAssayData from already-extracted spike trains.

    Use this if you run your own spike detector or have a custom export. Each
    value is spike times in seconds for that channel.
    """
    sp = {int(c): np.asarray(t, float) for c, t in spikes.items()}
    return NetworkAssayData(spikes=sp, duration=float(duration), fs=float(fs),
                            positions=positions or {}, electrode=electrode or {})


def load_network_assay(path: str, well: Optional[str] = None,
                       rec: Optional[str] = None,
                       fs: Optional[float] = None) -> NetworkAssayData:
    """Read MaxWell MaxLab Live online-detected spikes from a .raw.h5 file.

    Matches the documented MaxWell HDF5 schema (see neo MaxwellRawIO). A
    `version` dataset selects the layout:

      OLD format (version == 20160704, MaxOne):
        spikes  : /proc0/spikeTimes  -- compound (frameno, channel, amplitude)
        mapping : /mapping           -- compound (channel, electrode, x, y)
        fs      : 20000 (fixed)
      NEW format (version > 20160704, MaxOne new / MaxTwo multi-well):
        well    : /wells/well000, /wells/well001, ...  (ONE WELL PER STREAM --
                  for a 6-well MaxTwo plate, pick the well for this organoid)
        rec     : /wells/<well>/rec0000, ...
        spikes  : /wells/<well>/<rec>/spikes  -- compound (frameno, channel, ...)
        mapping : /wells/<well>/<rec>/settings/mapping
        fs      : /wells/<well>/<rec>/settings/sampling

    Notes:
      * Uses the ONLINE-detected spikes the assay already stored, so it does NOT
        need MaxWell's HDF5 compression plugin (that is only required to read the
        raw `sig`/`raw` traces, which we don't touch). mapping x/y are in um.
      * frameno is absolute; times are returned relative to the first spike,
        which preserves all cross-channel timing (all that connectivity needs).
      * For MaxTwo, call once per well (well='well000', 'well001', ...).

    If your file deviates, extract spikes yourself and use from_spike_lists().
    """
    if not _HAVE_H5PY:
        raise RuntimeError(
            "h5py not installed. `pip install h5py`, or extract spikes "
            "yourself and use connectivity.from_spike_lists(...).")

    def _channel_field(names):
        return "channel" if "channel" in names else names[1]

    def _time_seconds(ds, names, sampling):
        if "frameno" in names:
            return np.asarray(ds["frameno"], np.float64) / sampling
        if "time" in names:
            return np.asarray(ds["time"], np.float64)
        return np.asarray(ds[names[0]], np.float64) / sampling

    def _parse_mapping(m):
        mn = m.dtype.names or ()
        elec, pos = {}, {}
        for row in m:
            c = int(row["channel"])
            if c < 0:
                continue                       # unrouted slots are flagged -1
            if "electrode" in mn:
                elec[c] = int(row["electrode"])
            if "x" in mn and "y" in mn:
                pos[c] = (float(row["x"]), float(row["y"]))
        return elec, pos

    with h5py.File(path, "r") as f:
        version = None
        if "version" in f:
            try:                               # MaxWell stores it as a byte string
                v = np.asarray(f["version"]).ravel()[0]
                version = int(v.decode() if isinstance(v, bytes) else v)
            except Exception:                  # noqa
                version = None

        old_format = (version == 20160704) or ("wells" not in f and "sig" in f) \
            or ("proc0" in f)

        if old_format:
            sampling = fs if fs is not None else 20_000.0
            if "proc0/spikeTimes" not in f:
                raise ValueError("Old-format MaxWell file but no /proc0/spikeTimes "
                                 f"(keys: {list(f.keys())}). Use from_spike_lists().")
            spike_ds = f["proc0/spikeTimes"][()]
            mapping = f["mapping"][()] if "mapping" in f else None
        else:
            if "wells" not in f:
                raise ValueError("Unrecognised MaxWell file: no 'version', 'wells' "
                                 f"or 'sig' (keys: {list(f.keys())}). Use from_spike_lists().")
            wells = sorted(f["wells"].keys())
            well = well or wells[0]
            if well not in f["wells"]:
                raise ValueError(f"well '{well}' not found. Available: {wells}")
            recs = sorted(f["wells"][well].keys())
            rec = rec or recs[0]
            if rec not in f["wells"][well]:
                raise ValueError(f"rec '{rec}' not found in {well}. Available: {recs}")
            grp = f["wells"][well][rec]
            settings = grp["settings"]
            sampling = float(fs if fs is not None
                             else np.asarray(settings["sampling"])[0])
            if "spikes" not in grp:
                raise ValueError(f"No 'spikes' dataset in /wells/{well}/{rec} "
                                 f"(keys: {list(grp.keys())}). Was spike detection on?")
            spike_ds = grp["spikes"][()]
            mapping = settings["mapping"][()] if "mapping" in settings else None

        names = spike_ds.dtype.names
        ch_field = _channel_field(names)
        t = _time_seconds(spike_ds, names, sampling)
        ch = np.asarray(spike_ds[ch_field], int)
        electrode, positions = _parse_mapping(mapping) if mapping is not None else ({}, {})

        # keep only spikes on mapped (routed) channels when a mapping exists
        keep = np.ones(ch.shape, bool)
        if electrode:
            routed = set(electrode.keys())
            keep = np.array([c in routed for c in ch])
        t, ch = t[keep], ch[keep]
        t0 = t.min() if t.size else 0.0

    spikes: Dict[int, List[float]] = {}
    for c, ts in zip(ch, t):
        spikes.setdefault(int(c), []).append(float(ts - t0))
    duration = max((max(v) for v in spikes.values() if v), default=0.0)
    spikes_np = {c: np.asarray(sorted(v), float) for c, v in spikes.items()}
    return NetworkAssayData(spikes=spikes_np, duration=duration, fs=sampling,
                            positions=positions, electrode=electrode)


def list_wells(path: str) -> List[str]:
    """Return the well names in a MaxWell file ([''] for old single-well format)."""
    if not _HAVE_H5PY:
        raise RuntimeError("h5py not installed (`pip install h5py`).")
    with h5py.File(path, "r") as f:
        if "wells" in f:
            return sorted(f["wells"].keys())
        return [""]                            # old format: one implicit well


def inspect_h5(path: str, max_items: int = 200) -> None:
    """Print a MaxWell .h5 tree + detected format, to confirm the layout.

    Use this on a real file before the first run:  load_network_assay expects
    the paths neo's MaxwellRawIO documents; if yours differ this shows where the
    spikes / mapping actually live so you can pass the right well/rec (or fall
    back to from_spike_lists()).
    """
    if not _HAVE_H5PY:
        raise RuntimeError("h5py not installed (`pip install h5py`).")
    with h5py.File(path, "r") as f:
        ver = None
        if "version" in f:
            v = np.asarray(f["version"]).ravel()[0]
            ver = v.decode() if isinstance(v, bytes) else v
        fmt = ("OLD/MaxOne (proc0/spikeTimes)" if ("proc0" in f or "sig" in f)
               else "NEW/MaxTwo multi-well (wells/...)" if "wells" in f
               else "UNKNOWN")
        print(f"version={ver}  ->  {fmt}")
        if "wells" in f:
            for w in sorted(f["wells"].keys()):
                recs = sorted(f["wells"][w].keys())
                print(f"  well {w}: recs {recs}")
                for rc in recs:
                    g = f["wells"][w][rc]
                    print(f"    {rc}: has spikes={'spikes' in g}, "
                          f"has settings/mapping={'mapping' in g.get('settings', {})}")
        n = [0]
        def _show(name, obj):
            if n[0] >= max_items:
                return
            if isinstance(obj, h5py.Dataset):
                fields = f" fields={obj.dtype.names}" if obj.dtype.names else ""
                print(f"  /{name}  shape={obj.shape} dtype={obj.dtype}{fields}")
                n[0] += 1
        f.visititems(_show)


# ======================================================================
# 2. Binning
# ======================================================================
def binned_matrix(data: NetworkAssayData, bin_ms: float = 5.0,
                  channels: Optional[Sequence[int]] = None
                  ) -> Tuple[np.ndarray, List[int]]:
    """Return (n_channels, n_bins) spike-count matrix and the channel order."""
    chans = list(channels) if channels is not None else data.channels
    n_bins = int(np.ceil(data.duration / (bin_ms * 1e-3)))
    bw = bin_ms * 1e-3
    M = np.zeros((len(chans), n_bins), dtype=np.int32)
    for i, c in enumerate(chans):
        t = data.spikes.get(c, ())
        if len(t):
            idx = np.clip((np.asarray(t) / bw).astype(int), 0, n_bins - 1)
            np.add.at(M[i], idx, 1)
    return M, chans


# ======================================================================
# 3. Directed connectivity: delayed transfer entropy
# ======================================================================
def _te_pairwise_binary(B: np.ndarray, delay: int = 1) -> np.ndarray:
    """Delayed transfer entropy between all pairs of BINARY spike trains.

    TE(X -> Y) with history length 1 and lag `delay` (in bins):

        TE = sum p(y1, y0, x0) * log2[ p(y1|y0,x0) / p(y1|y0) ]

    where y1 = Y at t+1, y0 = Y at t, x0 = X at t-delay+1. Returns W with
    W[i, j] = TE(i -> j)  (non-negative; W has zero diagonal).

    B: (n_channels, n_bins) array of 0/1. Returns W with W[i, j] = TE(i -> j)
    (non-negative; zero diagonal).

    Fully vectorised over ALL source/target pairs. The 8-cell joint histogram
    N(y1=c, y0=a, x0=b) over every (target, source) pair is obtained with four
    matrix products: for each (a, c) the per-target mask M_ac (n_target x L) is
    multiplied by the source matrix (L x n_source), giving the b=1 counts; the
    b=0 counts follow by subtraction. This is O(N^2 * L) in BLAS rather than in
    Python, ~100x faster than the per-pair loop -- what makes the bin/delay
    sweep tractable. Still O(N^2): prefilter channels for large arrays.
    """
    n, T = B.shape
    if T <= delay + 1:
        return np.zeros((n, n))
    # aligned slices for TE_{X->Y}(d) = I(Y_{t+1}; X_{t-d+1} | Y_t):
    #   y1 = Y_{t+1}, y0 = Y_t, x0 = X_{t-d+1}  -- x0 and y0 share time index t
    Y1 = B[:, delay + 1:].astype(np.float64)     # (n, L) target future
    Y0 = B[:, delay:-1].astype(np.float64)       # (n, L) target present
    X0 = B[:, 1:-delay].astype(np.float64)       # (n, L) source, lag d
    L = Y1.shape[1]
    eps = 1e-12
    Y0_is = {0: (Y0 < 0.5).astype(np.float64), 1: Y0}        # 1{y0=a}
    Y1_is = {0: (Y1 < 0.5).astype(np.float64), 1: Y1}        # 1{y1=c}
    Xb = {1: X0, 0: (1.0 - X0)}                              # 1{x0=b}

    # triple counts N[a,c,b] as (n_target, n_source), and N_ac as (n_target,)
    N_acb, N_ac = {}, {}
    for a in (0, 1):
        for c in (0, 1):
            Mac = Y0_is[a] * Y1_is[c]            # (n_target, L)
            N_ac[(a, c)] = Mac.sum(axis=1)       # (n_target,)
            N_acb[(a, c, 1)] = Mac @ Xb[1].T     # (n_target, n_source)
            N_acb[(a, c, 0)] = N_ac[(a, c)][:, None] - N_acb[(a, c, 1)]

    TE = np.zeros((n, n))                        # TE[target, source]
    for a in (0, 1):
        Na = N_ac[(a, 0)] + N_ac[(a, 1)]         # (n_target,)
        for b in (0, 1):
            Nab = N_acb[(a, 0, b)] + N_acb[(a, 1, b)]    # (n_target, n_source)
            for c in (0, 1):
                Nabc = N_acb[(a, c, b)]                  # (n_target, n_source)
                p_joint = Nabc / L
                cond_full = Nabc / (Nab + eps)           # p(y1=c|y0=a,x0=b)
                cond_marg = (N_ac[(a, c)] / (Na + eps))[:, None]  # p(y1=c|y0=a)
                term = p_joint * np.log2((cond_full + eps) / (cond_marg + eps))
                TE += np.where(Nabc > 0, term, 0.0)
    W = TE.T                                     # W[i, j] = TE(source i -> target j)
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, 0.0)


@dataclass
class TEResult:
    """Raw TE plus the surrogate null, so callers can threshold or score."""
    W_raw: np.ndarray            # unthresholded TE, W[i, j] = TE(i -> j)
    null_mean: float
    null_std: float
    occupancy: float             # mean fraction of populated (binarised) bins
    chans: List[int]

    def threshold(self, sig_z: float = 3.0) -> np.ndarray:
        thr = self.null_mean + sig_z * self.null_std
        W = np.where(self.W_raw > thr, self.W_raw, 0.0)
        np.fill_diagonal(W, 0.0)
        return W

    def effect_size(self, sig_z: float = 3.0) -> float:
        """Mean (TE - null_mean)/null_std over surviving edges (signal SNR)."""
        W = self.threshold(sig_z)
        e = W[W > 0]
        if e.size == 0:
            return 0.0
        return float(((e - self.null_mean) / (self.null_std + 1e-12)).mean())


def te_and_null(data: NetworkAssayData,
                channels: Optional[Sequence[int]] = None,
                bin_ms: float = 5.0,
                delay_bins: int = 1,
                n_surrogates: int = 20,
                surrogate_jitter_ms: float = 20.0,
                rng_seed: int = 0) -> TEResult:
    """Raw TE matrix + jittered-spike surrogate null (no thresholding yet).

    Surrogates jitter every spike by up to +/- surrogate_jitter_ms, destroying
    fine cross-channel timing while preserving rate -- the null for "no real
    directed coupling at this rate". Real data and surrogates use the SAME bin
    and delay, so the threshold is on the same footing as the signal.
    """
    B0, chans = binned_matrix(data, bin_ms, channels)
    B0 = (B0 > 0).astype(np.int8)            # binarise
    occ = float(B0.mean())
    W = _te_pairwise_binary(B0, delay_bins)

    rng = np.random.default_rng(rng_seed)
    null_vals = []
    bw = bin_ms * 1e-3
    n_bins = B0.shape[1]
    iu = ~np.eye(len(chans), dtype=bool)
    for _ in range(max(n_surrogates, 1)):
        Bs = np.zeros_like(B0)
        for i, c in enumerate(chans):
            t = data.spikes.get(c, ())
            if len(t) == 0:
                continue
            jit = (np.asarray(t) +
                   rng.uniform(-surrogate_jitter_ms, surrogate_jitter_ms,
                               size=len(t)) * 1e-3)
            idx = np.clip((jit / bw).astype(int), 0, n_bins - 1)
            Bs[i, idx] = 1
        Ws = _te_pairwise_binary(Bs, delay_bins)
        null_vals.append(Ws[iu])
    null = np.concatenate(null_vals) if null_vals else np.array([0.0])
    return TEResult(W_raw=W, null_mean=float(null.mean()),
                    null_std=float(null.std()), occupancy=occ, chans=chans)


def directed_connectivity(data: NetworkAssayData,
                          channels: Optional[Sequence[int]] = None,
                          bin_ms: float = 5.0,
                          delay_bins: int = 1,
                          n_surrogates: int = 20,
                          surrogate_jitter_ms: float = 20.0,
                          sig_z: float = 3.0,
                          rng_seed: int = 0
                          ) -> Tuple[np.ndarray, List[int]]:
    """Directed TE matrix, thresholded against jittered-spike surrogates.

    Returns (W, channels) with W[i, j] = significant TE(i -> j), else 0.
    Thin wrapper over te_and_null(...).threshold(sig_z).
    """
    res = te_and_null(data, channels, bin_ms, delay_bins, n_surrogates,
                      surrogate_jitter_ms, rng_seed)
    return res.threshold(sig_z), res.chans


# ----------------------------------------------------------------------
# 3a. Parameter sweep & delay-agnostic connectivity
# ----------------------------------------------------------------------
def significant_edges(W: np.ndarray, chans: Sequence[int]) -> Dict[Tuple[int, int], float]:
    """{(src_channel, dst_channel): weight} for every non-zero edge in W."""
    out = {}
    for i, j in np.argwhere(W > 0):
        out[(chans[int(i)], chans[int(j)])] = float(W[int(i), int(j)])
    return out


def sweep_bin_delay(data: NetworkAssayData,
                    bins_ms: Sequence[float],
                    delays_ms: Sequence[float],
                    channels: Optional[Sequence[int]] = None,
                    min_rate_hz: float = 0.1,
                    max_channels: int = 128,
                    n_surrogates: int = 8,
                    sig_z: float = 3.0,
                    rng_seed: int = 0) -> Dict:
    """Sweep TE over a grid of bin sizes x delays.

    delay is given in MILLISECONDS and converted per-bin to
    delay_bins = max(1, round(delay_ms / bin_ms)); a delay shorter than one
    bin cannot be resolved, so it is clamped to 1 and flagged in `resolvable`.

    Returns a dict of (len(bins) x len(delays)) metric grids:
      n_edges      : number of significant directed edges
      mean_te      : mean weight of significant edges
      occupancy    : mean binarised-bin occupancy (want ~0.01-0.3)
      effect_size  : mean SNR of significant edges above the null
      resolvable   : bool, whether delay_ms >= bin_ms
      W            : the thresholded matrix (for later plotting), per cell
      chans        : channel order
    Plus 'edges': per-cell {(src,dst):w} for evaluation against ground truth.
    """
    rates = data.firing_rates()
    active = sorted((c for c in data.channels if rates.get(c, 0) >= min_rate_hz),
                    key=lambda c: rates[c], reverse=True)[:max_channels]
    if channels is not None:
        active = [c for c in channels if c in set(active)] or list(channels)

    nb, nd = len(bins_ms), len(delays_ms)
    grid = lambda: np.zeros((nb, nd))
    out = {"bins_ms": list(bins_ms), "delays_ms": list(delays_ms),
           "n_edges": grid(), "mean_te": grid(), "occupancy": grid(),
           "effect_size": grid(), "resolvable": np.zeros((nb, nd), bool),
           "W": [[None] * nd for _ in range(nb)],
           "edges": [[None] * nd for _ in range(nb)], "chans": active}
    for bi, bm in enumerate(bins_ms):
        for di, dm in enumerate(delays_ms):
            db = max(1, int(round(dm / bm)))
            res = te_and_null(data, channels=active, bin_ms=bm, delay_bins=db,
                              n_surrogates=n_surrogates, rng_seed=rng_seed)
            W = res.threshold(sig_z)
            edges = significant_edges(W, res.chans)
            out["W"][bi][di] = W
            out["edges"][bi][di] = edges
            out["n_edges"][bi, di] = len(edges)
            out["mean_te"][bi, di] = np.mean(list(edges.values())) if edges else 0.0
            out["occupancy"][bi, di] = res.occupancy
            out["effect_size"][bi, di] = res.effect_size(sig_z)
            out["resolvable"][bi, di] = dm >= bm
    return out


def connectivity_max_over_delays(data: NetworkAssayData,
                                 bin_ms: float,
                                 delays_ms: Sequence[float],
                                 channels: Optional[Sequence[int]] = None,
                                 n_surrogates: int = 10,
                                 sig_z: float = 3.0,
                                 rng_seed: int = 0
                                 ) -> Tuple[np.ndarray, List[int]]:
    """Delay-agnostic connectivity: element-wise max of TE over several delays.

    A SINGLE delay is a band-pass on connection latency (and hence, since delay
    grows with path length, on connection distance). Taking the per-edge max
    over a range of delays recovers short- AND long-latency edges together --
    the recommended way to build the connectivity matrix for selection.
    """
    Wmax, chans = None, None
    for dm in delays_ms:
        db = max(1, int(round(dm / bin_ms)))
        W, chans = directed_connectivity(data, channels=channels, bin_ms=bin_ms,
                                         delay_bins=db, n_surrogates=n_surrogates,
                                         sig_z=sig_z, rng_seed=rng_seed)
        Wmax = W if Wmax is None else np.maximum(Wmax, W)
    return Wmax, chans


# ----------------------------------------------------------------------
# 3b. Peak-delay connectivity (per-pair best delay, matched null)
# ----------------------------------------------------------------------
def _delay_bins_list(bin_ms: float, delays_ms: Sequence[float]
                     ) -> Tuple[List[int], List[float]]:
    """Unique delay_bins (>=1) and their resolved latency (delay_bins*bin_ms)."""
    seen: Dict[int, float] = {}
    for dm in delays_ms:
        db = max(1, int(round(dm / bin_ms)))
        seen.setdefault(db, db * bin_ms)
    dbs = sorted(seen)
    return dbs, [seen[db] for db in dbs]


def _te_max_over_delays(B0: np.ndarray, delay_bins: Sequence[int]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pair max TE over delays, and the argmax delay index per pair."""
    Wmax = None
    Didx = None
    for k, db in enumerate(delay_bins):
        W = _te_pairwise_binary(B0, db)
        if Wmax is None:
            Wmax, Didx = W, np.zeros_like(W, dtype=int)
        else:
            upd = W > Wmax
            Wmax = np.where(upd, W, Wmax)
            Didx = np.where(upd, k, Didx)
    return Wmax, Didx


@dataclass
class PeakDelayResult:
    W: np.ndarray            # thresholded peak TE, W[i,j] = max_d TE_d(i->j)
    delay_ms: np.ndarray     # per-edge peak latency (NaN where not significant)
    chans: List[int]
    W_raw: np.ndarray        # unthresholded peak TE


def directed_connectivity_peakdelay(
        data: NetworkAssayData,
        channels: Optional[Sequence[int]] = None,
        bin_ms: float = 5.0,
        delays_ms: Sequence[float] = (2, 4, 6, 8, 10, 14, 18),
        n_surrogates: int = 20,
        surrogate_jitter_ms: float = 20.0,
        sig_z: float = 3.0,
        rng_seed: int = 0) -> PeakDelayResult:
    """Connectivity where each pair keeps its BEST delay -- distance-fair.

    For every (i, j) we take W*[i,j] = max over delays of TE_d(i -> j), and
    record the delay that achieved it (an estimate of that edge's conduction
    latency). This stops a single fixed delay from band-passing connectivity to
    one distance band, so long-range drivers are not undercounted -- important
    for ranking sources/sinks during electrode selection.

    CRITICAL -- the significance test must match the statistic. Taking the max
    over K delays is biased upward (max of K noisy, positively-biased
    estimates), so we threshold against a null built the SAME way: each
    surrogate also takes its max over the same K delays. Thresholding a max
    statistic against a single-delay null would pass a flood of false
    positives. Keep `delays_ms` within physiological latencies (~1-20 ms); a
    wider range inflates the max and captures polysynaptic / common-drive paths.
    """
    B0, chans = binned_matrix(data, bin_ms, channels)
    B0 = (B0 > 0).astype(np.int8)
    dbs, resolved_ms = _delay_bins_list(bin_ms, delays_ms)
    Wmax, Didx = _te_max_over_delays(B0, dbs)

    rng = np.random.default_rng(rng_seed)
    bw = bin_ms * 1e-3
    n_bins = B0.shape[1]
    iu = ~np.eye(len(chans), dtype=bool)
    null_vals = []
    for _ in range(max(n_surrogates, 1)):
        Bs = np.zeros_like(B0)
        for i, c in enumerate(chans):
            t = data.spikes.get(c, ())
            if len(t) == 0:
                continue
            jit = (np.asarray(t) +
                   rng.uniform(-surrogate_jitter_ms, surrogate_jitter_ms,
                               size=len(t)) * 1e-3)
            idx = np.clip((jit / bw).astype(int), 0, n_bins - 1)
            Bs[i, idx] = 1
        Wsm, _ = _te_max_over_delays(Bs, dbs)        # matched max-over-delays null
        null_vals.append(Wsm[iu])
    null = np.concatenate(null_vals) if null_vals else np.array([0.0])
    thr = null.mean() + sig_z * null.std()

    W = np.where(Wmax > thr, Wmax, 0.0)
    np.fill_diagonal(W, 0.0)
    resolved = np.asarray(resolved_ms, float)
    D_ms = resolved[Didx]
    D_ms = np.where(W > 0, D_ms, np.nan)             # latency only for real edges
    return PeakDelayResult(W=W, delay_ms=D_ms, chans=chans, W_raw=Wmax)


# ======================================================================
# 4. Node scores
# ======================================================================
def node_scores(W: np.ndarray) -> Dict[str, np.ndarray]:
    """out/in strength and the directedness index used to rank sources/sinks."""
    out_s = W.sum(axis=1)                 # i -> all
    in_s = W.sum(axis=0)                  # all -> i
    denom = out_s + in_s + 1e-12
    directedness = (out_s - in_s) / denom   # +1 pure source, -1 pure sink
    return {"out_strength": out_s, "in_strength": in_s,
            "directedness": directedness}


# ======================================================================
# 5. Selection
# ======================================================================
def _spatially_filtered(ranked: List[int], positions: Dict[int, Tuple[float, float]],
                        chans: List[int], min_dist_um: float, n: int) -> List[int]:
    """Greedily take top-ranked channels while enforcing a min spacing."""
    if not positions or min_dist_um <= 0:
        return ranked[:n]
    picked: List[int] = []
    for ci in ranked:
        c = chans[ci]
        if c not in positions:
            picked.append(ci)
        else:
            x, y = positions[c]
            ok = True
            for pj in picked:
                cp = chans[pj]
                if cp in positions:
                    xp, yp = positions[cp]
                    if (x - xp) ** 2 + (y - yp) ** 2 < min_dist_um ** 2:
                        ok = False
                        break
            if ok:
                picked.append(ci)
        if len(picked) >= n:
            break
    return picked


def _greedy_decorrelated(B: np.ndarray, candidates: List[int],
                         seed_order: List[int], n: int) -> List[int]:
    """Pick n channels (indices into B) that are mutually decorrelated.

    Start from the best-seeded candidate, then repeatedly add the candidate
    whose worst-case (max) correlation to the already-picked set is smallest.
    Maximises coverage of distinct functional activity -> higher state rank.
    """
    if n >= len(candidates):
        return candidates
    Bz = B - B.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Bz, axis=1, keepdims=True) + 1e-12
    Bn = Bz / norm
    picked = [seed_order[0]]
    cand = [c for c in candidates if c != picked[0]]
    while len(picked) < n and cand:
        # correlation of each candidate to each picked channel
        C = np.abs(Bn[cand] @ Bn[picked].T)        # (n_cand, n_picked)
        worst = C.max(axis=1)
        nxt = cand[int(np.argmin(worst))]
        picked.append(nxt)
        cand.remove(nxt)
    return picked


@dataclass
class IOSelection:
    input_channels: List[int]
    output_channels: List[int]
    input_electrodes: List[int]       # routed electrode ids (if mapping known)
    output_electrodes: List[int]
    scores: Dict[str, np.ndarray]
    channels: List[int]               # channel order the scores index into
    W: np.ndarray
    delay_ms: np.ndarray              # per-edge peak latency (NaN off-edges)


def select_io_electrodes(data: NetworkAssayData,
                         n_inputs: int,
                         n_outputs: int,
                         bin_ms: float = 5.0,
                         delays_ms: Sequence[float] = (2, 4, 6, 8, 10, 14, 18),
                         min_rate_hz: float = 0.1,
                         max_channels: int = 256,
                         input_min_spacing_um: float = 100.0,
                         stimulable_channels: Optional[Sequence[int]] = None,
                         exclude_input_neighbours_um: float = 50.0,
                         n_surrogates: int = 20,
                         rng_seed: int = 0,
                         qc_plot_path: Optional[str] = None) -> IOSelection:
    """End-to-end: Network-assay data -> input (source) & output (sink) picks.

    n_inputs  : how many stimulation electrodes to shortlist (e.g. q+2l, i.e.
                CFG.hw.stim_clusters_per_well). Validate these with a stim scan.
    n_outputs : how many readout electrodes (<= CFG.hw.readout_channels_per_well).
    delays_ms : delays (ms) swept for the PEAK-DELAY connectivity -- each pair
                keeps its best delay, so long-range drivers are not undercounted
                (see directed_connectivity_peakdelay). Keep within ~1-20 ms.
    max_channels : prefilter to this many most-active channels before the
                O(N^2) TE -- raise if you have compute budget.
    stimulable_channels : restrict inputs to electrodes routable to a stim unit
                (only 32 per well). Strongly recommended on real hardware.

    Inputs are ranked by directedness (sources), spacing-filtered and limited to
    stimulable channels. Outputs are sinks, then chosen for decorrelation/
    coverage (in-strength seeds the order), excluding inputs and their
    neighbourhood (stim-artifact zone).
    """
    # --- prefilter to active channels (keeps TE tractable) ------------------
    rates = data.firing_rates()
    active = [c for c in data.channels if rates.get(c, 0) >= min_rate_hz]
    active.sort(key=lambda c: rates[c], reverse=True)
    active = active[:max_channels]
    if len(active) < n_inputs + n_outputs:
        warnings.warn(f"Only {len(active)} active channels after prefilter; "
                      "lower min_rate_hz or record longer.")

    pk = directed_connectivity_peakdelay(
        data, channels=active, bin_ms=bin_ms, delays_ms=delays_ms,
        n_surrogates=n_surrogates, rng_seed=rng_seed)
    W, chans, D_ms = pk.W, pk.chans, pk.delay_ms
    sc = node_scores(W)
    B, _ = binned_matrix(data, bin_ms, chans)

    stim_set = set(stimulable_channels) if stimulable_channels is not None else None

    # --- INPUTS: strongest sources, stimulable, spatially separated ----------
    src_rank = list(np.argsort(-sc["directedness"]))
    if stim_set is not None:
        src_rank = [i for i in src_rank if chans[i] in stim_set]
    in_idx = _spatially_filtered(src_rank, data.positions, chans,
                                 input_min_spacing_um, n_inputs)

    # --- OUTPUTS: sinks, decorrelated coverage, away from inputs -------------
    in_channels = {chans[i] for i in in_idx}
    excl = set(in_channels)
    if data.positions and exclude_input_neighbours_um > 0:
        for i in in_idx:
            c = chans[i]
            if c in data.positions:
                x, y = data.positions[c]
                for k, cc in enumerate(chans):
                    if cc in data.positions:
                        xx, yy = data.positions[cc]
                        if (x - xx) ** 2 + (y - yy) ** 2 < exclude_input_neighbours_um ** 2:
                            excl.add(cc)
    out_candidates = [k for k, c in enumerate(chans) if c not in excl]
    sink_order = sorted(out_candidates, key=lambda k: -sc["in_strength"][k])
    out_idx = _greedy_decorrelated(B, out_candidates, sink_order, n_outputs)

    def _elecs(idxs):
        return [data.electrode.get(chans[i], chans[i]) for i in idxs]

    sel = IOSelection(
        input_channels=[chans[i] for i in in_idx],
        output_channels=[chans[i] for i in out_idx],
        input_electrodes=_elecs(in_idx),
        output_electrodes=_elecs(out_idx),
        scores=sc, channels=chans, W=W, delay_ms=D_ms)
    if qc_plot_path is not None and data.positions:
        plot_selection_qc(sel, data, qc_plot_path)
    return sel


def plot_selection_qc(sel: IOSelection, data: NetworkAssayData, path: str):
    """Automatic QC for an electrode selection (delay-vs-distance + spatial map).

    Left  : per-edge peak delay vs inter-electrode distance. If the recovered
            edges are real timed propagation, delay rises with distance (a clear
            positive trend / high Pearson r). A flat cloud warns that edges may
            be zero-lag common-input or noise rather than directed flow.
    Right : electrode map coloured by directedness, with the CHOSEN inputs
            (red squares) and outputs (blue circles) marked, so you can see
            where the picks landed and that inputs are spatially separated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = data.positions
    chans = sel.channels
    ci = {c: k for k, c in enumerate(chans)}
    # gather significant edges with a known distance and finite peak delay
    dist, delay = [], []
    for i, j in np.argwhere(sel.W > 0):
        ci_, cj_ = chans[int(i)], chans[int(j)]
        if ci_ in pos and cj_ in pos and np.isfinite(sel.delay_ms[i, j]):
            (x0, y0), (x1, y1) = pos[ci_], pos[cj_]
            dist.append(float(np.hypot(x1 - x0, y1 - y0)))
            delay.append(float(sel.delay_ms[i, j]))
    dist, delay = np.asarray(dist), np.asarray(delay)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.5))
    if dist.size >= 2 and dist.std() > 0:
        r = float(np.corrcoef(dist, delay)[0, 1])
        a0.scatter(dist, delay, s=30, alpha=0.6, edgecolor="k", linewidth=0.3)
        z = np.polyfit(dist, delay, 1)
        xv = np.linspace(dist.min(), dist.max(), 50)
        a0.plot(xv, np.polyval(z, xv), "r--",
                label=f"Pearson r={r:.2f}\nslope={z[0]*1000:.1f} ms/mm")
        a0.legend(loc="upper left", fontsize=9)
        verdict = ("PASS: delay rises with distance (timed propagation)"
                   if r > 0.3 else
                   "CHECK: weak delay-distance trend -- possible common input")
        a0.set_title("QC: peak delay vs distance\n" + verdict, fontsize=10)
    else:
        a0.text(0.5, 0.5, "not enough spatially-resolved edges", ha="center")
        a0.set_title("QC: peak delay vs distance")
    a0.set_xlabel("inter-electrode distance (um)")
    a0.set_ylabel("edge peak delay (ms)")

    xs = np.array([pos[c][0] for c in chans if c in pos])
    ys = np.array([pos[c][1] for c in chans if c in pos])
    dv = np.array([sel.scores["directedness"][ci[c]] for c in chans if c in pos])
    sc = a1.scatter(xs, ys, c=dv, cmap="coolwarm", vmin=-1, vmax=1, s=70,
                    edgecolor="0.4", linewidth=0.3, zorder=2)
    for c in sel.input_channels:
        if c in pos:
            a1.scatter(*pos[c], marker="s", s=200, facecolor="none",
                       edgecolor="red", linewidth=2.2, zorder=3)
    for c in sel.output_channels:
        if c in pos:
            a1.scatter(*pos[c], marker="o", s=200, facecolor="none",
                       edgecolor="blue", linewidth=1.8, zorder=3)
    a1.scatter([], [], marker="s", facecolor="none", edgecolor="red",
               linewidth=2.2, s=120, label="input (source)")
    a1.scatter([], [], marker="o", facecolor="none", edgecolor="blue",
               linewidth=1.8, s=120, label="output (sink)")
    a1.legend(loc="upper right", fontsize=9)
    a1.set_xlabel("x (um)"); a1.set_ylabel("y (um)")
    a1.set_title("chosen electrodes on the array\n(fill = directedness)", fontsize=10)
    a1.set_aspect("equal"); a1.invert_yaxis()
    fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.04, label="(out-in)/(out+in)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def select_for_well(data: NetworkAssayData, cfg, **kwargs) -> IOSelection:
    """Convenience wrapper that reads counts straight from CFG.

    Feeds CFG.hw.stim_clusters_per_well inputs and
    CFG.hw.readout_channels_per_well outputs into select_io_electrodes.
    The resulting .input_electrodes / .output_electrodes plug directly into
    hardware.MaxLabSession.configure_well(well_id, output_electrodes,
    input_electrodes).
    """
    return select_io_electrodes(
        data,
        n_inputs=cfg.hw.stim_clusters_per_well,
        n_outputs=cfg.hw.readout_channels_per_well,
        **kwargs)


# ======================================================================
# 6. Self-test: synthesise a driver -> sink network, recover the roles
# ======================================================================
def _synthesize_demo(n_ch=40, duration=120.0, fs=20_000.0, seed=0
                     ) -> NetworkAssayData:
    """Make fake spikes where channels 0-2 drive 3-9 with a fixed delay.

    Lets you run `python connectivity.py` with no hardware and confirm the
    selector tags the drivers as inputs and downstream cells as outputs.
    """
    rng = np.random.default_rng(seed)
    bw = 1e-3                          # 1 ms base grid
    n_bins = int(duration / bw)
    drivers = [0, 1, 2]
    sinks = [3, 4, 5, 6, 7, 8, 9]
    delay = 5                         # ms  (== 1 bin at the 5 ms analysis bin)
    spikes: Dict[int, List[float]] = {c: [] for c in range(n_ch)}
    # drivers: sparse Poisson events (sparse -> distinct, TE-friendly)
    drive_trains = {}
    for d in drivers:
        times = np.where(rng.random(n_bins) < 0.005)[0]   # ~5 Hz
        drive_trains[d] = times
        spikes[d] = list(times * bw)
    # sinks: fire shortly after their driver (strong, fairly reliable) + noise
    for s in sinks:
        src = drivers[s % len(drivers)]
        relayed = drive_trains[src] + delay
        relayed = relayed[relayed < n_bins]
        keep = relayed[rng.random(len(relayed)) < 0.9]
        noise = np.where(rng.random(n_bins) < 0.001)[0]
        spikes[s] = sorted(set(list(keep * bw) + list(noise * bw)))
    # the rest: background noise only
    for c in range(n_ch):
        if c not in drivers and c not in sinks:
            noise = np.where(rng.random(n_bins) < 0.003)[0]
            spikes[c] = list(noise * bw)
    pos = {c: (float((c % 8) * 60), float((c // 8) * 60)) for c in range(n_ch)}
    return NetworkAssayData(spikes={c: np.asarray(v, float) for c, v in spikes.items()},
                            duration=duration, fs=fs, positions=pos,
                            electrode={c: 1000 + c for c in range(n_ch)})


def _synthesize_spatial_demo(seed=0, duration=90.0, nside=6, pitch_um=60.0,
                             velocity_um_per_ms=30.0
                             ) -> Tuple[NetworkAssayData, List[Tuple[int, int, float]]]:
    """Grid of electrodes with DISTANCE-DEPENDENT conduction delays.

    A handful of drivers each project to several targets; the propagation delay
    of every edge is distance / velocity, so nearby targets get short delays and
    distant targets get long delays. This is the right testbed for the question
    "does a single delay just pick short-range connections?" -- sweeping the
    delay should reveal near edges at small delays and far edges at large ones.

    Returns (data, true_edges) where true_edges = [(src, dst, delay_ms), ...].
    """
    rng = np.random.default_rng(seed)
    n_ch = nside * nside
    pos = {r * nside + c: (c * pitch_um, r * pitch_um)
           for r in range(nside) for c in range(nside)}
    bw = 1e-3
    n_bins = int(duration / bw)
    spikes: Dict[int, List[float]] = {}
    for ch in range(n_ch):                       # background firing everywhere
        spikes[ch] = list(np.where(rng.random(n_bins) < 0.003)[0] * bw)

    drivers = [0, nside - 1, n_ch - nside, n_ch - 1]   # four corners
    drive_trains = {}
    for d in drivers:
        times = np.where(rng.random(n_bins) < 0.006)[0]    # ~6 Hz
        drive_trains[d] = times
        spikes[d] = sorted(set(spikes[d]) | set(times * bw))

    true_edges: List[Tuple[int, int, float]] = []
    pool = [c for c in range(n_ch) if c not in drivers]
    for d in drivers:
        xd, yd = pos[d]
        targets = rng.choice(pool, size=5, replace=False)
        for t in targets:
            xt, yt = pos[t]
            dist = float(np.hypot(xt - xd, yt - yd))
            delay_ms = max(2.0, round(dist / velocity_um_per_ms))
            relayed = drive_trains[d] + int(delay_ms)
            relayed = relayed[relayed < n_bins]
            keep = relayed[rng.random(len(relayed)) < 0.85]
            spikes[t] = sorted(set(spikes[t]) | set(keep * bw))
            true_edges.append((int(d), int(t), float(delay_ms)))

    data = NetworkAssayData(
        spikes={c: np.asarray(v, float) for c, v in spikes.items()},
        duration=duration, positions=pos,
        electrode={c: 1000 + c for c in range(n_ch)})
    return data, true_edges


if __name__ == "__main__":
    print("Synthesising a driver->sink test network (no hardware)...")
    data = _synthesize_demo()
    sel = select_io_electrodes(data, n_inputs=3, n_outputs=6,
                               bin_ms=5.0, min_rate_hz=0.0,
                               input_min_spacing_um=0.0, n_surrogates=10)
    print("\nGround truth : drivers = [0,1,2], sinks = [3..9]")
    print("Picked INPUT  channels (expect the sources 0,1,2):", sorted(sel.input_channels))
    print("Picked OUTPUT channels: top sink by in-strength, then DECORRELATED")
    print("  coverage (the sinks relay the same drivers, so they are mutually")
    print("  redundant -- RC wants spread, not a redundant clump):",
          sorted(sel.output_channels))
    print("\ndirectedness (per channel, sorted high->low; >0 = source):")
    order = np.argsort(-sel.scores["directedness"])
    for k in order[:10]:
        print(f"  ch {sel.channels[k]:>3}  directedness={sel.scores['directedness'][k]:+.2f}"
              f"  out={sel.scores['out_strength'][k]:.3f}  in={sel.scores['in_strength'][k]:.3f}")
