"""
Compatibility test for load_network_assay against the MaxWell HDF5 schema.

We don't have a real rig file here (and the raw traces need MaxWell's
compression plugin), so we build FAITHFUL MOCK .h5 files that replicate the
documented layout used by neo's MaxwellRawIO -- both the OLD MaxOne format
(version 20160704: /sig, /mapping, /proc0/spikeTimes) and the NEW multi-well
format (/wells/wellXXX/recXXXX/{spikes, settings/{mapping, sampling}}) -- then
round-trip them through the loader and assert the spikes, channel map, electrode
ids, positions and sampling rate all come back correctly. The `spikes` and
`mapping` datasets are uncompressed in real files too, so this exercises exactly
the path a real Network-assay export takes.

Run:  python test_maxwell_compat.py
"""
import os
import tempfile
import numpy as np
import h5py

import connectivity as cx

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.gettempdir()             # mock files go here (kept off the repo)
FS = 20_000.0
FRAME_OFFSET = 1_000_000        # absolute-frame origin, like a real recording

MAP_DTYPE = np.dtype([("channel", "<i4"), ("electrode", "<i4"),
                      ("x", "<f8"), ("y", "<f8")])
SPIKE_DTYPE = np.dtype([("frameno", "<i8"), ("channel", "<i4"),
                        ("amplitude", "<f4")])


def _mapping_array(data):
    rows = [(c, data.electrode[c], data.positions[c][0], data.positions[c][1])
            for c in data.channels]
    return np.array(rows, dtype=MAP_DTYPE)


def _spike_array(data):
    fr, ch, amp = [], [], []
    for c in data.channels:
        for t in data.spikes[c]:
            fr.append(int(round(t * FS)) + FRAME_OFFSET)
            ch.append(c)
            amp.append(-20.0)
    order = np.argsort(fr)               # files are time-ordered
    arr = np.empty(len(fr), dtype=SPIKE_DTYPE)
    arr["frameno"] = np.array(fr)[order]
    arr["channel"] = np.array(ch)[order]
    arr["amplitude"] = np.array(amp)[order]
    return arr


def write_new_format(path, data, well="well000", rec="rec0000"):
    with h5py.File(path, "w") as f:
        f.create_dataset("version", data=np.array([b"20190530"]))
        g = f.create_group(f"wells/{well}/{rec}")
        s = g.create_group("settings")
        s.create_dataset("sampling", data=np.array([FS]))
        s.create_dataset("mapping", data=_mapping_array(data))
        g.create_dataset("spikes", data=_spike_array(data))
        # a stand-in for the (compressed) raw traces we deliberately don't read
        g.create_group("groups/routed").create_dataset(
            "raw", data=np.zeros((4, 4), dtype="uint16"))


def write_old_format(path, data):
    with h5py.File(path, "w") as f:
        f.create_dataset("version", data=np.array([b"20160704"]))
        f.create_dataset("sig", data=np.zeros((4, 4), dtype="uint16"))
        f.create_dataset("mapping", data=_mapping_array(data))
        f.create_group("proc0").create_dataset("spikeTimes", data=_spike_array(data))


def check(loaded, ref, label):
    assert abs(loaded.fs - FS) < 1e-6, f"{label}: fs {loaded.fs}"
    assert set(loaded.channels) == set(ref.channels), f"{label}: channel set"
    # electrode map + positions preserved
    for c in ref.channels:
        assert loaded.electrode[c] == ref.electrode[c], f"{label}: electrode {c}"
        assert np.allclose(loaded.positions[c], ref.positions[c]), f"{label}: pos {c}"
    # spike counts preserved, and cross-channel RELATIVE timing within one frame
    for c in ref.channels:
        assert len(loaded.spikes[c]) == len(ref.spikes[c]), f"{label}: count ch {c}"
    # relative timing: difference between two channels' first spikes preserved
    a, b = ref.channels[0], ref.channels[-1]
    if ref.spikes[a].size and ref.spikes[b].size:
        d_ref = ref.spikes[a].min() - ref.spikes[b].min()
        d_load = loaded.spikes[a].min() - loaded.spikes[b].min()
        assert abs(d_ref - d_load) < 1.5 / FS, f"{label}: relative timing"
    print(f"  [{label}] OK -- {len(loaded.channels)} ch, fs={loaded.fs:.0f}, "
          f"dur={loaded.duration:.1f}s, positions+electrodes match")


if __name__ == "__main__":
    print("Building reference network and writing mock MaxWell files...")
    ref, true_edges = cx._synthesize_spatial_demo(seed=3, duration=60.0)

    new_path = os.path.join(TMP, "_mock_maxtwo_new.raw.h5")
    old_path = os.path.join(TMP, "_mock_maxone_old.raw.h5")
    write_new_format(new_path, ref)
    write_old_format(old_path, ref)

    print("Round-tripping through load_network_assay:")
    loaded_new = cx.load_network_assay(new_path, well="well000", rec="rec0000")
    check(loaded_new, ref, "NEW / MaxTwo well000")
    loaded_old = cx.load_network_assay(old_path)
    check(loaded_old, ref, "OLD / MaxOne")

    # full pipeline on the loaded (not in-memory) data, incl. auto QC plot
    print("Running selection + QC on the LOADED new-format data...")
    sel = cx.select_io_electrodes(
        loaded_new, n_inputs=4, n_outputs=8, bin_ms=2.0,
        delays_ms=[2, 4, 6, 8, 10, 12, 14, 16], min_rate_hz=0.0,
        input_min_spacing_um=0.0, n_surrogates=6, rng_seed=3,
        qc_plot_path=os.path.join(HERE, "fig_conn4_selection_qc.png"))
    drivers = sorted({s for s, _, _ in true_edges})
    print(f"  inputs picked : {sorted(sel.input_channels)}")
    print(f"  true drivers  : {drivers}")
    print(f"  outputs picked: {sorted(sel.output_channels)} ({len(sel.output_channels)})")
    print("  QC figure     : fig_conn4_selection_qc.png")

    for p in (new_path, old_path):
        try:
            os.remove(p)
        except OSError:
            pass
    print("\nAll compatibility checks passed.")
