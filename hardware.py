"""
Thin wrapper around the MaxLab Live API for the MaxTwo 6-well system.

This is a SCAFFOLD: the method bodies marked TODO must be completed against
your MaxLab Live install (api-docs.mxwbio.com). Everything else in the codebase
is hardware-agnostic and already runs via the ESN surrogate.

Responsibilities:
  * open/close a session, power up wells
  * configure per-well electrode arrays from a connectivity.IOSelection:
    route the chosen stimulation electrodes to stim units, select the chosen
    readout electrodes for recording
  * deliver a stimulation Sequence to a well and return evoked spike events
  * deliver a quiet/washout interval

It consumes the output of connectivity.py directly:
    sel = connectivity.select_for_well(network_assay_data, CFG)   # one per well
    session.configure_from_selection({well_id: sel, ...})
sel.input_electrodes  -> stimulation electrodes (len == q+2l)
sel.output_channels   -> readout channel order that the SpikeDecoder consumes
sel.output_electrodes -> the electrodes routed to those readout channels

IMPORTANT (latency): the autoregressive loop reads spikes and emits the next
stimulus every frame. For tight closed-loop timing MaxWell recommend the C++
API. This Python wrapper is fine for slower frames and for Phases 0-2.
"""
try:
    import maxlab
    import maxlab.chip
    import maxlab.util
    import maxlab.system
    _HAVE_MAXLAB = True
except Exception:           # noqa
    _HAVE_MAXLAB = False


class MaxLabSession:
    def __init__(self, cfg, allow_dry=False):
        """
        allow_dry=True lets you build a session WITHOUT the MaxLab API, to test
        the wiring (electrode routing maps, encoder/decoder construction). A dry
        session cannot stimulate_and_record. On the rig, leave allow_dry=False.
        """
        self.cfg = cfg
        self.dry = not _HAVE_MAXLAB
        self.well_stim_map = {}      # well_id -> [stim unit handles] (1 per input)
        self.well_channels = {}      # well_id -> [readout channel ids] (decoder order)
        if self.dry and not allow_dry:
            raise RuntimeError(
                "MaxLab Live API not found. Use the ESN surrogate (run_demo.py) "
                "for dry runs, install MaxLab Live on the rig PC, or pass "
                "allow_dry=True to test electrode wiring only.")

    # -- lifecycle ----------------------------------------------------------
    def open(self):
        if self.dry:
            return
        maxlab.util.initialize()
        maxlab.send(maxlab.chip.Amplifier().set_gain(512))
        # TODO: power up the wells you are using

    def close(self):
        # TODO: power down stimulation units, stop recordings
        pass

    # -- configuration ------------------------------------------------------
    def configure_well(self, well_id, readout_channels, stim_electrodes,
                       readout_electrodes=None):
        """
        Route `stim_electrodes` (len == q+2l) to stimulation units and select
        `readout_electrodes` for recording. `readout_channels` is the channel
        order the SpikeDecoder will key on (== connectivity IOSelection
        .output_channels). Populates well_stim_map (one stim handle per input
        electrode) and well_channels.
        """
        stim_electrodes = list(stim_electrodes)
        self.well_channels[well_id] = list(readout_channels)
        if self.dry:
            # stand-in "handles": just the electrode ids, so the encoder can run
            self.well_stim_map[well_id] = list(stim_electrodes)
            return
        # TODO (hardware): build maxlab.chip.Array('stimulation'),
        #   array.select_stimulation_electrodes(stim_electrodes); array.route();
        #   for e in stim_electrodes: array.connect_electrode_to_stimulation(e);
        #   handles.append(array.query_stimulation_at_electrode(e));
        #   also select readout_electrodes for recording; array.download().
        handles = []                                   # fill with stim handles
        self.well_stim_map[well_id] = handles

    def configure_from_selection(self, selections):
        """
        selections: {well_id: connectivity.IOSelection}. Configures every well
        directly from the connectivity output. This is the single hand-off point
        between connectivity.py and the live experiment.
        """
        for well_id, sel in selections.items():
            self.configure_well(
                well_id,
                readout_channels=sel.output_channels,
                stim_electrodes=sel.input_electrodes,
                readout_electrodes=sel.output_electrodes)
        return self

    # -- per-frame I/O ------------------------------------------------------
    def stimulate_and_record(self, well_id, sequence, record_ms):
        """
        Send `sequence` to `well_id`, record for record_ms, return spike events
        as [(channel_id, t_seconds_from_window_start), ...] on the well's
        readout channels (well_channels[well_id]).
        """
        if self.dry:
            raise RuntimeError("dry session cannot stimulate_and_record; "
                               "install MaxLab Live on the rig.")
        # TODO: start recording; sequence.send(); collect online spike events
        raise NotImplementedError("connect to MaxLab Live spike stream")

    def quiet(self, well_id, ms):
        """Deliver no stimulus for `ms` (settle / washout)."""
        # TODO: optional explicit delay; often just a host-side sleep
        pass
