"""
Minimal stimulation test for the MaxWell host (run in VS Code on the rig PC).

PURPOSE: a self-contained "can I make an electrode stimulate from Python?" check,
independent of the rest of the pipeline. Once this works, the exact same calls
fill the TODOs in hardware.py (routing) and encoding.py (the pulse Sequence).

It follows the documented MaxLab Live pattern (api-docs.mxwbio.com -> Examples ->
Stimulation):
  1. initialize the system, set amplifier gain
  2. build an Array('stimulation'), select + route the stim electrode(s)
  3. connect each electrode to a stimulation unit, query which unit it got
  4. download config; power up each StimulationUnit in voltage mode on DAC 0
  5. build a charge-balanced biphasic pulse train as a maxlab.Sequence and send it
  6. power the units down

Run OFF the rig to read the plan (no hardware needed):
    python stim_test.py --dry --electrodes 12044,9810

Run ON the rig (MaxLab Live installed, MaxTwo connected):
    python stim_test.py --electrodes 12044 --amp 100 --n-pulses 10

!!! SAFETY -------------------------------------------------------------------
  * Pulses are charge-balanced biphasic. START LOW (amp ~ 50-100 DAC LSB) and
    increase slowly while watching impedance / evoked response.
  * Keep charge per phase within MaxWell's recommended limits to protect the
    electrodes and tissue. Do not leave units powered up.
-----------------------------------------------------------------------------

!!! VERIFY against YOUR installed API version (calls differ slightly across
    MaxLab versions): the names marked [VERIFY] below -- especially
    select_stimulation_electrodes / connect_electrode_to_stimulation /
    query_stimulation_at_electrode, and the MaxTwo WELL-SELECTION step (line
    marked [MAXTWO]). The api-docs Stimulation example is the source of truth.
"""
import argparse
import time

try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    _HAVE_MAXLAB = True
except Exception:                       # noqa
    _HAVE_MAXLAB = False

FS_HZ = 20_000.0                         # MaxTwo sampling rate


def append_biphasic_pulse(seq, amplitude_dac, phase_samples):
    """One charge-balanced biphasic voltage pulse on DAC 0.

    Voltage mode uses an INVERTING amplifier, so to get a positive-first pulse
    you SUBTRACT from the 512 midpoint first, then ADD, then return to 512.
    amplitude_dac is in DAC LSBs; phase_samples is the duration of EACH phase.
    """
    seq.append(maxlab.chip.DAC(0, 512 - amplitude_dac))   # phase 1
    seq.append(maxlab.system.DelaySamples(phase_samples))
    seq.append(maxlab.chip.DAC(0, 512 + amplitude_dac))   # phase 2 (balancing)
    seq.append(maxlab.system.DelaySamples(phase_samples))
    seq.append(maxlab.chip.DAC(0, 512))                   # back to baseline
    return seq


def print_plan(args, phase_samples, ipi_samples):
    print("=== STIM PLAN (dry run -- nothing sent) ===")
    print(f"  electrodes        : {args.electrodes}")
    print(f"  amplitude         : {args.amp} DAC LSB  (~{args.amp*2.9:.0f} mV, VERIFY mV/LSB)")
    print(f"  phase duration    : {phase_samples} samples = "
          f"{phase_samples/FS_HZ*1e6:.0f} us/phase  (biphasic -> "
          f"{2*phase_samples/FS_HZ*1e6:.0f} us total)")
    print(f"  pulses            : {args.n_pulses} at {args.rate_hz} Hz "
          f"(inter-pulse {ipi_samples} samples)")
    print(f"  gain              : {args.gain}")
    print("  sequence per pulse: DAC(0,512-A) wait, DAC(0,512+A) wait, DAC(0,512)")
    print("  (on the rig this routes the electrodes, powers the stim units, and")
    print("   sends the Sequence; run without --dry on the MaxLab host to do it.)")


def main():
    ap = argparse.ArgumentParser(description="Minimal MaxLab Live stimulation test")
    ap.add_argument("--electrodes", required=True,
                    help="comma-separated electrode ids to stimulate, e.g. 12044,9810")
    ap.add_argument("--well", default=None,
                    help="MaxTwo well to activate, e.g. well000 (see [MAXTWO] note)")
    ap.add_argument("--amp", type=int, default=100, help="amplitude in DAC LSB (START LOW)")
    ap.add_argument("--phase-us", type=int, default=200, help="duration per phase (us)")
    ap.add_argument("--n-pulses", type=int, default=10)
    ap.add_argument("--rate-hz", type=float, default=10.0, help="pulse rate within the train")
    ap.add_argument("--gain", type=int, default=512)
    ap.add_argument("--dry", action="store_true", help="print the plan, send nothing")
    args = ap.parse_args()

    electrodes = [int(e) for e in args.electrodes.split(",") if e.strip()]
    phase_samples = max(1, round(args.phase_us * 1e-6 * FS_HZ))
    ipi_samples = max(1, round(FS_HZ / args.rate_hz) - 2 * phase_samples)

    if args.dry or not _HAVE_MAXLAB:
        if not _HAVE_MAXLAB and not args.dry:
            print("maxlab not importable here -> showing plan only.\n")
        print_plan(args, phase_samples, ipi_samples)
        return

    # ---- 1. initialize ----------------------------------------------------
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Amplifier().set_gain(args.gain))
    # [MAXTWO] If on a MaxTwo plate, activate the target well here before routing.
    #   The well-selection call is install-specific -- check the api-docs MaxTwo
    #   example (commonly a maxlab.util/maxlab.system well/subchip selection).

    # ---- 2. route the stimulation electrodes ------------------------------
    array = maxlab.chip.Array("stimulation")
    array.reset()
    array.select_stimulation_electrodes(electrodes)         # [VERIFY]
    array.route()

    # ---- 3. connect each electrode to a stim unit; find the unit ----------
    units = []
    for e in electrodes:
        array.connect_electrode_to_stimulation(e)           # [VERIFY]
        u = array.query_stimulation_at_electrode(e)         # [VERIFY] returns unit id
        if not u:
            raise RuntimeError(f"electrode {e} could not be connected to a stim unit "
                               "(none free, or not routable). Pick another electrode.")
        units.append(u)
    array.download()
    maxlab.util.offset()

    # ---- 4. power up each stimulation unit (voltage mode, DAC source 0) ----
    stim_units = []
    for u in units:
        s = (maxlab.chip.StimulationUnit(u)
             .power_up(True).connect(True).set_voltage_mode().dac_source(0))
        maxlab.send(s)
        stim_units.append(s)

    # ---- 5. build and send the pulse train --------------------------------
    seq = maxlab.Sequence()
    for _ in range(args.n_pulses):
        append_biphasic_pulse(seq, args.amp, phase_samples)
        seq.append(maxlab.system.DelaySamples(ipi_samples))
    print(f"Sending {args.n_pulses} pulses to electrodes {electrodes} "
          f"(units {units}) ...")
    seq.send()
    time.sleep(args.n_pulses / args.rate_hz + 0.5)

    # ---- 6. power down ----------------------------------------------------
    for s in stim_units:
        maxlab.send(s.power_up(False))
    print("Done. Stim units powered down.")


if __name__ == "__main__":
    main()
