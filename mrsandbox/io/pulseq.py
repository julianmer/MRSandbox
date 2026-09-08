####################################################################################################
#                                            pulseq.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/09/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq import and export (via pypulseq): read_seq translates .seq blocks into the       #
#          event primitives with the ADC marking the readout, and write_seq turns a physical       #
#          event sequence back into a .seq file any Pulseq scanner or simulator can play.          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np

from mrsandbox.sequences.events import Delay, Gradient, Pulse, Sequence, ShapedPulse


#**************************************************************************************************#
#                                       Class PulseqSequence                                       #
#**************************************************************************************************#
#                                                                                                  #
# A Sequence read from a .seq file, carrying the readout the ADC block declared.                   #
#                                                                                                  #
#**************************************************************************************************#
class PulseqSequence(Sequence):
    """A Sequence read from a .seq file, carrying the readout the ADC block declared.

    Attributes:
        npoints: ADC samples of the declared readout.
        dwell_s: ADC dwell time in seconds.
        adc_freq_offset_hz, adc_phase_deg: The ADC frequency and phase offsets of the
            file, recorded for the caller (the engine reads out unshifted).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, events, npoints, dwell_s, adc_freq_offset_hz=0.0, adc_phase_deg=0.0):
        super().__init__(*events)
        self.npoints = npoints
        self.dwell_s = dwell_s
        self.adc_freq_offset_hz = adc_freq_offset_hz
        self.adc_phase_deg = adc_phase_deg


#******************#
#   rf to events   #
#******************#
def _shaped_pulse(rf, center_ppm, steps, traps, extent_mm):
    """One RF event as a ShapedPulse: amplitude and phase resampled on a uniform grid.

    Pulseq RF amplitudes are already in Hz, so the pulse plays in peak_hz mode (exact
    amplitudes, scaled by the drawn transmit field like any RF). The carrier sits at
    center_ppm plus the file's ppm offset; the file's Hz offset and any phase
    modulation become the per-step frequency track, with the half-step phase
    correction so the midpoint-sampled phase is exact.
    """
    signal = np.asarray(rf.signal, dtype=np.complex128)
    t = np.asarray(rf.t, dtype=np.float64)
    duration = float(rf.shape_dur)
    dt = duration / steps
    mid = (np.arange(steps) + 0.5) * dt
    amplitude = np.interp(mid, t, np.abs(signal))
    phase = np.interp(mid, t, np.unwrap(np.angle(signal)))
    peak = float(amplitude.max())
    freq_offset = float(rf.freq_offset)
    sweep = np.diff(phase, prepend=phase[0]) / (2.0 * math.pi * dt) + freq_offset
    phase0_deg = math.degrees(float(phase[0]) + float(rf.phase_offset)
                              - math.pi * freq_offset * dt)
    flat = [(channel, grad) for channel, grad in traps.items() if grad.amplitude]
    gradient = float(flat[0][1].amplitude) / 1000.0 if flat else 0.0
    axis = flat[0][0][1] if flat else "x"
    return ShapedPulse(
        flip_deg=None,
        duration_s=duration,
        envelope=(amplitude / peak).astype(np.float32),
        offset_ppm=center_ppm + float(getattr(rf, "freq_ppm", 0.0)),
        phase_deg=phase0_deg,
        frequency_hz=sweep.astype(np.float32) if np.any(sweep) else None,
        peak_hz=peak,
        gradient_hz_per_mm=gradient,
        gradient_axis=axis,
        extent_mm=extent_mm,
    )


#****************#
#   seq import   #
#****************#
def read_seq(path, center_ppm=4.65, steps=128, extent_mm=20.0):
    """Read a Pulseq .seq file into a PulseqSequence of event primitives.

    Blocks translate one to one: RF events become ShapedPulses (with an optional flat
    trapezoid under the pulse as the slice gradient), gradient-only blocks become a
    physical Gradient kick of the dominant axis's area between two half-duration
    delays, empty blocks become delays, and the first ADC block ends the sequence and
    declares the readout. The .seq frame puts the carrier at the scanner reference,
    which `center_ppm` places on the ppm axis (water at 4.65).

    Stated v1 idealizations: trapezoid gradients only (every axis's area becomes its
    own kick); ramp moments outside an RF pulse are not applied; blocks after the
    first ADC are ignored (one readout per file).

    Args:
        path: The .seq file.
        center_ppm: Position of the scanner carrier on the ppm axis.
        steps: Uniform resampling steps per RF pulse.
        extent_mm: Isochromat span handed to gradient-bearing events.
    """
    try:
        import pypulseq
    except ImportError:
        raise ImportError("Pulseq import needs pypulseq: pip install pypulseq") from None

    source = pypulseq.Sequence()
    source.read(str(path))
    events = []
    for index in sorted(source.block_events):
        block = source.get_block(index)
        adc = block.adc
        if adc is not None:
            if getattr(adc, "phase_modulation", None) is not None and np.any(adc.phase_modulation):
                raise ValueError("ADC phase modulation is not supported")
            if float(adc.delay) > 0.0:
                events.append(Delay(float(adc.delay)))
            return PulseqSequence(events, npoints=int(adc.num_samples), dwell_s=float(adc.dwell),
                                  adc_freq_offset_hz=float(adc.freq_offset),
                                  adc_phase_deg=math.degrees(float(adc.phase_offset)))
        traps = {axis: grad for axis in ("gx", "gy", "gz")
                 if (grad := getattr(block, axis)) is not None}
        for grad in traps.values():
            if grad.type != "trap":
                raise ValueError("only trapezoid gradients are supported")
        rf = block.rf
        duration = float(block.block_duration)
        if rf is not None:
            if float(getattr(rf, "phase_ppm", 0.0)):
                raise ValueError("RF phase_ppm is not supported")
            if float(rf.delay) > 0.0:
                events.append(Delay(float(rf.delay)))
            events.append(_shaped_pulse(rf, center_ppm, steps, traps, extent_mm))
            tail = duration - float(rf.delay) - float(rf.shape_dur)
            if tail > 1e-9:
                events.append(Delay(tail))
        elif traps:
            events.append(Delay(duration / 2.0))
            for channel, grad in traps.items():
                if float(grad.area):
                    events.append(Gradient(moment_per_mm=float(grad.area) / 1000.0,
                                           extent_mm=extent_mm, axis=channel[1]))
            events.append(Delay(duration / 2.0))
        elif duration > 0.0:
            events.append(Delay(duration))
    raise ValueError("the .seq file has no ADC block: nothing marks the readout")


#***********************#
#   waveform to hertz   #
#***********************#
def _export_waveform(pulse, f0_mhz, center_ppm):
    """The exact complex Hz waveform a ShapedPulse plays, with the carrier expressed
    as a Hz offset from the scanner reference at center_ppm."""
    envelope = np.asarray(pulse.envelope, dtype=np.float64)
    dt = float(pulse.duration_s) / envelope.size
    if pulse.peak_hz is not None:
        amplitude = envelope * float(pulse.peak_hz)
    else:
        amplitude = envelope * (float(pulse.flip_deg) / 360.0) / (envelope.sum() * dt)
    phase = np.full(envelope.size, math.radians(float(pulse.phase_deg)))
    if pulse.frequency_hz is not None:
        sweep = np.asarray(pulse.frequency_hz, dtype=np.float64)
        phase = phase + 2.0 * math.pi * np.cumsum(sweep) * dt
    offsets = pulse.offset_ppm if isinstance(pulse.offset_ppm, tuple) else (pulse.offset_ppm,)
    if len(offsets) != 1:
        raise ValueError("multi-band pulses have no single carrier to export")
    freq_offset = (float(offsets[0]) - center_ppm) * f0_mhz
    return amplitude * np.exp(1j * phase), dt, freq_offset


#****************#
#   seq export   #
#****************#
def write_seq(sequence, path, npoints, dwell_s, f0_mhz, center_ppm=4.65,
              hard_pulse_s=1e-4, system=None):
    """Write a physical event Sequence to a Pulseq .seq file, the ADC appended.

    Hardware-realizable events export exactly: hard Pulses play as hard_pulse_s block
    pulses, ShapedPulses as their exact Hz waveforms (with a flat slice trapezoid when
    the pulse carries a gradient, and the carrier as the Hz offset from the scanner
    reference at center_ppm), Delays as delays, and Gradients as minimal-duration
    trapezoids on their own axis's channel.
    CoherenceFilters and SelectivePulses are validated idealizations without a
    hardware counterpart and raise: export the physical design, with crushers in place
    of filters and shaped pulses in place of ideal bands.

    Ideal events are instantaneous in the simulation but not on hardware, so their
    block durations are borrowed from the following delays: the total sequence
    duration is preserved exactly (delays too short to lend raise), event centres
    shift by at most half the borrowed time, and emitted delays round to the block
    raster. Waveform steps must sit on the RF raster of `system`.
    """
    try:
        import pypulseq
    except ImportError:
        raise ImportError("Pulseq export needs pypulseq: pip install pypulseq") from None

    system = system if system is not None else pypulseq.Opts()
    seq = pypulseq.Sequence(system=system)
    raster = system.block_duration_raster
    owed = 0.0
    for event in sequence.events:
        if isinstance(event, Delay):
            remaining = float(event.t_s) - owed
            if remaining < -1e-9:
                owed = -remaining
                continue
            owed = 0.0
            rounded = round(remaining / raster) * raster
            if rounded > 1e-9:
                seq.add_block(pypulseq.make_delay(rounded))
        elif isinstance(event, ShapedPulse):
            signal, dt, freq_offset = _export_waveform(event, f0_mhz, center_ppm)
            use = "refocusing" if (event.peak_hz is not None
                                   or float(event.flip_deg) >= 135.0) else "excitation"
            rf = pypulseq.make_arbitrary_rf(
                signal=signal, flip_angle=2.0 * math.pi * np.abs(signal).sum() * dt,
                dwell=dt, freq_offset=freq_offset, no_signal_scaling=True,
                system=system, use=use)
            if event.gradient_hz_per_mm:
                slab = pypulseq.make_trapezoid(
                    channel=event.gradient_axis, amplitude=float(event.gradient_hz_per_mm) * 1000.0,
                    flat_time=float(event.duration_s), system=system)
                rf.delay = max(rf.delay, slab.rise_time)
                seq.add_block(rf, slab)
                owed += rf.delay + slab.fall_time + rf.ringdown_time
            else:
                seq.add_block(rf)
                owed += rf.delay + rf.ringdown_time
        elif isinstance(event, Pulse):
            use = "refocusing" if float(event.flip_deg) >= 135.0 else "excitation"
            rf = pypulseq.make_block_pulse(
                flip_angle=math.radians(float(event.flip_deg)), duration=hard_pulse_s,
                phase_offset=math.radians(float(event.phase_deg)), system=system, use=use)
            seq.add_block(rf)
            owed += rf.delay + hard_pulse_s + rf.ringdown_time
        elif isinstance(event, Gradient):
            trap = pypulseq.make_trapezoid(
                channel=event.axis, area=float(event.moment_per_mm) * 1000.0, system=system)
            seq.add_block(trap)
            owed += trap.rise_time + trap.flat_time + trap.fall_time
        else:
            raise ValueError(
                f"{type(event).__name__} is a validated idealization with no hardware "
                f"counterpart; export the physical design instead (crushers for "
                f"coherence filters, shaped pulses for ideal-selective bands)")
    if owed > 1e-9:
        raise ValueError(f"the delays are {owed * 1e3:.3f} ms too short to absorb the "
                         f"hardware event durations; lengthen them or shorten the pulses")
    seq.add_block(pypulseq.make_adc(num_samples=int(npoints), dwell=float(dwell_s),
                                    system=system))
    seq.write(str(path))
