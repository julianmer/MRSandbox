####################################################################################################
#                                             pulseq.py                                            #
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


#**********#
#   area   #
#**********#
def _area(knots, levels, a, b):
    """The integral over [a, b] of a waveform linear between its knots, zero outside them."""
    inside = knots[(knots > a) & (knots < b)]
    times = np.concatenate([[a], inside, [b]])
    return float(np.trapezoid(np.interp(times, knots, levels, left=0.0, right=0.0), times))


#****************#
#   seq import   #
#****************#
def read_seq(path, center_ppm=4.65, steps=128, extent_mm=20.0, readout=0):
    """Read a Pulseq .seq file into a PulseqSequence of event primitives.

    Blocks translate one to one: RF events become ShapedPulses (with a flat trapezoid
    under the pulse as the slice gradient, or an extended trapezoid or arbitrary waveform
    sampled at the pulse's steps, modulated where it is not flat, its course before and
    after the pulse within the block as kicks), gradient-only blocks become a
    physical Gradient kick of the dominant axis's area between two half-duration
    delays, empty blocks become delays, and each ADC block declares one readout.
    `readout` selects which ADC marks the returned sequence (0-based); every earlier
    ADC block elapses as free evolution of its full duration, with its gradients
    kicking. read_seq_all returns every readout at once. The .seq frame puts the
    carrier at the scanner reference, which `center_ppm` places on the ppm axis
    (water at 4.65).

    Stated idealizations: every gradient axis's area becomes its own kick
    (trapezoids and arbitrary waveforms alike); a flat trapezoid under an RF pulse plays
    as the slice gradient over the whole pulse, its ramp moments outside the pulse not
    applied; one axis plays under a pulse; the readout itself is ungradiented, so
    events sharing an ADC block act only on the readouts after it.

    Args:
        path: The .seq file.
        center_ppm: Position of the scanner carrier on the ppm axis.
        steps: Uniform resampling steps per RF pulse.
        extent_mm: Isochromat span handed to gradient-bearing events.
        readout: Index of the ADC block whose readout the sequence carries.
    """
    sequences = read_seq_all(path, center_ppm=center_ppm, steps=steps, extent_mm=extent_mm)
    if not 0 <= readout < len(sequences):
        raise ValueError(f"readout {readout} does not exist: the file holds "
                         f"{len(sequences)} readout(s)")
    return sequences[readout]


def read_seq_all(path, center_ppm=4.65, steps=128, extent_mm=20.0):
    """Every readout of a .seq file as a tuple of PulseqSequence, one per ADC block.

    Readout k carries all blocks before its ADC; see read_seq for the translation
    and the stated idealizations. Each sequence declares its own npoints and dwell,
    so the readouts of a file may differ freely.
    """
    try:
        import pypulseq
    except ImportError:
        raise ImportError("Pulseq import needs pypulseq: pip install pypulseq") from None

    source = pypulseq.Sequence()
    source.read(str(path))
    events, sequences = [], []
    for index in sorted(source.block_events):
        block = source.get_block(index)
        adc = block.adc
        traps = {axis: grad for axis in ("gx", "gy", "gz")
                 if (grad := getattr(block, axis)) is not None}
        rf = block.rf
        if adc is not None:
            if getattr(adc, "phase_modulation", None) is not None and np.any(adc.phase_modulation):
                raise ValueError("ADC phase modulation is not supported")
            pre = list(events)
            if float(adc.delay) > 0.0:
                pre.append(Delay(float(adc.delay)))
            sequences.append(PulseqSequence(pre, npoints=int(adc.num_samples),
                                            dwell_s=float(adc.dwell),
                                            adc_freq_offset_hz=float(adc.freq_offset),
                                            adc_phase_deg=math.degrees(float(adc.phase_offset))))
            # the block then elapses in full before any later readout: fall through,
            # so its gradients kick and its duration passes as free evolution
        duration = float(block.block_duration)
        if rf is not None:
            if float(getattr(rf, "phase_ppm", 0.0)):
                raise ValueError("RF phase_ppm is not supported")
            if float(rf.delay) > 0.0:
                events.append(Delay(float(rf.delay)))
            # the RF event as a ShapedPulse, amplitude and phase resampled on a uniform grid.
            # Pulseq RF amplitudes are already in Hz, so the pulse plays in peak_hz mode
            # (exact amplitudes, scaled by the drawn transmit field like any RF). The
            # carrier sits at center_ppm plus the file's ppm offset; the file's Hz offset
            # and any phase modulation become the per-step frequency track, with the
            # half-step phase correction so the midpoint-sampled phase is exact.
            signal = np.asarray(rf.signal, dtype=np.complex128)
            t = np.asarray(rf.t, dtype=np.float64)
            dt = float(rf.shape_dur) / steps
            mid = (np.arange(steps) + 0.5) * dt
            amplitude = np.interp(mid, t, np.abs(signal))
            phase = np.interp(mid, t, np.unwrap(np.angle(signal)))
            peak = float(amplitude.max())
            freq_offset = float(rf.freq_offset)
            sweep = np.diff(phase, prepend=phase[0]) / (2.0 * math.pi * dt) + freq_offset
            # every gradient of the block (Hz/m along its own times) split at the pulse: the
            # one axis that plays under the pulse is its slice gradient, a flat trapezoid over
            # the whole pulse (its ramps unapplied, stated), an extended trapezoid or an
            # arbitrary waveform at the pulse's steps, modulated where it is not flat; what
            # any axis plays before or after the pulse within the block kicks by its area
            start, end = float(rf.delay), float(rf.delay) + float(rf.shape_dur)
            slice_hz_per_mm, axis, shape, before, after = 0.0, "x", None, [], []
            for channel, grad in traps.items():
                if grad.type == "trap":
                    if not grad.amplitude:
                        continue
                    knots = float(grad.delay) + np.cumsum([0.0, float(grad.rise_time),
                                                           float(grad.flat_time),
                                                           float(grad.fall_time)])
                    levels = np.array([0.0, grad.amplitude, grad.amplitude, 0.0], dtype=float)
                else:
                    knots = np.asarray(grad.tt, dtype=np.float64) + float(grad.delay)
                    levels = np.asarray(grad.waveform, dtype=np.float64)
                course = np.interp(start + mid, knots, levels, left=0.0, right=0.0)
                reference = course[np.abs(course).argmax()]
                if reference:
                    if slice_hz_per_mm:
                        raise ValueError("two gradient axes under one RF pulse are not supported")
                    axis = channel[1]
                    if grad.type == "trap":
                        slice_hz_per_mm = float(grad.amplitude) / 1000.0
                        continue
                    slice_hz_per_mm = float(reference) / 1000.0
                    if course.max() - course.min() > 1e-6 * abs(reference):
                        shape = (course / reference).astype(np.float32)
                for lobes, (a, b) in ((before, (0.0, start)), (after, (end, duration))):
                    moment = _area(knots, levels, a, b) / 1000.0
                    if moment:
                        lobes.append(Gradient(moment_per_mm=moment, extent_mm=extent_mm,
                                              axis=channel[1]))
            events.extend(before)
            events.append(ShapedPulse(
                flip_deg=None,
                duration_s=float(rf.shape_dur),
                envelope=(amplitude / peak).astype(np.float32),
                offset_ppm=center_ppm + float(getattr(rf, "freq_ppm", 0.0)),
                phase_deg=math.degrees(float(phase[0]) + float(rf.phase_offset)
                                       - math.pi * freq_offset * dt),
                frequency_hz=sweep.astype(np.float32) if np.any(sweep) else None,
                peak_hz=peak,
                gradient_hz_per_mm=slice_hz_per_mm,
                gradient_axis=axis,
                gradient_shape=shape,
                extent_mm=extent_mm,
            ))
            events.extend(after)
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
    if not sequences:
        raise ValueError("the .seq file has no ADC block: nothing marks the readout")
    return tuple(sequences)


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
            # the exact complex Hz waveform the pulse plays, its carrier as a Hz offset from
            # the scanner reference at center_ppm
            envelope = np.asarray(event.envelope, dtype=np.float64)
            dt = float(event.duration_s) / envelope.size
            if event.peak_hz is not None:
                amplitude = envelope * float(event.peak_hz)
            else:
                amplitude = envelope * (float(event.flip_deg) / 360.0) / (envelope.sum() * dt)
            phase = np.full(envelope.size, math.radians(float(event.phase_deg)))
            if event.frequency_hz is not None:
                phase = phase + 2.0 * math.pi * np.cumsum(
                    np.asarray(event.frequency_hz, dtype=np.float64)) * dt
            offsets = (event.offset_ppm if isinstance(event.offset_ppm, tuple)
                       else (event.offset_ppm,))
            if len(offsets) != 1:
                raise ValueError("multi-band pulses have no single carrier to export")
            signal, freq_offset = (amplitude * np.exp(1j * phase),
                                   (float(offsets[0]) - center_ppm) * f0_mhz)
            use = "refocusing" if (event.peak_hz is not None
                                   or float(event.flip_deg) >= 135.0) else "excitation"
            rf = pypulseq.make_arbitrary_rf(
                signal=signal, flip_angle=2.0 * math.pi * np.abs(signal).sum() * dt,
                dwell=dt, freq_offset=freq_offset, no_signal_scaling=True,
                system=system, use=use)
            if event.gradient_hz_per_mm and event.gradient_shape is not None:
                raise ValueError("a modulated slice gradient (gradient_shape) does not export; "
                                 "only flat slice gradients play as trapezoids")
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
