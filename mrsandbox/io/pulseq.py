####################################################################################################
#                                            pulseq.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/09/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq import: read a .seq file (via pypulseq) and translate its blocks into the        #
#          event primitives, so any single-voxel Pulseq sequence runs through the engine with      #
#          the ADC marking the readout.                                                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np

from mrsandbox.sequences.events import Delay, Gradient, Sequence, ShapedPulse


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
    flat = [g for g in traps.values() if g.amplitude]
    gradient = float(flat[0].amplitude) / 1000.0 if flat else 0.0
    return ShapedPulse(
        flip_deg=None,
        duration_s=duration,
        envelope=(amplitude / peak).astype(np.float32),
        offset_ppm=center_ppm + float(getattr(rf, "freq_ppm", 0.0)),
        phase_deg=phase0_deg,
        frequency_hz=sweep.astype(np.float32) if np.any(sweep) else None,
        peak_hz=peak,
        gradient_hz_per_mm=gradient,
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

    Stated v1 idealizations: gradients act along one axis (the largest-area axis of
    each block), trapezoids only; ramp moments outside an RF pulse are not applied;
    blocks after the first ADC are ignored (one readout per file).

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
            axis = max(traps.values(), key=lambda grad: abs(float(grad.area)))
            events.append(Delay(duration / 2.0))
            events.append(Gradient(moment_per_mm=float(axis.area) / 1000.0, extent_mm=extent_mm))
            events.append(Delay(duration / 2.0))
        elif duration > 0.0:
            events.append(Delay(duration))
    raise ValueError("the .seq file has no ADC block: nothing marks the readout")
