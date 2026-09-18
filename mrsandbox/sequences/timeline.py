####################################################################################################
#                                           timeline.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The gradient waveforms a design plays on a scanner, on a time grid: the kicks as the    #
#          shortest trapezoids the scanner's limits allow, the slice gradients as their plateaus   #
#          with ramps, so anything the gradients induce (eddy currents, the field a gradient       #
#          impulse response predicts) can be computed from the design itself.                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox.sequences.events import Delay, Gradient, ShapedPulse, trapezoid


#*******************#
#   event spans     #
#*******************#
def event_spans(sequence):
    """When each event of a design plays, [(start_s, end_s)] in event order: delays
    and shaped pulses take their duration, everything else is instantaneous."""
    spans, now = [], 0.0
    for event in sequence.events:
        took = 0.0
        if isinstance(event, Delay):
            took = float(event.t_s)
        elif isinstance(event, ShapedPulse):
            took = float(event.duration_s)
        spans.append((now, now + took))
        now += took
    return spans


#*************************#
#   gradient waveforms    #
#*************************#
def gradient_waveforms(sequence, scanner, dt_s, until_s, gamma_mhz_t=42.5774785):
    """The gradient waveforms of a design on `scanner`, sampled every `dt_s` out to
    `until_s`, as {axis: array in mT/m} over the axes the design uses.

    Placement, all stated: every Gradient kick plays as the shortest trapezoid of its
    moment under the scanner's gradient and slew limits, in the delay next to it
    (starting where the kick sits when a delay follows, ending there when a pulse
    follows, so a crusher pair straddles its refocusing pulse); a ShapedPulse's slice
    gradient is flat for the pulse (or follows its gradient_shape) with its ramps
    just before and after (borrowed from the surrounding delays); ideal pulses and coherence filters take no time
    and play no gradient; the readout, from `sequence.duration_s` on, is
    ungradiented. Overlapping trapezoids add.
    """
    max_grad, max_slew = scanner.max_grad_mt_m * 1e-3, scanner.max_slew_t_m_s
    n = int(np.ceil(until_s / dt_s)) + 1
    t = np.arange(n) * dt_s
    waveforms = {}

    def play(axis, start_s, amplitude_t_m, ramp_s, flat_s):
        # the trapezoid of amplitude, ramp, and flat top starting at start_s
        shape = np.interp(t, [start_s, start_s + ramp_s, start_s + ramp_s + flat_s,
                              start_s + 2.0 * ramp_s + flat_s],
                          [0.0, 1.0, 1.0, 0.0], left=0.0, right=0.0)
        waveforms[axis] = waveforms.get(axis, np.zeros(n)) + amplitude_t_m * 1e3 * shape

    events = sequence.events
    for k, (event, (now, _)) in enumerate(zip(events, event_spans(sequence))):
        if isinstance(event, Gradient):
            area = float(event.moment_per_mm) * 1e3 / (gamma_mhz_t * 1e6)  # T s / m
            amplitude, ramp, flat = trapezoid(area, max_grad, max_slew)
            following = next((e for e in events[k + 1:] if not isinstance(e, Gradient)), None)
            start = now if isinstance(following, Delay) else now - (2.0 * ramp + flat)
            play(event.axis, start, amplitude, ramp, flat)
        elif isinstance(event, ShapedPulse):
            if event.gradient_hz_per_mm:
                amplitude = float(event.gradient_hz_per_mm) / (gamma_mhz_t * 1e3)  # T/m
                ramp = abs(amplitude) / max_slew
                play(event.gradient_axis, now - ramp, amplitude, ramp, float(event.duration_s))
                if event.gradient_shape is not None:
                    # the modulation rides on the flat top
                    shape = np.asarray(event.gradient_shape, dtype=np.float64)
                    inside = (t >= now) & (t <= now + float(event.duration_s))
                    at = (np.arange(shape.size) + 0.5) * float(event.duration_s) / shape.size
                    waveforms[event.gradient_axis][inside] += amplitude * 1e3 * (
                        np.interp(t[inside] - now, at, shape) - 1.0)
    return t, waveforms
