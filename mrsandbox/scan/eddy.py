####################################################################################################
#                                             eddy.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The field a design's gradients induce through the scanner's impulse response, and what  #
#          it does to a voxel: before the readout the kicks it puts on the design's coherences     #
#          (a uniform phase from the B0 term and the first-order terms at the voxel, moments       #
#          unbalancing the crushers across the box), during it the kernel on the FID (the same     #
#          terms integrated from the readout's start). The response is the parametric exponential  #
#          one of exponential_response, or a measured gradient impulse response (resources.GIRF,   #
#          Vannesjo et al. 2013) term by term.                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import copy

import numpy as np
import scipy.signal

from mrsandbox.sequences.events import InducedKick, ShapedPulse
from mrsandbox.sequences.timeline import event_spans, gradient_waveforms

TIME_STEP_S = 1e-5
# the band the fields before the readout are taken in: above it a measured response's
# integrals over the design pick up the measurement's noise, and 20 kHz is about what
# its authors' own filter passes (stated)
INDUCED_BANDWIDTH_HZ = 20000.0


#*************************#
#   exponential response  #
#*************************#
def induced_fields(waveform_mt_m, dt_s, taus_s, b0_hz_per_mt_m, linear_fractions, tr_s=None):
    """The B0 offset (Hz) and the gradient error (mT/m) one axis's waveform induces
    through a sum of exponential eddy-current terms: every step of the gradient
    excites, per term k, a field decaying as exp(-t / tau_k), of b0_k hertz per
    mT/m of step in the uniform term and of the fraction linear_k of the step,
    opposing it, in the term along the axis itself (the classic multi-exponential
    eddy-current model, de Graaf 2019 section 10.5; the response of an ideal
    pre-emphasis leaves only such residuals). Cross-axis terms are not modelled
    (stated).

    With a repetition time the waveform repeats every `tr_s` and the fields are the
    steady state of the train, not the first repetition's: what one repetition
    leaves of a term at the next one's start, D, arrives there from every earlier
    repetition too, D / (1 - exp(-tr_s / tau)) in all, and decays through this one.
    A term of seconds under a TR of two keeps most of itself that way."""
    steps = np.diff(waveform_mt_m, prepend=0.0)
    b0, error = np.zeros_like(waveform_mt_m), np.zeros_like(waveform_mt_m)
    t = np.arange(steps.size) * dt_s
    for tau, b0_k, linear_k in zip(taus_s, b0_hz_per_mt_m, linear_fractions):
        decayed = scipy.signal.lfilter([1.0], [1.0, -np.exp(-dt_s / tau)], steps)
        if tr_s is not None:
            left = decayed[-1] * np.exp(-(tr_s - t[-1]) / tau)
            decayed = decayed + left / (1.0 - np.exp(-tr_s / tau)) * np.exp(-t / tau)
        b0 = b0 + b0_k * decayed
        error = error - linear_k * decayed
    return b0, error


def exponential_response(taus_s, b0_hz_per_mt_m, linear_fractions, tr_s=None):
    """The exponential terms as the response `readout_kernel` takes: called as
    (axis, waveform_mt_m, dt_s, bandwidth_hz) it returns {"b0": Hz, axis: mT/m},
    the two fields of `induced_fields` (no cross terms; the receiver's band plays no
    part, the residuals being smooth), in the steady state of a train repeating
    every `tr_s` when one is given."""
    def response(axis, waveform_mt_m, dt_s, bandwidth_hz):
        b0, error = induced_fields(waveform_mt_m, dt_s, taus_s, b0_hz_per_mt_m, linear_fractions,
                                   tr_s)
        return {"b0": b0, axis: error}
    return response


#*******************#
#   induced phase   #
#*******************#
def induced_phase(sequence, scanner, response, until_s, gamma_mhz_t=42.5774785,
                  bandwidth_hz=INDUCED_BANDWIDTH_HZ):
    """The phase the fields a design's gradients induce build up on the timeline
    (TIME_STEP_S) out to `until_s`: (t, cycles, moments) with `cycles` the B0 term's
    integral in cycles and `moments` {axis: cycles per mm} the first-order terms'
    integrals, every input axis's fields added per term.

    `response(axis, waveform_mt_m, dt_s, bandwidth_hz)` gives the fields one axis's
    waveform induces as {"b0": Hz, "x": mT/m, "y": mT/m, "z": mT/m} (absent terms
    zero): the exponential form of `exponential_response`, or a measured gradient
    impulse response (resources.GIRF) with its cross terms, band-limited to
    `bandwidth_hz` (INDUCED_BANDWIDTH_HZ unless a readout's receiver band is given)."""
    t, waveforms = gradient_waveforms(sequence, scanner, TIME_STEP_S, until_s, gamma_mhz_t)
    b0 = np.zeros(t.size)
    errors = {axis: np.zeros(t.size) for axis in "xyz"}
    for axis, waveform in waveforms.items():
        fields = response(axis, waveform, TIME_STEP_S, bandwidth_hz)
        b0 = b0 + fields.get("b0", 0.0)
        for k in errors:
            errors[k] = errors[k] + fields.get(k, 0.0)
    cycles = np.cumsum(b0) * TIME_STEP_S
    moments = {axis: np.cumsum(error) * TIME_STEP_S * gamma_mhz_t for axis, error in errors.items()}
    return t, cycles, moments


def induced_sequence(sequence, scanner, response, position_mm, gamma_mhz_t=42.5774785):
    """The design with the phase its own gradients induce before the readout, at a
    voxel `position_mm` from the isocentre: the fields' integrals, the B0 term plus
    the first-order terms at the voxel as uniform cycles and the first-order terms as
    moments across the box, as an InducedKick after every delay and as the `induced`
    track of every shaped pulse, whose spins run off resonance by the field while it
    plays. With a TR an exponential response carries the fields of the earlier
    repetitions (`induced_fields`); a measured one remembers 200 ms and needs none.
    The copy keeps the design's own attributes."""
    spans = event_spans(sequence)
    t, cycles, moments = induced_phase(sequence, scanner, response, spans[-1][1], gamma_mhz_t)
    if not any(np.any(m) for m in moments.values()) and not np.any(cycles):
        return sequence
    def since(start, at):
        # [uniform cycles, moment x, y, z] built up from `start` to the times `at`
        rows = [np.interp(at, t, moments[axis]) - np.interp(start, t, moments[axis])
                for axis in "xyz"]
        uniform = np.interp(at, t, cycles) - np.interp(start, t, cycles)
        return np.stack([uniform + sum(row * float(p) for row, p in zip(rows, position_mm))] + rows)

    events = []
    for event, (start, end) in zip(sequence.events, spans):
        if isinstance(event, ShapedPulse):
            steps = len(event.envelope)
            at = start + (np.arange(steps) + 0.5) * (end - start) / steps
            event = copy.copy(event)
            event.induced = since(start, np.append(at, end)).astype(np.float32)
        events.append(event)
        if end > start and not isinstance(event, ShapedPulse):
            events.append(InducedKick(*(float(v) for v in since(start, end))))
    design = copy.copy(sequence)
    design.events = tuple(events)
    return design


#*******************#
#   readout kernel  #
#*******************#
def readout_kernel(sequence, scanner, position_mm, size_mm, npoints, dwell_s, response,
                   gamma_mhz_t=42.5774785, start_s=0.0):
    """The complex kernel the induced fields put on a voxel's FID, sampled at the
    readout's points: exp(i 2 pi phi(t)) with phi the integral from the readout's
    start of the B0 term plus the first-order terms times the voxel position (mm from
    the isocentre, the phantom's origin), times per axis the box mean of that axis's
    phase across the voxel, sinc(moment x extent) (a box of isochromats dephasing
    under a residual gradient moment in cycles per mm; the moment carried into the
    readout dephases in the design's own ensemble, `induced_sequence`, so the two
    sincs multiply where one of their sum would be exact, stated). `response` as in
    `induced_phase`, a measured one band-limited to the readout's receiver band; the
    points sit `start_s` after the readout's start (the receiver delay). None for a
    design that plays no gradient (an ideal localization): nothing is induced, and a
    kernel of ones would only cost interpolation."""
    start = float(sequence.duration_s)
    if not gradient_waveforms(sequence, scanner, TIME_STEP_S, start, gamma_mhz_t)[1]:
        return None
    t, cycles, moments = induced_phase(sequence, scanner, response,
                                       start + start_s + npoints * dwell_s,
                                       gamma_mhz_t, bandwidth_hz=1.0 / dwell_s)
    readout = start + start_s + np.arange(npoints) * dwell_s
    size = np.asarray(size_mm, dtype=np.float64) * np.ones(3)

    def from_start(cumulative):
        # the phase over the readout's points, from the readout's start
        return np.interp(readout, t, cumulative) - np.interp(start, t, cumulative)

    phase = from_start(cycles)  # cycles from the B0 term
    kernel = np.ones(npoints)
    for k, axis in enumerate("xyz"):
        moment = from_start(moments[axis])  # cycles per mm from the residual gradient
        phase = phase + moment * float(position_mm[k])
        kernel = kernel * np.sinc(moment * size[k])
    return (kernel * np.exp(2j * np.pi * phase)).astype(np.complex64)
