####################################################################################################
#                                             eddy.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The field a design's gradients induce through the scanner's impulse response, and the   #
#          kernel it puts on a voxel's readout: a global phase from the B0 term, and from the      #
#          linear term a phase at the voxel position plus the dephasing across the box. The        #
#          response here is the parametric exponential one; a measured gradient impulse response   #
#          (Vannesjo et al. 2013) replaces it term by term.                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import scipy.signal

from mrsandbox.sequences.timeline import gradient_waveforms

TIME_STEP_S = 1e-5


#*************************#
#   exponential response  #
#*************************#
def induced_fields(waveform_mt_m, dt_s, taus_s, b0_hz_per_mt_m, linear_fractions):
    """The B0 offset (Hz) and the gradient error (mT/m) one axis's waveform induces
    through a sum of exponential eddy-current terms: every step of the gradient
    excites, per term k, a field decaying as exp(-t / tau_k), of b0_k hertz per
    mT/m of step in the uniform term and of the fraction linear_k of the step,
    opposing it, in the term along the axis itself (the classic multi-exponential
    eddy-current model, de Graaf 2019 section 10.5; the response of an ideal
    pre-emphasis leaves only such residuals). Cross-axis terms are not modelled
    (stated)."""
    steps = np.diff(waveform_mt_m, prepend=0.0)
    b0, error = np.zeros_like(waveform_mt_m), np.zeros_like(waveform_mt_m)
    for tau, b0_k, linear_k in zip(taus_s, b0_hz_per_mt_m, linear_fractions):
        decayed = scipy.signal.lfilter([1.0], [1.0, -np.exp(-dt_s / tau)], steps)
        b0 = b0 + b0_k * decayed
        error = error - linear_k * decayed
    return b0, error


#*******************#
#   readout kernel  #
#*******************#
def readout_kernel(sequence, scanner, position_mm, size_mm, npoints, dwell_s, taus_s,
                   b0_hz_per_mt_m, linear_fractions, gamma_mhz_t=42.5774785):
    """The complex kernel the induced fields put on a voxel's FID, sampled at the
    readout's points: exp(i 2 pi phi(t)) with phi the integral from the readout's
    start of the B0 terms plus the linear terms times the voxel position (mm from
    the isocentre, the phantom's origin), times per axis the box mean of the
    linear term's phase across the voxel, sinc(moment x extent) (a box of isochromats
    dephasing under a residual gradient moment in cycles per mm)."""
    start = float(sequence.duration_s)
    t, waveforms = gradient_waveforms(sequence, scanner, TIME_STEP_S, start + npoints * dwell_s,
                                      gamma_mhz_t)
    readout = start + np.arange(npoints) * dwell_s
    size = np.asarray(size_mm, dtype=np.float64) * np.ones(3)
    phase = np.zeros(npoints)
    kernel = np.ones(npoints)
    for axis, waveform in waveforms.items():
        b0, error = induced_fields(waveform, TIME_STEP_S, taus_s, b0_hz_per_mt_m, linear_fractions)
        k = "xyz".index(axis)
        # cycles per mm from the residual gradient, cycles from the B0 term, both
        # accumulated from the readout's start
        moment = np.interp(readout, t, np.cumsum(error) * TIME_STEP_S) * gamma_mhz_t
        moment = moment - moment[0]
        cycles = np.interp(readout, t, np.cumsum(b0) * TIME_STEP_S)
        cycles = cycles - cycles[0]
        phase = phase + cycles + moment * float(position_mm[k])
        kernel = kernel * np.sinc(moment * size[k])
    return (kernel * np.exp(2j * np.pi * phase)).astype(np.complex64)
