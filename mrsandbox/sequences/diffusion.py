####################################################################################################
#                                           diffusion.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: The diffusion weighting a sequence's own gradient events build up, following the dMRS  #
#          consensus (Ligneul et al. 2024, MRM 91:860): the b-matrix from the squared accumulated #
#          gradient moments, crushers and their cross terms included by construction, and the     #
#          powder-averaged decay of randomly oriented cylinders (the sticks at zero radius) under #
#          those kicks, so standard sequences come out slightly diffusion weighted, as on a        #
#          scanner, and a dMRS design decays as the consensus's minimal model says.                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import scipy.special

from mrsandbox.sequences.events import CoherenceFilter, Delay, Gradient, Pulse, ShapedPulse

# the roots of J1' (the cylinder's radial modes) and the directions the powder average sums
# over, a Fibonacci sphere (stated: smooth on the sphere to a part in ten thousand)
ROOTS = scipy.special.jnp_zeros(1, 24)
DIRECTIONS = 1024


#***********#
#   kicks   #
#***********#
def kicks(sequence):
    """The gradient kicks the detected pathway sees, as (the readout's start in s,
    [(t_s, (kx, ky, kz) in cycles per mm)]) with the pathway's order folded into every
    kick.

    Following the consensus formulation (Ligneul et al. 2024, Box 2), every Gradient
    event adds its moment times the coherence order of the detected pathway to the
    running helix of its axis. The pathway order is set by the CoherenceFilters where the
    sequence declares them, defaults to +1 after excitation, and negates at refocusing
    pulses (flip >= 135 degrees) when no filter overrides; kicks played at order zero (a
    STEAM mixing time) add nothing while the stored helix keeps attenuating, exactly as a
    stimulated echo does. A filter standing for a crusher (its crusher_per_mm) kicks on
    the order it retains, along z (stated: the filter names no axis), so a pair around a
    pulse of length tau weights by (2 pi m)^2 tau and rephases. Gradient events are
    instantaneous kicks, the zero-duration limit of the Stejskal-Tanner delta/3 term;
    in-pulse slice gradients are not counted (stated).
    """
    order, now, found = 1, 0.0, []
    for event in sequence.events:
        if isinstance(event, CoherenceFilter):
            order = event.orders[0]
            if event.crusher_per_mm:
                found.append((now, (0.0, 0.0, order * float(event.crusher_per_mm))))
        elif isinstance(event, Gradient):
            kick = [0.0, 0.0, 0.0]
            kick["xyz".index(event.axis)] = order * float(event.moment_per_mm)
            found.append((now, tuple(kick)))
        elif isinstance(event, Delay):
            now += float(event.t_s)
        elif isinstance(event, ShapedPulse):
            now += float(event.duration_s)
            if event.peak_hz is not None or float(event.flip_deg) >= 135.0:
                order = -order
        elif isinstance(event, Pulse):
            if float(event.flip_deg) >= 135.0:
                order = -order
    return now, found


#**************#
#   b matrix   #
#**************#
def b_matrix_ms_um2(sequence):
    """The b-matrix of a sequence in ms per um^2, (3, 3): B_ij = (2 pi)^2 times the
    integral of kappa_i kappa_j over time, kappa the helix the kicks build per axis, so
    the crushers' cross terms between the axes are in it; its trace is `b_value_ms_um2`."""
    end, found = kicks(sequence)
    helix, matrix = np.zeros(3), np.zeros((3, 3))
    for (t_s, kick), until in zip(found, [t for t, _ in found[1:]] + [end]):
        helix = helix + np.asarray(kick)
        matrix = matrix + np.outer(helix, helix) * (until - t_s)
    return matrix * (2.0 * math.pi * 1e3) ** 2 * 1e-9  # 1/m^2 s to ms/um^2


#*************#
#   b value   #
#*************#
def b_value_ms_um2(sequence):
    """The b-value of a sequence in ms per um^2, the trace of its b-matrix: standard
    sequences come out slightly diffusion weighted by their crushers, as on a scanner."""
    return float(np.trace(b_matrix_ms_um2(sequence)))


#*****************#
#   attenuation   #
#*****************#
def attenuation(sequence, d_um2_ms, radius_um=0.0):
    """The powder-averaged signal of randomly oriented infinitely long cylinders under
    the sequence's own kicks: the consensus minimal model (Ligneul et al. 2024, section
    5.1) with the diffusivity D along the fibre and the radius R across it, the sticks at
    R = 0.

    Along a cylinder of direction n the diffusion is free, exp(-D n^T B n) with the
    b-matrix B, so a single-direction b gives exp(-b D cos^2 theta) and three-axis
    crushers nearly exp(-b D / 3). Across it the diffusion is restricted, in the Gaussian
    phase approximation of van Gelderen et al. 1994 (JMR B 103:255) taken to the kicks'
    zero-duration limit: ln E = -(2 pi)^2 sum_m c_m sum_ij (k_i . k_j across n)
    exp(-D a_m^2 |t_i - t_j|), with a_m the roots of J1' over R and c_m = 1 / (a_m^2
    (a_m^2 R^2 - 1)); a kick pair q apart by Delta is van Gelderen's pulsed pair at
    delta -> 0, and at long Delta it is the cylinder's form factor exp(-(2 pi q)^2
    R^2 / 4). The average over the fibre directions, a uniform powder (stated: the
    voxel carries no fibre orientation), is summed over DIRECTIONS points of a
    Fibonacci sphere, which reproduces the sticks' closed form sqrt(pi / (4 b D))
    erf(sqrt(b D)) for a single-direction b. Units: D in um^2/ms, R in um.
    """
    end, found = kicks(sequence)
    if not found:
        return 1.0
    times = np.asarray([t for t, _ in found]) * 1e3                       # ms
    moments = np.asarray([kick for _, kick in found]) * 1e-3               # cycles per um
    matrix = b_matrix_ms_um2(sequence)
    k = np.arange(DIRECTIONS) + 0.5
    polar, azimuth = np.arccos(1.0 - 2.0 * k / DIRECTIONS), math.pi * (1.0 + math.sqrt(5.0)) * k
    n = np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth),
                  np.cos(polar)], axis=1)                                   # (directions, 3)
    along = np.einsum("di,ij,dj->d", n, matrix, n) * d_um2_ms
    if radius_um > 0.0:
        # the kicks' components across every direction, and their pairwise correlation
        # through the cylinder's modes
        across = (moments[None, :, :]
                  - np.einsum("dj,kj->dk", n, moments)[:, :, None] * n[:, None, :])
        pairs = np.einsum("dik,djk->dij", across, across)                  # (directions, k, k)
        gaps = np.abs(times[:, None] - times[None, :])                     # ms
        alpha = ROOTS / radius_um
        weights = 1.0 / (alpha ** 2 * (ROOTS ** 2 - 1.0))
        decay = np.exp(-d_um2_ms * alpha[:, None, None] ** 2 * gaps[None, :, :])
        along = along + (2.0 * math.pi) ** 2 * np.einsum("m,dij,mij->d", weights, pairs, decay)
    return float(np.mean(np.exp(-along)))
