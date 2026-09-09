####################################################################################################
#                                           diffusion.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: The b-value a sequence's own gradient events build up, following the dMRS consensus     #
#          (Ligneul et al. 2024, MRM 91:860, Box 2): b integrates the squared accumulated gradient #
#          moment over the sequence, so crusher contributions and cross terms are included by      #
#          construction, and standard sequences come out slightly diffusion weighted, as on a      #
#          scanner.                                                                                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

from mrsandbox.sequences.events import CoherenceFilter, Delay, Gradient, Pulse, ShapedPulse


#*************#
#   b value   #
#*************#
def b_value_ms_um2(sequence):
    """The b-value of a sequence in ms per um^2, from its own gradient moments.

    Following the consensus formulation (Ligneul et al. 2024, Box 2), b integrates the
    squared accumulated moment over time: every Gradient event adds its moment times
    the coherence order of the detected pathway to the running helix of its axis, and
    every interval of duration t contributes (2 pi kappa)^2 t per axis. The pathway
    order is set by the CoherenceFilters where the sequence declares them, defaults to
    +1 after excitation, and negates at refocusing pulses (flip >= 135 degrees) when
    no filter overrides; kicks played at order zero (a STEAM mixing time) add nothing
    while the stored helix keeps attenuating, exactly as a stimulated echo does.
    Gradient events are instantaneous kicks, the zero-duration limit of the
    Stejskal-Tanner delta/3 term; in-pulse slice gradients are not counted (stated).
    """
    order = 1
    helix = {}
    b_si = 0.0

    def dwell(duration_s):
        return sum((2.0 * math.pi * kappa * 1e3) ** 2 for kappa in helix.values()) * duration_s

    for event in sequence.events:
        if isinstance(event, CoherenceFilter):
            order = event.orders[0]
        elif isinstance(event, Gradient):
            helix[event.axis] = helix.get(event.axis, 0.0) + order * float(event.moment_per_mm)
        elif isinstance(event, Delay):
            b_si += dwell(float(event.t_s))
        elif isinstance(event, ShapedPulse):
            b_si += dwell(float(event.duration_s))
            if event.peak_hz is not None or float(event.flip_deg) >= 135.0:
                order = -order
        elif isinstance(event, Pulse):
            if float(event.flip_deg) >= 135.0:
                order = -order
    return b_si * 1e-9  # 1/m^2 s to ms/um^2
