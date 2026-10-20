####################################################################################################
#                                         test_sequences.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Sequence gate: two classic analytic behaviours only a real engine reproduces - the      #
#          lactate doublet's J-modulation inverts under PRESS at TE = 1/J, and STEAM's stimulated  #
#          echo carries exactly half the signal - plus the fitting-challenge preset's stated       #
#          numbers.                                                                                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import Engine, Press, PulseAcquire, SpinSystem, Steam


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def sim():
    return Engine(3.0)


def ch3_amplitude(sim, sequence):
    """Summed in-phase Lac CH3 doublet amplitude at 3 T (real under the receiver convention)."""
    (lac,) = SpinSystem.from_database("Lac")
    freqs, amps = (ops.convert_to_numpy(x) for x in sim.lines(sequence, lac))
    f0 = lac.gamma_mhz_t * 3.0
    window = (np.abs(freqs) / f0 > 1.3142 - 0.05) & (np.abs(freqs) / f0 < 1.3142 + 0.05)
    return np.real(amps[window]).sum()


#**************************************************************************************************#
#                                          Class TestPress                                         #
#**************************************************************************************************#
class TestPress:

    #******************#
    #   j modulation   #
    #******************#
    def test_lac_doublet_inverts_at_te_of_one_over_j(self, sim):
        te = 1.0 / 6.933
        short = ch3_amplitude(sim, Press(te1_s=0.0005, te2_s=0.0005))
        inverted = ch3_amplitude(sim, Press(te1_s=te / 2.0, te2_s=te / 2.0))
        assert short * inverted < 0.0, "doublet must invert at TE = 1/J"
        assert abs(abs(inverted / short) - 1.0) < 0.05  # cos(pi J TE) = -1: full inversion


#**************************************************************************************************#
#                                          Class TestSteam                                         #
#**************************************************************************************************#
class TestSteam:

    #********************#
    #   half amplitude   #
    #********************#
    def test_stimulated_echo_is_half_the_signal(self, sim):
        (scyllo,) = SpinSystem.from_database("sIns")  # a singlet, so the ratio is exact
        reference = ops.convert_to_numpy(sim.lines(PulseAcquire(), scyllo)[1])
        steam = ops.convert_to_numpy(sim.lines(Steam(te_s=0.02, tm_s=0.01), scyllo)[1])
        ratio = np.abs(steam).sum() / np.abs(reference).sum()
        assert abs(ratio - 0.5) < 1e-3

    #***************#
    #   refocused   #
    #***************#
    def test_the_stimulated_echo_refocuses_the_shifts(self, sim):
        # the kept pathway is +1, 0, -1: the echo is in phase with a plain excitation
        # at any echo time (the pathway -1, 0, -1 has the same half amplitude and a
        # phase that turns with shift times TE)
        (scyllo,) = SpinSystem.from_database("sIns")
        reference = ops.convert_to_numpy(sim.lines(PulseAcquire(), scyllo)[1])
        k = np.argmax(np.abs(reference))
        for te_s in (0.0200, 0.0203, 0.0217):
            steam = ops.convert_to_numpy(sim.lines(Steam(te_s=te_s, tm_s=0.01), scyllo)[1])
            assert abs(np.angle(steam[k] / reference[k])) < 1e-3
