####################################################################################################
#                                       test_suppression.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Suppression gate: composed suppression modules act physically. CHESS in front of PRESS  #
#          kills the water while leaving the metabolites untouched, a deliberate flip error leaves #
#          exactly cos(flip)^n of the water, the selective band excludes near neighbours, and the  #
#          WET and VAPOR series deliver their published T1 and B1 insensitivity.                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest
from keras import ops

from mrsandbox import Chess, Engine, Press, PulseAcquire, SpinSystem, Vapor, Wet
from mrsandbox.database import suppression
from mrsandbox.phantom.water import WaterSource

TE1, TE2 = 0.011, 0.019


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


def total_amplitude(engine, sequence, system):
    _, amps = engine.lines(sequence, system)
    return float(np.abs(ops.convert_to_numpy(amps)).sum())


#**************************************************************************************************#
#                                      Class TestSuppression                                       #
#**************************************************************************************************#
class TestSuppression:

    #*************************************#
    #   water dies, metabolites survive   #
    #*************************************#
    def test_chess_kills_water_and_spares_naa(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        press, chess_press = Press(TE1, TE2), Chess(3) + Press(TE1, TE2)
        assert total_amplitude(engine, chess_press, water) < 1e-6 * total_amplitude(engine, press, water)
        for system in SpinSystem.from_database("NAA"):
            plain = total_amplitude(engine, press, system)
            suppressed = total_amplitude(engine, chess_press, system)
            assert abs(suppressed - plain) < 1e-3 * plain

    #*************************#
    #   analytic flip error   #
    #*************************#
    def test_flip_error_leaves_cos_flip_to_the_n(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        reference = total_amplitude(engine, PulseAcquire(), water)
        for n, flip in [(1, 60.0), (2, 60.0), (3, 80.0)]:
            residual = total_amplitude(engine, Chess(n, flip_deg=flip) + PulseAcquire(), water)
            expected = abs(math.cos(math.radians(flip))) ** n
            assert abs(residual / reference - expected) < 1e-3

    #**********************#
    #   band selectivity   #
    #**********************#
    def test_band_excludes_the_naa_aspartyl_neighbour(self, engine):
        # the NAA aspartyl proton at 4.3817 ppm sits 34 Hz from water at 3 T; a 60 Hz
        # band (half width 30 Hz) must leave it alone, a 100 Hz band must not
        (naa_aspartyl,) = [s for s in SpinSystem.from_database("NAA") if "nAspartyl" in s.name]
        plain = total_amplitude(engine, PulseAcquire(), naa_aspartyl)
        narrow = total_amplitude(engine, Chess(3, bandwidth_hz=60.0) + PulseAcquire(), naa_aspartyl)
        wide = total_amplitude(engine, Chess(3, bandwidth_hz=100.0) + PulseAcquire(), naa_aspartyl)
        assert abs(narrow - plain) < 1e-3 * plain
        assert wide < 0.9 * plain

    #*****************#
    #   composition   #
    #*****************#
    def test_composition_concatenates_events(self):
        combined = Chess(3) + Press(TE1, TE2)
        assert len(combined.events) == len(Chess(3).events) + len(Press(TE1, TE2).events)


#**************************************************************************************************#
#                                     Class TestNamedSchemes                                       #
#**************************************************************************************************#
#                                                                                                  #
# WET and VAPOR carry their published series and deliver the published insensitivity.              #
#                                                                                                  #
#**************************************************************************************************#
class TestNamedSchemes:

    #*********************#
    #   typed from print  #
    #*********************#
    def test_series_match_the_published_tables(self):
        assert Wet().flips_deg == suppression.WET["flips_deg"] == (81.4, 101.4, 69.3, 161.0)
        assert Vapor().delays_s == suppression.VAPOR["delays_s"]["4t"]
        assert Vapor(field="7t").delays_s == suppression.VAPOR["delays_s"]["7t"]
        # a +5 dB power offset scales the flip of a fixed-shape pulse by 10 ** (5 / 20)
        assert abs(Vapor().flips_deg[2] - 90.0 * 10.0 ** 0.25) < 1e-9

    #**********************#
    #   wet nulls water    #
    #**********************#
    def test_wet_nulls_water_across_t1(self, engine):
        for t1 in (1.0, 1.5, 2.0):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(), water)
            residual = total_amplitude(engine, Wet() + PulseAcquire(), water)
            assert residual < 0.005 * full

    #***************************#
    #   vapor b1 insensitive    #
    #***************************#
    def test_vapor_is_b1_and_t1_insensitive(self, engine):
        # the published claim: water stays nulled for nominal flips from 65 to 125
        # degrees and water T1 from 1 to 2 seconds (measured worst case 1.9 percent)
        for t1 in (1.0, 2.0):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(), water)
            for flip in (65.0, 90.0, 125.0):
                residual = total_amplitude(engine, Vapor(nominal_flip_deg=flip) + PulseAcquire(), water)
                assert residual < 0.025 * full

    #**************************#
    #   metabolites survive    #
    #**************************#
    def test_vapor_spares_the_metabolites(self, engine):
        for system in SpinSystem.from_database("NAA"):
            system = system.with_relaxation(t1_s=1.4, t2_s=0.25)
            plain = total_amplitude(engine, PulseAcquire(), system)
            suppressed = total_amplitude(engine, Vapor() + PulseAcquire(), system)
            assert abs(suppressed - plain) < 1e-3 * plain
