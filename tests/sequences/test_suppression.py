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

from mrsandbox import (Chess, CoherenceFilter, Cows, Engine, Press, PulseAcquire, Sequence,
                       ShapedPulse, SiemensChess, SpinSystem, Vapor, Wet)
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
        assert (total_amplitude(engine, chess_press, water)
                < 1e-6 * total_amplitude(engine, press, water))
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

    #*********************#
    #   siemens product   #
    #*********************#
    def test_siemens_train_nulls_water_across_t1_until_its_adjustment_is_off(self, engine):
        # the header's ratios and band; the overtipping third pulse crosses zero at the
        # excitation whatever the T1, and a scale the adjustment did not take out (10 percent
        # low) leaves water
        train = SiemensChess()
        assert train.flips_deg == tuple(
            90.0 * r for r in suppression.SIEMENS_SVS_SE["amplitude_ratios"])
        assert train.bandwidth_hz == 50.0 and train.delays_s == (0.030,) * 3
        assert train.scaled(0.9).flips_deg == pytest.approx(tuple(0.9 * f for f in train.flips_deg))
        assert train.scaled(0.9).delays_s == train.delays_s
        for t1 in (0.9, 1.3, 4.0):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(), water)
            assert total_amplitude(engine, train + PulseAcquire(), water) < 0.005 * full
            assert total_amplitude(engine, train.scaled(0.9) + PulseAcquire(), water) > 0.01 * full

    #***************************#
    #   vapor b1 insensitive    #
    #***************************#
    def test_vapor_is_b1_and_t1_insensitive(self, engine):
        # the published claim: water stays nulled for nominal flips from 65 to 125
        # degrees and water T1 from 1 to 2 seconds
        for t1 in (1.0, 2.0):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(), water)
            for flip in (65.0, 90.0, 125.0):
                residual = total_amplitude(engine, Vapor(nominal_flip_deg=flip) + PulseAcquire(),
                                           water)
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

    #*********************************#
    #   cows tables and the headers   #
    #*********************************#
    def test_cows_series_match_the_tables_and_the_dataset_headers(self):
        # Table 1 of Igwe et al. 2025: each scheme's delays sum to the duration in its name, and the
        # flips are what the dataset's headers played (OpenNeuro ds006812, sub-01, the amplitudes of
        # ss_rf_ws1 to ss_rf_ws12 in volts, 11.7936 V the 90 degree pulse; the study calibrated no
        # flip per voxel)
        volts = {"VAPOR7": (11.7936, 11.7936, 20.9926, 11.7936, 20.9926, 11.7936, 20.9926),
                 "COWS(7;236)": (9.8712, 11.2865, 11.3219, 9.4939, 10.7912, 15.2845, 21.0516),
                 "COWS(12;626)": (9.9279, 0.4411, 14.4873, 18.4853, 23.5872, 2.6288, 11.7464,
                                  9.9951, 15.3506, 10.4326, 16.7375, 21.9656)}
        for scheme, duration_s in (("COWS(7;236)", 0.236), ("COWS(12;626)", 0.626),
                                   ("VAPOR7", 0.626)):
            train = Cows(scheme)
            assert train.flips_deg == suppression.COWS[scheme]["flips_deg"]
            assert sum(train.delays_s) == pytest.approx(duration_s, abs=0.0005)
            assert train.bandwidth_hz == 68.0
            assert train.flips_deg == pytest.approx(
                tuple(90.0 * v / 11.7936 for v in volts[scheme]), abs=0.1)

    #**********************************#
    #   cows nulls across t1 and b1    #
    #**********************************#
    def test_cows_nulls_water_across_t1_and_b1_in_the_steady_state(self, engine):
        # the paper's design range, T1 0.5 to 5 s and a B1 factor 0.75 to 1.25, in the spoiled
        # steady state of its TR 2 s (COWS assumes no equilibrium): both COWS schemes leave under
        # 0.3 percent of the unsuppressed water at the excitation everywhere in it (ideal 68 Hz
        # boxes on resonance), VAPOR7 under 0.7 within 10 percent of B1
        for t1 in (0.5, 1.3, 4.5):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(tr_s=2.0), water)
            for scheme, b1s, bound in (("COWS(7;236)", (0.75, 1.0, 1.25), 0.003),
                                       ("COWS(12;626)", (0.75, 1.0, 1.25), 0.003),
                                       ("VAPOR7", (0.9, 1.0, 1.1), 0.007)):
                for b1 in b1s:
                    assert total_amplitude(
                        engine, Cows(scheme).scaled(b1) + PulseAcquire(tr_s=2.0), water
                    ) < bound * full

    #*************************#
    #   played trains        #
    #*************************#
    def test_a_played_train_carries_its_sources_pulse_and_still_nulls(self, engine):
        # played real, COWS's pulses are the paper's 17.92 ms Gaussians (their saturation, 1 - Mz,
        # at half at 34 Hz from the water: the paper's 68 Hz), the train's gaps follow the
        # peak-to-peak table (the first pulse starts half a pulse before its peak), VAPOR's are the
        # 35 ms three-lobe sincs of Tkac and Gruetter 2000, and the played COWS(7;236) still leaves
        # under 0.1 percent of the water at the nominal flips over T1
        from mrsandbox.sequences.timeline import event_spans

        train = Cows(real=True)
        pulse = train.events[0]
        assert isinstance(pulse, ShapedPulse) and pulse.duration_s == 0.01792 and not train.adjusted
        assert float(event_spans(train)[-1][1]) == pytest.approx(0.236157 + 0.01792 / 2.0, abs=1e-4)
        assert Vapor(real=True).events[0].duration_s == 0.035 and not Vapor(real=True).adjusted
        assert (SiemensChess(real=True).events[0].duration_s == pytest.approx(0.01792 * 68.0 / 50.0)
                and SiemensChess(real=True).adjusted)
        water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=1.3, t2_s=0.05)
        full = total_amplitude(engine, PulseAcquire(), water)
        ninety = ShapedPulse.gaussian(90.0, pulse.duration_s, offset_ppm=4.65,
                                      truncation=train.truncation)
        for offset_hz, saturation in ((0.0, 1.0), (34.0, 0.5), (100.0, 0.0)):
            _, amps = engine.lines(Sequence(ninety, CoherenceFilter(-1)), water,
                                   offset_hz=offset_hz)
            response = float(np.abs(ops.convert_to_numpy(amps)).sum()) / full
            assert (1.0 - math.cos(math.asin(min(response, 1.0)))
                    == pytest.approx(saturation, abs=0.1))
        for t1 in (0.5, 1.3, 4.5):
            water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=t1, t2_s=0.05)
            full = total_amplitude(engine, PulseAcquire(tr_s=2.0), water)
            assert total_amplitude(engine, train + PulseAcquire(tr_s=2.0), water) < 0.001 * full

