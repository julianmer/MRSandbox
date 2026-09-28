####################################################################################################
#                                         test_scanners.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The scanner limits as the constraints a design must satisfy: peak B1, gradient and     #
#          slew, and the delays absorbing the hardware durations of the ideal events.              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import pytest

from mrsandbox import Press, Scanner, SemiLaser
from mrsandbox.sequences.events import (CoherenceFilter, Delay, Gradient, Pulse, SelectivePulse,
                                        Sequence, ShapedPulse)

GAMMA = 42.5774785


#**************************************************************************************************#
#                                        Class TestLimits                                          #
#**************************************************************************************************#
class TestLimits:

    def test_presets_carry_the_stated_limits(self):
        scanner = Scanner.preset("generic_3t")
        assert ((scanner.peak_b1_ut, scanner.max_grad_mt_m, scanner.max_slew_t_m_s)
                == (25.0, 40.0, 200.0))
        assert Scanner.preset("generic_7t").peak_b1_ut == 30.0
        assert Scanner("custom", 3.0).max_grad_mt_m == 40.0

    def test_hard_pulses_take_the_time_the_peak_b1_allows(self):
        scanner = Scanner.preset("generic_3t")
        shortest = 0.25 / (GAMMA * scanner.peak_b1_ut)  # a 90 at the peak B1, in seconds
        assert Sequence(Pulse(90.0), Delay(shortest * 1.001), CoherenceFilter(-1)).feasible(scanner)
        reasons = Sequence(Pulse(90.0), Delay(shortest * 0.5)).violations(scanner)
        assert len(reasons) == 1 and "too short" in reasons[0]
        assert f"{shortest * 0.5 * 1e3:.3f} ms" in reasons[0]
        # the readout cannot start inside the last event
        assert not Sequence(Delay(0.010), Pulse(90.0)).feasible(scanner)
        # a coherence filter is an idealization and costs nothing
        assert Sequence(CoherenceFilter(-1)).feasible(scanner)

    def test_named_designs_report_their_minimum_echo_time(self):
        scanner = Scanner.preset("generic_3t")
        assert Press(te1_s=0.011, te2_s=0.019).feasible(scanner)
        assert not Press(te1_s=0.0002, te2_s=0.0002).feasible(scanner)
        # semi-LASER: the adiabatic pulses take their own time, the delays absorb the 90
        # and the library default (1000 Hz, 23.5 uT) fits the 3 T body coil, though not
        # the 15 uT of the Philips preset
        assert SemiLaser(te_s=0.030).feasible(scanner)
        assert SemiLaser(te_s=0.030, afp_peak_hz=1500.0).violations(scanner)[0].startswith(
            "shaped pulse peak 35.2 uT")
        assert SemiLaser(te_s=0.030).violations(Scanner.preset("philips_elition_3t"))[0].startswith(
            "shaped pulse peak 23.5 uT")

    def test_selective_bands_take_the_gaussian_length(self):
        scanner = Scanner.preset("generic_3t")
        band = SelectivePulse(180.0, 1.9, 60.0)
        assert Sequence(band, Delay(1.53 / 60.0 * 1.001)).feasible(scanner)
        assert not Sequence(band, Delay(1.53 / 60.0 * 0.999)).feasible(scanner)
        assert Sequence(SelectivePulse(90.0, 4.65, 60.0),
                        Delay(2.70 / 60.0 * 1.001)).feasible(scanner)

    def test_gradients_take_the_shortest_trapezoid(self):
        scanner = Scanner.preset("generic_3t")
        # a crusher of 30 mT/m for 2 ms: its area exceeds the triangle limit, so the
        # shortest trapezoid is the ramps plus a flat top at the maximum gradient
        crusher = Gradient.crusher(30.0, 0.002)
        area = 30e-3 * 2e-3
        ramps = 2.0 * scanner.max_grad_mt_m * 1e-3 / scanner.max_slew_t_m_s
        flat = ((area - (scanner.max_grad_mt_m * 1e-3) ** 2 / scanner.max_slew_t_m_s)
                / (scanner.max_grad_mt_m * 1e-3))
        assert Sequence(crusher, Delay((ramps + flat) * 1.001)).feasible(scanner)
        assert not Sequence(crusher, Delay((ramps + flat) * 0.999)).feasible(scanner)
        # a small kick fits under a triangle
        small = Gradient(moment_per_mm=0.1)
        triangle = 2.0 * math.sqrt(0.1e3 / (GAMMA * 1e6) / scanner.max_slew_t_m_s)
        assert Sequence(small, Delay(triangle * 1.001)).feasible(scanner)
        assert not Sequence(small, Delay(triangle * 0.999)).feasible(scanner)
        # a slice gradient over the limit is refused, under it costs its two ramps
        pulse = ShapedPulse.sinc(90.0, 0.002, gradient_hz_per_mm=GAMMA * 50.0)
        assert "slice gradient 50.0 mT/m" in Sequence(pulse, Delay(0.010)).violations(scanner)[0]
        pulse = ShapedPulse.sinc(90.0, 0.002, gradient_hz_per_mm=GAMMA * 20.0)
        rise = 20e-3 / scanner.max_slew_t_m_s
        assert Sequence(pulse, Delay(2.0 * rise * 1.001)).feasible(scanner)
        assert not Sequence(pulse, Delay(2.0 * rise * 0.999)).feasible(scanner)

    def test_a_replayed_design_keeps_its_feasibility(self):
        scanner = Scanner.preset("generic_3t")
        design = Press(te1_s=0.011, te2_s=0.019)
        assert design.bound(design.timings()).feasible(scanner)
        with pytest.raises(AttributeError):
            design.bound(design.timings()).te1_s  # a plain Sequence: only the events remain
