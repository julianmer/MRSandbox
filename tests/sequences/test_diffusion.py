####################################################################################################
#                                        test_diffusion.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: Diffusion gate: the b-value built from the sequence's own gradient events matches the   #
#          analytic law, stored helices keep attenuating through a mixing time, axes add           #
#          independently, and exp(-b ADC) reaches the rendered voxel exactly.                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest

from mrsandbox import (Brain, CoherenceFilter, Delay, Gradient, NoLipids, NoMM, NoWater,
                       Params, Press, Pulse, Scanner, Sequence, VirtualScan)
from mrsandbox.sequences.diffusion import b_value_ms_um2

K, DELTA = 40.0, 0.05  # cycles per mm, seconds
EXPECTED = (2.0 * math.pi * K * 1e3) ** 2 * DELTA * 1e-9  # the analytic kick-pair b


#**************************************************************************************************#
#                                         Class TestBValue                                         #
#**************************************************************************************************#
class TestBValue:

    #*******************#
    #   the kick pair   #
    #*******************#
    def test_kick_pair_matches_the_analytic_b(self):
        pgse = Sequence(Pulse(90.0), CoherenceFilter(-1), Gradient(K), Delay(DELTA),
                        Gradient(-K), Delay(0.005))
        assert abs(b_value_ms_um2(pgse) - EXPECTED) < 1e-9 * EXPECTED

    #*****************#
    #   spin echo     #
    #*****************#
    def test_same_sign_pair_around_a_refocusing_pulse_matches(self):
        echo = Sequence(Pulse(90.0), CoherenceFilter(+1), Gradient(K), Delay(DELTA / 2.0),
                        Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(DELTA / 2.0),
                        Gradient(K), Delay(0.010))
        assert abs(b_value_ms_um2(echo) - EXPECTED) < 1e-9 * EXPECTED
        assert b_value_ms_um2(Press(0.011, 0.019)) == 0.0  # filters carry no moments

    #*****************#
    #   storage       #
    #*****************#
    def test_stored_helix_keeps_attenuating_through_a_mixing_time(self):
        tm = 0.2
        steam = Sequence(Pulse(90.0), CoherenceFilter(-1), Gradient(K), Delay(DELTA),
                         Pulse(90.0), CoherenceFilter(0), Gradient(3.7), Delay(tm),
                         Pulse(90.0), CoherenceFilter(-1), Gradient(-K), Delay(0.005))
        expected = (2.0 * math.pi * K * 1e3) ** 2 * (DELTA + tm) * 1e-9
        # the TM crusher plays at coherence order zero and adds nothing, while the
        # stored helix attenuates through the whole mixing time, as a stimulated echo does
        assert abs(b_value_ms_um2(steam) - expected) < 1e-9 * expected

    #*************#
    #   axes      #
    #*************#
    def test_axes_add_independently(self):
        crossed = Sequence(Pulse(90.0), CoherenceFilter(-1), Gradient(K, axis="x"),
                           Gradient(K, axis="y"), Delay(DELTA), Gradient(-K, axis="x"),
                           Gradient(-K, axis="y"), Delay(0.005))
        assert abs(b_value_ms_um2(crossed) - 2.0 * EXPECTED) < 1e-9 * EXPECTED


#**************************************************************************************************#
#                                      Class TestAttenuation                                       #
#**************************************************************************************************#
class TestAttenuation:

    #**********************#
    #   the rendered law   #
    #**********************#
    @pytest.mark.render  # a full-scan render, skipped on the tensorflow CI leg
    def test_rendered_amplitudes_fade_by_exp_minus_b_adc(self):
        brain = Brain.load("icbm152")
        weighted = Sequence(Pulse(90.0), CoherenceFilter(+1), Gradient(30.0),
                            Delay(0.020), Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1),
                            Delay(0.020), Gradient(30.0), Delay(0.005))
        b = (2.0 * math.pi * 30.0 * 1e3) ** 2 * 0.040 * 1e-9  # the independent expectation
        conc = {name: 0.0 for name in VirtualScan.default_params().conc}
        conc["sIns"] = 5.0
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
                       conc=conc,
                       # pin the drawn mm ADC so both trees consume the same randomness
                       diffusion=dict(mm=0.0075))
        frozen = quiet | Params(diffusion=dict(
            adc={name: 0.0 for name in conc}, water=0.0, mm=0.0, lipids=0.0))
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        pieces = (Scanner.preset("generic_3t"), weighted)
        sources = dict(water=NoWater(), lipids=NoLipids(), mm=NoMM())
        still = VirtualScan(brain, *pieces, params=frozen, **sources).voxel(**kw)
        fading = VirtualScan(brain, *pieces, params=quiet, **sources).voxel(**kw)
        _, reference = still.spectrum(seed=7, npoints=2048)
        truth = fading.sample(seed=7)
        adc = truth.diffusion.adc["sIns"]
        f = fading.fractions
        total = f.gm + f.wm + f.csf
        adc = (f.gm * adc.gm + f.wm * adc.wm + f.csf * adc.csf) / total
        # the metabolite decay law is the drawn model: the monoexponential
        # representation, or the consensus sticks whose powder average is
        # sqrt(pi / (4 b D)) erf(sqrt(b D)) with D = 3 ADC (Ligneul 2024, 5.1)
        x = b * adc
        sticks = math.sqrt(math.pi / (12.0 * x)) * math.erf(math.sqrt(3.0 * x))
        for model, law in (("gaussian", math.exp(-x)), ("sticks", sticks)):
            modelled = VirtualScan(brain, *pieces, params=quiet | Params(diffusion=dict(model=model)),
                                   **sources).voxel(**kw)
            _, attenuated = modelled.spectrum(seed=7, npoints=2048)
            ratio = np.abs(np.asarray(attenuated)).max() / np.abs(np.asarray(reference)).max()
            assert abs(ratio - law) < 1e-3 * law
        # the sticks decay is weaker than the Gaussian one at this b and the two
        # agree to first order at low b (the ADC is the same low-b slope)
        assert sticks > math.exp(-x)
        low = 0.05 * adc
        assert abs(math.sqrt(math.pi / (12.0 * low)) * math.erf(math.sqrt(3.0 * low))
                   - math.exp(-low)) < 1e-3
