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
#          independently, the filters standing for crushers weight as the crushers would, and      #
#          exp(-b ADC) reaches the rendered voxel exactly, each water part with its tissue's ADC.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest

from mrsandbox import (Brain, CoherenceFilter, Delay, Gradient, LocalizedMegaPress, NoLipids,
                       NoMM, NoWater, Params, PerTissue, Press, Pulse, Scanner, Sequence,
                       Unsuppressed, VirtualScan)
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

    #*******************#
    #   filter pairs    #
    #*******************#
    def test_a_filter_pair_weights_like_the_crushers_it_stands_for(self):
        # an editing pair keeps the order and carries opposite moments; a refocusing pair
        # carries equal moments across the order's turn: both rephase and both weight by
        # (2 pi m)^2 tau, and nothing is left helical afterwards (the trailing delay is free)
        editing = Sequence(Pulse(90.0), CoherenceFilter(-1, K), Delay(DELTA),
                           CoherenceFilter(-1, -K), Delay(0.5))
        refocusing = Sequence(Pulse(90.0), CoherenceFilter(+1, K), Delay(DELTA / 2.0),
                              Pulse(180.0, phase_deg=90.0), Delay(DELTA / 2.0),
                              CoherenceFilter(-1, K), Delay(0.5))
        for pair in (editing, refocusing):
            assert abs(b_value_ms_um2(pair) - EXPECTED) < 1e-9 * EXPECTED

    def test_the_editing_designs_carry_their_crusher_pairs(self, monkeypatch):
        # LocalizedMegaPress's two editing pairs: 2 (2 pi m)^2 tau over the design without them
        kw = dict(f0_mhz=127.77, edit_s=0.015, size_mm=30.0)
        with_pairs = b_value_ms_um2(LocalizedMegaPress(7.43, **kw))
        m = LocalizedMegaPress.EDIT_CRUSHER_PER_MM
        monkeypatch.setattr(LocalizedMegaPress, "EDIT_CRUSHER_PER_MM", 0.0)
        without = b_value_ms_um2(LocalizedMegaPress(7.43, **kw))
        pairs = 2.0 * (2.0 * math.pi * m * 1e3) ** 2 * 0.015 * 1e-9
        assert abs(with_pairs - without - pairs) < 1e-9 * pairs
        assert without < 1e-3 * pairs  # the refocusing crushers stay unplayed


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
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
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
            modelled = VirtualScan(brain, *pieces,
                                   params=quiet | Params(diffusion=dict(model=model)),
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

    #*********************#
    #   the water parts   #
    #*********************#
    @pytest.mark.render
    def test_each_water_part_fades_with_its_tissues_adc(self):
        # the water's parts, one per tissue, each faded by exp(-b ADC) of its own tissue:
        # with the relaxation pinned equal across the tissues the parts are one line apart
        # from their shares, so the peak falls by the share-weighted mean of the fadings;
        # a CSF ADC alone fades the CSF's share by exp(-b 3.0) and nothing else
        brain = Brain.load("icbm152")
        pair = Sequence(Pulse(90.0), CoherenceFilter(-1, K / 3.0), Delay(DELTA),
                        CoherenceFilter(-1, -K / 3.0), Delay(0.005))
        b = EXPECTED / 9.0
        conc = {name: 0.0 for name in VirtualScan.default_params().conc}
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
                       conc=conc, diffusion=dict(mm=0.0075),
                       water=dict(t1_s=PerTissue(1.0, 1.0, 1.0), t2_s=PerTissue(0.08, 0.08, 0.08)))
        kw = dict(position_mm=(0.0, -20.0, 20.0), size_mm=20.0)
        pieces = (Scanner.preset("generic_3t"), pair)
        sources = dict(water=Unsuppressed(), lipids=NoLipids(), mm=NoMM())

        def peak(water_adc):
            params = quiet | Params(diffusion=dict(water=water_adc))
            voxel = VirtualScan(brain, *pieces, params=params, **sources).voxel(**kw)
            _, spectrum = voxel.spectrum(seed=7, npoints=2048)
            return np.abs(np.asarray(spectrum)).max(), voxel

        still, voxel = peak(PerTissue(0.0, 0.0, 0.0))
        f, content = voxel.fractions, brain.water_content
        share = {t: getattr(f, t) * content[t] for t in ("gm", "wm", "csf")}
        assert share["csf"] > 0.1 * sum(share.values())
        csf_only, _ = peak(PerTissue(0.0, 0.0, 3.0))
        law = (share["gm"] + share["wm"] + share["csf"] * math.exp(-b * 3.0)) / sum(share.values())
        assert abs(csf_only / still - law) < 1e-3 * law
        drawn, voxel = peak(VirtualScan.default_params().diffusion.water)
        adc = voxel.sample(seed=7).diffusion.water
        law = sum(share[t] * math.exp(-b * getattr(adc, t)) for t in share) / sum(share.values())
        assert abs(drawn / still - law) < 1e-3 * law

    #**************************#
    #   per metabolite adcs    #
    #**************************#
    def test_the_major_metabolites_carry_their_measured_adc_ratios(self):
        # Deelchand et al. 2018: tNAA 0.13, tCr 0.13, tCho 0.12, Glu 0.15, mIns 0.11
        # over their mean 0.128 scale the tissue ADCs; a metabolite without a row keeps them
        adc = VirtualScan.default_params().sample(0).diffusion.adc
        assert abs(adc["GABA"].gm - 0.10) < 1e-12 and abs(adc["GABA"].wm - 0.15) < 1e-12
        assert abs(adc["Glu"].gm / adc["Ins"].gm - 0.15 / 0.11) < 1e-9
        assert adc["NAA"].wm == adc["NAAG"].wm == adc["PCr"].wm
        assert abs(adc["GPC"].wm - 0.15 * 0.12 / 0.128) < 1e-9
