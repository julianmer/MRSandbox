####################################################################################################
#                                     test_macromolecules.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Macromolecule gate: the typed table matches the Osprey source, the background sits at   #
#          the published positions with Gaussian shape, scales linearly with the drawn level,      #
#          lands in the in vivo area fraction ballpark, and vanishes with NoMM.                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import (Brain, ChallengeMM, MeasuredMM, NoMM, ParameterizedMM, Params, Press,
                       Scanner, VirtualScan)
from mrsandbox.resources import Resources
from mrsandbox.database import macromolecules

# full-scan integration renders: the tensorflow CI leg skips these (-m 'not render')
pytestmark = pytest.mark.render

KW = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def spectra():
    brain = Brain.load("icbm152")
    pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
    quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                   acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))

    def spectrum(level, mm=None):
        params = quiet | Params(mm=dict(level_mm=level))
        source = mm if mm is not None else ParameterizedMM()  # these gates test the
        scan = VirtualScan(brain, *pieces, params=params, mm=source)  # parameterized source
        ppm, spec = scan.voxel(**KW).spectrum(seed=7, npoints=8192)
        return np.asarray(ppm), np.real(spec)

    return spectrum


#**************************************************************************************************#
#                                      Class TestParameters                                        #
#**************************************************************************************************#
class TestParameters:

    #**********************#
    #   typed from code    #
    #**********************#
    def test_table_matches_the_osprey_source(self):
        assert macromolecules.OSPREY["MM09"] == (("MM09", 0.91, 0.14, 3.0),)
        assert macromolecules.OSPREY["MM17"] == (("MM17", 1.67, 0.15, 2.0),)
        assert len(macromolecules.OSPREY["MM20"]) == 4  # the four-part composite
        assert sum(p for _, _, _, p in macromolecules.OSPREY["MM20"]) == pytest.approx(2.39)

    #**********************#
    #   consensus check    #
    #**********************#
    def test_osprey_positions_land_on_consensus_peaks(self):
        """Every simulated MM Gaussian must sit on a peak the consensus table describes
        (within its own stated 0.01 to 0.04 ppm inter-paper variation, plus width)."""
        consensus = [entry["ppm"] for entry in macromolecules.CUDALBU.values()
                     if not isinstance(entry["ppm"], tuple)]
        for name in ("MM09", "MM12", "MM14", "MM17", "MM20"):
            for _, ppm, _, _ in macromolecules.OSPREY[name]:
                assert min(abs(ppm - c) for c in consensus) < 0.06, (name, ppm)


#**************************************************************************************************#
#                                      Class TestBackground                                        #
#**************************************************************************************************#
class TestBackground:

    #*******************#
    #   where and how   #
    #*******************#
    def test_background_sits_at_the_published_positions(self, spectra):
        ppm, with_mm = spectra(5.0)
        _, without = spectra(5.0, mm=NoMM())
        background = with_mm - without
        at_mm09 = background[np.abs(ppm - 0.91) < 0.05].max()
        away = np.abs(background[(ppm > 4.2) & (ppm < 4.5)]).max()
        assert at_mm09 > 10.0 * max(away, 1e-6)

    #***************#
    #   linearity   #
    #***************#
    def test_level_scales_the_background_linearly(self, spectra):
        ppm, base = spectra(5.0, mm=NoMM())
        _, low = spectra(3.0)
        _, high = spectra(6.0)
        window = np.abs(ppm - 0.91) < 0.05
        ratio = (high - base)[window].max() / (low - base)[window].max()
        assert abs(ratio - 2.0) < 0.05

    #********************#
    #   area fraction    #
    #********************#
    def test_area_fraction_lands_in_the_in_vivo_ballpark(self, spectra):
        ppm, with_mm = spectra(3.0)
        _, without = spectra(3.0, mm=NoMM())
        window = (ppm > 0.5) & (ppm < 4.2)
        fraction = (with_mm - without)[window].sum() / with_mm[window].sum()
        assert 0.10 < fraction < 0.30


#**************************************************************************************************#
#                                        Class TestChallenge                                       #
#**************************************************************************************************#
#                                                                                                  #
# The challenge basis lands on its positions with the challenge's own amplitude model.             #
#                                                                                                  #
#**************************************************************************************************#
class TestChallenge:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def background(spectra):
        ppm, with_mm = spectra(3.0, mm=ChallengeMM())
        _, without = spectra(3.0, mm=NoMM())
        return ppm, with_mm - without

    #*******************#
    #   peak fidelity   #
    #*******************#
    def test_components_land_on_their_positions_in_absorption(self, spectra):
        ppm, background = self.background(spectra)
        for centre in (0.91, 1.21, 1.43):
            window = np.abs(ppm - centre) < 0.12
            assert abs(ppm[window][np.argmax(background[window])] - centre) < 0.04
        window = (ppm > 0.5) & (ppm < 4.2)
        assert background[window].min() > -0.05 * background[window].max()

    #***********************#
    #   amplitude model     #
    #***********************#
    def test_first_point_is_kappa_times_the_ratios_times_creatine(self):
        # b = 0.4 x r x c_tCr per component on the basis' own first points
        from mrsandbox.database import macromolecules

        brain = Brain.load("icbm152")
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           mm=ChallengeMM())
        voxel = scan.voxel(**KW)
        truth = voxel.sample(seed=7)
        fid = scan.mm.fid(voxel, truth, 42.5774785 * 3.0, np.zeros(1))
        ratios = macromolecules.MRSI_CHALLENGE["components"]
        weights = [voxel.fractions.gm * ratios[name]["gm"] + voxel.fractions.wm * ratios[name]["wm"]
                   for name in ChallengeMM.FILES]
        expected = 0.4 * (truth.conc.Cr + truth.conc.PCr) * np.dot(weights, scan.mm._load()[:, 0])
        assert abs(fid[0] - expected) < 1e-5 * abs(expected)


#**************************************************************************************************#
#                                       Class TestMeasured                                         #
#**************************************************************************************************#
#                                                                                                  #
# The measured consensus spectrum maps into the simulation frame peak-faithfully.                  #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("mm_consensus/3t_human_steam_fid"),
                    reason="consensus MM data not cached; MeasuredMM fetches it once")
class TestMeasured:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def background(spectra, level):
        ppm, with_mm = spectra(level, mm=MeasuredMM())
        _, without = spectra(level, mm=NoMM())
        return ppm, with_mm - without

    #*******************#
    #   peak fidelity   #
    #*******************#
    def test_m094_lands_on_the_consensus_position(self, spectra):
        ppm, background = self.background(spectra, 3.0)
        window = (ppm > 0.7) & (ppm < 1.1)
        apex = ppm[window][np.argmax(background[window])]
        assert abs(apex - 0.94) < 0.03  # the consensus M0.94, its stated variation

    #*******************#
    #   proper phase    #
    #*******************#
    def test_background_is_absorptive(self, spectra):
        ppm, background = self.background(spectra, 3.0)
        window = (ppm > 0.5) & (ppm < 4.2)
        assert background[window].min() > -0.05 * background[window].max()

    #***************#
    #   linearity   #
    #***************#
    def test_level_scales_the_measured_background(self, spectra):
        ppm, low = self.background(spectra, 2.0)
        _, high = self.background(spectra, 4.0)
        window = np.abs(ppm - 0.94) < 0.05
        assert abs(high[window].max() / low[window].max() - 2.0) < 0.05

    #*****************#
    #   same league   #
    #*****************#
    def test_measured_and_parameterized_agree_in_magnitude(self, spectra):
        ppm, measured = self.background(spectra, 3.0)
        _, parameterized = spectra(3.0)
        _, without = spectra(3.0, mm=NoMM())
        window = (ppm > 0.5) & (ppm < 4.2)
        ratio = measured[window].sum() / (parameterized - without)[window].sum()
        assert 0.5 < ratio < 2.0

    #****************#
    #   te decay     #
    #****************#
    @staticmethod
    def measured_background(te_s, seed=7, npoints=4096):
        brain = Brain.load("icbm152")
        tree = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))
        pieces = (Scanner.preset("generic_3t"), Press(te_s / 2.0, te_s / 2.0))
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        with_mm = VirtualScan(brain, *pieces, params=tree, mm=MeasuredMM())
        without = VirtualScan(brain, *pieces, params=tree, mm=NoMM())
        ppm, s1 = with_mm.voxel(**kw).spectrum(seed=seed, npoints=npoints)
        _, s0 = without.voxel(**kw).spectrum(seed=seed, npoints=npoints)
        return np.asarray(ppm), np.real(np.asarray(s1) - np.asarray(s0))

    def test_te_decay_follows_the_typed_t2(self):
        from mrsandbox.database.macromolecules import CONSENSUS_T2

        peaks = {}
        for te in (0.02, 0.06):
            ppm, background = self.measured_background(te)
            window = (ppm > 0.84) & (ppm < 1.04)
            peaks[te] = background[window].max()
        t2 = CONSENSUS_T2["t2_ms"]["M0.94"]["landheer_2020"][0] / 1000.0
        expected = np.exp(-0.04 / t2)  # the typed 3 T human M0.94 value
        assert abs(peaks[0.06] / peaks[0.02] - expected) < 0.05 * expected

    def test_measured_background_vanishes_at_long_echo_time(self):
        ppm, short = self.measured_background(0.02)
        _, long_te = self.measured_background(0.144)
        window = (ppm > 0.5) & (ppm < 4.2)
        assert np.abs(long_te[window]).max() < 0.08 * np.abs(short[window]).max()

    #*****************#
    #   tr recovery   #
    #*****************#
    @staticmethod
    def pulse_acquire_background(tr_s, seed=7, npoints=4096):
        from mrsandbox import PulseAcquire

        brain = Brain.load("icbm152")
        tree = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))
        pieces = (Scanner.preset("generic_3t"), PulseAcquire(tr_s=tr_s))
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        with_mm = VirtualScan(brain, *pieces, params=tree, mm=MeasuredMM())
        without = VirtualScan(brain, *pieces, params=tree, mm=NoMM())
        truth = with_mm.voxel(**kw).sample(seed=seed)
        ppm, s1 = with_mm.voxel(**kw).spectrum(seed=seed, npoints=npoints)
        _, s0 = without.voxel(**kw).spectrum(seed=seed, npoints=npoints)
        return np.asarray(ppm), np.real(np.asarray(s1) - np.asarray(s0)), truth

    def test_tr_saturates_each_component_by_its_own_t1(self):
        ppm, relaxed, truth = self.pulse_acquire_background(None)
        _, saturated, _ = self.pulse_acquire_background(0.3)
        for name, centre in (("mm091", 0.94), ("mm321", 3.21)):
            window = np.abs(ppm - centre) < 0.05
            ratio = saturated[window].max() / relaxed[window].max()
            expected = 1.0 - np.exp(-0.3 / truth.mm.t1[name])  # the pulse-acquire law
            assert abs(ratio - expected) < 0.03 * expected, (name, ratio, expected)


#**************************************************************************************************#
#                                        Class TestCoupled                                         #
#**************************************************************************************************#
#                                                                                                  #
# The coupled MM systems behave with the sequence: TE decay, co-editing, creatine anchoring.       #
#                                                                                                  #
#**************************************************************************************************#
class TestCoupled:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def background(sequence, seed=7, **params):
        brain = Brain.load("icbm152")
        tree = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
                      mm=dict(kappa=7.0), **params)
        pieces = (Scanner.preset("generic_3t"), sequence)
        with_mm = VirtualScan(brain, *pieces, params=tree)
        without = VirtualScan(brain, *pieces, params=tree, mm=NoMM())
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        ppm, s1 = with_mm.voxel(**kw).spectrum(seed=seed, npoints=4096)
        _, s0 = without.voxel(**kw).spectrum(seed=seed, npoints=4096)
        return np.asarray(ppm), np.real(s1 - s0)

    #**************#
    #   te decay   #
    #**************#
    def test_mm_vanish_at_long_echo_time(self):
        ppm, short = self.background(Press(te1_s=0.015, te2_s=0.015))
        _, long = self.background(Press(te1_s=0.072, te2_s=0.072))
        window = (ppm > 0.5) & (ppm < 4.2)
        assert np.abs(long[window]).max() < 0.05 * np.abs(short[window]).max()

    #********************#
    #   area fraction    #
    #********************#
    def test_short_te_fraction_in_the_consensus_ballpark(self):
        brain = Brain.load("icbm152")
        tree = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
                      mm=dict(kappa=10.4))
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.015, te2_s=0.015))
        with_mm = VirtualScan(brain, *pieces, params=tree)
        without = VirtualScan(brain, *pieces, params=tree, mm=NoMM())
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        ppm, s1 = with_mm.voxel(**kw).spectrum(seed=7, npoints=4096)
        _, s0 = without.voxel(**kw).spectrum(seed=7, npoints=4096)
        ppm = np.asarray(ppm)
        window = (ppm > 0.5) & (ppm < 4.2)
        fraction = np.real(s1 - s0)[window].sum() / np.real(s1)[window].sum()
        assert 0.10 < fraction < 0.30

    #*****************#
    #   co-editing    #
    #*****************#
    def test_mm_co_edit_in_the_mega_difference(self):
        from mrsandbox import MegaPress

        pair = MegaPress.pair()
        ppm, diff_on = self.background(pair["on"])
        _, diff_off = self.background(pair["off"])
        edited = diff_on - diff_off
        at_3 = np.abs(edited[np.abs(ppm - 3.0) < 0.15]).max()
        far = np.abs(edited[(ppm > 8.0) & (ppm < 9.5)]).max()
        assert at_3 > 10.0 * max(far, 1e-9)  # the consensus-flagged co-edited background

    #***********************#
    #   creatine anchoring  #
    #***********************#
    def test_background_scales_with_the_drawn_creatine(self):
        ppm, low = self.background(Press(te1_s=0.015, te2_s=0.015), conc=dict(Cr=4.0, PCr=2.0))
        _, high = self.background(Press(te1_s=0.015, te2_s=0.015), conc=dict(Cr=8.0, PCr=4.0))
        window = np.abs(ppm - 0.91) < 0.1
        assert abs(high[window].max() / low[window].max() - 2.0) < 0.05

    def test_background_reads_a_resolved_creatine(self):
        # the anchor reads the creatine like every other concentration leaf: a
        # per-tissue creatine of one value everywhere is that fixed value
        from mrsandbox import PerTissue

        sequence = Press(te1_s=0.015, te2_s=0.015)
        _, fixed = self.background(sequence, conc=dict(Cr=8.0, PCr=4.0))
        _, tissue = self.background(sequence, conc=dict(Cr=PerTissue(gm=8.0, wm=8.0, csf=8.0),
                                                        PCr=PerTissue(gm=4.0, wm=4.0, csf=4.0)))
        np.testing.assert_allclose(tissue, fixed, atol=1e-6 * np.abs(fixed).max())

    #***********************#
    #   couplings by table  #
    #***********************#
    def test_couplings_come_from_the_consensus_table(self):
        # the simulated J values are read from the cited CONSENSUS_COUPLINGS rows by
        # nearest peak position, so the table and the physics cannot drift apart
        from mrsandbox.phantom.macromolecules import CoupledMM

        assert CoupledMM._coupling_j_hz(0.91, 2.04) == 7.7  # leu/ile/val chain
        assert CoupledMM._coupling_j_hz(1.21, 4.24) == 6.6  # threonine
        assert CoupledMM._coupling_j_hz(1.67, 2.99) == 7.7  # the GABA-relevant chain
        with pytest.raises(ValueError, match="no consensus coupling"):
            CoupledMM._coupling_j_hz(6.0, 7.0)
