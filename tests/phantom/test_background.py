####################################################################################################
#                                        test_background.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Background-signal gate (residual water and lipids, the Kreis 2021 terminology): water   #
#          scales physically (unsuppressed dwarfs the metabolites, the suppressed residual         #
#          matches the sampled residual fraction, NoWater is silent) and lipids follow the         #
#          geometry (zero weight deep in the brain, signal at the skull, NoLipids kills it).       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import Brain, NoLipids, NoWater, Params, Press, Scanner, Unsuppressed, VirtualScan

# full-scan integration renders: the tensorflow CI leg skips these (-m 'not render')
pytestmark = pytest.mark.render

INTERIOR = (10.0, -20.0, 5.0)


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def brain():
    return Brain.load("icbm152")


def height_at(scan, position, ppm_at, seed=7):
    voxel = scan.voxel(position_mm=position, size_mm=20.0)
    ppm, spec = voxel.spectrum(seed=seed, npoints=8192, sw_hz=4000.0)
    ppm, spec = np.asarray(ppm), ops.convert_to_numpy(spec)
    return float(np.abs(spec[np.argmin(np.abs(ppm - ppm_at))]))


def make_scan(brain, params=None, **kwargs):
    # quiet the receiver-side imperfections: these gates compare component heights
    # across scans, and a drawn frequency shift or spurious echo would move or pollute
    # the read bins (the imperfections carry their own exact-form tests in tests/scan)
    quiet = Params(acquisition=dict(shift_hz=0.0), spurious=dict(amplitude=0.0))
    params = quiet | (params if params is not None else Params())
    return VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                       params=params, **kwargs)


#**************************************************************************************************#
#                                         Class TestWater                                          #
#**************************************************************************************************#
class TestWater:

    #****************************#
    #   unsuppressed dominates   #
    #****************************#
    def test_unsuppressed_water_dwarfs_the_metabolites(self, brain):
        scan = make_scan(brain, water=Unsuppressed())
        assert height_at(scan, INTERIOR, 4.65) > 50.0 * height_at(scan, INTERIOR, 2.008)

    #**********************#
    #   residual scaling   #
    #**********************#
    def test_suppressed_residual_matches_the_sampled_fraction(self, brain):
        # the leaked water off: it sits under the residual at the same frequency (its own gate
        # below)
        quiet = Params(oov=dict(fraction=0.0))
        suppressed, unsuppressed = (make_scan(brain, params=quiet),
                                    make_scan(brain, params=quiet, water=Unsuppressed()))
        truth = suppressed.voxel(position_mm=INTERIOR, size_mm=20.0).sample(seed=7)
        ratio = height_at(suppressed, INTERIOR, 4.65) / height_at(unsuppressed, INTERIOR, 4.65)
        assert abs(ratio - truth.water.residual_fraction) < 0.3 * truth.water.residual_fraction

    #*********************#
    #   per-tissue T1     #
    #*********************#
    def test_each_tissue_recovers_with_its_own_t1(self, brain):
        # equal per-tissue values render as the one value does; a slow CSF beside fast tissues
        # leaves MORE water at a short TR than the mean T1 says (recovery is convex in T1), which
        # one adjustment cannot null with the rest (the T2 held to one value, since each tissue's
        # water decays with its own otherwise)
        from mrsandbox.params import PerTissue

        def water(t1):
            quiet = Params(acquisition=dict(shift_hz=0.0), spurious=dict(amplitude=0.0),
                           water=dict(t1_s=t1, t2_s=0.1))
            scan = VirtualScan(brain, Scanner.preset("generic_3t"),
                               Press(te1_s=0.011, te2_s=0.019, tr_s=1.0),
                               params=quiet, water=Unsuppressed())
            return height_at(scan, INTERIOR, 4.65)

        assert water(PerTissue(gm=1.2, wm=1.2, csf=1.2)) == pytest.approx(water(1.2), rel=1e-5)
        fractions = make_scan(brain).voxel(position_mm=INTERIOR, size_mm=20.0).fractions
        mean = ((fractions.gm * 1.0 + fractions.wm * 1.0 + fractions.csf * 4.0)
                / (fractions.gm + fractions.wm + fractions.csf))
        if fractions.csf > 0.02:
            assert water(PerTissue(gm=1.0, wm=1.0, csf=4.0)) > 1.01 * water(mean)

    #*********************#
    #   per-tissue T2     #
    #*********************#
    def test_each_tissue_decays_with_its_own_t2(self, brain):
        # the water's parts: one per tissue present, each at the tissue's own T2 (the CSF's long
        # line beside the tissues' shorter ones), so what a train leaves is their sum and not one
        # line at the mean
        scan = make_scan(brain, water=Unsuppressed())
        voxel = scan.voxel(position_mm=INTERIOR, size_mm=20.0)
        truth = voxel.sample(seed=7)
        parts = scan.water.lines(scan.engine, voxel.design(truth), voxel, truth, voxel._f0_mhz())
        present = [name for name in ("gm", "wm", "csf") if getattr(voxel.fractions, name) > 0.0]
        assert [t2 for _, _, t2 in parts] == [getattr(truth.water.t2_s, name) for name in present]
        assert max(t2 for _, _, t2 in parts) > 2.0 * min(t2 for _, _, t2 in parts)

    #**********************#
    #   out-of-voxel water   #
    #**********************#
    def test_the_leaked_water_is_one_broad_line_a_fraction_of_the_full_water(self, brain):
        # the water outside the voxel that the localization leaks (scan.calibrations OOV_WATER
        # record): one line at the water's frequency plus its offset, decaying with its own T2* (and
        # the draw's microscopic envelope), a fraction of the full water at t = 0 after the echo's
        # decay, at the drawn phase against the voxel's own residual
        from mrsandbox.phantom.water import WaterSource

        def fid(fraction):
            quiet = Params(acquisition=dict(shift_hz=0.0, spurious=dict(amplitude=0.0),
                                            phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                            eddy_transient_rad=0.0),
                           spurious=dict(amplitude=0.0), lineshape=dict(gauss_hz=0.0),
                           water=dict(t2_s=0.08), oov=dict(fraction=fraction, t2star_s=0.02,
                                                           offset_hz=0.0, phase_deg=180.0))
            scan = VirtualScan(brain, Scanner.preset("generic_3t"),
                               Press(te1_s=0.011, te2_s=0.019), params=quiet,
                               water=Unsuppressed())
            voxel = scan.voxel(position_mm=INTERIOR, size_mm=20.0)
            return voxel, np.asarray(voxel.fid(seed=7, npoints=1024, dwell_s=1.0 / 4000.0)[1],
                                     np.complex128)

        voxel, plain = fid(0.0)
        _, leaked = fid(0.004)
        extra = leaked - plain
        t = np.arange(1024) / 4000.0
        decay = np.polyfit(t[:200], np.log(np.abs(extra[:200])), 1)[0]
        assert abs(-1.0 / decay - 0.02) < 0.002  # its own T2*
        full = WaterSource.full_amplitude_mm(voxel) * np.exp(-0.030 / 0.08)
        # the fraction of the full water after TE
        assert abs(abs(extra[0]) / (0.004 * full) - 1.0) < 0.05
        water = np.asarray(voxel.scan.water.lines(voxel.scan.engine,
                                                  voxel.design(voxel.sample(seed=7)), voxel,
                                                  voxel.sample(seed=7), voxel._f0_mhz())[0][1])
        # opposite to the voxel's own water, as drawn
        assert abs(abs(np.angle(extra[0] / np.sum(water))) - np.pi) < 0.2

    #*************#
    #   silence   #
    #*************#
    def test_no_water_is_silent(self, brain):
        # NoWater must equal a residual fraction forced to zero exactly: what remains at
        # 4.65 ppm is the genuine metabolite floor (the beta-glucose anomeric proton sits
        # 3 Hz away), not water
        none = height_at(make_scan(brain, water=NoWater()), INTERIOR, 4.65)
        zeroed = height_at(
            make_scan(brain, params=Params(water=dict(residual_fraction=0.0))), INTERIOR, 4.65
        )
        assert abs(none - zeroed) < 1e-4 * max(zeroed, 1.0)
        assert height_at(make_scan(brain), INTERIOR, 4.65) > 1.5 * none


#**************************************************************************************************#
#                                         Class TestLipids                                         #
#**************************************************************************************************#
class TestLipids:

    #**************#
    #   geometry   #
    #**************#
    def test_lipid_weight_follows_the_anatomy(self, brain):
        assert brain.lipid_weight(INTERIOR, 20.0) == 0.0
        skull = self.skull_position(brain)
        assert brain.lipid_weight(skull, 20.0) > 0.05
        assert brain.lipid_weight((500.0, 500.0, 500.0), 20.0) == 0.0

    #************************#
    #   skull voxel bleeds   #
    #************************#
    def test_skull_voxel_bleeds_and_no_lipids_kills_it(self, brain):
        skull = self.skull_position(brain)
        with_lipids = make_scan(brain)
        without = make_scan(brain, lipids=NoLipids())
        assert height_at(with_lipids, skull, 1.30) > 3.0 * height_at(without, skull, 1.30)
        # deep in the brain the geometric weight keeps lipids invisible
        interior_ratio = height_at(with_lipids, INTERIOR, 1.30) / height_at(without, INTERIOR, 1.30)
        assert interior_ratio == 1.0  # zero weight inside: no lipid lines at all

    #************#
    #   helper   #
    #************#
    @staticmethod
    def skull_position(brain):
        for x in np.arange(50.0, 95.0, 2.0):
            if brain.lipid_weight((x, -20.0, 5.0), 20.0) > 0.05:
                return (float(x), -20.0, 5.0)
        pytest.fail("no skull-adjacent voxel found along the search line")


#**************************************************************************************************#
#                                    Class TestWaterRelaxation                                     #
#**************************************************************************************************#
#                                                                                                  #
# Water relaxation follows the voxel's tissue composition through the cited per-tissue table.      #
#                                                                                                  #
#**************************************************************************************************#
class TestWaterRelaxation:

    #****************#
    #   weighting    #
    #****************#
    def test_water_relaxation_weighs_the_local_tissue(self, brain):
        from mrsandbox import Tissues, VirtualScan, Scanner, Press

        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        truth = scan.params.sample(seed=7)
        assert isinstance(truth.water.t1_s, Tissues)  # the PerTissue leaf realized
        pure_gm = Tissues(gm=1.0, wm=0.0, csf=0.0)
        pure_csf = Tissues(gm=0.0, wm=0.0, csf=1.0)
        mixed = Tissues(gm=0.5, wm=0.5, csf=0.0)
        # the voxel resolves the realized leaves by weighing them with its fractions
        t1_gm, t2_gm = truth.water.t1_s.weighted(pure_gm), truth.water.t2_s.weighted(pure_gm)
        t1_csf = truth.water.t1_s.weighted(pure_csf)
        t1_mix, t2_mix = truth.water.t1_s.weighted(mixed), truth.water.t2_s.weighted(mixed)
        # T1 draws from the crowdsourced challenge priors (BOUDREAU), T2 stays cited
        assert abs(t1_gm - 1.548) < 5 * 0.156 and t2_gm == 0.110
        assert t1_csf == 4.3
        t1_wm = truth.water.t1_s.wm
        assert abs(t1_wm - 0.828) < 5 * 0.038
        assert abs(t1_mix - (t1_gm + t1_wm) / 2.0) < 1e-9
        assert abs(t2_mix - (0.110 + 0.080) / 2.0) < 1e-9

    #****************#
    #   overrides    #
    #****************#
    def test_scalar_override_passes_through(self, brain):
        from mrsandbox import Params, Press, Scanner, VirtualScan

        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=Params(water=dict(t1_s=1.1, t2_s=0.05)))
        truth = scan.params.sample(seed=7)
        voxel = scan.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        assert voxel.resolve(truth.water.t1_s) == 1.1 and voxel.resolve(truth.water.t2_s) == 0.05


#**************************************************************************************************#
#                                       Class TestLipidModel                                       #
#**************************************************************************************************#
#                                                                                                  #
# The skull lipids render the Hamilton six-peak triglyceride spectrum with its measured ratios.    #
#                                                                                                  #
#**************************************************************************************************#
class TestLipidModel:

    #*******************#
    #   peak pattern    #
    #*******************#
    def test_lipid_spectrum_follows_the_hamilton_ratios(self, brain):
        from mrsandbox import NoMM, NoWater, Params, Press, Scanner, VirtualScan

        quiet = Params(noise=dict(sigma=0.0), acquisition=dict(shift_hz=0.0, phase_deg=0.0),
                       conc={name: 0.0 for name in
                             VirtualScan.default_params().conc})
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=quiet, water=NoWater(), mm=NoMM())
        edge = (0.0, 72.0, 5.0)  # a voxel reaching a centimetre beyond the brain clips the scalp
        assert brain.lipid_weight(edge, 25.0) > 0.0
        ppm, spec = scan.voxel(position_mm=edge, size_mm=25.0).spectrum(seed=7, npoints=8192)
        ppm, real = np.asarray(ppm), np.real(spec)
        methylene = real[np.abs(ppm - 1.30) < 0.1].max()
        methyl = real[np.abs(ppm - 0.90) < 0.08].max()
        olefinic = real[np.abs(ppm - 5.30) < 0.1].max()
        assert methylene > 3.0 * methyl  # 70 percent versus 8.8 percent, width-diluted
        assert olefinic > 0.0  # the 5.3 ppm peak under water exists
        assert methylene > 5.0 * olefinic
