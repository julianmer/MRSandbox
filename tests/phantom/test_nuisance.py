####################################################################################################
#                                         test_nuisance.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Nuisance gate: water scales physically (unsuppressed dwarfs the metabolites, the        #
#          suppressed residual matches the sampled residual fraction, NoWater is silent) and       #
#          lipids follow the geometry (zero weight deep in the brain, signal at the skull,         #
#          NoLipids kills it).                                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import Brain, NoLipids, NoWater, Press, Scanner, Unsuppressed, VirtualScan

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


def make_scan(brain, **kwargs):
    return VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019), **kwargs)


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
        suppressed, unsuppressed = make_scan(brain), make_scan(brain, water=Unsuppressed())
        truth = suppressed.voxel(position_mm=INTERIOR, size_mm=20.0).sample(seed=7)
        ratio = height_at(suppressed, INTERIOR, 4.65) / height_at(unsuppressed, INTERIOR, 4.65)
        assert abs(ratio - truth.water.residual_fraction) < 0.3 * truth.water.residual_fraction

    #*************#
    #   silence   #
    #*************#
    def test_no_water_is_silent(self, brain):
        # NoWater must equal a residual fraction forced to zero exactly: what remains at
        # 4.65 ppm is the genuine metabolite floor (the beta-glucose anomeric proton sits
        # 3 Hz away), not water
        from mrsandbox import Params

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
        from mrsandbox.phantom.water import WaterSource

        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        truth = scan.params.sample(seed=7)
        assert isinstance(truth.water.t1_s, Tissues)  # the PerTissue leaf realized
        pure_gm = Tissues(gm=1.0, wm=0.0, csf=0.0)
        pure_csf = Tissues(gm=0.0, wm=0.0, csf=1.0)
        mixed = Tissues(gm=0.5, wm=0.5, csf=0.0)
        t1_gm, t2_gm = WaterSource.relaxation(pure_gm, truth)
        t1_csf, _ = WaterSource.relaxation(pure_csf, truth)
        t1_mix, t2_mix = WaterSource.relaxation(mixed, truth)
        assert t1_gm == 1.23 and t2_gm == 0.110  # the cited pure-tissue values
        assert t1_csf == 4.3
        assert abs(t1_mix - (1.23 + 0.95) / 2.0) < 1e-9
        assert abs(t2_mix - (0.110 + 0.080) / 2.0) < 1e-9

    #****************#
    #   overrides    #
    #****************#
    def test_scalar_override_passes_through(self, brain):
        from mrsandbox import Params, Press, Scanner, Tissues, VirtualScan
        from mrsandbox.phantom.water import WaterSource

        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=Params(water=dict(t1_s=1.1, t2_s=0.05)))
        truth = scan.params.sample(seed=7)
        t1, t2 = WaterSource.relaxation(Tissues(gm=1.0, wm=0.0, csf=0.0), truth)
        assert t1 == 1.1 and t2 == 0.05


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
        edge = (0.0, 60.0, 5.0)  # a voxel straddling the boundary carries lipids
        assert brain.lipid_weight(edge, 25.0) > 0.0
        ppm, spec = scan.voxel(position_mm=edge, size_mm=25.0).spectrum(seed=7, npoints=8192)
        ppm, real = np.asarray(ppm), np.real(spec)
        methylene = real[np.abs(ppm - 1.30) < 0.1].max()
        methyl = real[np.abs(ppm - 0.90) < 0.08].max()
        olefinic = real[np.abs(ppm - 5.30) < 0.1].max()
        assert methylene > 3.0 * methyl  # 70 percent versus 8.8 percent, width-diluted
        assert olefinic > 0.0  # the 5.3 ppm peak under water now exists
        assert methylene > 5.0 * olefinic
