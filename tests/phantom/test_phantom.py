####################################################################################################
#                                         test_phantom.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Phantom gate: the bundled ICBM152 loads with its known geometry, tissue fractions       #
#          behave inside, outside, and on synthetic maps, and unknown names fail loudly.           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, Tissues
from mrsandbox.resources import Resources


#**************************************************************************************************#
#                                         Class TestBrain                                          #
#**************************************************************************************************#
class TestBrain:

    #**********************#
    #   bundled template   #
    #**********************#
    def test_icbm152_loads_with_known_geometry(self):
        brain = Brain.load("icbm152")
        assert brain.shape == (97, 115, 97)
        assert brain.name == "icbm152"
        assert 0.0 <= brain.gm.min() and brain.gm.max() <= 1.0

    #**********************************#
    #   fractions inside and outside   #
    #**********************************#
    def test_fractions_inside_and_outside(self):
        brain = Brain.load("icbm152")
        centre = brain.fractions(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
        assert isinstance(centre, Tissues)
        assert 0.3 < centre.gm + centre.wm + centre.csf <= 1.05
        outside = brain.fractions(position_mm=(500.0, 500.0, 500.0), size_mm=20.0)
        assert outside == Tissues(gm=0.0, wm=0.0, csf=0.0)

    #********************#
    #   synthetic maps   #
    #********************#
    def test_from_maps_with_identity_geometry(self):
        shape = (10, 10, 10)
        brain = Brain.from_maps(
            gm=np.ones(shape), wm=np.zeros(shape), csf=np.zeros(shape), affine=np.eye(4)
        )
        inside = brain.fractions(position_mm=(5.0, 5.0, 5.0), size_mm=4.0)
        assert inside == Tissues(gm=1.0, wm=0.0, csf=0.0)

    #**********************#
    #   anisotropic box    #
    #**********************#
    def test_anisotropic_box_averages_its_own_extent(self):
        # tissue in three z layers (cells 3, 4, 5 span z in [2.5, 5.5]): a box of
        # exactly that extent is pure tissue, stretching it along z alone dilutes the
        # mean by the exact partial volumes, and a half-cell offset weighs the edges
        gm = np.zeros((10, 10, 10))
        gm[:, :, 3:6] = 1.0
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=np.eye(4))
        thin = brain.fractions(position_mm=(5.0, 5.0, 4.0), size_mm=(4.0, 4.0, 3.0))
        thick = brain.fractions(position_mm=(5.0, 5.0, 4.0), size_mm=(4.0, 4.0, 7.0))
        shifted = brain.fractions(position_mm=(5.0, 5.0, 4.5), size_mm=(4.0, 4.0, 3.0))
        assert abs(thin.gm - 1.0) < 1e-6
        assert abs(thick.gm - 3.0 / 7.0) < 1e-6  # z in [0.5, 7.5] holds the same three
        assert abs(shifted.gm - 2.5 / 3.0) < 1e-6  # z in [3, 6]: half of cell 6 is empty

    #********************#
    #   unknown preset   #
    #********************#
    def test_unknown_brain_raises(self):
        with pytest.raises(KeyError, match="known:"):
            Brain.load("nonsense")


#**************************************************************************************************#
#                                       Class TestBigBrainMR                                       #
#**************************************************************************************************#
#                                                                                                  #
# The fetched BigBrain-MR preset behaves like any Brain; runs only once the cache holds it.        #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("bigbrain_mr/labels_800um"),
                    reason="bigbrain_mr not cached; Brain.load('bigbrain_mr') fetches it once")
class TestBigBrainMR:

    #****************#
    #   geometry     #
    #****************#
    def test_loads_with_centred_world_coordinates(self):
        brain = Brain.load("bigbrain_mr")
        assert brain.shape == (194, 240, 204)
        centre = brain.fractions((0.0, 0.0, 0.0), 20.0)
        assert centre.gm + centre.wm + centre.csf > 0.5  # the origin sits mid brain

    #****************#
    #   partiality   #
    #****************#
    def test_binary_labels_give_partial_volumes_by_averaging(self):
        brain = Brain.load("bigbrain_mr")
        f = brain.fractions((-30.0, 10.0, 10.0), 20.0)
        for fraction in (f.gm, f.wm):
            assert 0.0 < fraction < 1.0
        assert f.gm + f.wm + f.csf <= 1.0 + 1e-6


#**************************************************************************************************#
#                                        Class TestFatLayer                                        #
#**************************************************************************************************#
#                                                                                                  #
# A brain with a fat map puts the lipid layer where that tissue is, not at the brain boundary.     #
#                                                                                                  #
#**************************************************************************************************#
class TestFatLayer:

    def test_a_fat_map_is_the_lipid_layer(self):
        # brain in x < 8, two cells of bone (nothing), then two cells of scalp
        gm = np.zeros((16, 8, 8), dtype=np.float32)
        gm[:8] = 1.0
        fat = np.zeros_like(gm)
        fat[10:12] = 1.0
        affine = np.eye(4)
        zeros = np.zeros_like(gm)
        layered = Brain.from_maps(gm=gm, wm=zeros, csf=zeros, affine=affine, fat=fat)
        boundary = Brain.from_maps(gm=gm, wm=zeros, csf=zeros, affine=affine)
        # a 4 mm box over x in 6.5 to 10.5: a quarter brain, the bone, a quarter scalp
        position = (8.5, 3.5, 3.5)
        assert boundary.lipid_weight(position, 4.0) == pytest.approx(0.75)  # bone counted as fat
        assert layered.lipid_weight(position, 4.0) == pytest.approx(0.25)
        np.testing.assert_allclose(layered.lipid_offset(position, 4.0), (1.5, 0.0, 0.0), atol=1e-6)
        # inside the brain and in the bone alone there is no fat
        assert layered.lipid_weight((3.5, 3.5, 3.5), 4.0) == 0.0
        assert layered.lipid_weight((8.5, 3.5, 3.5), 1.0) == 0.0


#**************************************************************************************************#
#                                         Class TestQsmHead                                        #
#**************************************************************************************************#
#                                                                                                  #
# The QSM challenge 2.0 head: a whole head with skull and scalp, placed in the cache by hand.      #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("qsm_rc2/labels"),
                    reason="the QSM challenge 2.0 head is not redistributed; place it by hand")
class TestQsmHead:

    def test_the_head_has_a_brain_a_skull_and_a_scalp(self):
        brain = Brain.load("qsm_rc2")
        assert brain.shape == (256, 320, 320) and brain.name == "qsm_rc2"
        inside = brain.fractions((0.0, 0.0, 40.0), 10.0)
        assert inside.gm + inside.wm + inside.csf > 0.9
        assert brain.lipid_weight((0.0, 0.0, 40.0), 10.0) == 0.0
        # going up through the vertex: brain, then the skull, then the scalp, then air
        assert brain.lipid_weight((0.0, 0.0, 80.0), 10.0) > 0.5
        assert brain.fractions((0.0, 0.0, 80.0), 10.0) == Tissues(gm=0.0, wm=0.0, csf=0.0)
        assert brain.lipid_weight((0.0, 0.0, 95.0), 10.0) == 0.0
        # the scalp sits outward of a voxel that clips it from inside
        assert brain.lipid_offset((60.0, 0.0, 20.0), 10.0)[0] > 1.0

    def test_the_heads_maps_make_one_coherent_subject(self):
        # labels, field, and relaxation of one 7 T subject: the water T1 a voxel
        # resolves is the head's own R1 there, white-matter rich voxels read shorter
        from mrsandbox import FieldMap, Params, Press, Scanner, VirtualScan
        from mrsandbox.params import Mapped
        from mrsandbox.resources import QSMHead

        t1 = QSMHead.volume("T1map")
        assert t1.shape == (256, 320, 320) and QSMHead.volume("Chimap").shape == t1.shape
        grey, white = float(t1.mean((0.0, -40.0, 0.0), 20.0)), float(t1.mean((25.0, 10.0, 30.0), 20.0))
        assert 1.6 < grey < 2.3 and 1.1 < white < 1.6  # 7 T
        scan = VirtualScan(Brain.load("qsm_rc2"), Scanner.preset("generic_7t"),
                           Press(te1_s=0.011, te2_s=0.019), field=FieldMap.load("qsm_rc2", b0_t=7.0),
                           params=Params(water=dict(t1_s=Mapped(t1))))
        data = scan.voxel(position_mm=(25.0, 10.0, 30.0), size_mm=20.0).acquire(seed=3, npoints=128)
        local = data.hdr_ext["SimulationGroundTruth"]["local"]["water.t1_s"]
        assert local == pytest.approx(white, rel=1e-6)
        with pytest.raises(KeyError):
            QSMHead.volume("T2map")
