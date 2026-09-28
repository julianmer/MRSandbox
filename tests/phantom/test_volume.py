####################################################################################################
#                                          test_volume.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/09/26                                                                                #
#                                                                                                  #
# Purpose: Volume gate: a box reads exactly the cells it overlaps with their partial volumes,      #
#          through scaled, shifted, and rotated affines alike, trailing axes ride along, and a     #
#          box outside the grid is empty.                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Volume
from mrsandbox.resources import Resources


#*************#
#   helpers   #
#*************#
def ramp(shape=(10, 10, 10), axis=0):
    """A volume whose value is its index along one axis."""
    index = np.arange(shape[axis], dtype=np.float64)
    return np.broadcast_to(np.expand_dims(index, [a for a in range(3) if a != axis]),
                           shape).copy()


#**************************************************************************************************#
#                                         Class TestVolume                                         #
#**************************************************************************************************#
class TestVolume:

    #**********************#
    #   partial volumes    #
    #**********************#
    def test_box_weights_are_the_exact_partial_volumes(self):
        volume = Volume(ramp(), np.eye(4))
        # a 4 mm box centred on 5.0 spans [3, 7]: cells 3 and 7 half, 4 to 6 whole
        values, weights = volume.box((5.0, 5.0, 5.0), 4.0)
        assert weights.shape == (5, 5, 5) and values.shape == (5, 5, 5)
        assert abs(weights.sum() - 64.0) < 1e-12
        assert abs(weights[0, 0, 0] - 0.125) < 1e-12 and abs(weights[1, 1, 1] - 1.0) < 1e-12
        assert abs(volume.mean((5.0, 5.0, 5.0), 4.0) - 5.0) < 1e-12
        # a 3 mm box spans [3.5, 6.5], the whole cells 4 to 6 and nothing else
        _, whole = volume.box((5.0, 5.0, 5.0), 3.0)
        assert whole.shape == (3, 3, 3) and np.all(whole == 1.0)

    #*************************#
    #   scaled and shifted    #
    #*************************#
    def test_scaled_and_shifted_affine_places_the_box(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = -9.0  # world x = 2 i - 9, so i = 5 sits at x = 1
        volume = Volume(ramp(), affine)
        assert abs(volume.mean((1.0, 1.0, 1.0), 4.0) - 5.0) < 1e-12
        # 4 mm is two cells: x in [-1, 3] covers cells 4 (x = -1) to 6 (x = 3) by halves
        _, weights = volume.box((1.0, 1.0, 1.0), 4.0)
        assert weights.shape == (3, 3, 3)
        assert np.allclose(weights.sum(axis=(1, 2)) / 4.0, [0.5, 1.0, 0.5])

    #*************#
    #   rotated   #
    #*************#
    def test_rotated_affine_reads_the_world_axes(self):
        # the grid's first axis lies along world y: a box elongated along world y sees
        # the ramp, one elongated along world x does not
        affine = np.array([[0.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        volume = Volume(ramp(axis=0), affine)
        along_y, _ = volume.box((5.0, 5.0, 5.0), (1.0, 5.0, 1.0))
        along_x, _ = volume.box((5.0, 5.0, 5.0), (5.0, 1.0, 1.0))
        assert along_y.shape == (5, 1, 1) and np.ptp(along_y) == 4.0
        assert along_x.shape == (1, 5, 1) and np.ptp(along_x) == 0.0

    #*********************#
    #   trailing axes     #
    #*********************#
    def test_trailing_axes_ride_along(self):
        data = np.stack([ramp(axis=0), ramp(axis=1)], axis=-1)
        volume = Volume(data, np.eye(4))
        mean = volume.mean((4.0, 6.0, 5.0), 2.0)
        assert mean.shape == (2,) and np.allclose(mean, (4.0, 6.0))
        assert np.allclose(volume.nearest((4.4, 5.6, 5.0)), (4.0, 6.0))

    #***************#
    #   outside     #
    #***************#
    def test_outside_the_grid_is_empty(self):
        volume = Volume(ramp(), np.eye(4))
        values, weights = volume.box((50.0, 5.0, 5.0), 3.0)
        assert values.size == 0 and weights.size == 0
        assert volume.mean((50.0, 5.0, 5.0), 3.0) is None
        assert volume.nearest((50.0, 5.0, 5.0)) is None
        # a box straddling the edge keeps only its inside part: x in [-1.5, 1.5] holds
        # cells 0 and 1 whole, y and z in [3.5, 6.5] three whole cells each
        _, edge = volume.box((0.0, 5.0, 5.0), 3.0)
        assert abs(edge.sum() - 2.0 * 9.0) < 1e-12


#**************************************************************************************************#
#                                      Class TestBigBrainMaps                                      #
#**************************************************************************************************#
#                                                                                                  #
# The registered BigBrain-MR quantitative maps load as Volumes on the brain's own grid.            #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("bigbrain_mr/t1map_800um")
                    or not Resources.cached("bigbrain_mr/labels_800um"),
                    reason="bigbrain_mr maps not cached; Volume.load fetches them once")
class TestBigBrainMaps:

    #**************#
    #   t1 map     #
    #**************#
    def test_t1_map_shares_the_brain_grid_and_reads_in_seconds(self):
        from mrsandbox import Brain, Mapped, Params, Press, Scanner, VirtualScan

        t1 = Volume.load("bigbrain_mr/t1map_800um")
        brain = Brain.load("bigbrain_mr")
        assert t1.shape == brain.shape and np.allclose(t1.affine, brain.affine)
        box = dict(position_mm=(-30.0, 10.0, 10.0), size_mm=20.0)
        local = t1.mean(**box)
        assert 0.8 < local < 2.5  # 7 T MP2RAGE brain tissue, seconds
        # through the scan: the water T1 of that voxel is the map's box mean
        scan = VirtualScan(brain, Scanner.preset("generic_7t"), Press(te1_s=0.011, te2_s=0.019),
                           params=Params(water=dict(t1_s=Mapped(t1))))
        voxel = scan.voxel(**box)
        assert voxel.resolve(voxel.sample(seed=7).water.t1_s) == float(local)
