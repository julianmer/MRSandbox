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
        # tissue in three z layers: stretching the box along z alone dilutes the mean
        # exactly as the box-lookup convention (floor/ceil plus one) predicts
        gm = np.zeros((10, 10, 10))
        gm[:, :, 3:6] = 1.0
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=np.eye(4))
        thin = brain.fractions(position_mm=(5.0, 5.0, 4.0), size_mm=(4.0, 4.0, 3.0))
        thick = brain.fractions(position_mm=(5.0, 5.0, 4.0), size_mm=(4.0, 4.0, 7.0))
        assert abs(thin.gm - 3.0 / 5.0) < 1e-6  # layers 2..6 hold three GM layers
        assert abs(thick.gm - 3.0 / 9.0) < 1e-6  # layers 0..8 hold the same three

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
