####################################################################################################
#                                           test_uk7t.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The UK7T conversions and, when the archive is cached, the Oxford session's 7 T stack.   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.resources import UK7T


#**************************************************************************************************#
#                                       Class TestConversion                                       #
#**************************************************************************************************#
class TestConversion:

    def test_dream_map_in_tenths_of_a_degree_becomes_the_relative_field(self):
        flip = np.array([[600.0, 300.0], [780.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(UK7T.transmit(flip, 60.0), [[1.0, 0.5], [1.3, 0.0]], atol=1e-6)

    def test_unknown_site_and_map_are_refused(self):
        with pytest.raises(KeyError):
            UK7T.file("abc", "DREAM_FA")
        with pytest.raises(KeyError):
            UK7T.volume("T1map")


#**************************************************************************************************#
#                                         Class TestStack                                          #
#**************************************************************************************************#
#                                                                                                  #
# The Oxford session's derived stack, gated when the archive or its members are in the cache.      #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not UK7T.cached("oxf"), reason="UK7T archive not cached (7.1 GB)")
class TestStack:

    def test_maps_land_in_the_7t_ranges(self):
        # the transmit field near nominal at the head centre and well below it at the
        # periphery (the 7 T centre brightening), the field offset centred within the
        # unambiguous range, R2* of tens per second, each on its own grid
        transmit = UK7T.volume("TB1map", "oxf")
        assert transmit.data.shape == (56, 64, 44)
        inside = transmit.data > 0.0
        centre = np.array(np.nonzero(inside)).mean(axis=1).round().astype(int)
        box = transmit.data[tuple(slice(c - 3, c + 4) for c in centre)]
        assert 0.85 < np.median(box[box > 0.0]) < 1.2
        assert np.percentile(transmit.data[inside], 5) < 0.6
        field = UK7T.volume("fieldmap", "oxf")
        assert field.data.shape == (64, 64, 64)
        assert abs(np.median(field.data[24:40, 24:40, 24:40])) < 30.0
        assert np.abs(field.data).max() <= 1.0 / (2.0 * 0.00102) + 1e-3
        r2star = UK7T.volume("R2starmap", "oxf")
        assert r2star.data.shape == (120, 156, 192)
        assert 20.0 < np.median(r2star.data[r2star.data > 0.0]) < 80.0
        # the maps answer box queries in the session's scanner space: the grid centre
        middle = transmit.affine[:3, :3] @ ((np.asarray(transmit.data.shape) - 1) / 2.0) + transmit.affine[:3, 3]
        assert transmit.mean(middle, 20.0) is not None
