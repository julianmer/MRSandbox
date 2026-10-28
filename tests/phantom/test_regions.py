####################################################################################################
#                                          test_regions.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 28/10/26                                                                                #
#                                                                                                  #
# Purpose: Region gate: a voxel's tissue fractions split by region add up to the fractions, a      #
#          brain without regions has none, the atlases number their labels into the package's     #
#          regions, and CerebrA puts the thalamus and the cerebellum where they are.               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain
from mrsandbox.phantom.regions import BIGBRAIN, REGIONS, numbered
from mrsandbox.resources import Resources


#*************#
#   helpers   #
#*************#
def halves(regions=True):
    """An 8 mm cube of grey and white matter, frontal on its left half and thalamus on its
    right, one cell a millimetre centred on the origin."""
    shape, affine = (8, 8, 8), np.eye(4)
    affine[:3, 3] = -3.5
    gm, wm = np.full(shape, 0.6, np.float32), np.full(shape, 0.3, np.float32)
    labels = np.full(shape, REGIONS.index("frontal"), np.uint8)
    labels[4:] = REGIONS.index("thalamus")
    return Brain(gm=gm, wm=wm, csf=np.zeros(shape, np.float32), affine=affine,
                 regions=labels if regions else None)


#**************************************************************************************************#
#                                         Class TestRegions                                        #
#**************************************************************************************************#
class TestRegions:

    #***********#
    #   parts   #
    #***********#
    def test_the_parts_split_the_fractions_by_region(self):
        brain = halves()
        parts = brain.parts((-1.0, 0.0, 0.0), 4.0)
        # a box three quarters on the frontal half, one on the thalamic
        assert set(parts) == {"frontal", "thalamus"}
        assert parts["frontal"].gm == pytest.approx(0.6 * 0.75)
        assert parts["thalamus"].wm == pytest.approx(0.3 * 0.25)
        total = np.sum([list(part) for part in parts.values()], axis=0)
        assert np.allclose(total, list(brain.fractions((-1.0, 0.0, 0.0), 4.0)))

    def test_a_brain_without_regions_has_no_parts(self):
        brain = halves(regions=False)
        assert brain.regions is None and brain.parts((0.0, 0.0, 0.0), 4.0) == {}
        assert halves().parts((500.0, 0.0, 0.0), 4.0) == {}

    #*************#
    #   atlases   #
    #*************#
    def test_bigbrains_labels_number_into_the_regions(self):
        labels = np.array([[[0, 2, 6, 19, 99]]])
        assert [REGIONS[i] for i in numbered(labels, BIGBRAIN).ravel()] == [
            "none", "cortex", "thalamus", "substantia_nigra", "none"]

    @pytest.mark.skipif(not Resources.cached("cerebra/labels"),
                        reason="the CerebrA atlas is not in the cache")
    def test_cerebra_puts_the_thalamus_and_the_cerebellum_where_they_are(self):
        brain = Brain.load("icbm152")
        size = (10.0, 10.0, 10.0)

        def most(place):
            parts = brain.parts(place, size)
            return max(parts, key=lambda region: parts[region].gm + parts[region].wm)

        assert most((-10.0, -18.0, 8.0)) == "thalamus"
        assert most((-25.0, -65.0, -35.0)) == "cerebellum"
        assert most((30.0, 30.0, 30.0)) == "frontal"
        # the white matter under the occipital cortex belongs to the occipital lobe
        assert most((-25.0, -80.0, 10.0)) == "occipital"
