####################################################################################################
#                                          test_field.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Field map gate: a constant offset gives a pure phase ramp, the voxel shim nulls what a  #
#          shim can null (constant and linear fields) while quadratic structure survives, a box    #
#          outside the map dephases nothing, and a scan with a field renders broader lines than    #
#          the uniform field scan.                                                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, FieldMap, Press, Scanner, VirtualScan
from mrsandbox.resources import Resources

T = np.arange(1024) / 4000.0
BOX = dict(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)


#*************#
#   helpers   #
#*************#
def centred_map(values):
    affine = np.eye(4)
    affine[:3, 3] = -(np.asarray(values.shape) - 1) / 2.0
    return FieldMap.from_map(values.astype(np.float32), affine)


def quadratic_field(n=50, amplitude_hz=10.0):
    x = np.linspace(-1.0, 1.0, n)
    return amplitude_hz * (x[:, None, None] ** 2 + 0.0 * x[None, :, None] + 0.0 * x[None, None, :])


#**************************************************************************************************#
#                                        Class TestKernel                                          #
#**************************************************************************************************#
class TestKernel:

    #********************#
    #   constant field   #
    #********************#
    def test_constant_offset_is_a_pure_phase_ramp(self):
        field = centred_map(np.full((50, 50, 50), 5.0))
        kernel = field.kernel(**BOX, t_s=T, shim="none")
        assert np.abs(kernel - np.exp(2j * np.pi * 5.0 * T)).max() < 1e-5
        # and the shim nulls it entirely
        assert np.abs(field.kernel(**BOX, t_s=T) - 1.0).max() < 1e-5

    #******************#
    #   linear field   #
    #******************#
    def test_shim_nulls_a_linear_gradient(self):
        gradient = np.tile(np.linspace(-10.0, 10.0, 50), (50, 50, 1))
        field = centred_map(gradient)
        raw = field.kernel(**BOX, t_s=T, shim="none")
        shimmed = field.kernel(**BOX, t_s=T, shim="linear")
        assert np.abs(raw[-1]) < 0.2  # a 20 Hz spread dephases hard
        assert np.abs(shimmed - 1.0).max() < 1e-4  # the first-order shim removes it all

    #*********************#
    #   quadratic field   #
    #*********************#
    def test_quadratic_structure_survives_the_shim(self):
        field = centred_map(quadratic_field())
        shimmed = field.kernel(**BOX, t_s=T, shim="linear")
        assert np.abs(shimmed[-1]) < 0.9  # residual dephasing remains
        assert field.offsets(**BOX, shim="linear").std() > 0.5

    #***************#
    #   outside     #
    #***************#
    def test_outside_the_map_nothing_dephases(self):
        field = centred_map(quadratic_field())
        kernel = field.kernel(position_mm=(500.0, 0.0, 0.0), size_mm=20.0, t_s=T)
        assert np.abs(kernel - 1.0).max() == 0.0


#**************************************************************************************************#
#                                         Class TestScan                                           #
#**************************************************************************************************#
class TestScan:
    pytestmark = pytest.mark.render  # full-scan renders, skipped on the tensorflow CI leg

    #*******************#
    #   broader lines   #
    #*******************#
    def test_field_broadens_the_rendered_lines(self):
        # oscillation with a period below the box size: structure no shim can null
        brain = Brain.load("icbm152")
        wiggle = 4.0 * np.sin(2.0 * np.pi * np.arange(120) / 9.0)
        field = centred_map(np.tile(wiggle, (120, 120, 1)))
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))

        def naa_height(field_map):
            scan = VirtualScan(brain, *pieces, field=field_map)
            ppm, spec = scan.voxel(position_mm=(10, -20, 5), size_mm=20).spectrum(seed=7, npoints=8192)
            ppm, mag = np.asarray(ppm), np.abs(spec)
            return mag[np.abs(ppm - 2.008) < 0.1].max()

        assert naa_height(field) < 0.8 * naa_height(None)


#**************************************************************************************************#
#                                     Class TestBigBrainField                                      #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("bigbrain_mr/background_field_400um"),
                    reason="background field not cached; FieldMap.load('bigbrain_mr') fetches it once")
class TestBigBrainField:

    #*******************#
    #   preset loads    #
    #*******************#
    def test_preset_scales_with_field_and_shims_to_subhertz(self):
        at_3t = FieldMap.load("bigbrain_mr", b0_t=3.0)
        at_7t = FieldMap.load("bigbrain_mr", b0_t=7.0)
        raw_3t = at_3t.offsets((-30.0, 10.0, 10.0), 25.0, shim="none")
        raw_7t = at_7t.offsets((-30.0, 10.0, 10.0), 25.0, shim="none")
        assert abs(raw_7t.mean() / raw_3t.mean() - 7.0 / 3.0) < 1e-3
        shimmed = at_3t.offsets((-30.0, 10.0, 10.0), 25.0)
        assert shimmed.std() < 0.5 * raw_3t.std()  # the shim removes most structure
