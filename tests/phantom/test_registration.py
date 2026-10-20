####################################################################################################
#                                       test_registration.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The linear registration recovers a known affine between two views of one object, and    #
#          the composed affine makes the moved Volume read the target's values at its positions.   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox import Volume
from mrsandbox.phantom.registration import _matrix, overlap, register


#**************************************************************************************************#
#                                      Class TestRegistration                                      #
#**************************************************************************************************#
class TestRegistration:

    @staticmethod
    def blob(shape, affine, centre_mm, seed):
        """A smooth textured object: an ellipsoid with random low-frequency contrast."""
        grid = np.array(np.indices(shape)).reshape(3, -1)
        world = affine[:3, :3] @ grid + affine[:3, 3:4]
        d = (world - np.asarray(centre_mm)[:, None]) / np.array([[30.0], [40.0], [35.0]])
        rng = np.random.default_rng(seed)
        texture = sum(np.cos(world.T @ k + p) for k, p in
                      zip(rng.normal(0.0, 0.08, (6, 3)), rng.uniform(0.0, 6.0, 6)))
        # a soft edge (a 6 mm ramp) so two grids sample the boundary alike
        edge = np.clip((1.0 - np.sqrt((d * d).sum(axis=0))) * 6.0, 0.0, 1.0)
        values = edge * (1.5 + 0.3 * texture)
        return values.reshape(shape).astype(np.float32)

    def test_a_known_affine_is_recovered(self):
        target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        target_affine[:3, 3] = -60.0
        target = Volume(self.blob((60, 60, 60), target_affine, (0.0, 0.0, 0.0), 3), target_affine)
        # the same object seen from another world: world_target = truth @ world_source,
        # so the source's cells hold the object at truth @ (their own world position)
        truth = _matrix([7.0, -4.0, 3.0, 8.0, -5.0, 6.0, 1.1, 0.95, 1.05, 0.03, -0.02, 0.04],
                        np.zeros(3))
        source_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        source_affine[:3, 3] = -70.0
        source = Volume(self.blob((94, 94, 94), truth @ source_affine, (0.0, 0.0, 0.0), 3),
                        source_affine)
        matrix = register(source, target, coarse_mm=4.0)
        corners = np.array([[x, y, z, 1.0] for x in (-40.0, 40.0) for y in (-40.0, 40.0)
                            for z in (-40.0, 40.0)]).T
        error = np.linalg.norm((matrix @ corners - truth @ corners)[:3], axis=0)
        assert error.max() < 1.0, error  # mm at the object's corners
        assert overlap(source, target, matrix) > 0.97
        # the moved volume reads the target's values at the target's positions
        moved = Volume(source.data, matrix @ source.affine)
        for position in ((0.0, 0.0, 0.0), (10.0, -15.0, 5.0)):
            assert abs(moved.mean(position, 6.0) - target.mean(position, 6.0)) < 0.03
