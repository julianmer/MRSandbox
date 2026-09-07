####################################################################################################
#                                        test_generator.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Training-mode gate: the stream is seeded and reproducible, shapes and targets line up   #
#          with the params tree, the noise leaf controls the noise floor, and consecutive batches  #
#          are fresh draws.                                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, Params, Press, Scanner, VirtualScan

VOXEL = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)


#*****************#
#   shared scan   #
#*****************#
@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))


#**************************************************************************************************#
#                                       Class TestGenerator                                        #
#**************************************************************************************************#
class TestGenerator:

    #************************#
    #   shapes and targets   #
    #************************#
    def test_shapes_dtypes_and_target_order(self, scan):
        gen = scan.generator(**VOXEL, batch_size=4, seed=1, npoints=512)
        x, y = next(gen)
        assert x.shape == (4, 2, 512) and x.dtype == np.float32
        assert y.shape == (4, len(scan.params.conc)) and y.dtype == np.float32
        assert gen.metabolites == tuple(sorted(scan.params.conc))
        column = gen.metabolites.index("NAA")
        assert all(7.5 <= value < 12.0 for value in y[:, column])
        assert len(gen.truths) == 4

    #*************#
    #   seeding   #
    #*************#
    def test_stream_is_seeded_and_fresh(self, scan):
        a1, b1 = next(scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256))
        a2, b2 = next(scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256))
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
        gen = scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256)
        first, second = next(gen), next(gen)
        assert np.abs(first[0] - second[0]).max() > 0.0
        other, _ = next(scan.generator(**VOXEL, batch_size=3, seed=8, npoints=256))
        assert np.abs(a1 - other).max() > 0.0

    #*******************#
    #   noise control   #
    #*******************#
    def test_noise_leaf_controls_the_floor(self, scan):
        quiet_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                 params=Params(noise=dict(sigma=0.0)))
        loud_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                params=Params(noise=dict(sigma=2.0)))
        quiet, _ = next(quiet_scan.generator(**VOXEL, batch_size=1, seed=5, npoints=1024))
        loud, _ = next(loud_scan.generator(**VOXEL, batch_size=1, seed=5, npoints=1024))
        # same seed and identical draw order: the difference is exactly the added noise,
        # whose real and imaginary channels each carry the sampled sigma
        noise = loud - quiet
        assert np.abs(noise).max() > 0.0
        assert abs(np.std(noise) - 2.0) < 0.2
