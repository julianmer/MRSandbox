####################################################################################################
#                                          test_coils.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Coil array gate: the loop field reduces to the on-axis closed form, sensitivities       #
#          fall off with distance, the Roemer combination has unit gain and beats both the         #
#          sum-of-squares and the best single coil in SNR, generated noise carries the             #
#          requested covariance, and a multi-channel acquisition combines back to the single       #
#          channel truth.                                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, CoilArray, Loop, Params, Press, Scanner, VirtualScan

VOXEL = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def array():
    from mrsandbox import MappedCoilArray

    return MappedCoilArray.load("bigbrain_mr")


@pytest.fixture(scope="module")
def pieces():
    return Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019)


def synthetic_signal(b, npoints=512):
    s = np.exp(1j * 2.0 * np.pi * 0.01 * np.arange(npoints)).astype(np.complex64)
    return s, b[:, None] * s[None, :]


#**************************************************************************************************#
#                                           Class TestLoop                                         #
#**************************************************************************************************#
class TestLoop:

    #**********************#
    #   on axis analytic   #
    #**********************#
    def test_field_reduces_to_the_on_axis_closed_form(self):
        loop = Loop(40.0, (0.0, 0.0, 0.0))
        for z in (0.0, 30.0, 80.0):
            expected = 40.0 ** 2 / (2.0 * (40.0 ** 2 + z ** 2) ** 1.5)
            assert abs(loop.field((0.0, 0.0, z))[2] - expected) < 1e-12

    #**************#
    #   falloff    #
    #**************#
    def test_sensitivity_falls_off_with_distance(self):
        array = CoilArray.ring(8, ring_radius_mm=110.0)
        near_first = np.abs(array.sensitivities((80.0, 0.0, 0.0)))
        assert near_first[0] == near_first.max()  # loop 0 sits at +x
        assert near_first[0] > 5.0 * near_first[4]  # the opposite loop sees far less


#**************************************************************************************************#
#                                       Class TestCombination                                      #
#**************************************************************************************************#
class TestCombination:

    #*****************#
    #   unit gain     #
    #*****************#
    def test_noise_free_combination_returns_the_signal(self):
        array = CoilArray.ring(8)
        position = (30.0, -20.0, 5.0)
        s, clean = synthetic_signal(array.sensitivities(position))
        assert np.abs(array.combine(clean, position) - s).max() < 1e-6

    #********************#
    #   snr ordering     #
    #********************#
    def test_roemer_beats_sum_of_squares_and_the_best_coil(self):
        rng = np.random.default_rng(3)
        r = 0.2 * np.ones((8, 8)) + 0.8 * np.eye(8)  # correlated coils
        array = CoilArray(CoilArray.ring(8).loops, noise_resistance=r)
        position = (30.0, -20.0, 5.0)
        b = array.sensitivities(position)
        s, clean = synthetic_signal(b, npoints=4096)
        noisy = clean + array.noise(rng, 0.002, 4096)

        def snr(signal, reference):
            return np.abs(reference).mean() / np.abs(signal - reference).std()

        combined = snr(array.combine(noisy, position), s)
        squares = snr(array.sum_of_squares(noisy) / np.sqrt(np.real(np.vdot(b, b))), np.abs(s))
        single = max(snr(noisy[k] / b[k], s) for k in range(8))
        assert combined > squares
        assert combined > single
        # the analytic gain over the best coil is sqrt(b^H R^-1 b) / |b_best| / sqrt(R^-1_kk)-ish;
        # gate the simple lower bound instead of the exact expression
        assert combined > 1.2 * single

    #**********************#
    #   noise covariance   #
    #**********************#
    def test_generated_noise_matches_the_requested_covariance(self):
        r = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
        array = CoilArray(CoilArray.ring(4).loops, noise_resistance=r)
        noise = array.noise(np.random.default_rng(5), 2.0, 40000)
        covariance = (noise @ noise.conj().T).real / (2.0 * 40000)
        assert np.abs(covariance - 4.0 * r).max() < 0.15


#**************************************************************************************************#
#                                       Class TestAcquisition                                      #
#**************************************************************************************************#
class TestAcquisition:
    pytestmark = pytest.mark.slow  # full-scan renders, skipped on the tensorflow CI leg

    #****************#
    #   dimensions   #
    #****************#
    def test_acquire_carries_the_coil_dimension(self, pieces):
        brain, scanner, sequence = pieces
        scan = VirtualScan(brain, scanner, sequence, coils=CoilArray.ring(8))
        data = scan.voxel(**VOXEL).acquire(seed=7, npoints=512)
        assert data.shape == (1, 1, 1, 512, 8)
        assert data.dim_tags[0] == "DIM_COIL"

    #*******************#
    #   round trip      #
    #*******************#
    def test_combined_channels_match_the_single_channel_truth(self, pieces):
        brain, scanner, sequence = pieces
        quiet = Params(noise=dict(sigma=0.0))
        array = CoilArray.ring(8)
        plain = VirtualScan(brain, scanner, sequence, params=quiet)
        coiled = VirtualScan(brain, scanner, sequence, params=quiet, coils=array)
        reference = np.asarray(plain.voxel(**VOXEL).acquire(seed=7, npoints=512)[:])[0, 0, 0]
        channels = np.asarray(coiled.voxel(**VOXEL).acquire(seed=7, npoints=512)[:])[0, 0, 0].T
        combined = array.combine(channels, VOXEL["position_mm"])
        assert np.abs(combined - reference).max() < 1e-4 * np.abs(reference).max()


#**************************************************************************************************#
#                                      Class TestMappedArray                                       #
#**************************************************************************************************#
#                                                                                                  #
# The measured 32-channel maps serve positions and combine through the same Roemer machinery.      #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not __import__("mrsandbox").MappedCoilArray.cached(),
                    reason="coil map cache not derived; MappedCoilArray.load() builds it once")
class TestMappedArray:


    #**************#
    #   the maps   #
    #**************#
    def test_maps_serve_positions(self, array):
        assert len(array) == 32
        centre = array.sensitivities((0.0, 0.0, 0.0))
        assert centre.shape == (32,) and np.abs(centre).max() > 0.0
        assert np.ptp(np.angle(centre[np.abs(centre) > 1e-6])) > 1.0  # measured phases vary
        elsewhere = array.sensitivities((30.0, -20.0, 10.0))
        assert np.abs(centre - elsewhere).max() > 0.0
        assert np.abs(array.sensitivities((500.0, 0.0, 0.0))).max() == 0.0

    #*****************#
    #   combination   #
    #*****************#
    def test_inherited_combination_has_unit_gain(self, array):
        position = (10.0, -20.0, 5.0)
        s, clean = synthetic_signal(array.sensitivities(position))
        assert np.abs(array.combine(clean, position) - s).max() < 1e-5

    #***************#
    #   round trip  #
    #***************#
    def test_scan_round_trip_with_measured_coils(self, pieces, array):
        brain, scanner, sequence = pieces
        quiet = Params(noise=dict(sigma=0.0))
        plain = VirtualScan(brain, scanner, sequence, params=quiet)
        coiled = VirtualScan(brain, scanner, sequence, params=quiet, coils=array)
        reference = np.asarray(plain.voxel(**VOXEL).acquire(seed=7, npoints=256)[:])[0, 0, 0]
        data = coiled.voxel(**VOXEL).acquire(seed=7, npoints=256)
        assert data.shape == (1, 1, 1, 256, 32)
        channels = np.asarray(data[:])[0, 0, 0].T
        combined = array.combine(channels, VOXEL["position_mm"])
        assert np.abs(combined - reference).max() < 1e-3 * np.abs(reference).max()
