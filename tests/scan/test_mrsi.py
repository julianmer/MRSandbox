####################################################################################################
#                                           test_mrsi.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: MRSI gate: the slice lattice lands where the affine says, one truth serves the grid,    #
#          spatial contrast emerges from the physics (metabolites inside, the lipid ring at the    #
#          boundary, silence outside), acquisition is seeded, and coils add their dimension.       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, CoilArray, Params, Press, Scanner, VirtualScan

QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_deg=0.0))


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                       Press(te1_s=0.011, te2_s=0.019), params=QUIET)


@pytest.fixture(scope="module")
def acquisition(scan):
    mrsi = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(8, 8), voxel_mm=20.0)
    return mrsi, mrsi.acquire(seed=7, npoints=128)


#**************************************************************************************************#
#                                         Class TestLattice                                        #
#**************************************************************************************************#
class TestLattice:

    #**************#
    #   geometry   #
    #**************#
    def test_grid_is_centred_and_the_affine_matches(self, scan):
        mrsi = scan.mrsi(center_mm=(10.0, -20.0, 5.0), matrix=(8, 8), voxel_mm=20.0)
        positions = np.array([[mrsi.position(i, j) for j in range(8)] for i in range(8)])
        assert np.allclose(positions.mean(axis=(0, 1)), (10.0, -20.0, 5.0))
        # the affine maps grid indices to world positions
        for i, j in ((0, 0), (3, 5), (7, 7)):
            world = mrsi.affine @ np.array([i, j, 0.0, 1.0])
            assert np.allclose(world[:3], mrsi.position(i, j))

    def test_acquisition_shape_and_affine(self, acquisition):
        mrsi, data = acquisition
        assert data.shape == (8, 8, 1, 128)
        assert np.allclose(np.asarray(data.voxToWorldMat), mrsi.affine)


#**************************************************************************************************#
#                                        Class TestContrast                                        #
#**************************************************************************************************#
class TestContrast:

    #**********************#
    #   spatial physics    #
    #**********************#
    def test_contrast_follows_the_anatomy(self, scan, acquisition):
        mrsi, data = acquisition
        volume = np.asarray(data[:])[:, :, 0]
        power = np.abs(volume).sum(axis=-1)
        tissue = np.array([
            [sum(scan.brain.fractions(mrsi.position(i, j), 20.0)[:2]) for j in range(8)]
            for i in range(8)
        ])
        lipid = np.array([
            [scan.brain.lipid_weight(mrsi.position(i, j), 20.0) for j in range(8)]
            for i in range(8)
        ])
        deep = power[(tissue > 0.7) & (lipid == 0.0)]
        ring = power[lipid > 0.1]
        outside = power[(tissue < 0.01) & (lipid == 0.0)]
        assert deep.size and ring.size and outside.size  # the grid spans all three zones
        assert ring.min() > deep.max()  # the lipid ring dominates, as in vivo
        assert deep.min() > 100.0 * (outside.max() + 1e-9)  # noise-free outside is silent

    #*****************#
    #   metabolites   #
    #*****************#
    def test_interior_voxels_carry_the_naa_peak(self, scan, acquisition):
        mrsi, data = acquisition
        volume = np.asarray(data[:])[:, :, 0]
        spec = np.abs(np.fft.fft(volume[4, 4]))
        freqs = np.fft.fftfreq(128, d=1.0 / 4000.0) % 4000.0
        ppm = freqs / (42.5774785 * 3.0)
        naa = spec[np.abs(ppm - 2.008) < 0.3].max()
        empty = spec[(ppm > 6.0) & (ppm < 8.0)].max()
        # at 128 points the broad MM components leak across the short spectrum, so the
        # empty region carries truncation ringing; three to one still proves the peak
        assert naa > 3.0 * empty

    #****************#
    #   provenance   #
    #****************#
    def test_one_truth_serves_the_grid(self, acquisition):
        _, data = acquisition
        header = data.hdr_ext.to_dict()
        assert "NAA" in header["SimulationGroundTruth"]["params"]["conc"]
        assert header["SimulationGrid"]["matrix"] == [8, 8]


#**************************************************************************************************#
#                                          Class TestSeeds                                         #
#**************************************************************************************************#
class TestSeeds:

    #*****************#
    #   determinism   #
    #*****************#
    def test_same_seed_reproduces_the_slice(self, scan):
        mrsi = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(3, 3), voxel_mm=20.0)
        a = np.asarray(mrsi.acquire(seed=3, npoints=64)[:])
        b = np.asarray(mrsi.acquire(seed=3, npoints=64)[:])
        np.testing.assert_array_equal(a, b)

    #***********#
    #   coils   #
    #***********#
    def test_coils_add_their_dimension(self, scan):
        coiled = VirtualScan(scan.brain, scan.scanner, scan.sequence, params=QUIET,
                             coils=CoilArray.ring(4))
        mrsi = coiled.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(3, 3), voxel_mm=20.0)
        data = mrsi.acquire(seed=3, npoints=64)
        assert data.shape == (3, 3, 1, 64, 4)
        assert data.dim_tags[0] == "DIM_COIL"


#**************************************************************************************************#
#                                       Class TestEncoding                                         #
#**************************************************************************************************#
#                                                                                                  #
# The k-space encoding imprints the real point spread function and preserves amplitudes.           #
#                                                                                                  #
#**************************************************************************************************#
class TestEncoding:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def point_brain(n=64, hot=(33, 33)):
        """A synthetic brain with tissue in exactly one 1 mm voxel, centred coordinates."""
        gm = np.zeros((n, n, 3), dtype=np.float32)
        gm[hot[0], hot[1], 1] = 1.0
        affine = np.eye(4)
        affine[:3, 3] = (-(n - 1) / 2.0, -(n - 1) / 2.0, -1.0)
        return Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm), affine=affine)

    def acquire_point(self, weighting):
        from mrsandbox import NoLipids, NoMM, NoWater

        brain = self.point_brain()
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())
        mrsi = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(8, 8), voxel_mm=4.0,
                         oversampling=2, weighting=weighting)
        data = mrsi.acquire(seed=3, npoints=32)
        return np.abs(np.asarray(data[:])[:, :, 0]).sum(axis=-1)

    #**************#
    #   the psf    #
    #**************#
    def test_a_point_source_bleeds_into_neighbours(self):
        power = self.acquire_point(weighting=None)
        centre = np.unravel_index(np.argmax(power), power.shape)
        assert centre == (4, 4)  # the hot voxel's acquired cell keeps the maximum
        neighbours = power[3, 4], power[5, 4], power[4, 3], power[4, 5]
        assert all(n > 0.01 * power[centre] for n in neighbours)  # Gibbs bleed exists

    def test_hamming_suppresses_the_far_sidelobes(self):
        plain = self.acquire_point(weighting=None)
        hamming = self.acquire_point(weighting="hamming")
        # away from the main lobe the weighted PSF must sit well below the plain one
        far_plain = plain[0, 4] + plain[4, 0] + plain[7, 4] + plain[4, 7]
        far_hamming = hamming[0, 4] + hamming[4, 0] + hamming[7, 4] + hamming[4, 7]
        assert far_hamming < 0.5 * far_plain
        # and the main lobe broadens: nearest neighbours gain relative to the centre
        assert (hamming[3, 4] / hamming[4, 4]) > (plain[3, 4] / plain[4, 4])

    #**************************#
    #   dc-centred weighting   #
    #**************************#
    def test_hamming_preserves_a_uniform_object_exactly(self):
        from mrsandbox import NoLipids, NoMM, NoWater

        n = 64
        gm = np.ones((n, n, 3), dtype=np.float32)
        affine = np.eye(4)
        affine[:3, 3] = (-(n - 1) / 2.0, -(n - 1) / 2.0, -1.0)
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=affine)
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(4, 4), voxel_mm=4.0)
        a = np.asarray(scan.mrsi(**kw).acquire(seed=3, npoints=16)[:])[:, :, 0]
        b = np.asarray(scan.mrsi(**kw, weighting="hamming").acquire(seed=3, npoints=16)[:])[:, :, 0]
        # a uniform object lives at DC alone, where the centred window is exactly one;
        # a window peaking between samples (numpy's) attenuates this by 4.6 percent
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6 * np.abs(a).max())

    #*******************#
    #   explicit mask   #
    #*******************#
    def test_explicit_mask_reproduces_the_named_ellipse(self):
        from mrsandbox import NoLipids, NoMM, NoWater

        brain = self.point_brain()
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())
        u = (np.arange(8) - 4) / 4.0
        mask = ((u[:, None] ** 2 + u[None, :] ** 2) <= 1.0 + 1e-9).astype(np.float32)
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(8, 8), voxel_mm=4.0, oversampling=2)
        named = scan.mrsi(**kw, sampling="elliptical").acquire(seed=3, npoints=32)
        custom = scan.mrsi(**kw, sampling=mask).acquire(seed=3, npoints=32)
        np.testing.assert_array_equal(np.asarray(custom[:]), np.asarray(named[:]))
        assert custom.hdr_ext.to_dict()["SimulationGrid"]["sampling"] == mask.tolist()

    #***************************#
    #   elliptical sampling     #
    #***************************#
    def test_elliptical_sampling_never_measures_the_corners(self):
        from mrsandbox import NoLipids, NoMM, NoWater

        brain = self.point_brain()
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())
        volumes = {}
        for sampling in (None, "elliptical"):
            mrsi = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(8, 8), voxel_mm=4.0,
                             oversampling=2, sampling=sampling)
            data = mrsi.acquire(seed=3, npoints=32)
            assert data.hdr_ext.to_dict()["SimulationGrid"]["sampling"] == sampling
            volumes[sampling] = np.asarray(data[:])[:, :, 0]
        k_full = np.fft.fftshift(np.fft.fft2(volumes[None], axes=(0, 1)), axes=(0, 1))
        k_ell = np.fft.fftshift(np.fft.fft2(volumes["elliptical"], axes=(0, 1)), axes=(0, 1))
        scale = np.abs(k_full[4, 4]).max()
        assert np.abs(k_full[0, 0]).max() > 1e-3 * scale  # full sampling measures the corner
        assert np.abs(k_ell[0, 0]).max() < 1e-6 * scale  # elliptical never does
        assert np.abs(k_ell[0, 4]).max() > 1e-3 * scale  # the axis edge stays measured

    #********************#
    #   normalization    #
    #********************#
    def test_ideal_and_encoded_agree_deep_inside_the_brain(self, scan):
        ideal = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=16.0)
        encoded = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=16.0,
                            oversampling=2)
        elliptical = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=16.0,
                               oversampling=2, sampling="elliptical")
        a = np.abs(np.asarray(ideal.acquire(seed=3, npoints=32)[:])[:, :, 0]).sum(axis=-1)
        b = np.abs(np.asarray(encoded.acquire(seed=3, npoints=32)[:])[:, :, 0]).sum(axis=-1)
        c = np.abs(np.asarray(elliptical.acquire(seed=3, npoints=32)[:])[:, :, 0]).sum(axis=-1)
        # a central 4x4 patch deep in the brain is nearly uniform, so encoding at twice
        # the source resolution must reproduce the ideal amplitudes closely; dropping
        # the k corners barely moves a smooth object (Maudsley's argument)
        assert np.abs(a - b).max() < 0.15 * a.max()
        assert np.abs(a - c).max() < 0.15 * a.max()
