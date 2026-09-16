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
import importlib.util

import numpy as np
import pytest

from mrsandbox import Brain, CoilArray, Params, PerTissue, Press, Scanner, VirtualScan

# full-scan integration renders: the tensorflow CI leg skips these (-m 'not render')
pytestmark = pytest.mark.render

QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))


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

    #**************#
    #   the slab   #
    #**************#
    def test_source_voxels_keep_the_slab_thickness(self, scan):
        # oversampling refines in plane only: every source voxel spans the full slab
        fine = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=8.0,
                         oversampling=2)
        assert fine.voxels[0][0][0].size_mm == (4.0, 4.0, 8.0)
        thick = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=8.0,
                          thickness_mm=12.0)
        assert thick.voxels[0][0][0].size_mm == (8.0, 8.0, 12.0)
        # a profile splits the slab into that many sub-slabs, centres spanning it
        shaped = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(4, 4), voxel_mm=8.0,
                           thickness_mm=12.0, profile=np.ones(3))
        cell = shaped.voxels[0][0]
        assert [v.size_mm for v in cell] == [(8.0, 8.0, 4.0)] * 3
        assert [v.position_mm[2] for v in cell] == [1.0, 5.0, 9.0]


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
        """A synthetic brain with tissue in exactly one 1 mm voxel, placed so it straddles
        two 2 mm source cells (its centre at 2.0 mm, spanning 1.5 to 2.5): an object off
        the acquired grid, which the k-space truncation therefore rings around. A cell
        aligned with the acquired grid would be its own point spread function."""
        gm = np.zeros((n, n, 3), dtype=np.float32)
        gm[hot[0], hot[1], 1] = 1.0
        affine = np.eye(4)
        affine[:3, 3] = (-(n / 2.0 - 1.0), -(n / 2.0 - 1.0), -1.0)
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


#**************************************************************************************************#
#                                      Class TestSliceProfile                                      #
#**************************************************************************************************#
#                                                                                                  #
# The z profile weights sub-slab signals: an all-ones profile is the box, a soft profile keeps     #
# its mean excitation, and the weights select what the slab sees along z.                          #
#                                                                                                  #
#**************************************************************************************************#
class TestSliceProfile:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def synthetic_scan(gm):
        from mrsandbox import NoLipids, NoMM, NoWater

        n = gm.shape[0]
        affine = np.eye(4)
        affine[:3, 3] = (-(n - 1) / 2.0, -(n - 1) / 2.0, -(gm.shape[2] - 1) / 2.0)
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=affine)
        return VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())

    #***********************#
    #   box and mean gate   #
    #***********************#
    def test_flat_profile_is_the_box_and_gaussian_keeps_its_mean(self):
        scan = self.synthetic_scan(np.ones((64, 64, 9), dtype=np.float32))
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2), voxel_mm=4.0)
        box = np.asarray(scan.mrsi(**kw).acquire(seed=3, npoints=32)[:])[:, :, 0]
        flat = np.asarray(scan.mrsi(**kw, profile=np.ones(3)).acquire(seed=3, npoints=32)[:])[:, :, 0]
        np.testing.assert_allclose(flat, box, rtol=1e-5, atol=1e-6 * np.abs(box).max())
        soft = scan.mrsi(**kw, profile="gaussian")
        data = soft.acquire(seed=3, npoints=32)
        gaussian = np.asarray(data[:])[:, :, 0]
        # every sub-slab of a uniform object is identical, so the soft slab keeps
        # exactly the mean excitation of its profile: the signal a soft pulse loses
        np.testing.assert_allclose(gaussian, soft.profile.mean() * box, rtol=1e-4,
                                   atol=1e-6 * np.abs(box).max())
        assert data.hdr_ext.to_dict()["SimulationGrid"]["profile"] == soft.profile.tolist()

    #******************#
    #   z selection    #
    #******************#
    def test_profile_selects_what_the_slab_sees_along_z(self):
        gm = np.zeros((16, 16, 9), dtype=np.float32)
        gm[:, :, 6:] = 1.0  # tissue only above the slab centre
        scan = self.synthetic_scan(gm)
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2), voxel_mm=4.0, thickness_mm=6.0)
        bottom = np.asarray(scan.mrsi(**kw, profile=[1.0, 0.0, 0.0])
                            .acquire(seed=3, npoints=32)[:])[:, :, 0]
        top = np.asarray(scan.mrsi(**kw, profile=[0.0, 0.0, 1.0])
                         .acquire(seed=3, npoints=32)[:])[:, :, 0]
        assert np.abs(top).max() > 0.0  # the upper sub-slab reaches the tissue
        assert np.abs(bottom).max() < 1e-12 * np.abs(top).max()  # the lower never does


#**************************************************************************************************#
#                                 Class TestAcquisitionWeighting                                   #
#**************************************************************************************************#
#                                                                                                  #
# Acquisition-weighted averaging: quantized per-encode counts weight the signal, weak encodes      #
# drop out, and averaging buys the square root noise gain.                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestAcquisitionWeighting:

    #**********************#
    #   realized counts    #
    #**********************#
    def test_realized_counts_weight_the_signal_and_the_header(self):
        scan = TestSliceProfile.synthetic_scan(np.ones((64, 64, 9), dtype=np.float32))
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(8, 8), voxel_mm=2.0)
        plain = scan.mrsi(**kw)
        weighted = scan.mrsi(**kw, weighting="hamming", averages=8)
        for bad in ("four", 0, 2.5):  # non-numeric included: always the stated error
            with pytest.raises(ValueError, match="positive integer"):
                scan.mrsi(**kw, averages=bad)
        counts = weighted._averages_map()
        assert counts[4, 4] == 8  # the k-space centre gets the full count
        assert counts[0, 0] == 0  # the hamming corner rounds to unmeasured
        np.testing.assert_array_equal(counts, np.rint(8.0 * weighted._weight_map()).astype(int))
        a = np.asarray(plain.acquire(seed=3, npoints=16)[:])[:, :, 0]
        acquired = weighted.acquire(seed=3, npoints=16)
        b = np.asarray(acquired[:])[:, :, 0]
        # a uniform object lives at DC alone, where the count normalization is one
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6 * np.abs(a).max())
        grid = acquired.hdr_ext.to_dict()["SimulationGrid"]
        assert grid["averages"] == 8
        assert grid["transients"] == int(counts.sum()) == int(np.sum(counts))

    #*******************#
    #   the snr gain    #
    #*******************#
    def test_averaging_buys_the_square_root_noise_gain(self):
        from mrsandbox import NoLipids, NoMM, NoWater

        gm = np.zeros((16, 16, 3), dtype=np.float32)
        affine = np.eye(4)
        affine[:3, 3] = (-7.5, -7.5, -1.0)
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=affine)
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=Params(noise=dict(sigma=2.0)), water=NoWater(),
                           lipids=NoLipids(), mm=NoMM())
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2), voxel_mm=4.0)
        single = np.asarray(scan.mrsi(**kw, averages=1).acquire(seed=3, npoints=256)[:])
        four = np.asarray(scan.mrsi(**kw, averages=4).acquire(seed=5, npoints=256)[:])
        assert abs(np.std(single.real) - 2.0) < 0.2  # one transient keeps sigma
        assert abs(np.std(four.real) - 1.0) < 0.1  # four transients halve it


#**************************************************************************************************#
#                                        Class TestSlab3D                                          #
#**************************************************************************************************#
#                                                                                                  #
# The 3D slab: partitions resolve z, the sampling ellipsoid and 3D weighting extend over kz, and   #
# the profile weights the partitions.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestSlab3D:

    #*****************#
    #   z resolved    #
    #*****************#
    def test_partitions_resolve_the_z_axis(self):
        gm = np.zeros((16, 16, 9), dtype=np.float32)
        gm[:, :, 6:] = 1.0  # tissue only in the upper third of the maps
        scan = TestSliceProfile.synthetic_scan(gm)
        mrsi = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2, 3), voxel_mm=4.0,
                         thickness_mm=2.0)
        data = mrsi.acquire(seed=3, npoints=16)
        assert data.shape == (2, 2, 3, 16)
        assert data.hdr_ext.to_dict()["SimulationGrid"]["matrix"] == [2, 2, 3]
        assert mrsi.position(0, 0, 0)[2] == -2.0 and mrsi.position(0, 0, 2)[2] == 2.0
        assert np.allclose(np.diag(mrsi.affine)[:3], (4.0, 4.0, 2.0))
        power = np.abs(np.asarray(data[:])).sum(axis=(0, 1, 3))
        assert power[2] > 0.0  # the top partition reaches the tissue
        assert power[0] < 1e-12 * power[2]  # the bottom partition never does

    #*******************#
    #   the ellipsoid   #
    #*******************#
    def test_spherical_sampling_and_3d_weighting(self):
        scan = TestSliceProfile.synthetic_scan(np.ones((64, 64, 9), dtype=np.float32))
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2, 4), voxel_mm=4.0,
                  thickness_mm=1.0)
        mask = scan.mrsi(**kw, sampling="elliptical")._sampling_mask()
        assert mask.shape == (2, 2, 4)
        assert mask[0, 0, 0] == 0.0  # the ellipsoid corner is never measured
        assert mask[0, 1, 2] == 1.0  # the axis edge stays measured
        # a uniform object lives at DC alone, where every 3D map normalizes to one
        plain = np.asarray(scan.mrsi(**kw).acquire(seed=3, npoints=8)[:])
        soft = np.asarray(scan.mrsi(**kw, weighting="hamming", averages=8)
                          .acquire(seed=3, npoints=8)[:])
        np.testing.assert_allclose(soft, plain, rtol=1e-4, atol=1e-6 * np.abs(plain).max())

    #**********************#
    #   partition profile  #
    #**********************#
    def test_profile_weights_the_partitions(self):
        scan = TestSliceProfile.synthetic_scan(np.ones((64, 64, 9), dtype=np.float32))
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2, 3), voxel_mm=4.0,
                  thickness_mm=1.0)
        box = np.asarray(scan.mrsi(**kw).acquire(seed=3, npoints=8)[:])
        shaped = scan.mrsi(**kw, profile="gaussian")
        assert shaped.profile.size == 3  # sampled at the partition centres
        data = np.asarray(shaped.acquire(seed=3, npoints=8)[:])
        for l in range(3):  # a uniform object realizes the profile weight exactly
            np.testing.assert_allclose(data[:, :, l], shaped.profile[l] * box[:, :, l],
                                       rtol=1e-4, atol=1e-6 * np.abs(box).max())
        with pytest.raises(ValueError, match="per partition"):
            scan.mrsi(**kw, profile=np.ones(5))


#**************************************************************************************************#
#                                       Class TestTrajectory                                       #
#**************************************************************************************************#
#                                                                                                  #
# The EPSI readout: k lines sampled at their visit times, so the flyback clock shift is exact      #
# and the bidirectional odd-even ghost emerges at half the spectral width.                         #
#                                                                                                  #
#**************************************************************************************************#
class TestTrajectory:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def singlet_scan():
        from mrsandbox import NoLipids, NoMM, NoWater

        gm = np.ones((64, 64, 9), dtype=np.float32)
        affine = np.eye(4)
        affine[:3, 3] = (-31.5, -31.5, -4.0)
        brain = Brain.from_maps(gm=gm, wm=np.zeros_like(gm), csf=np.zeros_like(gm),
                                affine=affine)
        conc = {name: 0.0 for name in VirtualScan.default_params().conc}
        conc["sIns"] = 5.0  # the scyllo-inositol singlet: one exponential line
        params = QUIET | Params(conc=conc)
        from mrsandbox import NoDownfield

        return VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=params, water=NoWater(), lipids=NoLipids(), mm=NoMM(),
                           downfield=NoDownfield())

    def acquired(self, scan, npoints=64, **kw):
        data = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(4, 4), voxel_mm=4.0, **kw)
        return data, np.asarray(data.acquire(seed=3, npoints=npoints)[:])[:, :, 0]

    #********************#
    #   the exact clock  #
    #********************#
    def test_flyback_shifts_the_sampling_clock_exactly(self):
        scan = self.singlet_scan()
        _, csi = self.acquired(scan)
        mrsi, epsi = self.acquired(scan, trajectory="epsi")
        # a uniform single-line object lives at kx DC, visited mid sweep: every voxel
        # FID is the CSI one advanced by that visit time, one constant complex factor
        # for a single exponential line
        ratio = epsi[2, 2] / csi[2, 2]
        assert np.std(ratio) < 1e-3 * np.abs(np.mean(ratio))
        assert abs(np.angle(np.mean(ratio))) > 0.05  # the line is genuinely off DC
        grid = mrsi.acquire(seed=3, npoints=8).hdr_ext.to_dict()["SimulationGrid"]
        assert grid["trajectory"] == "epsi"
        assert grid["visit_order"] == [0, 1, 2, 3] == grid["visit_order_odd"]

    #****************#
    #   the ghost    #
    #****************#
    def test_bidirectional_ghosts_at_half_the_bandwidth(self):
        scan = self.singlet_scan()
        _, flyback = self.acquired(scan, trajectory="epsi")
        _, bidirectional = self.acquired(scan, trajectory="epsi-bidirectional")
        fly = np.abs(np.fft.fft(flyback[2, 2]))
        bid = np.abs(np.fft.fft(bidirectional[2, 2]))
        peak = int(np.argmax(fly))
        ghost = (peak + 32) % 64  # half the spectral width away
        assert bid[peak] > bid[ghost]  # the main line still dominates
        assert bid[ghost] > 10.0 * fly[ghost]  # the odd-even ghost is bidirectional only

    #*******************#
    #   finite flyback  #
    #*******************#
    def test_finite_flyback_keeps_the_clock_identity(self):
        # the visits compress into dwell - dead: the uniform singlet still lives at
        # kx DC with one constant advance, and the realized dead time is quantized
        # to fine samples and recorded
        scan = self.singlet_scan()
        _, csi = self.acquired(scan)
        mrsi, epsi = self.acquired(scan, trajectory="epsi", dead_time_s=0.5e-4)
        ratio = epsi[2, 2] / csi[2, 2]
        assert np.std(ratio) < 1e-3 * np.abs(np.mean(ratio))
        grid = mrsi.acquire(seed=3, npoints=8).hdr_ext.to_dict()["SimulationGrid"]
        # dwell 2.5e-4, nx 4: n_dead = round(0.5e-4 * 4 / 2e-4) = 1 fine sample
        assert abs(grid["dead_time_s"] - 2.5e-4 / 5.0) < 1e-12

    def test_dead_time_compresses_the_sweep_and_shrinks_the_ghost(self):
        # the spectral dwell is fixed, so a turnaround dead time compresses the nx
        # visits into dwell - dead: the odd-even visit offsets and with them the
        # ghost shrink, while the ghost still dwarfs the flyback residue
        scan = self.singlet_scan()
        _, flyback = self.acquired(scan, trajectory="epsi")
        _, tight = self.acquired(scan, trajectory="epsi-bidirectional")
        _, loose = self.acquired(scan, trajectory="epsi-bidirectional", dead_time_s=0.5e-4)
        peak = int(np.argmax(np.abs(np.fft.fft(tight[2, 2]))))
        ghost = (peak + 32) % 64
        assert np.abs(np.fft.fft(loose[2, 2]))[ghost] < np.abs(np.fft.fft(tight[2, 2]))[ghost]
        assert np.abs(np.fft.fft(loose[2, 2]))[ghost] > 5.0 * np.abs(np.fft.fft(flyback[2, 2]))[ghost]

    #*****************#
    #   slab sweeps   #
    #*****************#
    def test_slab_partitions_resolve_under_epsi(self):
        gm = np.zeros((16, 16, 9), dtype=np.float32)
        gm[:, :, 6:] = 1.0  # tissue only in the upper third of the maps
        scan = TestSliceProfile.synthetic_scan(gm)
        mrsi = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(2, 2, 3), voxel_mm=4.0,
                         thickness_mm=2.0, trajectory="epsi")
        data = mrsi.acquire(seed=3, npoints=16)
        assert data.shape == (2, 2, 3, 16)
        power = np.abs(np.asarray(data[:])).sum(axis=(0, 1, 3))
        assert power[2] > 0.0  # the top partition reaches the tissue
        # kz still resolves under EPSI: only float32 FFT round-trip noise leaks (the
        # CSI slab takes its identity shortcut and lands at exactly zero instead)
        assert power[0] < 1e-6 * power[2]

    #******************#
    #   composition    #
    #******************#
    def test_trajectory_rejects_the_csi_knobs(self):
        scan = self.singlet_scan()
        kw = dict(center_mm=(0.0, 0.0, 0.0), matrix=(4, 4), voxel_mm=4.0)
        for bad in (dict(weighting="hamming"), dict(oversampling=2), dict(averages=4),
                    dict(sampling="elliptical")):
            with pytest.raises(ValueError, match="conventional CSI"):
                scan.mrsi(**kw, trajectory="epsi", **bad)
        with pytest.raises(ValueError, match="needs a trajectory"):
            scan.mrsi(**kw, dead_time_s=1e-5)
        with pytest.raises(ValueError, match="below the spectral dwell"):
            scan.mrsi(**kw, trajectory="epsi", dead_time_s=1.0).acquire(seed=3, npoints=8)


#**************************************************************************************************#
#                                       Class TestNonuniform                                       #
#**************************************************************************************************#
#                                                                                                  #
# Augmentrum trajectories: rosettes and rings read the sources at their visit times through the    #
# gridding NUFFT, the reconstruction keeps a uniform object and places a point source.             #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(importlib.util.find_spec("augmentrum") is None,
                    reason="Augmentrum not installed (mrsandbox[augment])")
class TestNonuniform:

    #*****************#
    #   uniform gate  #
    #*****************#
    def test_rosette_keeps_a_uniform_singlet_object(self):
        # every rosette shot starts at DC, so a uniform single-line object is read at
        # slot zero of every period: the reconstruction is the CSI slice up to the
        # gridding of the padded field of view, uniform across the grid
        scan = TestTrajectory.singlet_scan()
        _, csi = TestTrajectory().acquired(scan)
        mrsi, rosette = TestTrajectory().acquired(
            scan, trajectory="rosette_2d",
            trajectory_params=dict(n_shots=4, samples_per_shot=64))
        ratio = np.abs(rosette).sum(axis=-1) / np.abs(csi).sum(axis=-1)
        assert abs(ratio.mean() - 1.0) < 0.1  # the stated scalar gain normalization
        assert ratio.std() < 0.1 * ratio.mean()
        # the line sits where CSI puts it
        for spectrum in (np.abs(np.fft.fft(csi[2, 2])), np.abs(np.fft.fft(rosette[2, 2]))):
            assert np.argmax(spectrum) == np.argmax(np.abs(np.fft.fft(csi[1, 1])))
        grid = mrsi.acquire(seed=3, npoints=8).hdr_ext.to_dict()["SimulationGrid"]
        assert grid["trajectory"] == "rosette_2d" and grid["shots"] == 4
        assert grid["samples_per_shot"] == 64 and "visit_order" not in grid
        assert grid["trajectory_params"] == {"n_shots": 4, "samples_per_shot": 64}

    #*******************#
    #   point source    #
    #*******************#
    def test_rings_place_a_point_source(self):
        from mrsandbox import NoLipids, NoMM, NoWater

        brain = TestEncoding.point_brain()  # the hot cell lies in source cell (4, 4)
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=QUIET, water=NoWater(), lipids=NoLipids(), mm=NoMM())
        mrsi = scan.mrsi(center_mm=(0.0, 0.0, 0.0), matrix=(8, 8), voxel_mm=4.0,
                         trajectory="concentric_rings_2d",
                         trajectory_params=dict(n_rings=4, samples_per_ring=48))
        power = np.abs(np.asarray(mrsi.acquire(seed=3, npoints=32)[:])[:, :, 0]).sum(axis=-1)
        assert np.unravel_index(np.argmax(power), power.shape) == (4, 4)
        assert power[4, 4] > 3.0 * np.sort(power.ravel())[-2]  # the peak stands clear

    #*****************#
    #   the guards    #
    #*****************#
    def test_unequal_shots_are_refused(self):
        scan = TestTrajectory.singlet_scan()
        with pytest.raises(ValueError, match="same sample count"):
            scan.mrsi(matrix=(4, 4), voxel_mm=4.0, trajectory="concentric_rings_2d")
        with pytest.raises(ValueError, match="trajectory_params"):
            scan.mrsi(matrix=(4, 4), voxel_mm=4.0, trajectory="epsi",
                      trajectory_params=dict(n_shots=2))
        with pytest.raises(ValueError, match="Unknown trajectory"):
            scan.mrsi(matrix=(4, 4), voxel_mm=4.0, trajectory="helix_2d")


#**************************************************************************************************#
#                                         Class TestBasis                                          #
#**************************************************************************************************#
#                                                                                                  #
# The per-truth metabolite basis reproduces the per-voxel render exactly: what MRSI relies on.     #
#                                                                                                  #
#**************************************************************************************************#
class TestBasis:

    #*****************#
    #   the identity  #
    #*****************#
    def test_basis_render_equals_the_per_voxel_render(self):
        # every source cell of a lattice: weights @ basis plus the other sources equals
        # the plain per-voxel render, quiet and with the drawn echo and eddy phase
        brain = Brain.load("icbm152")
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        for params in (QUIET, Params(noise=dict(sigma=0.0))):
            scan = VirtualScan(brain, *pieces, params=params)
            mrsi = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(3, 3), voxel_mm=20.0)
            truth = scan.params.sample(seed=5)
            for row in mrsi.voxels:
                for cell in row:
                    voxel = cell[0]
                    _, plain = voxel._clean_fid(truth, 256, 1.0 / 4000.0)
                    basis = voxel.metabolite_basis(truth, 256, 1.0 / 4000.0)
                    _, formed = voxel._clean_fid(truth, 256, 1.0 / 4000.0, basis=basis)
                    assert basis.shape == (len(truth.conc), 256)
                    np.testing.assert_allclose(formed, plain, atol=1e-4 * np.abs(plain).max())
            # one basis per truth: the cache serves every cell of the lattice
            assert len(scan._basis[1]) == 1

    def test_basis_frame_follows_a_per_tissue_t1(self):
        # cells of equal T2 but different resolved T1 (a per-tissue leaf, b1 off
        # nominal so recovery reaches the echo) must not share one basis
        brain = Brain.load("icbm152")
        params = QUIET | Params(t1=dict(NAA=PerTissue(gm=0.3, wm=3.0)), b1_scale=0.8)
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019),
                           params=params)
        mrsi = scan.mrsi(center_mm=(0.0, -20.0, 5.0), matrix=(3, 3), voxel_mm=20.0)
        truth = scan.params.sample(seed=5)
        for row in mrsi.voxels:
            for cell in row:
                voxel = cell[0]
                _, plain = voxel._clean_fid(truth, 256, 1.0 / 4000.0)
                basis = voxel.metabolite_basis(truth, 256, 1.0 / 4000.0)
                _, formed = voxel._clean_fid(truth, 256, 1.0 / 4000.0, basis=basis)
                np.testing.assert_allclose(formed, plain, atol=1e-4 * np.abs(plain).max())
        assert len(scan._basis[1]) > 1
