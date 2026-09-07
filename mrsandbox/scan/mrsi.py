####################################################################################################
#                                             mrsi.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: The spatial dimension: an MRSI slice as a lattice of voxels over the virtual brain.     #
#          One realized truth serves the whole acquisition; spatial contrast emerges from the      #
#          physics (tissue fractions, the lipid ring at the boundary, per-voxel B0 dephasing,      #
#          per-position coil sensitivities), and the result is NIfTI-MRS with real spatial         #
#          dimensions and a world affine.                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np


#**************************************************************************************************#
#                                            Class MRSI                                            #
#**************************************************************************************************#
#                                                                                                  #
# One MRSI slice acquisition: a grid of voxels sharing one realized parameter tree.                #
#                                                                                                  #
#**************************************************************************************************#
class MRSI:
    """One MRSI slice acquisition: a grid of voxels sharing one realized parameter tree.

    The lattice lies in the axial plane at the given centre; every grid point is a
    full Voxel, so tissue weighting, water, lipids, macromolecules, B0 dephasing, and
    coil reception all apply per position. With `oversampling` above one, the object is
    simulated on a finer source grid and phase encoded through k-space: the acquisition
    keeps only the central matrix of k, which imprints the real point spread function
    (Gibbs ringing, the lipid ring bleeding inward).

    The k-space conventions, exactly: after the spatial FFT the acquired grid holds the
    centred k indices p = i - n//2 per axis, so DC sits at index n//2 and an even
    matrix carries its single Nyquist row at p = -n/2. `weighting="hamming"` multiplies
    k by the DC-centred window w(p) = 0.54 + 0.46 cos(pi p / (n/2)) per axis (w(0) = 1,
    so a uniform object keeps its amplitude; w = 0.08 at the Nyquist edge). Note that
    numpy's window peaks between samples on an even grid and would both attenuate DC
    and shift the point spread function by half a k step. `sampling="elliptical"`
    measures only the encodes with (2 px / nx)^2 + (2 py / ny)^2 <= 1, edge inclusive,
    the standard reduced phase encoding (Maudsley et al. 1994): the corners (about
    1 - pi/4, one fifth of the encodes) are never measured and stay zero, buying
    acquisition time for a slightly broader response function. `sampling` also accepts
    an explicit (nx, ny) array multiplying k, ordered like the acquired grid with DC at
    (nx//2, ny//2): booleans give an arbitrary measured-point mask, floats an arbitrary
    acquisition density. Amplitudes are normalized so full sampling of a uniform object
    is exactly the identity.

    Stated limits: encoding is in-plane only; the z extent of every source voxel equals
    voxel_mm / oversampling (a cubic source box), so with oversampling above one the
    sampled slab is thinner than the nominal voxel until the z slice profile lands.
    Receiver noise is drawn per acquired voxel after decoding, independent of the
    weighting and sampling, so acquisition weighting shapes the signal response but
    never colours the noise (the filtering idealization).

    Args:
        scan: The VirtualScan to acquire with.
        center_mm: World centre of the slice (x, y, z).
        matrix: Acquired grid size (nx, ny).
        voxel_mm: Nominal voxel edge (also the acquired grid spacing).
        oversampling: Source grid refinement per dimension; 1 gives ideal box voxels.
        weighting: None, or "hamming" k-space acquisition weighting.
        sampling: None (full Cartesian), "elliptical" corner-free coverage, or an
            explicit (nx, ny) mask or density array on the acquired k grid.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, scan, center_mm=(0.0, 0.0, 0.0), matrix=(16, 16), voxel_mm=10.0,
                 oversampling=1, weighting=None, sampling=None):
        if weighting not in (None, "hamming"):
            raise ValueError(f"unknown weighting {weighting!r}; known: None, hamming")
        self.scan = scan
        self.center_mm = tuple(float(x) for x in center_mm)
        self.matrix = (int(matrix[0]), int(matrix[1]))
        self.voxel_mm = float(voxel_mm)
        self.oversampling = int(oversampling)
        self.weighting = weighting
        if sampling is None or (isinstance(sampling, str) and sampling == "elliptical"):
            self.sampling = sampling
        elif isinstance(sampling, str):
            raise ValueError(f"unknown sampling {sampling!r}; known: None, elliptical, or an array")
        else:
            mask = np.asarray(sampling, dtype=np.float32)
            if mask.shape != self.matrix:
                raise ValueError(f"sampling mask shape {mask.shape} does not match matrix {self.matrix}")
            self.sampling = mask
        nx, ny = self.source_matrix
        source_mm = self.voxel_mm / self.oversampling
        self.voxels = [
            [scan.voxel(self.source_position(i, j), source_mm, compiled=True) for j in range(ny)]
            for i in range(nx)
        ]

    #**************#
    #   geometry   #
    #**************#
    @property
    def source_matrix(self):
        """The simulated source grid, `oversampling` times finer than the acquisition."""
        return (self.matrix[0] * self.oversampling, self.matrix[1] * self.oversampling)

    def position(self, i, j):
        """World position of acquired grid point (i, j), the lattice centred on center_mm."""
        nx, ny = self.matrix
        return (self.center_mm[0] + (i - (nx - 1) / 2.0) * self.voxel_mm,
                self.center_mm[1] + (j - (ny - 1) / 2.0) * self.voxel_mm,
                self.center_mm[2])

    def source_position(self, i, j):
        """World position of source grid point (i, j)."""
        nx, ny = self.source_matrix
        spacing = self.voxel_mm / self.oversampling
        return (self.center_mm[0] + (i - (nx - 1) / 2.0) * spacing,
                self.center_mm[1] + (j - (ny - 1) / 2.0) * spacing,
                self.center_mm[2])

    @property
    def affine(self):
        """Voxel-to-world matrix of the acquisition grid."""
        affine = np.diag([self.voxel_mm, self.voxel_mm, self.voxel_mm, 1.0])
        affine[:3, 3] = self.position(0, 0)
        return affine

    #*****************#
    #   acquisition   #
    #*****************#
    def acquire(self, seed=None, npoints=512, dwell_s=1.0 / 4000.0):
        """Acquire the slice and return NIfTI-MRS carrying the full provenance.

        One parameter draw serves every voxel (a subject has one set of concentrations).
        The clean source grid is phase encoded (spatial FFT), truncated to the central
        acquisition matrix of k with the optional weighting, and decoded back; receiver
        noise is then drawn per acquired voxel and per coil. With a CoilArray on the
        scan the data gains a trailing DIM_COIL dimension.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        rng = np.random.default_rng(seed)
        truth = self.scan.params.sample(rng)
        coils = self.scan.coils
        nx, ny = self.matrix
        sx, sy = self.source_matrix
        source = np.zeros((sx, sy, npoints) + ((len(coils),) if coils is not None else ()),
                          dtype=np.complex64)
        for i in range(sx):
            for j in range(sy):
                voxel = self.voxels[i][j]
                _, fid = voxel._clean_fid(truth, npoints, dwell_s)
                if coils is None:
                    source[i, j] = fid
                else:
                    source[i, j] = np.transpose(coils.sensitivities(voxel.position_mm)[:, None]
                                                * fid[None, :])

        encoded = self._encode(source)
        shape = (nx, ny, 1, npoints) + ((len(coils),) if coils is not None else ())
        data = np.zeros(shape, dtype=np.complex64)
        for i in range(nx):
            for j in range(ny):
                signal = encoded[i, j]
                if truth.noise.sigma > 0.0:
                    if coils is None:
                        signal = signal + truth.noise.sigma * (
                            rng.standard_normal(npoints) + 1j * rng.standard_normal(npoints)
                        ).astype(np.complex64)
                    else:
                        signal = signal + np.transpose(coils.noise(rng, truth.noise.sigma, npoints))
                data[i, j, 0] = signal

        hdr = Hdr_Ext(self.voxels[0][0]._f0_mhz(), "1H")
        if coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "params": truth.to_dict()},
            "MRSandbox: the realized parameter tree of this acquisition.",
        )
        hdr.set_user_def(
            "SimulationGrid",
            {"brain": self.scan.brain.name, "center_mm": list(self.center_mm),
             "matrix": list(self.matrix), "voxel_mm": self.voxel_mm,
             "oversampling": self.oversampling, "weighting": self.weighting,
             "sampling": self.sampling.tolist() if isinstance(self.sampling, np.ndarray)
             else self.sampling},
            "MRSandbox: the virtual brain and the slice lattice.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr, affine=self.affine)

    #**************#
    #   encoding   #
    #**************#
    def _encode(self, source):
        """Phase encode the source grid and decode the central k-space matrix.

        With oversampling 1, no weighting, and full sampling this is exactly the
        identity (ideal box voxels). Otherwise the truncation to the acquired matrix
        imprints the point spread function. The forward FFT runs over the oversampled
        source grid and the inverse over the acquired grid, so dividing by
        oversampling^2 makes a uniform object keep its amplitude exactly; the sampling
        mask and the DC-centred weighting (w(0) = 1) both preserve that identity.
        """
        if self.oversampling == 1 and self.weighting is None and self.sampling is None:
            return source
        nx, ny = self.matrix
        k = np.fft.fftshift(np.fft.fft2(source, axes=(0, 1)), axes=(0, 1))
        sx, sy = self.source_matrix
        cx, cy = sx // 2, sy // 2
        k = k[cx - nx // 2:cx - nx // 2 + nx, cy - ny // 2:cy - ny // 2 + ny]
        mask = self._sampling_mask()
        if mask is not None:
            k = k * mask.reshape(nx, ny, *([1] * (k.ndim - 2)))
        if self.weighting == "hamming":
            # the k-space Hamming centred on DC: w(p) = 0.54 + 0.46 cos(pi p / (n/2)),
            # w(0) exactly 1; numpy's np.hamming peaks between samples on an even grid
            # and would attenuate DC and shift the point spread function half a k step
            wx = 0.54 + 0.46 * np.cos(np.pi * (np.arange(nx) - nx // 2) / (nx / 2.0))
            wy = 0.54 + 0.46 * np.cos(np.pi * (np.arange(ny) - ny // 2) / (ny / 2.0))
            k = k * np.outer(wx, wy).reshape(nx, ny, *([1] * (k.ndim - 2)))
        decoded = np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1))
        return (decoded / self.oversampling ** 2).astype(np.complex64)

    def _sampling_mask(self):
        """The sampling mask on the acquired k grid as float32, or None for full
        Cartesian: the inscribed ellipse (2 px / nx)^2 + (2 py / ny)^2 <= 1 (edge
        inclusive) for "elliptical", or the user's explicit array verbatim."""
        if self.sampling is None:
            return None
        if isinstance(self.sampling, np.ndarray):
            return self.sampling
        nx, ny = self.matrix
        u = (np.arange(nx) - nx // 2) / (nx / 2.0)
        v = (np.arange(ny) - ny // 2) / (ny / 2.0)
        return ((u[:, None] ** 2 + v[None, :] ** 2) <= 1.0 + 1e-9).astype(np.float32)
