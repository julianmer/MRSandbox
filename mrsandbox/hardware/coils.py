####################################################################################################
#                                            coils.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Receive coil arrays after Roemer et al. 1990 (The NMR phased array, Magn Reson Med      #
#          16, 192-225): per-coil complex sensitivities from the Biot-Savart field of circular     #
#          loops, correlated noise through the noise resistance matrix R (their Eq. [9]), and      #
#          the optimum combination alpha = lambda R^-1 b (their Eqs. [19] and [20]) with the       #
#          sum-of-squares fallback (their Eq. [24]).                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import copy
import dataclasses

import numpy as np
import scipy.special

from mrsandbox.phantom.volume import Volume


#**************************************************************************************************#
#                                            Class Loop                                            #
#**************************************************************************************************#
#                                                                                                  #
# One circular receive loop; its sensitivity is the B1- component of its Biot-Savart field.        #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Loop:
    """One circular receive loop; its sensitivity is the B1- component of its Biot-Savart
    field at the point of interest (reception couples to B1- = (Bx - iBy) / 2, with B0
    along z). The off-axis field of a circular loop is the standard closed form in
    complete elliptic integrals; on axis it reduces to a^2 / (2 (a^2 + z^2)^(3/2)) per
    unit current (mu0 omitted, sensitivities are relative).

    Attributes:
        radius_mm: Loop radius.
        center_mm: Loop centre position.
        normal: Loop axis direction (need not be normalized).
    """

    radius_mm: float
    center_mm: tuple
    normal: tuple = (0.0, 0.0, 1.0)

    #*****************#
    #   sensitivity   #
    #*****************#
    def sensitivity(self, position_mm):
        """Complex receive sensitivity B1- at a position, relative units."""
        bx, by, _ = self.field(position_mm)
        return complex(0.5 * (bx - 1j * by))

    #***************#
    #   potential   #
    #***************#
    def potential(self, points_mm):
        """The vector potential of the unit-current loop at many points, (n, 3), mu0
        omitted like the field: azimuthal about the loop's axis, the closed form in
        complete elliptic integrals, A = (1 / (pi k)) sqrt(a / rho) ((1 - k^2 / 2) K - E)
        with k^2 = 4 a rho / ((a + rho)^2 + z^2). The electric field a loop induces in
        a conductor is -i omega A, which is what its noise comes from."""
        a = float(self.radius_mm)
        n = np.asarray(self.normal, dtype=float)
        n = n / np.linalg.norm(n)
        r = np.asarray(points_mm, dtype=float) - np.asarray(self.center_mm, dtype=float)
        z = r @ n
        radial = r - z[:, None] * n
        rho = np.linalg.norm(radial, axis=1)
        safe = np.maximum(rho, 1e-9)
        k2 = 4.0 * a * safe / ((a + safe) ** 2 + z ** 2)
        k = np.sqrt(k2)
        size = (np.sqrt(a / safe) / (np.pi * k)
                * ((1.0 - k2 / 2.0) * scipy.special.ellipk(k2) - scipy.special.ellipe(k2)))
        azimuth = np.cross(n, radial / safe[:, None])
        return np.where(rho[:, None] > 1e-9, size[:, None] * azimuth, 0.0)

    #***********#
    #   field   #
    #***********#
    def field(self, position_mm):
        """Biot-Savart field vector of the unit-current loop at a position (mu0 omitted)."""
        a = float(self.radius_mm)
        n = np.asarray(self.normal, dtype=float)
        n = n / np.linalg.norm(n)
        r = np.asarray(position_mm, dtype=float) - np.asarray(self.center_mm, dtype=float)
        z = float(np.dot(r, n))
        radial = r - z * n
        rho = float(np.linalg.norm(radial))
        rho_hat = radial / rho if rho > 1e-9 else np.zeros(3)

        k2 = 4.0 * a * rho / ((a + rho) ** 2 + z ** 2)
        kk = scipy.special.ellipk(k2)
        ee = scipy.special.ellipe(k2)
        front = 1.0 / (2.0 * np.pi * np.sqrt((a + rho) ** 2 + z ** 2))
        denom = (a - rho) ** 2 + z ** 2
        b_axis = front * (kk + ee * (a ** 2 - rho ** 2 - z ** 2) / denom)
        if rho > 1e-9:
            b_radial = front * (z / rho) * (ee * (a ** 2 + rho ** 2 + z ** 2) / denom - kk)
        else:
            b_radial = 0.0
        return b_axis * n + b_radial * rho_hat


#**************************************************************************************************#
#                                         Class CoilArray                                          #
#**************************************************************************************************#
#                                                                                                  #
# A receive array: loops, their noise resistance matrix, and the Roemer combination.               #
#                                                                                                  #
#**************************************************************************************************#
class CoilArray:
    """A receive array: loops, their noise resistance matrix, and the Roemer combination.

    An array decides where a scan is quiet: after the optimum combination the noise of
    a voxel is sigma / sqrt(b^H R^-1 b), low where loops are near and the channels
    add, high in the middle of the head. `loaded` computes R from the geometry (the
    noise a conductive head induces in every pair of loops), `calibrated` fixes the
    scale the loops' relative units leave open: at one place, the one the scan's
    noise prior was calibrated at, the combined noise is the drawn sigma itself, and
    everywhere else it follows the array.

    Args:
        loops: The receive Loops.
        noise_resistance: The matrix R of Roemer Eq. [9]; diagonal entries are each
            coil's own noise resistance, off-diagonal entries the correlated noise
            between coils. Defaults to the identity (independent, equal coils).
        gain: The scale on every sensitivity (`calibrated` sets it).
        modes: The channels as combinations of the loops, (channels, loops) complex: a Tim
            coil's mode matrix, whose primary, secondary and tertiary signals of a cluster
            together carry what its raw loops do (Reykowski and Blasche, ISMRM 2004 #1587;
            hardware.arrays). None: every loop its own channel.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, loops, noise_resistance=None, gain=1.0, modes=None):
        self.loops = tuple(loops)
        self.modes = None if modes is None else np.asarray(modes, dtype=np.complex128)
        self.noise_resistance = self._resistance(len(self), noise_resistance)
        self.gain = float(gain)

    @staticmethod
    def _resistance(n, noise_resistance):
        """The (n, n) noise resistance matrix, the identity when none is given."""
        matrix = np.asarray(noise_resistance) if noise_resistance is not None else np.eye(n)
        matrix = matrix.astype(np.complex128 if np.iscomplexobj(matrix) else float)     # a mode matrix mixes the loops' noise with phase
        if matrix.shape != (n, n):
            raise ValueError(f"noise_resistance must be {n}x{n} for {n} channels")
        return matrix

    def __len__(self):
        return len(self.loops) if self.modes is None else self.modes.shape[0]

    #**********#
    #   ring   #
    #**********#
    @classmethod
    def ring(cls, n=8, ring_radius_mm=110.0, loop_radius_mm=40.0, z_mm=0.0):
        """N loops on a ring around the z axis, normals pointing inward (a head array)."""
        loops = []
        for k in range(n):
            phi = 2.0 * np.pi * k / n
            center = (ring_radius_mm * np.cos(phi), ring_radius_mm * np.sin(phi), z_mm)
            loops.append(Loop(loop_radius_mm, center, normal=(-np.cos(phi), -np.sin(phi), 0.0)))
        return cls(loops)

    # the centre of the brain in the phantoms' world (the box of the ICBM152 brain runs
    # from -106 to 74 mm in y and -70 to 82 mm in z): a head array sits around the head,
    # not around the world origin near the anterior commissure
    HEAD_CENTER_MM = (0.0, -16.0, 6.0)

    #************#
    #   helmet   #
    #************#
    @classmethod
    def helmet(cls, rows=((0.0, 1), (35.0, 6), (65.0, 10), (95.0, 15)),
               radius_mm=(95.0, 125.0, 115.0),
               loop_radius_mm=38.0, center_mm=HEAD_CENTER_MM):
        """Loops tiling a sphere about the head in rows, normals pointing inward: the
        close-fitting head arrays (the default rows, one loop on top and rings of 6,
        10, and 15 at 35, 65, and 95 degrees from the top, make 32 channels; stated
        layout, the vendors' differ in detail). `rows` is ((polar angle in degrees
        from +z, loops in the ring), ...); alternate rings are turned by half a loop.

        Checked on the COWS data (scan.calibrations, 10 subjects, a Prisma 32-channel
        array, the voxels placed through each subject's T1): the apodized SNR of the
        prefrontal, occipital, and posterior frontal voxels runs 1 : 0.91 : 0.69, this
        helmet gives 1 : 0.81 : 0.57, the right order and too steep (a sphere of 110 mm
        gives 1 : 0.77 : 0.50, and about the world origin 1 : 2.31 : 0.81). `radius_mm` is
        one radius or the (x, y, z) semi-axes of an ellipsoid; the default follows a
        head that is longer than wide (the brain's box plus 35 mm, rounded; stated)."""
        loops = []
        for index, (polar_deg, count) in enumerate(rows):
            theta = np.radians(polar_deg)
            for k in range(count):
                phi = 2.0 * np.pi * (k + 0.5 * (index % 2)) / count
                out = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                radii = np.broadcast_to(np.asarray(radius_mm, dtype=float), (3,))
                inward = -(out / radii) / np.linalg.norm(out / radii)  # the ellipsoid's normal
                loops.append(Loop(loop_radius_mm, tuple(np.asarray(center_mm) + radii * out),
                                  normal=tuple(inward)))
        return cls(loops)

    #*************#
    #   surface   #
    #*************#
    @classmethod
    def surface(cls, center_mm, normal, loop_radius_mm=50.0):
        """One surface loop at `center_mm` facing along `normal`: the single coil laid
        on the head, sensitive to about a radius' depth."""
        return cls([Loop(loop_radius_mm, tuple(center_mm), normal=tuple(normal))])

    #************#
    #   loaded   #
    #************#
    def loaded(self, radius_mm=95.0, step_mm=10.0, center_mm=HEAD_CENTER_MM, own_noise=0.25):
        """The same array with its noise resistance matrix computed from the geometry:
        R_ij is the volume integral of E_i . E_j over the conductive sample (Roemer
        Eq. [9]; sample noise dominant, uniform conductivity), with E of a loop
        proportional to its vector potential (`Loop.potential`), here over a sphere
        the size of a head on a `step_mm` grid, scaled to a mean diagonal of one.
        Neighbouring loops then share noise, as they do on a scanner. `own_noise` is
        the share of a channel's noise power that is its own (conductor and
        preamplifier) and correlates with no other channel. The default is measured:
        the noise adjustment of a 32-channel head array on a head at 3 T has a
        strongest partner per channel of 0.41 on average where the sample alone
        gives 0.61 (scan.calibrations COIL_OWN_NOISE_PRISMA_3T). A small phantom loads
        the loops less and the share grows: 0.21 measured in a 32-channel array."""
        axis = np.arange(-radius_mm, radius_mm + step_mm / 2.0, step_mm)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        points = grid[np.linalg.norm(grid, axis=1) <= radius_mm] + np.asarray(center_mm, dtype=float)
        potentials = np.stack([loop.potential(points) for loop in self.loops])
        resistance = np.einsum("ipk,jpk->ij", potentials, potentials)
        resistance = resistance / np.mean(np.diag(resistance))
        resistance = (1.0 - own_noise) * resistance + own_noise * np.eye(len(self.loops))
        if self.modes is not None:                       # the combiner mixes the loops' noise as it mixes their signals
            resistance = self.modes @ resistance @ self.modes.conj().T
        return type(self)(self.loops, resistance, gain=self.gain, modes=self.modes)

    #****************#
    #   calibrated   #
    #****************#
    def calibrated(self, reference_mm, size_mm=None):
        """The same array scaled so the optimum combination's noise at `reference_mm`
        is the scan's drawn sigma (b^H R^-1 b = 1 there): the place the noise prior was
        calibrated at (scan.defaults: a parietal midline voxel under a 32-channel head
        array). Everywhere else the noise follows the array, `noise_gain`."""
        scaled = copy.copy(self)
        scaled.gain = self.gain * self.noise_gain(reference_mm, size_mm)
        return scaled

    def noise_gain(self, position_mm, size_mm=None):
        """What the optimum combination makes of the channel noise at a position:
        1 / sqrt(b^H R^-1 b), the factor on sigma (one at the calibrated place)."""
        b = self.sensitivities(position_mm, size_mm).astype(np.complex128)
        return float(1.0 / np.sqrt(np.real(np.vdot(b, np.linalg.solve(self.noise_resistance, b)))))

    #********************#
    #   sensitivities    #
    #********************#
    def sensitivities(self, position_mm, size_mm=None):
        """Complex sensitivity vector b at a position (Roemer's transverse field vector).

        The loop fields are analytic point samples at the position; `size_mm` is the
        shared signature of the mapped arrays and is ignored here (stated)."""
        b = np.asarray([loop.sensitivity(position_mm) for loop in self.loops], dtype=np.complex128)
        return (self.gain * (b if self.modes is None else self.modes @ b)).astype(np.complex64)

    #*****************#
    #   combination   #
    #*****************#
    def combine(self, channels, position_mm, size_mm=None):
        """Optimum combination P = lambda p^T R^-1 b (Roemer Eq. [20]).

        lambda is chosen as 1 / (b^H R^-1 b) so a noise-free input b s returns exactly s:
        the combined data is phased and scaled as if it came from a single coil, which is
        what spectroscopy needs.

        Args:
            channels: Complex array (n_coils, npoints).
            position_mm: Where the voxel sits; sets the weights.
            size_mm: Its size; a mapped array then weights with the same box-averaged
                sensitivities the voxel received with.
        """
        b = self.sensitivities(position_mm, size_mm).astype(np.complex128)
        rinv_b = np.linalg.solve(self.noise_resistance, b)
        weights = rinv_b.conj() / np.real(np.vdot(b, rinv_b))
        return weights @ np.asarray(channels, dtype=np.complex128)

    def sum_of_squares(self, channels):
        """Field-blind fallback P = sqrt(p^T R^-1 p*) (Roemer Eq. [24]); loses phase."""
        p = np.asarray(channels, dtype=np.complex128)
        rinv_p = np.linalg.solve(self.noise_resistance, p)
        return np.sqrt(np.real(np.sum(p.conj() * rinv_p, axis=0)))

    #***********#
    #   noise   #
    #***********#
    def noise(self, rng, sigma, npoints):
        """Correlated complex Gaussian noise (n_coils, npoints) with covariance sigma^2 R."""
        chol = np.linalg.cholesky(self.noise_resistance)
        eps = rng.standard_normal((len(self), npoints)) + 1j * rng.standard_normal((len(self), npoints))
        return (sigma * (chol @ eps)).astype(np.complex64)


#**************************************************************************************************#
#                                      Class MappedCoilArray                                       #
#**************************************************************************************************#
#                                                                                                  #
# A receive array defined by measured sensitivity maps instead of loop geometry.                   #
#                                                                                                  #
#**************************************************************************************************#
class MappedCoilArray(CoilArray):
    """A receive array defined by measured complex sensitivity maps instead of loop
    geometry; combination and noise generation are inherited unchanged.

    The bigbrain_mr preset derives its maps once from the BigBrain-MR 400 um ESPIRiT
    magnitude and phase volumes (32-channel Nova head coil at 7 T) into a strided
    complex cache; the t2star_move preset takes the measured 3 T maps of the T2*-MOVE
    dataset through Augmentrum (see `load`). Both serve positions by nearest-cell
    lookup and voxel boxes by the overlap-weighted mean (see phantom.Volume). The
    shipped files carry no world origin, so the volume-centre convention applies.
    Positions outside the mapped volume have zero sensitivity.

    Args:
        sensitivities_map: Complex volume (nx, ny, nz, n_coils).
        affine: 4x4 voxel-to-world matrix.
        noise_resistance: As for CoilArray; identity (independent coils) by default.
        name: Identifier stored in provenance.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, sensitivities_map, affine, noise_resistance=None, name="custom"):
        self.volume = Volume(np.asarray(sensitivities_map, dtype=np.complex64), affine, name=name)
        self.map = self.volume.data
        self.affine = self.volume.affine
        self.name = name
        self.loops = ()
        self.gain = 1.0
        self.noise_resistance = self._resistance(self.map.shape[-1], noise_resistance)

    def __len__(self):
        return self.map.shape[-1]

    #*************#
    #   loading   #
    #*************#
    PRESETS = ("bigbrain_mr", "t2star_move")

    @classmethod
    def load(cls, name="bigbrain_mr", stride=5, subject=None):
        """Load a named preset.

        "bigbrain_mr" derives and caches its strided complex map on first use: the
        first call reads the two full 400 um volumes (about 10 GB each decompressed)
        and writes the strided complex cache; every later call loads the cache.

        "t2star_move" is the measured 3 T array of the T2*-MOVE dataset (Eichhorn, H.,
        Saks, E., Spieker, V. J., Preibisch, C., Schnabel, J. A. T2*-MOVE: an open
        k-space dataset to address motion in T2* quantification from gradient echo
        MRI. Magnetic Resonance in Medicine (2025), doi 10.1002/mrm.70050; data doi
        10.15134/2kek-3553, CC BY 4.0): a 32-channel Philips head coil (31 channels in
        the file) at 2 x 2 x 3 mm over 112 x 92 x 36 voxels, fetched and cached by
        Augmentrum's T2starMove source (5.4 GB once, a few MB after; `subject` picks
        one of its 22 volunteers, the first by default). The maps are root-sum-of-
        squares normalized to one inside the head and zero outside, so they carry the
        channel pattern rather than an absolute gain (stated), and ship no world
        origin: the volume-centre convention applies, the index axes taken as world
        x, y, z (stated). `stride` is ignored for this preset.
        """
        if name not in cls.PRESETS:
            raise KeyError(f"unknown coil map preset {name!r}; known: {', '.join(cls.PRESETS)}")
        if name == "t2star_move":
            try:
                from augmentrum.sampling.coil_sampling import T2starMove
            except ImportError as error:
                raise ImportError("the t2star_move coil maps come through Augmentrum (the "
                                  "mrsandbox[augment] extra)") from error
            maps = T2starMove.fetch(subject, progress=False)
            affine = np.diag([2.0, 2.0, 3.0, 1.0])
            affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(maps.shape[:3]) - 1) / 2.0)
            return cls(maps, affine, name=name)
        from mrsandbox.resources import Resources

        cache = Resources.cache_dir() / f"bigbrain_mr/coil_sens_stride{stride}.npz"
        if not cache.exists():
            import nibabel as nib

            print(f"deriving {cache} from the full 400 um volumes (one-time, RAM heavy)")
            magnitude = nib.load(str(Resources.path("bigbrain_mr/coil_sens_mag_400um")))
            shape = np.asarray(magnitude.shape[:3])
            data = np.asarray(magnitude.dataobj)[::stride, ::stride, ::stride, :].astype(np.float32)
            phase = nib.load(str(Resources.path("bigbrain_mr/coil_sens_pha_400um")))
            angles = np.asarray(phase.dataobj)[::stride, ::stride, ::stride, :].astype(np.float32)
            sens = (data * np.exp(1j * angles)).astype(np.complex64)
            affine = np.diag([0.4 * stride] * 3 + [1.0])
            affine[:3, 3] = -0.4 * (shape - 1) / 2.0
            np.savez_compressed(cache, sensitivities=sens, affine=affine)
        stored = np.load(cache)
        return cls(stored["sensitivities"], stored["affine"], name=name)

    @classmethod
    def cached(cls, name="bigbrain_mr", stride=5, subject=None):
        """True when the preset's map cache already exists (Augmentrum's for t2star_move)."""
        if name == "t2star_move":
            try:
                from augmentrum.utils import download
            except ImportError:
                return False
            return (download.cache_root() / "csm" / f"t2star_move_{subject or 'first'}.npz").exists()
        from mrsandbox.resources import Resources

        return (Resources.cache_dir() / f"bigbrain_mr/coil_sens_stride{stride}.npz").exists()

    #********************#
    #   sensitivities    #
    #********************#
    def sensitivities(self, position_mm, size_mm=None):
        """Complex sensitivity vector at a position: the nearest mapped cell, or with
        `size_mm` the overlap-weighted mean over that voxel box, what a voxel of that
        size receives with; zero outside the mapped volume."""
        value = (self.volume.nearest(position_mm) if size_mm is None
                 else self.volume.mean(position_mm, size_mm))
        if value is None:
            return np.zeros(len(self), dtype=np.complex64)
        return self.gain * np.asarray(value, dtype=np.complex64)
