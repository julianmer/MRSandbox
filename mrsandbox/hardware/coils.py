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

    Args:
        loops: The receive Loops.
        noise_resistance: The matrix R of Roemer Eq. [9]; diagonal entries are each
            coil's own noise resistance, off-diagonal entries the correlated noise
            between coils. Defaults to the identity (independent, equal coils).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, loops, noise_resistance=None):
        self.loops = tuple(loops)
        n = len(self.loops)
        self.noise_resistance = (
            np.asarray(noise_resistance, dtype=float) if noise_resistance is not None else np.eye(n)
        )
        if self.noise_resistance.shape != (n, n):
            raise ValueError(f"noise_resistance must be {n}x{n} for {n} loops")

    def __len__(self):
        return len(self.loops)

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

    #********************#
    #   sensitivities    #
    #********************#
    def sensitivities(self, position_mm, size_mm=None):
        """Complex sensitivity vector b at a position (Roemer's transverse field vector).

        The loop fields are analytic point samples at the position; `size_mm` is the
        shared signature of the mapped arrays and is ignored here (stated)."""
        return np.asarray([loop.sensitivity(position_mm) for loop in self.loops],
                          dtype=np.complex64)

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
        eps = rng.standard_normal((len(self.loops), npoints)) + 1j * rng.standard_normal(
            (len(self.loops), npoints)
        )
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
    complex cache, then serves positions by nearest-cell lookup and voxel boxes by the
    overlap-weighted mean (see phantom.Volume). The shipped files carry no world
    origin, so the volume-centre convention of the other BigBrain-MR resources
    applies. Positions outside the mapped volume have zero sensitivity.

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
        n = self.map.shape[-1]
        self.noise_resistance = (
            np.asarray(noise_resistance, dtype=float) if noise_resistance is not None else np.eye(n)
        )

    def __len__(self):
        return self.map.shape[-1]

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name="bigbrain_mr", stride=5):
        """Load a named preset, deriving and caching its strided complex map on first use.

        The first call reads the two full 400 um volumes (about 10 GB each decompressed)
        and writes the strided complex cache; every later call loads the cache.
        """
        if name != "bigbrain_mr":
            raise KeyError(f"unknown coil map preset {name!r}; known: bigbrain_mr")
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
    def cached(cls, name="bigbrain_mr", stride=5):
        """True when the derived map cache already exists."""
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
        return np.asarray(value, dtype=np.complex64)
