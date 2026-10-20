####################################################################################################
#                                             field.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: B0 inhomogeneity as a field offset map: every point of the voxel box precesses at its   #
#          own frequency, so the ensemble average over the box is a complex modulation of the      #
#          FID. That is the physical intra-voxel lineshape: shift from the mean offset,            #
#          broadening and non-Lorentzian shape from the spread.                                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import scipy.fft

from mrsandbox.phantom.volume import Volume


#**************************************************************************************************#
#                                          Class FieldMap                                          #
#**************************************************************************************************#
#                                                                                                  #
# A B0 offset map in Hz with a world affine, answering: how does a voxel box dephase.              #
#                                                                                                  #
#**************************************************************************************************#
class FieldMap:
    """A B0 offset map in Hz with a world affine, answering: how does a voxel box dephase.

    The dephasing kernel of a box is the isochromat average k(t) = sum_j w_j exp(i 2 pi f_j t)
    over the offsets inside the box, each map cell weighted by its overlap with the box
    (see phantom.Volume), computed from a weighted histogram of the offsets so the cost
    is independent of map resolution. Multiplying a noise-free FID by k(t) applies the
    shift of the mean offset and the broadening of the spread at once.

    Args:
        offsets_hz: Offset volume in Hz.
        affine: 4x4 voxel-to-world matrix (world coordinates in mm).
        name: Identifier stored in provenance.
        shim: The voxel shim every lookup applies unless told otherwise: "none",
            "mean" (the frequency adjustment), "linear" (plus the first-order shims,
            the default), or "quadratic" (plus the five second-order shims).

    Attributes:
        shim_box: None, or (centre_mm, size_mm) of a shim volume of the operator's own: the
            shim is then fitted over that box and every voxel sits under its fields, as on
            a scanner whose shim box is not the voxel. None fits over the voxel itself.
    """

    PRESETS = ("bigbrain_mr", "qsm_rc2", "qsm_rc2_measured")
    SHIMS = ("none", "mean", "linear", "quadratic")

    #**********#
    #   init   #
    #**********#
    def __init__(self, offsets_hz, affine, name="custom", shim="linear"):
        if shim not in self.SHIMS:
            raise ValueError(f"unknown shim {shim!r}; known: {', '.join(self.SHIMS)}")
        self.volume = Volume(np.asarray(offsets_hz, dtype=np.float32), affine, name=name)
        self.offsets_hz = self.volume.data
        self.affine = self.volume.affine
        self.name = name
        self.shim = shim
        self.shim_box = None
        # a Volume of what holds signal (a brain's water content per cell) weighs the cells of a
        # box; None counts every cell alike, so a box reaching into bone or air dephases by cells
        # that carry no signal
        self.weights = None

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name="bigbrain_mr", b0_t=3.0, shim="linear"):
        """Load a named field map preset, fetched on first use and cached.

        The BigBrain-MR background field (V-SHARP from 7 T ME-GRE, 400 um) is stored as
        Hz x 10 and measured at 7 T; susceptibility-induced offsets scale linearly with
        the field, so the map is scaled by b0_t / 7. The shipped file carries no world
        origin, so the origin moves to the volume centre, the same stated convention as
        the BigBrain-MR brain.

        Args:
            name: Preset name.
            b0_t: Field strength of the simulation in tesla.
            shim: The default voxel shim, as for the constructor.
        """
        if name not in cls.PRESETS:
            raise KeyError(f"unknown field map {name!r}; known: {', '.join(cls.PRESETS)}")
        if name == "qsm_rc2_measured":
            # the measured total field of the QSM challenge 2.0 head at 7 T on the model's
            # grid at 0.64 mm, with the texture the susceptibility model lacks and that
            # session's shim in it; ppm scales with the field. Placed by hand (resources.qsm)
            ppm = Volume.load("qsm_rc2/total_field")
            return cls(offsets_hz=ppm.data * np.float32(42.5774785 * float(b0_t)),
                       affine=ppm.affine, name=name, shim=shim)
        if name == "qsm_rc2":
            # the whole-head susceptibility model of the QSM challenge 2.0 head (air,
            # bone, and tissue), halved to 1.28 mm cells; placed by hand (resources.qsm)
            chi = Volume.load("qsm_rc2/chi")
            nx, ny, nz = (n // 2 for n in chi.shape)
            coarse = (chi.data[:2 * nx, :2 * ny, :2 * nz].reshape(nx, 2, ny, 2, nz, 2)
                      .mean(axis=(1, 3, 5)))
            halved = chi.affine @ np.array([[2.0, 0.0, 0.0, 0.5], [0.0, 2.0, 0.0, 0.5],
                                            [0.0, 0.0, 2.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
            return cls.from_susceptibility(coarse, halved, b0_t, name=name, shim=shim)
        volume = Volume.load("bigbrain_mr/background_field_400um")
        return cls(offsets_hz=volume.data * np.float32(float(b0_t) / 7.0), affine=volume.affine,
                   name=name, shim=shim)

    @classmethod
    def from_susceptibility(cls, chi_ppm, affine, b0_t, axis=2, gamma_mhz_t=42.5774785,
                            name="custom", shim="linear"):
        """The field a susceptibility distribution makes in a main field along the
        index axis `axis`, by the k-space dipole model (Salomir et al. 2003, Concepts
        Magn Reson B 19:26; Marques and Bowtell 2005, Concepts Magn Reson B 25:65):
        the relative field is the inverse transform of (1/3 - k_B^2 / k^2) times the
        transform of chi, the sphere of Lorentz included, zero at k = 0. The volume
        is padded to twice its size with its corner value (the air around a head, as
        the QSM challenge 2.0 simulation pads), so the periodic copies stay away, and
        a whole head then carries the fields of its sinuses and its skull.

        Args:
            chi_ppm: Susceptibility volume in ppm, any reference.
            affine: 4x4 voxel-to-world matrix (mm); the cell sizes set the kernel.
            b0_t: Main field in tesla.
            axis: The index axis the main field lies along (2: inferior to superior).
        """
        chi = np.asarray(chi_ppm, dtype=np.float32)
        padded = np.full(tuple(2 * n for n in chi.shape), chi[-1, -1, -1], dtype=np.float32)
        padded[:chi.shape[0], :chi.shape[1], :chi.shape[2]] = chi
        cell = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
        k = [np.fft.fftfreq(n, d=d).astype(np.float32) for n, d in zip(padded.shape, cell)]
        k[2] = np.fft.rfftfreq(padded.shape[2], d=cell[2]).astype(np.float32)
        k2 = k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2
        along = (k[axis] ** 2).reshape([-1 if a == axis else 1 for a in range(3)])
        kernel = 1.0 / 3.0 - along / np.where(k2 > 0.0, k2, 1.0)
        kernel[0, 0, 0] = 0.0
        field = scipy.fft.irfftn(scipy.fft.rfftn(padded) * kernel, s=padded.shape)
        field = field[:chi.shape[0], :chi.shape[1], :chi.shape[2]]
        return cls(offsets_hz=field * np.float32(gamma_mhz_t * float(b0_t)), affine=affine,
                   name=name, shim=shim)

    @classmethod
    def from_map(cls, offsets_hz, affine, name="custom", shim="linear"):
        """Wrap your own offset map (for example a measured B0 map in Hz)."""
        return cls(offsets_hz=offsets_hz, affine=affine, name=name, shim=shim)

    #*************#
    #   offsets   #
    #*************#
    def offsets(self, position_mm, size_mm, shim=None, shim_at_mm=None, turn=None):
        """The Hz offsets of the map cells a voxel box (cubic, or per-axis edges)
        overlaps, after the voxel shim, flat.

        A real MRS acquisition shims the voxel first, so the parts of the field a shim
        can null must not broaden the line: `shim` removes nothing ("none"), the mean
        ("mean", the frequency adjustment), the best-fit linear field over the box
        ("linear", frequency plus the first-order shims), or the best-fit field of
        first plus second order ("quadratic": the five second-order spherical
        harmonics 2z^2 - x^2 - y^2, xz, yz, x^2 - y^2, xy in world coordinates, the
        terms a second-order shim set provides; a non-harmonic r^2 component stays,
        as it does on a scanner), every fit weighted by the cells' overlap with the
        box. None takes the map's own default. Empty outside the map. A turned box (`turn`,
        its rotation from its own frame to the world's) is shimmed as it stands.
        """
        values, _ = self._shimmed(position_mm, size_mm, shim, shim_at_mm, turn)
        return values

    def _shimmed(self, position_mm, size_mm, shim, shim_at_mm=None, turn=None):
        """The post-shim offsets of the overlapped cells and their overlap weights.
        With `shim_at_mm` the shim is the one fitted for the box at that place and
        kept in the scanner while the object moved: the box now at `position_mm` in
        the object's frame sits under the shim fields of the planned place."""
        values, weights = self.volume.box(position_mm, size_mm, turn)
        # a cell the map holds nothing for (exactly zero, the void beyond its head) is no signal
        keep = (weights > 0.0) & (values != 0.0)
        xyz = (self.volume.centres(position_mm, size_mm, turn)[keep] if keep.any()
               else np.empty((0, 3)))
        v, w = values[keep].astype(np.float64), weights[keep]
        if self.weights is not None and xyz.size:
            w = w * np.asarray(self.weights.at(xyz), dtype=np.float64)
            keep = w > 0.0
            v, w, xyz = v[keep], w[keep], xyz[keep]
        # the moved box's cells under the planned shim: where they sit in the scanner
        moved = (0.0 if shim_at_mm is None
                 else np.asarray(shim_at_mm, float) - np.asarray(position_mm, float))
        return v - self.shim_field(xyz + moved,
                                   position_mm if shim_at_mm is None else shim_at_mm,
                                   size_mm, shim, turn), w

    def shim_field(self, xyz_mm, position_mm, size_mm, shim=None, turn=None):
        """The field in Hz the shim set plays at the world points `xyz_mm` (n, 3), once fitted
        for the voxel box at `position_mm` (or for `shim_box` where the operator set one): what
        the adjustment takes off the map everywhere, not in the box alone. Zeros for "none",
        outside the map, and for no points."""
        shim = self.shim if shim is None else shim
        if shim not in self.SHIMS:
            raise ValueError(f"unknown shim {shim!r}; known: {', '.join(self.SHIMS)}")
        # the operator's shim box stands square; the voxel's turns with it
        at, size, around = ((*self.shim_box, None) if self.shim_box is not None
                            else (position_mm, size_mm, turn))
        values, weights = self.volume.box(at, size, around)
        keep = (weights > 0.0) & (values != 0.0)     # the void beyond the map's head fits nothing
        if shim == "none" or not keep.any() or len(xyz_mm) == 0:
            return np.zeros(len(xyz_mm))
        planned = self.volume.centres(at, size, around)[keep]
        centre = np.average(planned, axis=0, weights=weights[keep])

        def columns(xyz):
            x, y, z = (xyz - centre).T
            out = [np.ones(len(xyz)), x, y, z]
            if shim == "quadratic":
                out += [2.0 * z * z - x * x - y * y, x * z, y * z, x * x - y * y, x * y]
            return np.stack(out if shim != "mean" else out[:1], axis=1)

        root = np.sqrt(weights[keep])
        fit, *_ = np.linalg.lstsq(columns(planned) * root[:, None],
                                  values[keep].astype(np.float64) * root, rcond=None)
        return columns(np.asarray(xyz_mm, dtype=np.float64)) @ fit

    #************#
    #   kernel   #
    #************#
    def kernel(self, position_mm, size_mm, t_s, shim=None, bins=256, shim_at_mm=None,
               turn=None):
        """The complex dephasing kernel k(t) of a voxel box on a time grid.

        The post-shim box offsets are histogrammed with their overlap weights (the
        isochromat weights), then k(t) = sum_b w_b exp(i 2 pi f_b t); `shim` as for
        `offsets`, None taking the map's default. A box outside the map dephases
        nothing.
        """
        t = np.asarray(t_s, dtype=np.float64)
        offsets, overlap = self._shimmed(position_mm, size_mm, shim, shim_at_mm, turn)
        if offsets.size == 0:
            return np.ones_like(t, dtype=np.complex64)
        return self._dephasing(offsets, overlap, t, bins)

    #***************#
    #   linewidth   #
    #***************#
    def linewidth(self, position_mm, size_mm, shim=None, shim_at_mm=None, turn=None):
        """The width in Hz the field gives a line of the voxel box after the shim: the full
        width at half maximum of the box's dephased line under a 1 Hz Lorentzian (which
        keeps a perfect shim finite), between the outermost crossings (a coarse map's cells
        give a comb of lines, one width); `shim`, `shim_at_mm` and `turn` as for `kernel`."""
        t = np.arange(8192) / 1000.0
        line = np.fft.fft(self.kernel(position_mm, size_mm, t, shim, shim_at_mm=shim_at_mm,
                                      turn=turn) * np.exp(-np.pi * t), 8 * t.size).real
        above = np.flatnonzero(np.fft.fftshift(line) > 0.5 * line.max())
        return float((above[-1] - above[0] + 1) * 1000.0 / (8 * t.size))

    def groups(self, position_mm, size_mm, t_s, count, shim=None, bins=256, shim_at_mm=None,
               spread_hz=0.0, turn=None):
        """The box's post-shim offsets split into `count` groups of equal weight along
        the offset axis, as [(mean_hz, weight, kernel)]: each group's mean offset, its
        share of the box, and its own dephasing kernel, so the weighted kernels sum
        to `kernel`. What a sequence does at an offset (an editing pulse off its
        band, a selective pulse off its line) can then be played per group at the
        group's mean. An empty box is one group at rest.

        `spread_hz` smears every cell's offset by a Gaussian of that FWHM (the
        microscopic field below the map's resolution, scan.defaults lineshape.gauss_hz)
        before the split, so the groups carry the spread and their kernels sum to
        `kernel` times the Gaussian's envelope: a spin is then played at the frequency
        it actually has, not at the map's mean over its cell."""
        t = np.asarray(t_s, dtype=np.float64)
        offsets, overlap = self._shimmed(position_mm, size_mm, shim, shim_at_mm, turn)
        if offsets.size == 0:
            return [(0.0, 1.0, np.ones_like(t, dtype=np.complex64))]
        if spread_hz > 0.0:
            nodes, node_weights = np.polynomial.hermite_e.hermegauss(32)
            offsets = (offsets[:, None]
                       + spread_hz / (2.0 * np.sqrt(2.0 * np.log(2.0))) * nodes[None, :]).ravel()
            overlap = (overlap[:, None] * node_weights[None, :] / node_weights.sum()).ravel()
        order = np.argsort(offsets)
        offsets, overlap = offsets[order], overlap[order]
        share = np.cumsum(overlap) / overlap.sum()
        member = np.minimum((share * count).astype(int), count - 1)
        out = []
        for g in range(count):
            inside = member == g
            if inside.any():
                weight = overlap[inside].sum() / overlap.sum()
                out.append((float(np.average(offsets[inside], weights=overlap[inside])),
                            float(weight),
                            self._dephasing(offsets[inside], overlap[inside], t, bins)))
        return out

    @staticmethod
    def _dephasing(offsets, overlap, t, bins):
        """sum_b w_b exp(i 2 pi f_b t) over a weighted histogram of the offsets."""
        counts, edges = np.histogram(offsets, bins=bins, weights=overlap)
        sums, _ = np.histogram(offsets, bins=edges, weights=overlap * offsets)
        filled = counts > 0
        centres = sums[filled] / counts[filled]  # per-bin mean, exact for narrow content
        weights = counts[filled] / counts.sum()
        kernel = (weights[:, None] * np.exp(2j * np.pi * centres[:, None] * t[None, :])).sum(axis=0)
        return kernel.astype(np.complex64)
