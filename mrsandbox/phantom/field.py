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
    """

    PRESETS = ("bigbrain_mr",)

    #**********#
    #   init   #
    #**********#
    def __init__(self, offsets_hz, affine, name="custom"):
        self.volume = Volume(np.asarray(offsets_hz, dtype=np.float32), affine, name=name)
        self.offsets_hz = self.volume.data
        self.affine = self.volume.affine
        self.name = name

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name="bigbrain_mr", b0_t=3.0):
        """Load a named field map preset, fetched on first use and cached.

        The BigBrain-MR background field (V-SHARP from 7 T ME-GRE, 400 um) is stored as
        Hz x 10 and measured at 7 T; susceptibility-induced offsets scale linearly with
        the field, so the map is scaled by b0_t / 7. The shipped file carries no world
        origin, so the origin moves to the volume centre, the same stated convention as
        the BigBrain-MR brain.

        Args:
            name: Preset name.
            b0_t: Field strength of the simulation in tesla.
        """
        if name not in cls.PRESETS:
            raise KeyError(f"unknown field map {name!r}; known: {', '.join(cls.PRESETS)}")
        import nibabel as nib

        from mrsandbox.resources import Resources

        entry = Resources.REGISTRY["bigbrain_mr/background_field_400um"]
        image = nib.load(str(Resources.path("bigbrain_mr/background_field_400um")))
        offsets = np.asarray(image.get_fdata(), dtype=np.float32) * entry["scale"]
        offsets = offsets * (float(b0_t) / 7.0)
        affine = np.array(image.affine, dtype=np.float64)
        affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(offsets.shape) - 1) / 2.0)
        return cls(offsets_hz=offsets, affine=affine, name=name)

    @classmethod
    def from_map(cls, offsets_hz, affine, name="custom"):
        """Wrap your own offset map (for example a measured B0 map in Hz)."""
        return cls(offsets_hz=offsets_hz, affine=affine, name=name)

    #**************#
    #   offsets    #
    #**************#
    def offsets(self, position_mm, size_mm, shim="linear"):
        """The Hz offsets of the map cells a voxel box (cubic, or per-axis edges)
        overlaps, after the voxel shim, flat.

        A real MRS acquisition shims the voxel first, so the parts of the field a shim
        can null must not broaden the line: `shim` removes nothing ("none"), the mean
        ("mean", the frequency adjustment), or the best-fit linear field over the box
        ("linear", frequency plus first-order shim, the default), both weighted by the
        cells' overlap with the box. Empty outside the map.
        """
        values, _ = self._shimmed(position_mm, size_mm, shim)
        return values

    def _shimmed(self, position_mm, size_mm, shim):
        """The post-shim offsets of the overlapped cells and their overlap weights."""
        values, weights = self.volume.box(position_mm, size_mm)
        keep = weights > 0.0
        if not keep.any():
            return np.empty(0), np.empty(0)
        v, w = values[keep].astype(np.float64), weights[keep]
        if shim == "none":
            return v, w
        if shim == "mean":
            return v - np.average(v, weights=w), w
        if shim == "linear":
            grids = np.meshgrid(*(np.arange(n) for n in weights.shape), indexing="ij")
            design = np.stack([np.ones_like(v)] + [g[keep].astype(np.float64) for g in grids],
                              axis=1)
            root = np.sqrt(w)
            fit, *_ = np.linalg.lstsq(design * root[:, None], v * root, rcond=None)
            return v - design @ fit, w
        raise ValueError(f"unknown shim {shim!r}; known: none, mean, linear")

    #*************#
    #   kernel    #
    #*************#
    def kernel(self, position_mm, size_mm, t_s, shim="linear", bins=256):
        """The complex dephasing kernel k(t) of a voxel box on a time grid.

        The post-shim box offsets are histogrammed with their overlap weights (the
        isochromat weights), then k(t) = sum_b w_b exp(i 2 pi f_b t). A box outside
        the map dephases nothing.
        """
        t = np.asarray(t_s, dtype=np.float64)
        offsets, overlap = self._shimmed(position_mm, size_mm, shim)
        if offsets.size == 0:
            return np.ones_like(t, dtype=np.complex64)
        counts, edges = np.histogram(offsets, bins=bins, weights=overlap)
        sums, _ = np.histogram(offsets, bins=edges, weights=overlap * offsets)
        filled = counts > 0
        centres = sums[filled] / counts[filled]  # per-bin mean, exact for narrow content
        weights = counts[filled] / counts.sum()
        kernel = (weights[:, None] * np.exp(2j * np.pi * centres[:, None] * t[None, :])).sum(axis=0)
        return kernel.astype(np.complex64)
