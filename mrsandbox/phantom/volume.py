####################################################################################################
#                                            volume.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/09/26                                                                                #
#                                                                                                  #
# Purpose: The one spatial container of the virtual world: values on a voxel grid with a world     #
#          affine, answering what a box in millimetres holds. Tissue fractions, field offsets,     #
#          receive sensitivities, and the quantitative maps to come all read their voxel boxes     #
#          through this one lookup, with exact partial-volume weights at the box edges.            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np


#**************************************************************************************************#
#                                           Class Volume                                           #
#**************************************************************************************************#
#                                                                                                  #
# Values on a voxel grid with a world affine, sampled over boxes in millimetres.                   #
#                                                                                                  #
#**************************************************************************************************#
class Volume:
    """Values on a voxel grid with a world affine, sampled over boxes in millimetres.

    A grid cell (i, j, k) covers [i - 1/2, i + 1/2] along each index axis, the NIfTI
    voxel-centre convention. A box is mapped into index space through the inverse
    affine and every cell it overlaps is weighted by its overlap fraction: the exact
    partial volume for an axis-aligned affine, and for an oblique affine the box's
    index-space bounding box, a stated approximation. Trailing axes beyond the three
    spatial ones (tissue classes, coil channels) ride along untouched.

    Args:
        data: Array (nx, ny, nz, ...).
        affine: 4x4 voxel-to-world matrix (world coordinates in mm).
        name: Identifier stored in provenance.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, data, affine, name="custom"):
        self.data = np.asarray(data)
        if self.data.ndim < 3:
            raise ValueError("a volume needs three spatial axes")
        self.affine = np.asarray(affine, dtype=np.float64)
        self._inverse = np.linalg.inv(self.affine)
        self.name = name

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name):
        """A registered NIfTI resource as a Volume, fetched on first use: values scaled
        to the registry's stated units where the entry carries a scale, and for the
        bigbrain_mr entries the volume-centre origin convention of the other
        BigBrain-MR loaders (the shipped files carry no world origin)."""
        import nibabel as nib

        from mrsandbox.resources import Resources

        entry = Resources.REGISTRY[name]
        image = nib.load(str(Resources.path(name)))
        data = np.asarray(image.get_fdata(), dtype=np.float32) * np.float32(entry.get("scale", 1.0))
        affine = np.array(image.affine, dtype=np.float64)
        if name.startswith("bigbrain_mr/"):
            affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(data.shape[:3]) - 1) / 2.0)
        return cls(data, affine, name=name)

    #**************#
    #   geometry   #
    #**************#
    @property
    def shape(self):
        """Grid shape in voxels, the three spatial axes."""
        return self.data.shape[:3]

    #*********#
    #   box   #
    #*********#
    def box(self, position_mm, size_mm):
        """The cells a box overlaps and their overlap weights.

        Args:
            position_mm: Box centre in world millimetres (x, y, z).
            size_mm: Edge length of a cubic box, or the three (x, y, z) edges.

        Returns:
            (values, weights): the overlapped cells (nx', ny', nz', ...) and their
            weights (nx', ny', nz'), which sum to the box volume in cells where the
            box lies inside the grid; both empty when it lies outside.
        """
        centre = np.asarray(position_mm, dtype=np.float64)
        half = np.asarray(size_mm, dtype=np.float64) * np.ones(3) / 2.0
        signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = (self._inverse @ np.c_[centre + signs * half, np.ones(8)].T)[:3]
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        # cell i overlaps [lo, hi] with positive measure iff lo - 1/2 < i < hi + 1/2
        first = np.maximum(np.floor(lo - 0.5).astype(int) + 1, 0)
        last = np.minimum(np.ceil(hi + 0.5).astype(int) - 1, np.asarray(self.shape) - 1)
        if np.any(first > last):
            return (np.empty((0, 0, 0) + self.data.shape[3:], dtype=self.data.dtype),
                    np.empty((0, 0, 0)))
        overlaps = []
        for axis in range(3):
            cells = np.arange(first[axis], last[axis] + 1)
            overlaps.append(np.clip(np.minimum(hi[axis], cells + 0.5)
                                    - np.maximum(lo[axis], cells - 0.5), 0.0, 1.0))
        weights = (overlaps[0][:, None, None] * overlaps[1][None, :, None]
                   * overlaps[2][None, None, :])
        values = self.data[tuple(slice(f, l + 1) for f, l in zip(first, last))]
        return values, weights

    #**********#
    #   mean   #
    #**********#
    def mean(self, position_mm, size_mm):
        """The overlap-weighted mean over a box, per trailing axis; None when the box
        lies outside the grid."""
        values, weights = self.box(position_mm, size_mm)
        total = weights.sum()
        if total <= 0.0:
            return None
        return np.tensordot(weights, values, axes=3) / total

    #*************#
    #   nearest   #
    #*************#
    def nearest(self, position_mm):
        """The value of the cell nearest to a position; None outside the grid."""
        index = np.rint((self._inverse @ np.array([*position_mm, 1.0]))[:3]).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(self.shape)):
            return None
        return self.data[tuple(index)]
