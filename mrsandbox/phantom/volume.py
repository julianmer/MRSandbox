####################################################################################################
#                                             volume.py                                            #
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


#*************#
#   turning   #
#*************#
def turning(angles_deg):
    """The rotation (3, 3) that turns a box about x, then y, then z by `angles_deg`, from the
    box's own frame to the world's: a turned voxel, a moved head."""
    rotation = np.eye(3)
    for axis, angle in enumerate(np.radians(np.asarray(angles_deg, dtype=np.float64))):
        c, s = np.cos(angle), np.sin(angle)
        i, j = [k for k in range(3) if k != axis]
        turn = np.eye(3)
        turn[i, i], turn[i, j], turn[j, i], turn[j, j] = c, -s, s, c
        rotation = turn @ rotation
    return rotation


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
    index-space bounding box, a stated approximation. A turned box (`turn`, its rotation
    from its own frame to the world's) weighs every cell by the share of SUBCELLS^3
    points of it that lie inside. Trailing axes beyond the three spatial ones (tissue
    classes, coil channels) ride along untouched.

    Args:
        data: Array (nx, ny, nz, ...).
        affine: 4x4 voxel-to-world matrix (world coordinates in mm).
        name: Identifier stored in provenance.
    """

    SUBCELLS = 4

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
        BigBrain-MR loaders (the shipped files carry no world origin); an entry's
        origin_index is the cell that becomes world (0, 0, 0)."""
        import nibabel as nib

        from mrsandbox.resources import Resources

        entry = Resources.REGISTRY[name]
        image = nib.load(str(Resources.path(name)))
        data = np.asarray(image.get_fdata(), dtype=np.float32) * np.float32(entry.get("scale", 1.0))
        affine = np.array(image.affine, dtype=np.float64)
        if name.startswith("bigbrain_mr/"):
            affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(data.shape[:3]) - 1) / 2.0)
        if "origin_index" in entry:
            affine[:3, 3] = -affine[:3, :3] @ np.asarray(entry["origin_index"], dtype=np.float64)
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
    def box(self, position_mm, size_mm, turn=None):
        """The cells a box overlaps and their overlap weights.

        Args:
            position_mm: Box centre in world millimetres (x, y, z).
            size_mm: Edge length of a cubic box, or the three (x, y, z) edges.
            turn: The box's rotation (3, 3) from its own frame to the world's, or None.

        Returns:
            (values, weights): the overlapped cells (nx', ny', nz', ...) and their
            weights (nx', ny', nz'), which sum to the box volume in cells where the
            box lies inside the grid; both empty when it lies outside.
        """
        span = self._span(position_mm, size_mm, turn)
        if span is None:
            return (np.empty((0, 0, 0) + self.data.shape[3:], dtype=self.data.dtype),
                    np.empty((0, 0, 0)))
        first, last, lo, hi = span
        values = self.data[tuple(slice(f, l + 1) for f, l in zip(first, last))]
        if turn is not None:
            # every cell's sub-points in the box's frame, counted where they lie inside it
            half = np.asarray(size_mm, dtype=np.float64) * np.ones(3) / 2.0
            linear = self.affine[:3, :3]
            inside = (self.centres(position_mm, size_mm, turn)
                      - np.asarray(position_mm, dtype=np.float64)) @ turn
            steps = (np.arange(self.SUBCELLS) + 0.5) / self.SUBCELLS - 0.5
            weights = np.zeros(inside.shape[:3])
            for step in np.stack(np.meshgrid(steps, steps, steps), axis=-1).reshape(-1, 3):
                weights += np.all(np.abs(inside + step @ linear.T @ turn) <= half, axis=-1)
            return values, weights / self.SUBCELLS ** 3
        overlaps = []
        for axis in range(3):
            cells = np.arange(first[axis], last[axis] + 1)
            overlaps.append(np.clip(np.minimum(hi[axis], cells + 0.5)
                                    - np.maximum(lo[axis], cells - 0.5), 0.0, 1.0))
        weights = (overlaps[0][:, None, None] * overlaps[1][None, :, None]
                   * overlaps[2][None, None, :])
        return values, weights

    def centres(self, position_mm, size_mm, turn=None):
        """World centres (nx', ny', nz', 3) of the cells `box` returns; empty outside."""
        span = self._span(position_mm, size_mm, turn)
        if span is None:
            return np.empty((0, 0, 0, 3))
        first, last, _, _ = span
        grids = np.meshgrid(*(np.arange(f, l + 1) for f, l in zip(first, last)), indexing="ij")
        index = np.stack(grids, axis=-1).astype(np.float64)
        return index @ self.affine[:3, :3].T + self.affine[:3, 3]

    def _span(self, position_mm, size_mm, turn=None):
        """The index range a box overlaps: (first, last, lo, hi) per axis, with first
        and last the inclusive cell indices and lo and hi the box edges in index
        space (a turned box's bounding box there); None when the box lies outside the
        grid."""
        centre = np.asarray(position_mm, dtype=np.float64)
        half = np.asarray(size_mm, dtype=np.float64) * np.ones(3) / 2.0
        signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        reach = signs * half if turn is None else (signs * half) @ np.asarray(turn).T
        corners = (self._inverse @ np.c_[centre + reach, np.ones(8)].T)[:3]
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        # cell i overlaps [lo, hi] with positive measure iff lo - 1/2 < i < hi + 1/2
        first = np.maximum(np.floor(lo - 0.5).astype(int) + 1, 0)
        last = np.minimum(np.ceil(hi + 0.5).astype(int) - 1, np.asarray(self.shape) - 1)
        if np.any(first > last):
            return None
        return first, last, lo, hi

    #**********#
    #   mean   #
    #**********#
    def mean(self, position_mm, size_mm, turn=None):
        """The overlap-weighted mean over a box (turned by `turn`, see `box`), per trailing
        axis; None when the box lies outside the grid."""
        values, weights = self.box(position_mm, size_mm, turn)
        total = weights.sum()
        if total <= 0.0:
            return None
        return np.tensordot(weights, values, axes=3) / total

    #*************#
    #   nearest   #
    #*************#
    def at(self, points_mm):
        """The values of the cells nearest to world points (n, 3); zero outside the grid."""
        index = np.rint((np.c_[points_mm, np.ones(len(points_mm))] @ self._inverse.T)[:, :3]
                        ).astype(int)
        inside = np.all((index >= 0) & (index < np.asarray(self.shape)), axis=1)
        values = np.zeros((len(points_mm),) + self.data.shape[3:], dtype=self.data.dtype)
        values[inside] = self.data[tuple(index[inside].T)]
        return values

    def nearest(self, position_mm):
        """The value of the cell nearest to a position; None outside the grid."""
        index = np.rint((self._inverse @ np.array([*position_mm, 1.0]))[:3]).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(self.shape)):
            return None
        return self.data[tuple(index)]
