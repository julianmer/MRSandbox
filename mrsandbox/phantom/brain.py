####################################################################################################
#                                             brain.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual brain: continuous tissue probability maps with a world affine, answering    #
#          one question for any voxel box in millimetres: what fractions of grey matter, white     #
#          matter, and CSF sit inside. Every preset is fetched on first use through resources      #
#          (ICBM152 from TemplateFlow, BigBrain-MR from Zenodo); your own maps load through the    #
#          same interface. The wheel ships no data.                                                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox.params import Tissues
from mrsandbox.phantom.volume import Volume


#**************************************************************************************************#
#                                           Class Brain                                            #
#**************************************************************************************************#
#                                                                                                  #
# Tissue probability maps plus the affine that places them in world millimetres.                   #
#                                                                                                  #
#**************************************************************************************************#
class Brain:
    """Tissue probability maps plus the affine that places them in world millimetres.

    Args:
        gm, wm, csf: Probability volumes with a shared shape, values in [0, 1].
        affine: 4x4 voxel-to-world matrix (world coordinates in mm).
        name: Identifier stored in provenance.
    """

    PRESETS = ("icbm152", "bigbrain_mr")

    # BigBrain-MR atlas labels (Sainz Martinez et al. 2023, Table I) grouped by tissue.
    # Stated conventions for the mixed structures: the brainstem counts as white matter,
    # the ventral diencephalon and pineal gland as gray matter.
    BIGBRAIN_TISSUES = {
        "wm": (1, 4, 13),
        "gm": (2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20),
        "csf": (3,),
    }

    #**********#
    #   init   #
    #**********#
    def __init__(self, gm, wm, csf, affine, name="custom"):
        maps = tuple(np.asarray(m, dtype=np.float32) for m in (gm, wm, csf))
        if not (maps[0].shape == maps[1].shape == maps[2].shape):
            raise ValueError("gm, wm, and csf maps must share one shape")
        self.maps = Volume(np.stack(maps, axis=-1), affine, name=name)
        self.gm, self.wm, self.csf = (self.maps.data[..., i] for i in range(3))
        self.affine = self.maps.affine
        self.name = name

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name="icbm152"):
        """Load a named brain preset; its maps are fetched on first use and cached."""
        if name not in cls.PRESETS:
            raise KeyError(f"unknown brain {name!r}; known: {', '.join(cls.PRESETS)}")
        if name == "bigbrain_mr":
            return cls._load_bigbrain()
        import nibabel as nib

        from mrsandbox.resources import Resources

        images = {
            label: nib.load(str(Resources.path(f"icbm152/{label}_2mm")))
            for label in ("gm", "wm", "csf")
        }
        return cls(
            gm=images["gm"].get_fdata(),
            wm=images["wm"].get_fdata(),
            csf=images["csf"].get_fdata(),
            affine=images["gm"].affine,
            name=name,
        )

    @classmethod
    def _load_bigbrain(cls):
        """The BigBrain-MR atlas at 800 um, tissue maps from the 20-ROI labels.

        The label volume is fetched from Zenodo on first use (about half a megabyte)
        and cached; partial volumes at voxel boundaries emerge from box averaging the
        binary tissue masks, exactly as for probability maps. The shipped files carry
        no world origin (the affine is a bare 0.8 mm scaling), so by stated convention
        the origin is moved to the volume centre (Volume.load), which puts (0, 0, 0)
        mid brain the way ICBM coordinates do.
        """
        volume = Volume.load("bigbrain_mr/labels_800um")
        labels = np.asarray(volume.data, dtype=np.uint8)
        masks = {
            tissue: np.isin(labels, cls.BIGBRAIN_TISSUES[tissue]).astype(np.float32)
            for tissue in ("gm", "wm", "csf")
        }
        return cls(gm=masks["gm"], wm=masks["wm"], csf=masks["csf"],
                   affine=volume.affine, name="bigbrain_mr")

    @classmethod
    def from_maps(cls, gm, wm, csf, affine, name="custom"):
        """Wrap your own probability maps (for example an MPRAGE segmentation)."""
        return cls(gm=gm, wm=wm, csf=csf, affine=affine, name=name)

    #**************#
    #   geometry   #
    #**************#
    @property
    def shape(self):
        """Map shape in voxels."""
        return self.gm.shape

    #**********************#
    #   tissue fractions   #
    #**********************#
    def fractions(self, position_mm, size_mm):
        """Mean tissue fractions inside a voxel box, the map cells weighted by their
        overlap with the box (see phantom.Volume).

        Args:
            position_mm: Voxel centre in world millimetres (x, y, z).
            size_mm: Edge length of the cubic voxel in millimetres, or the three
                (x, y, z) edges of an anisotropic box.

        Returns:
            Tissues(gm, wm, csf) mean probabilities; all zero when the box lies
            outside the maps.
        """
        mean = self.maps.mean(position_mm, size_mm)
        if mean is None:
            return Tissues(gm=0.0, wm=0.0, csf=0.0)
        return Tissues(gm=float(mean[0]), wm=float(mean[1]), csf=float(mean[2]))

    #*******************#
    #   lipid weight    #
    #*******************#
    def lipid_weight(self, position_mm, size_mm):
        """Fraction of the voxel box beyond the brain boundary, the v1 proxy for the
        fat layer.

        Counted only when the box straddles the boundary (it holds solid tissue and
        empty space at once): a box fully inside the brain weighs exactly zero, a box
        fully outside weighs zero too, and a box clipping the skull weighs the clipped
        fraction. The tissue probability maps carry no scalp, so the boundary stands in
        for the fat layer until segmented skull maps arrive.
        """
        values, weights = self.maps.box(position_mm, size_mm)
        if weights.sum() <= 0.0:
            return 0.0
        total = values.sum(axis=-1)
        if not bool((total[weights > 0.0] > 0.5).any()):
            return 0.0
        return float((weights * (total < 0.1)).sum() / weights.sum())

    def lipid_offset(self, position_mm, size_mm):
        """Where the fat layer sits: the centroid of the box's beyond-the-brain content
        (the cells `lipid_weight` counts) relative to the voxel centre, (x, y, z) in
        millimetres; zeros when the box holds none."""
        values, weights = self.maps.box(position_mm, size_mm)
        if weights.sum() <= 0.0:
            return np.zeros(3)
        outside = weights * (values.sum(axis=-1) < 0.1)
        if outside.sum() <= 0.0:
            return np.zeros(3)
        centres = self.maps.centres(position_mm, size_mm).reshape(-1, 3)
        return np.average(centres, axis=0, weights=outside.ravel()) - np.asarray(position_mm, dtype=np.float64)
