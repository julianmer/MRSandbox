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

from mrsandbox.database import tissue
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
        fat: Optional map of the fat-bearing tissue around the skull, an array on the
            same grid or a Volume on its own; with it the lipid layer is that tissue,
            without it the brain boundary.
    """

    PRESETS = ("icbm152", "bigbrain_mr", "qsm_rc2")

    # the water content per tissue class that anchors the water signal (phantom.water)
    water_content = tissue.MRSI_CHALLENGE["water_content"]

    # BigBrain-MR atlas labels (Sainz Martinez et al. 2023, Table I) grouped by tissue.
    # Stated conventions for the mixed structures: the brainstem counts as white matter,
    # the ventral diencephalon and pineal gland as gray matter.
    BIGBRAIN_TISSUES = {
        "wm": (1, 4, 13),
        "gm": (2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20),
        "csf": (3,),
    }

    # QSM challenge 2.0 head labels (resources.qsm LABELS) grouped by tissue. Stated:
    # the deep nuclei count as gray matter, blood and calcification as no tissue, and
    # the extracranial soft tissue (the deposit's "muscle"; its "fat" label is empty, so
    # scalp fat and muscle are one class) is the fat-bearing layer, beyond the bone only:
    # the same label also lines the brain as the meninges. SCALP_FROM_MM parts the two
    # sheets and SCALP_MM is where a brain without a fat map takes its scalp to lie, both
    # in mm from the brain as the head's labels place them.
    QSM_TISSUES = {"gm": (1, 2, 3, 4, 5, 6, 7, 9), "wm": (8,), "csf": (10,), "fat": (15,)}
    SCALP_FROM_MM, SCALP_MM = 6.0, (9.0, 15.0)

    #**********#
    #   init   #
    #**********#
    def __init__(self, gm, wm, csf, affine, name="custom", fat=None):
        maps = tuple(np.asarray(m, dtype=np.float32) for m in (gm, wm, csf))
        if not (maps[0].shape == maps[1].shape == maps[2].shape):
            raise ValueError("gm, wm, and csf maps must share one shape")
        self.fat = (fat if fat is None or isinstance(fat, Volume)
                    else Volume(np.asarray(fat, dtype=np.float32), affine, name=name))
        self.maps = Volume(np.stack(maps, axis=-1), affine, name=name)
        self.gm, self.wm, self.csf = (self.maps.data[..., i] for i in range(3))
        self.affine = self.maps.affine
        self.name = name
        self._shell = None

    #*************#
    #   loading   #
    #*************#
    @classmethod
    def load(cls, name="icbm152", fat=None):
        """Load a named brain preset; its maps are fetched on first use and cached.
        `fat="qsm_rc2"` lends a preset without a scalp the fat-bearing layer of the QSM
        challenge head, carried into its world (resources.QSMHead.fat; the head's
        files are placed by hand)."""
        if name not in cls.PRESETS:
            raise KeyError(f"unknown brain {name!r}; known: {', '.join(cls.PRESETS)}")
        if fat is not None:
            from mrsandbox.resources.qsm import QSMHead

            if fat != "qsm_rc2" or name == "qsm_rc2":
                raise KeyError(f"unknown fat layer {fat!r} for {name!r}; known: qsm_rc2, for "
                               f"icbm152 and bigbrain_mr")
            brain = cls.load(name)
            brain.fat = QSMHead.fat(name)
            return brain
        if name == "bigbrain_mr":
            return cls._load_bigbrain()
        if name == "qsm_rc2":
            return cls._load_qsm()
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
    def _load_qsm(cls):
        """The QSM challenge 2.0 head at 0.64 mm: a whole head, so the skull, the air,
        and the extracranial tissue are labelled and the fat layer has its geometry.
        The deposit is not redistributed; `Resources.path` says where to get it and
        where to place it (resources.qsm)."""
        volume = Volume.load("qsm_rc2/labels")
        labels = np.asarray(volume.data, dtype=np.uint8)
        masks = {tissue: np.isin(labels, members).astype(np.float32)
                 for tissue, members in cls.QSM_TISSUES.items()}
        masks["fat"] = cls.scalp(masks["fat"], masks["gm"] + masks["wm"] + masks["csf"],
                                 volume.affine)
        return cls(affine=volume.affine, name="qsm_rc2", **masks)

    #***********#
    #   scalp   #
    #***********#
    @classmethod
    def scalp(cls, layer, brain, affine):
        """The part of a soft tissue layer that lies beyond the bone, SCALP_FROM_MM or
        more from the brain: the sheet that lines the brain is not fat."""
        import scipy.ndimage

        cell = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
        distance = scipy.ndimage.distance_transform_edt(brain < 0.5, sampling=cell)
        return (layer * (distance >= cls.SCALP_FROM_MM)).astype(np.float32)

    @classmethod
    def from_maps(cls, gm, wm, csf, affine, name="custom", fat=None):
        """Wrap your own probability maps (for example an MPRAGE segmentation)."""
        return cls(gm=gm, wm=wm, csf=csf, affine=affine, name=name, fat=fat)

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
        """Fraction of the voxel box that holds scalp, the fat layer.

        A brain with a `fat` map weighs the box by that tissue. The tissue probability
        maps of the other presets carry no scalp, so the shell SCALP_MM from the brain
        stands in for it (the distance the scalp keeps on the QSM head, beyond the
        meninges and the bone): a voxel that touches the cortex holds none, as in vivo,
        and one that reaches a centimetre beyond the brain clips it. Counted only when
        the box holds brain as well.
        """
        if self.fat is None:
            values, weights = self.maps.box(position_mm, size_mm)
            if (weights.sum() <= 0.0
                    or not bool((values.sum(axis=-1)[weights > 0.0] > 0.5).any())):
                return 0.0
        mean = self.layer.mean(position_mm, size_mm)
        return 0.0 if mean is None else float(mean)

    @property
    def layer(self):
        """The fat layer as a Volume: the `fat` map, or without one the shell SCALP_MM
        from the brain, built on first use."""
        if self.fat is not None:
            return self.fat
        if self._shell is None:
            import scipy.ndimage

            cell = np.linalg.norm(self.affine[:3, :3], axis=0)
            distance = scipy.ndimage.distance_transform_edt(self.maps.data.sum(axis=-1) < 0.5,
                                                            sampling=cell)
            shell = (distance >= self.SCALP_MM[0]) & (distance <= self.SCALP_MM[1])
            self._shell = Volume(shell.astype(np.float32), self.affine,
                                 name=self.name + "/scalp")
        return self._shell

    def lipid_cells(self, position_mm, size_mm):
        """The fat inside a box, cell by cell: (offsets, volumes) with the cell centres
        in mm from the box centre, (n, 3), and the fat each holds inside the box in
        cubic mm, (n,); empty outside the layer."""
        values, weights = self.layer.box(position_mm, size_mm)
        held = (values * weights).ravel()
        if held.size == 0 or held.sum() <= 0.0:
            return np.empty((0, 3)), np.empty(0)
        keep = held > 0.0
        centres = self.layer.centres(position_mm, size_mm).reshape(-1, 3)[keep]
        cell_mm3 = abs(float(np.linalg.det(self.layer.affine[:3, :3])))
        return centres - np.asarray(position_mm, dtype=np.float64), held[keep] * cell_mm3

    def lipid_offset(self, position_mm, size_mm):
        """Where the fat layer sits: the centroid of the box's fat (the `layer` cells
        `lipid_weight` counts) relative to the voxel centre, (x, y, z) in millimetres;
        zeros when the box holds none."""
        source = self.layer
        values, weights = source.box(position_mm, size_mm)
        if weights.sum() <= 0.0:
            return np.zeros(3)
        outside = weights * values
        if outside.sum() <= 0.0:
            return np.zeros(3)
        centres = source.centres(position_mm, size_mm).reshape(-1, 3)
        return (np.average(centres, axis=0, weights=outside.ravel())
                - np.asarray(position_mm, dtype=np.float64))
