####################################################################################################
#                                              qsm.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The QSM reconstruction challenge 2.0 head phantom as named resources: whole-head        #
#          labels with bone, air, and extracranial tissue, and the quantitative maps behind them.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

#*********************#
#   qsm challenge 2   #
#*********************#
# Source: Marques, J. P., et al. QSM reconstruction challenge 2.0: a realistic in silico head
#         phantom for MRI data simulation and evaluation of susceptibility mapping procedures.
#         Magn Reson Med 86 (2021) 526-542. One healthy volunteer at 7 T (MP2RAGEME), 0.64 mm.
# Data:   doi:10.34973/m20r-jt17, the Donders Repository, under its data use agreement
#         RU-DI-HD-1.0: no redistribution, every user requests access. The entries carry a
#         page, not a url: nothing is fetched, the files are placed in the cache by hand with
#         the archive's own paths (its MANIFEST.txt lists a SHA256 per file).
# Units:  labels 1 to 16 as the deposit's chimodel/paramaters.mat names them (LABELS; 12 has
#         no voxels); R1 in 1/s, R2* in 1/s, M0 arbitrary, chi in ppm.
# Notes:  the files carry no world origin; origin_index puts (0, 0, 0) at the centre of mass of
#         the brain labels 1 to 10 (stated), the index axes are left-right, posterior to
#         anterior (the field of the sinuses settles the sense), and inferior to superior.
_PAGE = "https://doi.org/10.34973/m20r-jt17"
_ORIGIN = (130.0, 143.0, 163.0)

LABELS = {1: "caudate", 2: "globus pallidus", 3: "putamen", 4: "red nucleus", 5: "dentate nucleus",
          6: "substantia nigra and subthalamic nucleus", 7: "thalamus", 8: "white matter",
          9: "gray matter", 10: "csf", 11: "blood", 12: "fat", 13: "bone", 14: "air",
          15: "muscle", 16: "calcification"}

QSM_RC2 = {
    f"qsm_rc2/{name}": {"page": _PAGE, "filename": f"qsm_rc2/{path}", "units": units,
                        "origin_index": _ORIGIN}
    for name, path, units in (
        ("labels", "data/masks/SegmentedModel.nii.gz", "label"),
        ("r1", "data/maps/R1.nii.gz", "1/s"),
        ("r2star", "data/maps/R2star.nii.gz", "1/s"),
        ("m0", "data/maps/M0.nii.gz", "arbitrary"),
        ("chi", "data/chimodel/Chi.nii.gz", "ppm"),
        # the MEASURED total field of the same head at 7 T (MP2RAGE-ME, the dataset's raw
        # folder), resampled by the authors onto the R2* map's grid: its own header carries
        # another origin and zooms, but its cells are the model's, so it takes the model's
        # origin like the others
        ("total_field", "data/raw/MP2RAGEME_total-field_ppm_head_r2s-header.nii.gz", "ppm"),
    )
}


#**************************************************************************************************#
#                                           Class QSMHead                                          #
#**************************************************************************************************#
#                                                                                                  #
# The quantitative maps of the QSM challenge 2.0 head, served on the head's own grid.              #
#                                                                                                  #
#**************************************************************************************************#
class QSMHead:
    """The quantitative maps of the QSM challenge 2.0 head, served on the head's own
    grid: one subject at 7 T whose labels are the `Brain.load("qsm_rc2")` brain and
    whose susceptibility is the `FieldMap.load("qsm_rc2")` field, so tissue, field,
    and relaxation maps are one coherent subject. Names follow qMRI-BIDS; T1map is
    the reciprocal of R1map where that is positive, zero elsewhere (air and bone).
    """

    MAPS = {"R1map": "qsm_rc2/r1", "T1map": "qsm_rc2/r1", "R2starmap": "qsm_rc2/r2star",
            "M0map": "qsm_rc2/m0", "Chimap": "qsm_rc2/chi"}

    #*************#
    #   volumes   #
    #*************#
    @classmethod
    def volume(cls, name):
        """One map as a Volume in the head's world (mm from the brain's centre)."""
        from mrsandbox.phantom.volume import Volume

        if name not in cls.MAPS:
            raise KeyError(f"unknown QSM head map {name!r}; known: {', '.join(cls.MAPS)}")
        volume = Volume.load(cls.MAPS[name])
        data = volume.data
        if name == "T1map":
            data = np.where(data > 0.0, 1.0 / np.where(data > 0.0, data, 1.0),
                            0.0).astype(np.float32)
        return Volume(data, volume.affine, name="qsm_rc2/" + name)

    #************#
    #   params   #
    #************#
    @classmethod
    def params(cls):
        """The head's relaxation maps as map-valued leaves for a scan of its own brain
        (`Brain.load("qsm_rc2")`, the same grid): every voxel then reads its local
        values in place of the per-tissue priors.

        water.t1_s is T1map, 1 / R1. lineshape.gauss_hz is the width of the
        microscopic field distribution, R2' / pi with R2' = R2* - R2 floored at zero,
        R2 being the 7 T water T2 of the scan defaults (database.relaxation
        WATER_FIELDS, 47 ms in tissue, and the CSF value) weighted by each cell's
        tissue. R2' is a Lorentzian rate and the leaf a Gaussian width; the two are
        matched by their full width at half maximum (stated). The head is one subject
        at 7 T: at another field the T1 does not hold, and R2' scales with the field
        (stated, not applied).
        """
        from mrsandbox.database.relaxation import WATER, WATER_FIELDS
        from mrsandbox.params import Mapped, Params
        from mrsandbox.phantom.brain import Brain
        from mrsandbox.phantom.volume import Volume

        brain, r2star = Brain.load("qsm_rc2"), cls.volume("R2starmap")
        t2 = {**WATER["t2_s"], **WATER_FIELDS[7.0]["t2_s"]}
        tissue = brain.gm + brain.wm + brain.csf
        r2 = ((brain.gm / t2["gm"] + brain.wm / t2["wm"] + brain.csf / t2["csf"])
              / np.where(tissue > 0.0, tissue, 1.0))
        width = np.maximum(r2star.data - r2, 0.0) / np.pi
        t1 = cls.volume("T1map").data
        # a voxel reads a map as its box mean, so the cells that hold no brain (and the few
        # with a vanishing R1, a T1 beyond CSF's) take the brain's median: neutral in the
        # mean, where a zero or a hundred seconds would pull it (stated)
        inside = (tissue > 0.5) & (t1 > 0.2) & (t1 < 5.0)
        t1 = np.where(inside, t1, np.median(t1[inside])).astype(np.float32)
        width = np.where(inside, width, np.median(width[inside])).astype(np.float32)
        return Params(water=dict(t1_s=Mapped(Volume(t1, r2star.affine, name="qsm_rc2/T1map"))),
                      lineshape=dict(gauss_hz=Mapped(Volume(width, r2star.affine,
                                                            name="qsm_rc2/R2primewidth"))))

    #******************#
    #   registration   #
    #******************#
    @classmethod
    def registration(cls, space):
        """The world-to-world affine carrying the head into the world of another brain
        preset ("icbm152" or "bigbrain_mr"), fitted once and cached: a shape fit
        (phantom.registration.register) of the head's brain, its labels 1 to 10,
        against the preset's brain tissue. Another subject's head on this brain: an
        affine follows the size and the pose, not the folds (stated)."""
        from mrsandbox.phantom.brain import Brain
        from mrsandbox.phantom.registration import register
        from mrsandbox.phantom.volume import Volume
        from mrsandbox.resources import Resources

        if space not in ("icbm152", "bigbrain_mr"):
            raise KeyError(f"unknown space {space!r}; known: icbm152, bigbrain_mr")
        cache = Resources.cache_dir() / f"qsm_rc2/to_{space}.npy"
        if not cache.exists():
            labels = Volume.load("qsm_rc2/labels")
            # halved to 1.28 mm cells for the fit
            head = (labels.data[::2, ::2, ::2] >= 1) & (labels.data[::2, ::2, ::2] <= 10)
            halved = labels.affine @ np.diag([2.0, 2.0, 2.0, 1.0])
            target = Brain.load(space)
            tissue = (target.maps.data.sum(axis=-1) > 0.5).astype(np.float32)
            np.save(cache, register(Volume(head.astype(np.float32), halved),
                                    Volume(tissue, target.affine)))
        return np.load(cache)

    #*********#
    #   fat   #
    #*********#
    @classmethod
    def fat(cls, space, cell_mm=1.0):
        """The head's fat-bearing layer (label 15, Brain.QSM_TISSUES) carried into
        another brain preset's world and resampled there, linearly, onto an
        axis-aligned grid of `cell_mm` cells (so box queries stay exact), as a Volume
        of fractions, less what the preset's own tissue already fills there (a place is
        tissue or fat, and the brain is the preset's); derived once and cached beside
        the head's files, never shipped. It lends the presets without a scalp, ICBM152
        and BigBrain-MR, the geometry of a fat layer: `Brain.load(space, fat="qsm_rc2")`."""
        import scipy.ndimage

        from mrsandbox.phantom.brain import Brain
        from mrsandbox.phantom.volume import Volume
        from mrsandbox.resources import Resources

        cache = Resources.cache_dir() / f"qsm_rc2/scalp_in_{space}_{cell_mm:g}mm.npz"
        if not cache.exists():
            labels = Volume.load("qsm_rc2/labels")
            layer = np.isin(labels.data, Brain.QSM_TISSUES["fat"]).astype(np.float32)
            inside = np.isin(labels.data,
                             sum((Brain.QSM_TISSUES[t] for t in ("gm", "wm", "csf")), ()))
            layer = Brain.scalp(layer, inside.astype(np.float32), labels.affine)
            carried = cls.registration(space) @ labels.affine
            corners = np.array([[i, j, k, 1.0] for i in (0, labels.shape[0] - 1)
                                for j in (0, labels.shape[1] - 1)
                                for k in (0, labels.shape[2] - 1)])
            world = (carried @ corners.T)[:3]
            low, high = np.floor(world.min(axis=1)), np.ceil(world.max(axis=1))
            shape = tuple(int(n) for n in np.rint((high - low) / cell_mm) + 1)
            affine = np.diag([cell_mm, cell_mm, cell_mm, 1.0])
            affine[:3, 3] = low
            index = np.linalg.inv(carried) @ affine  # target index to head index
            data = scipy.ndimage.affine_transform(layer, index[:3, :3], offset=index[:3, 3],
                                                  output_shape=shape, order=1, cval=0.0)
            brain = Brain.load(space)
            index = np.linalg.inv(brain.affine) @ affine  # target index to brain index
            tissue = scipy.ndimage.affine_transform(
                brain.maps.data.sum(axis=-1), index[:3, :3], offset=index[:3, 3],
                output_shape=shape, order=1, cval=0.0)
            data = np.minimum(data, 1.0 - np.clip(tissue, 0.0, 1.0))
            np.savez_compressed(cache, data=data.astype(np.float32), affine=affine)
        stored = np.load(cache)
        return Volume(stored["data"], stored["affine"], name=f"qsm_rc2/fat_in_{space}")
