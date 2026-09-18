####################################################################################################
#                                              qsm.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The QSM reconstruction challenge 2.0 head phantom as named resources: whole-head       #
#          labels with bone, air, and extracranial tissue, and the quantitative maps behind them.  #
#                                                                                                  #
####################################################################################################

#**********************#
#   qsm challenge 2    #
#**********************#
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
    )
}


#**************************************************************************************************#
#                                          Class QSMHead                                           #
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
        import numpy as np

        from mrsandbox.phantom.volume import Volume

        if name not in cls.MAPS:
            raise KeyError(f"unknown QSM head map {name!r}; known: {', '.join(cls.MAPS)}")
        volume = Volume.load(cls.MAPS[name])
        data = volume.data
        if name == "T1map":
            data = np.where(data > 0.0, 1.0 / np.where(data > 0.0, data, 1.0), 0.0).astype(np.float32)
        return Volume(data, volume.affine, name="qsm_rc2/" + name)
