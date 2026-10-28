####################################################################################################
#                                            regions.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 28/10/26                                                                                #
#                                                                                                  #
# Purpose: The brain's regions: the lobes, the deep grey nuclei, the brainstem and the cerebellum  #
#          as one vocabulary, with the atlases that place them on a brain's grid (CerebrA for the  #
#          brains in the MNI frame, BigBrain-MR's own labels for its brain). Each region carries   #
#          its own metabolite levels (database.regional) over the tissue fractions it holds.       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import csv

import numpy as np

#*************#
#   regions   #
#*************#
# The vocabulary, index 0 for cells no atlas reaches. A lobe holds grey and white matter by the
# tissue maps (its white matter the lobe of the nearest cortex); "cortex" and "white" stand for
# the cerebrum of a brain whose atlas has no lobes.
REGIONS = ("none", "frontal", "parietal", "temporal", "occipital", "insula", "anterior_cingulate",
           "posterior_cingulate", "cortex", "white", "thalamus", "caudate", "putamen", "pallidum",
           "accumbens", "hippocampus", "amygdala", "diencephalon", "basal_forebrain",
           "substantia_nigra", "brainstem", "cerebellum", "vermis", "csf")
LOBES = ("frontal", "parietal", "temporal", "occipital", "insula", "anterior_cingulate",
         "posterior_cingulate", "cortex", "white")

#*************#
#   cerebra   #
#*************#
# CerebrA's labels (resources.icbm CEREBRA) by region, both hemispheres alike. The cortical
# labels follow the Desikan-Killiany lobes (Desikan et al. 2006, NeuroImage 31:968), the
# paracentral and precentral gyri frontal, the postcentral parietal, the cingulate and insula
# apart; the optic chiasm counts as diencephalon (stated).
CEREBRA = {
    "frontal": ("Caudal Middle Frontal", "Lateral Orbitofrontal", "Medial Orbitofrontal",
                "Paracentral", "Pars Opercularis", "Pars Orbitalis", "Pars Triangularis",
                "Precentral", "Rostral Middle Frontal", "Superior Frontal"),
    "parietal": ("Inferior Parietal", "Postcentral", "Precuneus", "Superior Parietal",
                 "Supramarginal"),
    "temporal": ("Entorhinal", "Fusiform", "Inferior temporal", "Middle Temporal",
                 "Parahippocampal", "Superior Temporal", "Transverse Temporal"),
    "occipital": ("Cuneus", "Lateral Occipital", "Lingual", "Pericalcarine"),
    "insula": ("Insula",),
    "anterior_cingulate": ("Caudal Anterior Cingulate", "Rostral Anterior Cingulate"),
    "posterior_cingulate": ("Posterior Cingulate", "Isthmus Cingulate"),
    "thalamus": ("Thalamus",),
    "caudate": ("Caudate",),
    "putamen": ("Putamen",),
    "pallidum": ("Pallidum",),
    "accumbens": ("Accumbens Area",),
    "hippocampus": ("Hippocampus",),
    "amygdala": ("Amygdala",),
    "diencephalon": ("Ventral Diencephalon", "Optic Chiasm"),
    "basal_forebrain": ("Basal Forebrain",),
    "brainstem": ("Brainstem",),
    "cerebellum": ("Cerebellum Gray Matter", "Cerebellum White Matter"),
    "vermis": ("Vermal lobules I-V", "Vermal lobules VI-VII", "Vermal lobules VIII-X"),
    "csf": ("Lateral Ventricle", "Inferior Lateral Ventricle", "Third Ventricle",
            "Fourth Ventricle"),
}

#**************#
#   bigbrain   #
#**************#
# BigBrain-MR's 20 labels (Sainz Martinez et al. 2023, Table I) by region; both pallidal
# segments are the pallidum, the subthalamic nuclei and the pineal gland count as diencephalon,
# the red nuclei as brainstem (stated).
BIGBRAIN = {1: "white", 2: "cortex", 3: "csf", 4: "cerebellum", 5: "cerebellum", 6: "thalamus",
            7: "caudate", 8: "putamen", 9: "pallidum", 10: "pallidum", 11: "basal_forebrain",
            12: "accumbens", 13: "brainstem", 14: "hippocampus", 15: "diencephalon",
            16: "amygdala", 17: "diencephalon", 18: "brainstem", 19: "substantia_nigra",
            20: "diencephalon"}


#*************#
#   numbers   #
#*************#
def numbered(labels, regions):
    """An atlas's label volume as region indices: `regions` maps each label number to a
    region name; labels it does not name become 0."""
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for label, region in regions.items():
        if label < len(lookup):
            lookup[label] = REGIONS.index(region)
    return lookup[np.asarray(labels, dtype=np.int64)]


#*************#
#   atlases   #
#*************#
def cerebra():
    """CerebrA as region indices on its own 1 mm grid, and its affine. Its unlabelled cells, the
    cerebral white matter among them, take the lobe of the nearest labelled cortex, so the white
    matter under each lobe belongs to it (and cells outside the brain too, where the tissue maps
    hold nothing)."""
    import nibabel as nib
    import scipy.ndimage

    from mrsandbox.resources import Resources

    with open(Resources.path("cerebra/names"), newline="") as f:
        rows = list(csv.DictReader(f))
    names = {name: region for region, members in CEREBRA.items() for name in members}
    by_number = {int(row[side]): names[row["Label Name"].strip()]
                 for row in rows for side in ("RH Label", "LH Labels")}
    image = nib.load(str(Resources.path("cerebra/labels")))
    indices = numbered(np.rint(image.get_fdata()).astype(np.int64), by_number)
    lobes = np.isin(indices, [REGIONS.index(lobe) for lobe in LOBES])
    _, nearest = scipy.ndimage.distance_transform_edt(~lobes, return_indices=True)
    filled = np.where(indices > 0, indices, indices[tuple(nearest)])
    return filled.astype(np.uint8), np.array(image.affine, dtype=np.float64)


def resample(indices, affine, target_affine, target_shape):
    """Region indices carried onto another grid by the world frame, each target cell taking the
    region of the nearest source cell; 0 where the source does not reach."""
    import scipy.ndimage

    matrix = np.linalg.inv(affine) @ np.asarray(target_affine, dtype=np.float64)
    return scipy.ndimage.affine_transform(indices, matrix[:3, :3], offset=matrix[:3, 3],
                                          output_shape=tuple(target_shape), order=0,
                                          mode="constant", cval=0)
