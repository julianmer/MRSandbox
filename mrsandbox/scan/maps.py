####################################################################################################
#                                              maps.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: What a quantitative map says per tissue, read on its subject's own tissue maps: the     #
#          water's T1 off an R1 or a T1 map (MPM, MP2RAGE), one record per tissue; and the FSL     #
#          steps the tissue maps come from (conform, strip, segment), shared by the console's      #
#          anatomy and the cohorts' builders.                                                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import pathlib

import numpy as np

from mrsandbox.scan.calibrations import Calibration
from mrsandbox.scan.run import _conditions

#***************#
#   constants   #
#***************#
# a cell is a tissue's own where its partial volume passes PURE (the rest mixes tissues, and the
# map's edges blur them); a tissue with fewer than FEW such cells gives no reading (stated)
PURE, FEW = 0.95, 100
TISSUES = ("gm", "wm", "csf")


#**************************************************************************************************#
#                                         Class MapReading                                         #
#**************************************************************************************************#
#                                                                                                  #
# A quantitative map read per tissue on its subject's own tissue maps.                             #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class MapReading:
    """A quantitative map read per tissue on its subject's own tissue maps (partial volumes on
    the map's grid, FSL fast of the same session's T1-weighted image): each tissue's values
    are the map in the cells where that tissue's share passes PURE, the reading their median,
    its range a quarter either side as every reading of one's own (the cells' spread is noise
    and partial volume, which a voxel averages away). `leaf` says what the map is of:
    water.t1_s off a T1 map in seconds (an R1 map in 1/s read as its inverse).

    Attributes:
        leaf: The PerTissue leaf the map reads, "water.t1_s".
        values: {tissue: the map's values in that tissue's own cells}.
        header: What the map's sidecar says of the scanner.
    """

    leaf: str
    values: dict
    header: dict

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, leaf, data, tissues, header=None):
        """`data` the map, `tissues` {gm, wm, csf: partial volume} on its grid; cells where the map
        is not finite or not above zero are left out (outside the head, a fit that failed)."""
        data = np.asarray(data, dtype=np.float64)
        valid = np.isfinite(data) & (data > 0.0)
        return cls(leaf=leaf,
                   values={tissue: data[valid & (np.asarray(tissues[tissue]) > PURE)]
                           for tissue in TISSUES},
                   header=dict(header or {}))

    #****************#
    #   from files   #
    #****************#
    @classmethod
    def from_files(cls, leaf, path, gm, wm, csf, r1=False, header=None):
        """A map and the three partial volumes as NIfTI files of one session, the map brought onto
        the tissues' grid by the files' affines (linear); `r1` reads an R1 map in 1/s as T1."""
        import nibabel as nib
        import nibabel.processing

        grid = nib.load(str(gm))
        data = np.asarray(
            nibabel.processing.resample_from_to(nib.load(str(path)), grid, order=1).dataobj,
            dtype=np.float64)
        if r1:
            data = np.divide(1.0, data, out=np.zeros_like(data), where=data > 0.0)
        return cls.from_arrays(leaf, data,
                               {tissue: np.asarray(nib.load(str(p)).dataobj)
                                for tissue, p in zip(TISSUES, (gm, wm, csf))},
                               header)

    #*************#
    #   records   #
    #*************#
    def records(self, dataset, citation=None, url=None, processing=None, method=""):
        """One Calibration per tissue with FEW own cells or more, `leaf`.gm, .wm, .csf: the
        median the value, a quarter either side the prior, the subject's (carried_by); the
        cells' 5th and 95th percentiles kept with the reading. `processing` says how the
        map and the tissues were made where the default does not; `method` how the map was
        measured, the rows' sequence family (scan.cohort pools a leaf's faithful methods,
        FAITHFUL)."""
        conditions = _conditions(self.header, dataset, citation, url, "quantitative map")
        conditions["sequence"]["family"] = method
        made = []
        for tissue, values in self.values.items():
            if values.size < FEW:
                continue
            low, median, high = (float(v) for v in np.percentile(values, (5.0, 50.0, 95.0)))
            made.append(Calibration(
                leaf=f"{self.leaf}.{tissue}", value=round(median, 4),
                prior=(round(0.75 * median, 4), round(1.25 * median, 4)),
                measured={"statistic": f"the map's median in the {tissue} cells past {PURE} of the "
                                       "partial volume (scan.maps.MapReading)",
                          "cells": int(values.size),
                          "percentiles_5_50_95": [round(low, 4), round(median, 4), round(high, 4)]},
                acquisition={}, voxel={},
                processing=processing or (
                    "the map onto the tissues' grid by the affines, linear; "
                    "the tissues FSL fast of the session's T1-weighted image"),
                **{**conditions, "note": "one subject's quantitative map; made by scan.maps"}))
        return tuple(made)


#***********#
#   tools   #
#***********#
def _fsl(*names):
    """FSL's tools by name, with the environment their runs take (FSL must be installed,
    fsl.fmrib.ox.ac.uk; it is no dependency of the package)."""
    import os
    import shutil

    tools = {name: shutil.which(name) or shutil.which(
                 name, path=os.path.expandvars("$FSLDIR/bin") + os.pathsep
                 + os.path.expanduser("~/fsl/share/fsl/bin"))
             for name in names}
    if None in tools.values():
        raise RuntimeError("FSL's " + ", ".join(names) + " are needed to segment an image "
                           "(fsl.fmrib.ox.ac.uk); or give the three tissue maps")
    environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get(
        "FSLDIR", str(pathlib.Path(next(iter(tools.values()))).parents[3]))}
    return tools, environment


#*************#
#   conform   #
#*************#
def conform(image):
    """The image on an axis-aligned 1 mm grid of 256 cells a side in scanner coordinates
    about its own centre, resampled by its affine (cubic), so an oblique acquisition is
    set straight."""
    import nibabel.processing

    grid = np.eye(4)
    grid[:3, 3] = (image.affine @ np.append(0.5 * (np.asarray(image.shape[:3]) - 1.0),
                                            1.0))[:3] - 127.5
    return nibabel.processing.resample_from_to(image, ((256, 256, 256), grid), order=3)


#***********#
#   strip   #
#***********#
def strip(path, folder, crop=True):
    """The brain mask of a structural image (FSL bet), written into `folder` and returned
    on the image's own grid. `crop` takes the neck first (robustfov): a sagittal 3D image
    reaches far below the brain, and bet then takes neck and scalp for brain; the cropped
    image only finds the brain, and its mask is carried back onto the whole image's grid
    by their affines."""
    import subprocess

    import nibabel as nib
    import nibabel.processing

    folder = pathlib.Path(folder)
    tools, environment = _fsl(*(("robustfov", "bet") if crop else ("bet",)))
    source = pathlib.Path(path)
    if crop:
        subprocess.run([tools["robustfov"], "-i", str(source), "-r", str(folder / "cropped")],
                       check=True, env=environment, capture_output=True)
        source = folder / "cropped"
    subprocess.run([tools["bet"], str(source), str(folder / "stripped"), "-R", "-f", "0.4",
                    "-m"], check=True, env=environment)
    mask = nib.load(str(folder / "stripped_mask.nii.gz"))
    if crop:
        mask = nibabel.processing.resample_from_to(mask, nib.load(str(path)), order=0)
    return mask


#*************#
#   segment   #
#*************#
def segment(path, folder, classes=3):
    """Tissue partial volumes of a brain-only T1-weighted image (FSL fast, `classes`
    classes): folder/tissue_pve_<k>.nii.gz on the image's own grid, ordered dark to bright
    (CSF, grey, white with three). Returns the folder."""
    import subprocess

    folder = pathlib.Path(folder)
    tools, environment = _fsl("fast")
    subprocess.run([tools["fast"], "-t", "1", "-n", str(classes), "-o", str(folder / "tissue"),
                    str(path)], check=True, env=environment)
    return folder


#*************#
#   tissues   #
#*************#
def tissues(path, folder, strip_on=None, classes=3, crop=True):
    """Tissue partial volumes of a structural image in `folder`: the brain found by `strip`
    on `strip_on` (the image itself unless given, e.g. an MP2RAGE's second inversion, a
    plain gradient echo with an ordinary background) and carried onto the image's grid,
    the image inside it (folder/brain.nii.gz) segmented by `segment`. Returns the brain
    mask on the image's grid."""
    import nibabel as nib
    import nibabel.processing

    folder = pathlib.Path(folder)
    image = nib.load(str(path))
    mask = nibabel.processing.resample_from_to(
        strip(strip_on if strip_on is not None else path, folder, crop=crop), image, order=0)
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.float32)
                             * (np.asarray(mask.dataobj) > 0), image.affine),
             str(folder / "brain.nii.gz"))
    segment(folder / "brain.nii.gz", folder, classes=classes)
    return mask


#**********************#
#   tissues by atlas   #
#**********************#
def tissues_by_atlas(path, folder):
    """Tissue partial volumes of an image whose contrast or background defeats `strip` and a
    plain `segment` (a radial acquisition's streaks, a weak grey-white contrast): FSL's MNI152
    template flirted onto it (twelve degrees of freedom, normalized mutual information, the
    contrasts differing), the template's brain mask carried over by the same transform, and
    fast guided by the standard tissue priors throughout (-P) on the image inside it:
    folder/tissue_pve_<k>.nii.gz as `segment` leaves them. Returns the brain mask."""
    import subprocess

    import nibabel as nib

    folder = pathlib.Path(folder)
    tools, environment = _fsl("flirt", "fast")
    standard = pathlib.Path(environment["FSLDIR"]) / "data" / "standard"
    subprocess.run([tools["flirt"], "-in", str(standard / "MNI152_T1_2mm"), "-ref", str(path),
                    "-dof", "12", "-cost", "normmi", "-omat", str(folder / "standard.mat")],
                   check=True, env=environment)
    subprocess.run([tools["flirt"], "-in", str(standard / "MNI152_T1_2mm_brain_mask"), "-ref",
                    str(path), "-applyxfm", "-init", str(folder / "standard.mat"), "-interp",
                    "nearestneighbour", "-out", str(folder / "stripped_mask")],
                   check=True, env=environment)
    image, mask = nib.load(str(path)), nib.load(str(folder / "stripped_mask.nii.gz"))
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.float32)
                             * (np.asarray(mask.dataobj) > 0.5), image.affine),
             str(folder / "brain.nii.gz"))
    subprocess.run([tools["fast"], "-t", "1", "-n", "3", "-a", str(folder / "standard.mat"), "-P",
                    "-o", str(folder / "tissue"), str(folder / "brain.nii.gz")],
                   check=True, env=environment)
    return mask
