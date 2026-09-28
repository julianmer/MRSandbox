####################################################################################################
#                                              maps.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: What a quantitative map says per tissue, read on its subject's own tissue maps: the     #
#          water's T1 off an R1 or a T1 map (MPM, MP2RAGE), one record per tissue.                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

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
    def records(self, dataset, citation=None, url=None):
        """One Calibration per tissue with FEW own cells or more, `leaf`.gm, .wm, .csf: the
        median the value, a quarter either side the prior, the subject's (carried_by); the
        cells' 5th and 95th percentiles kept with the reading."""
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
                processing="the map onto the tissues' grid by the affines, linear; "
                           "the tissues FSL fast of the session's T1-weighted image",
                **{**_conditions(self.header, dataset, citation, url, "quantitative map"),
                   "note": "one subject's quantitative map; made by scan.maps"}))
        return tuple(made)
