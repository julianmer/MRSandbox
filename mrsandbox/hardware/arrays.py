####################################################################################################
#                                             arrays.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: The receive head arrays of the scanners, as their papers, manuals and vendors describe  #
#          them: loops placed as published, what nobody publishes stated, and the Tim coils' mode  #
#          matrix, each preset naming what is cited and what is the package's.                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np

from mrsandbox.hardware.coils import CoilArray, Loop

#***************#
#   constants   #
#***************#
CENTER = np.asarray(CoilArray.HEAD_CENTER_MM, dtype=float)
# the Tim coils' inner surface: Head/Neck 20's inner diameters, 26.5 cm vertical (anterior to
# posterior, supine) and 23 cm horizontal (Schmitt and Rieger 2021, Front Neurosci 15:735290),
# lent to the Head Matrix and the mMR coil (stated)
TIM_SEMI_MM = (115.0, 132.5)
TIM_SOURCE = ("Reykowski and Blasche, ISMRM 2004 #1587 (the mode matrix: P primary, CP-like for "
              "the head centre; S secondary; P+S+T equivalent to the raw loops); Reykowski et "
              "al., ISMRM 2004 #2390 and 2005 #908 (Head Matrix: 4 clusters of 3 loops, Anterior "
              "1 and 2, Posterior 1 and 2, two rings of 6; Neck Matrix: 2 clusters of 2, a 90 "
              "degree hybrid, forward and reverse CP)")
MODES = ("the modes: each cluster's loops phased to add at the head centre and combined by the "
         "unitary DFT (stated)")
# the EN960 head form of Wiggins et al. 2005: 222 mm AP, 181 RL, 210 SI; the loops 10 mm out from
# it (stated)
HEAD_FORM_SEMI_MM = (90.5 + 10.0, 111.0 + 10.0, 105.0 + 10.0)


#**************************************************************************************************#
#                                         Class ArrayPreset                                        #
#**************************************************************************************************#
#                                                                                                  #
# One named receive array: how its loops are laid, what is published, what is stated.              #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class ArrayPreset:
    """One named receive array as its sources describe it.

    Attributes:
        label: The console's words for it.
        vendor: Who makes it ("research" for a lab's build).
        channels: How many channels it gives.
        names: The names a file's header gives it (ReceiveCoilName, the coil labels' prefix).
        cited: What is published, with its source.
        stated: What the package assumed where nothing is published.
        build: Makes the CoilArray.
    """

    label: str
    vendor: str
    channels: int
    names: tuple
    cited: str
    stated: str
    build: object = dataclasses.field(compare=False, repr=False)

    #***********#
    #   array   #
    #***********#
    def array(self):
        """The CoilArray, its loops placed and its channels formed."""
        made = self.build()
        if len(made) != self.channels:
            raise ValueError(f"{self.label} builds {len(made)} channels, not {self.channels}")
        return made


#********************#
#   elliptic rings   #
#********************#
def _rings(rings, semi_mm, loop_radius_mm, offset_deg=0.0):
    """Loops on an elliptic cylinder about the head's centre, normals inward: `rings` is ((z from
    the centre in mm, loops in the ring), ...), `semi_mm` the (x, y) semi-axes; angles counted from
    +x, the first loop at `offset_deg`."""
    loops = []
    for z_mm, count in rings:
        for k in range(count):
            phi = np.radians(offset_deg) + 2.0 * np.pi * k / count
            at = np.array([semi_mm[0] * np.cos(phi), semi_mm[1] * np.sin(phi), 0.0])
            inward = -np.array([at[0] / semi_mm[0] ** 2, at[1] / semi_mm[1] ** 2, 0.0])
            loops.append(Loop(loop_radius_mm, tuple(CENTER + at + np.array([0.0, 0.0, z_mm])),
                              normal=tuple(inward / np.linalg.norm(inward))))
    return loops


#**************#
#   head cap   #
#**************#
def _cap(count, semi_mm, polar_max_deg, radii_mm):
    """`count` loops spread evenly over an ellipsoid's cap from the vertex to `polar_max_deg` (the
    golden-angle spiral, equal area per loop), normals inward; `radii_mm` one radius a loop in
    order from the vertex."""
    semi = np.asarray(semi_mm, dtype=float)
    top = 1.0 - np.cos(np.radians(polar_max_deg))
    loops = []
    for k in range(count):
        theta = np.arccos(1.0 - top * (k + 0.5) / count)
        phi = k * np.pi * (3.0 - np.sqrt(5.0))
        out = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        inward = -(out / semi) / np.linalg.norm(out / semi)
        loops.append(Loop(float(radii_mm[k]), tuple(CENTER + semi * out), normal=tuple(inward)))
    return loops


#***************#
#   tim modes   #
#***************#
def _modes(loops, clusters, per_cluster):
    """The Tim mode matrix, (channels, loops): for each cluster (loop indices), its loops phased by
    their sensitivity at the head centre so they add there, combined by the unitary DFT of the
    cluster's size; the first `per_cluster` modes kept (1: CP, the primary alone; 2: dual; 3:
    triple)."""
    matrix = []
    for cluster in clusters:
        phases = np.exp(-1j * np.angle([loops[i].sensitivity(CENTER) for i in cluster]))
        n = len(cluster)
        for mode in range(per_cluster):
            row = np.zeros(len(loops), dtype=np.complex128)
            row[list(cluster)] = phases * np.exp(2j * np.pi * mode * np.arange(n) / n) / np.sqrt(n)
            matrix.append(row)
    return np.asarray(matrix)


#*****************#
#   head matrix   #
#*****************#
def _head_matrix(per_cluster, neck=False):
    """The Head Matrix's 12 loops, two rings of 6 (stated as circles of the pitch, 60 mm radius, 60
    mm above and below the head's centre), its clusters the anterior and posterior three of each
    ring; with `neck` the Neck Matrix's 4 loops in a ring below (stated at 165 mm below the centre,
    45 mm radius) in 2 clusters of 2, dual."""
    loops = _rings(((60.0, 6), (-60.0, 6)), TIM_SEMI_MM, 60.0, offset_deg=30.0)
    # Anterior 1, 2 (y > 0), Posterior 1, 2
    clusters = [(0, 1, 2), (6, 7, 8), (3, 4, 5), (9, 10, 11)]
    matrix = _modes(loops, clusters, per_cluster)
    if not neck:
        return CoilArray(loops, modes=matrix)
    neck_loops = _rings(((-165.0, 4),), TIM_SEMI_MM, 45.0, offset_deg=45.0)
    everything = loops + neck_loops
    head = np.pad(matrix, ((0, 0), (0, 4)))
    neck_modes = _modes(everything, [(12, 13), (14, 15)], 2)
    return CoilArray(everything, modes=np.vstack([head, neck_modes]))


#*************#
#   presets   #
#*************#
PRESETS = {
    "siemens_head_neck_20": ArrayPreset(
        "Head/Neck 20", "siemens", 20, ("HeadNeck_20",),
        cited="20 loops, two rings of 8 and one of 4, inner diameters 26.5 cm vertical and 23 cm "
              "horizontal (Schmitt and Rieger 2021, Front Neurosci 15:735290, reporting Siemens)",
        stated="circular loops of the ring's pitch (48 mm radius) on the published ellipse, the "
               "head rings 48 mm above and below the head's centre, the neck ring 145 mm below, "
               "the rings aligned; every loop its own channel",
        build=lambda: CoilArray(_rings(((48.0, 8), (-48.0, 8), (-145.0, 4)), TIM_SEMI_MM, 48.0,
                                       offset_deg=22.5))),
    "siemens_head_matrix_12_cp": ArrayPreset(
        "Head Matrix 12, CP mode", "siemens", 4, ("HeadMatrix",), cited=TIM_SOURCE,
        stated="the loops as circles on Head/Neck 20's ellipse (60 mm radius, rings 60 mm above "
               "and below the centre); " + MODES,
        build=lambda: _head_matrix(1)),
    "siemens_head_matrix_12_dual": ArrayPreset(
        "Head Matrix 12, dual mode", "siemens", 8, ("HeadMatrix",), cited=TIM_SOURCE,
        stated="the loops as circles on Head/Neck 20's ellipse (60 mm radius, rings 60 mm above "
               "and below the centre); " + MODES,
        build=lambda: _head_matrix(2)),
    "siemens_head_matrix_12_triple": ArrayPreset(
        "Head Matrix 12, triple mode", "siemens", 12, ("HeadMatrix",), cited=TIM_SOURCE,
        stated="the loops as circles on Head/Neck 20's ellipse (60 mm radius, rings 60 mm above "
               "and below the centre); " + MODES,
        build=lambda: _head_matrix(3)),
    "siemens_head_neck_mrpet": ArrayPreset(
        "mMR Head/Neck, 12 + 4", "siemens", 16, ("HeadNeck_MRPET",),
        cited="12 head and 4 neck channels, a split design (Siemens' Biograph mMR brochure; "
              "Navarro de Lara et al. 2019, Sensors 19:3297); " + TIM_SOURCE,
        stated="built as the Head Matrix in triple mode and the Neck Matrix in dual mode (the 12 + "
               "4 fits it); " + MODES,
        build=lambda: _head_matrix(3, neck=True)),
    "siemens_32ch_head": ArrayPreset(
        "32-ch Head", "siemens", 32, ("32Ch_Head", "32Ch Head", "Head_32"),
        cited="32 elements, 12 upper and 20 lower, housing 300 x 309 x 290 mm (Siemens' product "
              "page)",
        stated="the layout is unpublished: 32 loops spread evenly over the head form's cap to 110 "
               "degrees from the vertex, 40 mm radius (the MGH build's hexagon loops)",
        build=lambda: CoilArray(_cap(32, HEAD_FORM_SEMI_MM, 110.0, [40.0] * 32))),
    "ge_nova_32": ArrayPreset(
        "32-ch Head (Nova 3832016)", "ge", 32, ("32Ch Head", "Nova32"),
        cited="32 loops, 8 columns of 4 from the vertex to the base, the top 4 x 4 anterior and "
              "the bottom 4 x 4 posterior, housing 44 x 38.2 x 29.25 cm (Nova Medical's user "
              "manual, model 3832016, the MR750's head coil)",
        stated="the rows at 25, 50, 75 and 100 degrees from the vertex on the package's head "
               "ellipsoid, the columns every 45 degrees, 34 mm radius loops (the mean of the two "
               "pitches)",
        build=lambda: CoilArray([loop for theta in (25.0, 50.0, 75.0, 100.0)
                                 for loop in CoilArray.helmet(rows=((theta, 8),),
                                                              loop_radius_mm=34.0).loops])),
    "philips_32ch_sense_head": ArrayPreset(
        "32-ch SENSE Head", "philips", 32, ("SENSE-Head-32", "HeadNeck_32"),
        cited="inner diameter 26 cm at the opening, irregular polygonal loops (Panman et al. 2019, "
              "Front Neurosci 13:729, Suppl. Fig. 1, reprinting Philips)",
        stated="the layout is unpublished: 32 loops spread evenly over a cap of 130 mm radius to "
               "110 degrees, 40 mm radius",
        build=lambda: CoilArray(_cap(32, (130.0, 130.0, 130.0), 110.0, [40.0] * 32))),
    "philips_8ch_sense_head": ArrayPreset(
        "8-ch SENSE Head", "philips", 8, ("SENSE-Head-8",),
        cited="inner diameter 25.5 cm, large rectangular loops around a cylinder (Panman et al. "
              "2019, Suppl. Fig. 1)",
        stated="a ring of 8 circles of the pitch (50 mm radius) on the 25.5 cm cylinder at the "
               "head's centre",
        build=lambda: CoilArray(_rings(((0.0, 8),), (127.5, 127.5), 50.0, offset_deg=22.5))),
    "mgh_32ch_wiggins": ArrayPreset(
        "MGH 32-ch soccer ball (Wiggins 2006)", "research", 32, (),
        cited="32 loops on 26 hexagonal and 6 pentagonal tiles of about 40 mm sides, loops about "
              "80 mm (hexagons) and 60 mm (pentagons), on the EN960 head form (222 AP, 181 RL, 210 "
              "SI mm) (Wiggins et al. 2005, ISMRM #679; 2006, Magn Reson Med 56:216)",
        stated="the tiling's centres spread evenly over the head form's cap to 110 degrees, 10 mm "
               "out from it, the six nearest the vertex the pentagons",
        build=lambda: CoilArray(_cap(32, HEAD_FORM_SEMI_MM, 110.0, [30.0] * 6 + [40.0] * 26))),
    "mgh_64ch_keil": ArrayPreset(
        "MGH 64-ch (Keil 2013)", "research", 64, (),
        cited="64 loops, 42 posterior and 22 anterior, 65 mm hexagons and 55 mm pentagons on the "
              "95th percentile head contour dilated by about 3 mm (Keil et al. 2013, Magn Reson "
              "Med 70:248)",
        stated="spread evenly over the head form's cap to 115 degrees, the twelve nearest the "
               "vertex the pentagons",
        build=lambda: CoilArray(_cap(64, HEAD_FORM_SEMI_MM, 115.0, [27.5] * 12 + [32.5] * 52))),
}
