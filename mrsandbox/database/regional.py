####################################################################################################
#                                            regional.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 28/10/26                                                                                #
#                                                                                                  #
# Purpose: Metabolite concentrations by brain region, one dict per source, the values as published #
#          under the papers' own region and metabolite names, with the citation beside each. Each  #
#          source says which of the package's regions (phantom.regions) its rows stand for and in  #
#          which tissue (None for a nucleus, all of it), its reference rows and the metabolites it #
#          quantifies; how the regions become factors is code (params.Priors.regional).            #
#                                                                                                  #
####################################################################################################

#*****************#
#   wright 2024   #
#*****************#
# Source: Wright, A. M., Ziegs, T., Henning, A. Quantitative metabolic reference for healthy human
#         cerebrum derived from group averaged 9.4T 1H MRSI data. bioRxiv (2024),
#         doi 10.1101/2024.04.04.587927, Table 2: group-averaged maps of eight volunteers in the
#         MNI152 frame, masked by region.
# Units:  (mean, sd) in mmol per kg.
# Notes:  the temporal lobe grey matter lies in the slices the paper calls unstable (sds up to
#         twice the means), and the corpus callosum is no region of the package: neither stands
#         for one; the reference is the mean of the lobes in each tissue.
WRIGHT_2024 = {
    "values": {
        "Frontal Lobe GM": {
            "NAA": (9.39, 0.73), "tCr": (7.57, 0.32), "Glu": (8.41, 0.39), "mI": (4.06, 0.49),
            "tCho": (1.08, 0.08), "Gln": (2.37, 0.26), "NAAG": (1.78, 0.92), "Asp": (1.02, 0.18),
            "GABA": (0.86, 0.08), "GSH": (0.96, 0.12), "Tau": (2.64, 0.51), "Scyllo": (0.3, 0.04)
        },
        "Temporal Lobe GM": {
            "NAA": (9.99, 0.66), "tCr": (10.76, 6.33), "Glu": (14.15, 12.98), "mI": (8.65, 8.89),
            "tCho": (1.22, 0.12), "Gln": (8.01, 13.19), "NAAG": (2.8, 1.3), "Asp": (0.82, 0.26),
            "GABA": (0.85, 0.16), "GSH": (3.51, 3.84), "Tau": (22.16, 26.51),
            "Scyllo": (0.36, 0.04)
        },
        "Parietal Lobe GM": {
            "NAA": (10.23, 0.69), "tCr": (8.48, 0.37), "Glu": (9.07, 0.41), "mI": (4.14, 0.41),
            "tCho": (1.14, 0.08), "Gln": (2.63, 0.33), "NAAG": (1.65, 0.54), "Asp": (1.12, 0.27),
            "GABA": (1, 0.1), "GSH": (1.03, 0.16), "Tau": (2.76, 0.32), "Scyllo": (0.31, 0.03)
        },
        "Occipital Lobe GM": {
            "NAA": (10.4, 1.2), "tCr": (8.02, 0.74), "Glu": (8.94, 1.48), "mI": (3.62, 0.46),
            "tCho": (1, 0.14), "Gln": (2.74, 1.08), "NAAG": (2.07, 0.79), "Asp": (0.8, 0.4),
            "GABA": (0.83, 0.21), "GSH": (1.08, 0.33), "Tau": (2.78, 0.91), "Scyllo": (0.37, 0.1)
        },
        "Frontal Lobe WM": {
            "NAA": (11.16, 0.34), "tCr": (9.54, 0.66), "Glu": (9.5, 0.76), "mI": (5.24, 0.96),
            "tCho": (1.66, 0.11), "Gln": (3.22, 0.57), "NAAG": (3.09, 1.49), "Asp": (1.63, 0.31),
            "GABA": (1.38, 0.16), "GSH": (1.63, 0.34), "Tau": (5.94, 7.16),
            "Scyllo": (0.44, 0.05)
        },
        "Occipital Lobe WM": {
            "NAA": (11.27, 0.69), "tCr": (8.91, 0.68), "Glu": (8.05, 0.4), "mI": (5.63, 0.99),
            "tCho": (1.57, 0.16), "Gln": (3.06, 0.72), "NAAG": (2.47, 0.6), "Asp": (1.33, 0.38),
            "GABA": (1.11, 0.27), "GSH": (1.3, 0.2), "Tau": (3.41, 1.38), "Scyllo": (0.29, 0.07)
        },
        "Parietal Lobe WM": {
            "NAA": (12.4, 0.62), "tCr": (9.5, 0.79), "Glu": (9.34, 1.51), "mI": (5.03, 0.77),
            "tCho": (1.6, 0.12), "Gln": (3.22, 0.97), "NAAG": (2.92, 1.05), "Asp": (1.71, 0.35),
            "GABA": (1.44, 0.27), "GSH": (1.28, 0.2), "Tau": (4.25, 2.01), "Scyllo": (0.49, 0.09)
        },
        "Corpus Callosum": {
            "NAA": (8.64, 1.47), "tCr": (7.15, 1.08), "Glu": (7.24, 1.27), "mI": (4.36, 0.71),
            "tCho": (1.18, 0.22), "Gln": (2.29, 0.59), "NAAG": (2.26, 0.96), "Asp": (0.92, 0.35),
            "GABA": (0.83, 0.2), "GSH": (1, 0.23), "Tau": (7.5, 14.68), "Scyllo": (0.28, 0.06)
        },
    },
    "stands_for": {
        "Frontal Lobe GM": (("frontal",), "gm"), "Parietal Lobe GM": (("parietal",), "gm"),
        "Occipital Lobe GM": (("occipital",), "gm"), "Frontal Lobe WM": (("frontal",), "wm"),
        "Parietal Lobe WM": (("parietal",), "wm"), "Occipital Lobe WM": (("occipital",), "wm"),
    },
    "reference": {"gm": ("Frontal Lobe GM", "Parietal Lobe GM", "Occipital Lobe GM"),
                  "wm": ("Frontal Lobe WM", "Parietal Lobe WM", "Occipital Lobe WM")},
    "quantified": ("NAA", "tCr", "Glu", "mI", "tCho", "Gln", "NAAG", "Asp", "GABA", "GSH", "Tau",
                   "Scyllo"),
}

#*****************#
#   hangel 2021   #
#*****************#
# Source: Hangel, G., Spurny-Dworak, B., Lazen, P., Cadrien, C., Sharma, S., Hingerl, L., Heckova,
#         E., Strasser, B., Motyka, S., Lipka, A., Gruber, S., Brandner, C., Lanzenberger, R.,
#         Rossler, K., Trattnig, S., Bogner, W. Inter-subject stability and regional
#         concentration estimates of 3D-FID-MRSI in the human brain at 7 T. NMR in Biomedicine 34
#         (2021) e4596, doi 10.1002/nbm.4596, Supporting Information Table S3: 23 volunteers,
#         internal water reference.
# Units:  (mean, sd) in mM.
# Notes:  the rows the package's regions take; the table's hemispheres, sub-lobar areas and mixed
#         rows are not typed. The paper estimates tCho, tCr, Glu, mIns and NAA within its quality
#         thresholds, the others it lists are not quantified. The cingulate rows stand for both
#         cingulate regions; the reference is the whole cortex and the whole white matter.
HANGEL_2021 = {
    "values": {
        "Cortical GM (bilateral)": {
            "tCho": (1.69, 0.71), "tCr": (7.07, 2.85), "GABA": (1.92, 1.4), "Glu": (8.23, 3.7),
            "Gln": (2.33, 1.6), "Gly": (0.84, 0.59), "GSH": (0.79, 0.66), "mIns": (5.26, 2.12),
            "NAA": (9.33, 4.05), "NAAG": (1.84, 1.15), "Ser": (1.91, 1.33), "Tau": (2.36, 1.35)
        },
        "Subcortical WM (bilateral)": {
            "tCho": (1.92, 0.73), "tCr": (7.24, 2.62), "GABA": (1.75, 1.23), "Glu": (7.67, 3.29),
            "Gln": (2.05, 1.53), "Gly": (0.9, 0.59), "GSH": (0.91, 0.65), "mIns": (5.24, 1.85),
            "NAA": (9.73, 3.61), "NAAG": (1.94, 1.1), "Ser": (1.89, 1.21), "Tau": (2.1, 1.27)
        },
        "Frontal cortex GM": {
            "tCho": (1.74, 0.79), "tCr": (6.98, 3.01), "GABA": (1.84, 1.39), "Glu": (8.29, 3.98),
            "Gln": (2.29, 1.65), "Gly": (0.78, 0.64), "GSH": (0.87, 0.76), "mIns": (5.22, 2.29),
            "NAA": (9.44, 3.98), "NAAG": (1.85, 1.19), "Ser": (1.96, 1.58), "Tau": (2.39, 1.5)
        },
        "Frontal subcortex WM": {
            "tCho": (1.99, 0.79), "tCr": (7.39, 2.69), "GABA": (1.71, 1.13), "Glu": (7.99, 3.41),
            "Gln": (2.09, 1.57), "Gly": (0.87, 0.61), "GSH": (0.95, 0.73), "mIns": (5.2, 1.92),
            "NAA": (9.91, 3.81), "NAAG": (1.93, 1.11), "Ser": (1.93, 1.32), "Tau": (2.17, 1.34)
        },
        "Parietal cortex GM": {
            "tCho": (1.64, 0.5), "tCr": (7.57, 2.23), "GABA": (2.06, 1.22), "Glu": (8.55, 2.81),
            "Gln": (2.24, 1.33), "Gly": (0.92, 0.44), "GSH": (0.69, 0.46), "mIns": (5.61, 1.63),
            "NAA": (10.09, 3.02), "NAAG": (2.02, 0.95), "Ser": (1.78, 0.94), "Tau": (2.41, 1)
        },
        "Parietal subcortex WM": {
            "tCho": (1.84, 0.51), "tCr": (7.51, 2), "GABA": (1.85, 1.14), "Glu": (7.64, 2.61),
            "Gln": (1.9, 1.25), "Gly": (0.97, 0.45), "GSH": (0.8, 0.48), "mIns": (5.54, 1.38),
            "NAA": (10.21, 2.66), "NAAG": (2.15, 0.91), "Ser": (1.68, 0.85), "Tau": (2.14, 0.95)
        },
        "Occipital cortex GM": {
            "tCho": (1.46, 0.65), "tCr": (6.55, 2.92), "GABA": (2.07, 1.73), "Glu": (7.26, 3.72),
            "Gln": (2.3, 1.73), "Gly": (0.84, 0.59), "GSH": (0.82, 0.74), "mIns": (4.57, 2),
            "NAA": (8.58, 4.31), "NAAG": (1.77, 1.33), "Ser": (1.99, 1.29), "Tau": (2.11, 1.28)
        },
        "Occipital subcortex WM": {
            "tCho": (1.61, 0.68), "tCr": (6.61, 2.66), "GABA": (1.96, 1.53), "Glu": (6.68, 3.35),
            "Gln": (2.11, 1.62), "Gly": (0.9, 0.61), "GSH": (0.91, 0.65), "mIns": (4.59, 1.79),
            "NAA": (8.83, 3.91), "NAAG": (1.79, 1.25), "Ser": (2, 1.22), "Tau": (1.87, 1.15)
        },
        "Temporal subcortex WM": {
            "tCho": (1.95, 0.89), "tCr": (6.89, 3.17), "GABA": (1.55, 1.19), "Glu": (7.3, 3.68),
            "Gln": (2.32, 1.64), "Gly": (0.83, 0.65), "GSH": (0.94, 0.69), "mIns": (4.83, 2.15),
            "NAA": (8.56, 4.06), "NAAG": (1.66, 1.14), "Ser": (2, 1.34), "Tau": (1.97, 1.48)
        },
        "Cingulate cortex GM": {
            "tCho": (2.22, 0.74), "tCr": (8.45, 2.62), "GABA": (2.36, 1.28), "Glu": (10.14, 3.57),
            "Gln": (2.47, 1.52), "Gly": (1, 0.63), "GSH": (1.05, 0.72), "mIns": (6.6, 2.05),
            "NAA": (10.86, 3.41), "NAAG": (2.06, 1.04), "Ser": (2.4, 1.43), "Tau": (2.81, 1.47)
        },
        "Cingulate subcortex WM": {
            "tCho": (2.32, 0.81), "tCr": (7.55, 2.75), "GABA": (1.93, 1.25), "Glu": (8.06, 3.7),
            "Gln": (1.83, 1.52), "Gly": (1.06, 0.7), "GSH": (1.22, 0.73), "mIns": (5.99, 2.01),
            "NAA": (10.55, 3.57), "NAAG": (2.06, 1.08), "Ser": (2.26, 1.38), "Tau": (2.09, 1.47)
        },
        "Thalamus": {
            "tCho": (2.4, 0.83), "tCr": (8.81, 3.07), "GABA": (2.47, 1.48), "Glu": (9.18, 4.15),
            "Gln": (2.52, 2.13), "Gly": (1.16, 0.95), "GSH": (1.37, 0.82), "mIns": (6.24, 2.21),
            "NAA": (10.6, 4), "NAAG": (2.58, 1.55), "Ser": (2.69, 1.58), "Tau": (2.41, 2.08)
        },
        "Putamen": {
            "tCho": (2.27, 0.86), "tCr": (9.36, 3.32), "GABA": (1.83, 1.26), "Glu": (9.2, 3.59),
            "Gln": (3.13, 1.7), "Gly": (0.87, 0.84), "GSH": (1.33, 0.83), "mIns": (5.09, 2.04),
            "NAA": (9.58, 3.77), "NAAG": (1.83, 1.13), "Ser": (2.8, 1.58), "Tau": (2.26, 1.78)
        },
        "Pallidum": {
            "tCho": (2.11, 0.94), "tCr": (8.88, 3.75), "GABA": (2.41, 1.71), "Glu": (8.18, 4.3),
            "Gln": (2.89, 2.09), "Gly": (0.91, 0.94), "GSH": (1.63, 0.92), "mIns": (4.31, 2.03),
            "NAA": (8.43, 4.06), "NAAG": (1.95, 1.43), "Ser": (2.85, 1.84), "Tau": (2.01, 2.02)
        },
        "Hippocampus": {
            "tCho": (2.18, 1.01), "tCr": (7.37, 3.35), "GABA": (1.81, 1.55), "Glu": (6.83, 3.67),
            "Gln": (2.35, 1.72), "Gly": (1.07, 0.86), "GSH": (1.5, 0.97), "mIns": (5.78, 2.74),
            "NAA": (7.12, 3.73), "NAAG": (1.8, 1.42), "Ser": (2.06, 1.56), "Tau": (2.26, 1.82)
        },
    },
    "stands_for": {
        "Frontal cortex GM": (("frontal",), "gm"), "Frontal subcortex WM": (("frontal",), "wm"),
        "Parietal cortex GM": (("parietal",), "gm"),
        "Parietal subcortex WM": (("parietal",), "wm"),
        "Occipital cortex GM": (("occipital",), "gm"),
        "Occipital subcortex WM": (("occipital",), "wm"),
        "Temporal subcortex WM": (("temporal",), "wm"),
        "Cingulate cortex GM": (("anterior_cingulate", "posterior_cingulate"), "gm"),
        "Cingulate subcortex WM": (("anterior_cingulate", "posterior_cingulate"), "wm"),
        "Thalamus": (("thalamus",), None), "Putamen": (("putamen",), None),
        "Pallidum": (("pallidum",), None), "Hippocampus": (("hippocampus",), None),
    },
    "reference": {"gm": ("Cortical GM (bilateral)",), "wm": ("Subcortical WM (bilateral)",)},
    "quantified": ("tCho", "tCr", "Glu", "mIns", "NAA"),
}

#******************#
#   pouwels 1998   #
#******************#
# Source: Pouwels, P. J. W., Frahm, J. Regional metabolite concentrations in human brain as
#         determined by quantitative localized proton MRS. Magnetic Resonance in Medicine 39
#         (1998) 53-60, doi 10.1002/mrm.1910390110, Tables 1 to 3: single voxels at 2.0 T in 34
#         volunteers, white matter (Table 1), grey matter (Table 2), cerebellum and thalamus
#         (Table 3), each row under its table's heading.
# Units:  (mean, sd) in mM (mmol per litre of voxel), from calibrated model spectra.
# Notes:  Cr is the total creatine and Cho the choline-containing compounds, tNAA the NAA with
#         the NAAG; the tables' creatine line widths are not typed. The cerebellar hemisphere's
#         voxel (mainly white matter, the paper's) stands for the cerebellum, the central one for
#         the vermis. The reference is the mean of the frontal, parietal and occipital rows in
#         each tissue.
POUWELS_1998 = {
    "values": {
        "White matter, Frontal": {
            "Cr": (5.7, 0.5), "tNAA": (9.6, 1.1), "NAA": (8.1, 0.9), "NAAG": (1.5, 0.9),
            "Cho": (1.78, 0.41), "Ins": (3.8, 0.9), "Glu": (7.0, 2.6), "Gln": (1.8, 1.6)
        },
        "White matter, Parietal": {
            "Cr": (5.7, 0.6), "tNAA": (10.6, 0.8), "NAA": (8.0, 1.0), "NAAG": (2.7, 1.2),
            "Cho": (1.68, 0.27), "Ins": (3.1, 0.6), "Glu": (6.7, 1.8), "Gln": (1.5, 1.3)
        },
        "White matter, Occipital": {
            "Cr": (5.5, 0.8), "tNAA": (10.4, 0.9), "NAA": (7.8, 0.9), "NAAG": (2.6, 1.0),
            "Cho": (1.64, 0.21), "Ins": (4.1, 0.8), "Glu": (6.0, 1.2), "Gln": (2.2, 1.6)
        },
        "Gray matter, Frontal": {
            "Cr": (6.4, 0.7), "tNAA": (8.4, 1.0), "NAA": (7.7, 1.0), "NAAG": (0.7, 0.3),
            "Cho": (1.38, 0.17), "Ins": (4.3, 0.9), "Glu": (8.5, 1.0), "Gln": (4.4, 1.4)
        },
        "Gray matter, Parietal": {
            "Cr": (6.5, 0.6), "tNAA": (8.7, 0.8), "NAA": (8.2, 0.8), "NAAG": (0.5, 0.4),
            "Cho": (1.10, 0.14), "Ins": (4.3, 0.7), "Glu": (8.2, 1.1), "Gln": (3.8, 1.4)
        },
        "Gray matter, Occipital": {
            "Cr": (6.9, 0.7), "tNAA": (10.6, 0.8), "NAA": (9.2, 0.9), "NAAG": (1.4, 0.8),
            "Cho": (0.88, 0.10), "Ins": (4.1, 0.6), "Glu": (8.6, 1.1), "Gln": (3.9, 1.1)
        },
        "Gray matter, Insular": {
            "Cr": (7.0, 0.6), "tNAA": (9.6, 0.6), "NAA": (8.7, 0.8), "NAAG": (0.8, 0.5),
            "Cho": (1.30, 0.19), "Ins": (4.7, 0.6), "Glu": (8.8, 0.8), "Gln": (4.9, 1.6)
        },
        "Cerebellum, Hemisphere": {
            "Cr": (8.7, 1.3), "tNAA": (11.0, 0.8), "NAA": (7.4, 1.0), "NAAG": (3.6, 0.9),
            "Cho": (2.21, 0.29), "Ins": (5.8, 1.1), "Glu": (7.7, 2.5), "Gln": (4.3, 2.5)
        },
        "Cerebellum, Vermis": {
            "Cr": (9.0, 0.9), "tNAA": (8.7, 0.9), "NAA": (7.4, 0.8), "NAAG": (1.3, 0.8),
            "Cho": (2.19, 0.30), "Ins": (5.6, 1.1), "Glu": (7.5, 1.4), "Gln": (5.4, 2.2)
        },
        "Thalamus": {
            "Cr": (6.8, 0.7), "tNAA": (10.5, 0.7), "NAA": (7.3, 1.3), "NAAG": (3.2, 1.1),
            "Cho": (1.81, 0.21), "Ins": (3.6, 0.9), "Glu": (6.9, 2.2), "Gln": (2.9, 1.8)
        },
    },
    "stands_for": {
        "White matter, Frontal": (("frontal",), "wm"),
        "White matter, Parietal": (("parietal",), "wm"),
        "White matter, Occipital": (("occipital",), "wm"),
        "Gray matter, Frontal": (("frontal",), "gm"),
        "Gray matter, Parietal": (("parietal",), "gm"),
        "Gray matter, Occipital": (("occipital",), "gm"),
        "Gray matter, Insular": (("insula",), "gm"),
        "Cerebellum, Hemisphere": (("cerebellum",), None),
        "Cerebellum, Vermis": (("vermis",), None),
        "Thalamus": (("thalamus",), None),
    },
    "reference": {"gm": ("Gray matter, Frontal", "Gray matter, Parietal", "Gray matter, Occipital"),
                  "wm": ("White matter, Frontal", "White matter, Parietal",
                         "White matter, Occipital")},
    "quantified": ("Cr", "NAA", "NAAG", "Cho", "Ins", "Glu", "Gln"),
}

#*************#
#   sources   #
#*************#
# newest first: where several give a region, the newest holds it
STUDIES = (WRIGHT_2024, HANGEL_2021, POUWELS_1998)

# the papers' metabolite names as the package's: a total lends its pattern to its constituents
NAMES = {"NAA": ("NAA",), "NAAG": ("NAAG",), "tCr": ("Cr", "PCr"), "tCho": ("GPC", "PCho"),
         "Glu": ("Glu",), "Gln": ("Gln",), "mI": ("Ins",), "mIns": ("Ins",), "GABA": ("GABA",),
         "GSH": ("GSH",), "Asp": ("Asp",), "Tau": ("Tau",), "Scyllo": ("sIns",), "Gly": ("Gly",),
         "Ser": ("Ser",), "Cr": ("Cr", "PCr"), "Cho": ("GPC", "PCho"), "Ins": ("Ins",)}
