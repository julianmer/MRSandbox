####################################################################################################
#                                           relaxation.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Relaxation times in seconds, one dict per source, stored exactly as published with the  #
#          citation beside each dict. Pooling across sources is code, never data.                  #
#                                                                                                  #
####################################################################################################

#************#
#   simple   #
#************#
# Source: stated simple assumptions, not literature values.
# Units:  one global T2 and one global T1 in seconds, ballpark for 3 T.
# Notes:  the fallback fidelity level; per-metabolite sources replace these wherever
#         pooled literature rows exist.
SIMPLE = {
    "t2_s": 0.160,
    "t1_s": 1.300,
}

# The three 3 T metabolite T1 sources below share one shape: rows are (mean_s, error, n)
# per resonance per region, and the "error" field says whether the printed error is a
# standard error ("se") or a standard deviation ("sd"). Composite in vivo resonances keep
# their composite names (tCr, Cho as choline compounds, NA as NAA plus NAAG, Glx);
# assigning them to constituent molecules is a pooling approximation, never data.

#**************#
#   mlynarik   #
#**************#
# Source: Mlynarik, V., Gruber, S., Moser, E. Proton T1 and T2 relaxation times of human
#         brain metabolites at 3 Tesla. NMR in Biomedicine 14 (2001) 325-331. Table 1.
# Units:  T1 in seconds with standard errors, n = 5 unless noted.
# Notes:  inversion recovery STEAM, occipital lobe.
MLYNARIK = {
    "error": "se",
    "t1_s": {
        "occipital_gm": {
            "NAA": (1.47, 0.06, 5), "Glu": (1.27, 0.10, 4), "tCr_CH3": (1.46, 0.07, 5),
            "Cho": (1.30, 0.06, 5), "Ins_3.57": (1.23, 0.09, 5), "Ins_3.66": (1.19, 0.07, 2),
            "Glx": (1.20, 0.10, 5), "tCr_CH2": (0.97, 0.07, 5),
        },
        "occipital_wm": {
            "NAA": (1.35, 0.12, 5), "Glu": (1.17, 0.08, 4), "tCr_CH3": (1.24, 0.10, 5),
            "Cho": (1.08, 0.06, 5), "Ins_3.57": (1.01, 0.09, 5), "Ins_3.66": (0.93, 0.11, 5),
            "Glx": (0.96, 0.20, 5), "tCr_CH2": (0.87, 0.11, 5),
        },
    },
}

#*************#
#   ethofer   #
#*************#
# Source: Ethofer, T., Mader, I., Seeger, U., Helms, G., Erb, M., Grodd, W., Ludolph, A.,
#         Klose, U. Comparison of longitudinal metabolite relaxation times in different
#         regions of the human brain at 1.5 and 3 Tesla. Magnetic Resonance in Medicine
#         50 (2003) 1296-1301. Table 2, the 3 T rows.
# Units:  T1 in seconds with standard deviations, N = 8.
# Notes:  progressive saturation PRESS, six regions.
ETHOFER = {
    "error": "sd",
    "t1_s": {
        "occipital_gm": {"NA": (1.47, 0.08, 8), "Cho": (1.25, 0.22, 8),
                         "tCr": (1.33, 0.13, 8), "Ins": (1.12, 0.25, 8)},
        "occipital_wm": {"NA": (1.40, 0.15, 8), "Cho": (1.17, 0.15, 8),
                         "tCr": (1.31, 0.13, 8), "Ins": (0.91, 0.13, 8)},
        "motor_cortex": {"NA": (1.46, 0.22, 8), "Cho": (1.47, 0.20, 8),
                         "tCr": (1.38, 0.16, 8), "Ins": (0.92, 0.13, 8)},
        "frontoparietal_wm": {"NA": (1.56, 0.06, 8), "Cho": (1.21, 0.13, 8),
                              "tCr": (1.40, 0.06, 8), "Ins": (0.96, 0.10, 8)},
        "thalamus": {"NA": (1.57, 0.08, 8), "Cho": (1.38, 0.22, 8),
                     "tCr": (1.45, 0.16, 8), "Ins": (1.11, 0.32, 8)},
        "cerebellum": {"NA": (1.43, 0.19, 8), "Cho": (1.26, 0.11, 8),
                       "tCr": (1.38, 0.09, 8), "Ins": (0.96, 0.23, 8)},
    },
}

#************#
#   traber   #
#************#
# Source: Traber, F., Block, W., Lamerichs, R., Gieseke, J., Schild, H. H. 1H metabolite
#         relaxation times at 3.0 tesla: measurements of T1 and T2 values in normal brain
#         and determination of regional differences in transverse relaxation. Journal of
#         Magnetic Resonance Imaging 19 (2004) 537-545. Table 2, the 3 T rows.
# Units:  T1 in seconds with standard deviations, N = 8; one row per technique.
# Notes:  each row pools occipital WM, motor cortex, and fronto-lateral VOIs (the paper
#         found no regional T1 differences).
TRABER = {
    "error": "sd",
    "t1_s": {
        "inversion_recovery": {
            "NAA": (1.34, 0.08, 8), "tCr_CH3": (1.11, 0.11, 8), "Cho": (1.14, 0.07, 8),
            "Ins": (0.98, 0.16, 8), "tCr_CH2": (0.76, 0.13, 8),
        },
        "progressive_saturation": {
            "NAA": (1.39, 0.03, 8), "tCr_CH3": (1.47, 0.07, 8), "Cho": (1.15, 0.04, 8),
            "Ins": (1.36, 0.25, 8), "tCr_CH2": (1.02, 0.13, 8),
        },
    },
}

#*********#
#   xin   #
#*********#
# Source: Xin, L., Schaller, B., Mlynarik, V., Lu, H., Gruetter, R. Proton T1 relaxation
#         times of metabolites in human occipital white and gray matter at 7 T. Magnetic
#         Resonance in Medicine 69 (2013) 931-936. Table 2.
# Units:  T1 in seconds with standard deviations, n = 6 per tissue; 7 T.
# Notes:  inversion recovery semi-adiabatic SPECIAL (TE 12 ms), occipital lobe. tCr_CH3
#         is the 3.03 ppm methyl, tCr_CH2 the 3.92 ppm methylene, NAA the 2.01 ppm methyl
#         and NAA_CH2 its aspartate methylene; tCho and Glu are entire-molecule means.
#         The same table's water rows feed WATER_FIELDS, its macromolecule row the
#         consensus MM table (xin_2013 in database.macromolecules).
XIN = {
    "error": "sd",
    "b0_t": 7.0,
    "t1_s": {
        "occipital_wm": {
            "tCr_CH3": (1.78, 0.04, 6), "tCr_CH2": (1.10, 0.06, 6), "NAA": (1.90, 0.06, 6),
            "Cho": (1.32, 0.06, 6), "Ins": (1.19, 0.07, 6), "Glu": (1.75, 0.04, 6),
            "NAA_CH2": (1.19, 0.05, 6), "GSH": (1.06, 0.06, 6), "Gln": (1.74, 0.23, 6),
            "sIns": (1.23, 0.07, 6), "Tau": (2.09, 0.04, 6), "PE": (1.32, 0.30, 6),
            "NAAG": (0.94, 0.08, 6),
        },
        "occipital_gm": {
            "tCr_CH3": (1.74, 0.06, 6), "tCr_CH2": (1.13, 0.04, 6), "NAA": (1.83, 0.05, 6),
            "Cho": (1.51, 0.11, 6), "Ins": (1.28, 0.08, 6), "Glu": (1.61, 0.09, 6),
            "NAA_CH2": (1.24, 0.05, 6), "GSH": (1.14, 0.18, 6), "Gln": (1.64, 0.07, 6),
            "sIns": (1.31, 0.20, 6), "Tau": (2.15, 0.11, 6), "PE": (1.31, 0.18, 6),
            "NAAG": (1.21, 0.22, 6),
        },
    },
}

#**************************#
#   ethofer at 1.5 tesla   #
#**************************#
# Source: Ethofer et al. 2003 (the ETHOFER card above), Table 2, the 1.5 T rows.
# Units:  T1 in seconds with standard deviations, N = 8; 1.5 T.
# Notes:  progressive saturation PRESS, the same six regions and the same subjects as the
#         3 T rows; the paper's pure-tissue water extrapolation at 1.5 T (gm 1.04 +- 0.02,
#         wm 0.68 +- 0.02 s) stands behind Rooney's in vivo rows in WATER_FIELDS.
ETHOFER_1P5 = {
    "error": "sd",
    "b0_t": 1.5,
    "t1_s": {
        "occipital_gm": {"NA": (1.27, 0.05, 8), "Cho": (1.15, 0.12, 8),
                         "tCr": (1.24, 0.06, 8), "Ins": (1.00, 0.15, 8)},
        "occipital_wm": {"NA": (1.36, 0.06, 8), "Cho": (1.03, 0.15, 8),
                         "tCr": (1.24, 0.10, 8), "Ins": (0.97, 0.04, 8)},
        "motor_cortex": {"NA": (1.17, 0.07, 8), "Cho": (1.08, 0.16, 8),
                         "tCr": (1.16, 0.10, 8), "Ins": (1.04, 0.14, 8)},
        "frontoparietal_wm": {"NA": (1.19, 0.09, 8), "Cho": (1.04, 0.12, 8),
                              "tCr": (1.15, 0.08, 8), "Ins": (0.93, 0.07, 8)},
        "thalamus": {"NA": (1.41, 0.09, 8), "Cho": (1.10, 0.17, 8),
                     "tCr": (1.27, 0.11, 8), "Ins": (1.04, 0.21, 8)},
        "cerebellum": {"NA": (1.42, 0.15, 8), "Cho": (1.22, 0.16, 8),
                       "tCr": (1.33, 0.18, 8), "Ins": (1.25, 0.18, 8)},
    },
}

#*************#
#   rutgers   #
#*************#
# Source: Rutgers, D. R., Kingsley, P. B., van der Grond, J. Saturation-corrected T1 and
#         T2 relaxation times of choline, creatine and N-acetyl aspartate in human
#         cerebral white matter at 1.5 T. NMR in Biomedicine 16 (2003) 286-288 (a Letter
#         to the Editor). Table 1, the revised two-parameter fit.
# Units:  T1 in seconds with standard deviations, 15 volunteers; 1.5 T.
# Notes:  PRESS, cerebral white matter; the revision corrects the original values for
#         partial saturation (1 to 3 percent longer). Table 2 gives the matching
#         saturation-corrected T2 (Cho 375, Cr 239, NAA 436 ms over 60 to 864 ms TE);
#         T2 is pooled from the Gudmundson database instead, so it is not typed here.
RUTGERS = {
    "error": "sd",
    "b0_t": 1.5,
    "t1_s": {
        "cerebral_wm": {"Cho": (1.102, 0.137, 15), "tCr": (1.402, 0.157, 15),
                        "NAA": (1.304, 0.145, 15)},
    },
}

#***********#
#   water   #
#***********#
# Source: per entry, beside each value.
# Units:  water relaxation per tissue at 3 T, in seconds.
WATER = {
    "t1_s": {
        "gm": 1.23,   # Ethofer et al., MRM 50 (2003) 1296: pure-tissue extrapolation
        "wm": 0.95,   # Ethofer et al., same extrapolation
        "csf": 4.3,   # Rooney et al., MRM 57 (2007) 308: ventricular, field independent
    },
    "t2_s": {
        "csf": 0.5,   # stated assumption; literature spans roughly 0.3 to 2 seconds
    },
}

#*********************#
#   water t2 at 3 t   #
#*********************#
# Source: per entry, beside each row.
# Data:   in vivo water T2 of healthy adults at 3 T measured by multi-echo spin echo fitted with
#         the stimulated echoes modelled (extended phase graph or Bloch dictionaries, or a
#         corrected fast spin echo), the methods that read the tissue's own T2 (uncorrected
#         mono-exponential fits read it long); each study checked at its source. Rows are
#         (T2, sd between the study's subjects or None where the paper does not say what its
#         error is, n); grey is cortical, white cerebral or frontal.
# Units:  seconds.
WATER_T2_3T = {
    "gm": (
        (0.0664, 0.0017, 28),  # Radunsky et al. 2024, PLoS One 19:e0297244, Table 3, cortex
        (0.0733, 0.0026, 3),   # Snyder et al. 2022, MRM 87:2145, Table 2, the three subjects' mean
        (0.0841, 0.0018, 7),   # Corbin et al. 2025 (the CORBIN card), the seven volunteers' median
        (0.0805, None, 27),    # Hamid et al. 2023, Front Neurol 14:1175971, mid-cortex
        (0.0870, 0.0042, 40),  # Seiler et al. 2020, Aging 12:16195, Table 1, mid-cortex
    ),
    "wm": (
        (0.0588, 0.0014, 28),  # Radunsky et al. 2024, Table 3, cerebral white matter
        (0.0610, None, 38),    # Shepherd et al. 2017, NeuroImage Clin 14:363, Table 2, centrum
                               # semiovale
        (0.0581, 0.0016, 3),   # Snyder et al. 2022, Table 2, frontal white matter
        (0.0761, 0.0030, 7),   # Corbin et al. 2025 (the CORBIN card), the seven volunteers' median
        (0.0627, None, 27),    # Hamid et al. 2023, Table 2, cerebral white matter
    ),
}

#***************************#
#   water at other fields   #
#***************************#
# Source: per entry, beside each value.
# Data:   water relaxation per tissue where a field differs from the 3 T table above;
#         a missing entry means the 3 T value is reused (stated: no human in vivo water T2
#         off 3 T is typed yet). CSF stays the field-independent Rooney value.
# Units:  seconds; entries as (mean, sd) where the source prints a spread.
WATER_FIELDS = {
    1.5: {
        "t1_s": {
            "gm": (1.188, 0.069),  # Rooney et al. 2007 (the ROONEY card): cortical grey matter
            "wm": (0.656, 0.016),  # and frontal white matter in vivo, three subjects, sd
        },
        "t2_s": {
            "gm": (0.095, 0.008),  # Stanisz et al. 2005 (the STANISZ card): bovine grey and
            "wm": (0.072, 0.004),  # white matter in vitro at 37 C, three samples (stated: the
        },                         # only typed 1.5 T water T2; no human in vivo source yet)
    },
    4.0: {
        "t1_s": {
            "gm": (1.723, 0.093),  # Rooney et al. 2007 (the ROONEY card): cortical grey matter
            "wm": (1.010, 0.019),  # and frontal white matter in vivo, three subjects, sd
        },
    },
    7.0: {
        "t1_s": {
            "gm": (2.00, 0.12),  # Xin et al. 2013 (the XIN card), Table 2, occipital GM
            "wm": (1.55, 0.07),  # and WM, n = 6, sd
        },
        "t2_s": {
            "gm": 0.047,  # Marjanska et al., NMR Biomed 25 (2012) 332, Table 1: 47 +- 1 ms
            "wm": 0.047,  # in the occipital (GM rich) and motor (WM rich) voxels alike
        },
    },
}

#**************#
#   boudreau   #
#**************#
# Source: Boudreau, M., Karakuzu, A., Cohen-Adad, J., et al. Repeat it without me:
#         crowdsourcing the T1 mapping common ground via the ISMRM reproducibility
#         challenge. Magnetic Resonance in Medicine 92 (2024) 1115-1127.
# Data:   human brain water T1 at 3 T across all challenge submissions (inversion
#         recovery per Barral 2010, 27 sites, three vendors, 56 datasets), per region
#         of interest as (mean_s, sd_s); the sd is the intersubmission spread, a
#         realistic site-to-site variation.
# Units:  seconds.
BOUDREAU = {
    "error": "sd",
    "t1_s": {
        "wm_genu": (0.828, 0.038),
        "wm_splenium": (0.852, 0.049),
        "gm_cortex": (1.548, 0.156),
        "gm_deep": (1.188, 0.133),
    },
}

#************#
#   corbin   #
#************#
# Source: Corbin, N., Trotier, A. J., Anandra, S., et al. Whole-brain T2 mapping with radial
#         sampling and retrospective motion correction at 3T. Magnetic Resonance in Medicine 93
#         (2025) 1026-1042, Table 1.
# Data:   human brain water T2 at 3 T, seven healthy adults (5 F, 2 M, aged 28 to 50), Siemens
#         Prisma, a 3D radial multi-echo spin echo of 50 echoes (7 to 350 ms), T2 by an extended
#         phase graph fit with B1 (the stimulated echoes modelled); per region the median over
#         each volunteer's voxels (grey and white matter by SPM12 on their own MPRAGE,
#         probability over 0.9; deep nuclei by volBrain), one entry per volunteer, V1 to V7.
# Units:  seconds.
CORBIN = {
    "t2_s": {
        "gm": (0.0870, 0.0848, 0.0838, 0.0875, 0.0828, 0.0841, 0.0832),
        "wm": (0.0786, 0.0761, 0.0751, 0.0805, 0.0795, 0.0757, 0.0717),
        "caudate": (0.0747, 0.0751, 0.0752, 0.0698, 0.0729, 0.0715, 0.0735),
        "putamen": (0.0693, 0.0715, 0.0701, 0.0680, 0.0730, 0.0669, 0.0651),
        "thalamus": (0.0740, 0.0748, 0.0746, 0.0773, 0.0747, 0.0731, 0.0722),
        "pallidum": (0.0622, 0.0636, 0.0596, 0.0632, 0.0676, 0.0599, 0.0586),
    },
}

#*************#
#   cudalbu   #
#*************#
# Source: Cudalbu, C., Mlynarik, V., Xin, L., Gruetter, R. Comparison of T1 relaxation times of
#         the neurochemical profile in rat brain at 9.4 tesla and 14.1 tesla. Magnetic Resonance
#         in Medicine 62 (2009) 862-867. Figures 2 and 4; the paper prints no table.
# Data:   rat brain in vivo, six animals per field, one voxel. T1 by inversion recovery (IR,
#         AMARES on ten resonances, Figure 2) and by progressive saturation (PS, LCModel on 16
#         components, Figure 4). Every mean and error bar (sd over the animals) is read off the
#         bar chart against its 0.2 s grid, to about 0.02 s; the macromolecules' 0.66 +- 0.07 s
#         at 14.1 T and 0.51 +- 0.07 s at 9.4 T are the text's own numbers. The text: the IR T1
#         lie in 1.2 to 1.5 s, the PS T1 in 1.4 to 1.9 s apart from taurine; NAA alone rises
#         with the field (8 percent by IR, 11 percent by PS); the methods agree within 15
#         percent, PS reading a little higher. A star in the figure marks the methylene
#         (tCr_CH2) and the aspartyl (NAA_CH2) resonance; IR rows name the line's ppm.
# Units:  seconds, (mean, sd, n); the field in tesla. Rat, not human: kept apart from
#         Database.T1_SOURCES until a species switch exists.
CUDALBU = {
    "error": "sd",
    "species": "rat",
    "t1_s": {
        9.4: {
            "ir": {
                "tCr_CH3": (1.27, 0.03, 6), "tCr_CH2": (1.05, 0.08, 6), "NAA": (1.37, 0.03, 6),
                "Tau": (2.17, 0.15, 6), "Cho": (1.41, 0.07, 6), "Glu_2.35": (1.17, 0.05, 6),
                "Glx_3.75": (1.24, 0.04, 6), "Ins_3.57": (1.40, 0.18, 6),
                "Ins_3.65": (1.27, 0.22, 6), "Ins_4.05": (1.29, 0.12, 6),
            },
            "ps": {
                "tCr_CH3": (1.71, 0.06, 6), "tCr_CH2": (1.12, 0.08, 6), "NAA": (1.71, 0.03, 6),
                "NAA_CH2": (1.23, 0.10, 6), "Tau": (2.56, 0.20, 6), "Cho": (1.49, 0.17, 6),
                "Glu": (1.52, 0.05, 6), "Gln": (1.44, 0.04, 6), "Ins": (1.46, 0.08, 6),
                "MM": (0.51, 0.07, 6), "Ala": (1.74, 0.23, 6), "Asp": (1.57, 0.17, 6),
                "GABA": (1.86, 0.08, 6), "GSH": (1.14, 0.17, 6), "NAAG": (1.56, 0.32, 6),
                "PE": (1.22, 0.12, 6),
            },
        },
        14.1: {
            "ir": {
                "tCr_CH3": (1.36, 0.15, 6), "tCr_CH2": (0.94, 0.08, 6), "NAA": (1.48, 0.03, 6),
                "Tau": (2.17, 0.09, 6), "Cho": (1.34, 0.16, 6), "Glu_2.35": (1.43, 0.14, 6),
                "Glx_3.75": (1.33, 0.07, 6), "Ins_3.57": (1.43, 0.08, 6),
                "Ins_3.65": (1.40, 0.10, 6), "Ins_4.05": (1.40, 0.22, 6),
            },
            "ps": {
                "tCr_CH3": (1.78, 0.12, 6), "tCr_CH2": (1.17, 0.08, 6), "NAA": (1.91, 0.10, 6),
                "NAA_CH2": (1.28, 0.10, 6), "Tau": (2.57, 0.14, 6), "Cho": (1.25, 0.14, 6),
                "Glu": (1.65, 0.06, 6), "Gln": (1.70, 0.11, 6), "Ins": (1.42, 0.25, 6),
                "MM": (0.66, 0.07, 6), "Ala": (1.76, 0.24, 6), "Asp": (1.78, 0.15, 6),
                "GABA": (1.55, 0.27, 6), "GSH": (1.66, 0.43, 6), "NAAG": (1.38, 0.27, 6),
                "PE": (1.60, 0.14, 6),
            },
        },
    },
}

#**********#
#   wang   #
#**********#
# Source: Wang, Y., van Gelderen, P., de Zwart, J. A., Duyn, J. H. B0-field dependence of MRI T1
#         relaxation in human brain. NeuroImage 213 (2020) 116700. Table 2 and Section 3.
# Data:   eight healthy adults (21 to 32 years, three women) on one Siemens site, the same
#         inversion-recovery and saturation-transfer sequence at nominal 0.55, 1.5, 3 and 7 T
#         (2.89 and 6.98 T as measured), fitted jointly to a two-pool (Bloch-McConnell) model of
#         magnetization transfer. The water pool's slow longitudinal rate lambda_s, the inverse
#         of the observed T1, in the splenium of the corpus callosum where the two pools resolve
#         (3 and 7 T); the macromolecular pool's rate R_m at every field (at 0.55 and 1.5 T with
#         the pool fraction and exchange rate fixed at the 7 T means), and its power law in the
#         field, R_m = a B0^b: 12.2 B0^-1.00 in the splenium (R^2 0.997), 13.2 B0^-1.03 over the
#         averaged white matter.
# Units:  rates in 1/s as (mean, sd over the subjects, n); the field in tesla as measured.
WANG = {
    "error": "sd",
    "r1_water_per_s": {
        "wm_splenium": {2.89: (1.112, 0.025, 8), 6.98: (0.760, 0.008, 8)},
    },
    "r1_macromolecular_per_s": {
        "wm_splenium": {0.55: (22.8, 2.6, 8), 1.50: (8.2, 0.5, 8), 2.89: (3.89, 0.07, 8),
                        6.98: (1.85, 0.09, 8)},
        "wm": {0.55: (25.0, 4.0, 8), 1.50: (9.0, 0.5, 8), 2.89: (3.99, 0.05, 8),
               6.98: (1.90, 0.05, 8)},
    },
    "macromolecular_law": {"wm_splenium": (12.2, -1.00), "wm": (13.2, -1.03)},
}

#************#
#   rooney   #
#************#
# Source: Rooney, W. D., Johnson, G., Li, X., Cohen, E. R., Kim, S.-G., Ugurbil, K., Springer,
#         C. S. Magnetic field and tissue dependencies of human brain longitudinal 1H2O
#         relaxation in vivo. Magnetic Resonance in Medicine 57 (2007) 308-318. Table 1, the
#         rows of this study; the fits in the Results and Discussion.
# Data:   three men of 32 to 59 years, the same axial slice on five instruments within six
#         months (0.2, 1.0 and 1.5 T at NYU, 4.0 T at Brookhaven and Minnesota, 7.0 T at
#         Minnesota), 32 inversion-recovery images fitted pixel by pixel, regions of interest
#         averaged over the subjects: frontal white matter, cortical grey matter, caudate,
#         thalamus, putamen, globus pallidus, sagittal sinus blood and ventricular CSF. The
#         ventricular CSF rate does not vary with the field, 0.231 +- 0.009 1/s. Bottomley's
#         function T1 = C (gamma B0)^beta fitted to the data (gamma the magnetogyric ratio, B0
#         in tesla, T1 in seconds): white matter C 0.00071, beta 0.382; putamen C 0.00116, beta
#         0.376; sagittal sinus blood C 0.00335, beta 0.340. The excess rate over CSF of every
#         tissue follows B0^-0.39 (nonlinear regression over all tissues).
# Units:  seconds as (mean, sd over the three subjects); the field in tesla.
ROONEY = {
    "error": "sd",
    "n": 3,
    "t1_s": {
        0.2: {"wm": (0.361, 0.017), "gm": (0.635, 0.054), "caudate": (0.555, 0.019),
              "thalamus": (0.522, 0.044), "putamen": (0.524, 0.019), "pallidum": (0.411, 0.020),
              "blood": (0.776, 0.022), "csf": (4.408, 0.187)},
        1.0: {"wm": (0.555, 0.020), "gm": (1.036, 0.019), "caudate": (0.898, 0.045),
              "thalamus": (0.807, 0.047), "putamen": (0.815, 0.016), "pallidum": (0.625, 0.014),
              "blood": (1.351, 0.024), "csf": (4.276, 0.109)},
        1.5: {"wm": (0.656, 0.016), "gm": (1.188, 0.069), "caudate": (1.083, 0.052),
              "thalamus": (0.972, 0.032), "putamen": (0.981, 0.013), "pallidum": (0.746, 0.020),
              "blood": (1.540, 0.023), "csf": (4.070, 0.065)},
        4.0: {"wm": (1.010, 0.019), "gm": (1.723, 0.093), "caudate": (1.509, 0.053),
              "thalamus": (1.452, 0.087), "putamen": (1.446, 0.032), "pallidum": (1.143, 0.027),
              "blood": (1.914, 0.114), "csf": (4.472, 0.085)},
        7.0: {"wm": (1.220, 0.036), "gm": (2.132, 0.103), "caudate": (1.745, 0.064),
              "thalamus": (1.656, 0.084), "putamen": (1.700, 0.066), "pallidum": (1.347, 0.052),
              "blood": (2.587, 0.283), "csf": (4.425, 0.137)},
    },
    "field_law": {"wm": (0.00071, 0.382), "putamen": (0.00116, 0.376), "blood": (0.00335, 0.340)},
    "excess_rate_exponent": -0.39,
    "csf_r1_per_s": (0.231, 0.009),
}

#*************#
#   stanisz   #
#*************#
# Source: Stanisz, G. J., Odrobina, E. E., Pun, J., Escaravage, M., Graham, S. J., Bronskill,
#         M. J., Henkelman, R. M. T1, T2 relaxation and magnetization transfer in tissue at 3T.
#         Magnetic Resonance in Medicine 54 (2005) 507-512. Table 1, the rows of this study.
# Data:   in vitro at 37 C in a test tube, 3 T and 1.5 T on the same console, three samples
#         per tissue: bovine white and grey matter and optic nerve, rat spinal cord, human blood
#         at 95 percent oxygenation (the liver, muscle, heart, kidney and cartilage rows are
#         left out here). T1 by inversion recovery, T2 by CPMG at a 1 ms echo spacing. The T2
#         do not differ between the fields; the T1 rise by 22 percent in white matter and 62
#         percent in grey matter from 1.5 to 3 T. The paper's Table 2 holds the quantitative
#         magnetization transfer parameters at 3 T (not typed here).
# Units:  seconds as (mean, sd over the three samples); the field in tesla.
STANISZ = {
    "error": "sd",
    "n": 3,
    "t1_s": {
        3.0: {"wm": (1.084, 0.045), "gm": (1.820, 0.114), "optic_nerve": (1.083, 0.039),
              "spinal_cord": (0.993, 0.047), "blood": (1.932, 0.085)},
        1.5: {"wm": (0.884, 0.050), "gm": (1.124, 0.050), "optic_nerve": (0.815, 0.030),
              "spinal_cord": (0.745, 0.037), "blood": (1.441, 0.120)},
    },
    "t2_s": {
        3.0: {"wm": (0.069, 0.003), "gm": (0.099, 0.007), "optic_nerve": (0.078, 0.005),
              "spinal_cord": (0.078, 0.002), "blood": (0.275, 0.050)},
        1.5: {"wm": (0.072, 0.004), "gm": (0.095, 0.008), "optic_nerve": (0.077, 0.009),
              "spinal_cord": (0.074, 0.006), "blood": (0.290, 0.030)},
    },
}

#**********************#
#   de graaf, rat      #
#**********************#
# Source: de Graaf, R. A., Brown, P. B., McIntyre, S., Nixon, T. W., Behar, K. L., Rothman,
#         D. L. High magnetic field water and metabolite proton T1 and T2 relaxation in rat
#         brain in vivo. Magnetic Resonance in Medicine 56 (2006) 386-394. Tables 1 and 2.
# Data:   rat brain in vivo, five animals, at 4.0, 9.4 and 11.74 T under the same protocol.
#         Water: inversion-recovery and multi-echo EPI maps, nine structures (corpus callosum,
#         cerebral cortex, olfactory bulb, hippocampus, cerebellar white and grey matter,
#         striatum, thalamus, midbrain). Metabolites: localized spectra, the T1 of the resonances
#         named by their shift at the long TE, the macromolecules M1 (1.7 ppm), M2 (1.4), M3
#         (1.1) and M4 (0.9) at the short TE, M2 and M3 together; the T2 of the singlets alone
#         (the coupled spins' echo decay carries their J evolution). Myo-inositol and taurine
#         could not be read at 4.0 T; the blanks of the paper's table are left out.
# Units:  seconds as (mean, sd over the five animals); the field in tesla as the paper names it.
#         Rat, not human: kept apart from Database.T1_SOURCES until a species switch exists.
DEGRAAF_RAT = {
    "error": "sd",
    "n": 5,
    "water": {
        4.0: {"t1_s": {"corpus_callosum": (1.0968, 0.0493), "cortex": (1.2858, 0.0770),
                       "olfactory_bulb": (1.6406, 0.0208), "hippocampus": (1.3341, 0.0974),
                       "cerebellar_wm": (1.0469, 0.0533), "cerebellar_gm": (1.3526, 0.0826),
                       "striatum": (1.2882, 0.0873), "thalamus": (1.1694, 0.0695),
                       "midbrain": (1.0644, 0.0173)},
              "t2_s": {"corpus_callosum": (0.0579, 0.0016), "cortex": (0.0652, 0.0024),
                       "olfactory_bulb": (0.0802, 0.0020), "hippocampus": (0.0720, 0.0013),
                       "cerebellar_wm": (0.0588, 0.0015), "cerebellar_gm": (0.0653, 0.0020),
                       "striatum": (0.0697, 0.0020), "thalamus": (0.0617, 0.0021),
                       "midbrain": (0.0606, 0.0019)}},
        9.4: {"t1_s": {"corpus_callosum": (1.7521, 0.0521), "cortex": (1.9484, 0.0519),
                       "olfactory_bulb": (2.1291, 0.0637), "hippocampus": (2.0597, 0.0661),
                       "cerebellar_wm": (1.6603, 0.0793), "cerebellar_gm": (2.0972, 0.0682),
                       "striatum": (1.9270, 0.0547), "thalamus": (1.7931, 0.0643),
                       "midbrain": (1.7865, 0.0819)},
              "t2_s": {"corpus_callosum": (0.0358, 0.0012), "cortex": (0.0421, 0.0012),
                       "olfactory_bulb": (0.0481, 0.0019), "hippocampus": (0.0454, 0.0018),
                       "cerebellar_wm": (0.0372, 0.0020), "cerebellar_gm": (0.0417, 0.0016),
                       "striatum": (0.0435, 0.0024), "thalamus": (0.0406, 0.0012),
                       "midbrain": (0.0403, 0.0014)}},
        11.7: {"t1_s": {"corpus_callosum": (1.8613, 0.0735), "cortex": (2.0734, 0.1007),
                        "olfactory_bulb": (2.3043, 0.0634), "hippocampus": (2.2228, 0.0632),
                        "cerebellar_wm": (1.7451, 0.0360), "cerebellar_gm": (2.1094, 0.0950),
                        "striatum": (2.0465, 0.0553), "thalamus": (1.9032, 0.0612),
                        "midbrain": (1.8932, 0.0443)},
               "t2_s": {"corpus_callosum": (0.0307, 0.0010), "cortex": (0.0362, 0.0010),
                        "olfactory_bulb": (0.0389, 0.0011), "hippocampus": (0.0389, 0.0013),
                        "cerebellar_wm": (0.0271, 0.0013), "cerebellar_gm": (0.0373, 0.0023),
                        "striatum": (0.0364, 0.0022), "thalamus": (0.0338, 0.0016),
                        "midbrain": (0.0338, 0.0019)}},
    },
    "t1_s": {
        4.0: {"tCr_CH2": (1.0034, 0.0812), "tCho": (1.4514, 0.0691), "tCr_CH3": (1.6140, 0.0352),
              "Glx_H4": (1.2626, 0.0288), "NAA": (1.5206, 0.0513), "MM2_3": (0.3609, 0.0381),
              "MM1": (0.3535, 0.0389)},
        9.4: {"tCr_CH2": (1.0397, 0.0735), "Ins": (1.3701, 0.1204), "Tau": (2.3289, 0.3586),
              "tCho": (1.3483, 0.0206), "tCr_CH3": (1.6792, 0.0677), "Glx_H4": (1.4975, 0.1065),
              "NAA": (1.6740, 0.0311), "MM4": (0.5306, 0.0424), "MM2_3": (0.5809, 0.0221),
              "MM1": (0.6326, 0.0303)},
        11.7: {"tCr_CH2": (1.1564, 0.0845), "Ins": (1.6381, 0.1705), "Tau": (2.4475, 0.1678),
               "tCho": (1.6296, 0.1087), "tCr_CH3": (1.7673, 0.0802), "Glx_H4": (1.4505, 0.1468),
               "NAA": (1.7135, 0.0441), "MM4": (0.6271, 0.0429), "MM2_3": (0.6341, 0.0251),
               "MM1": (0.7622, 0.0304)},
    },
    "t2_s": {
        4.0: {"tCr_CH2": (0.1407, 0.0115), "tCho": (0.5698, 0.0288), "tCr_CH3": (0.2329, 0.0059),
              "NAA": (0.3919, 0.0151), "MM2_3": (0.0317, 0.0025), "MM1": (0.0234, 0.0011)},
        9.4: {"tCr_CH2": (0.1278, 0.0057), "tCho": (0.4413, 0.0362), "tCr_CH3": (0.1711, 0.0039),
              "NAA": (0.2943, 0.0051), "MM4": (0.0227, 0.0025), "MM2_3": (0.0335, 0.0019),
              "MM1": (0.0274, 0.0023)},
        11.7: {"tCr_CH2": (0.1248, 0.0073), "tCho": (0.3657, 0.0731), "tCr_CH3": (0.1591, 0.0066),
               "NAA": (0.2852, 0.0265), "MM4": (0.0199, 0.0023), "MM2_3": (0.0314, 0.0021),
               "MM1": (0.0253, 0.0019)},
    },
}
