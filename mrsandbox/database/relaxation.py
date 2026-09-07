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
        "occipital_gm": {"NA": (1.47, 0.08, 8), "Cho": (1.25, 0.22, 8), "tCr": (1.33, 0.13, 8), "Ins": (1.12, 0.25, 8)},
        "occipital_wm": {"NA": (1.40, 0.15, 8), "Cho": (1.17, 0.15, 8), "tCr": (1.31, 0.13, 8), "Ins": (0.91, 0.13, 8)},
        "motor_cortex": {"NA": (1.46, 0.22, 8), "Cho": (1.47, 0.20, 8), "tCr": (1.38, 0.16, 8), "Ins": (0.92, 0.13, 8)},
        "frontoparietal_wm": {"NA": (1.56, 0.06, 8), "Cho": (1.21, 0.13, 8), "tCr": (1.40, 0.06, 8), "Ins": (0.96, 0.10, 8)},
        "thalamus": {"NA": (1.57, 0.08, 8), "Cho": (1.38, 0.22, 8), "tCr": (1.45, 0.16, 8), "Ins": (1.11, 0.32, 8)},
        "cerebellum": {"NA": (1.43, 0.19, 8), "Cho": (1.26, 0.11, 8), "tCr": (1.38, 0.09, 8), "Ins": (0.96, 0.23, 8)},
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
        "gm": 0.110,  # Wansapura et al., JMRI 9 (1999) 531 (pair printed unlabeled;
        "wm": 0.080,  # assigned GM-high, consistent with the same abstract's T1 order)
        "csf": 0.5,   # stated assumption; literature spans roughly 0.3 to 2 seconds
    },
}
