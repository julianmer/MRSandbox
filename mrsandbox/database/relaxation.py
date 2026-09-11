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

#*************************#
#   ethofer at 1.5 tesla  #
#*************************#
# Source: Ethofer et al. 2003 (the ETHOFER card above), Table 2, the 1.5 T rows.
# Units:  T1 in seconds with standard deviations, N = 8; 1.5 T.
# Notes:  progressive saturation PRESS, the same six regions and the same subjects as the
#         3 T rows; the paper's pure-tissue water extrapolation at 1.5 T feeds WATER_FIELDS.
ETHOFER_1P5 = {
    "error": "sd",
    "b0_t": 1.5,
    "t1_s": {
        "occipital_gm": {"NA": (1.27, 0.05, 8), "Cho": (1.15, 0.12, 8), "tCr": (1.24, 0.06, 8), "Ins": (1.00, 0.15, 8)},
        "occipital_wm": {"NA": (1.36, 0.06, 8), "Cho": (1.03, 0.15, 8), "tCr": (1.24, 0.10, 8), "Ins": (0.97, 0.04, 8)},
        "motor_cortex": {"NA": (1.17, 0.07, 8), "Cho": (1.08, 0.16, 8), "tCr": (1.16, 0.10, 8), "Ins": (1.04, 0.14, 8)},
        "frontoparietal_wm": {"NA": (1.19, 0.09, 8), "Cho": (1.04, 0.12, 8), "tCr": (1.15, 0.08, 8), "Ins": (0.93, 0.07, 8)},
        "thalamus": {"NA": (1.41, 0.09, 8), "Cho": (1.10, 0.17, 8), "tCr": (1.27, 0.11, 8), "Ins": (1.04, 0.21, 8)},
        "cerebellum": {"NA": (1.42, 0.15, 8), "Cho": (1.22, 0.16, 8), "tCr": (1.33, 0.18, 8), "Ins": (1.25, 0.18, 8)},
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
        "cerebral_wm": {"Cho": (1.102, 0.137, 15), "tCr": (1.402, 0.157, 15), "NAA": (1.304, 0.145, 15)},
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

#*************************#
#   water at other fields #
#*************************#
# Source: per entry, beside each value.
# Data:   water relaxation per tissue where a field differs from the 3 T table above;
#         a missing entry means the 3 T value is reused (stated: no 1.5 T water T2
#         source is typed yet). CSF stays the field-independent Rooney value.
# Units:  seconds; t1_s entries as (mean, sd) where the source prints a spread.
WATER_FIELDS = {
    1.5: {
        "t1_s": {
            "gm": (1.04, 0.02),  # Ethofer et al. 2003 (the ETHOFER card): pure-tissue
            "wm": (0.68, 0.02),  # extrapolation of the 1.5 T regional values, N = 8
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
