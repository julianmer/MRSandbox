####################################################################################################
#                                         macromolecules.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Macromolecule background parameterizations, one dict per source, stored exactly as      #
#          published with the citation beside each dict.                                           #
#                                                                                                  #
####################################################################################################

#*************#
#   cudalbu   #
#*************#
# Source: Cudalbu, C., Behar, K. L., Bhattacharyya, P. K., Bogner, W., et al., Mlynarik, V.
#         Contribution of macromolecules to brain 1H MR spectra: experts' consensus
#         recommendations. NMR in Biomedicine 34 (2021) e4393. Table 1.
# Units:  recommended nomenclature; position in ppm (ranges as (low, high) tuples);
#         tentative assignment; confirmed = established in vivo vs still to be confirmed.
# Notes:  positions vary by 0.01 to 0.04 ppm between papers (the table's own footnote).
CUDALBU = {
    "M0.94": {"ppm": 0.94, "assignment": "leucine, isoleucine, valine", "confirmed": True},
    "M1.22": {"ppm": 1.22, "assignment": "threonine", "confirmed": True},
    "M1.43": {"ppm": 1.43, "assignment": "alanine", "confirmed": True},
    "M1.70": {"ppm": 1.70, "assignment": "lysine, arginine, leucine", "confirmed": True},
    "M1.81": {"ppm": 1.81, "assignment": None, "confirmed": False},
    "M1.9": {"ppm": 1.9, "assignment": None, "confirmed": False},
    "M2.05": {"ppm": 2.05, "assignment": "glutamate, glutamine", "confirmed": True},
    "M2.07": {"ppm": 2.07, "assignment": None, "confirmed": False},
    "M2.17": {"ppm": 2.17, "assignment": None, "confirmed": False},
    "M2.27": {"ppm": 2.27, "assignment": "glutamate, glutamine", "confirmed": True},
    "M2.37": {"ppm": 2.37, "assignment": None, "confirmed": False},
    "M2.50": {"ppm": 2.47, "assignment": None, "confirmed": False},
    "M2.54": {"ppm": 2.54, "assignment": "beta-methylene protons of aspartyl groups", "confirmed": True},
    "M2.74": {"ppm": 2.68, "assignment": "beta-methylene protons of aspartyl groups", "confirmed": True},
    "M2.97": {"ppm": 2.97, "assignment": None, "confirmed": False},
    "M3.00": {"ppm": 3.00, "assignment": "lysine, cystine", "confirmed": True},
    "M3.21": {"ppm": 3.21, "assignment": "valine H-beta, alpha-CH protons; phosphatidylcholine methyl may contribute", "confirmed": True},
    "M3.43-3.95": {"ppm": (3.43, 3.95), "assignment": None, "confirmed": False},
    "M3.71": {"ppm": 3.71, "assignment": "alpha-CH protons of protein amino acids", "confirmed": True},
    "M3.79": {"ppm": 3.77, "assignment": "alpha-CH protons of protein amino acids", "confirmed": True},
    "M3.87": {"ppm": 3.87, "assignment": "alpha-CH protons of protein amino acids", "confirmed": True},
    "M3.97": {"ppm": 3.97, "assignment": "alpha-CH protons of protein amino acids", "confirmed": True},
    "M4.05-4.43": {"ppm": (4.05, 4.42), "assignment": "threonine H-beta, H-gamma", "confirmed": True},
    "M4.20": {"ppm": 4.20, "assignment": "alpha-CH protons of protein amino acids", "confirmed": True},
}

#************#
#   osprey   #
#************#
# Source: Oeltzschner, G., Zoellner, H. J., Hui, S. C. N., Mikkelsen, M., Saleh, M. G.,
#         Tapper, S., Edden, R. A. E. Osprey: open-source processing, reconstruction and
#         estimation of magnetic resonance spectroscopy data. Journal of Neuroscience
#         Methods 343 (2020) 108827.
# Data:   fit_makeBasis.m, github.com/schorschinho/Osprey, retrieved 2026-09-05; the code
#         states its values follow the LCModel and TARQUIN algorithms.
# Units:  (label, position ppm, FWHM ppm, amplitude in one-proton areas relative to the
#         creatine CH3 singlet at 3.027 ppm).
# Notes:  short echo time parameterization; MM20 and Lip20 are composites of several
#         Gaussians; the in vivo level is a sampled parameter, never part of this data.
OSPREY = {
    "MM09": (("MM09", 0.91, 0.14, 3.0),),
    "MM12": (("MM12", 1.21, 0.15, 2.0),),
    "MM14": (("MM14", 1.43, 0.17, 2.0),),
    "MM17": (("MM17", 1.67, 0.15, 2.0),),
    "MM20": (("MM20a", 2.08, 0.15, 1.33), ("MM20b", 2.25, 0.20, 0.33),
             ("MM20c", 1.95, 0.15, 0.33), ("MM20d", 3.00, 0.20, 0.40)),
    "Lip09": (("Lip09", 0.89, 0.14, 3.0),),
    "Lip13": (("Lip13a", 1.28, 0.15, 2.0), ("Lip13b", 1.28, 0.89, 2.0)),
    "Lip20": (("Lip20a", 2.04, 0.15, 1.33), ("Lip20b", 2.25, 0.15, 0.67),
              ("Lip20c", 2.80, 0.20, 0.87)),
}

#**********************#
#   mrsi challenge     #
#**********************#
# Source: LaMaster, J. T., Merkofer, J. P., van de Sande, D. M. J., Soher, B. J.,
#         Strasser, B., Ma, C. The 2024 MRSI Data Processing and Quantification
#         Challenge Synthetic Dataset. arXiv:2609.02637. Table 3, after Povazan et al.
#         2018 (7 T double inversion recovery MRSI, six volunteers).
# Units:  per component: position in ppm and the amplitude ratios r to the total creatine
#         concentration for gray and white matter; the model is b = kappa x r x c_tCr
#         with the global scaling kappa = 0.4.
# Notes:  gm and wm ratios of the 1.21 to 1.67 components were constrained equal because
#         the measured contrast was not significant.
MRSI_CHALLENGE = {
    "kappa": 0.4,
    "components": {
        "M0.91": {"ppm": 0.91, "gm": 0.14049, "wm": 0.14225},
        "M1.21": {"ppm": 1.21, "gm": 0.24152, "wm": 0.24152},
        "M1.43": {"ppm": 1.43, "gm": 0.27404, "wm": 0.27404},
        "M1.67": {"ppm": 1.67, "gm": 0.24470, "wm": 0.24470},
        "M2.04": {"ppm": 2.04, "gm": 0.099662, "wm": 0.042158},
        "M2.26": {"ppm": 2.26, "gm": 0.045433, "wm": 0.020042},
        "M2.99": {"ppm": 2.99, "gm": 0.20234, "wm": 0.22049},
        "M3.21": {"ppm": 3.21, "gm": 0.35101, "wm": 0.39498},
        "M3.77": {"ppm": 3.77, "gm": 0.33077, "wm": 0.33268},
    },
}

#**************************#
#   consensus couplings    #
#**************************#
# Source: Cudalbu, C., et al. (the CUDALBU table above), Table 1: cross-peaks in 2D
#         spectra and the J column (coupling constants from 2D J-resolved spectra).
# Units:  (partner_a_ppm, partner_b_ppm, j_hz) for the amino-acid chains behind the main
#         MM peaks; J values vary by about 0.1 Hz between the paired rows, the mean is
#         stored.
# Notes:  the 1.70 to 3.00 coupling is the one the consensus flags as important for GABA
#         detection (the co-edited MM background).
CONSENSUS_COUPLINGS = (
    (0.94, 2.05, 7.7),  # leucine, isoleucine, valine: CH3 to CH
    (1.22, 4.24, 6.6),  # threonine: CH3 to H-beta
    (1.43, 4.32, 7.7),  # alanine: CH3 to CH
    (1.70, 3.00, 7.7),  # lysine, arginine: the GABA-relevant chain
)

#***************************#
#   consensus t1 (app s1)   #
#***************************#
# Source: Cudalbu, C., et al. (the CUDALBU table above), Appendix S1 (supporting
#         information of nbm.4393), Table S2: T1 relaxation times of MM at different
#         magnetic fields in humans and rats.
# Data:   per MM peak and study column: (value_ms, error_ms), error None where the table
#         prints none (the error type is not stated in the source); study metadata typed
#         from the table headers, echo and repetition times as printed.
# Units:  milliseconds; b0_t in tesla.
# Notes:  entries the source prints in brackets reached the FitAid fit limit and are
#         listed in fit_limit; entries printed with ~ are listed in approximate. The
#         source marks Murali-Manohar 2019 as updated values first published in this
#         table. Peaks without any reported value are omitted.
CONSENSUS_T1 = {
    "studies": {
        "lopez_kolkovsky_2016": dict(b0_t=17.2, species="rat", sequence="IR-LASER",
                                     te_ms="16.5", tr_ms="5000", roi="cerebral cortex"),
        "cudalbu_2009_14t1": dict(b0_t=14.1, species="rat", sequence="IR-SPECIAL",
                                  te_ms="2.8", tr_ms="4000",
                                  roi="frontal cortex, corpus callosum, striatum"),
        "cudalbu_2009_9t4": dict(b0_t=9.4, species="rat", sequence="IR-SPECIAL",
                                 te_ms="2.8", tr_ms="4000",
                                 roi="frontal cortex, corpus callosum, striatum"),
        "de_graaf_2006_11t7": dict(b0_t=11.7, species="rat", sequence="in house method",
                                   te_ms=None, tr_ms=None, roi="cerebral cortex"),
        "de_graaf_2006_9t4": dict(b0_t=9.4, species="rat", sequence="in house method",
                                  te_ms=None, tr_ms=None, roi="cerebral cortex"),
        "de_graaf_2006_4t": dict(b0_t=4.0, species="rat", sequence="in house method",
                                 te_ms=None, tr_ms=None, roi="cerebral cortex"),
        "pfeuffer_1999": dict(b0_t=9.4, species="rat", sequence="IR-STEAM",
                              te_ms="1", tr_ms="4000", roi="posterior to bregma"),
        "murali_manohar_2019_gm": dict(b0_t=9.4, species="human",
                                       sequence="DIR-MC-semiLASER", te_ms="24",
                                       tr_ms="8000", roi="occipital lobe, GM rich"),
        "murali_manohar_2019_wm": dict(b0_t=9.4, species="human",
                                       sequence="DIR-MC-semiLASER", te_ms="24",
                                       tr_ms="8000", roi="occipital lobe, WM rich"),
        "xin_2013": dict(b0_t=7.0, species="human", sequence="IR-SPECIAL", te_ms="12",
                         tr_ms="7500", roi="occipital lobe (GM rich)"),
        "hoefemann_2020": dict(b0_t=3.0, species="human", sequence="2DJ-IRMC-PRESS",
                               te_ms="20-95", tr_ms="3000-4000",
                               roi="occipital lobe (GM rich)"),
        "behar_1994": dict(b0_t=2.1, species="human", sequence="IR-3D ISIS", te_ms="17",
                           tr_ms="4450", roi="occipital lobe (mixed)"),
        "kreis_2005": dict(b0_t=1.5, species="human", sequence="IR-PRESS", te_ms="20",
                           tr_ms="400-9000", roi="mixed voxel"),
    },
    "t1_ms": {
        "M0.94": {"lopez_kolkovsky_2016": (748, 14), "cudalbu_2009_14t1": (660, 70),
                  "cudalbu_2009_9t4": (510, 70), "de_graaf_2006_11t7": (762, 30),
                  "de_graaf_2006_9t4": (633, 30), "de_graaf_2006_4t": (354, 39),
                  "pfeuffer_1999": (300, None), "murali_manohar_2019_gm": (280, 10),
                  "murali_manohar_2019_wm": (284, 10), "xin_2013": (420, 20),
                  "hoefemann_2020": (290, None), "behar_1994": (250, 36),
                  "kreis_2005": (220, 60)},
        "M1.22": {"de_graaf_2006_11t7": (634, 25), "de_graaf_2006_9t4": (581, 22),
                  "de_graaf_2006_4t": (361, 38), "murali_manohar_2019_gm": (273, 27),
                  "murali_manohar_2019_wm": (280, 28), "hoefemann_2020": (309, None),
                  "kreis_2005": (200, 50)},
        "M1.43": {"murali_manohar_2019_gm": (326, 21), "murali_manohar_2019_wm": (270, 24)},
        "M1.70": {"de_graaf_2006_11t7": (627, 43), "de_graaf_2006_9t4": (531, 42),
                  "murali_manohar_2019_gm": (231, 13), "murali_manohar_2019_wm": (253, 17),
                  "hoefemann_2020": (225, None)},
        "M1.81": {"hoefemann_2020": (247, None)},
        "M1.9": {"lopez_kolkovsky_2016": (652, 41)},
        "M2.05": {"murali_manohar_2019_gm": (279, 8), "murali_manohar_2019_wm": (319, 9)},
        "M2.27": {"murali_manohar_2019_gm": (281, 7), "murali_manohar_2019_wm": (313, 9),
                  "hoefemann_2020": (263, None)},
        "M2.57": {"murali_manohar_2019_gm": (309, 20), "murali_manohar_2019_wm": (327, 18)},
        "M2.74": {"murali_manohar_2019_gm": (306, 19), "murali_manohar_2019_wm": (383, 22),
                  "hoefemann_2020": (400, None)},
        "M2.97": {"lopez_kolkovsky_2016": (574, 22), "murali_manohar_2019_gm": (463, 15),
                  "murali_manohar_2019_wm": (518, 15), "hoefemann_2020": (400, None)},
        "M3.09": {"hoefemann_2020": (400, None)},
        "M3.21": {"murali_manohar_2019_gm": (204, 20), "murali_manohar_2019_wm": (379, 29)},
        "M3.26-3.28": {"hoefemann_2020": (400, None)},
        "M3.4-3.6": {"murali_manohar_2019_gm": (510, 30), "murali_manohar_2019_wm": (564, 35),
                     "hoefemann_2020": (400, None)},
        "M3.71": {"lopez_kolkovsky_2016": (792, 39), "murali_manohar_2019_gm": (280, 24),
                  "murali_manohar_2019_wm": (434, 25)},
        "M3.87": {"murali_manohar_2019_gm": (280, 27), "murali_manohar_2019_wm": (307, 22),
                  "hoefemann_2020": (347, None)},
    },
    "approximate": {("M0.94", "pfeuffer_1999")},
    "fit_limit": {("M2.74", "hoefemann_2020"), ("M2.97", "hoefemann_2020"),
                  ("M3.09", "hoefemann_2020"), ("M3.26-3.28", "hoefemann_2020"),
                  ("M3.4-3.6", "hoefemann_2020")},
}

#***************************#
#   consensus t2 (app s1)   #
#***************************#
# Source: Cudalbu, C., et al. (the CUDALBU table above), Appendix S1 (supporting
#         information of nbm.4393), Table S3: T2 relaxation times of MM at different
#         magnetic fields in humans and rats.
# Data:   per MM peak and study column: (value_ms, error_ms), error None where the table
#         prints none (the error type is not stated in the source); study metadata typed
#         from the table headers, echo time series and repetition times as printed.
# Units:  milliseconds; b0_t in tesla.
# Notes:  entries the source prints in brackets reached the FitAid fit limit and are
#         listed in fit_limit; entries printed with ~ are listed in approximate. Peaks
#         without any reported value are omitted. The 3 T human columns are Landheer
#         2020 (with errors) and Hoefemann 2020.
CONSENSUS_T2 = {
    "studies": {
        "lopez_kolkovsky_2016": dict(b0_t=17.2, species="rat", sequence="IR-LASER",
                                     te_ms="16.5-200", tr_ms="5000", roi="cerebral cortex"),
        "de_graaf_2006_11t7": dict(b0_t=11.7, species="rat", sequence="in house method",
                                   te_ms="10-450", tr_ms="3000", roi="cerebral cortex"),
        "de_graaf_2006_9t4": dict(b0_t=9.4, species="rat", sequence="in house method",
                                  te_ms="10-450", tr_ms="3000", roi="cerebral cortex"),
        "de_graaf_2006_4t": dict(b0_t=4.0, species="rat", sequence="in house method",
                                 te_ms="10-450", tr_ms="3000", roi="cerebral cortex"),
        "pfeuffer_1999": dict(b0_t=9.4, species="rat", sequence="STEAM", te_ms="100-200",
                              tr_ms="6000", roi="posterior to bregma"),
        "xin_2008": dict(b0_t=9.4, species="rat", sequence="SPECIAL", te_ms="2.8-300",
                         tr_ms="4000", roi="centered on the hippocampus"),
        "murali_manohar_2020_gm": dict(b0_t=9.4, species="human", sequence="DIR-semiLASER",
                                       te_ms="24-60", tr_ms="10000",
                                       roi="occipital lobe, GM rich"),
        "murali_manohar_2020_wm": dict(b0_t=9.4, species="human", sequence="DIR-semiLASER",
                                       te_ms="24-60", tr_ms="10000",
                                       roi="occipital lobe, WM rich"),
        "landheer_2020": dict(b0_t=3.0, species="human", sequence="DIR-semiLASER",
                              te_ms="20.1-62.1", tr_ms="2000",
                              roi="occipital and frontal lobe (GM rich)"),
        "hoefemann_2020": dict(b0_t=3.0, species="human", sequence="2DJ-IRMC-PRESS",
                               te_ms="20-95", tr_ms="3000-4000",
                               roi="occipital lobe (GM rich)"),
        "behar_1994": dict(b0_t=2.1, species="human", sequence="ISIS", te_ms="19.4-145.2",
                           tr_ms="3300", roi="occipital lobe (mixed)"),
    },
    "t2_ms": {
        "M0.94": {"lopez_kolkovsky_2016": (20, 2), "de_graaf_2006_11t7": (25, 2),
                  "de_graaf_2006_9t4": (27, 2), "de_graaf_2006_4t": (23, 1),
                  "pfeuffer_1999": (26, None), "xin_2008": (19, 0.4),
                  "murali_manohar_2020_gm": (36, 7), "murali_manohar_2020_wm": (35, 10),
                  "landheer_2020": (27, 2), "hoefemann_2020": (21, None),
                  "behar_1994": (44, 4)},
        "M1.22": {"de_graaf_2006_11t7": (31, 2), "de_graaf_2006_9t4": (34, 2),
                  "de_graaf_2006_4t": (32, 2), "murali_manohar_2020_gm": (34, 13),
                  "murali_manohar_2020_wm": (25, 0), "landheer_2020": (39, 13),
                  "hoefemann_2020": (19, None)},
        "M1.43": {"murali_manohar_2020_gm": (32, 7), "murali_manohar_2020_wm": (40, 24),
                  "landheer_2020": (18, 4)},
        "M1.70": {"de_graaf_2006_11t7": (20, 2), "de_graaf_2006_9t4": (20, 2),
                  "murali_manohar_2020_gm": (18, 4), "murali_manohar_2020_wm": (18, 8),
                  "landheer_2020": (17, 3), "hoefemann_2020": (27, None)},
        "M1.81": {"hoefemann_2020": (19, None)},
        "M1.9": {"lopez_kolkovsky_2016": (28, 3), "murali_manohar_2020_gm": (16, 2),
                 "murali_manohar_2020_wm": (19, 2), "landheer_2020": (14, 1)},
        "M2.17": {"murali_manohar_2020_gm": (18, 2), "murali_manohar_2020_wm": (18, 1),
                  "landheer_2020": (20, 2)},
        "M2.27": {"hoefemann_2020": (24, None)},
        "M2.57": {"murali_manohar_2020_gm": (20, 3), "murali_manohar_2020_wm": (24, 3),
                  "landheer_2020": (26, 4)},
        "M2.74": {"murali_manohar_2020_gm": (14, 3), "murali_manohar_2020_wm": (13, 2),
                  "hoefemann_2020": (16, None)},
        "M2.97": {"lopez_kolkovsky_2016": (55, 5), "murali_manohar_2020_gm": (19, 2),
                  "murali_manohar_2020_wm": (19, 1), "landheer_2020": (21, 4),
                  "hoefemann_2020": (35, None)},
        "M3.09": {"hoefemann_2020": (35, None)},
        "M3.21": {"murali_manohar_2020_gm": (18, 3), "murali_manohar_2020_wm": (16, 5),
                  "landheer_2020": (18, 5)},
        "M3.26-3.28": {"hoefemann_2020": (35, None)},
        "M3.4-3.6": {"murali_manohar_2020_gm": (14, 6), "murali_manohar_2020_wm": (16, 5),
                     "landheer_2020": (21, 2), "hoefemann_2020": (24, None)},
        "M3.71": {"lopez_kolkovsky_2016": (43, 6), "murali_manohar_2020_gm": (25, 7),
                  "murali_manohar_2020_wm": (25, 7)},
        "M3.87": {"murali_manohar_2020_gm": (22, 5), "murali_manohar_2020_wm": (24, 7),
                  "hoefemann_2020": (34, None)},
        "M3.97": {"murali_manohar_2020_gm": (21, 8)},
    },
    "approximate": {("M0.94", "pfeuffer_1999")},
    "fit_limit": {("M2.97", "hoefemann_2020"), ("M3.09", "hoefemann_2020"),
                  ("M3.26-3.28", "hoefemann_2020")},
}
