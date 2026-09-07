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
