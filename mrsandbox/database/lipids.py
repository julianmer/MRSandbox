####################################################################################################
#                                             lipids.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Lipid resonances, one dict per source: the Hamilton six-peak triglyceride spectrum      #
#          (the rendering default), the de Graaf Table 2.3 shift reference, and the two-peak       #
#          SIMPLE stated assumption kept as the minimal fallback.                                  #
#                                                                                                  #
####################################################################################################

#************#
#   simple   #
#************#
# Source: stated simple assumption, not a literature table.
# Units:  bulk methylene and terminal methyl at their standard ppm positions with an
#         assumed amplitude ratio.
# Notes:  to be replaced by real triglyceride spin systems (de Graaf Table 2.3 or the
#         SimnTG port).
SIMPLE = {
    "peaks": {"CH2": {"shift_ppm": 1.30, "relative_amplitude": 1.0},
              "CH3": {"shift_ppm": 0.90, "relative_amplitude": 0.3}},
}

#**************#
#   hamilton   #
#**************#
# Source: Hamilton, G., Yokoo, T., Bydder, M., Cruite, I., Schroeder, M. E., Sirlin,
#         C. B., Middleton, M. S. In vivo characterization of the liver fat 1H MR
#         spectrum. NMR in Biomedicine 24 (2011) 784-790.
# Units:  six triglyceride resonances: position in ppm and relative amplitude (fraction
#         of the total lipid signal), measured at 3 T in 121 subjects.
# Notes:  the 5.3 ppm olefinic peak hides under water in vivo; relative amplitudes were
#         found stable across subjects.
HAMILTON = {
    "peaks": {
        "olefinic": {"shift_ppm": 5.30, "relative_amplitude": 0.047},
        "glycerol": {"shift_ppm": 4.20, "relative_amplitude": 0.039},
        "diallylic": {"shift_ppm": 2.75, "relative_amplitude": 0.006},
        "alpha_allylic": {"shift_ppm": 2.10, "relative_amplitude": 0.120},
        "methylene": {"shift_ppm": 1.30, "relative_amplitude": 0.700},
        "methyl": {"shift_ppm": 0.90, "relative_amplitude": 0.088},
    },
}

#**************#
#   de graaf   #
#**************#
# Source: de Graaf, R. A. In Vivo NMR Spectroscopy: Principles and Techniques, 3rd
#         edition. Wiley, 2019. Table 2.3 (1H columns).
# Units:  proton chemical shifts in ppm of the triglyceride proton groups, with the
#         figure labels and assignments; shift errors typically below 0.01 ppm.
# Notes:  reference material for finer lipid modelling; the bulk methylene rows span
#         1.27 to 1.33 ppm and merge in vivo.
DE_GRAAF = {
    "A": {"shift_ppm": 0.89, "assignment": "CH3-CH2-"},
    "BLn": {"shift_ppm": 2.04, "assignment": "CH3-CH2-CH= (linolenic omega-3)"},
    "B": {"shift_ppm": 1.31, "assignment": "-(CH2)n- bulk methylenes (1.27 to 1.33)"},
    "BL": {"shift_ppm": 1.29, "assignment": "CH3-CH2-CH2-CH2- (linoleic omega-6)"},
    "C": {"shift_ppm": 2.03, "assignment": "=CH-CH2-CH2- allylic"},
    "E": {"shift_ppm": 2.76, "assignment": "=CH-CH2-CH= diallylic"},
    "F": {"shift_ppm": 1.58, "assignment": "-CH2-CH2-COOR"},
    "G": {"shift_ppm": 2.25, "assignment": "-CH2-CH2-COOR alpha carboxyl"},
    "H": {"shift_ppm": 5.22, "assignment": "HCOOR glycerol H2"},
    "I": {"shift_ppm": 4.30, "assignment": "H2COOR glycerol H1"},
    "J": {"shift_ppm": 4.07, "assignment": "H2COOR glycerol H1-prime"},
    "D": {"shift_ppm": 5.33, "assignment": "-CH2-CH=CH-CH2- olefinic (5.31 to 5.33)"},
}
