####################################################################################################
#                                            tissue.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Tissue water properties, one dict per source, stored exactly as published with the      #
#          citation beside each dict.                                                              #
#                                                                                                  #
####################################################################################################

#**********************#
#   mrsi challenge     #
#**********************#
# Source: LaMaster, J. T., Merkofer, J. P., van de Sande, D. M. J., Soher, B. J.,
#         Strasser, B., Ma, C. The 2024 MRSI Data Processing and Quantification
#         Challenge Synthetic Dataset. arXiv:2609.02637.
# Units:  water content as fraction of the water molarity per tissue; molarity in mM.
# Notes:  the 4.65 ppm water position is the usual convention, not from the descriptor.
MRSI_CHALLENGE = {
    "water_molarity_mm": 55200.0,
    "water_content": {"gm": 0.80, "wm": 0.70, "csf": 1.00},
    "water_shift_ppm": 4.65,
}

#*************************#
#   water thermometry     #
#*************************#
# Source: Cady, E. B., D'Souza, P. C., Penrice, J., Lorek, A. The estimation of local brain
#         temperature by in vivo 1H magnetic resonance spectroscopy. Magn Reson Med 33 (1995)
#         862-867: T = 286.9 - 94.0 x (water minus NAA methyl shift in ppm), calibrated in
#         piglet brain over 30 to 40 C.
# Units:  degrees Celsius and degrees Celsius per ppm.
# Notes:  the water resonance moves by -1 / 94.0 ppm per degree against the metabolites, whose
#         own shifts move at most 7.6e-4 ppm/K against each other (Wermter et al. 2017, MAGMA
#         30:579, Table 1), fourteen times less.
CADY_1995 = {"intercept_c": 286.9, "c_per_ppm": 94.0}
