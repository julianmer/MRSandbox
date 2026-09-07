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
