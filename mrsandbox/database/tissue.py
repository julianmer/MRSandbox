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

#***********************#
#   brain temperature   #
#***********************#
# Source: Rzechorzek, N. M., et al. A daily temperature rhythm in the human brain predicts survival
#         after brain injury. Brain 145 (2022) 2031-2048, doi:10.1093/brain/awab466: MRS
#         thermometry of 40 healthy adults (20 to 40 years), morning, afternoon and late evening,
#         T = 100 x (NAA minus water shift in ppm + 2.665) + 37.
# Units:  degrees Celsius.
# Notes:  the mean (38.5) is in their calibration; this package's water at 37 C sits 2.642 ppm
#         above NAA, which their equation reads as 38.7 C, so a draw keeps the package's anchor
#         and takes their spread between people alone (VirtualScan temperature_spread).
RZECHORZEK_2022 = {"healthy_mean_c": 38.5, "healthy_sd_c": 0.4, "healthy_range_c": (36.1, 40.9),
                   "luteal_offset_c": 0.36}

#***************************#
#   metabolite diffusion    #
#***************************#
# Source: Deelchand, D. K., Auerbach, E. J., Marjanska, M. Apparent diffusion coefficients of
#         the five major metabolites measured in the human brain in vivo at 3T. Magn Reson Med
#         79 (2018) 2896-2901, Table 1: trace/3 ADC with STEAM (TE/TM 21.2/105 ms), the
#         cross-terms removed, five subjects, a prefrontal voxel of 64 % grey matter, 23 %
#         white matter, 13 % CSF.
# Units:  um^2/ms (water 0.70 in the same table).
# Notes:  PRESS (TE 54.2 ms) gave 0.16, 0.14, 0.14, 0.15, 0.16 in the same order, not
#         significantly different. "members" are this database's metabolites behind each total.
DEELCHAND_2018 = {
    "adc_um2_ms": {"tNAA": 0.13, "tCr": 0.13, "tCho": 0.12, "Glu": 0.15, "mIns": 0.11},
    "sd_um2_ms": {"tNAA": 0.01, "tCr": 0.01, "tCho": 0.01, "Glu": 0.03, "mIns": 0.02},
    "members": {"tNAA": ("NAA", "NAAG"), "tCr": ("Cr", "PCr"), "tCho": ("GPC", "PCho", "Cho"),
                "Glu": ("Glu",), "mIns": ("Ins",)},
}
