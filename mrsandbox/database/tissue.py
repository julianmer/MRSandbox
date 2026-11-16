####################################################################################################
#                                             tissue.py                                            #
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

#********************#
#   mrsi challenge   #
#********************#
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

#***********************#
#   water thermometry   #
#***********************#
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

#**************************#
#   metabolite diffusion   #
#**************************#
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

#*****************************#
#   the sticks' diffusivity   #
#*****************************#
# Source: Lundell, H., Ingo, C., Dyrby, T. B., Ronen, I. Cytosolic diffusivity and microscopic
#         anisotropy of N-acetyl aspartate in human white matter with diffusion-weighted MRS at
#         7 T. NMR in Biomedicine 34 (2021) e4304, doi 10.1002/nbm.4304, Table 1: ten healthy
#         participants, powder-averaged DW-sLASER in the corpus callosum (CC) and the parietal
#         white matter (PWM), b up to 9.4 and 14.5 ms/um^2, gradients 59 and 65 ms apart.
# Units:  (mean, sd) in um^2/ms; the microscopic anisotropy has none.
# Notes:  tNAA. "tensor" is the paper's fit with a longitudinal and a transverse diffusivity,
#         "stick" the one with the transverse at zero (randomly oriented sticks, the model the
#         voxel's "sticks" plays); the table's p-values are not typed.
LUNDELL_2021 = {
    "tensor": {"CC": {"d_l": (0.51, 0.1), "d_t": (0.024, 0.01), "md": (0.19, 0.04),
                      "ufa": (0.94, 0.05)},
               "PWM": {"d_l": (0.50, 0.07), "d_t": (0.011, 0.01), "md": (0.17, 0.02),
                       "ufa": (0.98, 0.03)}},
    "stick": {"CC": {"d_l": (0.65, 0.1)}, "PWM": {"d_l": (0.58, 0.08)}},
}

#********************#
#   water kurtosis   #
#********************#
# Source: Latt, J., Nilsson, M., Wirestam, R., Stahlberg, F., Karlsson, N., Johansson, M.,
#         Sundgren, P. C., van Westen, D. Regional values of diffusional kurtosis estimates in
#         the healthy brain. Journal of Magnetic Resonance Imaging 37 (2013) 610-618, doi
#         10.1002/jmri.23857, Table 1: 36 volunteers of 19 to 64 years at 3 T, 26 structures.
# Units:  (mean, sd between subjects): the mean kurtosis (no unit) and the mean diffusivity in
#         um^2/ms.
# Notes:  the table's radial values, anisotropy and age correlations are not typed. "pooled" is
#         the text's average of the mean kurtosis over all white and all grey structures at 20
#         and at 60 years (the grey ones are the deep nuclei: the table has no cortex).
LATT_2013 = {
    "regions": {
        "External capsule": {"mk": (0.81, 0.05), "md": (0.9, 0.05)},
        "ALIC": {"mk": (1.04, 0.1), "md": (0.87, 0.05)},
        "PLIC": {"mk": (1.23, 0.09), "md": (0.89, 0.09)},
        "CC, body": {"mk": (1.17, 0.07), "md": (0.92, 0.07)},
        "CC, genu": {"mk": (1.06, 0.11), "md": (0.93, 0.06)},
        "CC, splenium": {"mk": (1.32, 0.09), "md": (0.89, 0.09)},
        "Centrum semiovale": {"mk": (1.09, 0.04), "md": (0.8, 0.04)},
        "Cingulate, body": {"mk": (1.07, 0.07), "md": (0.86, 0.07)},
        "Cingulate, temporal": {"mk": (0.85, 0.08), "md": (0.92, 0.12)},
        "Corona radiata": {"mk": (1.09, 0.04), "md": (0.84, 0.05)},
        "CST, cerebral crus": {"mk": (1.23, 0.07), "md": (0.88, 0.08)},
        "IFO, anterior basal": {"mk": (0.86, 0.07), "md": (0.89, 0.05)},
        "ILF, posterior": {"mk": (0.96, 0.06), "md": (0.9, 0.06)},
        "SLF, posterior": {"mk": (1.11, 0.04), "md": (0.83, 0.04)},
        "Frontal sWM": {"mk": (0.94, 0.05), "md": (0.91, 0.05)},
        "Parietal sWM": {"mk": (1.0, 0.05), "md": (0.86, 0.06)},
        "Temporal sWM": {"mk": (0.96, 0.07), "md": (0.88, 0.08)},
        "Caudate head": {"mk": (0.61, 0.08), "md": (0.87, 0.05)},
        "Globus pallidus": {"mk": (1.06, 0.08), "md": (0.86, 0.08)},
        "Putamen": {"mk": (0.67, 0.08), "md": (0.79, 0.03)},
        "Thalamus": {"mk": (0.86, 0.07), "md": (0.87, 0.1)},
        "IC, gradient 1": {"mk": (0.89, 0.1), "md": (0.85, 0.06)},
        "IC, gradient 2": {"mk": (1.05, 0.12), "md": (0.9, 0.07)},
        "IC, gradient 3": {"mk": (1.1, 0.1), "md": (0.92, 0.09)},
        "IC, gradient 4": {"mk": (1.21, 0.09), "md": (0.91, 0.12)},
        "IC, gradient 5": {"mk": (1.21, 0.11), "md": (0.87, 0.08)},
    },
    "pooled": {"wm": (1.08, 1.02), "gm": (0.83, 0.76)},
}
