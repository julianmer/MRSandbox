####################################################################################################
#                                          suppression.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/09/26                                                                                #
#                                                                                                  #
# Purpose: Published water suppression pulse series, one dict per source, stored exactly as        #
#          published with the citation beside each dict. Converting a power offset in dB to a      #
#          flip angle is code, never data.                                                         #
#                                                                                                  #
####################################################################################################

#*********#
#   wet   #
#*********#
# Source: Ogg, R. J., Kingsley, P. B., Taylor, J. S. WET, a T1- and B1-insensitive
#         water-suppression method for in vivo localized 1H NMR spectroscopy. Journal of
#         Magnetic Resonance, Series B 104 (1994) 1-10.
# Units:  four flip angles in degrees, numerically optimized to null the longitudinal
#         water insensitively to T1 and B1.
# Notes:  the paper optimizes the series for the sequence timing at hand, so the
#         interpulse delay stays a free timing.
WET = {
    "flips_deg": (81.4, 101.4, 69.3, 161.0),
}

#***********#
#   vapor   #
#***********#
# Source: Tkac, I., Gruetter, R. Methodology of 1H NMR spectroscopy of the human brain at
#         very high magnetic fields. Applied Magnetic Resonance 29 (2005) 139-157.
# Units:  peak power offsets in dB from the nominal water suppression pulse power, and the
#         delay series in seconds, optimized per field strength.
# Notes:  VAPOR was introduced as a seven pulse scheme in Tkac et al. 1999 (MRM 41,
#         649-656), whose flip table is not in any open text; this is the complete eight
#         pulse table.
VAPOR = {
    "power_db": (0.0, 0.0, 5.0, 0.0, 4.0, 0.0, 5.0, 5.0),
    "delays_s": {
        "4t": (0.150, 0.100, 0.140, 0.105, 0.105, 0.066, 0.072, 0.018),
        "7t": (0.150, 0.100, 0.122, 0.105, 0.102, 0.061, 0.067, 0.014),
    },
}

#*************************#
#   unsuppressed ratios   #
#*************************#
# Source: Bacon, J. B., Jezzard, P., Clarke, W. T. MR spectroscopy without water suppression
#         using the gradient impulse response function. MRM 96 (2026) 530-541, Table 2: the
#         ratios (mean +- SD, n = 8) of metabolite concentrations measured without and with
#         VAPOR water suppression (OVS on in both) on a 3 T Prisma, semi-LASER TE 30 ms in the
#         motor cortex and the MEGA-PRESS OFF spectra of TE 68 ms in the visual cortex, and
#         the semi-LASER ratio before over after system heating; the paper reads the excess
#         under suppression as water-mediated saturation (magnetization) transfer, creatine
#         the metabolite most affected, and finds none of it in a phantom (Table 3).
# Units:  dimensionless ratios of concentrations, (mean, SD).
# Notes:  labelled priors for the amplitude water suppression costs each metabolite; no
#         source applies them yet (a decision for the suppressed-water world).
BACON_2026_NON_SUPPRESSED = {
    "semi_laser": {
        "tNAA": (1.156, 0.051), "tCr": (1.535, 0.160), "mI": (1.242, 0.100),
        "tCho": (1.193, 0.053), "Glx": (0.927, 0.123),
    },
    "semi_laser_before_after_heating": {
        "tNAA": (0.998, 0.037), "tCr": (0.971, 0.074), "mI": (1.040, 0.117),
        "tCho": (0.975, 0.049), "Glx": (0.968, 0.174),
    },
    "mega_press_off": {
        "tNAA": (0.994, 0.030), "tCr": (1.069, 0.039), "tCho": (0.975, 0.052),
        "Glx": (1.009, 0.066),
    },
}
