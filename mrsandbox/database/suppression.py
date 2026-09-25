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
#         pulse table. The pulse is from Tkac and Gruetter, ISMRM 2000, abstract 1856 (the
#         4 T human scheme): "truncated 3 lobe asymmetric RF pulses of 35 ms duration" for
#         "less than 5% saturation outside the +- 0.4 ppm region", played "without water
#         suppression adjustment" (the ratio to the STEAM power holds); its timing there,
#         150, 100, 120, 105, 102, 65, 67, 15 ms, differs from the 2005 table's by a few ms
#         and is not typed. The lobes count the zero crossings a side of a symmetric sinc.
VAPOR = {
    "pulse": {"duration_s": 0.035, "lobes": 2, "adjusted": False},
    "power_db": (0.0, 0.0, 5.0, 0.0, 4.0, 0.0, 5.0, 5.0),
    "delays_s": {
        "4t": (0.150, 0.100, 0.140, 0.105, 0.105, 0.066, 0.072, 0.018),
        "7t": (0.150, 0.100, 0.122, 0.105, 0.102, 0.061, 0.067, 0.014),
    },
}

#**********#
#   cows   #
#**********#
# Source: Igwe, K. C., Gajdosik, M., Juchem, C., Landheer, K. Constrained optimized water
#         suppression for 1H MR spectroscopy. Magnetic Resonance in Medicine 94 (2025)
#         895-904, Table 1: the three schemes of its experiments (Prisma 3 T, sLASER, TR 2 s):
#         COWS(7;236) and COWS(12;626), flips and delays constrained optimized to null the
#         water over T1 0.5 to 5 s and a B1 factor 0.75 to 1.25 with no steady state assumed,
#         and VAPOR7, the seven pulse VAPOR the paper held them against.
# Units:  flip angles in degrees; delays in seconds from each pulse's peak to the next pulse's
#         peak, the last to the excitation's, so a scheme's delays sum to the duration in its
#         name. The pulses are 10 percent cutoff Gaussians of 68 Hz and 17.92 ms (the 68 Hz is
#         the full width at half of the saturation, 1 - Mz: that is what a Gaussian of that
#         truncation and duration gives, checked in the engine), each followed by a 9 ms crusher
#         (their DOTCOPS directions are in the table and not typed: the package keeps the
#         longitudinal magnetization alone).
# Notes:  CHECKED against the dataset's headers (OpenNeuro ds006812, sub-01, 25/09/26): the
#         amplitudes of ss_rf_ws1 to ss_rf_ws12 over the 11.7936 V of the 90 degree pulses
#         (reference 221.3 V; the study calibrated no flip per voxel) give every flip of the
#         table to 0.1 degrees. The header's lRFExcitationBandwidth reads 50 Hz, the product's
#         field, where the paper states its 68 Hz pulses.
COWS = {
    "pulse_bandwidth_hz": 68.0, "pulse_duration_s": 0.01792, "pulse_truncation": 0.1,
    "COWS(7;236)": {"flips_deg": (75.4, 86.2, 86.4, 72.4, 82.3, 116.7, 160.6),
                    "delays_s": (0.028, 0.028, 0.053032, 0.039794, 0.028, 0.038214, 0.021117)},
    "COWS(12;626)": {"flips_deg": (75.8, 3.4, 110.6, 141.1, 180.0, 20.1, 89.6, 76.3, 117.2, 79.6, 127.7, 167.6),
                     "delays_s": (0.082664, 0.028, 0.030553, 0.028, 0.061701, 0.051238, 0.099274, 0.062789,
                                  0.087689, 0.028, 0.044538, 0.021797)},
    "VAPOR7": {"flips_deg": (90.0, 90.0, 160.2, 90.0, 160.2, 90.0, 160.2),
               "delays_s": (0.150, 0.080, 0.160, 0.080, 0.100, 0.030, 0.026)},
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
# Siemens' product water saturation as svs_se plays it (syngo MR B17), read from the TWIX headers of
# Big GABA site S1, subjects S01, S03 and S05 (24/09/26, info/HANDOFF.md item 217):
# sSpecPara.lRFExcitationBandwidth 50 Hz, and the amplitudes of ss_rf_ws1, ws2, ws3 in the ratio
# 1 : 0.935 : 1.803 in every subject (16.30, 15.24, 29.38 V in S01; the scanner's water suppression
# adjustment scales all three at once, the protocol's 18.56, 17.35, 33.45 V became those). The third
# pulse overtips, as WET's last one does, so the water recovering through the last gap crosses zero
# at the excitation whatever its T1. The header holds no timing: the 30 ms spacing (a pulse of
# about 26 ms and its spoiler) is STATED.
SIEMENS_SVS_SE = {"bandwidth_hz": 50.0, "amplitude_ratios": (1.0, 0.935, 1.803), "spacing_s": 0.030}
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
