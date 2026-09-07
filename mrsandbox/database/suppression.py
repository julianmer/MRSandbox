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
