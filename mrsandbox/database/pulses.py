####################################################################################################
#                                             pulses.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Published characteristics of amplitude modulated RF pulses, stored exactly as           #
#          published with the citation beside the dict. These are the reference numbers shaped     #
#          pulse implementations are validated against.                                            #
#                                                                                                  #
####################################################################################################

#**************#
#   de graaf   #
#**************#
# Source: de Graaf, R. A. In Vivo NMR Spectroscopy: Principles and Techniques, 3rd
#         edition. Wiley, 2019. Table 5.1.
# Units:  keyed by (pulse name, nutation angle); r_bw = FWHM bandwidth (kHz) x length (ms),
#         measured on Mxy for 90 and on Mz for 180; r_tw = transition width product;
#         integrals over the shape normalized to [0, 1] (square pulse = 1).
# Notes:  sinc rows are the five-lobe pulse (third zero crossing); Gaussian rows state
#         their truncation level.
DE_GRAAF = {
    ("square", 90): {"r_bw": 1.37, "r_tw": 0.61, "integral": 1.0000, "rms_integral": 1.0000},
    ("square", 180): {"r_bw": 0.80, "r_tw": 0.60, "integral": 1.0000, "rms_integral": 1.0000},
    ("sinc", 90): {"r_bw": 5.95, "r_tw": 0.95, "integral": 0.1775, "rms_integral": 0.4011},
    ("sinc", 180): {"r_bw": 4.45, "r_tw": 2.27, "integral": 0.1775, "rms_integral": 0.4011},
    ("gaussian_1pct", 90): {"r_bw": 2.70, "r_tw": 1.86, "integral": 0.4116,
                            "rms_integral": 0.5401},
    ("gaussian_1pct", 180): {"r_bw": 1.53, "r_tw": 1.34, "integral": 0.4116,
                             "rms_integral": 0.5401},
    ("gaussian_10pct", 90): {"r_bw": 2.00, "r_tw": 1.14, "integral": 0.5650,
                             "rms_integral": 0.6416},
    ("gaussian_10pct", 180): {"r_bw": 1.15, "r_tw": 0.95, "integral": 0.5650,
                              "rms_integral": 0.6416},
    ("hermite", 90): {"r_bw": 5.43, "r_tw": 2.56, "integral": 0.1967, "rms_integral": 0.3956},
    ("hermite", 180): {"r_bw": 3.49, "r_tw": 1.81, "integral": 0.1541, "rms_integral": 0.3804},
}
