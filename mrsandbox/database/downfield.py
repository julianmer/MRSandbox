####################################################################################################
#                                           downfield.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: The downfield resonances between 6.83 and 8.49 ppm: the nine-peak parameterization of   #
#          the Johns Hopkins downfield MRSI papers, and the 3 T relaxation and water-exchange      #
#          rates that govern how these peaks vanish under water presaturation.                     #
#                                                                                                  #
####################################################################################################

#*******************#
#   nine df peaks   #
#*******************#
# Source: Ozdemir, I., Ganji, S., Gillen, J., Etyemez, S., Povazan, M., Barker, P. B. (2023).
#         Downfield proton MRSI with whole-brain coverage at 3T. Magnetic Resonance in
#         Medicine, 90(3), 814-822. doi:10.1002/mrm.29706. Positions and assignments from
#         Povazan, M., Schar, M., Gillen, J., Barker, P. B. (2022). Magnetic resonance
#         spectroscopic imaging of downfield proton resonances in the human brain at 3T.
#         Magnetic Resonance in Medicine, 87(4), 1661-1672. doi:10.1002/mrm.29142.
# Data:   the nine Gaussian basis peaks of the LCModel/AMARES downfield parameterization;
#         per peak: position and the reported concentration range in institutional units
#         (water referenced). Ozdemir reports the combined 8.18/8.24/8.37 amides at 5 to 8
#         i.u., the 7.90 NAA amide around 2 i.u., and the individual smaller peaks at 1 to
#         2 i.u.; the per-amide ranges below are stated so their sum lands in the reported
#         combined range.
# Units:  ppm; institutional units (approximately concentration equivalent mM, stated).
# Notes:  assignments per Povazan 2022: DF7.90 is the NAA amide (plus other amines),
#         DF8.18/8.24/8.37 are predominantly protein amides, DF8.24 and DF8.49 include ATP,
#         DF8.24 may include the glutathione glycine, DF7.09 likely homocarnosine.
OZDEMIR = {
    "df683": {"ppm": 6.83, "level_iu": (1.0, 2.0)},
    "df709": {"ppm": 7.09, "level_iu": (1.0, 2.0)},  # likely homocarnosine
    "df730": {"ppm": 7.30, "level_iu": (1.0, 2.0)},
    "df748": {"ppm": 7.48, "level_iu": (1.0, 2.0)},
    "df790": {"ppm": 7.90, "level_iu": (1.5, 2.5)},  # NAA amide, about 2 i.u.
    "df818": {"ppm": 8.18, "level_iu": (1.5, 3.0)},  # protein amide
    "df824": {"ppm": 8.24, "level_iu": (1.5, 3.0)},  # protein amide, ATP, GSH glycine
    "df837": {"ppm": 8.37, "level_iu": (1.5, 3.0)},  # protein amide
    "df849": {"ppm": 8.49, "level_iu": (1.0, 2.0)},  # ATP
}

#**********************************#
#   df t1 and exchange with water  #
#**********************************#
# Source: MacMillan, E. L., Chong, D. G. Q., Dreher, W., Henning, A., Boesch, C., Kreis, R.
#         (2011). Magnetization exchange with water and T1 relaxation of the downfield
#         resonances in human brain spectra at 3.0 T. Magnetic Resonance in Medicine,
#         65(5), 1239-1246. doi:10.1002/mrm.22813, Figure 6: median values over 11
#         volunteers from the two-pool Bloch-McConnell fit of the metabolite-cycled water
#         inversion transfer experiment.
# Data:   per prominent downfield peak: T1 in absence of exchange, and the forward
#         magnetization exchange rate with water.
# Units:  peak positions ppm; t1_ms milliseconds; kex_hz per second.
MACMILLAN = {
    "t1_ms": {8.50: 176.0, 8.24: 254.0, 7.86: 525.0, 7.32: 195.0, 7.09: 315.0, 6.84: 178.0},
    "kex_hz": {8.50: 8.9, 8.24: 7.5, 7.86: 0.5, 7.32: 1.3, 7.09: 0.9, 6.84: 4.2},
}
