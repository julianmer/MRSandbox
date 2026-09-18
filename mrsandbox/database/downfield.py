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
# Echo:   the levels belong to a spin echo at TE 22 ms (TR 287 ms, flip 78 degrees, 1331
#         excitation at 7.4 ppm, an 11 ms sinc-Gauss refocusing pulse at 7.8 ppm), with no
#         correction for T2 or exchange: a level is what is left at that echo time.
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

#*************************#
#   basis widths at 3 t   #
#*************************#
# Source: Povazan 2022 (above), Supporting Information Figure S1 A: the nine AMARES
#         Gaussians of the 1331-selref downfield spectrum at 3 T. The supporting
#         information tabulates no widths, so the drawn curves were digitized here
#         (11/09/26): nine Gaussians fitted to the curve pixels with the ppm axis
#         calibrated from the tick marks and the centres free within 0.06 ppm of the
#         nominal positions; residual 1.2 pixels, one pixel being 1.1 Hz, so the widths
#         carry about 2 Hz of digitizing uncertainty. Table S2 of the same document
#         reports the LCModel linewidth of the quantified resonances as 12.0 +- 2.4 Hz
#         across subjects (the resolved, narrower peaks; the amides near 7 ppm are broader).
# Data:   per peak: the fitted centre and the full width at half maximum of the basis
#         Gaussian, the observed 3 T width (intrinsic plus field).
# Units:  ppm; Hz FWHM at 127.76 MHz.
# Notes:  a digitized estimate, stated as such; the positions used for rendering stay
#         the published OZDEMIR values.
POVAZAN_WIDTHS = {
    "df683": {"ppm": 6.80, "fwhm_hz": 28.0},
    "df709": {"ppm": 7.07, "fwhm_hz": 30.0},
    "df730": {"ppm": 7.30, "fwhm_hz": 23.0},
    "df748": {"ppm": 7.50, "fwhm_hz": 18.0},
    "df790": {"ppm": 7.91, "fwhm_hz": 32.0},
    "df818": {"ppm": 8.16, "fwhm_hz": 18.0},
    "df824": {"ppm": 8.26, "fwhm_hz": 14.0},
    "df837": {"ppm": 8.35, "fwhm_hz": 16.0},
    "df849": {"ppm": 8.50, "fwhm_hz": 28.0},
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
