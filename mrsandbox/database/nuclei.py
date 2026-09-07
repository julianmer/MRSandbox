####################################################################################################
#                                            nuclei.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Nuclear constants, one dict per quantity, with the citation beside each table.          #
#                                                                                                  #
####################################################################################################

#**********************#
#   gamma per nucleus  #
#**********************#
# Source: Harris, R. K., et al. NMR nomenclature (IUPAC Recommendations 2001). Pure and
#         Applied Chemistry 73 (2001) 1795-1818. doi:10.1351/pac200173111795.
# Units:  gamma-bar = gamma / 2 pi, in MHz per tesla.
GAMMA_MHZ_PER_T = {
    "1H": 42.5774785,
    "2H": 6.5359026,
    "13C": 10.7084,
    "19F": 40.0776,
    "23Na": 11.2620,
    "31P": 17.2351,
}
