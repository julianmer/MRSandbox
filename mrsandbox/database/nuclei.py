####################################################################################################
#                                             nuclei.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Nuclear constants, one dict per quantity, with the citation beside each table.          #
#                                                                                                  #
####################################################################################################

#***********************#
#   gamma per nucleus   #
#***********************#
# Source: Harris, R. K., et al. NMR nomenclature (IUPAC Recommendations 2001). Pure and
#         Applied Chemistry 73 (2001) 1795-1818. doi:10.1351/pac200173111795.
# Units:  gamma-bar = gamma / 2 pi, in MHz per tesla.
GAMMA_MHZ_PER_T = {
    "1H": 42.5774785,
    "2H": 6.5359026,
    "13C": 10.7084,
    "14N": 3.077705,
    "19F": 40.0776,
    "23Na": 11.2620,
    "31P": 17.2351,
}


#**********************#
#   spin quantum no.   #
#**********************#
# Source: Harris, R. K., et al. NMR nomenclature (IUPAC Recommendations 2001). Pure and
#         Applied Chemistry 73 (2001) 1795-1818. doi:10.1351/pac200173111795.
# Units:  nuclear spin quantum number I (dimensionless); the Hilbert space of one spin
#         has 2 I + 1 levels.
SPIN = {
    "1H": 0.5,
    "2H": 1.0,
    "13C": 0.5,
    "14N": 1.0,
    "19F": 0.5,
    "23Na": 1.5,
    "31P": 0.5,
}


#******************#
#   heteronuclei   #
#******************#
# Source: the nitrogen and phosphorus of the GOVINDARAJU spin systems (spinsystems.py, FID-A's
#         spinSystems.mat), which the table writes as protons at 100 and 40 ppm, named after the
#         nucleiList of MARSS's metabList (Landheer, Swanberg, Juchem, NMR Biomed 34 (2021)
#         e4129, the 2023-07-30 release), matched by their couplings: 0.46 to 2.9 Hz to 14N,
#         6.0 to 7.3 Hz to 31P. In GPC's choline group, where MARSS's nucleiList names the 2.67 Hz
#         partner a proton and the 6.03 Hz one 14N, the couplings decide; GPC's trimethyl group, a
#         singlet in MARSS, is named by its 0.57 Hz coupling, choline's methyl to its nitrogen.
# Units:  {metabolite: {group: {spin label: nucleus}}}.
HETERONUCLEI = {
    "Cho": {"Ch_3CH3": {"S4": "14N"}},
    "GPC": {"gp": {"S6": "31P"}, "pCh1": {"S5": "14N", "S6": "31P"}, "pCh2": {"S4": "14N"}},
    "PCho": {"PCh_ch2ch2": {"S5": "14N", "S6": "31P"}},
    "PE": {"PE": {"S5": "14N", "S6": "31P"}},
}
