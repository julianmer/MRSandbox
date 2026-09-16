####################################################################################################
#                                          deuterium.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/09/26                                                                                #
#                                                                                                  #
# Purpose: The deuterium (2H) spin systems and constants of deuterium metabolic imaging, typed     #
#          from the literature with the citation beside each dict, in the record format of the    #
#          Govindaraju table so SpinSystem.from_database(name, source="dmi") reads them.           #
#                                                                                                  #
####################################################################################################
#******************#
#   deuterium      #
#******************#
# Source: Ruhm et al. (2024) Advances and prospects in deuterium metabolic imaging (DMI): a
#         systematic review of in vivo studies. Table 5 (in vivo 2H chemical shifts of the
#         detected deuterated metabolites) and the T1 values it quotes from de Graaf et al.
#         (2018) Deuterium metabolic imaging (DMI) for MRI-based 3D mapping of metabolism in
#         vivo, Science Advances 4:eaat7314 (doi 10.1126/sciadv.aat7314), rat brain at 11.7 T.
# Data:   One uncoupled 2H spin per resonance. The 2H-2H couplings are unresolved at the in
#         vivo linewidths and the quadrupolar interaction averages to zero in solution (it
#         drives the relaxation, not the spectrum), so every entry is a singlet (stated).
#         "scale" is one deuteron: concentrations are in mM of 2H, so a [6,6'-2H2]glucose
#         at 1 mM enters as 2 mM (stated convention).
# Units:  ppm, s. T2 values are not reported by either source and are not typed.
DMI = {
    "Water": {"nucleus": "2H", "t1_s": 0.320,
              "groups": [{"name": "HDO", "spins": {"D1": 4.8}}]},
    "Glc": {"nucleus": "2H", "t1_s": 0.064,
            "groups": [{"name": "Glc", "spins": {"D1": 3.8}}]},
    "Glx": {"nucleus": "2H", "t1_s": 0.146,
            "groups": [{"name": "Glx", "spins": {"D1": 2.4}}]},
    "Lac": {"nucleus": "2H", "t1_s": 0.297,
            "groups": [{"name": "Lac", "spins": {"D1": 1.4}}]},
}

# Source: as above (Ruhm et al. 2024, Background section).
# Data:   natural-abundance deuterated water in tissue.
# Units:  mM of 2H.
DMI_NATURAL_HDO_MM = 10.12

