####################################################################################################
#                                         concentrations.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Metabolite concentrations in mM, one dict per source, stored exactly as published with  #
#          the citation beside each dict. Pooling or merging across sources is code, never data.   #
#          Names are canonical and exact: compound totals (tCr, tNAA, Glx) are distinct keys from  #
#          single molecules and never interchangeable.                                             #
#                                                                                                  #
####################################################################################################

#**************#
#   de graaf   #
#**************#
# Source: de Graaf, R. A. In Vivo NMR Spectroscopy: Principles and Techniques, 3rd
#         edition. Wiley, 2019; as compiled in julianmer/WAND-for-MRS
#         simulation/simulationDefs.py (normConcs).
# Units:  (low, high) in mM, healthy adult human brain.
# Notes:  macromolecules excluded, their scale being implementation dependent.
DE_GRAAF = {
    "Ace": (0.0, 0.5),        # Acetate
    "Ala": (0.1, 1.6),        # Alanine
    "Asc": (0.4, 1.7),        # Ascorbate
    "Asp": (1.0, 2.0),        # Aspartate
    "ATP": (2.0, 4.0),        # Adenosine triphosphate
    "BCAA": (0.0, 0.5),       # Branched chain amino acids
    "Cr": (4.5, 10.5),        # Creatine
    "EA": (0.0, 1.6),         # Ethanolamine
    "GABA": (1.0, 2.0),       # Gamma-aminobutyric acid
    "Glc": (1.0, 2.0),        # Glucose
    "Gln": (3.0, 6.0),        # Glutamine
    "Glu": (6.0, 12.5),       # Glutamate
    "Gly": (0.2, 1.0),        # Glycine
    "Glycogen": (3.0, 6.0),   # Glycogen
    "GPC": (0.4, 1.7),        # Glycerophosphocholine
    "GSH": (1.7, 3.0),        # Glutathione
    "Ins": (4.0, 9.0),        # Myo-inositol
    "Lac": (0.2, 1.0),        # Lactate
    "NAA": (7.5, 12.0),       # N-acetylaspartate
    "NAAG": (0.5, 2.5),       # N-acetylaspartylglutamate
    "PCho": (0.2, 1.0),       # Phosphocholine
    "PCr": (3.0, 5.5),        # Phosphocreatine
    "PE": (1.0, 2.0),         # Phosphorylethanolamine
    "Ser": (0.2, 2.0),        # Serine
    "sIns": (0.2, 0.5),       # Scyllo-inositol
    "Tau": (3.0, 6.0),        # Taurine
    "Thr": (0.0, 0.5),        # Threonine
}
