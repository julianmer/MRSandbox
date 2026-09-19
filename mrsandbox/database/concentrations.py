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

#*******************************#
#   mrsi challenge per tissue   #
#*******************************#
# Source: the 2024 MRSI reconstruction challenge dataset paper (LaMaster, Merkofer, et al.,
#         arXiv:2609.02637), Table 1: the fixed simulation concentrations per tissue.
# Data:   per basis function: (gm_mm, wm_mm) at 3 T. The paper chose the values with
#         reference to the Gudmundson 2023 meta-analysis compilation and the in vivo
#         studies it cites, settled pragmatically since no single reference covers every
#         basis in both tissues, and reports them in full for traceability.
# Units:  mM.
# Notes:  the split rows tCr39, PE40, and Lac41 repeat their parents' values (basis
#         bookkeeping) and are not retyped; Lac carries zero by design.
MRSI_CHALLENGE = {
    "Asp": (3.70, 2.07),
    "GABA": (1.86, 0.88),
    "Gln": (3.90, 2.20),
    "Glu": (8.60, 6.00),
    "Lac": (0.0, 0.0),
    "NAA": (10.98, 7.49),
    "NAAG": (1.40, 2.60),
    "PE": (2.12, 2.12),
    "Tau": (1.93, 1.74),
    "mIns": (4.30, 3.10),
    "sIno": (0.50, 0.57),
    "tCho": (1.38, 1.78),
    "tCr": (6.90, 5.50),
}


#*************#
#   spectre   #
#*************#
# Source: Gold Standard Phantoms, SPECTRE (Spectroscopy Reference) information sheet,
#         PD-1004-1001 issue 4.0. https://www.goldstandardphantoms.com
# Units:  mM, one value per metabolite: a solution, in vitro.
# Notes:  a sphere of 180 mm (SPECTRE) or 110 mm (SPECTRE Pico) outer diameter, HDPE shell;
#         water based, 50 mM potassium phosphate buffer, pH 7.4 at 20 C, doped with Gd-DTPA
#         (amount not stated, so the relaxation times are not known from the sheet), a
#         CMIT:MIT preservative. Choline is choline chloride. No macromolecules, no lipids.
SPECTRE = {
    "NAA": 12.5,
    "Cr": 10.0,
    "Cho": 3.0,
    "Ins": 7.5,
    "Glu": 12.5,
    "Lac": 5.0,
    "GABA": 2.0,
}
