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

#**********************************#
#   grey and white matter levels   #
#**********************************#
# Source: the in vivo studies of healthy adults that take a metabolite's level in pure grey and
#         pure white matter by regressing their voxels on the tissue fractions, or by solving
#         voxels of known composition together, newest first; each entry cites its paper.
# Data:   per study, per metabolite under its paper's name (database.regional NAMES maps them):
#         (grey, white) as printed, or the grey over white ratio where the paper prints only
#         that.
# Notes:  the studies reference their levels differently (a phantom, the tissue water, an
#         institutional unit, a creatine scaling), and a study's ratio holds within its own
#         referencing: the ratios are taken as printed, none converted (Gasparovic's per kilogram
#         of tissue water stay so). The N-acetyl singlet (NAA with NAAG) stands for NAA; Glx is
#         not typed. Left out: Kirov et al. 2013 (J Neurol 260:242 prints Tal 2012's controls
#         again; Neurology 80:39 the same group and method), studies
#         of tissue-rich voxels or masks without the regression (partial volume pulls their
#         contrast toward one), long-echo studies without metabolite relaxation, ratios to
#         creatine or NAA alone, and GABA+ with its macromolecules (Zhu et al. 2011).
TISSUE_STUDIES = (
    {"cited": "Klauser, Strasser, Bogner, Hingerl et al., Imaging Neuroscience 2 (2024) 1-20, "
              "doi 10.1162/imag_a_00313: 7 T FID-MRSI, a general linear model of the tissue "
              "fractions per lobe, 5 adults; the ratios printed in the results, myo-inositol "
              "the means of Table 1's five lobes",
     "values": {"tCr": 1.19, "NAA": 1.12, "Glu": 1.2, "Gln": 1.34, "GABA": 1.15, "GSH": 1.11,
                "tCho": 0.96, "NAAG": 0.47, "mI": (3.606, 3.634)}},
    {"cited": "Tiwari, An, Ganji et al., NMR in Biomedicine 30 (2017) e3747, "
              "doi 10.1002/nbm.3747: 3 T, an occipital and a parietal voxel regressed to pure "
              "tissue, 6 adults; the text's pure tissue values, NAA's ratio as printed",
     "values": {"mI": (10.1, 4.7), "tCr": (9.1, 4.8), "Gly": (0.76, 0.29), "NAA": 1.4}},
    {"cited": "Ding, Maudsley, Sabati, Sheriff, Dellani, Lanfermann, Magnetic Resonance in "
              "Medicine 73 (2015) 921-928, doi 10.1002/mrm.25208: 3 T whole-brain EPSI at TE "
              "17.6 ms, regression per lobe, 20 adults; the means over Table 2's eight cerebral "
              "lobes",
     "values": {"NAA": (11.0263, 9.3275), "tCho": (1.5075, 1.7188), "tCr": (8.9562, 6.55),
                "mI": (5.205, 4.1213)}},
    {"cited": "Bhattacharyya, Phillips, Stone, Lowe, Magnetic Resonance Imaging 29 (2011) 374-379, "
              "doi 10.1016/j.mri.2010.10.009: 3 T MEGA-PRESS, regression on the grey fraction, 9 "
              "adults",
     "values": {"GABA": (2.87, 0.33)}},
    {"cited": "Gasparovic, Bedrick, Mayer et al., Magnetic Resonance in Medicine 66 (2011) "
              "324-332, doi 10.1002/mrm.22858: 3 T PRESS MRSI at TE 40 ms, tissue water "
              "referenced, relaxation corrected, regression, 21 adults; Table 1",
     "values": {"NAA": (19.54, 14.23), "tCr": (17.52, 9.67), "tCho": (3.27, 3.17),
                "mI": (14.48, 7.96), "Glu": (22.06, 10.49)}},
    {"cited": "Gasparovic, Yeo, Mannell et al., Journal of Neurotrauma 26 (2009) 1635-1643, "
              "doi 10.1089/neu.2009.0896: 3 T PRESS MRSI at TE 40 ms, regression, 9 controls; "
              "Table 4",
     "values": {"NAA": (20.96, 17.22), "tCho": (3.16, 2.90), "tCr": (17.40, 9.05),
                "Glu": (21.44, 8.39)}},
    {"cited": "Ganji, An, Banerjee, Madan, Hulsey, Choi, NMR in Biomedicine 27 (2014) 1167-1175, "
              "doi 10.1002/nbm.3170: 7 T PRESS, four voxels regressed, 9 adults; the intercepts' "
              "ratio",
     "values": {"GABA": 7.5}},
    {"cited": "Tal, Kirov, Grossman, Gonen, NMR in Biomedicine 25 (2012) 1392-1400, "
              "doi 10.1002/nbm.2812: 3 T 3D MRSI at TE 35 ms, global levels from the segmented "
              "fractions, 18 adults; the abstract",
     "values": {"NAA": (8.5, 7.7), "tCr": (6.9, 4.9), "tCho": (1.2, 1.4), "mI": (5.3, 4.4)}},
    {"cited": "Banerjee, Ganji, Hulsey et al., Magnetic Resonance in Medicine 68 (2012) 325-331, "
              "doi 10.1002/mrm.24368: 7 T PRESS at TE 150 ms, 44 voxels regressed, 11 adults",
     "values": {"mI": (6.9, 1.6), "Gly": (1.1, 0.1)}},
    {"cited": "Choi, Bhardwaj, Kalra et al., Magnetic Resonance in Medicine 58 (2007) 27-33, "
              "doi 10.1002/mrm.21275: 3 T double quantum filter, two voxels solved with their "
              "water segmentation, 3 adults; the abstract",
     "values": {"GABA": (1.1, 0.4)}},
    {"cited": "Choi, Coupland, Bhardwaj et al., Magnetic Resonance in Medicine 56 (2006) 971-977, "
              "doi 10.1002/mrm.21055: 3 T PRESS, two voxels solved with their water segmentation, "
              "T2 corrected, 5 adults; the abstract",
     "values": {"Glu": (10.37, 5.06), "tCr": (8.87, 5.16)}},
    {"cited": "Choi, Lee, Merkle, Shen, NeuroImage 33 (2006) 85-93, "
              "doi 10.1016/j.neuroimage.2006.06.016: 3 T multiple quantum filtered CSI, regression "
              "on the grey fraction, 13 adults; the abstract",
     "values": {"GABA": (1.30, 0.16)}},
    {"cited": "Gasparovic, Song, Devier et al., Magnetic Resonance in Medicine 55 (2006) "
              "1219-1226, doi 10.1002/mrm.20901: MRSI, tissue water referenced, segmented; the "
              "abstract (its field and size not in it)",
     "values": {"NAA": (17.16, 14.26), "tCho": (3.27, 2.65), "tCr": (13.98, 7.10)}},
    {"cited": "Pan, Venkatraman, Vives, Spencer, NMR in Biomedicine 19 (2006) 209-216, "
              "doi 10.1002/nbm.1019: glutamate spectroscopic imaging, segmented and regressed, "
              "hippocampal and temporal grey against temporal white matter; the abstract",
     "values": {"Glu": (8.8, 3.7)}},
    {"cited": "Jensen, Frederick, Renshaw, NMR in Biomedicine 18 (2005) 570-576, "
              "doi 10.1002/nbm.994: 4 T J-resolved MRSI, regression, 6 adults; the abstract",
     "values": {"GABA": (0.96, 0.44)}},
    {"cited": "McLean, Woermann, Barker, Duncan, Magnetic Resonance in Medicine 44 (2000) 401-411: "
              "short echo MRSI, LCModel, segmented, 13 adults; the ratios of the abstract",
     "values": {"NAA": 1.16, "tCr": 1.7, "mI": 1.6, "tCho": 0.9}},
    {"cited": "Noworolski, Nelson, Henry et al., Magnetic Resonance in Medicine 41 (1999) 21-29: "
              "1.5 T PRESS MRSI, segmented, 23 adults; the ratios of the abstract, choline the "
              "occipital lobe's",
     "values": {"NAA": 1.4, "tCr": 1.4, "tCho": 0.73}},
    {"cited": "Wang, Li, Magnetic Resonance in Medicine 39 (1998) 28-33, "
              "doi 10.1002/mrm.1910390107: 1.5 T occipital single voxel at TE 136 ms, segmented, "
              "tissue water referenced with each tissue's water content, relaxation corrected; the "
              "abstract",
     "values": {"NAA": (11.0, 7.5), "tCr": (9.7, 5.2), "tCho": (1.9, 1.6)}},
    {"cited": "Hetherington, Mason, Pan et al., Magnetic Resonance in Medicine 32 (1994) 565-571, "
              "doi 10.1002/mrm.1910320504: 4.1 T spectroscopic imaging, each tissue's T1 and T2 "
              "corrected; the abstract's white over grey creatine content of 0.8",
     "values": {"tCr": (1.0, 0.8)}},
)
