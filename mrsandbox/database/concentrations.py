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
#         edition. Wiley, 2019, Figure 2.3 (cerebral metabolite concentration ranges).
# Data:   the figure's bars, the approximate minimum and maximum reported in the literature
#         for normal adult human and rat brain; they stand on half millimolars, the lows of
#         0.1 and 0.2 apart. The book prints no numbers beside them.
# Units:  (low, high) in mM.
# Notes:  macromolecules excluded, their scale being implementation dependent; the figure's
#         total choline bar (0.5 to 2.5) is not typed, its two compounds are.
DE_GRAAF = {
    "Ace": (0.0, 0.5),        # Acetate
    "Ala": (0.1, 1.5),        # Alanine
    "Asc": (0.5, 1.5),        # Ascorbate
    "Asp": (1.0, 2.0),        # Aspartate
    "ATP": (2.0, 4.0),        # Adenosine triphosphate
    "BCAA": (0.0, 0.5),       # Branched chain amino acids
    "Cr": (4.5, 10.5),        # Creatine
    "EA": (0.0, 1.5),         # Ethanolamine
    "GABA": (1.0, 2.0),       # Gamma-aminobutyric acid
    "Glc": (1.0, 2.0),        # Glucose
    "Gln": (3.0, 6.0),        # Glutamine
    "Glu": (6.0, 12.5),       # Glutamate
    "Gly": (0.2, 1.0),        # Glycine
    "Glycogen": (3.0, 6.0),   # Glycogen
    "GPC": (0.5, 1.5),        # Glycerophosphocholine
    "GSH": (1.5, 3.0),        # Glutathione
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
# Data:   per study: `values`, per metabolite under its paper's name (database.regional NAMES
#         maps them), (grey, white) as printed or the grey over white ratio where the paper
#         prints only that; the acquisition (`field_t`, `sequence`, `te_ms`, `edited`), how the
#         pure tissues were taken (`pure`), the subjects (`n`); `referencing`, what the values are
#         per: "tissue" (a litre of tissue: a phantom, or the tissue water with each tissue's
#         water content folded in), "water" (the tissue water: a kilogram of it, or the voxel's
#         water signal with no content folded in), or None where the paper as readable does not
#         say or leaves a bias in (`unconverted` says which).
# Units:  as printed, each study its own; Priors.tissue_contrast converts them to the package's,
#         mmol per litre of tissue (a value per tissue water times that tissue's water content,
#         so the simulated metabolite over water signal is the measured one).
# Notes:  the N-acetyl singlet (NAA with NAAG) stands for NAA; Glx is not typed. Left out: Kirov
#         et al. 2013 (J Neurol 260:242 prints Tal 2012's controls again; Neurology 80:39 the same
#         group and method), tissue-rich voxels or masks without the regression (partial volume
#         pulls their contrast toward one), long-echo studies without metabolite relaxation,
#         ratios to creatine or NAA alone, and GABA+ with its macromolecules (Zhu et al. 2011).
TISSUE_STUDIES = (
    {"cited": "Klauser, Strasser, Bogner, Hingerl et al., Imaging Neuroscience 2 (2024) 1-20, "
              "doi 10.1162/imag_a_00313: FID-MRSI, a general linear model of the FreeSurfer "
              "fractions per lobe, LCModel water scaling; the ratios printed in the results, "
              "myo-inositol the means of Table 1's five lobes",
     "field_t": 7.0, "sequence": "FID-MRSI", "te_ms": 0.9, "edited": False, "pure": "model",
     "n": 5, "referencing": None,
     "unconverted": "the water's and the metabolites' T1 saturation at TR 275 ms is left in",
     "values": {"tCr": 1.19, "NAA": 1.12, "Glu": 1.2, "Gln": 1.34, "GABA": 1.15, "GSH": 1.11,
                "tCho": 0.96, "NAAG": 0.47, "mI": (3.606, 3.634)}},
    {"cited": "Tiwari, An, Ganji et al., NMR in Biomedicine 30 (2017) e3747, "
              "doi 10.1002/nbm.3747: triple refocusing, an occipital and a parietal voxel "
              "regressed, normalised to the voxel's tissue water (no water content stated) and "
              "scaled to an occipital tCr of 8 mM; the text's pure tissue values, NAA's ratio as "
              "printed",
     "field_t": 3.0, "sequence": "triple refocusing", "te_ms": 134.0, "edited": False,
     "pure": "regression", "n": 6, "referencing": "water",
     "values": {"mI": (10.1, 4.7), "tCr": (9.1, 4.8), "Gly": (0.76, 0.29), "NAA": 1.4}},
    {"cited": "Ding, Maudsley, Sabati, Sheriff, Dellani, Lanfermann, Magnetic Resonance in "
              "Medicine 73 (2015) 921-928, doi 10.1002/mrm.25208: whole-brain EPSI, regression "
              "per lobe, MIDAS units to the voxel's water made a 100 percent water equivalent "
              "(Maudsley et al. 2009); the means over Table 2's eight cerebral lobes",
     "field_t": 3.0, "sequence": "EPSI", "te_ms": 17.6, "edited": False, "pure": "regression",
     "n": 20, "referencing": "tissue",
     "values": {"NAA": (11.0263, 9.3275), "tCho": (1.5075, 1.7188), "tCr": (8.9562, 6.55),
                "mI": (5.205, 4.1213)}},
    {"cited": "Bhattacharyya, Phillips, Stone, Lowe, Magnetic Resonance Imaging 29 (2011) "
              "374-379, doi 10.1016/j.mri.2010.10.009: regression on the grey fraction, GABA over "
              "creatine times creatine by Gasparovic's tissue water method",
     "field_t": 3.0, "sequence": "MEGA-PRESS", "te_ms": 68.0, "edited": True,
     "pure": "regression", "n": 9, "referencing": "water",
     "values": {"GABA": (2.87, 0.33)}},
    {"cited": "Gasparovic, Bedrick, Mayer et al., Magnetic Resonance in Medicine 66 (2011) "
              "324-332, doi 10.1002/mrm.22858: PRESS MRSI, tissue water referenced, relaxation "
              "corrected, regression; Table 1",
     "field_t": 3.0, "sequence": "PRESS MRSI", "te_ms": 40.0, "edited": False,
     "pure": "regression", "n": 21, "referencing": "water",
     "values": {"NAA": (19.54, 14.23), "tCr": (17.52, 9.67), "tCho": (3.27, 3.17),
                "mI": (14.48, 7.96), "Glu": (22.06, 10.49)}},
    {"cited": "Gasparovic, Yeo, Mannell et al., Journal of Neurotrauma 26 (2009) 1635-1643, "
              "doi 10.1089/neu.2009.0896: PRESS MRSI, mmol per kg of MR-visible water, "
              "regression, controls; Table 4",
     "field_t": 3.0, "sequence": "PRESS MRSI", "te_ms": 40.0, "edited": False,
     "pure": "regression", "n": 9, "referencing": "water",
     "values": {"NAA": (20.96, 17.22), "tCho": (3.16, 2.90), "tCr": (17.40, 9.05),
                "Glu": (21.44, 8.39)}},
    {"cited": "Ganji, An, Banerjee, Madan, Hulsey, Choi, NMR in Biomedicine 27 (2014) 1167-1175, "
              "doi 10.1002/nbm.3170: four voxels regressed, normalised to the grey and white "
              "water (no water content stated); the intercepts' ratio",
     "field_t": 7.0, "sequence": "PRESS", "te_ms": 92.0, "edited": False, "pure": "regression",
     "n": 9, "referencing": "water",
     "values": {"GABA": 7.5}},
    {"cited": "Tal, Kirov, Grossman, Gonen, NMR in Biomedicine 25 (2012) 1392-1400, "
              "doi 10.1002/nbm.2812: 3D PRESS MRSI, phantom replacement per volume of grey and "
              "white matter, 480 voxels solved together; the abstract",
     "field_t": 3.0, "sequence": "PRESS MRSI", "te_ms": 35.0, "edited": False,
     "pure": "solved", "n": 18, "referencing": "tissue",
     "values": {"NAA": (8.5, 7.7), "tCr": (6.9, 4.9), "tCho": (1.2, 1.4), "mI": (5.3, 4.4)}},
    {"cited": "Banerjee, Ganji, Hulsey et al., Magnetic Resonance in Medicine 68 (2012) 325-331, "
              "doi 10.1002/mrm.24368: 44 voxels regressed, tissue water with the contents 0.81 "
              "and 0.71 folded in",
     "field_t": 7.0, "sequence": "PRESS", "te_ms": 150.0, "edited": False,
     "pure": "regression", "n": 11, "referencing": "tissue",
     "values": {"mI": (6.9, 1.6), "Gly": (1.1, 0.1)}},
    {"cited": "Choi, Bhardwaj, Kalra et al., Magnetic Resonance in Medicine 58 (2007) 27-33, "
              "doi 10.1002/mrm.21275: double quantum filter, two voxels solved with their water "
              "segmentation; the abstract",
     "field_t": 3.0, "sequence": "double quantum filter", "te_ms": 82.0, "edited": True,
     "pure": "solved", "n": 3, "referencing": None,
     "unconverted": "the abstract does not say what the levels are per",
     "values": {"GABA": (1.1, 0.4)}},
    {"cited": "Choi, Coupland, Bhardwaj et al., Magnetic Resonance in Medicine 56 (2006) 971-977, "
              "doi 10.1002/mrm.21055: two voxels solved with their water segmentation, T2 "
              "corrected; the abstract",
     "field_t": 3.0, "sequence": "PRESS", "te_ms": 128.0, "edited": False, "pure": "solved",
     "n": 5, "referencing": None,
     "unconverted": "the abstract does not say what the levels are per",
     "values": {"Glu": (10.37, 5.06), "tCr": (8.87, 5.16)}},
    {"cited": "Choi, Lee, Merkle, Shen, NeuroImage 33 (2006) 85-93, "
              "doi 10.1016/j.neuroimage.2006.06.016: GABA over creatine regressed on the grey "
              "fraction; the abstract",
     "field_t": 3.0, "sequence": "multiple quantum filtered CSI", "te_ms": None, "edited": True,
     "pure": "regression", "n": 13, "referencing": None,
     "unconverted": "the creatine level the ratios were scaled by is not in the abstract",
     "values": {"GABA": (1.30, 0.16)}},
    {"cited": "Gasparovic, Song, Devier et al., Magnetic Resonance in Medicine 55 (2006) "
              "1219-1226, doi 10.1002/mrm.20901: MRSI, tissue water referenced, segmented; the "
              "abstract",
     "field_t": None, "sequence": "MRSI", "te_ms": None, "edited": False, "pure": None,
     "n": None, "referencing": None,
     "unconverted": "the abstract prints mM, the method's later papers mmol per kg water",
     "values": {"NAA": (17.16, 14.26), "tCho": (3.27, 2.65), "tCr": (13.98, 7.10)}},
    {"cited": "Pan, Venkatraman, Vives, Spencer, NMR in Biomedicine 19 (2006) 209-216, "
              "doi 10.1002/nbm.1019: LASER spectroscopic imaging, the cistern's CSF water as the "
              "reference, regression, hippocampal and temporal grey against temporal white matter",
     "field_t": 4.0, "sequence": "LASER MRSI", "te_ms": 37.5, "edited": False,
     "pure": "regression", "n": 10, "referencing": None,
     "unconverted": "the CSF water concentration it assumes is not stated",
     "values": {"Glu": (8.8, 3.7)}},
    {"cited": "Jensen, Frederick, Renshaw, NMR in Biomedicine 18 (2005) 570-576, "
              "doi 10.1002/nbm.994: J-resolved MRSI, regression; the abstract",
     "field_t": 4.0, "sequence": "J-resolved MRSI", "te_ms": None, "edited": False,
     "pure": "regression", "n": 6, "referencing": None,
     "unconverted": "the abstract does not say what the levels are per",
     "values": {"GABA": (0.96, 0.44)}},
    {"cited": "McLean, Woermann, Barker, Duncan, Magnetic Resonance in Medicine 44 (2000) 401-411: "
              "short echo MRSI, LCModel, segmented; the ratios of the abstract",
     "field_t": None, "sequence": "MRSI", "te_ms": None, "edited": False, "pure": None, "n": 13,
     "referencing": None, "unconverted": "ratios only, their referencing not in the abstract",
     "values": {"NAA": 1.16, "tCr": 1.7, "mI": 1.6, "tCho": 0.9}},
    {"cited": "Noworolski, Nelson, Henry et al., Magnetic Resonance in Medicine 41 (1999) 21-29: "
              "PRESS MRSI with surface coils, segmented; the ratios of the abstract, choline the "
              "occipital lobe's",
     "field_t": 1.5, "sequence": "PRESS MRSI", "te_ms": None, "edited": False, "pure": None,
     "n": 23, "referencing": None,
     "unconverted": "ratios only, signal or level not said in the abstract",
     "values": {"NAA": 1.4, "tCr": 1.4, "tCho": 0.73}},
    {"cited": "Wang, Li, Magnetic Resonance in Medicine 39 (1998) 28-33, "
              "doi 10.1002/mrm.1910390107: occipital single voxel, segmented, tissue water with "
              "each tissue's water content, relaxation corrected; the abstract",
     "field_t": 1.5, "sequence": "single voxel", "te_ms": 136.0, "edited": False, "pure": None,
     "n": None, "referencing": None,
     "unconverted": "the water contents it folded in are not in the abstract",
     "values": {"NAA": (11.0, 7.5), "tCr": (9.7, 5.2), "tCho": (1.9, 1.6)}},
    {"cited": "Hetherington, Mason, Pan et al., Magnetic Resonance in Medicine 32 (1994) 565-571, "
              "doi 10.1002/mrm.1910320504: spectroscopic imaging, each tissue's T1 and T2 "
              "corrected; the abstract's white over grey creatine content of 0.8",
     "field_t": 4.1, "sequence": "MRSI", "te_ms": None, "edited": False, "pure": None,
     "n": None, "referencing": None,
     "unconverted": "a content ratio whose referencing the abstract does not give",
     "values": {"tCr": (1.0, 0.8)}},
)
