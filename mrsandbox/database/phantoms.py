####################################################################################################
#                                           phantoms.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Phantoms as their makers and the literature describe them: the contents of the MRS      #
#          spheres, the relaxation times measured in them, and the arrays of the ISMRM/NIST        #
#          system phantom. In vitro numbers; none of them sets a tissue leaf.                      #
#                                                                                                  #
####################################################################################################

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
#         No measured relaxation times of it were found in the literature (searched 19/09/26).
SPECTRE = {
    "NAA": 12.5,
    "Cr": 10.0,
    "Cho": 3.0,
    "Ins": 7.5,
    "Glu": 12.5,
    "Lac": 5.0,
    "GABA": 2.0,
}

#************#
#   braino   #
#************#
# Source: Lattanzi, R., Grant, A. K., Polimeni, J. R., Ohliger, M. A., Wiggins, G. C., Wald, L. L.,
#         Sodickson, D. K. Performance evaluation of a 32-element head array with respect to the
#         ultimate intrinsic SNR. NMR Biomed 23:142 (2010), doi 10.1002/nbm.1435, the earliest
#         statement read of the recipe of the GE "Braino" sphere (MRS HD sphere, model 2152220);
#         five later methods sections repeat it word for word (Lin 2014, doi 10.1002/mrm.24732;
#         Tur 2014, doi 10.1002/hbm.22229; Chiew 2018, doi 10.1002/nbm.3838; Jalnefjord 2021,
#         doi 10.1007/s10334-020-00900-1; Najac 2022, doi 10.1002/nbm.4801). GE's own document
#         was not seen: the recipe as the literature reports it.
# Units:  mM.
# Notes:  creatine is creatine hydrate, choline is choline chloride, glutamate L-glutamic acid,
#         lactate DL-lactic acid; besides them 0.1 percent sodium azide, 50 mM potassium phosphate
#         monobasic, 56 mM sodium hydroxide, 1 mL/L Gd-DPTA (Magnevist; its molarity is not in
#         the sources). The diameter is 169 mm in Lattanzi 2010, who modelled the sphere (An 2018,
#         doi 10.1002/mrm.26884, writes 10 cm); the pH is in none of the sources. The solution's
#         conductivity is 0.97 S/m and its relative permittivity 81.3 at 123 MHz (Lattanzi 2010).
BRAINO = {
    "NAA": 12.5,
    "Cr": 10.0,
    "Cho": 3.0,
    "Ins": 7.5,
    "Glu": 12.5,
    "Lac": 5.0,
}

#***********************#
#   braino relaxation   #
#***********************#
# Source: water T1 near 3 T: 0.450 s (Lattanzi 2010, above, 2.89 T) and 0.388 +- 0.043 s (Landheer,
#         K., Gajdosik, M., Juchem, C. J Magn Reson 2020, doi 10.1016/j.jmr.2019.106670).
#         Metabolite T1 at 7 T: An, L., Li, S., Shen, J. Magn Reson Med 78:2072 (2017), doi
#         10.1002/mrm.26612, Table 5 (inversion recovery PRESS, seven repeats), named as a Braino.
#         Metabolite T2 at 7 T: Li, S., An, L., Shen, J. Magn Reson Med 2018, doi 10.1002/mrm.26941,
#         Table 1 (multi-TE PRESS); the same group's doped sphere, NOT named as a Braino, its T1
#         (0.759, 0.449, 0.297 s) close to the named ones.
# Units:  seconds; (value, standard deviation) where one is printed.
# Notes:  no temperature is stated for any of them, and none was found at 1.5 or 3 T for the
#         metabolites. The gadolinium relaxes the metabolites unequally: choline's T1 is a third of
#         the NAA singlet's. T2 by multi-TE PRESS carries the sequence with it.
BRAINO_RELAXATION = {
    "water_t1_s": {"lattanzi_2010_3t": 0.450, "landheer_2020_3t": (0.388, 0.043)},
    "t1_s_7t": {"NAA": (0.799, 0.006), "NAA_CH2": (0.537, 0.015), "Glu": (0.541, 0.009),
                "Cr": (0.420, 0.008), "Cho": (0.264, 0.004)},
    "t2_s_7t": {"NAA": (0.530, 0.037), "Cr": (0.310, 0.024), "Cho": (0.200, 0.023),
                "Glu": 0.217, "Lac": 0.533},
}

#*****************#
#   nist system   #
#*****************#
# Source: Stupic, K. F., Ainslie, M., Boss, M. A., et al. A standard system phantom for magnetic
#         resonance imaging. Magn Reson Med 86:1194 (2021), doi 10.1002/mrm.28779: the geometry
#         from the text, the arrays from Supporting Information Tables S1 to S3 (the PROTOTYPE
#         phantom, properties at 20 C, concentrations by ICP), parsed from the supplement's text.
# Units:  per sphere ((x, y, z) mm from the centre, concentration, (T1, T2) ms at 1.5 T, (T1, T2)
#         ms at 3 T); the concentration is mM of NiCl2 (t1_array) and of MnCl2 (t2_array), and
#         percent of D2O in the water (pd_array: 95 is 5 percent of the protons).
# Notes:  a water-filled polycarbonate sphere of 200 mm inner diameter; each array is fourteen
#         polypropylene spheres of 15 mm inner diameter on one plate (y is the plate's height),
#         placed to 0.4 mm. The proton density spheres carry 0.101 mM NiCl2. PD-14's T2 at 1.5 T
#         repeats PD-1's in the source, a probable misprint. The commercial phantom (CaliberMRI
#         model 130) differs a little and has its own certificate per serial number, at 3 T from
#         16 to 26 C (github.com/MRIStandards/SystemPhantom, CalibrationData): use that for a
#         phantom in hand. Relaxivities at 20 C, 3 T (per mM per second): NiCl2 r1 0.652, r2
#         0.899; MnCl2 r1 7.002, r2 116.6; the water's own R1 0.316, R2 0.398 per second.
NIST_SYSTEM = {
    "inner_diameter_mm": 200.0,
    "sphere_inner_diameter_mm": 15.0,
    "water_r1_r2_3t": (0.316, 0.398),
    "t1_array": (
        ((0, 56.5, 50), 0.268, (2033, 1669), (1989, 1465)),
        ((29.4, 56.5, 40.5), 0.556, (1489, 1244), (1454, 1077)),
        ((47.6, 56.5, 15.5), 1.053, (1012, 859.3), (984.1, 717.9)),
        ((47.6, 56.5, -15.5), 1.683, (730.8, 628.5), (706, 510.1)),
        ((29.4, 56.5, -40.5), 2.579, (514.1, 446.3), (496.7, 359.6)),
        ((0, 56.5, -50), 3.835, (367.9, 321.2), (351.5, 255.5)),
        ((-29.4, 56.5, -40.5), 5.674, (260.1, 227.7), (247.13, 180.8)),
        ((-47.6, 56.5, -15.5), 8.142, (184.6, 161.9), (175.3, 127.3)),
        ((-47.6, 56.5, 15.5), 11.716, (132.7, 117.1), (125.9, 90.3)),
        ((-29.4, 56.5, 40.5), 16.83, (92.7, 81.9), (89, 64.3)),
        ((-20, 56.5, 20), 23.827, (65.4, 57.7), (62.7, 45.7)),
        ((20, 56.5, 20), 33.922, (46.32, 41), (44.53, 31.86)),
        ((20, 56.5, -20), 48.609, (32.45, 28.7), (30.84, 22.38)),
        ((-20, 56.5, -20), 68.884, (22.859, 20.2), (21.719, 15.83)),
    ),
    "t2_array": (
        ((0, 16.5, 50), 0.012, (2376, 939.4), (2480, 581.3)),
        ((29.4, 16.5, 40.5), 0.019, (2183, 594.3), (2173, 403.5)),
        ((47.6, 16.5, 15.5), 0.028, (1870, 416.5), (1907, 278.1)),
        ((47.6, 16.5, -15.5), 0.042, (1539, 267), (1604, 190.94)),
        ((29.4, 16.5, -40.5), 0.062, (1237, 184.9), (1332, 133.27)),
        ((0, 16.5, -50), 0.091, (1030, 140.6), (1044, 96.89)),
        ((-29.4, 16.5, -40.5), 0.131, (752.2, 91.76), (801.7, 64.07)),
        ((-47.6, 16.5, -15.5), 0.187, (550.2, 64.84), (608.6, 46.42)),
        ((-47.6, 16.5, 15.5), 0.267, (413.4, 45.28), (458.4, 31.97)),
        ((-29.4, 16.5, 40.5), 0.381, (292.2, 30.62), (336.5, 22.56)),
        ((-20, 16.5, 20), 0.54, (194.9, 19.76), (244.2, 15.813)),
        ((20, 16.5, 20), 0.762, (160.2, 15.99), (176.6, 11.237)),
        ((20, 16.5, -20), 1.08, (106.4, 10.47), (126.9, 7.911)),
        ((-20, 16.5, -20), 1.527, (83.33, 8.15), (90.9, 5.592)),
    ),
    "pd_array": (
        ((0, -23.5, 50), 95, (378.08, 320.61), (341.74, 236.25)),
        ((29.4, -23.5, 40.5), 90, (371.76, 311.21), (336.69, 233.59)),
        ((47.6, -23.5, 15.5), 85, (367.96, 313.8), (335.1, 233.89)),
        ((47.6, -23.5, -15.5), 80, (364.15, 308.79), (334.51, 233.02)),
        ((29.4, -23.5, -40.5), 75, (353.17, 299.65), (323.94, 227.16)),
        ((0, -23.5, -50), 70, (354.84, 300.41), (323.03, 226.04)),
        ((-29.4, -23.5, -40.5), 65, (365.43, 310.05), (343.58, 240.48)),
        ((-47.6, -23.5, -15.5), 60, (345.95, 295.2), (324.53, 228.05)),
        ((-47.6, -23.5, 15.5), 50, (343.76, 287.02), (319.87, 225.25)),
        ((-29.4, -23.5, 40.5), 40, (339.28, 285.67), (320.14, 226.31)),
        ((-20, -23.5, 20), 30, (340.05, 292.96), (316.07, 224.48)),
        ((20, -23.5, 20), 20, (328.78, 284.15), (309.31, 220.68)),
        ((20, -23.5, -20), 10, (329.23, 287.47), (311.68, 224.09)),
        ((-20, -23.5, -20), 0, (331.16, 320.61), (305.08, 219.06)),
    ),
}
