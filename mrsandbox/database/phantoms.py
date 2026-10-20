####################################################################################################
#                                            phantoms.py                                           #
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
#         No measured relaxation times of it are known in the literature.
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

#********************#
#   nist model 130   #
#********************#
# Source: NIST Report of Measurement 80100S-20200610-C1, the solutions of the commercial system
#         phantom (CaliberMRI model 130), serial numbers 130-0042 to 130-0132; github.com/
#         MRIStandards/SystemPhantom, CalibrationData; parsed from the certificate's text.
# Units:  per sphere (concentration in mM as the maker supplied it, {temperature in C: (T1, T2)
#         in ms}) at 3.00672 T; the uncertainties (three standard deviations, one to two
#         percent) are in the certificate.
# Notes:  the spheres sit where the prototype's do (NIST_SYSTEM). MnCl2-1 (T2 1928 ms) and
#         MnCl2-5 (as MnCl2-4) fall off their array's trend; the certificate notes a
#         recalibration after those two solutions were replaced: not to be used unread. A
#         phantom in hand has the certificate of its own serial number. No 1.5 T values here.
NIST_MODEL_130 = {
    "t1_array": (
        (0.29, {16: (1766.68, 1348.91), 18: (1830.34, 1455.09), 20: (1883.97, 1489.41),
               22: (1937.34, 1520.77), 24: (1987.5, 1532.32), 26: (2066.95, 1578.57)}),
        (0.6, {16: (1274.07, 990.77), 18: (1317.71, 1010.28), 20: (1330.16, 1026.78),
              22: (1355.29, 1040.25), 24: (1367.79, 1048.13), 26: (1395.94, 1067.14)}),
        (1.04, {16: (950.71, 731.7), 18: (963.56, 735.31), 20: (987.27, 736.97),
               22: (1000.81, 744.39), 24: (1015.7, 755.94), 26: (1030.78, 768.69)}),
        (1.64, {16: (675.07, 519.26), 18: (686.88, 520.94), 20: (690.08, 521.27),
               22: (695.01, 523.08), 24: (701.06, 528.55), 26: (709.48, 538.22)}),
        (2.52, {16: (483.91, 358.92), 18: (482.91, 355.15), 20: (484.97, 354.38),
               22: (486.92, 356.14), 24: (490.24, 359.8), 26: (494.55, 365.21)}),
        (3.68, {16: (343, 252.21), 18: (341.53, 249.23), 20: (341.58, 248.03),
               22: (342.58, 248.15), 24: (344.23, 250.05), 26: (346.67, 253.8)}),
        (5.43, {16: (243.77, 178.68), 18: (241.84, 176.11), 20: (240.86, 174.99),
               22: (240.75, 175.07), 24: (241.31, 176.15), 26: (242.45, 178.08)}),
        (7.74, {16: (177.68, 129.78), 18: (175.94, 127.33), 20: (174.95, 126.33),
               22: (174.59, 126.49), 24: (174.78, 126.91), 26: (175.48, 128.28)}),
        (11.3, {16: (122.99, 89.01), 18: (121.79, 87.65), 20: (121.08, 86.91),
               22: (120.8, 86.85), 24: (120.9, 87.27), 26: (121.34, 88.37)}),
        (16.5, {16: (87.47, 62.36), 18: (86.41, 61.6), 20: (85.75, 60.86),
               22: (85.03, 60.96), 24: (85.01, 61.27), 26: (85.28, 62.12)}),
        (23.3, {16: (61.49, 43.57), 18: (60.7, 42.91), 20: (60.21, 42.44),
               22: (59.97, 42.42), 24: (60, 42.67), 26: (60.17, 43.22)}),
        (32.7, {16: (43.79, 31.55), 18: (43.24, 30.97), 20: (42.89, 30.66),
               22: (42.72, 30.63), 24: (42.7, 30.83), 26: (42.8, 30.71)}),
        (46, {16: (31.05, 22.37), 18: (30.65, 21.99), 20: (30.4, 21.76),
             22: (30.27, 21.74), 24: (30.25, 21.84), 26: (30.31, 22.09)}),
        (65.3, {16: (21.94, 15.8), 18: (21.62, 15.47), 20: (21.44, 15.3),
               22: (21.28, 15.22), 24: (21.26, 15.33), 26: (21.31, 15.49)}),
    ),
    "t2_array": (
        (0.0108, {16: (2669.83, 1705.84), 18: (2862.34, 1823.56), 20: (3025.17, 1928.33),
                 22: (3191.54, 2023.15), 24: (3365.05, 2139.79), 26: (3536.79, 2205.31)}),
        (0.0181, {16: (1874.96, 357.4), 18: (2073.94, 370.54), 20: (2185.54, 379.48),
                 22: (2302.78, 392.49), 24: (2424.62, 406.02), 26: (2546.04, 420.75)}),
        (0.0282, {16: (1657.68, 247.93), 18: (1783.21, 257.47), 20: (1901.28, 267.29),
                 22: (2009.05, 276.98), 24: (2119.34, 287.48), 26: (2232.44, 299.08)}),
        (0.0434, {16: (1341.07, 161.07), 18: (1466.78, 169.02), 20: (1549.98, 175.05),
                 22: (1626.53, 181.48), 24: (1715.06, 187.21), 26: (1801.61, 192.23)}),
        (0.0626, {16: (1307.88, 154.07), 18: (1417.53, 159.77), 20: (1510.76, 166.9),
                 22: (1589.88, 172.51), 24: (1687.39, 180.8), 26: (1774.24, 188.16)}),
        (0.0934, {16: (892.29, 81.86), 18: (972.48, 85.92), 20: (1026.43, 88.89),
                 22: (1083.42, 92.3), 24: (1141.5, 95.97), 26: (1202.4, 99.89)}),
        (0.1353, {16: (711.11, 59.45), 18: (757.34, 61.31), 20: (805.1, 63.42),
                 22: (848.64, 65.65), 24: (892.41, 67.82), 26: (937.12, 70.57)}),
        (0.193, {16: (531.6, 41.53), 18: (563.86, 42.63), 20: (599.96, 44.24),
                22: (634.59, 45.86), 24: (669.56, 47.66), 26: (703.51, 49.26)}),
        (0.2768, {16: (384.99, 28.04), 18: (408.05, 28.93), 20: (431.22, 29.88),
                 22: (454.82, 30.92), 24: (478.81, 32.06), 26: (503.3, 33.25)}),
        (0.4276, {16: (262.14, 18.22), 18: (277.38, 18.78), 20: (292.87, 19.4),
                 22: (308.92, 20.07), 24: (325.23, 20.78), 26: (341.74, 21.56)}),
        (0.5555, {16: (203.53, 13.94), 18: (214.8, 14.28), 20: (226.53, 14.74),
                 22: (238.33, 15.24), 24: (250.61, 15.8), 26: (263.09, 16.39)}),
        (0.7902, {16: (141.21, 9.88), 18: (149.28, 10.17), 20: (158.17, 10.52),
                 22: (166.64, 10.88), 24: (175.31, 11.25), 26: (184.07, 11.67)}),
        (1.1274, {16: (104.57, 6.83), 18: (110.49, 7.04), 20: (116.7, 7.27),
                 22: (123.22, 7.49), 24: (129.67, 7.76), 26: (136.34, 8.06)}),
        (1.5996, {16: (73.65, 4.79), 18: (77.93, 4.93), 20: (82.46, 5.1),
                 22: (87.08, 5.28), 24: (91.82, 5.48), 26: (96.54, 5.69)}),
    ),
}
