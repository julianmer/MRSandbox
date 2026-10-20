####################################################################################################
#                                            bigbrain.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Registry entries for the BigBrain-MR digital phantom, one dict for this source with     #
#          the citation beside it (the database convention applied to remote data).                #
#                                                                                                  #
####################################################################################################

#*****************#
#   bigbrain mr   #
#*****************#
# Source: Sainz Martinez, C., Bach Cuadra, M., Jorge, J. BigBrain-MR: a new digital
#         phantom with anatomically-realistic magnetic resonance properties at 100-um
#         resolution for magnetic resonance methods development. NeuroImage 273 (2023)
#         120074.
# Data:   https://doi.org/10.5281/zenodo.7432527, CC BY-NC-SA 4.0; checksums are Zenodo's.
# Units:  per entry where scaled: value in units = raw x scale, typed from data_info.txt.
BIGBRAIN_MR = {
    "bigbrain_mr/labels_800um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_Labels_800um.nii.gz/content",
        "md5": "9a777d9ceeb5eeb7a13532a7af08b9fa",
        "filename": "bigbrain_mr/BigBrainMR_Labels_800um.nii.gz",
    },
    "bigbrain_mr/t1map_800um": {
        "url": "https://zenodo.org/api/records/7432527/files/BigBrainMR_T1map_800um.nii.gz/content",
        "md5": "faa12e5f08ed87eb3012705f028f3335",
        "filename": "bigbrain_mr/BigBrainMR_T1map_800um.nii.gz",
        "units": "s", "scale": 1e-3,  # stored in ms, 7 T MP2RAGE
    },
    "bigbrain_mr/r2smap_800um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_T2smodel_R2smap_800um.nii.gz/content",
        "md5": "c49cc28d5278e303041dba8a55ad863d",
        "filename": "bigbrain_mr/BigBrainMR_T2smodel_R2smap_800um.nii.gz",
        "units": "1/s", "scale": 1e-2,  # stored as 1/s x 100, 7 T ME-GRE
    },
    "bigbrain_mr/qsm_800um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_Phase_QSM_800um.nii.gz/content",
        "md5": "7eef45c0702c39953731a1b48d5d60ea",
        "filename": "bigbrain_mr/BigBrainMR_Phase_QSM_800um.nii.gz",
        "units": "ppm", "scale": 1e-4,  # stored as ppm x 1e4, STAR-QSM at 7 T
    },
    "bigbrain_mr/background_field_400um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_Phase_Background_400um.nii.gz/content",
        "md5": "be84f174317993ee15acba8c1224d950",
        "filename": "bigbrain_mr/BigBrainMR_Phase_Background_400um.nii.gz",
        "units": "Hz", "scale": 0.1,  # stored as Hz x 10, V-SHARP at 7 T; about 24 MB
    },
    "bigbrain_mr/bias_field_400um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_BiasField_400um.nii.gz/content",
        "md5": "f0bcd3f8fb1e886995ca34135316a49d",
        "filename": "bigbrain_mr/BigBrainMR_BiasField_400um.nii.gz",
        "units": "arbitrary", "scale": 1.0,  # 7 T receive bias magnitude; about 49 MB
    },
    # measured 32-channel receive sensitivities (Nova head coil at 7 T, ESPIRiT); the pair
    # is large (about 6.4 GB each), which is exactly what fetch-on-demand is for
    "bigbrain_mr/coil_sens_mag_400um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_CoilSensMag_400um.nii.gz/content",
        "md5": "8ea2adf73182317ce648ef24610cab57",
        "filename": "bigbrain_mr/BigBrainMR_CoilSensMag_400um.nii.gz",
        "units": "arbitrary", "scale": 1.0,
    },
    "bigbrain_mr/coil_sens_pha_400um": {
        "url": "https://zenodo.org/api/records/7432527/files/"
               "BigBrainMR_CoilSensPha_400um.nii.gz/content",
        "md5": "e0caf38161a62b8090eeac7d08c94cbf",
        "filename": "bigbrain_mr/BigBrainMR_CoilSensPha_400um.nii.gz",
        "units": "rad", "scale": 1.0,
    },
}
