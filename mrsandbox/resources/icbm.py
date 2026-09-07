####################################################################################################
#                                             icbm.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Registry entries for the ICBM152 2009c template, one dict for this source with the      #
#          citation beside it.                                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   icbm152   #
#*************#
# Source: Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C.,
#         Collins, D. L. Unbiased average age-appropriate atlases for pediatric studies.
#         NeuroImage 54 (2011) 313-327.
# Data:   https://www.templateflow.org, tpl-MNI152NLin2009cAsym res-02 probseg.
# Notes:  Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre,
#         Montreal Neurological Institute, McGill University; free to use, copy, modify,
#         and distribute provided the copyright notice appears.
ICBM152 = {
    "icbm152/gm_2mm": {
        "url": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
               "tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz",
        "md5": "9e38c9dbb352625abc89e23449114705",
        "filename": "icbm152/gm_2mm.nii.gz",
    },
    "icbm152/wm_2mm": {
        "url": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
               "tpl-MNI152NLin2009cAsym_res-02_label-WM_probseg.nii.gz",
        "md5": "c4124028b3829bd059722b7aa02c02d2",
        "filename": "icbm152/wm_2mm.nii.gz",
    },
    "icbm152/csf_2mm": {
        "url": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
               "tpl-MNI152NLin2009cAsym_res-02_label-CSF_probseg.nii.gz",
        "md5": "7a36780a3efdb8f73fcb3c96ee101f9c",
        "filename": "icbm152/csf_2mm.nii.gz",
    },
}
