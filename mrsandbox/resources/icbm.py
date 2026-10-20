####################################################################################################
#                                              icbm.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Registry entries for the ICBM152 2009c template and the NIHPD pediatric templates,      #
#          one dict per source with the citation beside it, and the age band a child stands in.    #
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

#***********#
#   nihpd   #
#***********#
# Source: Fonov et al. 2011 (above), the NIHPD objective 1 templates of healthy children.
# Data:   https://www.templateflow.org, tpl-MNIPediatricAsym res-02 probseg, one template per
#         age band in years (the cohorts of TemplateFlow's description), in the MNI frame the
#         ICBM152 template has: a child's tissue at each place, not a child's head size.
# Notes:  the copyright notice above (Copyright (C) 1993-2019 Vladimir S. Fonov, Louis Collins,
#         McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University).
NIHPD_BANDS = {"4.5-18.5": 1, "4.5-8.5": 2, "7-11": 3, "7.5-13.5": 4, "10-14": 5, "13-18.5": 6}
NIHPD_MD5 = {
    1: ("23c13a965c7a960ede6bfaf6b6220695", "e2bb7b55fc4d32a187ade6a995ac29f1",
        "ef11944c7efa70fc500d22c7f839879a"),
    2: ("03668e9f3e0c186d5bae71ae51c5fd02", "41839f41344605cc0b6a5a698050d330",
        "c9d3e49a39c2112efbc3abf3dadf3a3d"),
    3: ("3f9f09d25654eed3707b5769c4ef83e9", "fc4183095da0a950a5ce574e2fbff3bc",
        "ca19374fa2267a81d3fffa471ea5cf37"),
    4: ("1e85e60e4fbe733f23b941813a54bbcc", "1c87d54ad00dd7928ba96f99d08db13c",
        "d4711cbaf9db4dafb7b86b01842bc4a6"),
    5: ("28f56beae09ec1261c9515cbbcc0215c", "f02a3365e60d47946f1fa3f73e38a4b4",
        "40fbc4744fcfd56c3139c96c13194746"),
    6: ("27e80dc1555b55f85ed59090a0ead729", "3edfd3612192752ba5716a3aee181bc8",
        "2366070e5a63acb053f96da0d9a686dd"),
}
NIHPD = {
    f"nihpd_{band}/{tissue}_2mm": {
        "url": f"https://templateflow.s3.amazonaws.com/tpl-MNIPediatricAsym/cohort-{cohort}/"
               f"tpl-MNIPediatricAsym_cohort-{cohort}_res-2_label-{tissue.upper()}_probseg.nii.gz",
        "md5": md5,
        "filename": f"nihpd_{band}/{tissue}_2mm.nii.gz",
    }
    for band, cohort in NIHPD_BANDS.items()
    for tissue, md5 in zip(("gm", "wm", "csf"), NIHPD_MD5[cohort])
}


#**************#
#   the band   #
#**************#
def nihpd_band(age_y):
    """The pediatric template's band of a person `age_y` years old: the one centred nearest the age
    among those holding it (the youngest band under 4.5 years); None past 18.5 or with no age,
    an adult."""
    if age_y is None or age_y > 18.5:
        return None
    bands = {band: tuple(float(v) for v in band.split("-")) for band in NIHPD_BANDS}
    held = [band for band, (low, high) in bands.items() if low <= max(age_y, 4.5) <= high]
    return min(held, key=lambda band: (abs(sum(bands[band]) / 2.0 - age_y),
                                       bands[band][1] - bands[band][0]))
