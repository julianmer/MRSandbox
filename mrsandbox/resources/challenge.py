####################################################################################################
#                                           challenge.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The macromolecule basis of the 2024 MRSI challenge as named resources.                  #
#                                                                                                  #
####################################################################################################

#**********************#
#   mrsi challenge     #
#**********************#
# Source: LaMaster, J. T., Merkofer, J. P., van de Sande, D. M. J., Soher, B. J.,
#         Strasser, B., Ma, C. The 2024 MRSI Data Processing and Quantification
#         Challenge Synthetic Dataset. arXiv:2609.02637.
# Data:   the example dataset of the challenge (bsoher.github.io/FittingChallenge2024),
#         basis_macromol_3T, gzipped unchanged into this repository's data/ folder: a
#         source checkout serves the files from there, an installed package fetches them
#         from the repository.
# Units:  NIfTI-MRS, one complex FID of 32768 points at 80.389 us per component.
_ROOT = "https://raw.githubusercontent.com/julianmer/MRSandbox/main/data/mrsi_challenge/"
_MD5 = {
    "mrs_mm_0_9": "68463d6843cdd50e5fc68a7d956dd01e",
    "mrs_mm_1_21": "7b7b646ecf23e100b1c81913d17239f0",
    "mrs_mm_1_43": "ea39343a9258cf02fe9ba87458764c56",
    "mrs_mm_1_67": "630363314344e05bf033d7494eb59be9",
    "mrs_mm_2_04": "11a3b6cb214fe3a07c20e89b51952cfa",
    "mrs_mm_2_26": "75fb1c8821dc909963a7a47d50d45879",
    "mrs_mm_2_99": "166d6abdaa791b17c38b64f300546562",
    "mrs_mm_3_21": "76c8fd7e2e7944221668fc3c972e814d",
    "mrs_mm_3_77": "78d7dd5823d05d0ece751596e0d1f526",
}
MRSI_CHALLENGE = {
    f"mrsi_challenge/{stem}": {
        "url": _ROOT + stem + ".nii.gz",
        "md5": md5,
        "filename": f"mrsi_challenge/{stem}.nii.gz",
        "repo": f"data/mrsi_challenge/{stem}.nii.gz",
    }
    for stem, md5 in _MD5.items()
}
