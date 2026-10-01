####################################################################################################
#                                            colin27.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Registry entries for the Colin27 2008 fuzzy phantom, one dict for this source with the  #
#          citation beside it.                                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   colin27   #
#*************#
# Source: Aubert-Broche, B., Evans, A. C., Collins, L. A new improved version of the realistic
#         digital brain phantom. NeuroImage 32 (2006) 138-145: one subject (Colin27, Holmes et
#         al. 1998) as twelve fuzzy tissue classes at 0.5 mm in the MNI frame, each cell's
#         fractions summing to one; the classes part the scalp's fat from its muscle.
# Data:   https://nist.mni.mcgill.ca/colin-27-average-brain-2008/, the NIfTI archive, the members
#         below taken out of it by range requests (the archive itself is never downloaded).
# Notes:  Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal
#         Neurological Institute, McGill University; free to use, copy, modify, and distribute
#         provided the copyright notice appears.
COLIN27 = {
    f"colin27/{label}": {
        "url": "https://packages.bic.mni.mcgill.ca/mni-models/colin27/"
               "mni_colin27_2008_fuzzy_nifti.zip",
        "member": f"{member}.nii",
        "md5": md5,
        "filename": f"colin27/{member}.nii",
    }
    for label, member, md5 in (("gm", "GM", "8ec96314400d0f82e82084df3d007c57"),
                               ("wm", "WM", "a3dcff5c13f348a4288650de88323adc"),
                               ("csf", "CSF", "58fe0bf9479652926a0c0278c4aa5a9e"),
                               ("fat", "FAT", "8984b814032461d1039aca4ca8d2aa52"),
                               ("marrow", "MARROW", "0fc18afad9e4500386edb01563653d27"))
}
