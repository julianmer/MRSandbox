####################################################################################################
#                                            mrshub.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Registry entries for the MRSHub macromolecule consensus data collection, one dict for   #
#          this source with the citation beside it.                                                #
#                                                                                                  #
####################################################################################################

#******************#
#   mm consensus   #
#******************#
# Source: Cudalbu, C., et al. Contribution of macromolecules to brain 1H MR spectra:
#         experts' consensus recommendations. NMR in Biomedicine 34 (2021) e4393.
# Data:   https://github.com/mrshub/mm-consensus-data-collection, MIT licensed.
# Units:  Varian format, a binary fid plus its procpar text header.
# Notes:  measured metabolite-nulled spectra: multi-subject, frequency corrected, summed,
#         residual water and the 3.93 ppm Cr/PCr peak removed; the 3 T human spectrum is
#         STEAM TE 8 ms (Dinesh Deelchand, CMRR).
MM_CONSENSUS = {
    "mm_consensus/3t_human_steam_fid": {
        "url": "https://raw.githubusercontent.com/mrshub/mm-consensus-data-collection/master/"
               "MM_database/3T_MM_human_STEAM_CMRR.fid/fid",
        "md5": "244d36bb9dd2e5d7414c670cbbafd87a",
        "filename": "mm_consensus/3T_MM_human_STEAM_CMRR.fid/fid",
    },
    "mm_consensus/3t_human_steam_procpar": {
        "url": "https://raw.githubusercontent.com/mrshub/mm-consensus-data-collection/master/"
               "MM_database/3T_MM_human_STEAM_CMRR.fid/procpar",
        "md5": "adf08ed583121c73fd6a066528b5580b",
        "filename": "mm_consensus/3T_MM_human_STEAM_CMRR.fid/procpar",
    },
}
