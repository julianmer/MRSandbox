####################################################################################################
#                                          gudmundson.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: The tables of the MRS meta-analysis database (Gudmundson et al. 2023, Anal Biochem      #
#          676:115227; github.com/agudmundson/mrs-database, MIT) as registered resources: a source  #
#          checkout reads them from the externals/mrs-database submodule, an installed package      #
#          fetches them from GitHub on first use (database.gudmundson).                            #
#                                                                                                  #
####################################################################################################

_ROOT = "https://raw.githubusercontent.com/agudmundson/mrs-database/main/"
TABLES = ("References_Concentrations.csv", "References_T2.csv",
          "Values_Concentrations.csv", "Values_T2.csv")
# no checksum: the database grows with the literature
GUDMUNDSON = {
    f"gudmundson/{name}": {
        "url": _ROOT + name,
        "md5": None,
        "filename": f"gudmundson/{name}",
        "repo": f"externals/mrs-database/{name}",
    }
    for name in TABLES
}
