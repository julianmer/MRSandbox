####################################################################################################
#                                           cohorts.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: The package's measured cohorts (scan.cohort), one JSON of readings per dataset, as      #
#          registered resources: a source checkout reads them from the repository's data/cohorts,  #
#          an installed package fetches them from there on first use. The wheel ships no data.     #
#                                                                                                  #
####################################################################################################

_ROOT = "https://raw.githubusercontent.com/julianmer/MRSandbox/main/data/cohorts/"
NAMES = (
    "big_gaba_ge",
    "big_gaba_philips",
    "big_gaba_trio",
    "cows_prisma",
    "ds001168_siemens_7t",
    "ds004611_prisma_3t",
    "ds005917_siemens_7t",
    "ds006444_ge_mr750",
    "fmrs_pain_achieva",
    "hmri_demo_prisma_3t",
    "ixi_guys_1p5t",
    "ixi_hh_3t",
    "neuromet_siemens_7t",
    "radial_mese_prisma_3t",
    "uk7t_oxford_7t",
)
# no checksum: the cohorts are re-read as the readers improve, and a JSON that did not arrive
# whole fails to parse
COHORTS = {
    f"cohorts/{name}": {
        "url": _ROOT + name + ".json",
        "md5": None,
        "filename": f"cohorts/{name}.json",
        "repo": f"data/cohorts/{name}.json",
    }
    for name in NAMES
}
