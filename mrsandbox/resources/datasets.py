####################################################################################################
#                                           datasets.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: The package's cohorts by the dataset each is read off: where its local copy lies under  #
#          one data folder, where the dataset comes from, and the recipe that builds the cohort;   #
#          one call rebuilds them all.                                                             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pathlib

from mrsandbox.resources.functional import RestingState7T
from mrsandbox.resources.hmri import HMRI
from mrsandbox.resources.relaxometry import IXI, Developmental, RadialMESE
from mrsandbox.resources.spectroscopy import (COWS, BigGABA, FMRSPain, Ketamine, NeuroMET,
                                              TissueSegmentation)
from mrsandbox.resources.uk7t import UK7T
from mrsandbox.scan import cohort as cohorts

#**************#
#   datasets   #
#**************#
# Data: per package cohort, its dataset's local copy under the data folder (None where the
#       package fetches the data itself, resources.Resources), where the dataset comes from, and
#       the recipe of (the local copy, the data folder), which keeps its work (conversions,
#       segmentations) under the data folder and reuses it.
DATASETS = {
    "big_gaba_trio": ("BigGABA", BigGABA.CITED[1],
                      lambda root, data: BigGABA.cohort(root, "trio")),
    "big_gaba_ge": ("BigGABA", BigGABA.CITED[1],
                    lambda root, data: BigGABA.cohort(root, "ge", root / "PRESS_ge_nifti")),
    "big_gaba_philips": ("BigGABA", BigGABA.CITED[1],
                         lambda root, data: BigGABA.cohort(root, "philips")),
    "cows_prisma": ("openneuro_ds006812", COWS.CITED[1], lambda root, data: COWS.cohort(root)),
    "fmrs_pain_achieva": ("fMRSinPain", FMRSPain.CITED[1],
                          lambda root, data: FMRSPain.cohort(root)),
    "ds006444_ge_mr750": ("openneuro_ds006444", TissueSegmentation.CITED[1],
                          lambda root, data: TissueSegmentation.cohort(root)),
    "ds005917_siemens_7t": ("openneuro_ds005917", Ketamine.CITED[1],
                            lambda root, data: Ketamine.cohort(root)),
    "neuromet_siemens_7t": ("neuromet", NeuroMET.CITED[1],
                            lambda root, data: NeuroMET.cohort(root)),
    "ds001168_siemens_7t": ("openneuro_ds001168", RestingState7T.CITED["url"],
                            lambda root, data: RestingState7T.cohort(root)),
    "hmri_demo_prisma_3t": (None, "https://github.com/hMRI-group/hMRI-toolbox-demodata",
                            lambda root, data: HMRI.cohort(data / "maps" / "hmri_demo_prisma_3t")),
    "uk7t_oxford_7t": (None, "https://ora.ox.ac.uk/objects/"
                             "uuid:55ca977f-62df-4cbf-b300-2dc893e36647",
                       lambda root, data: UK7T.cohort(data / "maps" / "uk7t_oxford_7t")),
    "ds004611_prisma_3t": ("openneuro_ds004611", Developmental.URL,
                           lambda root, data: Developmental.cohort(root,
                                                                   data / "maps" / "ds004611")),
    "ixi_guys_1p5t": ("ixi", IXI.URL,
                      lambda root, data: IXI.cohort(root, data / "maps" / "ixi_t1", "Guys")),
    "ixi_hh_3t": ("ixi", IXI.URL,
                  lambda root, data: IXI.cohort(root, data / "maps" / "ixi_t1", "HH")),
    "radial_mese_prisma_3t": ("radial_mese", RadialMESE.URL,
                              lambda root, data: RadialMESE.cohort(
                                  root, data / "maps" / "radial_mese_prisma_3t")),
}


#*************#
#   rebuild   #
#*************#
def rebuild(data, names=None, into=None):
    """Every package cohort (those named in `names`) rebuilt from the datasets' local copies
    under `data`, each saved into `into` (the package's cohorts unless given), and the pool
    forgotten so this process draws on the new rows. Every copy asked for is looked for
    first: a missing one stops the call before anything is built, naming where the dataset
    comes from. Returns the saved paths."""
    data = pathlib.Path(data)
    into = pathlib.Path(into) if into is not None else cohorts.COHORTS
    names = list(DATASETS) if names is None else list(names)
    missing = [f"{name} ({data / DATASETS[name][0]}, from {DATASETS[name][1]})" for name in names
               if DATASETS[name][0] is not None and not (data / DATASETS[name][0]).is_dir()]
    if missing:
        raise FileNotFoundError("no local copy of " + "; ".join(missing))
    into.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in names:
        folder, _, recipe = DATASETS[name]
        recipe(None if folder is None else data / folder, data).save(into / f"{name}.json")
        saved.append(into / f"{name}.json")
    cohorts.package.cache_clear()
    cohorts.forget()
    return saved
