####################################################################################################
#                                           datasets.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: The package's cohorts by the dataset each is read off: where its local copy lies under  #
#          one data folder, how the files its recipe reads are fetched (or where to get them by    #
#          hand), and the recipe that builds the cohort; one call fetches what is missing and      #
#          rebuilds them all.                                                                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import concurrent.futures
import io
import json
import pathlib
import re
import shutil
import urllib.parse
import urllib.request
import zipfile

from mrsandbox.resources.fetch import Resources, _read
from mrsandbox.resources.functional import RestingState7T
from mrsandbox.resources.hmri import HMRI
from mrsandbox.resources.relaxometry import IXI, Developmental, RadialMESE
from mrsandbox.resources.spectroscopy import (COWS, BigGABA, FMRSPain, Ketamine, NeuroMET,
                                              TissueSegmentation)
from mrsandbox.resources.uk7t import UK7T
from mrsandbox.scan import cohort as cohorts

OPENNEURO = "https://s3.amazonaws.com/openneuro.org"
XNAT = "https://www.nitrc.org/ir"


#*************#
#   fetched   #
#*************#
def _openneuro(accession, pattern):
    """A fetcher of the files of OpenNeuro `accession` whose paths within the dataset match
    `pattern` (a regular expression), off the public bucket into the local copy, laid out as
    the dataset is; a file there at its size is kept, one stored whole checked by its md5."""
    def fetch(root):
        token, found = None, []
        while True:
            listed = _read(f"{OPENNEURO}?list-type=2&prefix={accession}/" + (
                f"&continuation-token={urllib.parse.quote(token)}" if token else "")).decode()
            for item in re.findall(r"<Contents>(.*?)</Contents>", listed, re.S):
                key = re.search(r"<Key>([^<]+)</Key>", item).group(1)
                tag = re.search(r"<ETag>&quot;([^&]+)&quot;</ETag>", item)
                if re.search(pattern, key[len(accession) + 1:]):
                    found.append((key, int(re.search(r"<Size>(\d+)</Size>", item).group(1)),
                                  # a multipart upload's tag is no md5
                                  tag.group(1) if tag and "-" not in tag.group(1) else None))
            token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", listed)
            if token is None:
                break
            token = token.group(1)
        for key, size, md5 in found:
            path = root / key[len(accession) + 1:]
            if not (path.exists() and path.stat().st_size == size):
                Resources._fetch(f"{OPENNEURO}/{urllib.parse.quote(key)}", path, md5)
    return fetch


def _zipped(url, members):
    """A fetcher of `members` of a zip archive at `url` into the local copy at their paths in
    the archive, read by range requests (zipfile checks each member's CRC as it reads it)."""
    def fetch(root):
        with Resources._archive(url) as archive:
            for member in members:
                path = root / member
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".partial")
                with archive.open(member) as source, open(partial, "wb") as target:
                    shutil.copyfileobj(source, target, 1 << 22)
                partial.replace(path)
    return fetch


def _ixi(root):
    """IXI off NITRC's XNAT mirror (project ixi), a folder per subject: its T1, PD and T2 NIfTI
    and subject.json, what the session states: age, gender, field, each scan's parameters (TE,
    TR, flip). A folder that holds its subject.json is kept."""
    def one(row):
        tree = json.loads(_read(f"{XNAT}/data/subjects/{row['ID']}/experiments?format=json"))
        info = {}

        def walk(item):
            fields = item.get("data_fields", {})
            kind = item.get("meta", {}).get("xsi:type")
            if kind == "xnat:mrSessionData":
                info.update(exp=fields["ID"], label=fields["label"],
                            **({"age": fields["age"]} if "age" in fields else {}))
            elif kind == "xnat:demographicData":
                info["gender"] = fields.get("gender")
            for child in item.get("children", []):
                for grandchild in child["items"]:
                    walk(grandchild)
        for item in tree["items"]:
            walk(item)
        if "exp" not in info or (root / info["label"] / "subject.json").exists():
            return
        folder = root / info["label"]
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(_read(
                f"{XNAT}/data/experiments/{info['exp']}/scans/T1,T2,PD/resources/NIfTI/files"
                "?format=zip"))) as archive:
            for name in archive.namelist():
                if name.endswith(".nii.gz"):
                    (folder / pathlib.Path(name).name).write_bytes(archive.read(name))
        session = json.loads(_read(f"{XNAT}/data/experiments/{info['exp']}?format=json"))
        for item in session["items"]:
            info["field"] = item["data_fields"].get("fieldStrength")
            info["scans"] = {
                scan["data_fields"]["ID"]: {key.split("/")[1]: value
                                            for key, value in scan["data_fields"].items()
                                            if key.startswith("parameters/")}
                for child in item.get("children", []) for scan in child["items"]
                if scan.get("meta", {}).get("xsi:type") == "xnat:mrScanData"
                and "ID" in scan.get("data_fields", {})}
        (folder / "subject.json").write_text(json.dumps(info))

    rows = json.loads(_read(f"{XNAT}/data/projects/ixi/subjects?format=json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, rows["ResultSet"]["Result"]))


#**************#
#   datasets   #
#**************#
# Data: per package cohort, its dataset's local copy under the data folder (None where the
#       package fetches the data itself, resources.Resources), where the dataset comes from, the
#       fetcher of the files the recipe reads (None where the host wants a login: then fetched by
#       hand, and the source says where to and how the recipe reads them), and the recipe of
#       (the local copy, the data folder), which keeps its work (conversions, segmentations)
#       under the data folder and reuses it.
BIG_GABA = ("log in at https://www.nitrc.org/frs/?group_id=1312, download the PRESS sites' zips "
            "(G*_P, P*_P, S*_P) and demographics.csv, and unpack them as the recipe reads them: "
            "PRESS/<site>_P/S<k>/, demographics.csv")
DATASETS = {
    "big_gaba_trio": ("BigGABA", BIG_GABA, None,
                      lambda root, data: BigGABA.cohort(root, "trio")),
    "big_gaba_ge": ("BigGABA", BIG_GABA, None,
                    lambda root, data: BigGABA.cohort(root, "ge", root / "PRESS_ge_nifti")),
    "big_gaba_philips": ("BigGABA", BIG_GABA, None,
                         lambda root, data: BigGABA.cohort(root, "philips")),
    "cows_prisma": ("openneuro_ds006812", COWS.CITED[1], _openneuro(
                        "ds006812", r"^sub-[^/]+/mrs/sourcedata/[^/]+_metab_[^/]+\.dat$"
                                    r"|^participants\.(tsv|json)$"),
                    lambda root, data: COWS.cohort(root)),
    "fmrs_pain_achieva": ("fMRSinPain", "log in at https://www.nitrc.org/frs/?group_id=1516, "
                                        "download the raw data and unpack it as the recipe "
                                        "reads it: SUBJECTS/RAW/GLUPI_S<k>/*.list and *.data",
                          None, lambda root, data: FMRSPain.cohort(root)),
    "ds006444_ge_mr750": ("openneuro_ds006444", TissueSegmentation.CITED[1], _openneuro(
                              "ds006444", r"^sub-[^/]+/ses-[^/]+/mrs/[^/]+_svs\.nii\.gz$"
                                          r"|^participants\.(tsv|json)$"),
                          lambda root, data: TissueSegmentation.cohort(root)),
    "ds005917_siemens_7t": ("openneuro_ds005917", Ketamine.CITED[1], _openneuro(
                                "ds005917", r"^sub-[^/]+/ses-b0/mrs/[^/]+_svs\.nii\.gz$"
                                            r"|^participants\.(tsv|json)$"),
                            lambda root, data: Ketamine.cohort(root)),
    "neuromet_siemens_7t": ("neuromet", NeuroMET.CITED[1], _zipped(
                                "https://zenodo.org/records/5500320/files/"
                                "RawData-mrm29034_20211012_Riemann-Fillmer.zip?download=1",
                                [f"p{k}_T1_1_HS.dat" for k in range(1, 10)]),
                            lambda root, data: NeuroMET.cohort(root)),
    "ds001168_siemens_7t": ("openneuro_ds001168", RestingState7T.CITED["url"], _openneuro(
                                "ds001168", r"^sub-[^/]+/ses-1/func/[^/]+_acq-prefrontal_"
                                            r"(physio\.tsv\.gz|bold\.nii\.gz)$"
                                            r"|^task-rest_acq-prefrontal_(bold|physio)\.json$"
                                            r"|^participants\.tsv$"),
                            lambda root, data: RestingState7T.cohort(root)),
    "hmri_demo_prisma_3t": (None, "https://github.com/hMRI-group/hMRI-toolbox-demodata", None,
                            lambda root, data: HMRI.cohort(data / "maps" / "hmri_demo_prisma_3t")),
    "uk7t_oxford_7t": (None, "https://ora.ox.ac.uk/objects/"
                             "uuid:55ca977f-62df-4cbf-b300-2dc893e36647", None,
                       lambda root, data: UK7T.cohort(data / "maps" / "uk7t_oxford_7t")),
    "ds004611_prisma_3t": ("openneuro_ds004611", Developmental.URL, _openneuro(
                               "ds004611", r"^sub-[^/]+/ses-[^/]+/anat/[^/]+"
                                           r"(UNIT1|inv-2_MP2RAGE|T1map|T2map)\.(nii\.gz|json)$"
                                           r"|^sub-[^/]+/[^/]+_sessions\.tsv$"
                                           r"|^participants\.(tsv|json)$"),
                           lambda root, data: Developmental.cohort(root,
                                                                   data / "maps" / "ds004611")),
    "ixi_guys_1p5t": ("ixi", IXI.URL, _ixi,
                      lambda root, data: IXI.cohort(root, data / "maps" / "ixi_t1", "Guys")),
    "ixi_hh_3t": ("ixi", IXI.URL, _ixi,
                  lambda root, data: IXI.cohort(root, data / "maps" / "ixi_t1", "HH")),
    "radial_mese_prisma_3t": ("radial_mese", RadialMESE.URL, _zipped(
                                  "https://zenodo.org/records/10908743/files/"
                                  "Reconstruction.zip?download=1",
                                  [f"{RadialMESE.FOLDER}/{name}"
                                   for name in ("Echoes4D.nii", "Anima_T2map.nii")]),
                              lambda root, data: RadialMESE.cohort(
                                  root, data / "maps" / "radial_mese_prisma_3t")),
}


#*************#
#   rebuild   #
#*************#
def rebuild(data, names=None, into=None, fetch=True):
    """Every package cohort (those named in `names`) rebuilt from the datasets' local copies
    under `data`, each saved into `into` (the package's cohorts unless given), and the pool
    forgotten so this process draws on the new rows. A copy that is missing is fetched first
    (with `fetch`) where its host serves it openly; one whose host wants a login stops the call
    before anything is fetched or built, saying where to get it and how to lay it out. Returns
    the saved paths."""
    data = pathlib.Path(data)
    into = pathlib.Path(into) if into is not None else cohorts.COHORTS
    names = list(DATASETS) if names is None else list(names)
    absent = [name for name in names
              if DATASETS[name][0] is not None and not (data / DATASETS[name][0]).is_dir()]
    by_hand = [name for name in absent if not fetch or DATASETS[name][2] is None]
    if by_hand:
        raise FileNotFoundError("no local copy of " + "; ".join(
            f"{name} ({data / DATASETS[name][0]}: {DATASETS[name][1]})" for name in by_hand))
    # once per folder: the IXI sites share one
    for folder, fetcher in {DATASETS[name][0]: DATASETS[name][2] for name in absent}.items():
        fetcher(data / folder)
    into.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in names:
        folder, _, _, recipe = DATASETS[name]
        recipe(None if folder is None else data / folder, data).save(into / f"{name}.json")
        saved.append(into / f"{name}.json")
    cohorts.package.cache_clear()
    cohorts.forget()
    return saved
