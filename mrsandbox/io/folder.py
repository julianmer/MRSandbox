####################################################################################################
#                                             folder.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: A folder of scans as a scanner gives them, whatever the vendor and the format, told     #
#          apart by what they are and brought to what the package reads: raw spectroscopy through  #
#          spec2nii to NIfTI-MRS, DICOM images through dcm2niix, the rest by name.                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import json
import pathlib
import shutil
import subprocess
import sys

#***************#
#   the kinds   #
#***************#
# raw spectroscopy formats and the spec2nii subcommand that converts each
# (spec2nii: Clarke et al. 2022)
RAW = {".dat": "twix", ".7": "ge", ".sdat": "philips", ".list": "philips_dl", ".rda": "rda"}
# what a file's name says of an image: the words looked for, lowercase, in order
IMAGES = (("t1map", "T1 map"), ("r1map", "R1 map"),
          # star maps first: R2* is not R2 (a gradient echo's decay carries the field's
          # inhomogeneity), so neither feeds the water's T2
          ("t2starmap", "other image"), ("r2starmap", "other image"),
          ("t2map", "T2 map"), ("r2map", "R2 map"), ("t1w", "T1w"), ("mprage", "T1w"),
          ("t1_", "T1w"), ("_t1", "T1w"), ("fieldmap", "B0"), ("phasediff", "B0"), ("b0map", "B0"),
          ("_b0", "B0"), ("tb1", "B1+"), ("b1map", "B1+"), ("b1plus", "B1+"), ("flipangle", "B1+"),
          ("csm", "coils"), ("sens", "coils"), ("coilmap", "coils"),
          ("t2w", "T2w"), ("flair", "other image"))
# what each kind gives the room
GIVES = {"MRS run": "the noise level, the line width; with its reference the residual water, the "
                    "receiver delay, the eddy transient, the leaked water",
         "MRS reference": "with its run: the residual water, the receiver delay, the eddy "
                          "transient, the leaked water",
         "T1w": "the anatomy: tissues and the lipid layer (segmented with FSL)",
         "T1 map": "the water's T1 per tissue, read on the anatomy's tissues",
         "R1 map": "the water's T1 per tissue (1 / R1), read on the anatomy's tissues",
         "T2 map": "the water's T2 per tissue, read on the anatomy's tissues",
         "R2 map": "the water's T2 per tissue (1 / R2), read on the anatomy's tissues",
         "B0": "the field map, shimmed per voxel", "B1+": "the transmit field the pulses see",
         "coils": "the receive array's sensitivity maps",
         "noise scan": "the channels' covariance and own noise share",
         "Pulseq": "the sequence as blocks",
         "scanner file": "a saved scanner with its records", "T2w": "nothing yet",
         "other image": "nothing yet", "DICOM series": "images by their series",
         "physio": "the subject's breathing and heart rates over the run",
         "image series": "the head's motion over the run: events a minute and their size",
         "unknown": "nothing: not a kind the package knows"}


# which kinds of file give which leaf of the parameter tree: what "yours" on a line can be read from
READS = {"noise.sigma": ("MRS run",), "lineshape.gauss_hz": ("MRS run",),
         "coils.loaded.own_noise": ("MRS run", "noise scan"),
         "water.adjusted_residual": ("MRS run", "MRS reference"),
         "acquisition.delay_s": ("MRS run", "MRS reference"),
         "acquisition.eddy_transient_rad": ("MRS run", "MRS reference"),
         "oov.fraction": ("MRS run", "MRS reference"),
         "run.resp_freq_hz": ("physio",), "run.cardiac_hz": ("physio",),
         "run.events_per_min": ("image series",), "run.motion_mm": ("image series",),
         "contrast": ("T1w",), "lipids": ("T1w",), "water.t1_s": ("T1 map", "R1 map"),
         "water.t2_s": ("T2 map", "R2 map")}


def reads(leaf):
    """The kinds of file a leaf can be read from, by its path or its group; empty where no reader
    exists yet."""
    return READS.get(leaf) or READS.get(leaf.split(".")[0]) or ()


#**************************************************************************************************#
#                                           Class Found                                            #
#**************************************************************************************************#
#                                                                                                  #
# One file of the folder: what it is, and what the package can take from it.                       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass
class Found:
    """One file of the folder: what it is, and what the package can take from it.

    Args:
        path: The file.
        kind: One of GIVES' keys.
        gives: What the room takes from it.
        converted: The NIfTI-MRS a raw spectroscopy file was brought to, or None.
        note: How it was recognized, or why it could not be used.
    """
    path: pathlib.Path
    kind: str
    gives: str = ""
    converted: pathlib.Path = None
    note: str = ""


#************#
#   survey   #
#************#
def survey(folder):
    """Every file under `folder` told apart by what it is: NIfTI-MRS runs and references (the
    header's dimensions and its water suppression, or the name's "ref" or "water"), raw
    spectroscopy by its extension (Siemens twix .dat and .rda, GE .7, Philips .SDAT and .list,
    Bruker fid beside a method file), images by their name (T1w, MPRAGE, field maps, B1 maps, coil
    maps), DICOM files by their extension, ISMRMRD noise scans (.mrd, .h5), Pulseq files (.seq),
    saved scanners (.json with a scanner and records). Returns [Found] in the folder's order."""
    folder = pathlib.Path(folder)
    found = []
    for path in ([folder] if folder.is_file()
                 else sorted(p for p in folder.rglob("*")
                             if p.is_file() and not p.name.startswith("."))):
        name = path.name.lower()
        suffix = ("".join(path.suffixes[-2:]).lower() if name.endswith(".nii.gz")
                  else path.suffix.lower())
        if suffix in (".nii", ".nii.gz"):
            found.append(_nifti(path))
        elif suffix in RAW:
            if suffix == ".sdat" and not any(p.suffix.lower() == ".spar"
                                             for p in path.parent.glob(path.stem + ".*")):
                found.append(Found(path, "unknown", note="an SDAT without its SPAR"))
            else:
                found.append(Found(path, "MRS reference" if _says_reference(name) else "MRS run",
                                   note=f"raw spectroscopy, {RAW[suffix]} through spec2nii"))
        elif name == "fid" and (path.parent / "method").exists():
            found.append(Found(path, "MRS run", note="Bruker, through spec2nii"))
        elif suffix in (".dcm", ".ima") or name == "dicomdir":
            found.append(Found(path, "DICOM series", note="DICOM: images through dcm2niix, "
                                                          "spectroscopy through spec2nii"))
        elif suffix in (".mrd", ".h5"):
            found.append(Found(path, "noise scan", note="ISMRMRD"))
        elif name.endswith("_physio.tsv.gz") or name.endswith("_physio.tsv"):
            found.append(Found(path, "physio", note="a BIDS physiological recording"))
        elif name.endswith("_bold.nii.gz") or name.endswith("_bold.nii"):
            found.append(Found(path, "image series", note="a BIDS functional series"))
        elif suffix == ".seq":
            found.append(Found(path, "Pulseq", note="a Pulseq sequence file"))
        elif suffix == ".json":
            if any((path.parent / (path.stem + ending)).exists()
                   for ending in (".nii", ".nii.gz", ".tsv.gz", ".tsv")):
                continue  # an image's or a recording's sidecar: read with it
            try:
                stored = json.loads(path.read_text())
                found.append(Found(path, "scanner file" if {"scanner", "records"} <= set(stored)
                                   else "unknown", note="JSON"))
            except (ValueError, UnicodeDecodeError):
                found.append(Found(path, "unknown", note="JSON the package cannot read"))
        elif suffix in (".spar", ".method", ".acqp", ".txt", ".md", ".tsv", ".csv", ".log", ".png",
                        ".pdf"):
            continue  # sidecars and notes, taken with what they belong to or not at all
        else:
            found.append(Found(path, "unknown",
                               note=f"{suffix or 'no extension'}: not a kind the package knows"))
    for item in found:
        item.gives = GIVES[item.kind]
    return found


def _says_reference(name):
    return any(word in name
               for word in ("mrsref", "_ref", "-ref", "water", "h2o", "unsuppressed", "wref"))


def _nifti(path):
    """A NIfTI file: NIfTI-MRS by its extension header, a functional series by its name, else an
    image by its name."""
    if path.name.lower().endswith(("_bold.nii.gz", "_bold.nii")):
        return Found(path, "image series", note="a BIDS functional series")
    try:
        import nibabel as nib

        image = nib.load(str(path))
        extensions = list(image.header.extensions)
        if extensions and extensions[0].get_code() == 44:  # the NIfTI-MRS extension
            meta = json.loads(extensions[0].get_content())
            tags = [meta.get(f"dim_{k}") for k in (5, 6, 7)]
            suppressed = meta.get("WaterSuppressed")
            reference = ((suppressed is False)
                         or (suppressed is None and _says_reference(path.name.lower())))
            note = "NIfTI-MRS" + (" with the transients apart" if "DIM_DYN" in tags
                                  else ", the transients averaged: "
                                       "the noise cannot be read per transient")
            return Found(path, "MRS reference" if reference else "MRS run", note=note)
    except Exception as error:                                                # noqa: BLE001
        return Found(path, "unknown", note=f"NIfTI the package cannot read: {error}"[:120])
    name = path.name.lower()
    for word, kind in IMAGES:
        if word in name:
            return Found(path, kind, note=f"an image, by its name ({word})")
    return Found(path, "other image", note="an image whose name says nothing the package knows")


#*************#
#   convert   #
#*************#
def convert(item, work):
    """Bring a raw spectroscopy file to NIfTI-MRS under `work` with spec2nii (auto), and note the
    result on the Found; the NIfTI-MRS path or None."""
    tool = shutil.which("spec2nii") or str(pathlib.Path(sys.executable).parent / "spec2nii")
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stem = item.path.stem if item.path.name != "fid" else item.path.parent.name
    done = subprocess.run([tool, "auto", "-o", str(work), "-f", stem,
                           str(item.path if item.path.name != "fid" else item.path.parent)],
                          capture_output=True, text=True)
    made = sorted(work.glob(stem + "*.nii.gz"))
    if done.returncode != 0 or not made:
        item.note = f"spec2nii could not convert it: {(done.stderr or done.stdout).strip()[-160:]}"
        return None
    item.converted = made[0]
    if len(made) > 1:
        item.note += f"; spec2nii wrote {len(made)} files, the first taken"
    return made[0]


#***********#
#   pairs   #
#***********#
def pairs(found):
    """The MRS runs with their water references, [(run, reference or None)]: a reference belongs to
    the run whose name it shares up to the part that says it is one, else the run of its folder."""
    runs = [f for f in found if f.kind == "MRS run"]
    references = [f for f in found if f.kind == "MRS reference"]

    def stem(item):
        name = item.path.name.lower()
        for word in ("_mrsref", "_ref", "-ref", "_water", "_h2o", "_unsuppressed", "_wref", "_svs",
                     "_press", "_slaser", "_metab"):
            name = name.split(word)[0]
        return name

    paired = []
    for run in runs:
        same = ([r for r in references if stem(r) == stem(run)]
                or [r for r in references if r.path.parent == run.path.parent])
        paired.append((run, same[0] if same else None))
    return paired
