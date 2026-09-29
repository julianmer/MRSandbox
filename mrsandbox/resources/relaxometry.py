####################################################################################################
#                                          relaxometry.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 29/09/26                                                                                #
#                                                                                                  #
# Purpose: The open relaxometry datasets the water's T1 and T2 are read off, each a cohort of its  #
#          subjects' own rows: a local copy of the dataset in, one Cohort out, every subject's     #
#          maps read per tissue on its own segmentation (scan.maps).                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json
import pathlib

import numpy as np

from mrsandbox.hardware import Scanner
from mrsandbox.resources import participants
from mrsandbox.scan import maps
from mrsandbox.scan.cohort import Cohort, Member
from mrsandbox.scan.maps import MapReading
from mrsandbox.scan.profile import ScannerProfile


#*************#
#   subject   #
#*************#
def _subject(name, scanner, records, traits):
    """A cohort member of one subject's rows and traits, placed as the map cohorts place
    theirs."""
    return Member(name, ScannerProfile(Scanner.preset(scanner), records), (0.0, -35.0, 20.0),
                  (20.0, 20.0, 20.0), "icbm152", traits)


#**************************************************************************************************#
#                                        Class Developmental                                       #
#**************************************************************************************************#
#                                                                                                  #
# The Lausanne developmental dataset: MP2RAGE T1 and GRAPPATINI T2 maps of children.               #
#                                                                                                  #
#**************************************************************************************************#
class Developmental:
    """The Lausanne developmental dataset: MP2RAGE T1 and GRAPPATINI T2 maps of children.

    Source: Romascano, D., et al. Normative volumes and relaxation times at 3T during brain
            development. Scientific Data 11 (2024) 429. doi:10.1038/s41597-024-03267-3.
    Data:   OpenNeuro ds004611 v1.0.2, CC0: 67 healthy children aged 3 to 18 (the paper:
            healthy subjects), one Siemens Prisma fit
            session each (64-channel head and neck coil), the vendor's MP2RAGE T1 map and, for
            59 of them, the GRAPPATINI T2 map (a model-based fit of an accelerated 16-echo spin
            echo train), 0.72 x 0.72 x 3 mm.
    Units:  both maps in ms as stored (the T2 sidecars say "ms x10", which neither the tissues'
            values nor the authors' own tables bear out).
    """

    DATASET = "Developmental relaxometry, Lausanne, OpenNeuro ds004611 (CC0)"
    CITATION = "Romascano et al. 2024, Scientific Data 11:429"
    URL = "https://openneuro.org/datasets/ds004611"

    #**************#
    #   cohort     #
    #**************#
    @classmethod
    def cohort(cls, root, folder):
        """The ds004611_prisma_3t cohort from the dataset's local copy at `root`: each
        subject's UNIT1 image segmented inside the brain found on its second inversion (a
        plain gradient echo with an ordinary background), its T1 and T2 maps read on those
        tissues, grey and white matter only (neither protocol resolves CSF's T1 of about 4 s
        or its T2 of over a second). The maps are aligned by their affines,
        the two acquisitions minutes apart (stated). Each subject's first session is read (some
        have only a second). The work lands in `folder`/<subject> and is reused when there;
        each member carries its age, sex and health."""
        root, folder = pathlib.Path(root), pathlib.Path(folder)
        people = participants.bids(root, health="healthy")
        members = []
        for subject in sorted(p.name for p in root.glob("sub-*") if p.is_dir()):
            anat = sorted(root.glob(f"{subject}/ses-*/anat"))[0]
            session = anat.parts[-2]
            stem = f"{subject}_{session}_run-1"
            work = folder / subject
            work.mkdir(parents=True, exist_ok=True)
            if not (work / "tissue_pve_2.nii.gz").exists():
                maps.tissues(anat / f"{stem}_UNIT1.nii.gz", work,
                             strip_on=anat / f"{stem}_inv-2_MP2RAGE.nii.gz", crop=False)
            age = cls._age(root / subject / f"{subject}_sessions.tsv", session)
            records = ()
            for kind, leaf in (("T1map", "water.t1_s"), ("T2map", "water.t2_s")):
                if (anat / f"{stem}_{kind}.nii.gz").exists():
                    records += cls._read(anat / f"{stem}_{kind}", leaf, work, age)
            if records:
                traits = {**people.get(subject, {"health": "healthy"}),
                          **({} if age is None else {"age_y": age})}
                members.append(_subject(subject, "siemens_prisma_3t", records, traits))
        return Cohort("ds004611_prisma_3t", tuple(members))

    @classmethod
    def _read(cls, stem, leaf, work, age):
        import dataclasses

        import nibabel as nib

        image = nib.load(f"{stem}.nii.gz")
        seconds = work / f"{leaf}.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.float32) / 1000.0,
                                 image.affine), str(seconds))
        header = json.loads(pathlib.Path(f"{stem}.json").read_text())
        reading = MapReading.from_files(leaf, seconds,
                                        *(work / f"tissue_pve_{k}.nii.gz" for k in (1, 2, 0)),
                                        header=header)
        what = ("the vendor's MP2RAGE T1 map" if leaf == "water.t1_s"
                else "the GRAPPATINI T2 map (model-based, 16 echoes)")
        return tuple(dataclasses.replace(
            record, note=f"one child's {what}, aged {age} years; CSF not resolved by the "
                         "protocol, left to the literature")
            for record in reading.records(
                cls.DATASET, cls.CITATION, cls.URL,
                processing="the map onto the tissues' grid by the affines, linear; the tissues "
                           "FSL fast of the UNIT1 image inside the brain bet finds on the "
                           "second inversion")
            if not record.leaf.endswith(".csf"))

    @staticmethod
    def _age(path, session):
        """The session's age in years, None where the table has none."""
        ages = {row[0]: row[1] for row in (line.split("\t") for line in
                                            path.read_text().splitlines()[1:])
                if len(row) > 1 and row[1].strip()}
        return float(ages[session]) if session in ages else None


#**************************************************************************************************#
#                                            Class IXI                                             #
#**************************************************************************************************#
#                                                                                                  #
# The IXI dataset: dual-echo turbo spin echo PD and T2 images of adults at three London sites.     #
#                                                                                                  #
#**************************************************************************************************#
class IXI:
    """The IXI dataset: dual-echo turbo spin echo PD and T2 images of adults at three London sites.

    Source: the IXI dataset, Imperial College London, brain-development.org/ixi-dataset (CC BY-SA
            3.0), mirrored on NITRC (www.nitrc.org/projects/ixi_dataset), which serves each
            session's per-scan TE, TR and flip and the subject's age.
    Data:   about 580 normal, healthy adults aged 20 to 86 (the site's words): Guy's Hospital
            (Philips Gyroscan Intera 1.5 T), Hammersmith Hospital (Philips Intera 3 T), the
            Institute of Psychiatry (GE 1.5 T); per subject a sagittal 3D T1-weighted image and
            one dual-echo turbo spin echo (echo train 16) giving PD at TE 8 ms and T2 at TE 100
            ms, one TR, on one grid (the IOP headers state TE 8.72 and 104.64 ms, and "3T",
            which the NITRC session's 1.5 T and the site's description overrule).
    """

    DATASET = "IXI, {site} (CC BY-SA 3.0)"
    URL = "https://brain-development.org/ixi-dataset/"
    SITES = {
        "Guys": ("ixi_guys_1p5t", "generic_1p5t", "Guy's Hospital",
                 {"Manufacturer": "Philips", "ManufacturersModelName": "Gyroscan Intera",
                  "MagneticFieldStrength": 1.5}),
        "HH": ("ixi_hh_3t", "generic_3t", "Hammersmith Hospital",
               {"Manufacturer": "Philips", "ManufacturersModelName": "Intera",
                "MagneticFieldStrength": 3.0}),
        "IOP": ("ixi_iop_1p5t", "generic_1p5t", "Institute of Psychiatry",
                {"Manufacturer": "GE", "MagneticFieldStrength": 1.5}),
    }

    #**************#
    #   cohort     #
    #**************#
    @classmethod
    def cohort(cls, root, folder, site):
        """The cohort of one site from the local copy at `root` (one folder per subject
        holding its T1, PD and T2 NIfTI and the NITRC session's subject.json): each subject's
        T1-weighted image segmented (scan.maps.tissues, neck cropped), its T2 two-point from
        the PD and T2 echoes, (TE2 - TE1) / ln(PD / T2) where PD exceeds T2, read on those
        tissues, grey and white matter only (CSF's T2 of over a second is beyond a 100 ms
        echo). The T2 map is aligned to the T1 by the affines, the scans of one session
        (stated: a rigid refinement changes the tissues' readings little). The work lands
        in `folder`/<subject> and is reused when there; each member carries its age, sex and
        health."""
        import dataclasses
        import re

        import nibabel as nib
        import nibabel.processing
        import scipy.ndimage

        name, scanner, place, stated = cls.SITES[site]
        root, folder = pathlib.Path(root), pathlib.Path(folder)
        members = []
        for subject in sorted(root.glob(f"*_{site}")):
            found = {kind: next(subject.glob(f"*-{kind}.nii.gz"), None)
                     for kind in ("T1", "PD", "T2")}
            if None in found.values() or not (subject / "subject.json").exists():
                continue
            info = json.loads((subject / "subject.json").read_text())
            scans = info.get("scans", {})
            if "te" in scans.get("PD", {}) and "te" in scans.get("T2", {}):
                te = (scans["PD"]["te"] / 1000.0, scans["T2"]["te"] / 1000.0)
                tr = scans["T2"].get("tr")
            else:
                header = [nib.load(str(found[k])).header["descrip"].tobytes().decode(
                    errors="replace") for k in ("PD", "T2")]
                te = tuple(float(re.search(r"TE=([\d.]+)ms", h).group(1)) / 1000.0
                           for h in header)
                tr = float(re.search(r"TR=([\d.]+)ms", header[1]).group(1))
            work = folder / subject.name
            work.mkdir(parents=True, exist_ok=True)
            if not (work / "tissue_pve_2.nii.gz").exists():
                maps.tissues(found["T1"], work)
            image = nib.load(str(found["PD"]))
            pd = np.asarray(image.dataobj, dtype=np.float64)
            t2w = np.asarray(nib.load(str(found["T2"])).dataobj, dtype=np.float64)
            valid = (t2w > 0.0) & (pd > t2w)
            t2 = np.where(valid, (te[1] - te[0]) / np.log(np.where(valid, pd, 2.0)
                                                         / np.where(valid, t2w, 1.0)), 0.0)
            nib.save(nib.Nifti1Image(t2.astype(np.float32), image.affine),
                     str(work / "water.t2_s.nii.gz"))
            # the echoes' scaling: CSF's T2 of a second or more leaves its PD a few percent
            # over its T2 intensity, so pure CSF (eroded twice off the partial volume's edge)
            # at or below 1 says the two series were scaled apart and no two-point T2 holds
            csf = scipy.ndimage.binary_erosion(np.asarray(nibabel.processing.resample_from_to(
                nib.load(str(work / "tissue_pve_0.nii.gz")), image, order=1).dataobj)
                > maps.PURE, iterations=2)
            ratio = float(np.median(pd[csf] / np.maximum(t2w[csf], 1.0)))
            failed = "" if ratio > 1.0 else (
                f"pure CSF reads PD over T2 at {ratio:.3f}, at or below 1: the echoes are "
                "not scaled alike")
            reading = MapReading.from_files(
                "water.t2_s", work / "water.t2_s.nii.gz",
                *(work / f"tissue_pve_{k}.nii.gz" for k in (1, 2, 0)),
                header={**stated, "SequenceName": "dual-echo turbo spin echo, PD and T2",
                        "EchoTime": list(te), "RepetitionTime": tr and tr / 1000.0})
            age = info.get("age")
            note = (f"one adult's two-point T2 off a dual-echo turbo spin echo (TE "
                    f"{te[0] * 1e3:g} and {te[1] * 1e3:g} ms, echo train 16), aged {age} years: a "
                    "turbo train's stimulated echoes lengthen a two-point T2 over a multi-echo "
                    "spin echo's; CSF left to the literature")
            records = tuple(dataclasses.replace(
                record, note=note, failed=failed,
                measured={**record.measured, "csf_pd_over_t2": round(ratio, 4)})
                for record in reading.records(
                    cls.DATASET.format(site=place), None, cls.URL,
                    processing="the two-point T2 of the PD and T2 echoes onto the tissues' grid "
                               "by the affines, linear; the tissues FSL fast of the session's "
                               "T1-weighted image (robustfov, bet)")
                if not record.leaf.endswith(".csf"))
            if records:
                sex = {"female": "F", "male": "M"}.get(info.get("gender"))
                traits = {"health": "healthy", **({} if age is None else {"age_y": age}),
                          **({} if sex is None else {"sex": sex})}
                members.append(_subject(subject.name, scanner, records, traits))
        return Cohort(name, tuple(members))


#**************************************************************************************************#
#                                        Class RadialMESE                                          #
#**************************************************************************************************#
#                                                                                                  #
# One subject's 3D radial multi-echo spin echo at 3 T with its EPG-fitted T2 map.                  #
#                                                                                                  #
#**************************************************************************************************#
class RadialMESE:
    """One subject's 3D radial multi-echo spin echo at 3 T with its EPG-fitted T2 map.

    Source: Corbin, N., et al. Whole-brain T2 mapping with radial sampling and retrospective
            motion correction at 3T. Magn Reson Med 93 (2025) 1026, doi 10.1002/mrm.30328; the
            data: Zenodo (2024), doi 10.5281/zenodo.10908743, CC BY 4.0, the paper's experiment
            4 (one of seven healthy participants, 5 F and 2 M, aged 28 to 50, scanned without and
            with deliberate motion).
    Data:   a house-made sequence on a Siemens Prisma 3 T, 50 echoes from 7 to 350 ms (7 ms
            apart), TR 1 s, 1.5 mm isotropic; the T2 map fitted by Anima's extended phase graph
            estimation with B1 and M0 (the stimulated echoes modelled), the reconstruction
            without motion ("Nomotion/Nomoco"); the paper's MPRAGE is not deposited.
    """

    DATASET = "3D radial multi-echo spin echo, Bordeaux, Zenodo 10908743 (CC BY 4.0)"
    URL = "https://zenodo.org/records/10908743"
    FOLDER = pathlib.Path("Reconstruction") / "Nomotion" / "Nomoco" / "RecoEchoes"

    #**************#
    #   cohort     #
    #**************#
    @classmethod
    def cohort(cls, root, folder):
        """The radial_mese_prisma_3t cohort from the deposit's Reconstruction archive unpacked
        at `root`: the subject's first echo (TE 7 ms, T1-weighted by the 1 s repetition) is
        segmented with the standard tissue priors (scan.maps.tissues_by_atlas: the fits fill the
        field of view and the radial streaks defeat bet, the grey-white contrast is weak), the
        EPG T2 map read on those tissues, white matter only: without the subject's own
        structural image the thin cortex is not parted from CSF, whose T2 of over a second
        lies past a 350 ms train, so grey (and CSF) stay with the literature (the paper's own
        grey, off each participant's MPRAGE, reads lower than these grey cells). The work
        lands in `folder` and is reused when there."""
        import dataclasses

        import nibabel as nib

        echoes, folder = pathlib.Path(root) / cls.FOLDER, pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "tissue_pve_2.nii.gz").exists():
            series = nib.load(str(echoes / "Echoes4D.nii"))
            nib.save(nib.Nifti1Image(np.abs(np.asarray(series.dataobj[..., 0])).astype(np.float32),
                                     series.affine), str(folder / "echo1.nii.gz"))
            maps.tissues_by_atlas(folder / "echo1.nii.gz", folder)
        t2 = nib.load(str(echoes / "Anima_T2map.nii"))
        nib.save(nib.Nifti1Image(np.asarray(t2.dataobj, dtype=np.float32) / 1000.0, t2.affine),
                 str(folder / "water.t2_s.nii.gz"))
        reading = MapReading.from_files(
            "water.t2_s", folder / "water.t2_s.nii.gz",
            *(folder / f"tissue_pve_{k}.nii.gz" for k in (1, 2, 0)),
            header={"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma",
                    "MagneticFieldStrength": 3.0,
                    "SequenceName": "3D radial multi-echo spin echo, 50 echoes",
                    "EchoTime": [0.007, 0.350], "RepetitionTime": 1.0})
        records = tuple(dataclasses.replace(
            record, note="one subject's EPG-fitted T2 (Anima, B1 and M0 fitted) of a 50-echo "
                         "radial spin echo train, white matter only: the paper's Table 1 reads "
                         "white matter 71.7 to 80.5 ms over its seven participants off their "
                         "own MPRAGE; grey and CSF left to the literature")
            for record in reading.records(
                cls.DATASET, "Corbin et al. 2025, Magn Reson Med 93:1026", cls.URL,
                processing="the EPG T2 map onto the tissues' grid by the affines, linear; the "
                           "tissues FSL fast with the standard priors throughout on the first "
                           "echo, inside MNI152's brain mask flirted onto it")
            if record.leaf.endswith(".wm"))
        return Cohort("radial_mese_prisma_3t", (_subject(
            "the subject", "siemens_prisma_3t", records,
            {"age_y": (28.0, 50.0), "health": "healthy"}),))
