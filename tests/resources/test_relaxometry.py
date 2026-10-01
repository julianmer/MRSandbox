####################################################################################################
#                                       test_relaxometry.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 29/09/26                                                                                #
#                                                                                                  #
# Purpose: The relaxometry datasets' readers give back each subject's stated T1 and T2 per tissue  #
#          off synthetic subjects laid out as the datasets are, their segmentations in place.      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from mrsandbox.resources.relaxometry import IXI, Developmental, RadialMESE

T2 = {"gm": 0.100, "wm": 0.085, "csf": 2.0}
T1 = {"gm": 1.50, "wm": 0.90, "csf": 4.0}


def slabs():
    """White, grey and CSF slabs of 12 cells each along x on a 1 mm grid, pure throughout."""
    tissues = {t: np.zeros((36, 16, 16), dtype=np.float32) for t in ("wm", "gm", "csf")}
    for k, tissue in enumerate(tissues):
        tissues[tissue][12 * k:12 * (k + 1)] = 1.0
    return tissues


def segmented(work):
    """The segmentation `maps.tissues` leaves, in place so the reader reuses it."""
    import nibabel as nib

    work.mkdir(parents=True)
    for k, tissue in ((0, "csf"), (1, "gm"), (2, "wm")):
        nib.save(nib.Nifti1Image(slabs()[tissue], np.eye(4)), work / f"tissue_pve_{k}.nii.gz")


def saved(path, data):
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)


#**************************************************************************************************#
#                                            Class TestIXI                                         #
#**************************************************************************************************#
class TestIXI:

    def subject(self, root, work, name, pd_scale=1.0):
        """One IXI subject: the dual echo of the stated T2s at TE 8 and 100 ms, NITRC's timings."""
        folder = root / name
        folder.mkdir(parents=True)
        tissues = slabs()
        rho = {"gm": 800.0, "wm": 700.0, "csf": 1000.0}
        echo = {te: sum(rho[t] * tissues[t] * np.exp(-te / T2[t]) for t in T2)
                for te in (0.008, 0.100)}
        stem = name.replace("_", "-")
        saved(folder / f"IXI{stem}-PD.nii.gz", pd_scale * echo[0.008])
        saved(folder / f"IXI{stem}-T2.nii.gz", echo[0.100])
        saved(folder / f"IXI{stem}-T1.nii.gz", sum(tissues.values()))
        (folder / "subject.json").write_text(json.dumps({
            "age": 42.0, "gender": "female", "field": "1.5T",
            "scans": {"PD": {"te": 8, "tr": 8178.34}, "T2": {"te": 100, "tr": 8178.34}}}))
        segmented(work / name)

    def test_the_two_point_t2_comes_back_per_tissue(self, tmp_path):
        self.subject(tmp_path / "ixi", tmp_path / "work", "001_Guys")
        cohort = IXI.cohort(tmp_path / "ixi", tmp_path / "work", "Guys")
        records = {r.leaf: r for r in cohort.members[0].profile.records}
        assert set(records) == {"water.t2_s.gm", "water.t2_s.wm"}   # CSF to the literature
        for tissue in ("gm", "wm"):
            assert records[f"water.t2_s.{tissue}"].value == pytest.approx(T2[tissue], rel=1e-3)
        read = records["water.t2_s.wm"]
        assert (not read.failed and read.scanner["field_t"] == 1.5
                and read.sequence["te_s"] == [0.008, 0.1])
        assert cohort.members[0].traits == {"age_y": 42.0, "sex": "F", "health": "healthy"}
        assert cohort.name == "ixi_guys_1p5t" and "1.5 T" in cohort.label()

    def test_echoes_scaled_apart_fail_by_the_csf(self, tmp_path):
        # PD scaled down 10 percent: CSF's PD falls under its T2 intensity, which no finite T2
        # gives, so the subject's rows fail and are never pooled
        self.subject(tmp_path / "ixi", tmp_path / "work", "002_Guys", pd_scale=0.9)
        records = IXI.cohort(tmp_path / "ixi", tmp_path / "work", "Guys").members[0].profile.records
        assert all("not scaled alike" in r.failed for r in records)
        assert records[0].measured["csf_pd_over_t2"] < 1.0

    def test_a_site_whose_echoes_come_out_scaled_apart_fails_whole(self, tmp_path):
        self.subject(tmp_path / "ixi", tmp_path / "work", "003_IOP")
        cohort = IXI.cohort(tmp_path / "ixi", tmp_path / "work", "IOP")
        # its CSF passes, its rows fail all the same
        records = cohort.members[0].profile.records
        assert records[0].measured["csf_pd_over_t2"] > 1.0
        assert all("scaled apart" in r.failed for r in records)
        assert {r.sequence["family"] for r in records} == {"two-point turbo spin echo"}


#**************************************************************************************************#
#                                      Class TestDevelopmental                                     #
#**************************************************************************************************#
class TestDevelopmental:

    def test_the_vendor_maps_come_back_in_seconds_per_tissue(self, tmp_path):
        tissues = slabs()
        anat = tmp_path / "ds" / "sub-X" / "ses-1" / "anat"
        anat.mkdir(parents=True)
        stem = anat / "sub-X_ses-1_run-1"
        header = {"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma_fit",
                  "MagneticFieldStrength": 3}
        for kind, values in (("T1map", T1), ("T2map", T2)):     # both stored in ms
            saved(f"{stem}_{kind}.nii.gz", sum(1000.0 * values[t] * tissues[t] for t in values))
            (anat / f"sub-X_ses-1_run-1_{kind}.json").write_text(json.dumps(header))
        (tmp_path / "ds" / "participants.tsv").write_text("participant_id\tsex\nsub-X\tM\n")
        sessions = tmp_path / "ds" / "sub-X" / "sub-X_sessions.tsv"
        sessions.write_text("session_id\tage\nses-1\t9.3\n")
        segmented(tmp_path / "work" / "sub-X")
        cohort = Developmental.cohort(tmp_path / "ds", tmp_path / "work")
        records = {r.leaf: r for r in cohort.members[0].profile.records}
        assert set(records) == {f"water.{k}_s.{t}" for k in ("t1", "t2") for t in ("gm", "wm")}
        for tissue in ("gm", "wm"):
            assert records[f"water.t1_s.{tissue}"].value == pytest.approx(T1[tissue], rel=1e-3)
            assert records[f"water.t2_s.{tissue}"].value == pytest.approx(T2[tissue], rel=1e-3)
        assert cohort.members[0].traits == {"sex": "M", "health": "healthy", "age_y": 9.3}
        assert cohort.members[0].brain == "nihpd_7-11"   # the band centred nearest the age
        assert records["water.t2_s.wm"].scanner["model"] == "Prisma_fit"


#**************************************************************************************************#
#                                       Class TestRadialMESE                                       #
#**************************************************************************************************#
class TestRadialMESE:

    def test_the_epg_map_comes_back_in_seconds_per_tissue(self, tmp_path):
        tissues = slabs()
        echoes = tmp_path / "deposit" / RadialMESE.FOLDER
        echoes.mkdir(parents=True)
        saved(echoes / "Anima_T2map.nii", sum(1000.0 * T2[t] * tissues[t] for t in T2))  # ms
        segmented(tmp_path / "work")
        cohort = RadialMESE.cohort(tmp_path / "deposit", tmp_path / "work")
        records = {r.leaf: r for r in cohort.members[0].profile.records}
        assert set(records) == {"water.t2_s.wm"}   # the cortex not parted from CSF
        assert records["water.t2_s.wm"].value == pytest.approx(T2["wm"], rel=1e-3)
        # one of the paper's seven healthy participants, aged 28 to 50
        assert cohort.members[0].traits == {"age_y": (28.0, 50.0), "health": "healthy"}
