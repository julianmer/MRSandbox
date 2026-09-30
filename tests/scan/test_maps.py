####################################################################################################
#                                           test_maps.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: A quantitative map read per tissue gives back each tissue's value, and its records give #
#          the water's per-tissue leaf a source per tissue, the operator's or the package's pool.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Scanner
from mrsandbox.params import PerTissue, Sourced, Uniform
from mrsandbox.scan import Cohort, Member, ScannerProfile
from mrsandbox.scan import cohort as cohorts
from mrsandbox.scan.defaults import default_params
from mrsandbox.scan.maps import MapReading

T1 = {"gm": 1.45, "wm": 0.85, "csf": 4.0}


def subject():
    """Three slabs of tissue on a 1 mm grid, their borders blurred over three cells, a T1 map of
    the stated values mixed by the partial volumes, one percent of noise, and the head's outside
    at zero."""
    x = np.arange(40.0)
    shares = [np.clip((x - 12.0) / 3.0 + 0.5, 0.0, 1.0)
              * np.clip((24.0 - x) / 3.0 + 0.5, 0.0, 1.0)]  # grey in the middle
    shares = [1.0 - shares[0]] + shares
    wm = np.clip((12.0 - x) / 3.0 + 0.5, 0.0, 1.0)
    tissues = {"wm": wm, "gm": shares[1], "csf": np.clip(1.0 - wm - shares[1], 0.0, 1.0)}
    tissues = {t: np.broadcast_to(v[:, None, None], (40, 20, 20)).copy()
               for t, v in tissues.items()}
    tissues["csf"][:, :, 18:] = 0.0  # the head's outside: no tissue, no map
    t1 = sum(T1[t] * tissues[t] for t in T1) * (
        1.0 + 0.01 * np.random.default_rng(0).standard_normal((40, 20, 20)))
    return np.where(tissues["csf"] + tissues["gm"] + tissues["wm"] > 0.5, t1, 0.0), tissues


#**************************************************************************************************#
#                                        Class TestMapReading                                      #
#**************************************************************************************************#
class TestMapReading:

    def test_a_map_reads_back_each_tissues_value_off_its_own_cells(self, tmp_path):
        import nibabel as nib

        t1, tissues = subject()
        reading = MapReading.from_arrays("water.t1_s", t1, tissues, {"MagneticFieldStrength": 3})
        records = {r.leaf: r for r in reading.records("a synthetic subject",
                                                      method="inversion recovery")}
        assert set(records) == {"water.t1_s.gm", "water.t1_s.wm", "water.t1_s.csf"}
        # the method is the rows' sequence family
        assert {r.sequence["family"] for r in records.values()} == {"inversion recovery"}
        for tissue, value in T1.items():  # the mixed cells left out: the stated value
            assert records[f"water.t1_s.{tissue}"].value == pytest.approx(value, rel=0.005)
            # a quarter either side, as every reading of one's own; not the cells' spread
            read = records[f"water.t1_s.{tissue}"]
            assert read.prior == pytest.approx((0.75 * read.value, 1.25 * read.value), abs=1e-4)
        assert (records["water.t1_s.gm"].carried_by == "subject"
                and records["water.t1_s.gm"].scanner["field_t"] == 3.0)
        # the same off files, the map an R1 map on a grid twice as coarse, brought onto the
        # tissues' grid by the affines
        coarse = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(np.where(t1 > 0.0, 1.0 / np.where(t1 > 0.0, t1, 1.0),
                                          0.0)[::2, ::2, ::2].astype(np.float32), coarse),
                 tmp_path / "r1.nii.gz")
        for tissue in ("gm", "wm", "csf"):
            nib.save(nib.Nifti1Image(tissues[tissue].astype(np.float32), np.eye(4)),
                     tmp_path / f"{tissue}.nii.gz")
        read = MapReading.from_files("water.t1_s", tmp_path / "r1.nii.gz",
                                     *(tmp_path / f"{t}.nii.gz" for t in ("gm", "wm", "csf")),
                                     r1=True)
        assert (np.median(read.values["wm"]) == pytest.approx(T1["wm"], rel=0.02)
                and np.median(read.values["gm"]) == pytest.approx(T1["gm"], rel=0.02))

    def test_its_records_give_the_water_its_own_source_per_tissue(self):
        t1, tissues = subject()
        tissues["csf"][:] = 0.0  # no CSF of its own: that tissue stays the literature's
        records = MapReading.from_arrays("water.t1_s", t1, tissues).records("my map")
        leaf = default_params(records=records)["water"]["t1_s"]
        assert (isinstance(leaf, Sourced) and leaf.chosen == "own"
                and isinstance(leaf.own, PerTissue))
        wm = next(r for r in records if r.leaf == "water.t1_s.wm")
        assert leaf.own.wm == Uniform(*wm.prior) and leaf.own.csf == leaf.literature.csf
        # a draw lies within its reading's quarter either side
        assert leaf.sample(np.random.default_rng(0)).wm == pytest.approx(T1["wm"], rel=0.25)

    def test_the_package_rows_pool_per_tissue(self, monkeypatch):
        t1, tissues = subject()
        rows = MapReading.from_arrays("water.t1_s", t1, tissues,
                                      {"MagneticFieldStrength": 3}).records("a map dataset")
        monkeypatch.setattr(cohorts, "package", lambda: (Cohort("maps", (
            Member("S1", ScannerProfile(Scanner.preset("generic_3t"), rows),
                   (0.0, 0.0, 0.0), (20.0, 20.0, 20.0),
                   traits={"age_y": 30.0, "health": "healthy"}),)),))
        cohorts.forget()
        try:
            leaf = default_params(scanner=Scanner.preset("generic_3t"))["water"]["t1_s"]
            assert leaf.chosen == "literature" and set(leaf.available) == {
                "literature", "pooled", "mixed"}
            assert np.median(leaf.pooled.gm.values) == pytest.approx(T1["gm"], rel=0.005)
            assert "gm: pooled over 1 subject" in leaf.provenance["pooled"]
        finally:
            cohorts.forget()
