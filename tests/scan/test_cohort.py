####################################################################################################
#                                          test_cohort.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 26/09/26                                                                                #
#                                                                                                  #
# Purpose: A cohort draws one real subject with its measured values fixed; the package's cohorts   #
#          are its measured database, pooled for a scanner by the conditions it matches.           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Fixed, Press, Scanner
from mrsandbox.scan import Cohort, Member, ScannerProfile
from mrsandbox.scan import cohort as cohorts
from mrsandbox.scan.calibrations import Calibration, calibrations

PLACE, SIZE = (0.0, -35.0, 20.0), (30.0, 30.0, 30.0)


def record(leaf, value, vendor="siemens", model="TrioTim", field_t=2.89, channels=32, failed=""):
    return Calibration(leaf=leaf, value=value, prior=tuple(sorted((0.75 * value, 1.25 * value))), measured={},
                       source={"dataset": "a cohort", "citation": None, "url": None},
                       scanner={"vendor": vendor, "model": model, "software": None, "field_t": field_t},
                       coil={"channels": (channels,), "name": None}, sequence={"name": "svs_se", "family": "Press"},
                       acquisition={}, voxel={}, processing="", date="27/09/26", failed=failed)


def member(subject, sigma, residual, **conditions):
    records = (record("noise.sigma", sigma, **conditions), record("water.adjusted_residual", residual, **conditions))
    return Member(subject, ScannerProfile(Scanner.preset("siemens_trio_3t"), records), PLACE, SIZE)


#**************************************************************************************************#
#                                         Class TestCohort                                         #
#**************************************************************************************************#
class TestCohort:

    #**********#
    #   draw   #
    #**********#
    def test_a_draw_is_one_subject_with_its_values_fixed(self):
        cohort = Cohort("two subjects", (member("S01", 6.0, -0.006), member("S02", 9.0, -0.012)))
        drawn = {cohort.draw(seed).subject for seed in range(24)}
        assert drawn == {"S01", "S02"}
        one = next(m for m in cohort.members if m.subject == "S02")
        scan = one.scan(Press(te1_s=0.011, te2_s=0.024))
        sigma, residual = scan.params.noise.sigma, scan.params.water.adjusted_residual
        assert sigma == Fixed(9.0) and residual == Fixed(-0.012)                    # the subject's own, not a range about it
        assert sigma.chosen == "dataset" and scan.voxel(PLACE, SIZE).sample(seed=3).noise.sigma == 9.0     # a cohort subject is the dataset source
        assert one.profile.records[0].prior == (6.75, 11.25) and one.fixed().records[0].prior == (9.0, 9.0)   # the member's own untouched

    def test_a_failed_reading_is_dropped_from_the_member(self):
        failed = record("acquisition.eddy_transient_rad", 2.8, failed="the phase 5 ms in is past a quarter turn")
        one = Member("S07", ScannerProfile(Scanner.preset("siemens_trio_3t"), (record("noise.sigma", 4.0), failed)), PLACE, SIZE)
        assert [r.leaf for r in one.fixed().records] == ["noise.sigma"]
        scan = one.scan(Press(te1_s=0.011, te2_s=0.024))
        assert scan.params.noise.sigma == Fixed(4.0) and scan.params.acquisition.eddy_transient_rad != Fixed(2.8)   # the pool serves the leaf

    #*******************#
    #   save and load   #
    #*******************#
    def test_a_cohort_is_saved_and_loaded_as_one_file(self, tmp_path):
        cohort = Cohort("two subjects", (member("S01", 6.0, -0.006), member("S02", 9.0, -0.012)))
        cohort.save(tmp_path / "cohort.json")
        loaded = Cohort.load(tmp_path / "cohort.json")
        assert loaded.name == cohort.name and [m.subject for m in loaded.members] == ["S01", "S02"]
        assert loaded.members[1].profile.calibrations()["noise.sigma"].value == 9.0
        assert loaded.members[1].position_mm == PLACE and loaded.members[1].brain == "icbm152"
        header = loaded.members[0].scan(Press(te1_s=0.011, te2_s=0.024)).voxel(PLACE, SIZE).acquire(
            seed=0, npoints=256).hdr_ext["SimulationPhantom"]["calibrations"]
        assert header["noise.sigma"]["source"] == "subject S01 of a cohort"           # the member names itself


#**************************************************************************************************#
#                                         Class TestPooled                                         #
#**************************************************************************************************#
#                                                                                                  #
# The measured database pooled for a scanner: as specific as the scanner, failed rows out.         #
#                                                                                                  #
#**************************************************************************************************#
class TestPooled:
    """The measured database pooled for a scanner: as specific as the scanner, failed rows out."""

    @pytest.fixture(autouse=True)
    def own_database(self):
        yield
        cohorts.forget()                                                             # the package's pool again for the tests after

    TRIO = [4.0, 4.4, 5.0, 5.5, 6.1]
    GE = [7.0, 7.5, 9.0]

    def database(self, monkeypatch):
        trio = Cohort("trio", tuple(member(f"S{k}", sigma, -0.001 * k) for k, sigma in enumerate(self.TRIO)) + (
            Member("S9", ScannerProfile(Scanner.preset("siemens_trio_3t"), (record("noise.sigma", 100.0, failed="a stated rule"),)), PLACE, SIZE),))
        ge = Cohort("ge", tuple(member(f"G{k}", sigma, 0.03, vendor="ge", model=None, field_t=3.0, channels=8) for k, sigma in enumerate(self.GE)))
        monkeypatch.setattr(cohorts, "package", lambda: (trio, ge))
        cohorts.forget()                                                             # the pool is cached: this database is another

    def test_the_pool_is_the_median_over_the_matching_subjects_without_the_failed(self, monkeypatch):
        self.database(monkeypatch)
        pooled = cohorts.pooled("noise.sigma", Scanner.preset("siemens_trio_3t"), channels=32)
        assert pooled.value == np.median(self.TRIO) and pooled.prior == tuple(np.round(np.percentile(self.TRIO, (5, 95)), 7))
        assert pooled.measured["subjects"] == 5 and pooled.measured["matched_on"] == {"field_t": 2.895, "vendor": "siemens", "model": "TrioTim", "channels": 32}
        assert pooled.scanner == {"vendor": "siemens", "model": "TrioTim", "software": None, "field_t": 2.89}
        assert pooled.source["dataset"] == "pooled over 5 subjects of 1 cohort: trio" and pooled.sequence["family"] == "Press"

    def test_the_pool_is_as_specific_as_the_scanner(self, monkeypatch):
        self.database(monkeypatch)
        everything = cohorts.pooled("noise.sigma", Scanner.preset("generic_3t"))
        assert everything.measured["subjects"] == 8 and everything.measured["matched_on"] == {"field_t": 3.0}
        assert everything.value == np.median(self.TRIO + self.GE) and everything.scanner["vendor"] == "pooled"
        assert everything.prior == tuple(np.round(np.percentile(self.TRIO + self.GE, (5, 95)), 7))
        ge = cohorts.pooled("noise.sigma", Scanner.preset("ge_mr750_3t"), channels=8)            # rows without a model do not fail the model
        assert ge.measured["subjects"] == 3 and ge.prior == (7.0, 9.0) and ge.scanner["model"] == "unstated"    # the range over fewer than five
        many = cohorts.pooled("noise.sigma", Scanner.preset("ge_mr750_3t"), channels=32)          # nothing at 32 channels: the count is dropped
        assert many.measured["subjects"] == 3 and many.measured["matched_on"] == {"field_t": 3.001, "vendor": "ge"}
        philips = cohorts.pooled("noise.sigma", Scanner.preset("philips_achieva_3t"))            # no vendor rows: every system at the field
        assert philips.measured["subjects"] == 8 and philips.measured["matched_on"] == {"field_t": 3.0}
        low = cohorts.pooled("noise.sigma", Scanner.preset("generic_1p5t"))                       # nothing at the field: everything, the departure names it
        assert low.measured["subjects"] == 8 and low.measured["matched_on"] == {}
        assert cohorts.pooled("conc.Cr", Scanner.preset("generic_3t")) is None

    def test_the_catalogue_serves_the_pool_and_own_records_outrank_it(self, monkeypatch):
        self.database(monkeypatch)
        served = calibrations("siemens_trio_3t", channels=32)
        assert served["noise.sigma"] == cohorts.pooled("noise.sigma", Scanner.preset("siemens_trio_3t"), channels=32)
        assert served["water.adjusted_residual"].measured["subjects"] == 5 and served["conc.Cr"].source["dataset"].startswith("Big GABA")
        own = record("noise.sigma", 8.0)
        assert calibrations("siemens_trio_3t", records=(own,), channels=32)["noise.sigma"] is own

    #***********************#
    #   the package's own   #
    #***********************#
    def test_the_package_cohorts_are_the_records(self):
        # every record of a 3 T scanner is the median over the package's rows read under its conditions
        assert "big_gaba_trio" in Cohort.names()
        trio = Scanner.preset("siemens_trio_3t")
        served = ScannerProfile.preset("siemens_trio_3t").calibrations()
        for leaf in cohorts.leaves():
            matching = [r.value for _, _, r in cohorts.rows(leaf) if cohorts._matches(r, trio.b0_t, trio.vendor, trio.model, None)]
            assert served[leaf].value == round(float(np.median(matching)), 7) and served[leaf].measured["subjects"] == len(matching)
        failed = [r for c in cohorts.package() for m in c.members for r in m.profile.records if r.failed]
        assert failed and all(r.value not in [row[2].value for row in cohorts.rows(r.leaf)] or r.failed for r in failed)
        assert all(abs(r.value) <= 0.5 * np.pi for _, _, r in cohorts.rows("acquisition.eddy_transient_rad"))
