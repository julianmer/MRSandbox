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
import dataclasses

import numpy as np
import pytest

from mrsandbox import Empirical, Fixed, Press, Scanner
from mrsandbox.scan import Cohort, Member, ScannerProfile
from mrsandbox.scan import cohort as cohorts
from mrsandbox.scan.calibrations import Calibration, calibrations

PLACE, SIZE = (0.0, -35.0, 20.0), (30.0, 30.0, 30.0)


def record(leaf, value, vendor="siemens", model="TrioTim", field_t=2.89, channels=32, failed=""):
    return Calibration(leaf=leaf, value=value, prior=tuple(sorted((0.75 * value, 1.25 * value))),
                       measured={}, source={"dataset": "a cohort", "citation": None, "url": None},
                       scanner={"vendor": vendor, "model": model, "software": None,
                                "field_t": field_t},
                       coil={"channels": (channels,), "name": None},
                       sequence={"name": "svs_se", "family": "Press"},
                       acquisition={}, voxel={}, processing="", date="27/09/26", failed=failed)


def member(subject, sigma, residual, **conditions):
    records = (record("noise.sigma", sigma, **conditions),
               record("water.adjusted_residual", residual, **conditions))
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
        # the subject's own, not a range about it
        assert sigma == Fixed(9.0) and residual == Fixed(-0.012)
        # a cohort subject is the dataset source
        assert (sigma.chosen == "dataset"
                and scan.voxel(PLACE, SIZE).sample(seed=3).noise.sigma == 9.0)
        # the member's own untouched
        assert (one.profile.records[0].prior == (6.75, 11.25)
                and one.fixed().records[0].prior == (9.0, 9.0))

    def test_a_failed_reading_is_dropped_from_the_member(self):
        failed = record("acquisition.eddy_transient_rad", 2.8,
                        failed="the phase 5 ms in is past a quarter turn")
        one = Member("S07", ScannerProfile(Scanner.preset("siemens_trio_3t"),
                                           (record("noise.sigma", 4.0), failed)), PLACE, SIZE)
        assert [r.leaf for r in one.fixed().records] == ["noise.sigma"]
        scan = one.scan(Press(te1_s=0.011, te2_s=0.024))
        # the pool serves the leaf
        assert (scan.params.noise.sigma == Fixed(4.0)
                and scan.params.acquisition.eddy_transient_rad != Fixed(2.8))

    def test_a_run_whose_noise_reads_over_twice_the_median_is_a_failed_acquisition(self):
        cohort = Cohort("four runs", tuple(member(f"S0{k}", sigma, -0.006)
                                           for k, sigma in enumerate((4.0, 4.5, 5.0, 17.0))))
        checked = cohort.checked()
        failed = {m.subject: [r.leaf for r in m.profile.records if r.failed]
                  for m in checked.members}
        # its noise fails, its other readings stand; the other runs untouched
        assert failed == {"S00": [], "S01": [], "S02": [], "S03": ["noise.sigma"]}

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
        # the member names itself
        assert header["noise.sigma"]["source"] == "subject S01 of a cohort"

    def test_an_own_cohort_joins_the_database(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cohorts, "OWN", tmp_path)
        Cohort("my_site", (member("S01", 6.0, -0.006),)).save(tmp_path / "my_site.json")
        cohorts.forget()
        cohorts.package.cache_clear()
        try:
            assert "my_site" in Cohort.names() and "big_gaba_trio" in Cohort.names()
            assert any(name == "my_site" for name, _, _ in cohorts.rows("noise.sigma"))
        finally:
            cohorts.package.cache_clear()
            cohorts.forget()

    def test_the_label_reads_the_field_to_the_half_tesla(self):
        for field_t, shown in ((1.5, "1.5 T"), (2.89, "3 T"), (7.0, "7 T")):
            cohort = Cohort("c", (member("S01", 6.0, -0.006, field_t=field_t),))
            assert f" {shown} · " in cohort.label()


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
        cohorts.forget()  # the package's pool again for the tests after

    TRIO = [4.0, 4.4, 5.0, 5.5, 6.1]
    GE = [7.0, 7.5, 9.0]

    def database(self, monkeypatch):
        trio = Cohort("trio", tuple(member(f"S{k}", sigma, -0.001 * k)
                                    for k, sigma in enumerate(self.TRIO)) + (
            Member("S9", ScannerProfile(Scanner.preset("siemens_trio_3t"),
                                        (record("noise.sigma", 100.0, failed="a stated rule"),)),
                   PLACE, SIZE),))
        ge = Cohort("ge", tuple(member(f"G{k}", sigma, 0.03, vendor="ge", model=None, field_t=3.0,
                                       channels=8) for k, sigma in enumerate(self.GE)))
        monkeypatch.setattr(cohorts, "package", lambda: (trio, ge))
        cohorts.forget()  # the pool is cached: this database is another

    def test_the_pool_is_the_median_over_the_matching_subjects_without_the_failed(
            self, monkeypatch):
        self.database(monkeypatch)
        pooled = cohorts.pooled("noise.sigma", Scanner.preset("siemens_trio_3t"), channels=32)
        assert (pooled.value == np.median(self.TRIO)
                and pooled.prior == tuple(np.round(np.percentile(self.TRIO, (5, 95)), 7)))
        assert pooled.measured["subjects"] == 5 and pooled.measured["matched_on"] == {
            "field_t": 2.895, "vendor": "siemens", "model": "TrioTim", "channels": 32}
        assert pooled.scanner == {"vendor": "siemens", "model": "TrioTim", "software": None,
                                  "field_t": 2.89}
        assert (pooled.source["dataset"] == "pooled over 5 subjects of 1 cohort: trio"
                and pooled.sequence["family"] == "Press")

    def test_the_pool_is_as_specific_as_the_scanner(self, monkeypatch):
        self.database(monkeypatch)
        everything = cohorts.pooled("noise.sigma", Scanner.preset("generic_3t"))
        assert (everything.measured["subjects"] == 8
                and everything.measured["matched_on"] == {"field_t": 3.0})
        # each dataset a cluster: the two disagree far beyond their sizes, so each takes about
        # half the draws and a GE subject (three of them) weighs more than a Trio one (five)
        share = everything.measured["dataset_share"]
        weights = dict(zip(everything.measured["values"], everything.measured["weights"]))
        assert share["trio"] == pytest.approx(0.5, abs=0.05) and weights[7.0] > weights[4.0]
        population = Empirical(everything.measured["values"], everything.measured["weights"])
        assert (everything.value == round(population.value, 7)
                and everything.value > np.median(self.TRIO + self.GE)
                and everything.scanner["vendor"] == "pooled")
        assert everything.prior == (round(population.low, 7), round(population.high, 7))
        # rows without a model do not fail the model
        ge = cohorts.pooled("noise.sigma", Scanner.preset("ge_mr750_3t"), channels=8)
        # the range over fewer than five
        assert (ge.measured["subjects"] == 3 and ge.prior == (7.0, 9.0)
                and ge.scanner["model"] == "unstated")
        # nothing at 32 channels: the count is dropped
        many = cohorts.pooled("noise.sigma", Scanner.preset("ge_mr750_3t"), channels=32)
        assert (many.measured["subjects"] == 3
                and many.measured["matched_on"] == {"field_t": 3.001, "vendor": "ge"})
        # no vendor rows: every system at the field
        philips = cohorts.pooled("noise.sigma", Scanner.preset("philips_achieva_3t"))
        assert (philips.measured["subjects"] == 8
                and philips.measured["matched_on"] == {"field_t": 3.0})
        # nothing at the field: everything, the departure names it
        low = cohorts.pooled("noise.sigma", Scanner.preset("generic_1p5t"))
        assert low.measured["subjects"] == 8 and low.measured["matched_on"] == {}
        assert cohorts.pooled("conc.Cr", Scanner.preset("generic_3t")) is None

    def test_datasets_weigh_by_size_when_they_agree_and_alike_when_they_disagree(self):
        rng = np.random.default_rng(0)
        agree = {"big": list(5.0 + rng.normal(0, 1, 100)), "small": list(5.0 + rng.normal(0, 1, 4))}
        _, between, share = cohorts._clusters(agree)
        assert between < 0.2 and share["big"] > 0.9               # one population: by size
        apart = {"big": list(5.0 + rng.normal(0, 1, 100)), "small": list(9.0 + rng.normal(0, 1, 4))}
        _, between, share = cohorts._clusters(apart)
        assert between > 4.0 and 0.4 < share["small"] < 0.6        # two populations: alike

    def test_the_person_is_a_condition_of_what_the_subject_carries(self, monkeypatch, tmp_path):
        def person(name, t2, **traits):
            carried = dataclasses.replace(record("water.t2_s.wm", t2), carried_by="subject")
            return Member(name, ScannerProfile(Scanner.preset("siemens_trio_3t"),
                                               (carried, record("noise.sigma", 5.0))),
                          PLACE, SIZE, traits=traits)
        people = Cohort("people", (
            person("A", 0.080, age_y=30.0, sex="F", health="healthy"),
            person("B", 0.084, age_y=70.0, sex="M", health="healthy"),
            person("C", 0.090, age_y=9.0, sex="F", health="healthy"),
            person("D", 0.095, age_y=45.0, sex="F", health="multiple sclerosis"),
            person("E", 0.082, age_y=(18.0, None), health="healthy"),  # "healthy adults" alone
            person("F", 0.200)))                                        # nothing stated
        people.save(tmp_path / "people.json")
        people = Cohort.load(tmp_path / "people.json")
        assert people.members[4].traits == {"age_y": (18.0, None), "health": "healthy"}
        monkeypatch.setattr(cohorts, "package", lambda: (people,))
        cohorts.forget()
        trio = Scanner.preset("siemens_trio_3t")

        def values(**subject):
            return cohorts.pooled("water.t2_s.wm", trio, subject=subject).measured["values"]
        # unasked, a healthy adult: the child, the patient and the unstated stay out
        assert values() == [0.080, 0.082, 0.084]
        assert values(age_y=(60.0, None)) == [0.084] and values(sex="F") == [0.080]
        assert values(health="multiple sclerosis") == [0.095]
        # asked nothing at all: everyone
        assert values(age_y=None, health=None) == [0.080, 0.082, 0.084, 0.090, 0.095, 0.200]
        # an age no one holds gives way, named; a condition no one holds never does
        old = cohorts.pooled("water.t2_s.wm", trio, subject={"age_y": (90.0, None)})
        assert old.measured["subject_dropped"] == ["age_y"] and old.measured["subjects"] == 4
        assert cohorts.pooled("water.t2_s.wm", trio, subject={"health": "glioma"}) is None
        # the scanner's rows ask nothing of the person
        noise = cohorts.pooled("noise.sigma", trio, subject={"health": "glioma"})
        assert noise.measured["subjects"] == 6 and "subject" not in noise.measured

    def test_a_map_row_of_an_unfaithful_method_joins_only_when_asked_or_alone(self, monkeypatch):
        def mapped(name, t1, method, model="Prisma", age_y=30.0):
            row = dataclasses.replace(record("water.t1_s.wm", t1, model=model),
                                      carried_by="subject",
                                      sequence={"name": "a map", "family": method})
            return Member(name, ScannerProfile(Scanner.preset("siemens_prisma_3t"), (row,)),
                          PLACE, SIZE, traits={"age_y": age_y, "health": "healthy"})
        inversion, mp2rage = mapped("A", 0.85, "inversion recovery"), mapped("B", 0.82, "MP2RAGE")
        flip = mapped("C", 1.03, "variable flip angle")
        prisma = Scanner.preset("siemens_prisma_3t")

        def pool(*members, **asked):
            monkeypatch.setattr(cohorts, "package", lambda: (Cohort("maps", members),))
            cohorts.forget()
            return cohorts.pooled("water.t1_s.wm", prisma, **asked).measured
        # the faithful methods by default, named
        faithful = pool(inversion, mp2rage, flip)
        assert (faithful["values"] == [0.82, 0.85] and not faithful["method_dropped"]
                and faithful["matched_on"]["method"] == cohorts.FAITHFUL["water.t1_s"])
        # another method asked by name, or any
        assert pool(inversion, mp2rage, flip, method="variable flip angle")["values"] == [1.03]
        assert pool(inversion, mp2rage, flip, method=())["values"] == [0.82, 0.85, 1.03]
        # no faithful row of an adult: the method gives way, named, before the person's age
        alone = pool(flip, mapped("D", 1.20, "MP2RAGE", age_y=9.0))
        assert (alone["values"] == [1.03] and alone["method_dropped"]
                and alone["methods"] == ["variable flip angle"])
        # the model gives way before the method
        trio = pool(flip, mapped("E", 0.84, "inversion recovery", model="TrioTim"))
        assert trio["values"] == [0.84] and "model" not in trio["matched_on"]
        # a leaf without a faithful table asks no method
        self.database(monkeypatch)
        assert "methods" not in cohorts.pooled("noise.sigma", prisma).measured

    def test_a_tissues_relaxation_asks_its_field_and_no_vendor(self, monkeypatch):
        def mapped(name, t2, vendor, model, field_t):
            row = dataclasses.replace(
                record("water.t2_s.wm", t2, vendor=vendor, model=model, field_t=field_t),
                carried_by="subject",
                sequence={"name": "a map", "family": "EPG multi-echo spin echo"})
            return Member(name, ScannerProfile(Scanner.preset("siemens_prisma_3t"), (row,)),
                          PLACE, SIZE, traits={"age_y": 30.0, "health": "healthy"})
        monkeypatch.setattr(cohorts, "package", lambda: (Cohort("maps", (
            mapped("A", 0.072, "siemens", "Prisma", 2.89),
            mapped("B", 0.090, "philips", "Achieva", 1.5))),))
        cohorts.forget()
        # a Philips at 3 T takes the Siemens row at its field: no tissue depends on the vendor
        philips = cohorts.pooled("water.t2_s.wm", Scanner.preset("philips_achieva_3t")).measured
        assert philips["values"] == [0.072] and "vendor" not in philips["matched_on"]
        # no row at 7 T: the field never gives way, the leaf stays with the literature
        assert cohorts.pooled("water.t2_s.wm", Scanner.preset("generic_7t")) is None

    def test_the_database_is_one_table_a_row_per_reading(self, monkeypatch):
        self.database(monkeypatch)
        rows = cohorts.table()
        # every reading, the failed one with its reason
        assert len(rows) == 2 * 5 + 1 + 2 * 3 and rows["failed"].ne("").sum() == 1
        noise = rows.query("leaf == 'noise.sigma' and failed == '' and vendor == 'siemens'")
        assert sorted(noise["value"]) == self.TRIO and set(noise["dataset"]) == {"trio"}

    def test_the_catalogue_serves_the_pool_and_own_records_outrank_it(self, monkeypatch):
        self.database(monkeypatch)
        served = calibrations("siemens_trio_3t", channels=32)
        assert served["noise.sigma"] == cohorts.pooled(
            "noise.sigma", Scanner.preset("siemens_trio_3t"), channels=32)
        assert (served["water.adjusted_residual"].measured["subjects"] == 5
                and served["conc.Cr"].source["dataset"].startswith("Big GABA"))
        own = record("noise.sigma", 8.0)
        assert calibrations("siemens_trio_3t", records=(own,), channels=32)["noise.sigma"] is own

    #***********************#
    #   the package's own   #
    #***********************#
    def test_the_package_cohorts_are_the_records(self):
        # every record of a 3 T scanner is the median over the package's rows read under its
        # conditions: the plain median where one dataset serves, and where several do (a
        # tissue's relaxation, read on every scanner) the datasets' weighted median, which
        # lies between the datasets' own
        assert "big_gaba_trio" in Cohort.names()
        trio = Scanner.preset("siemens_trio_3t")
        served = ScannerProfile.preset("siemens_trio_3t").calibrations()
        for leaf in cohorts.leaves():
            # what the pool matched, as specific as it could, the person too for the rows the
            # subject carries
            on, person = served[leaf].measured["matched_on"], served[leaf].measured.get("subject")
            matching = [r.value for c in cohorts.package() for m in c.members
                        for r in m.profile.records if r.leaf == leaf and not r.failed
                        and cohorts._matches(r, on.get("field_t"), on.get("vendor"),
                                             on.get("model"), on.get("channels"),
                                             on.get("method"))
                        and (person is None or cohorts._suits(m.traits, person))]
            per_cohort = served[leaf].measured["per_cohort_median"].values()
            assert served[leaf].measured["subjects"] == len(matching)
            if len(per_cohort) == 1:
                assert served[leaf].value == round(float(np.median(matching)), 7)
            else:
                assert min(per_cohort) <= served[leaf].value <= max(per_cohort)
        assert served["noise.sigma"].measured["matched_on"]["model"] == trio.model
        failed = [r for c in cohorts.package() for m in c.members for r in m.profile.records
                  if r.failed]
        assert failed and all(r.value not in [row[2].value for row in cohorts.rows(r.leaf)]
                              or r.failed for r in failed)
        assert all(abs(r.value) <= 0.5 * np.pi
                   for _, _, r in cohorts.rows("acquisition.eddy_transient_rad"))
