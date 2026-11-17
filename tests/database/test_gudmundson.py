####################################################################################################
#                                        test_gudmundson.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Gudmundson loader gate: rows come back merged with their study metadata and the filters #
#          narrow them; everything skips cleanly when the submodule is not checked out.            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pytest

from mrsandbox.database.gudmundson import Gudmundson

pytestmark = pytest.mark.skipif(
    not Gudmundson.available(), reason="externals/mrs-database not checked out"
)


#**************************************************************************************************#
#                                       Class TestGudmundson                                       #
#**************************************************************************************************#
class TestGudmundson:

    #************************#
    #   concentration rows   #
    #************************#
    def test_concentration_rows_carry_study_metadata(self):
        rows = Gudmundson().concentrations("NAA")
        assert len(rows) > 100
        assert all(row["Metabolite"] == "NAA" for row in rows)
        assert all("DOI" in row and "Group" in row for row in rows)
        with_field = sum("Tesla" in row for row in rows)
        assert with_field > 0.9 * len(rows)  # a few value rows lack a matching reference row

    #*************#
    #   filters   #
    #*************#
    def test_filters_narrow_the_rows(self):
        db = Gudmundson()
        all_rows = db.concentrations("NAA")
        healthy = db.concentrations("NAA", group="Healthy")
        three_t = db.concentrations("NAA", tesla=3)
        assert 0 < len(healthy) < len(all_rows)
        assert 0 < len(three_t) < len(all_rows)
        assert all(row["Group"] == "Healthy" for row in healthy)
        assert all(row["Tesla"] == "3" for row in three_t)

    #*************#
    #   t2 rows   #
    #*************#
    def test_t2_rows_exist(self):
        rows = Gudmundson().t2("NAA")
        assert len(rows) > 5

    #**********************#
    #   missing tables     #
    #**********************#
    def test_without_the_submodule_or_a_fetch_the_tables_are_unavailable(self, monkeypatch):
        from mrsandbox.resources import Resources

        def refused(name):
            raise FileNotFoundError(name)
        monkeypatch.setattr(Gudmundson, "ROOT", Gudmundson.ROOT / "nope")
        monkeypatch.setattr(Resources, "path", staticmethod(refused))
        assert not Gudmundson.available()
        with pytest.raises(FileNotFoundError):
            Gudmundson().concentrations("NAA")


#**************************************************************************************************#
#                                         Class TestPooling                                        #
#**************************************************************************************************#
class TestPooling:

    #*******************#
    #   pooled values   #
    #*******************#
    def test_pooled_t2_lands_in_the_known_range(self):
        pooled = Gudmundson().pooled_t2("NAA")
        assert pooled.n_studies >= 50
        assert 0.2 < pooled.mean_s < 0.35
        assert pooled.sd_s > 0.0

    #*************#
    #   aliases   #
    #*************#
    def test_aliases_gather_same_molecule_rows(self):
        db = Gudmundson()
        assert db.pooled_t2("Ins").n_studies > 0  # via mI, mIns, Myo
        assert db.pooled_t2("Cr").n_studies >= 50  # via Cre and moiety variants
        assert db.pooled_t2("Ser") is None  # honestly absent, fallback applies upstream

    #*************#
    #   tissues   #
    #*************#
    def test_the_t2_prior_parts_the_tissues_as_their_rows_do(self):
        from mrsandbox import PerTissue, Priors

        db = Gudmundson()
        # at 3 T both tissues have rows: each tissue's leaf is its own pooled mean, white
        # matter's NAA the longer as most studies with both find
        leaf = Priors.gudmundson_t2(("NAA",), tesla=3.0).NAA
        assert isinstance(leaf, PerTissue)
        for tissue in ("GM", "WM"):
            own = db.pooled_t2("NAA", tesla=3.0, tissues=(tissue,)).mean_s
            assert getattr(leaf, tissue.lower()).mean == pytest.approx(own)
        assert leaf.wm.mean > leaf.gm.mean
        # at 9.4 T no white matter row: the field's pool times the ratio at the closest field
        # with white matter rows (7 T)
        assert 9.4 not in db.t2_fields("NAA", tissues=("WM",))
        leaf = Priors.gudmundson_t2(("NAA",), tesla=9.4).NAA
        ratio = (db.pooled_t2("NAA", tesla=7.0, tissues=("WM",)).mean_s
                 / db.pooled_t2("NAA", tesla=7.0).mean_s)
        assert leaf.wm.mean == pytest.approx(ratio * db.pooled_t2("NAA", tesla=9.4).mean_s)
        # no row anywhere: the stated fallback, one value for every tissue
        assert Priors.gudmundson_t2(("Ser",), tesla=3.0).Ser.value == pytest.approx(0.160)


#**************************************************************************************************#
#                                      Class TestDiseasePriors                                     #
#**************************************************************************************************#
#                                                                                                  #
# Clinical cohorts pool into priors; disease values differ from healthy the way the field knows.   #
#                                                                                                  #
#**************************************************************************************************#
class TestDiseasePriors:

    #*************#
    #   cohorts   #
    #*************#
    def test_healthy_pool_lands_in_the_de_graaf_range(self):
        pooled = Gudmundson().pooled_concentration("NAA")
        assert pooled.n_studies > 20
        assert 7.5 < pooled.mean_s < 12.0  # inside the de Graaf healthy range

    def test_disease_cohorts_shift_the_prior(self):
        db = Gudmundson()
        healthy = db.pooled_concentration("NAA")
        ms = db.pooled_concentration("NAA", groups="Multiple Sclerosis")
        assert ms is not None and ms.mean_s < healthy.mean_s  # the classic NAA reduction

    #*****************#
    #   composition   #
    #*****************#
    def test_priors_compose_over_the_healthy_defaults(self):
        from mrsandbox import Normal, Params, Priors

        healthy = Priors.de_graaf(metabolites=("NAA", "Cr", "Ser"))
        disease = Priors.gudmundson_concentrations(("NAA", "Cr", "Ser"),
                                                   groups="Multiple Sclerosis")
        composed = Params(conc=healthy) | Params(conc=disease)
        assert isinstance(composed.conc.NAA, Normal)  # overridden by the cohort pool
        assert composed.conc.Ser == healthy.Ser  # no rows: the healthy default stands
