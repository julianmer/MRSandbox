####################################################################################################
#                                         test_database.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Database gate: tables load through the typed access path, hold the values they claim,   #
#          fail loudly on unknown keys, and the priors builders turn sources into Params subtrees. #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pytest

from mrsandbox import Database, Fixed, Params, Priors, Uniform


#**************************************************************************************************#
#                                        Class TestDatabase                                        #
#**************************************************************************************************#
class TestDatabase:

    #************#
    #   nuclei   #
    #************#
    def test_gamma(self):
        assert abs(Database().gamma("1H") - 42.5774785) < 1e-6
        with pytest.raises(KeyError, match="known:"):
            Database().gamma("42X")

    #******************#
    #   spin systems   #
    #******************#
    def test_metabolites_and_lactate(self):
        db = Database()
        for name in ("NAA", "Cr", "Cho", "Lac"):
            assert name in db.metabolites()
        group = db.spin_system("Lac")["groups"][0]
        assert len(group["spins"]) == 4
        assert sorted(j for *_, j in group["j"]) == [6.933, 6.933, 6.933]

    #********************#
    #   concentrations   #
    #********************#
    def test_de_graaf_concentrations(self):
        db = Database()
        assert db.concentration("NAA") == (7.5, 12.0)
        assert db.concentration("Glu") == (6.0, 12.5)
        assert db.concentration("Cr") == (4.5, 10.5)
        with pytest.raises(KeyError, match="known:"):
            db.concentration("NAA", source="nonsense")

    #****************#
    #   relaxation   #
    #****************#
    def test_simple_relaxation(self):
        db = Database()
        assert db.t2() == 0.160
        assert db.t1("anything") == 1.300

    #*****************#
    #   pooled t1     #
    #*****************#
    def test_pooled_t1_lands_in_the_known_range(self):
        db = Database()
        pooled = db.pooled_t1("NAA")
        assert pooled.n_studies == 10  # 2 Mlynarik + 6 Ethofer (as NA) + 2 Traber
        assert 1.3 < pooled.mean_s < 1.5
        assert pooled.sd_s > 0.0

    def test_pooled_t1_composite_aliases_and_fallback(self):
        db = Database()
        # constituents inherit their composite resonance: GPC and PCho share the
        # choline-compound rows, Cr and PCr share the tCr rows (both moieties)
        assert db.pooled_t1("GPC") == db.pooled_t1("PCho")
        assert db.pooled_t1("Cr") == db.pooled_t1("PCr")
        assert db.pooled_t1("Cr").n_studies == 14
        assert db.pooled_t1("Ser") is None  # honestly absent, fallback applies upstream

    def test_pooled_t1_technique_disagreement_widens_the_prior(self):
        # Traber's inversion recovery and progressive saturation rows disagree for tCr
        # (1.11 vs 1.47 s); the between-study spread must absorb that, not hide it
        db = Database()
        assert db.pooled_t1("Cr").sd_s > db.pooled_t1("Cho").sd_s


#**************************************************************************************************#
#                                         Class TestPriors                                         #
#**************************************************************************************************#
class TestPriors:

    #**************#
    #   de graaf   #
    #**************#
    def test_de_graaf_builds_uniform_leaves(self):
        priors = Priors.de_graaf()
        assert isinstance(priors, Params)
        assert len(priors) == 27
        assert priors.NAA == Uniform(7.5, 12.0)
        truth = priors.sample(seed=1)
        assert 7.5 <= truth.NAA < 12.0

    #***************************#
    #   subset and relaxation   #
    #***************************#
    def test_subset_and_simple_relaxation(self):
        priors = Priors.de_graaf(metabolites=("NAA", "Cr"))
        assert sorted(priors) == ["Cr", "NAA"]
        relax = Priors.simple_relaxation()
        assert relax.t2_s == Fixed(0.160) and relax.t1_s == Fixed(1.300)
