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
import numpy as np
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
        assert db.t1() == 1.300

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


#**************************************************************************************************#
#                                      Class TestFieldPriors                                       #
#**************************************************************************************************#
#                                                                                                  #
# The default relaxation priors follow the scanner field, falling back to 3 T where a field has    #
# no rows.                                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestFieldPriors:

    #***************#
    #   7 tesla     #
    #***************#
    def test_seven_tesla_priors_follow_the_typed_sources(self):
        from mrsandbox.scan.defaults import default_params

        p3, p7 = default_params(b0_t=3.0), default_params(b0_t=7.0)
        assert p7.t2.NAA.mean < p3.t2.NAA.mean  # the Gudmundson 7 T pool: shorter T2
        assert p7.t1.NAA.mean > p3.t1.NAA.mean  # Xin 2013: longer T1
        assert abs(p7.t1.Tau.mean - 2.11) < 0.05  # Xin: 2.09 WM and 2.15 GM pooled
        assert p7.water.t1_s.gm.value.mean == 2.00 and p7.water.t1_s.wm.value.mean == 1.55
        assert p7.water.t2_s.gm.value == 0.047 and p7.water.t2_s.csf == p3.water.t2_s.csf
        assert p7.mm.t1.mm091.value == 0.420  # the one component with a 7 T human row
        assert p7.mm.t1.mm121 == p3.mm.t1.mm121 and p7.mm.t2 == p3.mm.t2
        assert p7.t2.GPC == p3.t2.GPC  # no rows at any field: the fixed fallback

    #*****************#
    #   1.5 tesla     #
    #*****************#
    def test_one_and_a_half_tesla_priors_and_the_three_tesla_fallback(self):
        from mrsandbox.scan.defaults import default_params

        p3, p15 = default_params(b0_t=3.0), default_params(b0_t=1.5)
        assert p15.t1.NAA.mean < p3.t1.NAA.mean  # Ethofer 1.5 T rows and Rutgers 2003
        assert p15.water.t1_s.wm.value.mean == 0.68 and p15.water.t1_s.gm.value.mean == 1.04
        assert p15.water.t2_s == p3.water.t2_s  # no 1.5 T water T2 typed: 3 T reused, stated
        assert p15.t2.Gln == p3.t2.Gln  # no 1.5 T rows for Gln: the 3 T pool
        assert p15.t2.NAA.mean != p3.t2.NAA.mean  # NAA has 1.5 T rows of its own
        assert p15.mm.t1.mm091.value == 0.220 and p15.mm.t1.mm121.value == 0.200  # Kreis 2005

    #*********************#
    #   scanner drives    #
    #*********************#
    def test_scans_take_the_field_from_the_scanner(self):
        import numpy as np

        from mrsandbox import Brain, Press, Scanner, VirtualScan
        from mrsandbox.scan.defaults import default_params

        shape = (4, 4, 4)
        brain = Brain.from_maps(gm=np.ones(shape), wm=np.zeros(shape), csf=np.zeros(shape),
                                affine=np.eye(4))
        press = Press(te1_s=0.011, te2_s=0.019)
        at_7t = VirtualScan(brain, Scanner.preset("generic_7t"), press)
        assert at_7t.params.t1.NAA == default_params(b0_t=7.0).t1.NAA
        assert VirtualScan.default_params().t1.NAA == default_params(b0_t=3.0).t1.NAA
        # any other field takes the nearest typed one
        assert default_params(b0_t=4.0).t1.NAA == default_params(b0_t=3.0).t1.NAA
        assert default_params(b0_t=9.4).t1.NAA == default_params(b0_t=7.0).t1.NAA

    #*******************#
    #   lineshape       #
    #*******************#
    def test_the_microscopic_width_scales_with_the_field(self):
        from mrsandbox import Uniform
        from mrsandbox.scan.defaults import default_params

        assert default_params(b0_t=3.0).lineshape.gauss_hz == Uniform(2.0, 6.0)
        assert default_params(b0_t=7.0).lineshape.gauss_hz == Uniform(2.0 * 7.0 / 3.0, 6.0 * 7.0 / 3.0)


#**************************************************************************************************#
#                                        Class TestShifts                                          #
#**************************************************************************************************#
#                                                                                                  #
# The shifts move with temperature and pH by the published functions, and agree with the older     #
# paper's slopes of the differences within a metabolite.                                           #
#**************************************************************************************************#
class TestShifts:

    def test_the_typed_conditions_change_nothing(self):
        from mrsandbox import SpinSystem

        database = Database()
        assert database.shift_change("NAA", 2.008, 37.0, 7.0) == 0.0
        assert database.shift_change("GPC", 3.212, 32.0, 6.5) == 0.0  # not in the table
        assert database.shift_change("NAA", 5.0, 32.0, 7.0) == 0.0    # no group near 5 ppm
        warm, typed = SpinSystem.from_database("Glu", temperature_c=37.0), SpinSystem.from_database("Glu")
        assert [s.shifts_ppm for s in warm] == [s.shifts_ppm for s in typed]

    def test_the_functions_carry_the_2017_difference_slopes(self):
        # Wermter et al. 2017 (MAGMA 30:579) Table 1, measured on its own: the slopes
        # of shift differences within a metabolite, in 1e-4 ppm per kelvin
        database = Database()

        def slope(metabolite, upper_ppm, lower_ppm):
            change = [database.shift_change(metabolite, upper_ppm, t, 7.0)
                      - database.shift_change(metabolite, lower_ppm, t, 7.0) for t in (32.0, 40.0)]
            return (change[1] - change[0]) / 8.0 * 1e4

        assert slope("Cr", 3.913, 3.027) == pytest.approx(-6.2166, abs=1.5)
        assert slope("PCr", 3.930, 3.029) == pytest.approx(-6.6944, abs=1.5)
        assert slope("Lac", 4.0974, 1.3142) == pytest.approx(-3.1866, abs=1.0)
        assert slope("Tau", 3.4206, 3.2459) == pytest.approx(3.2098, abs=1.0)
        # the GABA 4CH2 row as printed contradicts that table (a misprinted sign, see the
        # card) and sits 0.06 ppm off the typed shift: the match leaves it alone
        assert database.shift_change("GABA", 3.013, 32.0, 7.0) == 0.0

    def test_the_couplings_the_table_holds_move_too(self):
        from mrsandbox import SpinSystem

        database = Database()
        # the function itself at 37 C and pH 7 is the typed coupling to a tenth of a hertz
        from mrsandbox.database.shifts import WERMTER_2026_J

        assert database._titration(WERMTER_2026_J["NAA"]["2-3"], 37.0, 7.0) == pytest.approx(3.861, abs=0.12)
        assert database.coupling_change("NAA", 4.3817, 2.6727, 37.0, 7.0) == 0.0
        assert database.coupling_change("Glu", 3.7433, 2.0375, 32.0, 7.0) == 0.0  # not in the table
        assert database.coupling_change("NAA", 7.8205, 4.3817, 32.0, 7.0) == 0.0  # the NH's is not
        moved = database.coupling_change("NAA", 4.3817, 2.6727, 32.0, 6.8)
        assert 0.01 < abs(moved) < 0.2
        typed = SpinSystem.from_database("NAA")[1]
        cold = SpinSystem.from_database("NAA", temperature_c=32.0, ph=6.8)[1]
        assert cold.j_hz[1][2] == pytest.approx(typed.j_hz[1][2] + moved)
        # GABA's one function serves all four 2-3 couplings
        gaba = SpinSystem.from_database("GABA", temperature_c=32.0)[0]
        changed = np.asarray(gaba.j_hz) - np.asarray(SpinSystem.from_database("GABA")[0].j_hz)
        assert np.count_nonzero(np.triu(changed)) == 4

    def test_a_scan_moves_its_systems_and_its_water_with_the_temperature(self):
        from mrsandbox import Brain, Press, Scanner, VirtualScan

        brain = Brain.from_maps(*(3 * [[[[1.0]]]]), affine=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                                          [0, 0, 0, 1]])
        pieces = (brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        fever, body = VirtualScan(*pieces, temperature_c=40.0, ph=6.8), VirtualScan(*pieces)
        assert fever.params.water.temperature_c.value == 40.0
        assert body.params.water.temperature_c.value == 37.0
        moved = fever.systems("Lac")[0].shifts_ppm[0] - body.systems("Lac")[0].shifts_ppm[0]
        assert abs(moved - Database().shift_change("Lac", 4.0974, 40.0, 6.8)) < 1e-12 and moved != 0.0
