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

    def test_de_graafs_table_covers_every_metabolite_as_printed(self):
        # every metabolite builds from it, a sample reads as the book prints it (Table 2.1), and
        # GPC's heteronuclei carry their nuclei, 14N at spin 1
        from mrsandbox import SpinSystem

        db = Database()
        assert set(db.metabolites()) <= set(db.SPIN_SYSTEMS["degraaf_2019"])
        for name in db.metabolites():
            assert SpinSystem.from_database(name, source="degraaf_2019")
        glu = db.spin_system("Glu", source="degraaf_2019")["groups"][0]
        assert glu["spins"]["2"] == 3.748 and ("3'", "4", 8.77) in glu["j"]
        ser = db.spin_system("Ser", source="degraaf_2019")["groups"][0]
        assert ("2", "3'", 3.56) in ser["j"] and ("3", "3'", -12.3) in ser["j"]
        (choline,) = [s for s in SpinSystem.from_database("GPC", source="degraaf_2019")
                      if s.nuclei and "14N" in s.nuclei]
        assert set(choline.nuclei) == {"1H", "14N", "31P"} and choline.dim == 2 ** 4 * 3 * 2

    def test_govindarajus_table_carries_the_authors_corrections(self):
        # every metabolite but ascorbate (not in the paper) builds; the 2015 corrections are in
        # (taurine's 6.792 Hz at 1'-2', GABA's H2 at 2.2840 ppm, the geminal couplings), choline's
        # CH2 couples to its nitrogen, and PCho's 1-N is the paper's 2.680 Hz
        from mrsandbox import SpinSystem

        db = Database()
        table = db.SPIN_SYSTEMS["govind_2015"]
        assert set(db.metabolites()) - set(table) == {"Asc"}
        for name in table:
            assert SpinSystem.from_database(name, source="govind_2015")
        record = lambda name, k=0: db.spin_system(name, source="govind_2015")["groups"][k]
        assert ("1'", "2'", 6.792) in record("Tau")["j"]
        assert ("1", "1'", -12.438) in record("Tau")["j"]
        assert record("GABA")["spins"]["2"] == 2.2840
        assert ("1", "N", 2.572) in record("Cho", 1)["j"]
        assert ("1", "N", 2.680) in record("PCho")["j"]
        methyl = [s for s in SpinSystem.from_database("Cho", source="govind_2015") if s.n == 2]
        assert methyl[0].nuclei == ("1H", "14N") and methyl[0].scale == 9.0

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
    def test_the_water_t2_is_the_corrected_studies_median_and_each_a_voice_of_the_mix(self):
        from mrsandbox.database.relaxation import WATER_T2_3T
        from mrsandbox.scan.calibrations import calibrations
        from mrsandbox.scan.defaults import default_params

        t2 = default_params(b0_t=3.0, scanner="siemens_prisma_3t").water.t2_s
        # the median of the five stimulated-echo-corrected studies, their spread the range
        assert (t2.literature.gm.value.mean == 0.0805 and t2.literature.wm.value.mean == 0.061
                and 0.005 < t2.literature.wm.value.sd < 0.01)
        # the mix: each study and each measured dataset one voice
        record = calibrations("siemens_prisma_3t")["water.t2_s.wm"]
        datasets = len(set(record.measured["datasets"]))
        assert len(t2.mixed.wm.components) == len(WATER_T2_3T["wm"]) + datasets
        assert t2.chosen == "literature" and "each an equal voice" in t2.provenance["mixed"]

    def test_one_and_a_half_tesla_priors_and_the_three_tesla_fallback(self):
        from mrsandbox.scan.defaults import default_params

        p3, p15 = default_params(b0_t=3.0), default_params(b0_t=1.5)
        assert p15.t1.NAA.mean < p3.t1.NAA.mean  # Ethofer 1.5 T rows and Rutgers 2003
        assert p15.water.t1_s.wm.value.mean == 0.68 and p15.water.t1_s.gm.value.mean == 1.04
        # no 1.5 T water T2 typed: 3 T reused, stated
        assert p15.water.t2_s.literature == p3.water.t2_s.literature
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
        assert default_params(b0_t=7.0).lineshape.gauss_hz == Uniform(
            2.0 * 7.0 / 3.0, 6.0 * 7.0 / 3.0
        )

    #*****************************#
    #   signed eddy field terms   #
    #*****************************#
    def test_the_gradient_eddy_field_terms_carry_either_sign(self):
        # the eddy phase read on the water references shows both signs (scan.calibrations
        # EDDY_PHASE records)
        from mrsandbox import Uniform
        from mrsandbox.scan.defaults import default_params

        gradients = default_params(b0_t=3.0).gradients
        assert (gradients.b0_short_hz_per_mt_m == Uniform(-0.5, 0.5)
                and gradients.b0_long_hz_per_mt_m == Uniform(-0.2, 0.2))

    #********************#
    #   three sources    #
    #********************#
    def test_the_literature_is_chosen_where_it_exists_and_the_pooled_readings_elsewhere(self):
        # every leaf a reading serves is Sourced: the literature (de Graaf's creatine, the stated
        # width) chosen where one is published, the pooled readings where none is (the noise, the
        # residual, the delay, the transient, the leak: a zero for want of a number is not
        # literature), each source with its provenance
        from mrsandbox import Empirical, Sourced, Uniform
        from mrsandbox.scan.defaults import default_params

        tree = default_params(b0_t=3.0, scanner="siemens_trio_3t")
        # magnetization transfer's rates: published, beside the readings of Bacon et al.'s spectra
        transfer = {f"mt.rate.{name}" for name in ("Cr", "PCr", "NAA", "NAAG", "PCho", "GPC")}
        assert set(tree.sources()) == {"conc.Cr", "lineshape.gauss_hz", "noise.sigma",
                                       "water.adjusted_residual", "acquisition.delay_s",
                                       "acquisition.eddy_transient_rad", "oov.fraction", "water.t1_s",
                                       "water.t2_s", "mm.level_mm", "mm.kappa", *(f"run.{key}" for key in (
                                           "drift_hz_per_min", "ar_sigma_hz", "resp_freq_hz",
                                           "resp_amp_hz", "phase_jitter_deg", "amp_jitter_frac",
                                           "broaden_hz", "events_per_min", "motion_mm")),
                                       *transfer}
        assert {path: leaf.chosen for path, leaf in tree.sources().items()
                if not path.startswith("run.")} == {
            "conc.Cr": "literature", "lineshape.gauss_hz": "literature", "noise.sigma": "pooled",
            "water.adjusted_residual": "pooled", "acquisition.delay_s": "pooled",
            "acquisition.eddy_transient_rad": "pooled", "oov.fraction": "pooled",
            "water.t1_s": "literature", "water.t2_s": "literature", "mm.level_mm": "literature", "mm.kappa": "literature",
            **{path: "literature" for path in transfer}}
        # the run: the envelope until read; the head's move has no literature, the rows serve it
        assert all(leaf.chosen == "literature" for path, leaf in tree.sources().items()
                   if path.startswith("run.") and path != "run.motion_mm")
        assert tree.conc.Cr == Uniform(3.0, 5.5) and tree.lineshape.gauss_hz == Uniform(2.0, 6.0)
        assert tree.noise.sigma.literature is None and tree.acquisition.delay_s.literature is None
        # the rows themselves, one subject per draw
        assert (isinstance(tree.acquisition.eddy_transient_rad.pooled, Empirical)
                and isinstance(tree.acquisition.delay_s.pooled, Empirical))
        assert (tree.conc.Cr.provenance["literature"].startswith("de Graaf")
                and "big_gaba_trio" in tree.noise.sigma.provenance["pooled"])
        measured = tree.select("pooled")
        assert measured.conc.Cr.chosen == "pooled" and measured.conc.Cr == tree.conc.Cr.pooled
        assert (isinstance(measured.lineshape.gauss_hz, Sourced)
                and measured.lineshape.gauss_hz.chosen == "pooled")
        literature = tree.select("literature")
        assert (literature.noise.sigma == tree.noise.sigma
                and literature.acquisition.delay_s == tree.acquisition.delay_s)

    #**********************#
    #   eddy transient     #
    #**********************#
    def test_the_eddy_transient_is_the_pooled_rows_spread_about_zero(self):
        # the phase transient at the FID's start: the package's rows of either sign pooled for the
        # scanner, a Normal about zero with their 95th percentile as its spread; the failed readings
        # (a half turn off) are not among them
        import numpy as np

        from mrsandbox import Empirical
        from mrsandbox.scan import cohort
        from mrsandbox.scan.calibrations import calibrations
        from mrsandbox.scan.defaults import default_params

        for scanner in ("siemens_trio_3t", "ge_mr750_3t", "generic_3t"):
            record = calibrations(scanner)["acquisition.eddy_transient_rad"]
            assert record.prior[0] < 0.0 < record.prior[1]
            drawn = default_params(b0_t=3.0, scanner=scanner).acquisition.eddy_transient_rad
            assert (isinstance(drawn.current, Empirical)
                    and (round(drawn.low, 7), round(drawn.high, 7)) == record.prior)
            assert (round(drawn.value, 7) == record.value
                    and list(drawn.current.values) == record.measured["values"])
        assert all(abs(record.value) <= 0.5 * np.pi
                   for _, _, record in cohort.rows("acquisition.eddy_transient_rad"))
        assert any(record.failed for c in cohort.package() for m in c.members
                   for record in m.profile.records
                   if record.leaf == "acquisition.eddy_transient_rad")


#**************************************************************************************************#
#                                     Class TestScannerSurvey                                      #
#**************************************************************************************************#
class TestScannerSurvey:

    def test_the_survey_names_the_scanners_the_presets_stand_for(self):
        from mrsandbox.database.scanners import OPENNEURO_SURVEY

        prisma, trio = (OPENNEURO_SURVEY[("siemens", "Prisma", 3.0)],
                        OPENNEURO_SURVEY[("siemens", "TrioTim", 3.0)])
        assert prisma["datasets"] > 100 and "HeadNeck_64" in prisma["receive_coils"]
        # the reference amplitude: the Prisma's body coil needs three quarters of the Trio's
        assert prisma["tx_ref_amp_v"]["median"] < 0.8 * trio["tx_ref_amp_v"]["median"]
        assert len(prisma["shim_setting"]["median"]) == 8
        assert any("DV24" in release
                   for release in OPENNEURO_SURVEY[("ge", "DISCOVERY MR750", 3.0)]["software"])
        for (vendor, model, field), row in OPENNEURO_SURVEY.items():
            assert vendor in ("siemens", "ge", "philips", "bruker") and row["datasets"] >= 3


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
        warm, typed = (SpinSystem.from_database("Glu", temperature_c=37.0),
                       SpinSystem.from_database("Glu"))
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

        assert database._titration(WERMTER_2026_J["NAA"]["2-3"], 37.0, 7.0) == pytest.approx(
            3.861, abs=0.12)
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

        brain = Brain.from_maps(*(3 * [[[[1.0]]]]),
                                affine=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        pieces = (brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        fever, body = VirtualScan(*pieces, temperature_c=40.0, ph=6.8), VirtualScan(*pieces)
        assert fever.params.water.temperature_c.value == 40.0
        assert body.params.water.temperature_c.value == 37.0
        moved = fever.systems("Lac")[0].shifts_ppm[0] - body.systems("Lac")[0].shifts_ppm[0]
        assert (abs(moved - Database().shift_change("Lac", 4.0974, 40.0, 6.8)) < 1e-12
                and moved != 0.0)
