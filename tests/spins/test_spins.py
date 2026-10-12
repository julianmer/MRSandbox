####################################################################################################
#                                          test_spins.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Operator-algebra gate: commutation relations, coherence orders, Hamiltonian Hermiticity #
#          and field scaling, thermal scaling - the constants every simulation propagates must be  #
#          exactly right.                                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import SpinSystem


#*********************#
#   lactate fixture   #
#*********************#
@pytest.fixture(scope="module")
def lac():
    (system,) = SpinSystem.from_database("Lac")
    return system


#**************************************************************************************************#
#                                       Class TestOperators                                        #
#**************************************************************************************************#
class TestOperators:

    #*****************#
    #   commutation   #
    #*****************#
    def test_single_spin_commutation_relations(self, lac):
        op = lac.operators()
        for i in range(lac.n):
            np.testing.assert_allclose(op.ix[i] @ op.iy[i] - op.iy[i] @ op.ix[i], 1j * op.iz[i],
                                       atol=1e-6)

    #******************#
    #   independence   #
    #******************#
    def test_different_spins_commute(self, lac):
        op = lac.operators()
        comm = op.ix[0] @ op.iy[1] - op.iy[1] @ op.ix[0]
        np.testing.assert_allclose(comm, np.zeros_like(comm), atol=1e-6)

    #*********************#
    #   coherence order   #
    #*********************#
    def test_coherence_order_matrix(self, lac):
        op = lac.operators()
        np.testing.assert_array_equal(op.coherence_order, -op.coherence_order.T)
        assert op.coherence_order.max() == lac.n and op.coherence_order.min() == -lac.n
        support = np.abs(op.fplus) > 1e-9
        # F+ pairs with (detects) the -1 coherences
        assert np.all(op.coherence_order[support] == +1)

    #**********************#
    #   transverse basis   #
    #**********************#
    def test_hadamard_diagonalizes_the_transverse_operators(self, lac):
        # H Iz_i H = Ix_i for every spin (H unitary), so ideal pulses rotate about any
        # transverse axis with the per-spin mz rows as eigenvalues and no
        # eigendecomposition in the graph
        op = lac.operators()
        h = op.hadamard
        np.testing.assert_allclose(h @ h.conj().T, np.eye(lac.dim), atol=1e-6)
        for i in range(lac.n):
            np.testing.assert_allclose(op.mz[i], np.real(np.diag(op.iz[i])), atol=1e-6)
            np.testing.assert_allclose(h @ np.diag(op.mz[i]) @ h.conj().T, op.ix[i], atol=1e-6)

    #*************#
    #   caching   #
    #*************#
    def test_relaxed_variants_share_one_cached_stack(self, lac):
        # relaxation never changes the operator algebra, so per-draw relaxed copies
        # must hit the structural system's cache entry instead of growing it
        base = lac.operators()
        for i in range(8):
            assert lac.with_relaxation(t1_s=1.0 + i, t2_s=0.1 + i).operators() is base


#**************************************************************************************************#
#                                       Class TestSpinSystem                                       #
#**************************************************************************************************#
class TestSpinSystem:

    #***********************************#
    #   hermiticity and field scaling   #
    #***********************************#
    def test_hamiltonian_hermitian_and_scales_with_field(self, lac):
        h3, h7 = lac.hamiltonian(3.0), lac.hamiltonian(7.0)
        np.testing.assert_allclose(h3, h3.conj().T, atol=1e-4)
        op = lac.operators()
        project = lambda h: np.real(np.trace(h @ op.iz[0])) / np.real(np.trace(op.iz[0] @ op.iz[0]))
        assert abs(project(h7) / project(h3) - 7.0 / 3.0) < 1e-3  # shifts scale with B0, J does not

    #*********************#
    #   thermal scaling   #
    #*********************#
    def test_thermal_rho_carries_the_scale(self):
        (scyllo,) = SpinSystem.from_database("sIns")
        np.testing.assert_allclose(scyllo.thermal_rho(), 6.0 * scyllo.operators().fz, atol=0)


#**************************************************************************************************#
#                                      Class TestHigherSpin                                        #
#**************************************************************************************************#
#                                                                                                  #
# The operator algebra of any spin I: deuterium (1) and sodium (3/2) beside the proton (1/2).      #
#                                                                                                  #
#**************************************************************************************************#
class TestHigherSpin:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def single(nucleus, spin, shift_ppm=0.0):
        from mrsandbox.database import Database

        return SpinSystem(name=nucleus, nucleus=nucleus, shifts_ppm=(shift_ppm,), j_hz=((0.0,),),
                          scale=1.0, gamma_mhz_t=Database().gamma(nucleus), spin=spin)

    #***************#
    #   algebra     #
    #***************#
    @pytest.mark.parametrize("nucleus, spin", [("1H", 0.5), ("2H", 1.0), ("23Na", 1.5)])
    def test_operators_obey_the_spin_algebra(self, nucleus, spin):
        # [Ix, Iy] = i Iz, the Casimir I (I + 1), the m ladder of Iz, and the
        # transverse eigenbasis H diag(m) H^H = Ix, on the (2 I + 1) levels
        system = self.single(nucleus, spin)
        op = system.operators()
        levels = int(round(2 * spin + 1))
        assert system.dim == levels and op.ix.shape == (1, levels, levels)
        ix, iy, iz = op.ix[0], op.iy[0], op.iz[0]
        np.testing.assert_allclose(ix @ iy - iy @ ix, 1j * iz, atol=1e-6)
        np.testing.assert_allclose(ix @ ix + iy @ iy + iz @ iz,
                                   spin * (spin + 1.0) * np.eye(levels), atol=1e-6)
        np.testing.assert_allclose(np.real(np.diag(iz)), spin - np.arange(levels), atol=1e-6)
        h = op.hadamard
        np.testing.assert_allclose(h @ h.conj().T, np.eye(levels), atol=1e-6)
        np.testing.assert_allclose(h @ np.diag(op.mz[0]) @ h.conj().T, ix, atol=1e-6)
        assert op.coherence_order.max() == int(round(2 * spin))

    #*****************#
    #   two spins     #
    #*****************#
    def test_two_deuterons_share_the_product_space(self):
        from mrsandbox.database import Database

        pair = SpinSystem(name="pair", nucleus="2H", shifts_ppm=(1.0, 2.0),
                          j_hz=((0.0, 0.0), (0.0, 0.0)), scale=1.0,
                          gamma_mhz_t=Database().gamma("2H"), spin=1.0)
        op = pair.operators()
        assert pair.dim == 9
        comm = op.ix[0] @ op.iy[1] - op.iy[1] @ op.ix[0]
        np.testing.assert_allclose(comm, np.zeros_like(comm), atol=1e-6)
        for i in range(2):
            np.testing.assert_allclose(op.hadamard @ np.diag(op.mz[i]) @ op.hadamard.conj().T,
                                       op.ix[i], atol=1e-6)

    #*****************#
    #   amplitudes    #
    #*****************#
    def test_amplitude_per_spin_follows_i_times_i_plus_one(self):
        # 2 Fz / d is the thermal state of every spin, so the detected amplitude of
        # one spin scales as I (I + 1): 8/3 for deuterium and 5 for sodium against
        # the proton (the polarization and detection gain of a nucleus stay outside)
        from keras import ops

        from mrsandbox import Engine, PulseAcquire

        def detected(nucleus, spin):
            engine = Engine(3.0)
            _, amps = engine.lines(PulseAcquire(), self.single(nucleus, spin))
            return float(np.abs(np.asarray(ops.convert_to_numpy(amps))).sum())

        proton = detected("1H", 0.5)
        assert abs(detected("2H", 1.0) / proton - 8.0 / 3.0) < 1e-5
        assert abs(detected("23Na", 1.5) / proton - 5.0) < 1e-5

    #***********#
    #   dmi     #
    #***********#
    def test_dmi_spectrum_shows_the_four_deuterated_resonances(self):
        # the typed deuterium set (water, glucose, Glx, lactate) renders at its in
        # vivo positions, as spin 1 with its 11.7 T T1
        from keras import ops

        from mrsandbox import Engine, PulseAcquire

        systems = [SpinSystem.from_database(name, source="dmi")[0]
                   for name in ("Water", "Glc", "Glx", "Lac")]
        assert all(s.nucleus == "2H" and s.spin == 1.0 and s.dim == 3 for s in systems)
        assert systems[1].t1_s == 0.064
        ppm, spec = Engine(11.7).spectrum(PulseAcquire(), *systems, npoints=4096, sw_hz=2000.0)
        real = np.real(np.asarray(ops.convert_to_numpy(spec)))
        peaks = np.sort(np.asarray(ppm)[np.argsort(real)[-4:]])
        np.testing.assert_allclose(peaks, [1.4, 2.4, 3.8, 4.8], atol=0.01)


#**************************************************************************************************#
#                                       Class TestHeteronuclei                                     #
#**************************************************************************************************#
#                                                                                                  #
# A spin of another nucleus in a proton system: its own levels, never pulsed nor detected.         #
#                                                                                                  #
#**************************************************************************************************#
class TestHeteronuclei:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def pair(other, j_hz=7.0):
        # a proton at 3.2 ppm coupled to one spin: of another nucleus, or a proton at 100 ppm
        # (the way the table wrote the heteronuclei before they were named)
        from mrsandbox.database import Database

        return SpinSystem(name="pair", nucleus="1H", shifts_ppm=(3.2, 100.0),
                          j_hz=((0.0, j_hz), (j_hz, 0.0)), scale=1.0,
                          gamma_mhz_t=Database().gamma("1H"),
                          nuclei=None if other == "1H" else ("1H", other))

    @staticmethod
    def detected(sequence, *systems):
        from keras import ops

        from mrsandbox import Engine

        freqs, amps = Engine(3.0).lines(sequence, *systems)
        return (np.asarray(ops.convert_to_numpy(freqs)),
                np.asarray(ops.convert_to_numpy(amps)).astype(np.complex128))

    #**************#
    #   database   #
    #**************#
    def test_the_choline_compounds_name_their_nitrogen_and_phosphorus(self):
        # the table's protons at 100 and 40 ppm are the 14N (spin 1) and 31P of the compounds
        systems = {s.name: s for m in ("Cho", "PCho") for s in SpinSystem.from_database(m)}
        methyl = systems["Cho.Ch_3CH3"]
        assert methyl.nuclei == ("1H", "1H", "1H", "14N")
        assert methyl.levels == (2, 2, 2, 3) and methyl.dim == 24
        ring = systems["PCho.PCh_ch2ch2"]
        assert ring.nuclei[-2:] == ("14N", "31P") and ring.dim == 2 ** 5 * 3
        assert all(s.nuclei is None for s in SpinSystem.from_database("NAA"))

    #*************#
    #   triplet   #
    #*************#
    def test_the_choline_methyl_is_a_triplet_on_its_nitrogen(self):
        # three lines J apart in 1:1:1, every one a proton line, the protons' total the
        # uncoupled methyl's: the nitrogen adds its levels and nothing of its own
        from mrsandbox import PulseAcquire

        (methyl,) = [s for s in SpinSystem.from_database("Cho") if s.name == "Cho.Ch_3CH3"]
        freqs, amps = self.detected(PulseAcquire(), methyl)
        keep = np.abs(amps) > 1e-6 * np.abs(amps).max()
        lines = {}
        for f, a in zip(np.round(freqs[keep], 3), amps[keep]):
            lines[f] = lines.get(f, 0.0) + a
        positions = np.sort(np.asarray(list(lines)))
        np.testing.assert_allclose(np.diff(positions), 0.57, atol=1e-3)
        heights = np.abs(np.asarray([lines[f] for f in positions]))
        np.testing.assert_allclose(heights / heights.mean(), 1.0, atol=1e-5)
        bare = SpinSystem(name="bare", nucleus="1H", shifts_ppm=(3.185,), j_hz=((0.0,),),
                          scale=3.0 * methyl.scale, gamma_mhz_t=methyl.gamma_mhz_t)
        _, single = self.detected(PulseAcquire(), bare)
        assert abs(heights.sum() - np.abs(single.sum())) < 1e-5 * np.abs(single.sum())

    #************#
    #   echoes   #
    #************#
    @pytest.mark.parametrize("te_s", [0.03, 0.05, 0.1])
    def test_a_proton_echo_refocuses_a_heteronuclear_coupling(self, te_s):
        # the 180s turn the proton alone, so J Iz Sz refocuses at the echo top like a shift:
        # the coupled proton's signal there is the uncoupled one's at any echo time, against
        # cos(pi J TE) and worse when the partner is a proton the pulses turn too
        from mrsandbox import Press

        press = Press(te1_s=te_s / 2.0, te2_s=te_s / 2.0)
        bare = SpinSystem(name="bare", nucleus="1H", shifts_ppm=(3.2,), j_hz=((0.0,),),
                          scale=1.0, gamma_mhz_t=self.pair("31P").gamma_mhz_t)
        f0_mhz = 3.0 * bare.gamma_mhz_t

        def near(system):
            # the proton's lines at 3.2 ppm (a proton partner has its own at 100)
            freqs, amps = self.detected(press, system)
            return amps[np.abs(freqs - 3.2 * f0_mhz) < 50.0].sum()

        reference = near(bare)
        for other in ("31P", "14N"):
            assert abs(near(self.pair(other)) - reference) < 1e-5 * abs(reference)
        assert abs(abs(near(self.pair("1H"))) - abs(reference)) > 0.05 * abs(reference)


#**************************************************************************************************#
#                                      Class TestConnectedParts                                    #
#**************************************************************************************************#
#                                                                                                  #
# Spins no coupling connects are their own systems, and the split changes no spectrum.             #
#                                                                                                  #
#**************************************************************************************************#
class TestConnectedParts:

    def test_uncoupled_spins_split_and_merge_by_shift(self):
        # creatine's record is six uncoupled spins: three one-spin systems counted
        # 1, 2, and 3 times instead of one 64-dimensional system
        parts = SpinSystem.from_database("Cr")
        assert sorted((s.shifts_ppm, s.scale) for s in parts) == [
            ((3.027,), 3.0), ((3.913,), 2.0), ((6.649,), 1.0)]
        # a coupled group stays whole, its uncoupled amide leaves it
        gsh = {s.name: s for s in SpinSystem.from_database("GSH")}
        assert len(gsh["GSH.cyst.0"].shifts_ppm) == 3 and gsh["GSH.cyst.1"].shifts_ppm == (8.177,)
        assert all(s.dim <= 2 ** 6 or s.name.startswith("Glc")
                   for name in ("NAA", "Cr", "PCr", "GSH")
                   for s in SpinSystem.from_database(name))

    def test_the_split_renders_the_record_as_one_system_would(self):
        from keras import ops

        from mrsandbox import Engine, Press

        engine, sequence = Engine(3.0), Press(te1_s=0.011, te2_s=0.019)

        def spectrum(systems):
            total = 0.0
            for system in systems:
                freqs, amps = engine.lines(sequence, system)
                _, spec = engine.render_fid(freqs, amps, npoints=2048)
                total = total + np.asarray(ops.convert_to_numpy(spec))
            return total

        whole = SpinSystem(name="Cr.whole", nucleus="1H",
                           shifts_ppm=(6.649, 3.913, 3.913, 3.027, 3.027, 3.027),
                           j_hz=tuple((0.0,) * 6 for _ in range(6)), scale=1.0,
                           gamma_mhz_t=42.5774785)
        split, one = spectrum(SpinSystem.from_database("Cr")), spectrum((whole,))
        np.testing.assert_allclose(split, one, atol=1e-4 * np.abs(one).max())
