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
