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
            np.testing.assert_allclose(op.ix[i] @ op.iy[i] - op.iy[i] @ op.ix[i], 1j * op.iz[i], atol=1e-6)

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
        assert np.all(op.coherence_order[support] == +1)  # F+ pairs with (detects) the -1 coherences

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
