####################################################################################################
#                                         test_backend.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Substrate gate: the eigh propagator is unitary, trace-preserving, matches a float64     #
#          numpy reference, jit-compiles, and its timing gradients are exact - including at        #
#          degenerate eigenvalues, which equivalent spins make the norm.                           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest
from keras import ops

from nifti_mrs_plus.backend import jit, value_and_grad

from mrsandbox import Backend


#**************************************************************************************************#
#                                       Class TestPropagator                                       #
#**************************************************************************************************#
class TestPropagator:

    #***************#
    #   unitarity   #
    #***************#
    def test_is_unitary(self, random_hermitian):
        u = Backend.expm_herm(ops.convert_to_tensor(random_hermitian(8)), 0.01)
        eye = ops.convert_to_numpy(ops.einsum("ij,kj->ik", u, ops.conj(u)))
        np.testing.assert_allclose(eye, np.eye(8), atol=2e-5)

    #************************#
    #   trace preservation   #
    #************************#
    def test_preserves_trace(self, random_hermitian):
        u = Backend.expm_herm(ops.convert_to_tensor(random_hermitian(8, seed=1)), 0.003)
        rho = ops.convert_to_tensor(random_hermitian(8, seed=2))
        rho_t = ops.einsum("ij,jk,lk->il", u, rho, ops.conj(u))
        traces = [complex(ops.convert_to_numpy(ops.trace(r))) for r in (rho, rho_t)]
        # complex64 einsum accumulation lands near 1e-5 relative on some CPU BLAS builds
        assert abs(traces[0] - traces[1]) < 1e-5 * abs(traces[0])

    #***********************#
    #   float64 reference   #
    #***********************#
    def test_matches_numpy_reference(self, random_hermitian):
        h64 = random_hermitian(8, seed=3).astype(np.complex128)
        lam, v = np.linalg.eigh(h64)
        u_ref = v @ np.diag(np.exp(-2j * math.pi * lam * 0.0042)) @ v.conj().T
        u = ops.convert_to_numpy(
            Backend.expm_herm(ops.convert_to_tensor(h64.astype(np.complex64)), 0.0042)
        )
        np.testing.assert_allclose(u, u_ref, atol=3e-5)

    #****************#
    #   batched dt   #
    #****************#
    def test_batches_over_dt(self, random_hermitian):
        h = ops.convert_to_tensor(random_hermitian(4, seed=4))
        u_batch = ops.convert_to_numpy(Backend.expm_herm(h, np.float32([0.001, 0.002, 0.005])))
        assert u_batch.shape == (3, 4, 4)
        for i, dt in enumerate([0.001, 0.002, 0.005]):
            np.testing.assert_allclose(
                u_batch[i], ops.convert_to_numpy(Backend.expm_herm(h, dt)), atol=2e-5
            )


#**************************************************************************************************#
#                                       Class TestGradients                                        #
#**************************************************************************************************#
class TestGradients:

    #***********************#
    #   finite difference   #
    #***********************#
    def test_grad_wrt_dt_matches_finite_difference(self, random_hermitian, needs_autograd):
        h = ops.convert_to_tensor(random_hermitian(8, seed=5))
        w = ops.convert_to_tensor(np.random.default_rng(6).normal(size=(8, 8)).astype(np.float32))

        def loss(dt):
            u = Backend.expm_herm(h, dt)
            return ops.sum(ops.real(u) * w) + ops.sum(ops.imag(u) * w)

        _, grad = value_and_grad(loss)(ops.convert_to_tensor(np.float32(0.0123)))
        eps = 1e-5
        fd = (
            float(ops.convert_to_numpy(loss(ops.convert_to_tensor(np.float32(0.0123 + eps)))))
            - float(ops.convert_to_numpy(loss(ops.convert_to_tensor(np.float32(0.0123 - eps)))))
        ) / (2 * eps)
        assert abs(float(ops.convert_to_numpy(grad)) - fd) < 2e-2 * max(1.0, abs(fd))

    #****************************#
    #   degenerate eigenvalues   #
    #****************************#
    def test_grad_wrt_dt_at_degenerate_eigenvalues(self, needs_autograd):
        # d/dt cos(2 pi f t) = -2 pi f sin(2 pi f t); degeneracy must not disturb it.
        h = ops.convert_to_tensor(np.diag([10.0, 10.0, 25.0]).astype(np.complex64))

        def loss(dt):
            return ops.real(Backend.expm_herm(h, dt))[0, 0]

        dt0 = 0.013
        value, grad = value_and_grad(loss)(ops.convert_to_tensor(np.float32(dt0)))
        assert abs(float(ops.convert_to_numpy(value)) - math.cos(2 * math.pi * 10.0 * dt0)) < 1e-5
        assert abs(
            float(ops.convert_to_numpy(grad))
            + 2 * math.pi * 10.0 * math.sin(2 * math.pi * 10.0 * dt0)
        ) < 1e-2


#**************************************************************************************************#
#                                          Class TestJit                                           #
#**************************************************************************************************#
class TestJit:

    #*******************#
    #   compile smoke   #
    #*******************#
    def test_compiles_the_propagator(self, random_hermitian, backend):
        if backend == "numpy":
            pytest.skip("jit is identity on numpy")
        h = ops.convert_to_tensor(random_hermitian(4, seed=7))
        rho = ops.convert_to_tensor(random_hermitian(4, seed=8))

        def observable(dt):
            u = Backend.expm_herm(h, dt)
            return ops.real(ops.trace(ops.einsum("ij,jk,lk->il", u, rho, ops.conj(u))))

        dt = ops.convert_to_tensor(np.float32(0.002))
        eager = float(ops.convert_to_numpy(observable(dt)))
        compiled = float(ops.convert_to_numpy(jit(observable)(dt)))
        assert abs(eager - compiled) < 1e-4
