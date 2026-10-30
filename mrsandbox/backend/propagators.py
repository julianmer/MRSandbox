####################################################################################################
#                                          propagators.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The single escape hatch: everything Keras 3 does not expose or does inconsistently      #
#          (jit, autograd, the Hermitian propagator). No other package may contain a backend-      #
#          conditional line. Complex helpers come from nifti_mrs_plus.ops - one tested copy across #
#          the ecosystem.                                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.backend import precision


#**************************************************************************************************#
#                                           Class Backend                                          #
#**************************************************************************************************#
#                                                                                                  #
# Static namespace for the per-backend primitives the physics is built on.                         #
#                                                                                                  #
#**************************************************************************************************#
class Backend:
    """Static namespace for the propagator primitives the physics is built on.

    House rules: complex64 inside the physics, Hamiltonians in Hz, durations in seconds
    (the exp(-i 2 pi H t) convention). For jit and value_and_grad use
    nifti_mrs_plus.backend, which dispatches at call time on the tensors you pass.
    """

    #**********#
    #   vmap   #
    #**********#
    @staticmethod
    def vmap(fn, in_axes):
        """`fn` mapped over a leading axis of the arguments `in_axes` marks with 0 (None
        passes an argument whole), the outputs stacked on a new leading axis: jax's
        vmap on jax tensors, where the batch axis stays hidden from the traced code and
        one compiled graph serves the whole batch, and a loop over the batch with the
        results stacked everywhere else (the same values, one dispatch per element).
        Dispatches at call time on the tensors passed, like jit."""
        def mapped(*args):
            if any(nops.is_jax(arg) for arg in args):
                import jax

                return jax.vmap(fn, in_axes=in_axes)(*args)
            size = next(int(arg.shape[0]) for arg, axis in zip(args, in_axes) if axis == 0)
            outputs = [fn(*(arg[i] if axis == 0 else arg for arg, axis in zip(args, in_axes)))
                       for i in range(size)]
            if isinstance(outputs[0], tuple):
                return tuple(ops.stack(column) for column in zip(*outputs))
            return ops.stack(outputs)

        return mapped

    #********************#
    #   hermitian eigh   #
    #********************#
    @staticmethod
    def eigh_herm(h):
        """Real eigenvalues and complex64 eigenvectors of Hermitian matrices (..., d, d).

        Eigenvalues pass through `real` immediately (TF returns them complex); eigenvectors
        are cast to complex64 so downstream products never mix dtypes. Where the tensor's
        device has no eigh (torch's MPS before it gained one), the CPU computes it and the
        result moves back; Keras's own eigh would move the tensor back to the device first.
        """
        try:
            lam, v = ops.eigh(h)
        except NotImplementedError:
            import torch

            lam, v = (t.to(h.device) for t in torch.linalg.eigh(h.cpu()))
        return nops.real(lam), nops.cast(v, precision.COMPLEX)

    #**************************#
    #   propagator from eigh   #
    #**************************#
    @staticmethod
    def propagator_from_eigh(lam, v, dt):
        """U = V diag(exp(-i 2 pi lam dt)) V^H from a precomputed eigendecomposition.

        Gradients with respect to `dt` flow through the phase ramp only (V is constant in
        dt), so timing gradients stay exact even at degenerate eigenvalues. `dt` is a scalar
        or a tensor broadcast against the leading dimensions of `lam` after a trailing axis
        is appended.
        """
        dt_t = ops.cast(ops.convert_to_tensor(dt), lam.dtype)
        if ops.ndim(dt_t) > 0:
            dt_t = ops.expand_dims(dt_t, -1)
        ang = (-2.0 * math.pi) * lam * dt_t
        phase = ops.exp(nops.complex_from(ops.zeros_like(ang), ang))
        return ops.einsum("...ij,...j,...kj->...ik", v, phase, ops.conj(v))

    #***************#
    #   expm herm   #
    #***************#
    SQUARINGS = 8

    @staticmethod
    def expm_herm(h, dt):
        """exp(-i 2 pi H dt) for Hermitian H (Hz), batched over leading axes of H and dt,
        by scaling and squaring: A = -i 2 pi H dt is scaled by 2^-s to a Frobenius norm
        below two, exponentiated by its degree-18 Taylor polynomial (truncation 4e-12,
        below float32; seven matmuls in the Paterson-Stockmeyer grouping by powers of
        A^6), and squared s times. The scaling s is chosen per matrix in the graph and
        the squarings run a fixed SQUARINGS budget with a mask, so the graph stays
        static, the whole propagator is matmuls (no eigendecomposition and no LAPACK
        call inside a compiled graph, which every backend and platform fuses alike),
        and the gradient with respect to H or dt is exact at degenerate eigenvalues,
        where an eigendecomposition's derivative divides by zero. Matrices of norm
        above 2^(SQUARINGS + 1) (a step of hundreds of radians) exceed the budget; no
        pulse step comes near it. Every squaring doubles the float32 rounding, so the
        window is as wide as the polynomial allows."""
        dt_t = ops.cast(ops.convert_to_tensor(dt), precision.REAL)
        if ops.ndim(dt_t) > 0:
            dt_t = ops.reshape(dt_t, tuple(dt_t.shape) + (1, 1))
        a = (nops.complex_from(ops.zeros_like(nops.real(h)),
                               (-2.0 * math.pi) * nops.real(h) * dt_t)
             + nops.complex_from((2.0 * math.pi) * nops.imag(h) * dt_t,
                                 ops.zeros_like(nops.real(h))))
        norm = ops.sqrt(ops.sum(ops.abs(a) ** 2, axis=(-2, -1)))
        s = ops.maximum(ops.ceil(ops.log2(ops.maximum(norm / 2.0, 1e-30))), 0.0)
        scale = nops.cast(ops.exp2(-s), precision.COMPLEX)
        a = a * ops.expand_dims(ops.expand_dims(scale, -1), -1)
        eye = ops.eye(int(a.shape[-1]), dtype=precision.COMPLEX)
        a2 = a @ a
        a3 = a2 @ a
        a4 = a2 @ a2
        a5 = a4 @ a
        a6 = a4 @ a2
        powers = (eye, a, a2, a3, a4, a5)
        inverse_factorial = [1.0 / math.factorial(k) for k in range(19)]

        def block(j):
            return sum(inverse_factorial[6 * j + k] * powers[k] for k in range(6))

        u = block(2) + inverse_factorial[18] * a6
        u = block(1) + a6 @ u
        u = block(0) + a6 @ u
        for k in range(Backend.SQUARINGS):
            keep = ops.expand_dims(ops.expand_dims(s > float(k), -1), -1)
            u = ops.where(keep, u @ u, u)
        return u

    #****************#
    #   split step   #
    #****************#
    @staticmethod
    def split_step(h, basis, m, f1, dt, pieces=(1.0,)):
        """exp(-i 2 pi (H + f1 A) dt) for Hermitian H and A = basis diag(m) basis^H (Hz),
        batched over the leading axes of f1, as a composition of symmetric (Strang) pieces
        whose durations are the fractions `pieces` of dt, in order: each piece of duration
        tau is E(tau/2) R(tau) E(tau/2), with E the free evolution and
        R(tau) = exp(-i 2 pi f1 A tau) = basis diag(phase) basis^H. The free halves of
        neighbouring pieces join, so the step is E basis P (C P) ... (C P) basis^H E with
        C = basis^H E basis and P the diagonal phases: one matmul per piece, the free
        factors a few exponentials (expm_herm) shared by every step. Equal pieces are
        Strang's sub-steps (error falling as the square of their number); fractions that
        cancel the third-order error give a fourth-order step (Yoshida's triple jump). The
        phases are smooth in f1, so the gradient with respect to f1 and dt is exact at
        degenerate eigenvalues too.

        The basis is the system's constant, read on the host: a basis of spins 1/2 in
        float32 has its entries +-2^(-r/2) all rounded alike, so each round trip
        basis^H basis would shrink a state by a part in 1e7, the same way at every step;
        its signs, with their inverse the transpose over 2^r, are exact. A basis with
        another spin keeps its own adjoint."""
        dt = ops.cast(ops.convert_to_tensor(dt), precision.REAL)
        basis = np.asarray(ops.convert_to_numpy(basis), dtype=np.complex128)
        r = round(-2.0 * math.log2(np.abs(basis).max()))
        signs = np.rint(basis * 2.0 ** (r / 2.0))
        if np.abs(signs - basis * 2.0 ** (r / 2.0)).max() < 1e-4:
            basis, back = signs, signs.T / 2.0 ** r
        else:
            back = basis.conj().T
        basis, back = (nops.cast(ops.convert_to_tensor(x), precision.COMPLEX)
                       for x in (basis, back))
        ends = {pieces[0] / 2.0, pieces[-1] / 2.0}
        joins = {(a + b) / 2.0 for a, b in zip(pieces[:-1], pieces[1:])}
        free = {f: Backend.expm_herm(h, dt * f) for f in ends | joins}
        u = None
        for k, piece in enumerate(pieces):
            angle = (-2.0 * math.pi * piece) * ops.expand_dims(f1, -1) * dt * m
            phase = ops.expand_dims(ops.exp(nops.complex_from(ops.zeros_like(angle), angle)), -1)
            if u is None:
                u = phase * (back @ free[piece / 2.0])
            else:
                u = phase * ((back @ free[(pieces[k - 1] + piece) / 2.0] @ basis) @ u)
        return (free[pieces[-1] / 2.0] @ basis) @ u
