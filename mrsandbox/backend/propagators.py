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

from keras import ops
from nifti_mrs_plus import ops as nops


#**************************************************************************************************#
#                                          Class Backend                                           #
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

    #********************#
    #   hermitian eigh   #
    #********************#
    @staticmethod
    def eigh_herm(h):
        """Real eigenvalues and complex64 eigenvectors of Hermitian matrices (..., d, d).

        Eigenvalues pass through `real` immediately (TF returns them complex); eigenvectors
        are cast to complex64 so downstream products never mix dtypes.
        """
        lam, v = ops.eigh(h)
        return nops.real(lam), nops.cast(v, "complex64")

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
    @staticmethod
    def expm_herm(h, dt):
        """exp(-i 2 pi H dt) for Hermitian H (Hz), batched over leading axes of H and dt."""
        lam, v = Backend.eigh_herm(h)
        return Backend.propagator_from_eigh(lam, v, dt)
