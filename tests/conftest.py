####################################################################################################
#                                           conftest.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Test configuration: pin the Keras backend BEFORE keras is imported anywhere (default    #
#          jax) and share the fixtures.                                                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest


#*********************#
#   backend fixture   #
#*********************#
@pytest.fixture(scope="session")
def backend():
    """Name of the active Keras backend."""
    import keras

    return keras.backend.backend()


#********************#
#   autograd guard   #
#********************#
@pytest.fixture()
def needs_autograd(backend):
    """Skip on backends without autograd (numpy)."""
    if backend == "numpy":
        pytest.skip("no autograd on the numpy backend")


#***********************#
#   hermitian factory   #
#***********************#
@pytest.fixture()
def random_hermitian():
    """Reproducible random Hermitian matrices (complex64), values in Hz."""

    def make(d, seed=0, scale_hz=50.0):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        return (scale_hz * 0.5 * (a + a.conj().T)).astype(np.complex64)

    return make
