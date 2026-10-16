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
import pathlib

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

if os.environ["KERAS_BACKEND"] == "jax":
    # the engine's graphs compile once per machine, not once per run, as the console has it
    # (console.app.main): the same cache directory, so a test and the console share what either
    # compiled
    import jax

    jax.config.update(
        "jax_compilation_cache_dir", str(pathlib.Path.home() / ".cache" / "mrsandbox" / "jax")
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


#******************#
#   module reset   #
#******************#
@pytest.fixture(scope="module", autouse=True)
def released():
    """After each test module, jax's compiled graphs and the buffers they hold go: a run of every
    module keeps no more than the module at hand does (the disk cache keeps recompiling cheap)."""
    yield
    if os.environ["KERAS_BACKEND"] == "jax":
        import gc

        import jax

        jax.clear_caches()
        gc.collect()


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
