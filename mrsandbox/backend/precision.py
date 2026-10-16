####################################################################################################
#                                           precision.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The floating-point precision of the spin simulation and the line render: float32 by     #
#          default, float64 for convergence checks. Every graph reads REAL and COMPLEX when it is  #
#          traced, and the engine keys its graphs by them.                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import keras
from keras.src.backend.common import dtypes

#*************#
#   setting   #
#*************#
REAL, COMPLEX = "float32", "complex64"
COMPLEX_OF = {"float32": "complex64", "float64": "complex128"}
# Keras types every op's result as jax does with 64 bits off, forcing float64 and complex128 to 32
# bits on every backend (its private table keras.src.backend.common.dtypes.BIT64_TO_BIT32_DTYPE):
# its own entries for float32, kept 64-bit ones for float64
NARROWED = {wide: dtypes.BIT64_TO_BIT32_DTYPE[wide] for wide in ("float64", "complex128")}


#*******************#
#   set precision   #
#*******************#
def set_precision(name):
    """Simulate in `name`, "float32" (the default) or "float64": the operators, the
    Hamiltonians, the propagators and the density matrix, and the lines' render. The
    voxel's sources and noise stay float32 data. There is no 16-bit choice: the density
    matrix is complex and no backend has a 16-bit complex type. On jax, float64 turns
    jax_enable_x64 on and float32 turns it off again, so a float32 run after a float64 one
    converts as before; on every backend Keras's ops then keep 64-bit results."""
    global REAL, COMPLEX
    if name not in COMPLEX_OF:
        raise ValueError(f"precision is float32 or float64, not {name!r} (no backend has a "
                         "16-bit complex type)")
    if keras.config.backend() == "jax":
        import jax

        jax.config.update("jax_enable_x64", name == "float64")
    for wide, narrow in NARROWED.items():
        dtypes.BIT64_TO_BIT32_DTYPE[wide] = wide if name == "float64" else narrow
    REAL, COMPLEX = name, COMPLEX_OF[name]
