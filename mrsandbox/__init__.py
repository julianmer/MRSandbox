####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: MRSandbox: a simulation sandbox for magnetic resonance spectroscopy. The package        #
#          exports grow as the areas land.                                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   backend   #
#*************#
# Keras falls to TensorFlow when no backend is named, and writes that default into ~/.keras/keras.json
# on its first import. The package runs on any of the three, so where KERAS_BACKEND is unset and the
# configured backend is not installed, the first one that is installed is taken.
import importlib.util
import json
import os
import pathlib

if "KERAS_BACKEND" not in os.environ:
    config = pathlib.Path.home() / ".keras" / "keras.json"
    named = json.loads(config.read_text()).get("backend", "tensorflow") if config.exists() else "tensorflow"
    if importlib.util.find_spec(named) is None:
        os.environ["KERAS_BACKEND"] = next(
            (name for name in ("jax", "torch", "tensorflow") if importlib.util.find_spec(name)), named)

#*************#
#   imports   #
#*************#
from mrsandbox.backend import Backend  # noqa: E402
from mrsandbox.database import Database
from mrsandbox.engine import Context, Engine
from mrsandbox.hardware import CoilArray, Loop, MappedCoilArray, Receiver, Scanner
from mrsandbox.params import (Clipped, Fixed, Mapped, Normal, Params, PerTissue, Priors, Sourced, Tissues,
                              Uniform, Value)
from mrsandbox.phantom import (Brain, ChallengeMM, CoupledMM, FieldMap, MeasuredMM, NoDownfield,
                               NoLipids, NoMM, NoWater, ParameterizedDownfield, ParameterizedMM, Phantom,
                               Skull, Suppressed, Unsuppressed, Volume)
from mrsandbox.scan import MRSI, Generator, Multiplex, ScannerProfile, VirtualScan, Voxel
from mrsandbox.sequences import (Chess, Cows, CoherenceFilter, Cycled, Delay, Gradient, HadamardEditing,
                                 Hercules, Hermes, InducedKick, LocalizedMegaPress,
                                 LocalizedMegaSemiLaser, LocalizedPress, LocalizedSemiLaser,
                                 LocalizedSteam, MegaPress, OuterVolumeSuppression, Press,
                                 Projection, Pulse, PulseAcquire, SelectivePulse, SemiLaser,
                                 Sequence, ShapedPulse, SiemensChess, Steam, SuppressionTrain, Vapor, Wet)
from mrsandbox.spins import Operators, SpinSystem

__version__ = "0.0.1"
