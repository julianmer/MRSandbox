####################################################################################################
#                                            __init__.py                                           #
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
# Keras falls to TensorFlow when no backend is named, and writes that default into
# ~/.keras/keras.json on its first import. The package runs on any of the three, so where
# KERAS_BACKEND is unset and the configured backend is not installed, the first one that is
# installed is taken.
import importlib.util
import json
import os
import pathlib

if "KERAS_BACKEND" not in os.environ:
    config = pathlib.Path.home() / ".keras" / "keras.json"
    named = (json.loads(config.read_text()).get("backend", "tensorflow") if config.exists()
             else "tensorflow")
    if importlib.util.find_spec(named) is None:
        installed = next((name for name in ("jax", "torch", "tensorflow")
                          if importlib.util.find_spec(name)), None)
        if installed is None:
            raise ImportError("mrsandbox needs a Keras backend: install jax, torch or "
                              "tensorflow, as in pip install \"mrsandbox[jax]\"")
        os.environ["KERAS_BACKEND"] = installed

#*************#
#   exports   #
#*************#
# the package's names by the module that holds them, imported on first use (a module __getattr__):
# importing a part of the package (the console's window) does not load the engine and its backend
EXPORTS = {
    "mrsandbox.backend": ("Backend",),
    "mrsandbox.database": ("Database",),
    "mrsandbox.engine": ("Context", "Engine"),
    "mrsandbox.hardware": ("CoilArray", "Loop", "MappedCoilArray", "Receiver", "Scanner"),
    "mrsandbox.params": ("Clipped", "Empirical", "Fixed", "Mapped", "Mixture", "Normal", "Params",
                         "PerTissue", "Priors", "Sourced", "Tissues", "Uniform", "Value"),
    "mrsandbox.phantom": ("Brain", "ChallengeMM", "CoupledMM", "FieldMap", "MeasuredMM",
                          "NoDownfield", "NoLipids", "NoMM", "NoWater", "ParameterizedDownfield",
                          "ParameterizedMM", "Phantom", "Skull", "Suppressed", "Unsuppressed",
                          "Volume"),
    "mrsandbox.scan": ("MRSI", "Generator", "Multiplex", "ScannerProfile", "VirtualScan", "Voxel"),
    "mrsandbox.sequences": ("Chess", "Cows", "CoherenceFilter", "Cycled", "Delay", "Gradient",
                            "HadamardEditing", "Hercules", "Hermes", "InducedKick",
                            "LocalizedMegaPress", "LocalizedMegaSemiLaser", "LocalizedPress",
                            "LocalizedSemiLaser", "LocalizedSteam", "MegaPress",
                            "OuterVolumeSuppression", "Press", "Projection", "Pulse",
                            "PulseAcquire", "SelectivePulse", "SemiLaser", "Sequence",
                            "ShapedPulse", "SiemensChess", "Steam", "SuppressionTrain", "Vapor",
                            "Wet"),
    "mrsandbox.spins": ("Operators", "SpinSystem"),
}
OWNER = {name: module for module, names in EXPORTS.items() for name in names}
__all__ = [*OWNER, "replicate"]


def __getattr__(name):
    import importlib

    if name in OWNER:
        value = getattr(importlib.import_module(OWNER[name]), name)
        globals()[name] = value
        return value
    # a subpackage asked for as an attribute (mrsandbox.scan) is imported as Python would
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__version__ = "0.1.3"


#***************#
#   replicate   #
#***************#
def replicate(folder):
    """A folder of scans read at once into a room: every file told apart by what it is and
    used for what it can give (a run with its water reference calibrates the scanner, a T1
    image is the anatomy, field and B1 maps the room's, coil maps the array, a noise scan the
    channels' own noise share, a saved scanner its calibration; console.session.Session
    .replicate), the report on `report` as [(file, kind, what it gave or why not)]. The room is
    the console's session: `twin(sequence)` is a VirtualScan of it, `scanner`, `subject` and
    the rest as the console shows them."""
    from mrsandbox.console.session import Session

    room = Session("Brain, ICBM152 average, 2 mm")
    room.report = room.replicate(folder)
    return room
