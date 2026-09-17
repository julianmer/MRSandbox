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
#   imports   #
#*************#
from mrsandbox.backend import Backend
from mrsandbox.database import Database
from mrsandbox.engine import Context, Engine
from mrsandbox.hardware import CoilArray, Loop, MappedCoilArray, Scanner
from mrsandbox.params import (Clipped, Fixed, Mapped, Normal, Params, PerTissue, Priors, Tissues,
                              Uniform, Value)
from mrsandbox.phantom import (Brain, CoupledMM, FieldMap, MeasuredMM, NoDownfield, NoLipids,
                               NoMM, NoWater, ParameterizedDownfield, ParameterizedMM, Skull,
                               Suppressed, Unsuppressed, Volume)
from mrsandbox.scan import MRSI, Generator, Multiplex, VirtualScan, Voxel
from mrsandbox.sequences import (Chess, CoherenceFilter, Cycled, Delay, Gradient, HadamardEditing,
                                 Hercules, Hermes, MegaPress, OuterVolumeSuppression, Press, Pulse,
                                 PulseAcquire, SelectivePulse, SemiLaser, Sequence, ShapedPulse,
                                 Steam, SuppressionTrain, Vapor, Wet)
from mrsandbox.spins import Operators, SpinSystem

__version__ = "0.0.1"
