####################################################################################################
#                                            __init__.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual brain area: the anatomies and phantoms, the B0 field, and the voxel's       #
#          sources of water, lipids, macromolecules and the downfield peaks.                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.phantom.brain import Brain
from mrsandbox.phantom.solution import Phantom
from mrsandbox.phantom.downfield import DownfieldSource, NoDownfield, ParameterizedDownfield
from mrsandbox.phantom.field import FieldMap
from mrsandbox.phantom.lipids import LipidSource, NoLipids, Skull
from mrsandbox.phantom.macromolecules import (ChallengeMM, CoupledMM, MeasuredMM, MMSource,
                                              NoMM, ParameterizedMM)
from mrsandbox.phantom.registration import overlap, register
from mrsandbox.phantom.volume import Volume
from mrsandbox.phantom.water import NoWater, Suppressed, Unsuppressed, WaterSource
