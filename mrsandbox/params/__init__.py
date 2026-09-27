####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The parameter system, the configuration heart of MRSandbox. Every configurable quantity #
#          is a Value provider inside a composable Params tree; one seeded sample realizes the     #
#          whole tree and records the ground truth. Priors turn database sources into Params       #
#          subtrees.                                                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.params.values import (Clipped, Fixed, Mapped, Normal, PerTissue, Sourced, Tissues, Uniform,
                                     Value)
from mrsandbox.params.trees import Params
from mrsandbox.params.priors import Priors
