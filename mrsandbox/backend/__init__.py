####################################################################################################
#                                            __init__.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The single escape hatch for per-backend primitives; the propagator mathematics          #
#          lives in propagators.py. This package init only re-exports.                             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.backend.propagators import Backend
