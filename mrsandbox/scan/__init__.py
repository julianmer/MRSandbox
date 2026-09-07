####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The scan area: VirtualScan and Voxel compose brain, scanner, sequence, and Params into  #
#          acquisitions; the training generators join here next.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.scan.mrsi import MRSI
from mrsandbox.scan.multiplex import Multiplex
from mrsandbox.scan.virtual_scan import VirtualScan, Voxel
