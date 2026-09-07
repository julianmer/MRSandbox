####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Hardware descriptions the simulations consume; scanners first, field maps, coils, and   #
#          limits as they earn their place.                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.hardware.coils import CoilArray, Loop, MappedCoilArray
from mrsandbox.hardware.scanners import Scanner
