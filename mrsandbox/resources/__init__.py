####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Named remote data, fetched on first use into a local checksum verified cache. The       #
#          machinery lives in fetch.py, one cited registry module per data source beside it.       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.resources.fetch import Resources
from mrsandbox.resources.hmri import HMRI
from mrsandbox.resources.uk7t import UK7T
