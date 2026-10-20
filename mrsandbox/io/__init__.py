####################################################################################################
#                                            __init__.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: File format readers for external data; NIfTI comes through nibabel and NIfTI-MRS        #
#          through nifti_mrs, so only formats without a maintained reader live here.               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.io.dicom import read_dicom
from mrsandbox.io.philips import DataList, LabRaw, SdatSpar
from mrsandbox.io.folder import Found, convert, pairs, survey
from mrsandbox.io.fsl import FslSequence, read_fsl_sequence
from mrsandbox.io.pulseq import PulseqSequence, read_seq, read_seq_all, write_seq
from mrsandbox.io.varian import VarianFid
