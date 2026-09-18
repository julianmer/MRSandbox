####################################################################################################
#                                           receiver.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The digital receiver: the signal is sampled faster than the spectral dwell, low-pass    #
#          filtered by a linear-phase FIR, corrected for the filter's group delay, and decimated.  #
#          The filter meets the abrupt start of the acquisition, which is what corrupts the first  #
#          points of a real FID.                                                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np
from scipy import signal


#**************************************************************************************************#
#                                          Class Receiver                                          #
#**************************************************************************************************#
#                                                                                                  #
# A digital receiver: oversampled acquisition, a linear-phase FIR low-pass, and decimation.        #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Receiver:
    """A digital receiver: oversampled acquisition, a linear-phase FIR low-pass, and
    decimation.

    The scan renders the FID on the receiver's own grid, `oversampling` samples per
    spectral dwell, with nothing before the acquisition's start. The windowed-sinc
    low-pass (Hamming, unit gain at DC, cut at the output Nyquist frequency) spans
    `span_dwells` output dwells on each side of its centre; its group delay is removed
    exactly (the centre tap is aligned with every output sample) and every
    `oversampling`-th sample is kept. Away from the start the output is the ideal FID
    band-limited to the spectral width; at the start the filter straddles the onset,
    so the first point loses about half its amplitude and the next ones ring, within
    about `span_dwells` points. Both numbers are stated assumptions: vendors do not
    publish their filters, and their group delays run from a few to some seventy
    points.

    Args:
        oversampling: Receiver samples per spectral dwell.
        span_dwells: Half length of the filter in spectral dwells.
    """

    oversampling: int = 8
    span_dwells: int = 8

    #**********#
    #   grid   #
    #**********#
    def grid(self, npoints, dwell_s):
        """The (npoints, dwell_s) of the receiver's own grid behind an output grid: the
        output's span plus the filter's half length, at the oversampled rate."""
        half = self.oversampling * self.span_dwells
        return (npoints - 1) * self.oversampling + half + 1, dwell_s / self.oversampling

    #**************#
    #   decimate   #
    #**************#
    def decimate(self, fid, npoints):
        """Filter a FID on the receiver's grid and keep the `npoints` output samples,
        the group delay removed."""
        half = self.oversampling * self.span_dwells
        taps = signal.firwin(2 * half + 1, 1.0 / self.oversampling, window="hamming")
        filtered = np.convolve(fid, taps)[half::self.oversampling][:npoints]
        return filtered.astype(np.complex64)
