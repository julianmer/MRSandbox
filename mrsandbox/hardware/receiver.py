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
    points. The receiver's noise passes the same filter with no onset (it was there
    before the acquisition), so it rolls off toward the edges of the spectrum.

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

    #**********#
    #   taps   #
    #**********#
    @property
    def taps(self):
        """The filter on the receiver's grid: a Hamming windowed sinc cut at the output
        Nyquist frequency, unit gain at DC."""
        return signal.firwin(2 * self.oversampling * self.span_dwells + 1, 1.0 / self.oversampling,
                             window="hamming")

    #***********#
    #   noise   #
    #***********#
    def noise_points(self, npoints):
        """The white samples on the receiver's grid behind `npoints` of filtered noise:
        the filter's full length around every output point."""
        return (npoints - 1) * self.oversampling + 2 * self.oversampling * self.span_dwells + 1

    def colour(self, white):
        """White noise on the receiver's grid (time last, `noise_points` long), filtered
        and decimated."""
        taps = self.taps.reshape((1,) * (white.ndim - 1) + (-1,))
        filtered = signal.fftconvolve(white, taps, mode="valid", axes=-1)
        return filtered[..., ::self.oversampling].astype(np.complex64)

    def power(self, fractions):
        """The power white noise keeps after the filter and the decimation at `fractions` of
        the output band from its centre: the filter's power response summed over the
        oversampled band's aliases that fold onto each output frequency."""
        count = 1 << 14
        response = np.fft.fftshift(np.abs(np.fft.fft(self.taps, count)) ** 2)
        grid = np.fft.fftshift(np.fft.fftfreq(count))
        fractions = np.asarray(fractions, dtype=np.float64)
        return sum(np.interp(np.mod((fractions + k) / self.oversampling + 0.5, 1.0) - 0.5, grid,
                             response) for k in range(self.oversampling))

    #**************#
    #   decimate   #
    #**************#
    def decimate(self, fid, npoints):
        """Filter a FID on the receiver's grid and keep the `npoints` output samples,
        the group delay removed."""
        half = self.oversampling * self.span_dwells
        filtered = np.convolve(fid, self.taps)[half::self.oversampling][:npoints]
        return filtered.astype(np.complex64)


#****************************#
#   the vendors' receivers   #
#****************************#
# The receiver filters raw data show (scan.calibrate ReceiverReading, the transients' tails): GE's
# MR750 decimates through a Hamming-windowed sinc of four output dwells a side (Big GABA's GE
# files). A scanner not listed plays no filter (stated).
RECEIVERS = {"ge_mr750_3t": Receiver(oversampling=8, span_dwells=4)}

