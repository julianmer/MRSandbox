####################################################################################################
#                                           downfield.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: Downfield sources: the 6.83 to 8.49 ppm resonances (amides, the NAA amide, ATP,         #
#          homocarnosine) as a swappable source like water and macromolecules. These protons       #
#          exchange with water, so water presaturation transfers saturation onto them: each peak   #
#          then retains R1 / (R1 + kex), the two-site steady state at saturated water, with the    #
#          measured 3 T relaxation and exchange rates.                                             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc

import numpy as np

from mrsandbox.database import downfield


#**************************************************************************************************#
#                                      Class DownfieldSource                                       #
#**************************************************************************************************#
#                                                                                                  #
# How the downfield resonances enter the acquisition.                                              #
#                                                                                                  #
#**************************************************************************************************#
class DownfieldSource(abc.ABC):
    """How the downfield resonances enter the acquisition.

    A source contributes parameterized lines; the default is silence. A source that
    models the downfield empirically declares `replaces_above_ppm`, and the scan then
    drops the spin systems' own lines above that position from the render: those are
    exchangeable NH protons whose exchange broadening and saturation the density
    matrix does not carry, so rendering them at full narrow amplitude would both be
    unphysical and double count the parameterization.
    """

    replaces_above_ppm = None

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        """Return (freqs_hz, amps, gauss_hz), or None when silent. The shared source
        signature: the placed scan.Voxel, the realized truth, and the spectrometer
        frequency in MHz."""
        return None


#**************************************************************************************************#
#                                       Class NoDownfield                                          #
#**************************************************************************************************#
#                                                                                                  #
# Silence: no downfield signal at all (the conventional 0 to 4.2 ppm world).                       #
#                                                                                                  #
#**************************************************************************************************#
class NoDownfield(DownfieldSource):
    """Silence: no downfield signal at all (the conventional 0 to 4.2 ppm world)."""


#**************************************************************************************************#
#                                  Class ParameterizedDownfield                                    #
#**************************************************************************************************#
#                                                                                                  #
# The nine downfield Gaussians, vanishing physically under water presaturation.                    #
#                                                                                                  #
#**************************************************************************************************#
class ParameterizedDownfield(DownfieldSource):
    """The nine downfield Gaussians of the Johns Hopkins parameterization (Ozdemir
    2023, Povazan 2022), vanishing physically under water presaturation.

    Each peak is a Gaussian at its published position with the drawn level
    (institutional units treated as concentration equivalent mM, stated) times the
    tissue presence of the voxel. With truth.downfield.presaturated the exchangeable
    protons are partially saturated through their exchange with the saturated water:
    each peak retains R1 / (R1 + kex), the two-site Bloch-McConnell steady state at
    full water saturation, with the MacMillan 2011 median 3 T values mapped by peak
    position and assignment (EXCHANGE below; the 8.37 amide follows the 8.24 amide
    pool, the 7.48 peak its nearest measured neighbour at 7.32).

    The parameterization is empirical: it describes the whole observed downfield,
    metabolite NH protons included, so the spin systems' own lines above
    `replaces_above_ppm` (6.5, stated: below the 6.83 ppm first peak and above the
    upfield spectrum) are dropped from the render in its favour.

    Stated limits: the peaks render as static Gaussians (no echo time or sequence
    response, like the measured MM shapes), the retention assumes complete water
    saturation, and exchange dynamics (lineshape, TE loss during exchange) are not
    modelled.
    """

    replaces_above_ppm = 6.5

    # measured MacMillan peak carrying each parameterized peak's T1 and kex
    EXCHANGE = {"df683": 6.84, "df709": 7.09, "df730": 7.32, "df748": 7.32,
                "df790": 7.86, "df818": 8.24, "df824": 8.24, "df837": 8.24,
                "df849": 8.50}

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        tissue = voxel.fractions.gm + voxel.fractions.wm
        if tissue <= 0.0:
            return None
        freqs, amps, gauss = [], [], []
        for name, peak in downfield.OZDEMIR.items():
            level = truth.downfield.level[name] * tissue
            if truth.downfield.presaturated:
                level *= self.retention(name)
            freqs.append(peak["ppm"] * f0_mhz)
            amps.append(level)
            gauss.append(truth.downfield.gauss_hz)
        return (np.asarray(freqs, dtype=np.float32),
                np.asarray(amps, dtype=np.complex64),
                np.asarray(gauss, dtype=np.float32))

    #***************#
    #   retention   #
    #***************#
    @classmethod
    def retention(cls, name):
        """The fraction a peak keeps under water presaturation: R1 / (R1 + kex) with
        the MacMillan 2011 median T1 and exchange rate of its measured pool."""
        measured = cls.EXCHANGE[name]
        r1 = 1000.0 / downfield.MACMILLAN["t1_ms"][measured]
        return r1 / (r1 + downfield.MACMILLAN["kex_hz"][measured])
