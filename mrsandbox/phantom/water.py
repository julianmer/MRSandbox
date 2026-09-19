####################################################################################################
#                                             water.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Water sources: swappable strategies for how the voxel's water appears in the            #
#          acquisition. Water is just another spin system run through the same engine and          #
#          sequence, its full amplitude anchored at the tissue water molarity times the local      #
#          water content; suppression scales the residual. All knobs are Params leaves, so the     #
#          ground truth records them.                                                              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc

import numpy as np
from keras import ops

from mrsandbox.database import tissue
from mrsandbox.spins import SpinSystem


#**************************************************************************************************#
#                                        Class WaterSource                                         #
#**************************************************************************************************#
#                                                                                                  #
# How the voxel's water enters the acquisition.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class WaterSource(abc.ABC):
    """How the voxel's water enters the acquisition."""

    #***********#
    #   lines   #
    #***********#
    @abc.abstractmethod
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        """Return (freqs, amps) line tensors, or None when this source is silent.

        The shared source signature: the placed scan.Voxel (fractions and geometry),
        the realized truth, and the spectrometer frequency in MHz."""

    #********************#
    #   full amplitude   #
    #********************#
    @staticmethod
    def full_amplitude_mm(voxel):
        """Unsuppressed water in concentration equivalent mM: molarity times the local
        content, which is the subject's (a brain's tissues, a phantom's solution)."""
        fractions, content = voxel.fractions, voxel.scan.brain.water_content
        return tissue.MRSI_CHALLENGE["water_molarity_mm"] * (
            fractions.gm * content["gm"] + fractions.wm * content["wm"] + fractions.csf * content["csf"]
        )

    #******************#
    #   water system   #
    #******************#
    @staticmethod
    def system(gamma_mhz_t):
        """Water as a spin system: one shift at 4.65 ppm, two protons."""
        return SpinSystem(name="water", nucleus="1H",
                          shifts_ppm=(tissue.MRSI_CHALLENGE["water_shift_ppm"],),
                          j_hz=((0.0,),), scale=2.0, gamma_mhz_t=gamma_mhz_t)

    #******************#
    #   engine lines   #
    #******************#
    def _engine_lines(self, engine, sequence, voxel, truth, f0_mhz):
        """The water lines of this draw through the engine's compiled path: relaxation
        and b1 stay inputs, so the first call compiles and every voxel replays. The
        relaxation leaves resolve at the voxel (per-tissue values weighted by its
        composition, maps read over its box)."""
        t1_s, t2_s = voxel.resolve(truth.water.t1_s), voxel.resolve(truth.water.t2_s)
        fn = engine.compiled(sequence, self.system(f0_mhz / engine.b0_t), shift_mm=voxel.shift_mm)
        return fn(*(ops.convert_to_tensor(np.float32(v)) for v in
                    (t1_s, t2_s, voxel.b1_input(truth), voxel.offset_hz(truth))))


#**************************************************************************************************#
#                                          Class NoWater                                           #
#**************************************************************************************************#
#                                                                                                  #
# Silence: no water signal at all.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class NoWater(WaterSource):
    """Silence: no water signal at all."""

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        return None


#**************************************************************************************************#
#                                        Class Unsuppressed                                        #
#**************************************************************************************************#
#                                                                                                  #
# The full water signal, as in a water reference acquisition.                                      #
#                                                                                                  #
#**************************************************************************************************#
class Unsuppressed(WaterSource):
    """The full water signal, as in a water reference acquisition."""

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        freqs, amps = self._engine_lines(engine, sequence, voxel, truth, f0_mhz)
        return freqs, amps * complex(self.full_amplitude_mm(voxel))


#**************************************************************************************************#
#                                         Class Suppressed                                         #
#**************************************************************************************************#
#                                                                                                  #
# A residual after water suppression, scaled by the sampled residual fraction.                     #
#                                                                                                  #
#**************************************************************************************************#
class Suppressed(WaterSource):
    """A residual after water suppression, scaled by the sampled residual fraction.

    The residual fraction comes from the realized params (truth.water.residual_fraction),
    so it is sampled, recorded, and overridable like every other quantity.
    """

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        freqs, amps = self._engine_lines(engine, sequence, voxel, truth, f0_mhz)
        residual = truth.water.residual_fraction * self.full_amplitude_mm(voxel)
        return freqs, amps * complex(residual)
