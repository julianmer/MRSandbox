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
from mrsandbox.params import Tissues
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
    def lines(self, engine, sequence, fractions, truth, gamma_mhz_t):
        """Return (freqs, amps) line tensors, or None when this source is silent."""

    #******************#
    #   relaxation     #
    #******************#
    @staticmethod
    def relaxation(fractions, truth):
        """The voxel's water T1 and T2: per-tissue leaves weighted by the local
        composition; scalar overrides pass through unchanged."""

        def weigh(value):
            if not isinstance(value, Tissues):
                return value
            total = fractions.gm + fractions.wm + fractions.csf
            if total <= 0.0:
                return value.gm
            return (fractions.gm * value.gm + fractions.wm * value.wm
                    + fractions.csf * value.csf) / total

        return weigh(truth.water.t1_s), weigh(truth.water.t2_s)

    #********************#
    #   full amplitude   #
    #********************#
    @staticmethod
    def full_amplitude_mm(fractions):
        """Unsuppressed water in concentration equivalent mM: molarity times local content."""
        source = tissue.MRSI_CHALLENGE
        content = source["water_content"]
        return source["water_molarity_mm"] * (
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
    def _engine_lines(self, engine, sequence, fractions, truth, gamma_mhz_t):
        """The water lines of this draw through the engine's compiled path: relaxation
        and b1 stay inputs, so the first call compiles and every voxel replays."""
        t1_s, t2_s = self.relaxation(fractions, truth)
        fn = engine.compiled(sequence, self.system(gamma_mhz_t))
        return fn(ops.convert_to_tensor(np.float32(t1_s)),
                  ops.convert_to_tensor(np.float32(t2_s)),
                  ops.convert_to_tensor(np.float32(truth.b1_scale)))


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
    def lines(self, engine, sequence, fractions, truth, gamma_mhz_t):
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
    def lines(self, engine, sequence, fractions, truth, gamma_mhz_t):
        freqs, amps = self._engine_lines(engine, sequence, fractions, truth, gamma_mhz_t)
        return freqs, amps * complex(self.full_amplitude_mm(fractions))


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
    def lines(self, engine, sequence, fractions, truth, gamma_mhz_t):
        freqs, amps = self._engine_lines(engine, sequence, fractions, truth, gamma_mhz_t)
        residual = truth.water.residual_fraction * self.full_amplitude_mm(fractions)
        return freqs, amps * complex(residual)
