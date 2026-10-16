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
from mrsandbox.params.values import Tissues
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
        """Return the water's parts, [(freqs, amps, t2_s, tissue)] line tensors each at
        the T2 its line decays with and the tissue it is ("gm", "wm", "csf"; None for
        the voxel's water as one), or None when this source is silent.

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
            fractions.gm * content["gm"] + fractions.wm * content["wm"]
            + fractions.csf * content["csf"]
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
    def _engine_lines(self, engine, sequence, voxel, truth, f0_mhz, t1_s=None, t2_s=None):
        """The water lines of this draw through the engine's compiled path: relaxation
        and b1 stay inputs, so the first call compiles and every voxel replays. The
        relaxation leaves resolve at the voxel (per-tissue values weighted by its
        composition, maps read over its box); `t1_s` and `t2_s` play one tissue's own
        instead. A draw `prepare` replayed with its batch is read from the cache."""
        t1_s = voxel.resolve(truth.water.t1_s) if t1_s is None else t1_s
        t2_s = voxel.resolve(truth.water.t2_s) if t2_s is None else t2_s
        held = getattr(self, "_lines", {}).get(id(truth))
        key = (engine, sequence, float(t1_s), float(t2_s), voxel.frame(truth)[3],
               voxel.offset_hz(truth))
        if held is not None and held[0] is truth and key in held[1]:
            return held[1][key]
        fn = engine.compiled(sequence, self.system(f0_mhz / engine.b0_t), shift_mm=voxel.shift_mm)
        return fn(*(ops.convert_to_tensor(np.float32(v)) for v in
                    (t1_s, t2_s, voxel.b1_input(truth), voxel.offset_hz(truth))))

    #*************#
    #   prepare   #
    #*************#
    def prepare(self, engine, sequence, voxel, truths, f0_mhz):
        """The water lines of many draws in one replay mapped over the draws
        (Engine.replay_draws), at each draw's mean relaxation, left for `_engine_lines`;
        a draw with a transmit profile keeps its own replay, a tissue's own T1 too."""
        held = self.__dict__.setdefault("_lines", {})
        members = []
        for truth in truths:
            if voxel.b1_profile(truth) is not None:
                continue
            key = (engine, sequence, float(voxel.resolve(truth.water.t1_s)),
                   float(voxel.resolve(truth.water.t2_s)), voxel.frame(truth)[3],
                   voxel.offset_hz(truth))
            entry = held.get(id(truth))
            if entry is None or entry[0] is not truth:
                if len(held) >= 256:
                    held.pop(next(iter(held)))
                entry = held[id(truth)] = (truth, {})
            if key not in entry[1]:
                members.append((truth, key, entry[1]))
        if not members:
            return
        replay, _ = engine.replay_draws(sequence, ((self.system(f0_mhz / engine.b0_t),),),
                                        shift_mm=voxel.shift_mm)
        t1, t2, b1, offset = (
            ops.convert_to_tensor(np.asarray(values, dtype=np.float32)) for values in (
                [[key[2]] for _, key, _ in members], [[key[3]] for _, key, _ in members],
                [key[4] for _, key, _ in members], [key[5] for _, key, _ in members]))
        freqs, amps = replay(t1, t2, b1, offset)
        for row, (_, key, cache) in enumerate(members):
            cache[key] = (freqs[row], amps[row])


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
    """The full water signal, as in a water reference acquisition, and what a suppression
    train in the sequence leaves of it.

    Each tissue's water runs the sequence with its own T1 and decays with its own T2
    where the leaves are per tissue (a map or a number runs once): what recovers
    through a train's gaps and over the TR is the tissue's, so CSF's slow water and the
    tissues' fast water are never nulled by one adjustment at once, and what a train
    leaves is their sum, the CSF's long narrow line against the tissues' broader ones,
    partly cancelling as raw residuals do.
    """

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        t1_s, t2_s = truth.water.t1_s, truth.water.t2_s
        if not isinstance(t1_s, Tissues):
            freqs, amps = self._engine_lines(engine, sequence, voxel, truth, f0_mhz)
            return [(freqs, amps * complex(self.full_amplitude_mm(voxel)), voxel.resolve(t2_s),
                     None)]
        fractions, content = voxel.fractions, voxel.scan.brain.water_content
        parts = []
        for name in ("gm", "wm", "csf"):
            share = getattr(fractions, name) * content[name]
            if share <= 0.0:
                continue
            own_t2 = getattr(t2_s, name) if isinstance(t2_s, Tissues) else voxel.resolve(t2_s)
            freqs, amps = self._engine_lines(engine, sequence, voxel, truth, f0_mhz,
                                             t1_s=getattr(t1_s, name), t2_s=own_t2)
            parts.append((freqs,
                          amps * complex(tissue.MRSI_CHALLENGE["water_molarity_mm"] * share),
                          own_t2, name))
        return parts or None


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
        return [(freqs, amps * complex(residual), voxel.resolve(truth.water.t2_s), None)]
