####################################################################################################
#                                            lipids.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Lipid sources: swappable strategies for subcutaneous fat entering the voxel. The v1     #
#          Skull source weighs lipids by the head-but-not-brain fraction of the voxel box (a       #
#          stated proxy for the fat layer read straight off the tissue maps); donor-registered and #
#          SimnTG-parametric sources join later.                                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc

import numpy as np
from keras import ops

from mrsandbox.database import lipids as lipid_table
from mrsandbox.spins import SpinSystem


#**************************************************************************************************#
#                                        Class LipidSource                                         #
#**************************************************************************************************#
#                                                                                                  #
# How fat enters the voxel.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class LipidSource(abc.ABC):
    """How fat enters the voxel."""

    #***********#
    #   lines   #
    #***********#
    @abc.abstractmethod
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        """Return (freqs, amps) line tensors, or None when this source is silent.

        The shared source signature: the placed scan.Voxel (fractions and geometry),
        the realized truth, and the spectrometer frequency in MHz."""

    #*******************#
    #   lipid systems   #
    #*******************#
    @staticmethod
    def systems(gamma_mhz_t):
        """The lipid resonances as uncoupled systems: the Hamilton six-peak triglyceride
        spectrum with its measured relative amplitudes."""
        peaks = lipid_table.HAMILTON["peaks"]
        return tuple(
            SpinSystem(name=f"lipid_{name}", nucleus="1H", shifts_ppm=(peak["shift_ppm"],),
                       j_hz=((0.0,),), scale=peak["relative_amplitude"], gamma_mhz_t=gamma_mhz_t)
            for name, peak in peaks.items()
        )


#**************************************************************************************************#
#                                          Class NoLipids                                          #
#**************************************************************************************************#
#                                                                                                  #
# Silence: no lipid signal at all.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class NoLipids(LipidSource):
    """Silence: no lipid signal at all."""

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        return None


#**************************************************************************************************#
#                                           Class Skull                                            #
#**************************************************************************************************#
#                                                                                                  #
# Lipids weighted by the voxel's head-but-not-brain content.                                       #
#                                                                                                  #
#**************************************************************************************************#
class Skull(LipidSource):
    """Lipids weighted by the voxel's head-but-not-brain content, rendered where that
    content sits.

    The sampled amplitude (truth.lipids.amplitude_mm, a concentration equivalent at full
    lipid weight) scales with the geometric weight, so interior voxels stay clean and
    voxels touching the skull bleed. The lipid systems run the design with the
    isochromat ensemble moved to the fat's centroid (Brain.lipid_offset, along the
    axes the design's gradients use), so a slice profile's transition band, the
    chemical shift displacement, and an outer-volume saturation band act on the fat
    where it is; the ensemble keeps the voxel's extent about that centroid (a stated
    approximation of the layer's true shape).
    """

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        weight = voxel.scan.brain.lipid_weight(voxel.position_mm, voxel.size_mm)
        if weight == 0.0:
            return None
        amplitude = truth.lipids.amplitude_mm * weight
        offset = voxel.scan.brain.lipid_offset(voxel.position_mm, voxel.size_mm)
        shift = {axis: float(offset[k]) for k, axis in enumerate("xyz")}
        args = [ops.convert_to_tensor(np.float32(v)) for v in
                (truth.lipids.t1_s, truth.lipids.t2_s, voxel.b1_input(truth),
                 voxel.offset_hz(truth))]
        freqs_all, amps_all = [], []
        for system in self.systems(f0_mhz / engine.b0_t):
            freqs, amps = engine.compiled(sequence, system, shift_mm=shift)(*args)
            freqs_all.append(freqs)
            amps_all.append(amps * complex(amplitude))
        return ops.concatenate(freqs_all), ops.concatenate(amps_all)
