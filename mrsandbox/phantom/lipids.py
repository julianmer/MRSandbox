####################################################################################################
#                                            lipids.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Lipid sources: swappable strategies for the subcutaneous fat that enters the voxel.     #
#          The Skull source weighs lipids by the head-but-not-brain fraction of the voxel box      #
#          (a stated proxy for the fat layer read straight off the tissue maps).                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc

import numpy as np
from keras import ops

from mrsandbox.database import lipids as lipid_table
from mrsandbox.sequences.events import Projection
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

    A projected design (one with `Projection` events: the localized designs) on a
    brain with a fat map takes the layer's true shape instead: the lipid peaks are
    uncoupled, so what the localization leaves of them at a position is the product
    of its three axis profiles (Engine.profile) over the centre's, and the signal is
    that response summed over every fat cell inside the ensemble's span, in units of
    the nominal voxel volume. Fat outside the nominal box that a profile's foot or a
    displaced slab reaches is then in the spectrum, as on a scanner.
    """

    PROFILE_POINTS = 129

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        if (voxel.scan.brain.fat is not None
                and any(isinstance(e, Projection) for e in sequence.events)):
            return self._profiled(engine, sequence, voxel, truth, f0_mhz)
        weight = voxel.scan.brain.lipid_weight(voxel.position_mm, voxel.size_mm)
        if weight == 0.0:
            return None
        amplitude = truth.lipids.amplitude_mm * weight
        offset = voxel.scan.brain.lipid_offset(voxel.position_mm, voxel.size_mm)
        placed = voxel.shift_mm or {}
        shift = {axis: float(offset[k]) + placed.get(axis, 0.0) for k, axis in enumerate("xyz")}
        args = [ops.convert_to_tensor(np.float32(v)) for v in
                (truth.lipids.t1_s, truth.lipids.t2_s, voxel.b1_input(truth),
                 voxel.offset_hz(truth))]
        freqs_all, amps_all = [], []
        for system in self.systems(f0_mhz / engine.b0_t):
            freqs, amps = engine.compiled(sequence, system, shift_mm=shift)(*args)
            freqs_all.append(freqs)
            amps_all.append(amps * complex(amplitude))
        return ops.concatenate(freqs_all), ops.concatenate(amps_all)

    #**************#
    #   profiled   #
    #**************#
    def _profiled(self, engine, sequence, voxel, truth, f0_mhz):
        """The fat layer through a projected design's voxel profile, cell by cell."""
        extents = engine.extents(sequence)
        edges = np.asarray(voxel.size_mm, dtype=np.float64) * np.ones(3)
        span = [extents.get(axis, edges[k]) for k, axis in enumerate("xyz")]
        offsets, volumes = voxel.scan.brain.lipid_cells(voxel.position_mm, span)
        if volumes.size == 0:
            return None
        b1_scale, offset_hz = voxel.b1_scale(truth), voxel.offset_hz(truth)
        placed = voxel.shift_mm or {}  # a cell of a box: the profiles are the box's
        freqs_all, amps_all = [], []
        for system in self.systems(f0_mhz / engine.b0_t):
            system = system.with_relaxation(t1_s=truth.lipids.t1_s, t2_s=truth.lipids.t2_s)
            response, centre = np.ones(volumes.size, dtype=np.complex128), None
            for k, axis in enumerate("xyz"):
                if axis not in extents:
                    continue
                grid = np.linspace(-span[k] / 2.0, span[k] / 2.0, self.PROFILE_POINTS)
                at = np.append(placed.get(axis, 0.0) + grid, 0.0)  # and the design's centre
                freqs, amps = engine.profile(sequence, system, axis, at, b1_scale, offset_hz)
                centre, amps = amps[-1], amps[:-1]
                line = np.argmax(np.abs(centre))
                along = amps[:, line] / centre[line]
                response *= (np.interp(offsets[:, k], grid, along.real)
                             + 1j * np.interp(offsets[:, k], grid, along.imag))
            seen = (response * volumes).sum() / float(np.prod(edges))
            freqs_all.append(freqs.astype(np.float32))
            amps_all.append((centre * seen * truth.lipids.amplitude_mm).astype(np.complex64))
        return (ops.convert_to_tensor(np.concatenate(freqs_all)),
                ops.convert_to_tensor(np.concatenate(amps_all)))
