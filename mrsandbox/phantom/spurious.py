####################################################################################################
#                                            spurious.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/11/26                                                                                #
#                                                                                                  #
# Purpose: Spurious echo sources: the out-of-voxel echo that reaches the readout, either drawn   #
#          in the measured form or played from the scalp's coherence pathways, what the design's #
#          crushers leave beyond what its coherence filters idealize away.                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc
import copy

import numpy as np
from keras import ops

from mrsandbox.database import tissue
from mrsandbox.phantom.lipids import Skull
from mrsandbox.phantom.water import WaterSource
from mrsandbox.sequences.events import (CoherenceFilter, Gradient, Pulse, SelectivePulse,
                                        ShapedPulse)


#**************************************************************************************************#
#                                       Class SpuriousSource                                       #
#**************************************************************************************************#
#                                                                                                  #
# How a spurious echo enters the readout.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class SpuriousSource(abc.ABC):
    """How a spurious echo enters the readout."""

    @abc.abstractmethod
    def fid(self, voxel, truth, fids, npoints, dwell_s, start_s):
        """The echo's FID on the readout grid (complex64, `npoints` samples `dwell_s`
        apart from `start_s` after the readout's start), or None when silent. `fids`
        are the other sources' rendered FIDs by name, which a drawn echo is sized against."""


#**************************************************************************************************#
#                                       Class ParametricEcho                                       #
#**************************************************************************************************#
#                                                                                                  #
# The drawn echo of the measured form: a Gaussian bump at a drawn time, place, size and phase.     #
#                                                                                                  #
#**************************************************************************************************#
class ParametricEcho(SpuriousSource):
    """The drawn echo of the measured form: a Gaussian bump at a drawn time, place, size
    and phase (scan.defaults spurious, the AGNOSTIC form), its size relative to the
    maximum of the summed FID without the water (what comes from outside the voxel does
    not grow with the water a suppression train leaves inside it), rendered by
    Augmentrum's localized echo with its envelope exp(-(t - t_echo)^2 / (2 T2^2)), so
    T2 = 1 / sqrt(2 W). The default."""

    #*********#
    #   fid   #
    #*********#
    def fid(self, voxel, truth, fids, npoints, dwell_s, start_s):
        spur = truth.spurious
        if spur.amplitude <= 0.0:
            return None
        try:
            from augmentrum.augmentation import SpuriousEchoes
        except ImportError as error:
            raise ImportError("the drawn spurious echo needs Augmentrum (the mrsandbox[augment] "
                              "extra; see pyproject.toml for the install recipe)") from error
        f0_mhz = voxel._f0_mhz()
        echo = SpuriousEchoes(mode="echo", echoes=[dict(
            alpha=spur.amplitude, t_echo=spur.tau_s - start_s,
            T2=1.0 / np.sqrt(2.0 * spur.width_s2), freq_hz=spur.ppm * f0_mhz,
            phase_deg=spur.phase_deg + 360.0 * spur.ppm * f0_mhz * start_s,
            gaussian_env=True)])
        total = sum((fid for name, fid in fids.items() if name != "water"),
                    np.zeros(npoints, dtype=np.complex64))
        return (np.asarray(echo.process_tensor(total[None, :], sw_hz=1.0 / dwell_s)[0][0])
                - total)


#**************************************************************************************************#
#                                         Class PathwayEcho                                        #
#**************************************************************************************************#
#                                                                                                  #
# The echo played from the scalp's coherence pathways through the design's crushers.               #
#                                                                                                  #
#**************************************************************************************************#
class PathwayEcho(SpuriousSource):
    """The echo played from the scalp's coherence pathways through the design's crushers.

    The region is the scalp beyond the voxel's nearest face: a box of the fat layer's own
    thickness along that face's normal (the layer's cells' spread, phantom.Brain
    lipid_cells) and the voxel's edges along the other two axes, its fat the layer's
    cells inside it and its water the rest at SCALP_WATER of pure water (skin and muscle,
    stated), the lipids' and the voxel's water relaxation. The design plays on that
    region as one three-dimensional ensemble (`isochromats` per axis) with every
    coherence filter a crusher: the filters on both sides of a pulse are a pair of CYCLES
    over the voxel edge (alternating per pair), equal where the order turns over and
    opposite where it stays (an editing pulse); a filter on order zero (a train's, a
    band's) a lone spoiler of LONE cycles; a transverse filter on its own (after an
    excitation) stands for a pair elsewhere and gets none; each on its pulse's slab axis,
    a train's spoilers over x, y and z in turn (stated: a scanner's own scheme is its
    vendor's; equal spoilers on one axis would refocus a pathway stored between them). A
    storing pulse (STEAM) has no pairs, and the source stays silent.
    Whole cycles over an uncut region dephase exactly, so the echo is what the slab edges
    cut: a displaced slab leaves part of the region outside one 180, over which a pair is
    no longer whole, and that part, excited by a suppression pulse and refocused by the
    other 180 alone, rephases inside the readout at the pathway's own time. The echo is the
    crushed play less the filtered one, what the crushers leave beyond what the filters
    idealize away (the filtered play is what the voxel's own sources already carry), so a
    perfect crush, the filters, gives nothing. The region's field spread is a comb of
    offsets over SPAN standard deviations of a Gaussian of the drawn echo width
    (spurious.width_s2, so the echo's envelope is the measured one), each row playing
    the design at its offset and rendering at its own frequency, which puts every
    pathway's refocusing where its phase history says; the comb is spaced by the inverse
    of the readout plus the design's duration, so its aliases fall beyond the readout
    (a hundred rows, each a draw of the two replays: minutes on a laptop CPU, seconds on
    a GPU). The cycles stay small (CYCLES) so the
    grid resolves every unbalanced moment: a moment of more cycles than half the
    positions aliases. The echo's size is the physics' (spurious.amplitude only switches
    it off; the run still carries it by its rate, scan.Multiplex).

    Args:
        isochromats: Positions per axis of the region's ensemble, ISOCHROMATS by default.
    """

    CYCLES, LONE = (1.0, 2.0), 1.0
    SPAN, ISOCHROMATS, SCALP_WATER = 3.0, 10, 0.7

    #**********#
    #   init   #
    #**********#
    def __init__(self, isochromats=ISOCHROMATS):
        self.isochromats = int(isochromats)

    #*************#
    #   crushed   #
    #*************#
    def crushed(self, design, edge_mm):
        """The design with every coherence filter a crusher (see the class), or None for a
        design without filters or with a storing pulse (STEAM: its crushers are no pairs)."""
        events = list(design.events)
        if not any(isinstance(e, CoherenceFilter) for e in events):
            return None
        pulses = (Pulse, SelectivePulse, ShapedPulse)
        out, pair, lone, axis, k = [], 0, 0, None, 0
        while k < len(events):
            event = events[k]
            if isinstance(event, pulses):
                axis = event.gradient_axis if getattr(event, "gradient_hz_per_mm", 0.0) else None
            if not isinstance(event, CoherenceFilter):
                out.append(event)
                k += 1
            elif (k + 2 < len(events) and isinstance(events[k + 1], pulses)
                  and isinstance(events[k + 2], CoherenceFilter)
                  and event.orders != (0,)):
                # a pair about a pulse: equal moments where the order turns over, opposite where
                # it stays; a storing pulse keeps the longitudinal part instead
                before, pulse, after = event.orders, events[k + 1], events[k + 2].orders
                if 0 in after:
                    return None
                moment = self.CYCLES[pair % len(self.CYCLES)] / edge_mm
                pair += 1
                axis = pulse.gradient_axis if getattr(pulse, "gradient_hz_per_mm", 0.0) else "z"
                out += [Gradient(moment, edge_mm, axis=axis), pulse,
                        Gradient(-moment if after == before else moment, edge_mm, axis=axis)]
                k += 3
            else:
                # a saturation pulse's spoiler keeps the longitudinal part alone; a transverse
                # filter on its own (after an excitation) stands for a pair elsewhere
                if event.orders == (0,):
                    out.append(Gradient(self.LONE / edge_mm, edge_mm, axis=axis or "xyz"[lone % 3]))
                    lone += axis is None
                k += 1
        crushed = copy.copy(design)
        crushed.events = tuple(out)
        return crushed

    #************#
    #   region   #
    #************#
    def region(self, voxel):
        """The scalp beyond the voxel's nearest face: (edges mm, centre mm from the voxel's
        along its axes, the fat inside in cubic mm), or None away from the scalp."""
        brain = voxel.scan.brain
        if not hasattr(brain, "lipid_cells"):
            return None
        edges = np.asarray(voxel.size_mm, dtype=np.float64) * np.ones(3)
        extents = voxel.scan.engine.extents(voxel.scan.sequence)
        span = [extents.get(axis, edges[k]) for k, axis in enumerate("xyz")]
        offsets, volumes = brain.lipid_cells(voxel.position_mm, span, voxel.turn)
        if volumes.size == 0:
            return None
        centroid = np.average(offsets, axis=0, weights=volumes)
        normal = int(np.argmax(np.abs(centroid)))
        spread = np.average((offsets[:, normal] - centroid[normal]) ** 2, weights=volumes)
        size = edges.copy()
        size[normal] = np.clip(np.sqrt(12.0 * spread), 2.0, span[normal])
        centre = np.zeros(3)
        centre[normal] = np.sign(centroid[normal]) * (edges[normal] + size[normal]) / 2.0
        placed = centre if voxel.turn is None else centre @ voxel.turn.T
        fat_mm3 = brain.lipid_cells(np.asarray(voxel.position_mm) + placed, size, voxel.turn)[1]
        return size, centre, float(fat_mm3.sum())

    #*********#
    #   fid   #
    #*********#
    def fid(self, voxel, truth, fids, npoints, dwell_s, start_s):
        spur = truth.spurious
        if spur.amplitude <= 0.0:
            return None
        region = self.region(voxel)
        edge_mm = float(np.max(voxel.size_mm))
        design = voxel.design(truth)
        crushed = self.crushed(design, edge_mm)
        if region is None or crushed is None:
            return None
        size, centre, fat_mm3 = region
        engine, f0_mhz = voxel.scan.engine, voxel._f0_mhz()
        gamma_mhz_t = f0_mhz / engine.b0_t
        groups = ((WaterSource.system(gamma_mhz_t),), Skull.systems(gamma_mhz_t))
        placed = voxel.shift_mm or {}
        shift = {axis: float(centre[k]) + placed.get(axis, 0.0) for k, axis in enumerate("xyz")}
        # the region's field spread as a comb of offsets: a Gaussian of the drawn echo
        # width, spaced so that the comb's aliases (a copy of every pathway's echo one
        # period later) fall beyond the readout
        sigma_hz = np.sqrt(float(spur.width_s2) / 2.0) / np.pi
        spacing_hz = 1.0 / (start_s + npoints * dwell_s + float(design.duration_s))
        half = int(np.ceil(self.SPAN * sigma_hz / spacing_hz))
        rows = spacing_hz * np.arange(-half, half + 1)
        weights = np.exp(-0.5 * (rows / sigma_hz) ** 2)
        weights = weights / weights.sum()
        t1 = (voxel.resolve(truth.water.t1_s), float(truth.lipids.t1_s))
        t2 = (voxel.resolve(truth.water.t2_s), float(truth.lipids.t2_s))
        args = [ops.convert_to_tensor(np.asarray(values, dtype=np.float32)) for values in (
            [t1] * rows.size, [t2] * rows.size, [voxel.b1_scale(truth)] * rows.size,
            voxel.offset_hz(truth) + rows)]
        # in units of the nominal voxel volume, as every source renders
        water_mm = (tissue.MRSI_CHALLENGE["water_molarity_mm"] * self.SCALP_WATER
                    * (float(np.prod(size)) - fat_mm3) / edge_mm ** 3)
        lipid_mm = float(truth.lipids.amplitude_mm) * fat_mm3 / edge_mm ** 3
        echo = np.zeros(npoints, dtype=np.complex64)
        for sign, played in ((1.0, crushed), (-1.0, design)):
            replay, bounds = engine.replay_draws(played.confined(size).as_whole(), groups,
                                                 isochromats=self.isochromats, shift_mm=shift)
            freqs, amps = (np.asarray(ops.convert_to_numpy(v)) for v in replay(*args))
            water = np.arange(amps.shape[1]) < bounds[1]
            lw = np.where(water, 1.0 / (np.pi * t2[0]), 1.0 / (np.pi * t2[1])).astype(np.float32)
            scale = np.where(water, water_mm, lipid_mm).astype(np.complex64)
            for row, (offset_hz, weight) in enumerate(zip(rows, weights)):
                echo += np.complex64(sign * weight) * voxel._render_lines(
                    (freqs[row] + np.float32(offset_hz), amps[row] * scale, lw),
                    npoints, dwell_s, start_s)
        return echo
