####################################################################################################
#                                             voxel.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: One placed voxel: realized parameters, weighted component lines, renderings, and        #
#          NIfTI-MRS out. The seed and receiver-noise helpers every acquisition path shares        #
#          live here too.                                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.special
from keras import ops

from mrsandbox.database.tissue import CADY_1995
from mrsandbox.params import Params, Tissues
from mrsandbox.phantom.volume import Volume
from mrsandbox.scan.calibrations import calibrations
from mrsandbox.sequences.events import ShapedPulse


#**********************#
#   acquisition seed   #
#**********************#
def acquisition_seed(seed):
    """A serializable seed for an acquisition header: an int passes through, None draws
    a fresh one, so the recorded value always regenerates the file. A Generator is
    refused because its state cannot be recorded; use fid() for generator-driven draws.

    A fresh seed stays below 2**53 so every JSON reader keeps it exact (readers that
    parse integers as doubles round anything above), not only Python's."""
    if isinstance(seed, np.random.Generator):
        raise TypeError("acquire needs a serializable seed (an int, or None for a fresh "
                        "one) so the file can regenerate itself; use fid() for "
                        "generator-driven draws")
    return int(np.random.SeedSequence().entropy) % 2 ** 53 if seed is None else int(seed)


#***********************#
#   subject and noise   #
#***********************#
def draw(params, seed, step=0):
    """One subject and its receiver noise from one seed: (seed, truth, rng). The seed
    (an int, or None for a fresh recorded one) is the root of every leaf's path-keyed
    draw, so sample(seed), spectrum(seed), fid(seed), and acquire(seed) all mean the
    same subject and a seed read from a header regenerates it anywhere; the noise
    generator is the seed's own spawned stream number `step`, so drawing noise never
    reaches the leaves and a loop acquiring one subject many times takes fresh noise
    at every step."""
    seed = acquisition_seed(seed)
    return seed, params.sample(seed), np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(int(step),)))


#***************#
#   nifti mrs   #
#***************#
def nifti_mrs(data, dwell_s, hdr, **kwargs):
    """Simulated FIDs (the time axis fourth) as NIfTI-MRS in the standard's frame. The engine
    renders with 0 ppm on the carrier; the standard has no field for that and every reader
    (FSL-MRS, Osprey, spec2nii's own tools) puts the receiver's centre of a proton spectrum at
    4.65 ppm, as a scanner with water on resonance stores it. The FIDs are moved there, so a
    line the engine has at 2.01 ppm is read at 2.01 ppm from the file."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext

    data = np.asarray(data)
    turn = np.exp(-2j * np.pi * 4.65 * float(hdr.to_dict()["SpectrometerFrequency"][0])
                  * np.arange(data.shape[3]) * dwell_s)
    return gen_nifti_mrs_hdr_ext(
        (data * turn.reshape((1, 1, 1, -1) + (1,) * (data.ndim - 4))).astype(np.complex64),
        dwell_s, hdr, **kwargs)


#*****************#
#   noise scale   #
#*****************#
# the reference acquisition the drawn noise sigma belongs to: one transient of a 20 mm
# cube (8 mL) at 3 T sampled at 5 kHz (the calibration's receiver bandwidth); SNR
# scales with the voxel volume, with the field (body-noise dominated, stated), and
# with the square root of the averages, so the per-point sigma scales inversely, and
# the receiver's noise is white, so the per-point sigma grows with the square root of
# the bandwidth (the noise of a spectrum of fixed duration does not depend on it)
NOISE_REFERENCE = {"volume_ml": 8.0, "b0_t": 3.0, "bandwidth_hz": 5000.0}


def noise_scale(volume_ml, b0_t, dwell_s, averages=1):
    """The factor on the drawn sigma for an acquisition of `volume_ml` at `b0_t` tesla
    sampled every `dwell_s` and averaged `averages` times, one at the reference."""
    return ((NOISE_REFERENCE["volume_ml"] / volume_ml) * (NOISE_REFERENCE["b0_t"] / b0_t)
            * np.sqrt(1.0 / (dwell_s * NOISE_REFERENCE["bandwidth_hz"]) / averages))


#********************#
#   receiver noise   #
#********************#
def receiver_noise(rng, sigma, npoints, receiver=None, coils=None):
    """One draw of the receiver's complex Gaussian noise: standard deviation `sigma`
    per real and imaginary FID point, complex64; with `coils` the array's correlated
    channels, (n_coils, npoints). With a hardware.Receiver the white noise is drawn
    on the receiver's grid at the same density (the root of the oversampling more per
    point) and passes its filter."""
    n = npoints if receiver is None else receiver.noise_points(npoints)
    scale = sigma if receiver is None else sigma * np.sqrt(receiver.oversampling)
    if coils is not None:
        noise = coils.noise(rng, scale, n)
    else:
        noise = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return noise if receiver is None else receiver.colour(noise)


#**************************************************************************************************#
#                                            Class Voxel                                           #
#**************************************************************************************************#
#                                                                                                  #
# One placed voxel: realized parameters, weighted lines, renderings, and NIfTI-MRS out.            #
#                                                                                                  #
#**************************************************************************************************#
class Voxel:
    """One placed voxel: realized parameters, weighted lines, renderings, and NIfTI-MRS out.

    Metabolite amplitudes scale with the sampled concentration split per tissue by the
    drawn GM to WM contrast r: gm 2r/(1+r) + wm 2/(1+r), so an equal-mix voxel carries
    exactly the drawn value and pure tissues realize the published contrast; the
    linewidth follows the sampled T2 as 1 / (pi T2).

    Every realized leaf the render reads (concentrations, metabolite and water
    relaxation, diffusion) resolves at this voxel through `resolve`: scalars pass,
    per-tissue values weigh by the local composition, and spatial maps read their
    overlap-weighted mean over the box, so one truth serves every position.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, scan, position_mm, size_mm, compiled=False, box_center_mm=None,
                 box_axes="xyz", shim_at_mm=None, turn=None):
        self.scan = scan
        # a turned voxel: its rotation from its own frame (the design's axes) to the world's;
        # the subject is read over the turned box, the design plays in the voxel's frame as
        # ever (its eddy currents too, a stated approximation)
        self.turn = None if turn is None else np.asarray(turn, dtype=np.float64)
        # a cell of a lattice inside the box a localized design selects: the design is
        # played confined to the cell, its ensemble shifted from the box's centre
        # (along `box_axes`; the design keeps its own extent along the others, a slab
        # the lattice does not resolve)
        self.box_center_mm = (None if box_center_mm is None
                              else tuple(float(x) for x in box_center_mm))
        self.box_axes = box_axes
        # a head that moved after the shim: the shim of the planned place stays
        self.shim_at_mm = None if shim_at_mm is None else tuple(float(x) for x in shim_at_mm)
        self.position_mm = tuple(float(x) for x in position_mm)
        self.size_mm = (tuple(float(x) for x in size_mm) if hasattr(size_mm, "__len__")
                        else float(size_mm))
        self.fractions = scan.brain.fractions(position_mm, self.size_mm, self.turn)
        self._parts = None  # the fractions by region, read when a tree carries regions
        self.compiled = compiled
        self._kernels = {}
        self._field_offset_hz = 0.0  # the field group the sequence is being played for
        self._locals = {}
        self._transmit = 1.0 if scan.transmit is None else self.resolve(scan.transmit)

    #*************#
    #   resolve   #
    #*************#
    def resolve(self, value):
        """A realized leaf at this voxel: a Tissues weighted by the local fractions, a
        phantom.Volume read as its overlap-weighted mean over the box (cached; None
        outside the map resolves to 0.0), anything else unchanged."""
        if isinstance(value, Tissues):
            return value.weighted(self.fractions)
        if isinstance(value, Volume):
            if value not in self._locals:
                mean = value.mean(self.position_mm, self.size_mm, self.turn)
                self._locals[value] = 0.0 if mean is None else float(mean)
            return self._locals[value]
        return value

    def local(self, truth, path=()):
        """Every map-valued leaf of a realized tree resolved at this voxel, keyed by
        its dotted path, plus the transmit scale the pulses ran at when the scan
        carries a transmit map: the per-voxel truth the acquisition header records."""
        found = {}
        for key, value in truth.items():
            if isinstance(value, Params):
                found.update(self.local(value, path + (key,)))
            elif isinstance(value, Volume):
                found[".".join(path + (key,))] = self.resolve(value)
        if not path and self.scan.transmit is not None:
            found["b1_scale"] = self.b1_scale(truth)
        return found

    #********************#
    #   transmit frame   #
    #********************#
    def b1_scale(self, truth):
        """The transmit scale the pulses run at in this voxel: the drawn calibration
        scatter times the local relative B1+ of the scan's transmit map (1.0 without
        one, 0.0 outside it), the box mean."""
        return self.resolve(truth.b1_scale) * self._transmit

    def b1_profile(self, truth):
        """The transmit scale per isochromat, (isochromats,) float32: the drawn
        calibration scatter times the transmit map read at each isochromat's
        position (the ensemble of the scan's design, Engine.positions, about the
        voxel centre; a map cell missing at a position reads the box mean); None
        without a transmit map or without an ensemble (a design without gradients),
        where the box mean `b1_scale` is all there is."""
        positions = self.scan.engine.positions(self.scan.sequence)
        if self.scan.transmit is None or not positions:
            return None
        count = len(next(iter(positions.values())))
        offsets = np.zeros((count, 3), dtype=np.float64)
        for axis, values in positions.items():
            offsets[:, "xyz".index(axis)] = values
        if self.turn is not None:  # the ensemble lies along the voxel's own axes
            offsets = offsets @ self.turn.T
        local = [self.scan.transmit.nearest(np.asarray(self.position_mm) + offset)
                 for offset in offsets]
        values = np.asarray([self._transmit if v is None else float(v) for v in local])
        return (self.resolve(truth.b1_scale) * values).astype(np.float32)

    def b1_input(self, truth):
        """What the engine's graphs take as the transmit scale: the per-isochromat
        profile where there is one, else the box mean."""
        profile = self.b1_profile(truth)
        return self.b1_scale(truth) if profile is None else profile

    def offset_hz(self, truth):
        """The frequency offset the sequence runs at in this voxel: the drawn scanner
        frequency error, which the selective, shaped, and suppression pulses see; the
        readout shifts by the same value afterwards (the acquisition modulation)."""
        return self.resolve(truth.acquisition.shift_hz) - self._field_offset_hz

    def water_shift_hz(self, truth):
        """How far the water line sits below its typed place at the drawn tissue temperature,
        in hertz (the thermometry calibration, tissue.CADY_1995; the metabolites take theirs
        in their systems, `_metabolite_lines`): the water's sequence runs at the voxel's
        offset plus this, so a train sees the warm water where it is, and its lines move
        down by it afterwards (`_water_parts`)."""
        return ((self.resolve(truth.water.temperature_c) - 37.0) / CADY_1995["c_per_ppm"]
                * self._f0_mhz())

    def frame(self, truth):
        """The resolved frame the metabolite lines and basis are rendered in at this
        voxel: (the metabolites present, T1 per metabolite, T2 per metabolite, transmit,
        offset_hz), the metabolites of truth.conc whose concentration is not zero here (one
        at 0 mM costs no simulation and has no lines), every value resolved here, the
        transmit being `b1_scale` or, with a per-isochromat profile, (b1_scale, the
        profile's bytes). Together
        with the scan's `sequence.structure()` and timings (and, under a gradient
        response, this voxel's position, `_design_key`) it is the complete key of a
        rendered basis, so a caller caching bases across voxels or draws keys them
        by this tuple."""
        names = tuple(name for name, concentration in truth.conc.items()
                      if float(np.max(np.abs(self.resolve(concentration)))) > 0.0)
        profile = self.b1_profile(truth)
        transmit = (self.b1_scale(truth) if profile is None
                    else (self.b1_scale(truth), profile.tobytes()))
        return (names, tuple(float(self.resolve(truth.t1[name])) for name in names),
                tuple(float(self.resolve(truth.t2[name])) for name in names),
                transmit, self.offset_hz(truth))

    #***********#
    #   noise   #
    #***********#
    @property
    def volume_ml(self):
        """The voxel volume in millilitres."""
        size = np.asarray(self.size_mm, dtype=np.float64) * np.ones(3)
        return float(np.prod(size)) / 1000.0

    def noise_sigma(self, truth, dwell_s, averages=1):
        """The per-point receiver noise of this voxel's acquisition: the drawn sigma
        (one transient of the reference 8 mL voxel at 3 T and 5 kHz) scaled by the
        volume, the field, the receiver bandwidth, and the averages (noise_scale)."""
        return (truth.noise.sigma * self.noise_place()
                * noise_scale(self.volume_ml, self.scan.engine.b0_t, dwell_s, averages))

    def noise_place(self):
        """The scan's noise map at this voxel over the map at the place the noise prior
        belongs to (VirtualScan noise_map): an array's combined noise, or a measured
        noise level's box mean; one without a map, with channels, or outside it."""
        if "noise_place" not in self._kernels:
            noise, factor = self.scan.noise_map, 1.0
            if noise is not None and self.scan.coils is None:
                if hasattr(noise, "noise_gain"):
                    factor = noise.noise_gain(self.position_mm, self.size_mm, self.turn)
                else:
                    here = noise.mean(self.position_mm, self.size_mm, self.turn)
                    there = noise.mean(self.scan.noise_reference_mm, self.size_mm)
                    if here is not None and there is not None and min(here, there) > 0.0:
                        factor = float(here) / float(there)
            self._kernels["noise_place"] = factor
        return self._kernels["noise_place"]

    #****************#
    #   broadening   #
    #****************#
    def gauss_hz(self, truth, tissue=None):
        """The Gaussian broadening this voxel plays: the drawn lineshape.gauss_hz less, in
        quadrature, the width the scan's field map alone gives a line here after the shim.
        The width records and ranges are read against twins without a map, as the whole
        broadening beyond relaxation, so under a map the voxel plays only what the map does
        not; a map wider than the draw leaves none. `tissue` names one tissue's own: grey
        and white part the voxel's by the drawn lineshape.contrast as the levels part by
        theirs (an equal mix plays the draw, pure tissues the ratio), the CSF plays its
        drawn fraction of it (lineshape.csf)."""
        gauss = float(self.resolve(truth.lineshape.gauss_hz))
        if self.scan.field is not None:
            if "map width" not in self._kernels:
                width = self.scan.field.linewidth(self.position_mm, self.size_mm,
                                                  shim_at_mm=self.shim_at_mm, turn=self.turn)
                self._kernels["map width"] = max(width - 1.0, 0.0)     # less the Lorentzian's
            gauss = float(np.sqrt(max(gauss ** 2 - self._kernels["map width"] ** 2, 0.0)))
        if tissue is None:
            return gauss
        if tissue == "csf":
            return gauss * float(truth.lineshape.get("csf", 1.0))
        ratio = float(truth.lineshape.get("contrast", 1.0))
        return gauss * (2.0 * ratio if tissue == "gm" else 2.0) / (1.0 + ratio)

    #*****************#
    #   realization   #
    #*****************#
    def sample(self, seed=None):
        """Realize the scan's Params tree for this draw; the result is the ground truth."""
        return self.scan.params.sample(seed)

    #****************#
    #   components   #
    #****************#
    def components(self, truth, metabolites=True):
        """Every signal component as (freqs, amps, lw_hz, gauss_hz): metabolites, water,
        lipids, and the macromolecule background.

        Every metabolite is its own component with its own pooled T1 and T2 (linewidth
        and echo attenuation stay consistent), and water and lipids carry theirs;
        renderings flatten all components into one pruned line list and render once.
        With `metabolites` False the metabolite parts are left out, for a caller that
        forms them from a per-truth basis (`metabolite_basis`). A metabolite's grey and
        white lines (`_grouped`, apart where the tissues' widths part) are one component
        here, their amplitudes added.
        """
        flat = {}
        for (name, _), parts in self._grouped(truth, metabolites).items():
            if name == "metabolites" and name in flat:
                flat[name] = [(f, a + b, lw, g) for (f, a, lw, g), (_, b, _, _)
                              in zip(flat[name], parts)]
            else:
                flat.setdefault(name, []).extend(parts)
        return [part for parts in flat.values() for part in parts]

    def _water_parts(self, truth, engine, sequence, f0_mhz, faded, source=None):
        """The water's line parts at the offset the voxel is being played at, each with
        its tissue (one per tissue under a per-tissue draw, each at its own T2 and fading
        with its own ADC; None for the voxel's), [(part, tissue)] or None; from `source`
        (a phantom.WaterSource) in place of the scan's."""
        water = (source or self.scan.water).lines(engine, sequence, self, truth, f0_mhz)
        if water is None:
            return None
        transferred = self.transfer(truth).get("water", 1.0)
        adc = truth.diffusion.water
        # the water line moves with the temperature (the sequence saw it there, `water_shift_hz`)
        shift = np.float32(self.water_shift_hz(truth))
        return [((w_freqs - shift,
                  w_amps * np.complex64(faded(getattr(adc, part) if part and isinstance(adc, Tissues)
                                              else adc) * transferred),
                  1.0 / (np.pi * t2_s), 0.0), part) for w_freqs, w_amps, t2_s, part in water]

    def _grouped(self, truth, metabolites=True):
        """The parts of `components` grouped by source and by the Gaussian width their
        Lorentzian lines render under, {(name, gauss_hz): [parts]} in the order
        metabolites, water, oov (the water the localization leaks), lipids, mm,
        downfield: the metabolites and the water by tissue where the tissues' widths
        part (`_metabolite_widths`), the rest at the voxel's; silent sources are left
        out."""
        engine, sequence = self.scan.engine, self.design(truth)
        f0_mhz = self._f0_mhz()
        faded = self._diffusion_factor(truth)
        # an empirical downfield source carries everything above its cutoff, so the
        # spin systems' exchangeable NH lines up there leave the render (see
        # phantom.downfield.DownfieldSource)
        cutoff = self.scan.downfield.replaces_above_ppm
        groups, gauss_hz = {}, self.gauss_hz(truth)
        # the metabolites' graphs replay in their pool while this thread prepares the
        # other sources (the transfer, the water, the macromolecules: eager work of their
        # own); the groups come out in the order metabolites, water, oov, lipids, mm,
        # downfield as before
        with ThreadPoolExecutor(1) as ahead:
            lines = ahead.submit(self._metabolite_lines, truth) if metabolites else None
            widths = ({tissue: (width, self.metabolite_weights(truth, tissue))
                       for tissue, width in self._metabolite_widths(truth).items()}
                      if metabolites else {})
            rest = self._other_groups(truth, engine, sequence, f0_mhz, faded, gauss_hz)
            lines = lines.result() if metabolites else {}
        if metabolites:
            for tissue, (width, weights) in widths.items():
                parts = []
                for name, weight in zip(truth.conc, weights):
                    # a metabolite absent here keeps its place, with no lines
                    freqs, amps = lines.get(name, (np.zeros(0, np.float32),
                                                   np.zeros(0, np.complex64)))
                    if cutoff is not None:
                        amps = np.where(freqs > cutoff * f0_mhz, np.complex64(0.0), amps)
                    parts.append((freqs, amps * weight,
                                  1.0 / (np.pi * self.resolve(truth.t2[name])), 0.0))
                groups[("metabolites", width)] = parts
        groups.update(rest)
        return groups

    def _other_groups(self, truth, engine, sequence, f0_mhz, faded, gauss_hz):
        """`_grouped`'s sources beside the metabolites, in their order."""
        groups = {}
        water = self._water_parts(truth, engine, sequence, f0_mhz, faded)
        if water is not None:
            for part, tissue in water:
                groups.setdefault(("water", self.gauss_hz(truth, tissue)), []).append(part)
            residual = sum(complex(np.sum(np.asarray(ops.convert_to_numpy(a))))
                           for (_, a, _, _), _ in water)
            if truth.oov.fraction > 0.0 and residual != 0.0:
                # the water outside the voxel that the localization leaks: one broad line at
                # the water's frequency (its own offset), a fraction of the full water at
                # t = 0 after the echo's decay, at a drawn phase against the voxel's own
                # residual at t = 0; a residual forced to nothing takes the leak with it (the
                # console's switch, the silent gates)
                from mrsandbox.phantom.macromolecules import _echo_time_s
                from mrsandbox.phantom.water import WaterSource

                freqs, amps = (np.asarray(ops.convert_to_numpy(v)) for v in water[0][0][:2])
                at = (float(np.sum(freqs * np.abs(amps)) / np.sum(np.abs(amps)))
                      + float(truth.oov.offset_hz))
                full = WaterSource.full_amplitude_mm(self) * np.exp(
                    -(_echo_time_s(sequence) or 0.0) / self.resolve(truth.water.t2_s))
                amplitude = float(truth.oov.fraction) * full * np.exp(
                    1j * (np.angle(residual) + np.radians(float(truth.oov.phase_deg))))
                groups[("oov", gauss_hz)] = [
                    (np.array([at], np.float32), np.array([amplitude], np.complex64),
                     1.0 / (np.pi * float(truth.oov.t2star_s)), 0.0)]
        lipid = self.scan.lipids.lines(engine, sequence, self, truth, f0_mhz)
        if lipid is not None:
            l_freqs, l_amps = lipid
            groups[("lipids", gauss_hz)] = [
                (l_freqs, l_amps * np.complex64(faded(truth.diffusion.lipids)),
                 1.0 / (np.pi * truth.lipids.t2_s), 0.0)]
        mm = self.scan.mm.lines(engine, sequence, self, truth, f0_mhz)
        if mm is not None:
            # real positive amplitudes render absorptive as they are; the engine's
            # receiver factor compensates density matrix phases the MM lines never had
            freqs, amps, gauss = mm
            groups[("mm", gauss_hz)] = [(ops.convert_to_tensor(freqs),
                                         ops.convert_to_tensor(amps * faded(truth.diffusion.mm)),
                                         0.0, ops.convert_to_tensor(gauss))]
        df = self.scan.downfield.lines(engine, sequence, self, truth, f0_mhz)
        if df is not None:
            freqs, amps, gauss = df
            groups[("downfield", gauss_hz)] = [(ops.convert_to_tensor(freqs),
                                                ops.convert_to_tensor(amps),
                                                0.0, ops.convert_to_tensor(gauss))]
        return groups

    #*************#
    #   weights   #
    #*************#
    def metabolite_weights(self, truth, tissue=None):
        """The complex weight of every metabolite's unit lines in this voxel, in
        truth.conc order: the resolved concentration times the voxel's share of it
        (`share`, one tissue's with `tissue`), times the diffusion attenuation of the
        local ADC under the drawn diffusion model."""
        faded = self._diffusion_factor(truth)
        sticks = truth.diffusion.model == "sticks"
        transferred = self.transfer(truth)
        weights = []
        for name, concentration in truth.conc.items():
            weights.append(self.resolve(concentration) * self.share(truth, name, tissue)
                           * faded(truth.diffusion.adc[name], sticks)
                           * transferred.get(name, 1.0))
        return np.asarray(weights, dtype=np.complex64)

    def _metabolite_widths(self, truth):
        """The Gaussian width the metabolite lines render under here, {tissue: Hz}: the grey
        and the white lines apart where the voxel holds both and the drawn contrast parts
        their widths, else one entry (None, every line) at the one width."""
        widths = {tissue: self.gauss_hz(truth, tissue) for tissue in ("gm", "wm")
                  if getattr(self.fractions, tissue) > 0.0}
        if len(set(widths.values())) == 2:
            return widths
        return {None: next(iter(widths.values()), self.gauss_hz(truth))}

    def share(self, truth, name, tissue=None):
        """How much of a metabolite's drawn concentration this voxel holds: its tissue split by
        the drawn GM to WM contrast r as gm 2r/(1+r) + wm 2/(1+r) (an equal-mix voxel carries
        exactly the drawn value, pure tissues the published contrast), each region's part
        scaled by the metabolite's factors there where the tree carries regions
        (database.regional) and the brain has them, times the within-region texture here
        (`texture`); `tissue` takes the grey or the white term alone."""
        r = truth.contrast[name]
        factors = truth.get("regions", {}).get(name)
        if factors and self._parts is None:
            self._parts = self.scan.brain.parts(self.position_mm, self.size_mm, self.turn)
        grain = self.texture(truth).get(name, 1.0)
        unit = Tissues(1.0, 1.0)
        parts = self._parts if factors and self._parts else {None: self.fractions}
        held = {"gm": sum(2.0 * r * part.gm * (factors or {}).get(region, unit).gm
                          for region, part in parts.items()),
                "wm": sum(2.0 * part.wm * (factors or {}).get(region, unit).wm
                          for region, part in parts.items())}
        return grain * (held[tissue] if tissue else sum(held.values())) / (1.0 + r)

    def texture(self, truth):
        """Each metabolite's within-region factor at this voxel's centre, {name: factor}: one
        plus the drawn spread (texture.sd) times a smooth random field of unit variance and
        the drawn scale (a dozen cosines in directions and phases seeded by texture.seed,
        every metabolite's its own), read here alone, never as a volume; empty, every factor
        one, where the spread is zero or the anatomy has no regions (a phantom, a bare
        brain)."""
        grains = getattr(truth, "texture", None)
        if grains is None or float(grains.sd) <= 0.0 or self.scan.brain.regions is None:
            return {}
        centre = tuple(self.box_center_mm or self.position_mm)
        held, key = self._lines_cache(truth), ("texture", centre)
        if key not in held:
            names, modes = tuple(truth.conc), 12
            rng = np.random.default_rng(int(grains.seed))
            directions = rng.standard_normal((len(names), modes, 3))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            phases = rng.uniform(0.0, 2.0 * np.pi, (len(names), modes))
            field = np.cos(directions @ (np.asarray(centre) * 2.0 * np.pi / float(grains.scale_mm))
                           + phases).sum(axis=1) * np.sqrt(2.0 / modes)
            held[key] = dict(zip(names, 1.0 + float(grains.sd) * field))
        return held[key]

    #**************#
    #   transfer   #
    #**************#
    def transfer(self, truth):
        """What magnetization transfer through the tissue's bound pool leaves of each metabolite
        and of the water in this voxel under the design's saturation pulses (phantom.transfer),
        {name: factor} with "water": empty where the tree carries no transfer or the voxel no
        bound pool (CSF, a solution). The pulses' responses stay on the scan per design and
        transmit scale, read to two percent (stated)."""
        mt = getattr(truth, "mt", None)
        fraction = 0.0 if mt is None else float(self.resolve(mt.pool_fraction))
        if fraction <= 0.0:
            return {}
        from mrsandbox.phantom.transfer import Transfer
        from mrsandbox.phantom.water import WaterSource

        rates = {name: float(self.resolve(rate)) for name, rate in mt.rate.items()
                 if name in truth.conc}
        model = Transfer(fraction, float(self.resolve(mt.exchange_hz)),
                         round(float(self.resolve(mt.t2_s)), 7))
        return model.factors(
            self.scan.engine, self.design(truth),
            WaterSource.system(self._f0_mhz() / self.scan.engine.b0_t),
            float(self.resolve(truth.water.t1_s)), rates,
            {name: float(self.resolve(truth.t1[name])) for name in rates},
            b1_scale=round(float(np.mean(self.b1_scale(truth))) / 0.02) * 0.02,
            kept=self.scan._transfer)

    #***********#
    #   basis   #
    #***********#
    def metabolite_basis(self, truth, npoints, dwell_s):
        """The unit FIDs of every metabolite on a time grid, (K, npoints) complex64 in
        truth.conc order: each metabolite's lines at its resolved linewidth with the
        downfield cutoff applied, rendered in one call (Engine.render_fids) and cached
        per truth on the scan under this voxel's resolved frame, so a lattice of voxels
        sharing one truth renders the metabolites once and forms every cell as
        `metabolite_weights` @ basis. The frame is `frame(truth)` plus the grid."""
        lines = self._metabolite_lines(truth)
        names = tuple(truth.conc)
        start_s = float(truth.acquisition.delay_s)
        frame = self.frame(truth) + (npoints, float(dwell_s), start_s) + self._design_key(truth)
        cached = self.scan._basis
        if cached is None or cached[0] is not truth:
            cached = self.scan._basis = (truth, {})
        if frame in cached[1]:
            return cached[1][frame]
        cutoff = self.scan.downfield.replaces_above_ppm
        f0_mhz = self._f0_mhz()
        t2 = dict(zip(frame[0], frame[2]))
        freqs, amps, lw, rows = [], [], [], []
        for k, name in enumerate(names):
            if name not in lines:              # absent: its row stays zero
                continue
            f, a = lines[name]
            if cutoff is not None:
                a = np.where(f > cutoff * f0_mhz, np.complex64(0.0), a)
            keep = np.abs(a) > 1e-7 * np.abs(a).max() if a.size else np.zeros(0, bool)
            freqs.append(f[keep])
            amps.append(a[keep])
            lw.append(np.full(int(keep.sum()), t2[name], dtype=np.float32))
            rows.append(np.full(int(keep.sum()), k, dtype=np.int64))
        freqs, amps, rows = np.concatenate(freqs), np.concatenate(amps), np.concatenate(rows)
        lw = 1.0 / (np.pi * np.concatenate(lw))
        amps = amps * self._advance(freqs, lw, start_s)
        # one amplitude row per metabolite over the shared line list, padded to a
        # stable bucket so the jitted graph keeps its shapes across draws
        pad = -(-freqs.size // 512) * 512 - freqs.size
        matrix = np.zeros((len(names), freqs.size + pad), dtype=np.complex64)
        matrix[rows, np.arange(freqs.size)] = amps
        _, fids = self.scan.engine.render_fids(
            ops.convert_to_tensor(np.pad(freqs, (0, pad))), ops.convert_to_tensor(matrix),
            npoints=npoints, dwell_s=dwell_s,
            lw_hz=ops.convert_to_tensor(np.pad(lw, (0, pad), constant_values=1.0)))
        basis = np.asarray(ops.convert_to_numpy(fids), dtype=np.complex64)
        cached[1][frame] = basis
        return basis

    #************#
    #   fading   #
    #************#
    def _diffusion_factor(self, truth):
        """The diffusion attenuation of a component for this scan's b-value, given its
        ADC leaf (per-tissue leaves weighted by the local composition, scalars passing
        through); the identity at b = 0.

        The Gaussian form exp(-b ADC) is every component's default. With `sticks`
        the component is an intracellular metabolite in the consensus minimal model
        (Ligneul et al. 2024, MRM 91:860, section 5.1: randomly oriented infinitely
        long cylinders of zero radius, the sticks whose powder average is
        S = sqrt(pi / (4 b D)) erf(sqrt(b D)) with D the intracellular diffusivity along
        the fibre), so the decay is non-monoexponential above b of 3 to 5 ms/um^2 as
        measured. The leaf stays the low-b ADC, which is D / 3 for uniformly oriented
        sticks (the consensus's own identity), so the ADC tables and every b = 0 or
        low-b result keep their meaning and D = 3 ADC.
        """
        b = self.scan.b_value_ms_um2

        def faded(adc, sticks=False):
            if b == 0.0:
                return 1.0
            x = b * self.resolve(adc)
            if not sticks or x <= 0.0:
                return float(np.exp(-x))
            return float(np.sqrt(np.pi / (12.0 * x)) * scipy.special.erf(np.sqrt(3.0 * x)))

        return faded

    #****************#
    #   renderings   #
    #****************#
    def spectrum(self, seed=None, npoints=4096, sw_hz=4000.0):
        """Quick look: (ppm, complex spectrum), the FFT of the noise-free FID.

        One render path for everything: the spectrum is the Fourier transform of the
        same clean FID the acquisitions use, so B0 dephasing and every other time
        domain effect shows up here too. The dwell scaling keeps peak heights at the
        analytic Lorentzian values.
        """
        truth = self.sample(seed)
        dwell_s = 1.0 / sw_hz
        _, fid = self._clean_fid(truth, npoints, dwell_s)
        spec = np.fft.fft(fid) * dwell_s
        grid = np.fft.fftfreq(npoints, d=dwell_s) % sw_hz
        order = np.argsort(grid)
        return grid[order] / self._f0_mhz(), spec[order]

    def fid(self, seed=None, npoints=2048, dwell_s=1.0 / 4000.0, averages=1):
        """(t, complex FID) at any grid: all components summed, the sampled noise of
        `averages` averaged transients added."""
        if isinstance(seed, np.random.Generator):
            rng, truth = seed, self.scan.params.sample(seed)
        else:
            _, truth, rng = draw(self.scan.params, seed)
        t, fid = self._render(truth, rng, npoints, dwell_s, averages)
        return np.asarray(ops.convert_to_numpy(t)), fid

    #*****************#
    #   realization   #
    #*****************#
    def _render(self, truth, rng, npoints, dwell_s, averages=1):
        """Render all components in one call and add the sampled complex Gaussian noise
        of `averages` averaged transients."""
        t, fid = self._clean_fid(truth, npoints, dwell_s)
        sigma = self.noise_sigma(truth, dwell_s, averages)
        if sigma > 0.0:
            fid = fid + receiver_noise(rng, sigma, npoints, self.scan.receiver)
        return t, fid

    def _clean_fid(self, truth, npoints, dwell_s, basis=None):
        """The noise-free voxel FID: the components of `_component_fids` summed."""
        grid = np.arange(npoints) * dwell_s
        t = ops.convert_to_tensor(grid.astype(np.float32))
        fid = np.zeros(npoints, dtype=np.complex64)
        for component in self._component_fids(truth, npoints, dwell_s, basis).values():
            fid = fid + component
        return t, fid

    def _component_fids(self, truth, npoints, dwell_s, basis=None):
        """Every component's noise-free FID as the receiver delivers it: the rendered
        components of `_rendered_fids`, with a hardware.Receiver on the scan rendered
        on the receiver's oversampled grid and each filtered and decimated (a linear
        step, so the components still sum to the clean FID exactly)."""
        receiver = self.scan.receiver
        if receiver is None:
            return self._rendered_fids(truth, npoints, dwell_s, basis)
        fine_n, fine_dwell = receiver.grid(npoints, dwell_s)
        if basis is not None:
            basis = self.metabolite_basis(truth, fine_n, fine_dwell)
        return {name: receiver.decimate(fid, npoints)
                for name, fid in self._rendered_fids(truth, fine_n, fine_dwell, basis).items()}

    def _rendered_fids(self, truth, npoints, dwell_s, basis=None):
        """Every component's noise-free FID on the grid, {name: complex64} in the order
        metabolites, water, lipids, mm, downfield, spurious (silent ones left out):
        each source's lines rendered on their own, with the B0 dephasing, the drawn
        eddy-current phase, frequency shift, and zero-order phase applied to every
        one alike, and the spurious echo (drawn relative to the maximum of the summed
        FID without the water) as its own entry, so the components sum to the clean FID
        exactly.

        With `basis` (the per-truth metabolite unit FIDs of `metabolite_basis`) the
        metabolites enter as `metabolite_weights` @ basis and only the other sources
        render here; a multi-voxel acquisition renders the metabolites once that way.
        Line lists small enough not to be worth a dispatch render on the host in
        float64 (exact), the rest through the engine's jitted graphs.

        The spurious echo is the drawn AGNOSTIC-form echo (amplitude relative to the
        FID maximum, centre, Gaussian width, position, phase) rendered by Augmentrum's
        localized echo with its Gaussian envelope exp(-(t - t_echo)^2 / (2 T2^2)),
        so T2 = 1 / sqrt(2 W). The eddy phase is Augmentrum's synthetic trajectory
        (low-pass filtered Gaussian phase noise, its linear trend removed) with the
        drawn standard deviation, cut-off, and seed, so it regenerates from the truth;
        its phasor depends on the truth alone and is cached per truth; it regenerates on
        the readout's grid, so the same truth carries another trajectory at another
        npoints or dwell (stated).
        """
        try:
            from augmentrum.augmentation import EddyCurrent, SpuriousEchoes
        except ImportError as error:
            raise ImportError("acquisitions need Augmentrum for the spurious echo and eddy "
                              "phase (the mrsandbox[augment] extra; see pyproject.toml "
                              "for the install recipe)") from error
        # the receiver delay: every sample sits start_s after the readout's start
        start_s = float(truth.acquisition.delay_s)
        grid = start_s + np.arange(npoints) * dwell_s
        # the microscopic field distribution: a Gaussian envelope on the spin-system
        # components, each bucket of lines at its tissue's width (the measured MM and
        # downfield widths already carry theirs)
        gauss_hz = self.gauss_hz(truth)
        def sources(basis):
            fids = {}
            if basis is not None:
                fids["metabolites"] = sum(
                    (self.metabolite_weights(truth, tissue) @ basis).astype(np.complex64)
                    * self._envelope(width, grid)
                    for tissue, width in self._metabolite_widths(truth).items())
            held = {} if self._field_offset_hz else self._lines_cache(truth)
            ready = held.get(("metabolite fid", npoints, float(dwell_s), start_s))
            if ready is not None:               # rendered with the batch (`prepare`)
                fids["metabolites"] = ready
            if ready is not None and held.get(("every source", npoints, float(dwell_s), start_s)):
                for name in ("water", "oov", "lipids", "mm", "downfield"):
                    if (name, npoints, float(dwell_s), start_s) in [
                            key[1:] for key in held if key[0] == "fid"]:
                        fids[name] = held[("fid", name, npoints, float(dwell_s), start_s)]
                grouped = {}
            else:
                grouped = self._grouped(truth, metabolites=basis is None and ready is None)
            for (name, width), parts in grouped.items():
                lorentzian, voigt = self._flat_lines(parts)
                fid = fids.get(name, np.zeros(npoints, dtype=np.complex64))
                if lorentzian is not None:
                    fid = fid + (self._render_lines(lorentzian, npoints, dwell_s, start_s)
                                 * self._envelope(width, grid))
                if voigt is not None:
                    fid = fid + self._render_lines(voigt, npoints, dwell_s, start_s)
                fids[name] = fid
            mm_fid = self.scan.mm.fid(self, truth, self._f0_mhz(), grid,
                                      sequence=self.design(truth))
            if mm_fid is not None:
                fids["mm"] = fids.get("mm", 0.0) + mm_fid * np.float32(
                    self._diffusion_factor(truth)(truth.diffusion.mm))
            return fids

        if self.scan.field is None:
            fids = sources(basis)
        elif self.scan.field_groups == 1:
            kernel = self._field_kernel(npoints, dwell_s, start_s)
            groups = [] if self.scan.water_groups <= 1 else list(self.scan.field.groups(
                self.position_mm, self.size_mm, grid, self.scan.water_groups,
                shim_at_mm=self.shim_at_mm, spread_hz=float(gauss_hz), turn=self.turn))
            if groups:
                # the water's rows at the voxel's and at every group's offset in one replay
                self.scan.water.prepare(self.scan.engine, self.design(truth), self, [truth],
                                        self._f0_mhz(), [0.0, *(mean for mean, _, _ in groups)])
            fids = {name: fid * kernel for name, fid in sources(basis).items()}
            if "water" in fids and groups:
                # the water alone at every field group's own offset under its own dephasing
                # (VirtualScan water_groups): a suppression pulse misses the water where the
                # field is off its band, which the voxel's mean cannot show. The microscopic
                # spread goes into the groups, not onto the result: a train is a band-stop a
                # few hertz wide, so what it leaves is narrower than the water's distribution
                engine, sequence, faded = (self.scan.engine, self.design(truth),
                                           self._diffusion_factor(truth))
                water = np.zeros(npoints, dtype=np.complex64)
                for mean_hz, weight, part_kernel in groups:
                    self._field_offset_hz = mean_hz
                    try:
                        lorentzian, _ = self._flat_lines([part for part, _ in self._water_parts(
                            truth, engine, sequence, self._f0_mhz(), faded)])
                    finally:
                        self._field_offset_hz = 0.0
                    if lorentzian is not None:     # a train can take a group's water whole
                        water = water + np.float32(weight) * self._render_lines(
                            lorentzian, npoints, dwell_s, start_s) * part_kernel
                fids["water"] = water
        else:
            # the field inside the pulses: every group of the voxel's field distribution
            # plays the sequence at its own mean offset and dephases by its own kernel
            fids = {}
            for mean_hz, weight, kernel in self.scan.field.groups(
                    self.position_mm, self.size_mm, grid, self.scan.field_groups,
                    shim_at_mm=self.shim_at_mm, turn=self.turn):
                self._field_offset_hz = mean_hz
                try:
                    part = sources(None if basis is None
                                   else self.metabolite_basis(truth, npoints, dwell_s))
                finally:
                    self._field_offset_hz = 0.0
                for name, fid in part.items():
                    fids[name] = fids.get(name, 0.0) + np.float32(weight) * fid * kernel
        kernel = self._gradient_kernel(truth, npoints, dwell_s, start_s)
        if kernel is not None:
            fids = {name: fid * kernel for name, fid in fids.items()}
        spur, acquisition = truth.spurious, truth.acquisition
        sw_hz = 1.0 / dwell_s
        if spur.amplitude > 0.0:
            # the out-of-voxel echo enters before the receiver-side modulation like
            # any received signal
            echo = SpuriousEchoes(mode="echo", echoes=[dict(
                alpha=spur.amplitude, t_echo=spur.tau_s - start_s,
                T2=1.0 / np.sqrt(2.0 * spur.width_s2), freq_hz=spur.ppm * self._f0_mhz(),
                phase_deg=spur.phase_deg + 360.0 * spur.ppm * self._f0_mhz() * start_s,
                gaussian_env=True)])
            # relative to the voxel's signal WITHOUT its water: signal from outside the voxel
            # does not grow with the water a suppression train leaves inside it (the source's
            # own FIDs carry next to none)
            total = sum((fid for name, fid in fids.items() if name != "water"),
                        np.zeros(npoints, dtype=np.complex64))
            fids["spurious"] = (np.asarray(echo.process_tensor(total[None, :], sw_hz=sw_hz)[0][0])
                                - total)
        modulation = np.exp(1j * (2.0 * np.pi * self.offset_hz(truth) * grid
                                  + np.radians(acquisition.phase_deg)))
        if acquisition.eddy_transient_rad:
            # the phase transient at the FID's start, exp(-t / tau) grown back from 5 ms, scaled
            # so that the reader's measure of it (calibrate._eddy_transient) is the leaf
            from mrsandbox.scan.calibrate import EDDY_AT_S, EDDY_LINE_S, _eddy_transient

            t = np.arange(max(npoints, int(round(EDDY_LINE_S[1] / dwell_s)) + 1)) * dwell_s
            shape = np.exp((EDDY_AT_S - t) / float(acquisition.eddy_transient_tau_s))
            modulation = modulation * np.exp(
                1j * float(acquisition.eddy_transient_rad) / _eddy_transient(shape, dwell_s)
                * shape[:npoints])
        if acquisition.eddy_std_rad > 0.0:
            key = (int(acquisition.eddy_seed), float(acquisition.eddy_std_rad),
                   float(acquisition.eddy_lp_cut_hz), npoints, float(dwell_s))
            cached = self.scan._eddy
            if cached is None or cached[0] is not truth:
                cached = self.scan._eddy = (truth, {})
            if key not in cached[1]:
                eddy = EddyCurrent(std_rad=acquisition.eddy_std_rad,
                                   lp_cut_hz=acquisition.eddy_lp_cut_hz,
                                   seed=int(acquisition.eddy_seed))
                ones = np.ones((1, npoints), dtype=np.complex64)
                cached[1][key] = np.asarray(eddy.process_tensor(ones, sw_hz=sw_hz)[0][0])
            modulation = modulation * cached[1][key]
        return {name: (fid * modulation).astype(np.complex64) for name, fid in fids.items()}

    @staticmethod
    def _envelope(gauss_hz, t):
        """The envelope a Gaussian line of FWHM `gauss_hz` puts on an FID over the times
        `t`, exp(-(pi w)^2 t^2 / 4 ln 2), float32."""
        return np.exp(-(np.pi * gauss_hz) ** 2 / (4.0 * np.log(2.0)) * t * t).astype(np.float32)

    @staticmethod
    def _advance(freqs, lw_hz, start_s):
        """The factor a Lorentzian line's amplitude takes over `start_s` of evolution,
        exp((i 2 pi f - pi lw) start_s): a render of the advanced amplitudes from zero
        is the render of the lines from `start_s` exactly."""
        rate = 1j * 2.0 * np.pi * freqs.astype(np.float64) - np.pi * np.asarray(lw_hz, np.float64)
        return np.exp(rate * start_s).astype(np.complex64)

    def _render_lines(self, bucket, npoints, dwell_s, start_s=0.0):
        """One line bucket, (freqs, amps, lw_hz[, gauss_hz]) as numpy, to a complex64
        FID sampled from `start_s`: on the host in float64 when the bucket holds at
        most 64 lines (exact, no dispatch), otherwise through the engine's jitted
        graphs from the advanced lines (a Gaussian line's cross term
        exp(-2 g start_s t) is a Lorentzian width, so the advance stays exact)."""
        freqs, amps, lw = bucket[0], bucket[1], bucket[2]
        gauss = bucket[3] if len(bucket) > 3 else None
        if freqs.size <= 64:
            t = start_s + np.arange(npoints, dtype=np.float64) * dwell_s
            rate = 1j * 2.0 * np.pi * freqs.astype(np.float64) - np.pi * lw.astype(np.float64)
            phase = rate[:, None] * t[None, :]
            if gauss is not None:
                g = (np.pi * gauss.astype(np.float64)) ** 2 / (4.0 * np.log(2.0))
                phase = phase - g[:, None] * t[None, :] ** 2
            return ((amps.astype(np.complex128)[:, None] * np.exp(phase)).sum(axis=0)
                    .astype(np.complex64))
        if start_s:
            amps = amps * self._advance(freqs, lw, start_s)
            if gauss is not None:
                g = (np.pi * gauss.astype(np.float64)) ** 2 / (4.0 * np.log(2.0))
                amps = (amps * np.exp(-g * start_s ** 2)).astype(np.complex64)
                lw = (lw + 2.0 * g * start_s / np.pi).astype(lw.dtype)
        tensors = [ops.convert_to_tensor(np.pad(a, (0, -(-freqs.size // 512) * 512 - freqs.size),
                                                constant_values=fill))
                   for a, fill in ((freqs, 0.0), (amps, 0.0), (lw, 1.0))]
        if gauss is None:
            _, fid = self.scan.engine.render_fid(tensors[0], tensors[1], npoints=npoints,
                                                 dwell_s=dwell_s, lw_hz=tensors[2])
        else:
            padded = ops.convert_to_tensor(
                np.pad(gauss, (0, -(-freqs.size // 512) * 512 - freqs.size)))
            _, fid = self.scan.engine.render_fid(tensors[0], tensors[1], npoints=npoints,
                                                 dwell_s=dwell_s, lw_hz=tensors[2],
                                                 gauss_hz=padded)
        return np.asarray(ops.convert_to_numpy(fid), dtype=np.complex64)

    def _field_kernel(self, npoints, dwell_s, start_s=0.0):
        """The cached intra-voxel B0 dephasing kernel of this voxel on this time grid."""
        key = (npoints, dwell_s, start_s)
        if key not in self._kernels:
            t = start_s + np.arange(npoints) * dwell_s
            self._kernels[key] = self.scan.field.kernel(self.position_mm, self.size_mm, t,
                                                        shim_at_mm=self.shim_at_mm,
                                                        turn=self.turn)
        return self._kernels[key]

    def _response(self, truth):
        """The gradient response this voxel plays its design and readout through, as
        (key, callable): the scan's measured one when it has one, else the
        exponential response of the drawn gradients leaves; None when every drawn
        term is zero."""
        from mrsandbox.scan.eddy import exponential_response

        if self.scan.girf is not None:
            return ("girf", self.scan.girf.name), self.scan.girf
        g = truth.gradients
        terms = ((float(g.tau_short_s), float(g.tau_long_s)),
                 (float(g.b0_short_hz_per_mt_m), float(g.b0_long_hz_per_mt_m)),
                 (float(g.linear_short), float(g.linear_long)))
        if not any(terms[1]) and not any(terms[2]):
            return None
        tr_s = self.scan.sequence.tr_s
        tr_s = None if tr_s is None else float(tr_s)
        return ("gradients", tr_s) + terms, exponential_response(*terms, tr_s=tr_s)

    def _scanner(self):
        """The scan's scanner with its limits (a bare field strength gets the generic
        limits)."""
        from mrsandbox.hardware import Scanner

        scanner = self.scan.scanner
        return scanner if hasattr(scanner, "max_grad_mt_m") else Scanner("custom", float(scanner))

    @property
    def shift_mm(self):
        """Where this voxel sits from the centre of the box its design selects,
        {axis: mm}; None for a voxel that is the box."""
        if self.box_center_mm is None:
            return None
        return {axis: self.position_mm[k] - self.box_center_mm[k] for k, axis in enumerate("xyz")}

    def _played(self):
        """The scan's design, confined to this voxel when it is a cell of a box."""
        if self.box_center_mm is None:
            return self.scan.sequence
        if "confined" not in self._kernels:
            edges = np.asarray(self.size_mm, dtype=np.float64) * np.ones(3)
            self._kernels["confined"] = self.scan.sequence.confined(
                [edge if axis in self.box_axes else None for axis, edge in zip("xyz", edges)])
        return self._kernels["confined"]

    def design(self, truth):
        """The scan's design as this voxel plays it: with the phase its own gradients
        induce before the readout at this position (scan.eddy.induced_sequence,
        through `_response`), the design itself without a response; cached per
        response. A cell of a box plays the design confined to itself."""
        from mrsandbox.scan.eddy import induced_sequence

        response = self._response(truth)
        if response is None:
            return self._played()
        key = ("design",) + response[0]
        if key not in self._kernels:
            self._kernels[key] = induced_sequence(self._played(), self._scanner(), response[1],
                                                  self.position_mm)
        return self._kernels[key]

    def _design_key(self, truth):
        """What a line or basis cache adds for this voxel's design: nothing while the
        voxel plays the scan's own design (no response, or one with nothing to act
        on), the position once it carries kicks (they depend on it) or once the voxel
        is a cell of a box (its place in the box decides what the design leaves)."""
        kicked = self._response(truth) is not None
        if self.box_center_mm is not None:
            # cells at one place in the box (every interior cell sits at its centre)
            # and of one size share their lines
            placed = ("cell", self.size_mm) + tuple(round(v, 6) for v in self.shift_mm.values())
            return placed + ((self.position_mm,) if kicked else ())
        return ((self.position_mm,) if kicked and self.design(truth) is not self.scan.sequence
                else ())

    def _gradient_kernel(self, truth, npoints, dwell_s, start_s=0.0):
        """The cached eddy-current kernel of this voxel's readout (scan.eddy), the
        design's gradients played within the scanner's limits through `_response`;
        None without a response."""
        from mrsandbox.scan.eddy import readout_kernel

        response = self._response(truth)
        if response is None:
            return None
        key = ("kernel", npoints, dwell_s, start_s) + response[0]
        if key not in self._kernels:
            self._kernels[key] = readout_kernel(self.scan.sequence, self._scanner(),
                                                self.position_mm, self.size_mm, npoints, dwell_s,
                                                response[1], start_s=start_s)
        return self._kernels[key]

    #*****************#
    #   acquisition   #
    #*****************#
    def acquire(self, seed=None, npoints=2048, dwell_s=1.0 / 4000.0, averages=1, step=0):
        """Acquire the voxel and return NIfTI-MRS carrying the full provenance.

        The header extension holds SimulationGroundTruth (seed, realized parameters,
        tissue fractions), SimulationSequence (name and timings), and SimulationPhantom
        (brain, position, size), so any spectrum can be regenerated from its own file.
        The data is the average of `averages` transients (the noise falls by their
        square root), and `step` selects the noise stream of the seed, so a loop
        acquiring one subject repeatedly draws fresh noise each time.

        With a CoilArray on the scan, every coil receives the FID weighted by its
        complex sensitivity at this voxel plus correlated noise (covariance sigma^2 R),
        and the data gains a DIM_COIL dimension.
        """
        seed, truth, rng = draw(self.scan.params, seed, step)
        coils = self.scan.coils
        sigma = self.noise_sigma(truth, dwell_s, averages)
        if coils is None:
            _, fid = self._render(truth, rng, npoints, dwell_s, averages)
            data = fid.reshape(1, 1, 1, npoints)
        else:
            _, fid = self._clean_fid(truth, npoints, dwell_s)
            channels = (coils.sensitivities(self.position_mm, self.size_mm, self.turn)[:, None]
                        * fid[None, :])
            if sigma > 0.0:
                channels = channels + receiver_noise(rng, sigma, npoints, self.scan.receiver, coils)
            data = np.transpose(channels.astype(np.complex64)).reshape(1, 1, 1, npoints, len(coils))
        hdr = self._header(seed, step, averages, truth)
        if coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
        return nifti_mrs(data, dwell_s, hdr)

    def acquire_components(self, seed=None, npoints=2048, dwell_s=1.0 / 4000.0, averages=1,
                           step=0):
        """Acquire the voxel component by component and return NIfTI-MRS on DIM_USER_0:
        the same subject, noise, header, and provenance as `acquire` with the same
        arguments, the data split into the signal components named in the header's
        SimulationComponents (metabolites, water, lipids, mm, downfield, spurious as
        present, and noise last), so the components sum to the acquisition exactly.
        With a CoilArray on the scan every component is received per coil (DIM_COIL
        first, the components on the dimension after it) and the noise component is
        the coils' correlated noise. The second file of a training pair: one
        acquisition, one decomposition, never squeezed into the acquisition's own
        dimensions."""
        seed, truth, rng = draw(self.scan.params, seed, step)
        coils = self.scan.coils
        sigma = self.noise_sigma(truth, dwell_s, averages)
        fids = self._component_fids(truth, npoints, dwell_s)
        names = list(fids)
        stack = [fids[name] for name in names]
        if coils is None:
            stack.append(receiver_noise(rng, sigma, npoints, self.scan.receiver) if sigma > 0.0
                         else np.zeros(npoints, dtype=np.complex64))
            data = np.stack(stack, axis=-1).reshape(1, 1, 1, npoints, len(stack))
        else:
            sensitivities = coils.sensitivities(self.position_mm, self.size_mm, self.turn)
            channels = [sensitivities[:, None] * fid[None, :] for fid in stack]
            channels.append(receiver_noise(rng, sigma, npoints, self.scan.receiver, coils)
                            if sigma > 0.0
                            else np.zeros((len(coils), npoints), dtype=np.complex64))
            data = np.stack([np.transpose(c.astype(np.complex64)) for c in channels], axis=-1)
            data = data.reshape(1, 1, 1, npoints, len(coils), len(channels))
        names.append("noise")
        hdr = self._header(seed, step, averages, truth)
        if coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_dim_info(0 if coils is None else 1, "DIM_USER_0",
                         info="MRSandbox: the signal components named in SimulationComponents.")
        hdr.set_user_def("SimulationComponents", {"names": names},
                         "MRSandbox: the components on DIM_USER_0, in order; they sum to "
                         "the acquisition of the same seed and step.")
        return nifti_mrs(data, dwell_s, hdr)

    def _header(self, seed, step, averages, truth):
        """The header extension every acquisition of this voxel carries."""
        from nifti_mrs.hdr_ext import Hdr_Ext

        hdr = Hdr_Ext(self._f0_mhz(), "1H")
        timings = {k: float(v) for k, v in vars(self.scan.sequence).items()
                   if k.endswith("_s") and isinstance(v, (int, float))}
        timings["tr_s"] = (None if self.scan.sequence.tr_s is None
                           else float(self.scan.sequence.tr_s))
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "step": int(step), "averages": int(averages), "params": truth.to_dict(),
             "tissue_fractions": dict(self.fractions._asdict()), "local": self.local(truth)},
            "MRSandbox: the realized parameter tree of this acquisition, with every "
            "map-valued leaf resolved at the voxel under local.",
        )
        hdr.set_user_def(
            "SimulationSequence",
            {"name": type(self.scan.sequence).__name__,
             "b_ms_um2": float(self.scan.b_value_ms_um2), **timings},
            "MRSandbox: the sequence and its timings in seconds.",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.scan.brain.name, "position_mm": list(self.position_mm),
             "size_mm": list(self.size_mm) if isinstance(self.size_mm, tuple) else self.size_mm,
             "transmit": None if self.scan.transmit is None else self.scan.transmit.name,
             "receiver": None if self.scan.receiver is None
             else dataclasses.asdict(self.scan.receiver),
             "noise_place": float(self.noise_place()),
             "calibrations": {leaf: c.record(self.scan)
                              for leaf, c in calibrations(
                                  self.scan.scanner, self.scan.records,
                                  subject=self.scan.subject).items()}},
            "MRSandbox: the virtual brain, the transmit map, the receiver, and the voxel "
            "placement.",
        )
        return hdr

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def _flat_lines(parts):
        """Parts of `components` as two pruned numpy line lists with per-line widths:
        the Lorentzian lines (metabolites, water, lipids) as (freqs, amps, lw_hz), and
        the Voigt lines carrying a Gaussian width (macromolecules, downfield) as
        (freqs, amps, lw_hz, gauss_hz); either is None when empty. The split lets the
        engine render the many Lorentzian lines through its factorized graph and the
        few Voigt ones densely (`_render` pads each to a stable bucket, so every draw
        renders at the same shapes and the jitted graphs never retrace).

        Most of the d*d transition elements are silent (the coherence filters zero
        them), so dropping them before rendering cuts the render cost by orders of
        magnitude without touching the result.
        """
        if not parts:
            return None, None
        freqs = np.concatenate([np.asarray(ops.convert_to_numpy(f)) for f, _, _, _ in parts])
        amps = np.concatenate([np.asarray(ops.convert_to_numpy(a)) for _, a, _, _ in parts])
        lw = np.concatenate([np.full(f.shape[0], lw, dtype=np.float32) for f, _, lw, _ in parts])
        gauss = np.concatenate([
            np.asarray(ops.convert_to_numpy(g), dtype=np.float32) * np.ones(f.shape[0], np.float32)
            for f, _, _, g in parts
        ])
        if not amps.size:                      # the parts hold no line (metabolites absent)
            return None, None
        keep = np.abs(amps) > 1e-7 * np.abs(amps).max()
        voigt = keep & (gauss > 0.0)
        lorentz = keep & ~voigt
        lorentzian = (freqs[lorentz], amps[lorentz], lw[lorentz]) if lorentz.any() else None
        if not voigt.any():
            return lorentzian, None
        return lorentzian, (freqs[voigt], amps[voigt], lw[voigt], gauss[voigt])

    def _metabolite_lines(self, truth):
        """Unweighted numpy lines of every metabolite for one truth and this voxel's
        resolved frame, as {name: (freqs, amps)}.

        The lines depend on the draw (relaxation, b1, offset) but not on the voxel
        position beyond the resolved values, so they are cached on the scan keyed by
        those values (`_share`): one truth serving many voxels (MRSI, multiplex), or
        many draws at one frame (a closed loop over concentrations and noise, which
        never enter the lines), simulate the metabolites once, and every draw applies
        only its own weights. Compiled, all systems replay in one jitted graph
        (Engine.compiled_batch), so a draw costs one dispatch and one transfer, or,
        where shaped pulses play on an ensemble, in one graph per operator algebra
        (Engine.algebra); plain, each metabolite runs through Engine.lines.
        """
        frame = self.frame(truth)
        names, t1, t2, _, offset_hz = frame
        b1_scale = self.b1_input(truth)
        # the draw's tissue temperature: the metabolites' shifts and couplings follow it
        temperature_c = float(self.resolve(truth.water.temperature_c))
        key = (self.compiled, temperature_c) + frame + self._design_key(truth)
        if key in self.scan._frame_lines:
            return self.scan._frame_lines[key]
        if not names:                          # no metabolite here: nothing to simulate
            self._share(key, {})
            return {}
        groups = tuple(self.scan.systems(name, temperature_c) for name in names)
        sequence = self.design(truth)
        # a design played as a whole (a tilted slab) runs plain, its ensemble in chunks: a
        # graph would hold every chunk at once
        if self.compiled and not sequence.unfolded:
            # one graph for all systems, or, where shaped pulses play on an isochromat
            # ensemble, one per operator algebra (a graph holds its intermediates, and
            # the step propagators of an ensemble of every system at once outgrow
            # memory); each graph plays its systems as one batch
            engine = self.scan.engine
            systems = [(i, system) for i, group in enumerate(groups) for system in group]
            heavy = engine.positions(sequence) and any(
                isinstance(event, ShapedPulse) for event in sequence.events)
            batches = {}
            for position, (_, system) in enumerate(systems):
                batches.setdefault(engine.algebra(system) if heavy else None, []).append(position)
            # the biggest systems first: the longest graph sets the pool's wall
            batches = sorted(batches.values(), key=lambda b: -max(systems[p][1].dim for p in b))

            def played(batch):
                fn, kept = engine.compiled_batch(sequence, tuple((systems[p][1],) for p in batch),
                                                 shift_mm=self.shift_mm, pruned=True)
                f, a = fn(ops.convert_to_tensor(np.asarray([t1[systems[p][0]] for p in batch],
                                                           dtype=np.float32)),
                          ops.convert_to_tensor(np.asarray([t2[systems[p][0]] for p in batch],
                                                           dtype=np.float32)),
                          ops.convert_to_tensor(np.asarray(b1_scale, dtype=np.float32)),
                          ops.convert_to_tensor(np.float32(offset_hz)))
                return (np.asarray(ops.convert_to_numpy(f)),
                        np.asarray(ops.convert_to_numpy(a)), kept)

            # the graphs side by side: a compilation and a replay run without holding python
            with ThreadPoolExecutor(min(len(batches), self.scan.threads or os.cpu_count() or 1)) as pool:
                done = list(pool.map(played, batches))
            # each system's kept elements (the graph pruned to what carries signal), by name
            parts = {}
            for batch, (f, a, kept) in zip(batches, done):
                for k, position in enumerate(batch):
                    parts[position] = (f[kept[k]:kept[k + 1]], a[kept[k]:kept[k + 1]])
            bounds = np.cumsum([0] + [len(group) for group in groups])
            lines = {name: tuple(np.concatenate([parts[p][j]
                                                 for p in range(bounds[i], bounds[i + 1])])
                                 for j in (0, 1))
                     for i, name in enumerate(names)}
        else:
            lines = {}
            for i, (name, group) in enumerate(zip(names, groups)):
                f, a = self.scan.engine.lines(
                    sequence, *(s.with_relaxation(t1_s=t1[i], t2_s=t2[i]) for s in group),
                    b1_scale=b1_scale, offset_hz=offset_hz, shift_mm=self.shift_mm)
                lines[name] = (np.asarray(ops.convert_to_numpy(f)),
                               np.asarray(ops.convert_to_numpy(a)))
        self._share(key, lines)
        return lines

    def _share(self, key, lines):
        """Metabolite lines into the scan's lines by frame, {key of `_metabolite_lines`:
        {name: (freqs, amps)}}; the scan keeps the last 256 frames."""
        shared = self.scan._frame_lines
        if len(shared) >= 256:
            shared.pop(next(iter(shared)))
        shared[key] = lines

    def _lines_cache(self, truth):
        """This truth's rendered FIDs on the scan (`prepare`), {key: fid}; the scan keeps
        the last 256 truths (a generator's batch and its neighbours)."""
        held = self.scan._lines
        entry = held.get(id(truth))
        if entry is None or entry[0] is not truth:
            if len(held) >= 256:
                held.pop(next(iter(held)))
            entry = held[id(truth)] = (truth, {})
        return entry[1]

    #*************#
    #   prepare   #
    #*************#
    def _framed_lines(self, design, names, temperature_c, fresh):
        """The metabolite lines of the frames `fresh` ({key of `_metabolite_lines`: a
        truth at it}) under a played design on an ensemble, into the scan's lines by
        frame: one graph per operator algebra split at the transmit frame
        (Engine.framed, its systems as one batch), the pulses of each transmit frame
        once, held on the scan (the latest transmit frame alone, a closed loop's), the
        frames' relaxation mapped over them; the graphs side by side in a thread pool
        as `_metabolite_lines` runs them."""
        engine = self.scan.engine
        groups = tuple(self.scan.systems(name, temperature_c) for name in names)
        systems = [(i, system) for i, group in enumerate(groups) for system in group]
        batches = {}
        for position, (_, system) in enumerate(systems):
            batches.setdefault(engine.algebra(system), []).append(position)
        batches = sorted(batches.values(), key=lambda b: -max(systems[p][1].dim for p in b))
        by_transmit = {}
        for key, truth in fresh.items():
            _, _, _, t1, t2, transmit, offset_hz, *tail = key
            by_transmit.setdefault((transmit, offset_hz, tuple(tail)),
                                   (self.b1_input(truth), []))[1].append((key, t1, t2))
        held = self.scan._propagators
        for (transmit, offset_hz, tail), (b1, frames) in by_transmit.items():
            at = (temperature_c, id(design), transmit, offset_hz) + tail
            if at not in held:
                held.clear()
                held[at] = [None] * len(batches)
            b1, offset = (ops.convert_to_tensor(np.asarray(b1, dtype=np.float32)),
                          ops.convert_to_tensor(np.float32(offset_hz)))
            t1 = np.asarray([t1 for _, t1, _ in frames], dtype=np.float32)
            t2 = np.asarray([t2 for _, _, t2 in frames], dtype=np.float32)

            def played(b):
                batch = batches[b]
                frame, _, draws, kept = engine.framed(
                    design, tuple((systems[p][1],) for p in batch), shift_mm=self.shift_mm)
                if held[at][b] is None:
                    held[at][b] = frame(b1, offset)
                columns = [systems[p][0] for p in batch]
                # eight draws at a time: a mapped pass holds every draw's state over the
                # ensemble, and more gain nothing
                rows = [draws(ops.convert_to_tensor(t1[first:first + 8, columns]),
                              ops.convert_to_tensor(t2[first:first + 8, columns]), b1, offset,
                              held[at][b]) for first in range(0, len(frames), 8)]
                return (np.concatenate([np.asarray(ops.convert_to_numpy(f)) for f, _ in rows]),
                        np.concatenate([np.asarray(ops.convert_to_numpy(a)) for _, a in rows]),
                        kept)

            with ThreadPoolExecutor(min(len(batches), self.scan.threads or os.cpu_count() or 1)) as pool:
                done = list(pool.map(played, range(len(batches))))
            parts = {}
            for batch, (f, a, kept) in zip(batches, done):
                for k, position in enumerate(batch):
                    parts[position] = (f[:, kept[k]:kept[k + 1]], a[:, kept[k]:kept[k + 1]])
            bounds = np.cumsum([0] + [len(group) for group in groups])
            for row, (key, _, _) in enumerate(frames):
                self._share(key, {name: tuple(np.concatenate([parts[p][j][row] for p in
                                                              range(bounds[i], bounds[i + 1])])
                                              for j in (0, 1))
                                  for i, name in enumerate(names)})

    def prepare(self, truths, npoints=None, dwell_s=None):
        """The metabolite lines of many draws ahead of their renders, into the scan's
        lines by frame (`_metabolite_lines`): what a generator does with its batch.
        Draws sharing their systems (the temperature step) and their design go
        together and every distinct frame among them (the relaxation, the transmit
        scale, the frequency offset, the temperature, and the gradient response the
        design is induced with: the drawn gradients leaves, or the scan's GIRF) is made
        once: in ONE replay of the design's graph mapped over the frames
        (Engine.replay_draws), or, under a
        played design on an ensemble (a graph per operator algebra), through the graphs
        split at the transmit frame (`_framed_lines`). A draw with a transmit profile
        on an ideal design renders on its own, as does a plain voxel. Given the grid,
        the draws' metabolite FIDs are rendered in one call too (Engine.render_fids, a
        width per line and row) and left for `_rendered_fids`, where the field plays
        as one group."""
        if not self.compiled:
            return
        engine, shared = self.scan.engine, self.scan._frame_lines
        batches = {}
        for truth in truths:
            design = self.design(truth)
            temperature_c = float(self.resolve(truth.water.temperature_c))
            key = (self.compiled, temperature_c) + self.frame(truth) + self._design_key(truth)
            batches.setdefault((round(4.0 * temperature_c) / 4.0, id(design), key[2]),
                               []).append((truth, key, design))
        for (temperature_c, _, names), members in batches.items():
            design = members[0][2]
            fresh = {}
            for truth, key, _ in members:
                if key not in shared:
                    fresh.setdefault(key, truth)
            if not names:                      # no metabolite here: nothing to simulate
                for key in fresh:
                    self._share(key, {})
            elif engine.positions(design) and any(
                    isinstance(event, ShapedPulse) for event in design.events):
                self._framed_lines(design, names, temperature_c, fresh)
            else:
                plain = {key: truth for key, truth in fresh.items()
                         if self.b1_profile(truth) is None}
                if plain:
                    replay, bounds = engine.replay_draws(
                        design, tuple(self.scan.systems(name, temperature_c) for name in names),
                        shift_mm=self.shift_mm)
                    t1, t2, b1, offset = (
                        ops.convert_to_tensor(np.asarray(values, dtype=np.float32))
                        for values in (
                            [[self.resolve(truth.t1[name]) for name in names]
                             for truth in plain.values()],
                            [[self.resolve(truth.t2[name]) for name in names]
                             for truth in plain.values()],
                            [self.b1_scale(truth) for truth in plain.values()],
                            [self.offset_hz(truth) for truth in plain.values()]))
                    freqs, amps = (np.asarray(ops.convert_to_numpy(value))
                                   for value in replay(t1, t2, b1, offset))
                    for row, key in enumerate(plain):
                        self._share(key, {
                            name: (freqs[row, bounds[i]:bounds[i + 1]],
                                   amps[row, bounds[i]:bounds[i + 1]])
                            for i, name in enumerate(names)})
            self.scan.mm.prepare(engine, design, self, [truth for truth, _, _ in members],
                                 self._f0_mhz())
            self.scan.water.prepare(engine, design, self, [truth for truth, _, _ in members],
                                    self._f0_mhz())
            if npoints is None or self.scan.field_groups != 1:
                continue
            # the draws' metabolite FIDs as `_grouped` and `_render_lines` would make them one
            # by one: each line's weight, the downfield cutoff, the draw's widths and receiver
            # delay, the Gaussian envelope after, a row per tissue where the tissues' widths
            # part; the draws whose lines were made (a profile draw on an ideal design renders
            # alone)
            members = [member for member in members if member[1] in shared]
            if not members or not names:
                continue
            n, dwell = ((npoints, dwell_s) if self.scan.receiver is None
                        else self.scan.receiver.grid(npoints, dwell_s))
            lines = shared[members[0][1]]
            f = np.concatenate([lines[name][0] for name in names])
            bounds = np.cumsum([0] + [lines[name][0].size for name in names])
            cutoff = self.scan.downfield.replaces_above_ppm
            # the lines above the cutoff carry nothing, so they leave the render
            keep = (np.flatnonzero(f <= cutoff * self._f0_mhz()) if cutoff is not None
                    else np.arange(f.size))
            pad = -(-keep.size // 512) * 512 - keep.size
            grid = np.arange(n) * dwell
            jobs = [(truth, key, tissue, width) for truth, key, _ in members
                    for tissue, width in self._metabolite_widths(truth).items()]
            for truth, _, _ in members:
                self._lines_cache(truth).pop(("metabolite fid", n, float(dwell),
                                              float(truth.acquisition.delay_s)), None)
            for first in range(0, len(jobs), 32):
                chunk = jobs[first:first + 32]
                rows_amps = np.zeros((len(chunk), f.size), np.complex64)
                rows_lw = np.ones((len(chunk), f.size), np.float32)
                for row, (truth, key, tissue, _) in enumerate(chunk):
                    weights = dict(zip(truth.conc, self.metabolite_weights(truth, tissue)))
                    start_s, lines = float(truth.acquisition.delay_s), shared[key]
                    for i, name in enumerate(names):
                        part, lw = (slice(bounds[i], bounds[i + 1]),
                                    1.0 / (np.pi * self.resolve(truth.t2[name])))
                        rows_lw[row, part] = lw
                        rows_amps[row, part] = (lines[name][1] * weights[name]
                                                * self._advance(f[part], lw, start_s))
                _, fids = self.scan.engine.render_fids(
                    ops.convert_to_tensor(np.pad(f[keep], (0, pad))),
                    ops.convert_to_tensor(np.pad(rows_amps[:, keep], ((0, 0), (0, pad)))),
                    npoints=n, dwell_s=dwell,
                    lw_hz=np.pad(rows_lw[:, keep], ((0, 0), (0, pad)), constant_values=1.0))
                fids = np.asarray(ops.convert_to_numpy(fids), dtype=np.complex64)
                for row, (truth, _, _, width) in enumerate(chunk):
                    start_s = float(truth.acquisition.delay_s)
                    held, at = self._lines_cache(truth), ("metabolite fid", n, float(dwell), start_s)
                    held[at] = held.get(at, 0.0) + fids[row] * self._envelope(width, start_s + grid)
            # the other sources the same way: their parts per draw (the engine parts cached
            # above), the draws with one line list rendered as rows, Lorentzian lines through
            # the factorized graph, Voigt ones densely; a draw's `_rendered_fids` then takes
            # every source from the cache and skips its own grouping and renders
            alike, made = {}, {}
            for truth, _, _ in members:
                start_s = float(truth.acquisition.delay_s)
                for (name, width), parts in self._grouped(truth, metabolites=False).items():
                    freqs = np.concatenate([np.asarray(ops.convert_to_numpy(f))
                                            for f, _, _, _ in parts])
                    amps = np.concatenate([np.asarray(ops.convert_to_numpy(a))
                                           for _, a, _, _ in parts])
                    lw = np.concatenate([np.full(f.shape[0], w, np.float32)
                                         for f, _, w, _ in parts])
                    gauss = np.concatenate([np.asarray(ops.convert_to_numpy(g), np.float32)
                                            * np.ones(f.shape[0], np.float32)
                                            for f, _, _, g in parts])
                    alike.setdefault((name, freqs.tobytes(), bool(gauss.any())),
                                     (freqs, []))[1].append((truth, amps, lw, gauss, start_s,
                                                             width))
            for (name, _, voigt), (f, draws) in alike.items():
                for first in range(0, len(draws), 32):
                    chunk = draws[first:first + 32]
                    if f.size * len(chunk) <= 64:
                        # too few lines to be worth a dispatch: on the host, as a lone draw
                        # renders them (`_render_lines`)
                        fids = np.stack([self._render_lines(
                            (f, amps, lw, gauss if voigt else None), n, dwell, start_s)
                            for _, amps, lw, gauss, start_s, _ in chunk])
                    else:
                        pad = -(-f.size // (64 if voigt else 512)) * (64 if voigt else 512) - f.size
                        rows_amps, rows_lw = (np.zeros((len(chunk), f.size + pad), np.complex64),
                                              np.ones((len(chunk), f.size + pad), np.float32))
                        rows_gauss = np.zeros((len(chunk), f.size + pad), np.float32)
                        for row, (truth, amps, lw, gauss, start_s, _) in enumerate(chunk):
                            amps = amps * self._advance(f, lw, start_s)
                            if voigt:
                                # a Gaussian line's cross term over the delay is a Lorentzian
                                # width (as `_render_lines`)
                                g = (np.pi * gauss.astype(np.float64)) ** 2 / (4.0 * np.log(2.0))
                                amps, lw = ((amps * np.exp(-g * start_s ** 2)).astype(np.complex64),
                                            (lw + 2.0 * g * start_s / np.pi).astype(np.float32))
                            (rows_amps[row, :f.size], rows_lw[row, :f.size],
                             rows_gauss[row, :f.size]) = amps, lw, gauss
                        padded = ops.convert_to_tensor(np.pad(f, (0, pad)))
                        _, fids = self.scan.engine.render_fids(
                            padded, ops.convert_to_tensor(rows_amps), npoints=n, dwell_s=dwell,
                            lw_hz=rows_lw, gauss_hz=rows_gauss if voigt else None)
                        fids = np.asarray(ops.convert_to_numpy(fids), dtype=np.complex64)
                    for row, (truth, _, _, _, start_s, width) in enumerate(chunk):
                        fid = fids[row] if voigt else fids[row] * self._envelope(width,
                                                                                 start_s + grid)
                        at = (id(truth), ("fid", name, n, float(dwell), start_s))
                        made[at] = (truth, made[at][1] + fid if at in made else fid)
            for truth, key in made.values():
                pass
            for (_, key), (truth, fid) in made.items():
                self._lines_cache(truth)[key] = fid
            for truth, _, _ in members:
                self._lines_cache(truth)[("every source", n, float(dwell),
                                          float(truth.acquisition.delay_s))] = True

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
