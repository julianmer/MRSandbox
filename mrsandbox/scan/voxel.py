####################################################################################################
#                                            voxel.py                                              #
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


#**********************#
#   subject and noise  #
#**********************#
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
    turn = np.exp(-2j * np.pi * 4.65 * float(hdr.to_dict()["SpectrometerFrequency"][0]) * np.arange(data.shape[3]) * dwell_s)
    return gen_nifti_mrs_hdr_ext((data * turn.reshape((1, 1, 1, -1) + (1,) * (data.ndim - 4))).astype(np.complex64), dwell_s, hdr, **kwargs)


#**********************#
#   noise scale        #
#**********************#
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


#**********************#
#   receiver noise     #
#**********************#
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
#                                           Class Voxel                                            #
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
                 box_axes="xyz", shim_at_mm=None):
        self.scan = scan
        # a cell of a lattice inside the box a localized design selects: the design is
        # played confined to the cell, its ensemble shifted from the box's centre
        # (along `box_axes`; the design keeps its own extent along the others, a slab
        # the lattice does not resolve)
        self.box_center_mm = None if box_center_mm is None else tuple(float(x) for x in box_center_mm)
        self.box_axes = box_axes
        # a head that moved after the shim: the shim of the planned place stays
        self.shim_at_mm = None if shim_at_mm is None else tuple(float(x) for x in shim_at_mm)
        self.position_mm = tuple(float(x) for x in position_mm)
        self.size_mm = (tuple(float(x) for x in size_mm) if hasattr(size_mm, "__len__")
                        else float(size_mm))
        self.fractions = scan.brain.fractions(position_mm, self.size_mm)
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
                mean = value.mean(self.position_mm, self.size_mm)
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

    #*******************#
    #   transmit frame  #
    #*******************#
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
        local = [self.scan.transmit.nearest(np.asarray(self.position_mm) + offset) for offset in offsets]
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

    def frame(self, truth):
        """The resolved frame the metabolite lines and basis are rendered in at this
        voxel: (T1 per metabolite, T2 per metabolite, transmit, offset_hz) in
        truth.conc order, every value resolved here, the transmit being `b1_scale`
        or, with a per-isochromat profile, (b1_scale, the profile's bytes). Together
        with the scan's `sequence.structure()` and timings (and, under a gradient
        response, this voxel's position, `_design_key`) it is the complete key of a
        rendered basis, so a caller caching bases across voxels or draws keys them
        by this tuple."""
        names = tuple(truth.conc)
        profile = self.b1_profile(truth)
        transmit = self.b1_scale(truth) if profile is None else (self.b1_scale(truth), profile.tobytes())
        return (tuple(float(self.resolve(truth.t1[name])) for name in names),
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
                    factor = noise.noise_gain(self.position_mm, self.size_mm)
                else:
                    here = noise.mean(self.position_mm, self.size_mm)
                    there = noise.mean(self.scan.noise_reference_mm, self.size_mm)
                    if here is not None and there is not None and min(here, there) > 0.0:
                        factor = float(here) / float(there)
            self._kernels["noise_place"] = factor
        return self._kernels["noise_place"]

    #*****************#
    #   realization   #
    #*****************#
    def sample(self, seed=None):
        """Realize the scan's Params tree for this draw; the result is the ground truth."""
        return self.scan.params.sample(seed)

    #******************#
    #   components     #
    #******************#
    def components(self, truth, metabolites=True):
        """Every signal component as (freqs, amps, lw_hz, gauss_hz): metabolites, water,
        lipids, and the macromolecule background.

        Every metabolite is its own component with its own pooled T1 and T2 (linewidth
        and echo attenuation stay consistent), and water and lipids carry theirs;
        renderings flatten all components into one pruned line list and render once.
        With `metabolites` False the metabolite parts are left out, for a caller that
        forms them from a per-truth basis (`metabolite_basis`).
        """
        return [part for parts in self._grouped(truth, metabolites).values() for part in parts]

    def _water_parts(self, truth, engine, sequence, f0_mhz, faded):
        """The water's line parts at the offset the voxel is being played at, or None."""
        water = self.scan.water.lines(engine, sequence, self, truth, f0_mhz)
        if water is None:
            return None
        w_freqs, w_amps = water
        # the water line moves with the temperature, fourteen times more than the metabolites
        # (which take their systems at the same temperature, `_metabolite_lines`)
        warm = self.resolve(truth.water.temperature_c) - 37.0
        w_freqs = w_freqs - np.float32(warm / CADY_1995["c_per_ppm"] * f0_mhz)
        return [(w_freqs, w_amps * np.complex64(faded(truth.diffusion.water)),
                 1.0 / (np.pi * self.resolve(truth.water.t2_s)), 0.0)]

    def _grouped(self, truth, metabolites=True):
        """The parts of `components` grouped by source, {name: [parts]} in the order
        metabolites, water, lipids, mm, downfield; silent sources are left out."""
        engine, sequence = self.scan.engine, self.design(truth)
        f0_mhz = self._f0_mhz()
        faded = self._diffusion_factor(truth)
        # an empirical downfield source carries everything above its cutoff, so the
        # spin systems' exchangeable NH lines up there leave the render (see
        # phantom.downfield.DownfieldSource)
        cutoff = self.scan.downfield.replaces_above_ppm
        groups = {}
        if metabolites:
            lines = self._metabolite_lines(truth)
            weights = self.metabolite_weights(truth)
            parts = []
            for name, weight in zip(truth.conc, weights):
                freqs, amps = lines[name]
                if cutoff is not None:
                    amps = np.where(freqs > cutoff * f0_mhz, np.complex64(0.0), amps)
                parts.append((freqs, amps * weight, 1.0 / (np.pi * self.resolve(truth.t2[name])),
                              0.0))
            groups["metabolites"] = parts
        water = self._water_parts(truth, engine, sequence, f0_mhz, faded)
        if water is not None:
            groups["water"] = water
        lipid = self.scan.lipids.lines(engine, sequence, self, truth, f0_mhz)
        if lipid is not None:
            l_freqs, l_amps = lipid
            groups["lipids"] = [(l_freqs, l_amps * np.complex64(faded(truth.diffusion.lipids)),
                                 1.0 / (np.pi * truth.lipids.t2_s), 0.0)]
        mm = self.scan.mm.lines(engine, sequence, self, truth, f0_mhz)
        if mm is not None:
            # real positive amplitudes render absorptive as they are; the engine's
            # receiver factor compensates density matrix phases the MM lines never had
            freqs, amps, gauss = mm
            groups["mm"] = [(ops.convert_to_tensor(freqs),
                             ops.convert_to_tensor(amps * faded(truth.diffusion.mm)),
                             0.0, ops.convert_to_tensor(gauss))]
        df = self.scan.downfield.lines(engine, sequence, self, truth, f0_mhz)
        if df is not None:
            freqs, amps, gauss = df
            groups["downfield"] = [(ops.convert_to_tensor(freqs), ops.convert_to_tensor(amps),
                                    0.0, ops.convert_to_tensor(gauss))]
        return groups

    #*************#
    #   weights   #
    #*************#
    def metabolite_weights(self, truth):
        """The complex weight of every metabolite's unit lines in this voxel, in
        truth.conc order: the resolved concentration split per tissue by the drawn GM
        to WM contrast r as gm 2r/(1+r) + wm 2/(1+r) (an equal-mix voxel carries exactly
        the drawn value, pure tissues the published contrast), times the diffusion
        attenuation of the local ADC under the drawn diffusion model."""
        faded = self._diffusion_factor(truth)
        sticks = truth.diffusion.model == "sticks"
        weights = []
        for name, concentration in truth.conc.items():
            r = truth.contrast[name]
            tissue = (2.0 * r * self.fractions.gm + 2.0 * self.fractions.wm) / (1.0 + r)
            weights.append(self.resolve(concentration) * tissue
                           * faded(truth.diffusion.adc[name], sticks))
        return np.asarray(weights, dtype=np.complex64)

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
        freqs, amps, lw, rows = [], [], [], []
        for k, name in enumerate(names):
            f, a = lines[name]
            if cutoff is not None:
                a = np.where(f > cutoff * f0_mhz, np.complex64(0.0), a)
            keep = np.abs(a) > 1e-7 * np.abs(a).max() if a.size else np.zeros(0, bool)
            freqs.append(f[keep])
            amps.append(a[keep])
            lw.append(np.full(int(keep.sum()), frame[1][k], dtype=np.float32))
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

    #**************#
    #   fading     #
    #**************#
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
        one alike, and the spurious echo (drawn relative to the maximum of the summed FID without the water)
        as its own entry, so the components sum to the clean FID exactly.

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
        its phasor depends on the truth alone and is cached per truth.
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
        # components (the measured MM and downfield widths already carry theirs)
        gauss_hz = self.resolve(truth.lineshape.gauss_hz)
        envelope = np.exp(-(np.pi * gauss_hz) ** 2 / (4.0 * np.log(2.0)) * grid * grid).astype(np.float32)
        def sources(basis):
            fids = {}
            if basis is not None:
                fids["metabolites"] = (self.metabolite_weights(truth) @ basis).astype(np.complex64) * envelope
            ready = None if self._field_offset_hz else self._lines_cache(truth).get(("metabolite fid", npoints, float(dwell_s), start_s))
            if ready is not None:               # rendered with the batch (`prepare`)
                fids["metabolites"] = ready
            for name, parts in self._grouped(truth, metabolites=basis is None and ready is None).items():
                lorentzian, voigt = self._flat_lines(parts)
                fid = np.zeros(npoints, dtype=np.complex64)
                if lorentzian is not None:
                    fid = fid + self._render_lines(lorentzian, npoints, dwell_s, start_s) * envelope
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
            fids = {name: fid * kernel for name, fid in sources(basis).items()}
            if "water" in fids and self.scan.water_groups > 1:
                # the water alone at every field group's own offset under its own dephasing
                # (VirtualScan water_groups): a suppression pulse misses the water where the
                # field is off its band, which the voxel's mean cannot show
                engine, sequence, faded = self.scan.engine, self.design(truth), self._diffusion_factor(truth)
                water = np.zeros(npoints, dtype=np.complex64)
                for mean_hz, weight, part_kernel in self.scan.field.groups(
                        self.position_mm, self.size_mm, grid, self.scan.water_groups, shim_at_mm=self.shim_at_mm):
                    self._field_offset_hz = mean_hz
                    try:
                        lorentzian, _ = self._flat_lines(self._water_parts(truth, engine, sequence, self._f0_mhz(), faded))
                    finally:
                        self._field_offset_hz = 0.0
                    water = water + np.float32(weight) * self._render_lines(lorentzian, npoints, dwell_s, start_s) * envelope * part_kernel
                fids["water"] = water
        else:
            # the field inside the pulses: every group of the voxel's field distribution
            # plays the sequence at its own mean offset and dephases by its own kernel
            fids = {}
            for mean_hz, weight, kernel in self.scan.field.groups(
                    self.position_mm, self.size_mm, grid, self.scan.field_groups,
                    shim_at_mm=self.shim_at_mm):
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
            # relative to the voxel's signal WITHOUT its water: signal from outside the voxel does not grow with the water
            # a suppression train leaves inside it (the source's own FIDs carry next to none)
            total = sum((fid for name, fid in fids.items() if name != "water"), np.zeros(npoints, dtype=np.complex64))
            fids["spurious"] = np.asarray(echo.process_tensor(total[None, :], sw_hz=sw_hz)[0][0]) - total
        modulation = np.exp(1j * (2.0 * np.pi * self.offset_hz(truth) * grid
                                  + np.radians(acquisition.phase_deg)))
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
            return (amps.astype(np.complex128)[:, None] * np.exp(phase)).sum(axis=0).astype(np.complex64)
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
            padded = ops.convert_to_tensor(np.pad(gauss, (0, -(-freqs.size // 512) * 512 - freqs.size)))
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
                                                        shim_at_mm=self.shim_at_mm)
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
        return (self.position_mm,) if kicked and self.design(truth) is not self.scan.sequence else ()

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
            self._kernels[key] = readout_kernel(self.scan.sequence, self._scanner(), self.position_mm,
                                                self.size_mm, npoints, dwell_s, response[1],
                                                start_s=start_s)
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
            channels = coils.sensitivities(self.position_mm, self.size_mm)[:, None] * fid[None, :]
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
            sensitivities = coils.sensitivities(self.position_mm, self.size_mm)
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
        timings["tr_s"] = None if self.scan.sequence.tr_s is None else float(self.scan.sequence.tr_s)
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
                              for leaf, c in calibrations(self.scan.scanner, self.scan.records).items()}},
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
        position beyond the resolved values, so they are cached per truth on the scan
        keyed by those values: one truth serving many voxels (MRSI, multiplex)
        simulates the metabolites once, and every voxel applies only its own weights.
        Compiled, all systems replay in one jitted graph (Engine.compiled_batch), so a
        draw costs one dispatch and one transfer; plain, each metabolite runs through
        Engine.lines.
        """
        names = tuple(truth.conc)
        frame = self.frame(truth)
        t1, t2, _, offset_hz = frame
        b1_scale = self.b1_input(truth)
        cached = self._lines_cache(truth)
        # the draw's tissue temperature: the metabolites' shifts and couplings follow it
        temperature_c = float(self.resolve(truth.water.temperature_c))
        key = (self.compiled, temperature_c) + frame + self._design_key(truth)
        if key in cached:
            return cached[key]
        groups = tuple(self.scan.systems(name, temperature_c) for name in names)
        sequence = self.design(truth)
        if self.compiled:
            # one graph for all metabolites, or, where shaped pulses play on an
            # isochromat ensemble, one per metabolite: a graph holds its intermediates,
            # and the step propagators of an ensemble of every system at once run to
            # tens of GB
            engine = self.scan.engine
            heavy = engine.positions(sequence) and any(
                isinstance(event, ShapedPulse) for event in sequence.events)
            batches = [[i] for i in range(len(groups))] if heavy else [range(len(groups))]
            freqs, amps, bounds = [], [], [0]
            for batch in batches:
                fn, kept = engine.compiled_batch(sequence, tuple(groups[i] for i in batch),
                                                 shift_mm=self.shift_mm, pruned=True)
                f, a = fn(ops.convert_to_tensor(np.asarray([t1[i] for i in batch], dtype=np.float32)),
                          ops.convert_to_tensor(np.asarray([t2[i] for i in batch], dtype=np.float32)),
                          ops.convert_to_tensor(np.asarray(b1_scale, dtype=np.float32)),
                          ops.convert_to_tensor(np.float32(offset_hz)))
                freqs.append(np.asarray(ops.convert_to_numpy(f)))
                amps.append(np.asarray(ops.convert_to_numpy(a)))
                bounds.extend(bounds[-1] + kept[1:])       # the kept elements per group, the graph pruned to what carries signal
            freqs, amps = np.concatenate(freqs), np.concatenate(amps)
        else:
            parts = [self.scan.engine.lines(
                sequence, *(s.with_relaxation(t1_s=t1[i], t2_s=t2[i]) for s in group),
                b1_scale=b1_scale, offset_hz=offset_hz, shift_mm=self.shift_mm)
                for i, group in enumerate(groups)]
            freqs = np.concatenate([np.asarray(ops.convert_to_numpy(f)) for f, _ in parts])
            amps = np.concatenate([np.asarray(ops.convert_to_numpy(a)) for _, a in parts])
            bounds = np.cumsum([0] + [sum(s.dim ** 2 for s in group) for group in groups])
        lines = {name: (freqs[bounds[i]:bounds[i + 1]], amps[bounds[i]:bounds[i + 1]])
                 for i, name in enumerate(names)}
        cached[key] = lines
        return lines

    def _lines_cache(self, truth):
        """This truth's metabolite lines on the scan, {frame key: lines}; the scan keeps
        the last 256 truths (a generator's batch and its neighbours, one truth serving
        many voxels)."""
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
    def prepare(self, truths, npoints=None, dwell_s=None):
        """The metabolite lines of many draws in ONE replay of the design's graph mapped
        over the draws (Engine.replay_draws), left in the lines cache for each draw's
        own render: what a generator does with its batch before rendering it. Draws
        sharing their systems (the temperature step) and their design replay together;
        a draw with a transmit profile, or under a played design on an ensemble (a graph
        per metabolite), renders on its own as before, as does a plain voxel. Given the
        grid, the draws' metabolite FIDs are rendered in one call too (Engine.render_fids,
        a width per line and row) and left for `_rendered_fids`, where the field plays
        as one group."""
        if not self.compiled:
            return
        engine = self.scan.engine
        batches = {}
        for truth in truths:
            design = self.design(truth)
            if self.b1_profile(truth) is not None or (engine.positions(design) and any(
                    isinstance(event, ShapedPulse) for event in design.events)):
                continue
            temperature_c = float(self.resolve(truth.water.temperature_c))
            key = (self.compiled, temperature_c) + self.frame(truth) + self._design_key(truth)
            if key not in self._lines_cache(truth):
                batches.setdefault((round(4.0 * temperature_c) / 4.0, id(design)), []).append((truth, key, design))
        for (temperature_c, _), members in batches.items():
            names, design = tuple(members[0][0].conc), members[0][2]
            replay, bounds = engine.replay_draws(design, tuple(self.scan.systems(name, temperature_c) for name in names),
                                                 shift_mm=self.shift_mm)
            t1, t2, b1, offset = (ops.convert_to_tensor(np.asarray(values, dtype=np.float32)) for values in (
                [[self.resolve(truth.t1[name]) for name in names] for truth, _, _ in members],
                [[self.resolve(truth.t2[name]) for name in names] for truth, _, _ in members],
                [self.b1_scale(truth) for truth, _, _ in members], [self.offset_hz(truth) for truth, _, _ in members]))
            freqs, amps = (np.asarray(ops.convert_to_numpy(value)) for value in replay(t1, t2, b1, offset))
            for row, (truth, key, _) in enumerate(members):
                self._lines_cache(truth)[key] = {name: (freqs[row, bounds[i]:bounds[i + 1]], amps[row, bounds[i]:bounds[i + 1]])
                                                 for i, name in enumerate(names)}
            self.scan.mm.prepare(engine, design, self, [truth for truth, _, _ in members], self._f0_mhz())
            self.scan.water.prepare(engine, design, self, [truth for truth, _, _ in members], self._f0_mhz())
            if npoints is None or self.scan.field_groups != 1:
                continue
            # the draws' metabolite FIDs as `_grouped` and `_render_lines` would make them one by one: each line's
            # weight, the downfield cutoff, the draw's widths and receiver delay, the Gaussian envelope after
            n, dwell = (npoints, dwell_s) if self.scan.receiver is None else self.scan.receiver.grid(npoints, dwell_s)
            f, cutoff = freqs[0], self.scan.downfield.replaces_above_ppm
            keep = f <= cutoff * self._f0_mhz() if cutoff is not None else np.ones(f.size, bool)
            pad = -(-f.size // 512) * 512 - f.size
            grid = np.arange(n) * dwell
            for first in range(0, len(members), 32):
                chunk = members[first:first + 32]
                rows_amps = np.zeros((len(chunk), f.size + pad), np.complex64)
                rows_lw = np.ones((len(chunk), f.size + pad), np.float32)
                for row, (truth, _, _) in enumerate(chunk):
                    weights, start_s = self.metabolite_weights(truth), float(truth.acquisition.delay_s)
                    for i, name in enumerate(names):
                        part, lw = slice(bounds[i], bounds[i + 1]), 1.0 / (np.pi * self.resolve(truth.t2[name]))
                        rows_lw[row, part] = lw
                        rows_amps[row, part] = amps[first + row, part] * weights[i] * self._advance(f[part], lw, start_s)
                    rows_amps[row, :f.size] *= keep
                _, fids = self.scan.engine.render_fids(ops.convert_to_tensor(np.pad(f, (0, pad))), ops.convert_to_tensor(rows_amps),
                                                       npoints=n, dwell_s=dwell, lw_hz=rows_lw)
                fids = np.asarray(ops.convert_to_numpy(fids), dtype=np.complex64)
                for row, (truth, _, _) in enumerate(chunk):
                    start_s, gauss = float(truth.acquisition.delay_s), self.resolve(truth.lineshape.gauss_hz)
                    envelope = np.exp(-(np.pi * gauss) ** 2 / (4.0 * np.log(2.0)) * (start_s + grid) ** 2).astype(np.float32)
                    self._lines_cache(truth)[("metabolite fid", n, float(dwell), start_s)] = fids[row] * envelope

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
