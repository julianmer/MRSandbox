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
import numpy as np
import scipy.special
from keras import ops

from mrsandbox.params import Params, Tissues
from mrsandbox.phantom.volume import Volume


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


#**********************#
#   noise scale        #
#**********************#
# the reference acquisition the drawn noise sigma belongs to: one transient of a 20 mm
# cube (8 mL) at 3 T; SNR scales with the voxel volume, with the field (body-noise
# dominated, stated), and with the square root of the averages, so the per-point
# sigma scales inversely
NOISE_REFERENCE = {"volume_ml": 8.0, "b0_t": 3.0}


def noise_scale(volume_ml, b0_t, averages=1):
    """The factor on the drawn sigma for an acquisition of `volume_ml` at `b0_t` tesla
    averaged `averages` times, one at the reference."""
    return (NOISE_REFERENCE["volume_ml"] / volume_ml) * (NOISE_REFERENCE["b0_t"] / b0_t) / np.sqrt(averages)


#**********************#
#   receiver noise     #
#**********************#
def receiver_noise(rng, sigma, npoints):
    """One draw of the receiver's complex Gaussian noise: standard deviation `sigma`
    per real and imaginary FID point, complex64."""
    return sigma * (rng.standard_normal(npoints)
                    + 1j * rng.standard_normal(npoints)).astype(np.complex64)


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
    def __init__(self, scan, position_mm, size_mm, compiled=False):
        self.scan = scan
        self.position_mm = tuple(float(x) for x in position_mm)
        self.size_mm = (tuple(float(x) for x in size_mm) if hasattr(size_mm, "__len__")
                        else float(size_mm))
        self.fractions = scan.brain.fractions(position_mm, self.size_mm)
        self.compiled = compiled
        self._kernels = {}
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
        one, 0.0 outside it)."""
        return self.resolve(truth.b1_scale) * self._transmit

    def offset_hz(self, truth):
        """The frequency offset the sequence runs at in this voxel: the drawn scanner
        frequency error, which the selective, shaped, and suppression pulses see; the
        readout shifts by the same value afterwards (the acquisition modulation)."""
        return self.resolve(truth.acquisition.shift_hz)

    #***********#
    #   noise   #
    #***********#
    @property
    def volume_ml(self):
        """The voxel volume in millilitres."""
        size = np.asarray(self.size_mm, dtype=np.float64) * np.ones(3)
        return float(np.prod(size)) / 1000.0

    def noise_sigma(self, truth, averages=1):
        """The per-point receiver noise of this voxel's acquisition: the drawn sigma
        (one transient of the reference 8 mL voxel at 3 T) scaled by the volume, the
        field, and the averages (noise_scale)."""
        return truth.noise.sigma * noise_scale(self.volume_ml, self.scan.engine.b0_t, averages)

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
        engine, sequence = self.scan.engine, self.scan.sequence
        f0_mhz = self._f0_mhz()
        faded = self._diffusion_factor(truth)
        # an empirical downfield source carries everything above its cutoff, so the
        # spin systems' exchangeable NH lines up there leave the render (see
        # phantom.downfield.DownfieldSource)
        cutoff = self.scan.downfield.replaces_above_ppm
        parts = []
        if metabolites:
            lines = self._metabolite_lines(truth)
            weights = self.metabolite_weights(truth)
            for name, weight in zip(truth.conc, weights):
                freqs, amps = lines[name]
                if cutoff is not None:
                    amps = np.where(freqs > cutoff * f0_mhz, np.complex64(0.0), amps)
                parts.append((freqs, amps * weight, 1.0 / (np.pi * self.resolve(truth.t2[name])),
                              0.0))
        water = self.scan.water.lines(engine, sequence, self, truth, f0_mhz)
        if water is not None:
            water_t2 = self.resolve(truth.water.t2_s)
            w_freqs, w_amps = water
            parts.append((w_freqs, w_amps * np.complex64(faded(truth.diffusion.water)),
                          1.0 / (np.pi * water_t2), 0.0))
        lipid = self.scan.lipids.lines(engine, sequence, self, truth, f0_mhz)
        if lipid is not None:
            l_freqs, l_amps = lipid
            parts.append((l_freqs, l_amps * np.complex64(faded(truth.diffusion.lipids)),
                          1.0 / (np.pi * truth.lipids.t2_s), 0.0))
        mm = self.scan.mm.lines(engine, sequence, self, truth, f0_mhz)
        if mm is not None:
            # real positive amplitudes render absorptive as they are; the engine's
            # receiver factor compensates density matrix phases the MM lines never had
            freqs, amps, gauss = mm
            parts.append((ops.convert_to_tensor(freqs),
                          ops.convert_to_tensor(amps * faded(truth.diffusion.mm)),
                          0.0, ops.convert_to_tensor(gauss)))
        df = self.scan.downfield.lines(engine, sequence, self, truth, f0_mhz)
        if df is not None:
            freqs, amps, gauss = df
            parts.append((ops.convert_to_tensor(freqs), ops.convert_to_tensor(amps),
                          0.0, ops.convert_to_tensor(gauss)))
        return parts

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
        `metabolite_weights` @ basis."""
        lines = self._metabolite_lines(truth)
        names = tuple(truth.conc)
        frame = (tuple(float(self.resolve(truth.t1[name])) for name in names),
                 tuple(float(self.resolve(truth.t2[name])) for name in names),
                 self.b1_scale(truth), self.offset_hz(truth), npoints, float(dwell_s))
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
        sigma = self.noise_sigma(truth, averages)
        if sigma > 0.0:
            fid = fid + receiver_noise(rng, sigma, npoints)
        return t, fid

    def _clean_fid(self, truth, npoints, dwell_s, basis=None):
        """The noise-free voxel FID: components summed, B0 dephasing, the drawn spurious
        echo and eddy-current phase applied through Augmentrum's modules, then the
        drawn frequency shift and zero-order phase.

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
        grid = np.arange(npoints) * dwell_s
        t = ops.convert_to_tensor(grid.astype(np.float32))
        fid = np.zeros(npoints, dtype=np.complex64)
        if basis is not None:
            fid = fid + (self.metabolite_weights(truth) @ basis).astype(np.complex64)
        lorentzian, voigt = self._flat_lines(truth, metabolites=basis is None)
        if lorentzian is not None:
            fid = fid + self._render_lines(lorentzian, npoints, dwell_s)
        # the microscopic field distribution: a Gaussian envelope on the spin-system
        # components (the measured MM and downfield widths already carry theirs)
        gauss_hz = self.resolve(truth.lineshape.gauss_hz)
        if gauss_hz > 0.0:
            damping = (np.pi * gauss_hz) ** 2 / (4.0 * np.log(2.0))
            fid = fid * np.exp(-damping * grid * grid).astype(np.float32)
        if voigt is not None:
            fid = fid + self._render_lines(voigt, npoints, dwell_s)
        mm_fid = self.scan.mm.fid(self, truth, self._f0_mhz(), grid,
                                  sequence=self.scan.sequence)
        if mm_fid is not None:
            fid = fid + mm_fid * np.float32(
                self._diffusion_factor(truth)(truth.diffusion.mm))
        if self.scan.field is not None:
            fid = fid * self._field_kernel(npoints, dwell_s)
        spur, acquisition = truth.spurious, truth.acquisition
        sw_hz = 1.0 / dwell_s
        if spur.amplitude > 0.0:
            # the out-of-voxel echo enters before the receiver-side modulation like
            # any received signal
            echo = SpuriousEchoes(mode="echo", echoes=[dict(
                alpha=spur.amplitude, t_echo=spur.tau_s,
                T2=1.0 / np.sqrt(2.0 * spur.width_s2), freq_hz=spur.ppm * self._f0_mhz(),
                phase_deg=spur.phase_deg, gaussian_env=True)])
            fid = np.asarray(echo.process_tensor(fid[None, :], sw_hz=sw_hz)[0][0])
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
            fid = fid * cached[1][key]
        modulation = np.exp(1j * (2.0 * np.pi * self.offset_hz(truth) * grid
                                  + np.radians(acquisition.phase_deg)))
        return t, (fid * modulation).astype(np.complex64)

    def _render_lines(self, bucket, npoints, dwell_s):
        """One line bucket, (freqs, amps, lw_hz[, gauss_hz]) as numpy, to a complex64
        FID: on the host in float64 when the bucket holds at most 64 lines (exact, no
        dispatch), otherwise through the engine's jitted graphs."""
        freqs, amps, lw = bucket[0], bucket[1], bucket[2]
        gauss = bucket[3] if len(bucket) > 3 else None
        if freqs.size <= 64:
            t = np.arange(npoints, dtype=np.float64) * dwell_s
            rate = 1j * 2.0 * np.pi * freqs.astype(np.float64) - np.pi * lw.astype(np.float64)
            phase = rate[:, None] * t[None, :]
            if gauss is not None:
                g = (np.pi * gauss.astype(np.float64)) ** 2 / (4.0 * np.log(2.0))
                phase = phase - g[:, None] * t[None, :] ** 2
            return (amps.astype(np.complex128)[:, None] * np.exp(phase)).sum(axis=0).astype(np.complex64)
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

    def _field_kernel(self, npoints, dwell_s):
        """The cached intra-voxel B0 dephasing kernel of this voxel on this time grid."""
        key = (npoints, dwell_s)
        if key not in self._kernels:
            t = np.arange(npoints) * dwell_s
            self._kernels[key] = self.scan.field.kernel(self.position_mm, self.size_mm, t)
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
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        seed, truth, rng = draw(self.scan.params, seed, step)
        coils = self.scan.coils
        sigma = self.noise_sigma(truth, averages)
        if coils is None:
            _, fid = self._render(truth, rng, npoints, dwell_s, averages)
            data = fid.reshape(1, 1, 1, npoints)
        else:
            _, fid = self._clean_fid(truth, npoints, dwell_s)
            channels = coils.sensitivities(self.position_mm, self.size_mm)[:, None] * fid[None, :]
            if sigma > 0.0:
                channels = channels + coils.noise(rng, sigma, npoints)
            data = np.transpose(channels.astype(np.complex64)).reshape(1, 1, 1, npoints, len(coils))

        hdr = Hdr_Ext(self._f0_mhz(), "1H")
        if coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
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
             "transmit": None if self.scan.transmit is None else self.scan.transmit.name},
            "MRSandbox: the virtual brain, the transmit map, and the voxel placement.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr)

    #*************#
    #   helpers   #
    #*************#
    def _flat_lines(self, truth, metabolites=True):
        """All components as two pruned numpy line lists with per-line widths: the
        Lorentzian lines (metabolites, water, lipids) as (freqs, amps, lw_hz), and
        the Voigt lines carrying a Gaussian width (macromolecules, downfield) as
        (freqs, amps, lw_hz, gauss_hz); either is None when empty. The split lets the
        engine render the many Lorentzian lines through its factorized graph and the
        few Voigt ones densely (`_render` pads each to a stable bucket, so every draw
        renders at the same shapes and the jitted graphs never retrace).

        Most of the d*d transition elements are silent (the coherence filters zero
        them), so dropping them before rendering cuts the render cost by orders of
        magnitude without touching the result.
        """
        parts = self.components(truth, metabolites=metabolites)
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
        t1 = tuple(float(self.resolve(truth.t1[name])) for name in names)
        t2 = tuple(float(self.resolve(truth.t2[name])) for name in names)
        b1_scale, offset_hz = self.b1_scale(truth), self.offset_hz(truth)
        cached = self.scan._lines
        if cached is None or cached[0] is not truth:
            cached = self.scan._lines = (truth, {})
        key = (self.compiled, t1, t2, b1_scale, offset_hz)
        if key in cached[1]:
            return cached[1][key]
        groups = tuple(self.scan.systems(name) for name in names)
        sequence = self.scan.sequence
        if self.compiled:
            fn, bounds = self.scan.engine.compiled_batch(sequence, groups)
            f, a = fn(ops.convert_to_tensor(np.asarray(t1, dtype=np.float32)),
                      ops.convert_to_tensor(np.asarray(t2, dtype=np.float32)),
                      ops.convert_to_tensor(np.float32(b1_scale)),
                      ops.convert_to_tensor(np.float32(offset_hz)))
            freqs, amps = np.asarray(ops.convert_to_numpy(f)), np.asarray(ops.convert_to_numpy(a))
        else:
            parts = [self.scan.engine.lines(
                sequence, *(s.with_relaxation(t1_s=t1[i], t2_s=t2[i]) for s in group),
                b1_scale=b1_scale, offset_hz=offset_hz) for i, group in enumerate(groups)]
            freqs = np.concatenate([np.asarray(ops.convert_to_numpy(f)) for f, _ in parts])
            amps = np.concatenate([np.asarray(ops.convert_to_numpy(a)) for _, a in parts])
            bounds = np.cumsum([0] + [sum(s.dim ** 2 for s in group) for group in groups])
        lines = {name: (freqs[bounds[i]:bounds[i + 1]], amps[bounds[i]:bounds[i + 1]])
                 for i, name in enumerate(names)}
        cached[1][key] = lines
        return lines

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
