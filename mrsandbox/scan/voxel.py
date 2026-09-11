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
    return int(np.random.SeedSequence().entropy) % 2 ** 53 if seed is None else seed


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

    #*****************#
    #   realization   #
    #*****************#
    def sample(self, seed=None):
        """Realize the scan's Params tree for this draw; the result is the ground truth."""
        return self.scan.params.sample(seed)

    #******************#
    #   components     #
    #******************#
    def components(self, truth):
        """Every signal component as (freqs, amps, lw_hz, gauss_hz): metabolites, water,
        lipids, and the macromolecule background.

        Every metabolite is its own component with its own pooled T1 and T2 (linewidth
        and echo attenuation stay consistent), and water and lipids carry theirs;
        renderings flatten all components into one pruned line list and render once.
        """
        engine, sequence = self.scan.engine, self.scan.sequence
        f0_mhz = self._f0_mhz()
        faded = self._diffusion_factor(truth)
        # an empirical downfield source carries everything above its cutoff, so the
        # spin systems' exchangeable NH lines up there leave the render (see
        # phantom.downfield.DownfieldSource)
        cutoff = self.scan.downfield.replaces_above_ppm
        parts = []
        for name, concentration in truth.conc.items():
            concentration = self.resolve(concentration)
            t1, t2 = self.resolve(truth.t1[name]), self.resolve(truth.t2[name])
            freqs, amps = self._metabolite_lines(name, truth, t1, t2)
            if cutoff is not None:
                amps = ops.where(freqs > cutoff * f0_mhz, ops.zeros_like(amps), amps)
            # the drawn concentration is the equal-mix value; the GM to WM contrast
            # ratio r splits it per tissue, keeping gm + wm at r = 1 exactly
            r = truth.contrast[name]
            tissue = (2.0 * r * self.fractions.gm + 2.0 * self.fractions.wm) / (1.0 + r)
            weight = ops.convert_to_tensor(
                np.complex64(concentration * tissue * faded(truth.diffusion.adc[name])))
            parts.append((freqs, amps * weight, 1.0 / (np.pi * t2), 0.0))
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

    #**************#
    #   fading     #
    #**************#
    def _diffusion_factor(self, truth):
        """exp(-b ADC) for this scan's b-value: per-tissue ADC leaves weighted by the
        local composition, scalars passing through; the identity at b = 0."""
        b = self.scan.b_value_ms_um2

        def faded(adc):
            if b == 0.0:
                return 1.0
            return float(np.exp(-b * self.resolve(adc)))

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

    def fid(self, seed=None, npoints=2048, dwell_s=1.0 / 4000.0):
        """(t, complex FID) at any grid: all components summed, sampled noise added."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        truth = self.scan.params.sample(rng)
        t, fid = self._render(truth, rng, npoints, dwell_s)
        return np.asarray(ops.convert_to_numpy(t)), fid

    #*****************#
    #   realization   #
    #*****************#
    def _render(self, truth, rng, npoints, dwell_s):
        """Render all components in one call and add the sampled complex Gaussian noise."""
        t, fid = self._clean_fid(truth, npoints, dwell_s)
        if truth.noise.sigma > 0.0:
            fid = fid + receiver_noise(rng, truth.noise.sigma, npoints)
        return t, fid

    def _clean_fid(self, truth, npoints, dwell_s):
        """The noise-free voxel FID: components summed, B0 dephasing, the drawn spurious
        echo and eddy-current phase applied through Augmentrum's modules, then the
        drawn frequency shift and zero-order phase.

        The spurious echo is the drawn AGNOSTIC-form echo (amplitude relative to the
        FID maximum, centre, Gaussian width, position, phase) rendered by Augmentrum's
        localized echo with its Gaussian envelope exp(-(t - t_echo)^2 / (2 T2^2)),
        so T2 = 1 / sqrt(2 W). The eddy phase is Augmentrum's synthetic trajectory
        (low-pass filtered Gaussian phase noise, its linear trend removed) with the
        drawn standard deviation, cut-off, and seed, so it regenerates from the truth.
        """
        try:
            from augmentrum.augmentation import EddyCurrent, SpuriousEchoes
        except ImportError as error:
            raise ImportError("acquisitions need Augmentrum for the spurious echo and eddy "
                              "phase (the mrsandbox[augment] extra; see pyproject.toml "
                              "for the install recipe)") from error
        lorentzian, voigt = self._flat_lines(truth)
        t, total = self.scan.engine.render_fid(lorentzian[0], lorentzian[1], npoints=npoints,
                                               dwell_s=dwell_s, lw_hz=lorentzian[2])
        fid = np.asarray(ops.convert_to_numpy(total), dtype=np.complex64)
        if voigt is not None:
            _, extra = self.scan.engine.render_fid(voigt[0], voigt[1], npoints=npoints,
                                                   dwell_s=dwell_s, lw_hz=voigt[2],
                                                   gauss_hz=voigt[3])
            fid = fid + np.asarray(ops.convert_to_numpy(extra), dtype=np.complex64)
        grid = np.arange(npoints) * dwell_s
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
            eddy = EddyCurrent(std_rad=acquisition.eddy_std_rad,
                               lp_cut_hz=acquisition.eddy_lp_cut_hz,
                               seed=int(acquisition.eddy_seed))
            fid = np.asarray(eddy.process_tensor(fid[None, :], sw_hz=sw_hz)[0][0])
        modulation = np.exp(1j * (2.0 * np.pi * acquisition.shift_hz * grid
                                  + np.radians(acquisition.phase_deg)))
        return t, (fid * modulation).astype(np.complex64)

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
    def acquire(self, seed=None, npoints=2048, dwell_s=1.0 / 4000.0):
        """Acquire the voxel and return NIfTI-MRS carrying the full provenance.

        The header extension holds SimulationGroundTruth (seed, realized parameters,
        tissue fractions), SimulationSequence (name and timings), and SimulationPhantom
        (brain, position, size), so any spectrum can be regenerated from its own file.

        With a CoilArray on the scan, every coil receives the FID weighted by its
        complex sensitivity at this voxel plus correlated noise (covariance sigma^2 R),
        and the data gains a DIM_COIL dimension.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        seed = acquisition_seed(seed)
        rng = np.random.default_rng(seed)
        truth = self.scan.params.sample(rng)
        coils = self.scan.coils
        if coils is None:
            _, fid = self._render(truth, rng, npoints, dwell_s)
            data = fid.reshape(1, 1, 1, npoints)
        else:
            _, fid = self._clean_fid(truth, npoints, dwell_s)
            channels = coils.sensitivities(self.position_mm, self.size_mm)[:, None] * fid[None, :]
            if truth.noise.sigma > 0.0:
                channels = channels + coils.noise(rng, truth.noise.sigma, npoints)
            data = np.transpose(channels.astype(np.complex64)).reshape(1, 1, 1, npoints, len(coils))

        hdr = Hdr_Ext(self._f0_mhz(), "1H")
        if coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
        timings = {k: float(v) for k, v in vars(self.scan.sequence).items()
                   if k.endswith("_s") and isinstance(v, (int, float))}
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "params": truth.to_dict(), "tissue_fractions": dict(self.fractions._asdict()),
             "local": self.local(truth)},
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
    def _flat_lines(self, truth):
        """All components as two pruned line lists with per-line widths: the
        Lorentzian lines (metabolites, water, lipids) as (freqs, amps, lw_hz), and
        the Voigt lines carrying a Gaussian width (macromolecules, downfield) as
        (freqs, amps, lw_hz, gauss_hz), or None when there are none. The split lets
        the engine render the many Lorentzian lines through its factorized graph and
        the few Voigt ones densely.

        Most of the d*d transition elements are silent (the coherence filters zero
        them), so dropping them before rendering cuts the render cost by orders of
        magnitude without touching the result.
        """
        parts = self.components(truth)
        freqs = np.concatenate([np.asarray(ops.convert_to_numpy(f)) for f, _, _, _ in parts])
        amps = np.concatenate([np.asarray(ops.convert_to_numpy(a)) for _, a, _, _ in parts])
        lw = np.concatenate([np.full(f.shape[0], lw, dtype=np.float32) for f, _, lw, _ in parts])
        gauss = np.concatenate([
            np.asarray(ops.convert_to_numpy(g), dtype=np.float32) * np.ones(f.shape[0], np.float32)
            for f, _, _, g in parts
        ])
        keep = np.abs(amps) > 1e-7 * np.abs(amps).max()
        voigt = keep & (gauss > 0.0)

        def bucket(mask, *columns):
            # pad to a stable bucket size so every draw renders at the same shapes (the
            # jax op cache is keyed by shape; a fresh shape per draw would recompile)
            count = int(mask.sum())
            pad = -(-count // 512) * 512 - count
            return tuple(ops.convert_to_tensor(np.pad(values[mask], (0, pad), constant_values=fill))
                         for values, fill in columns)

        lorentzian = bucket(keep & ~voigt, (freqs, 0.0), (amps, 0.0), (lw, 1.0))
        if not voigt.any():
            return lorentzian, None
        return lorentzian, bucket(voigt, (freqs, 0.0), (amps, 0.0), (lw, 1.0), (gauss, 0.0))

    def _metabolite_lines(self, name, truth, t1_s, t2_s):
        """Unweighted lines of one metabolite for one truth and its resolved relaxation
        times, plain or through the compiled per-system path.

        The lines depend on the draw (relaxation, b1) but not on the voxel position
        beyond the resolved times, so they are cached per truth on the scan keyed by
        those times: one truth serving many voxels (MRSI, multiplex) simulates each
        metabolite once, and every voxel applies only its own weights. Compiled, the
        jitted (t1_s, t2_s, b1_scale) -> (freqs, amps) functions are cached on the
        engine, so drawing fresh values per sample replays compiled graphs instead of
        re-tracing the whole simulation.
        """
        cached = self.scan._lines
        if cached is None or cached[0] is not truth:
            cached = self.scan._lines = (truth, {})
        b1_scale, offset_hz = self.b1_scale(truth), self.offset_hz(truth)
        key = (name, self.compiled, t1_s, t2_s, b1_scale, offset_hz)
        if key in cached[1]:
            return cached[1][key]
        systems = self.scan.systems(name)
        if self.compiled:
            args = [ops.convert_to_tensor(np.float32(v)) for v in (t1_s, t2_s, b1_scale, offset_hz)]
            parts = [self.scan.engine.compiled(self.scan.sequence, s)(*args) for s in systems]
            lines = (ops.concatenate([f for f, _ in parts]),
                     ops.concatenate([a for _, a in parts]))
        else:
            relaxed = tuple(s.with_relaxation(t1_s=t1_s, t2_s=t2_s) for s in systems)
            lines = self.scan.engine.lines(self.scan.sequence, *relaxed, b1_scale=b1_scale,
                                           offset_hz=offset_hz)
        cached[1][key] = lines
        return lines

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
