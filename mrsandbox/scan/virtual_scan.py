####################################################################################################
#                                         virtual_scan.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The composition layer and public face: VirtualScan binds a brain, a scanner, and a      #
#          sequence under one Params tree; a Voxel realizes the parameters at a position, weights  #
#          each metabolite by its concentration and the local tissue, and returns the acquisition  #
#          as NIfTI-MRS with the seed and the full ground truth in the header.                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from keras import ops

from mrsandbox.database import Database, concentrations
from mrsandbox.database.relaxation import BOUDREAU, WATER
from mrsandbox.engine import Engine
from mrsandbox.params import Clipped, Normal, Params, PerTissue, Priors, Tissues, Uniform
from mrsandbox.phantom.lipids import Skull
from mrsandbox.phantom.macromolecules import CoupledMM
from mrsandbox.phantom.water import Suppressed
from mrsandbox.spins import SpinSystem


#**********************#
#   acquisition seed   #
#**********************#
def acquisition_seed(seed):
    """A serializable seed for an acquisition header: an int passes through, None draws
    a fresh one, so the recorded value always regenerates the file. A Generator is
    refused because its state cannot be recorded; use fid() for generator-driven draws."""
    if isinstance(seed, np.random.Generator):
        raise TypeError("acquire needs a serializable seed (an int, or None for a fresh "
                        "one) so the file can regenerate itself; use fid() for "
                        "generator-driven draws")
    return int(np.random.SeedSequence().entropy) if seed is None else seed


#**************************************************************************************************#
#                                        Class VirtualScan                                         #
#**************************************************************************************************#
#                                                                                                  #
# A brain, a scanner, and a sequence, configured by one Params tree.                               #
#                                                                                                  #
#**************************************************************************************************#
class VirtualScan:
    """A brain, a scanner, and a sequence, configured by one Params tree.

    Args:
        brain: A phantom.Brain.
        scanner: A hardware.Scanner, or a field strength in tesla.
        sequence: A sequences.Sequence.
        params: Params overriding the defaults, composed as defaults | params.
        database: Database instance; defaults to the packaged one.
        coils: A hardware.CoilArray for multi-channel reception; None gives one ideal
            channel.
        field: A phantom.FieldMap for B0 inhomogeneity; None gives a uniform field.
        mm: A phantom.MMSource; defaults to the LCModel-convention parameterization.
        isochromats: Ensemble size for gradient-bearing sequences, handed to the
            engine; raise it for dense multi-axis gradient schemes.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequence, params=None, database=None, water=None,
                 lipids=None, coils=None, field=None, mm=None, isochromats=32):
        self.brain = brain
        self.scanner = scanner
        self.sequence = sequence
        self.database = database if database is not None else Database()
        self.engine = Engine(scanner, isochromats=isochromats)
        self.water = water if water is not None else Suppressed()
        self.lipids = lipids if lipids is not None else Skull()
        self.mm = mm if mm is not None else CoupledMM()
        self.coils = coils
        self.field = field
        self.params = self.default_params(self.database) | (params if params is not None else Params())
        self._lines = None
        self._b_value = None

    #*************************#
    #   diffusion weighting   #
    #*************************#
    @property
    def b_value_ms_um2(self):
        """The b-value the sequence's own gradient events build up, cached; see
        sequences.diffusion for the consensus formulation."""
        if self._b_value is None:
            from mrsandbox.sequences.diffusion import b_value_ms_um2

            self._b_value = b_value_ms_um2(self.sequence)
        return self._b_value

    #************************#
    #   default parameters   #
    #************************#
    @staticmethod
    def default_params(database=None):
        """Concentration priors and pooled literature relaxation for every metabolite that
        has both a spin system and a de Graaf range. Total choline is covered by GPC and
        PCho, its two constituents, so there is no separate Cho entry.
        """
        db = database if database is not None else Database()
        names = [name for name in db.metabolites() if name in concentrations.DE_GRAAF]
        conc = Priors.de_graaf(metabolites=names, database=db)
        t2 = Priors.gudmundson_t2(names)  # pooled where the literature has rows, Fixed fallback
        nuisance = Params(
            # residual fraction after suppression is a stated assumption; the per-tissue
            # water relaxation is weighted by each voxel's composition at render time.
            # T1 draws from the crowdsourced 3 T reproducibility challenge (the typed
            # BOUDREAU table: cortical GM, and the genu for WM, the challenge's
            # cleanest ROI; the intersubmission sd is a realistic site-to-site spread,
            # clipped at a stated 0.2 s floor); CSF T1 and the T2 values keep the
            # cited WATER table entries
            water=dict(residual_fraction=(1e-4, 3e-3),
                       t1_s=PerTissue(
                           gm=Clipped(Normal(*BOUDREAU["t1_s"]["gm_cortex"]), low=0.2),
                           wm=Clipped(Normal(*BOUDREAU["t1_s"]["wm_genu"]), low=0.2),
                           csf=WATER["t1_s"]["csf"]),
                       t2_s=PerTissue(**WATER["t2_s"])),
            # adipose proton concentration is water-like (about 60 to 70 M), which is why
            # a voxel clipping the skull drowns in fat; the range is a stated assumption
            lipids=dict(amplitude_mm=(30000.0, 70000.0), t2_s=0.05, t1_s=0.3),
            # complex Gaussian noise per FID point, acquisition side; range is an assumption
            noise=dict(sigma=(0.5, 3.0)),
            # transmit field scale: every pulse flip is multiplied by the drawn value;
            # a few percent calibration scatter at 3 T is a stated assumption
            b1_scale=Normal(1.0, 0.03),
            # receiver-side imperfections, stated assumptions: scanner frequency error
            # and drift as a global shift, an uncorrected zero-order phase, and a
            # residual eddy current as a decaying phase on the FID (the leading term of
            # the multi-exponential gradient response, de Graaf 2019; in vivo the bulk
            # is removed by Klose 1990 ECC, so the residual amplitude stays modest)
            acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0),
                             eddy_deg=Uniform(-30.0, 30.0), eddy_tau_s=(0.01, 0.1)),
            # spurious echo: out-of-voxel magnetization that imperfect crushing or phase
            # cycling fails to remove and that rephases mid-acquisition (the mechanism
            # is demonstrated physically by the crusher gates in tests/sequences). The
            # form and every range are typed from the AGNOSTIC OOV synthesis
            # (Gudmundson et al. 2023, Imaging Neuroscience, imag_a_00025): a Gaussian
            # envelope echo a exp(-W (t - tau)^2) exp(i (2 pi f t + phi)) with tau
            # 10-400 ms, W 500-8000 1/s^2 (FWHM 18-74 ms), position 1-4 ppm, amplitude
            # 0.1-20% of the FID maximum, free phase; the source adds one echo to 85%
            # of examples, here it is always on and amplitude=0.0 switches it off
            spurious=dict(amplitude=(0.001, 0.2), tau_s=(0.010, 0.400),
                          width_s2=(500.0, 8000.0), ppm=(1.0, 4.0),
                          phase_deg=(0.0, 360.0)),
            # scanner and subject instabilities across repeated transients. The drift
            # rate is grounded in Hui et al. 2021 (NeuroImage 241:118430, 80 sites,
            # 99 scanners at 3 T): median 0.29 Hz/min before fMRI heating with an
            # IQR-derived spread, 0.43 after. The shot-to-shot jitter, phase scatter,
            # amplitude instability, and motion event model are stated assumptions,
            # every leaf overridable.
            transients=dict(
                drift_hz_per_min=Normal(0.29, 0.18),
                jitter_hz=0.2,
                phase_jitter_deg=4.0,
                amplitude_sd=0.02,
                motion_rate_per_min=(0.0, 0.5),
                motion_shift_hz=(1.0, 8.0),
                motion_phase_deg=(-45.0, 45.0),
            ),
            # macromolecules: the coupled default scales as kappa times the challenge
            # tissue ratios times the drawn creatine. The challenge kappa of 0.4 applies
            # to its own measured basis normalization, not to concentration units, so
            # kappa here is recalibrated (stated) to land the short echo time MM area
            # fraction in the consensus 10 to 25 percent ballpark; level_mm drives the
            # ParameterizedMM alternative. Per-component relaxation is typed from the
            # consensus Appendix S1 3 T human columns (T2: Landheer 2020, else
            # Hoefemann 2020; T1: Hoefemann 2020, fit-limit rows excluded; the
            # database.macromolecules CONSENSUS tables), components mapped by peak
            # identity (challenge M0.91 is consensus M0.94 and so on, positions vary
            # 0.01 to 0.04 ppm between papers). Where the table prints an error the
            # leaf draws from it, clipped at a 5 ms floor (a stated plausibility
            # bound); values without errors and components without a 3 T value stay
            # fixed at the typed value or the stated fallback
            mm=dict(level_mm=(1.5, 4.0), kappa=(5.5, 9.5),
                    t1=dict(mm091=0.290, mm121=0.309, mm143=0.400, mm167=0.225,
                            mm226=0.263, mm321=0.400, mm377=0.400),
                    t2=dict(mm091=Clipped(Normal(0.027, 0.002), low=0.005),
                            mm121=Clipped(Normal(0.039, 0.013), low=0.005),
                            mm143=Clipped(Normal(0.018, 0.004), low=0.005),
                            mm167=Clipped(Normal(0.017, 0.003), low=0.005),
                            mm226=0.024,
                            mm321=Clipped(Normal(0.018, 0.005), low=0.005),
                            mm377=0.030)),
            # diffusion: every component attenuates by exp(-b ADC) with b built from
            # the sequence's own gradient events (sequences.diffusion, the consensus
            # Box 2 of Ligneul et al. 2024, MRM 91:860), so a standard sequence is
            # only as diffusion weighted as its crushers make it. ADCs in um^2/ms:
            # metabolites follow the human long-diffusion-time summary the consensus
            # defers to (Valette et al. 2018, Front Neurosci 12:2: about 0.10 in grey
            # and 0.15 in white matter, weighted by the voxel composition); the MM
            # range is the consensus's own 0.005 to 0.01; water and lipids are stated
            diffusion=dict(
                adc={name: PerTissue(gm=0.10, wm=0.15, csf=0.10) for name in names},
                water=0.75,
                mm=(0.005, 0.01),
                lipids=0.0),
        )
        relaxation = Params(t2=t2, t1=Priors.literature_t1(names))
        return Params(conc=conc) | relaxation | nuisance

    #***********#
    #   voxel   #
    #***********#
    def voxel(self, position_mm, size_mm, compiled=False):
        """Place a voxel (a size_mm cube, or an (x, y, z) box) and get its acquisition
        handle.

        With `compiled`, metabolite lines run through jitted per-system functions cached
        on the scan: the first draw compiles, every further draw replays. Worth it for
        many draws from one scan (the training mode turns it on); a one-shot look is
        faster plain.
        """
        return Voxel(self, position_mm, size_mm, compiled=compiled)

    #***********#
    #   mrsi    #
    #***********#
    def mrsi(self, center_mm=(0.0, 0.0, 0.0), matrix=(16, 16), voxel_mm=10.0,
             thickness_mm=None, oversampling=1, weighting=None, sampling=None,
             profile=None, averages=None, trajectory=None):
        """Place an MRSI slice lattice and get its acquisition handle; see scan.mrsi.MRSI."""
        from mrsandbox.scan.mrsi import MRSI

        return MRSI(self, center_mm=center_mm, matrix=matrix, voxel_mm=voxel_mm,
                    thickness_mm=thickness_mm, oversampling=oversampling,
                    weighting=weighting, sampling=sampling, profile=profile,
                    averages=averages, trajectory=trajectory)

    #*****************#
    #   training      #
    #*****************#
    def generator(self, position_mm, size_mm, batch_size=16, seed=None, npoints=2048, dwell_s=1.0 / 4000.0):
        """A seeded on-the-fly training stream for this voxel; see scan.generator.Generator."""
        from mrsandbox.scan.generator import Generator

        return Generator(self.voxel(position_mm, size_mm, compiled=True), batch_size=batch_size,
                         seed=seed, npoints=npoints, dwell_s=dwell_s)


#**************************************************************************************************#
#                                           Class Voxel                                            #
#**************************************************************************************************#
#                                                                                                  #
# One placed voxel: realized parameters, weighted lines, renderings, and NIfTI-MRS out.            #
#                                                                                                  #
#**************************************************************************************************#
class Voxel:
    """One placed voxel: realized parameters, weighted lines, renderings, and NIfTI-MRS out.

    Metabolite amplitudes scale with the sampled concentration times the tissue presence
    (gm + wm) of the voxel; the linewidth follows the sampled T2 as 1 / (pi T2).
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

    #*****************#
    #   realization   #
    #*****************#
    def sample(self, seed=None):
        """Realize the scan's Params tree for this draw; the result is the ground truth."""
        return self.scan.params.sample(seed)

    #********************#
    #   weighted lines   #
    #********************#
    def lines(self, seed=None):
        """Concatenated metabolite transition lines, amplitude weighted.

        Returns:
            (freqs, amps, truth): line tensors plus the realized Params.
        """
        truth = self.sample(seed)
        tissue = self.fractions.gm + self.fractions.wm
        all_freqs, all_amps = [], []
        for name, concentration in truth.conc.items():
            systems = SpinSystem.from_database(name, database=self.scan.database)
            freqs, amps = self.scan.engine.lines(self.scan.sequence, *systems)
            weight = ops.convert_to_tensor(np.complex64(concentration * tissue))
            all_freqs.append(freqs)
            all_amps.append(amps * weight)
        return ops.concatenate(all_freqs), ops.concatenate(all_amps), truth

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
        gamma = self.scan.database.gamma("1H")
        engine, sequence = self.scan.engine, self.scan.sequence
        tissue = self.fractions.gm + self.fractions.wm
        faded = self._diffusion_factor(truth)
        parts = []
        for name, concentration in truth.conc.items():
            t2 = truth.t2[name]
            freqs, amps = self._metabolite_lines(name, truth)
            weight = ops.convert_to_tensor(
                np.complex64(concentration * tissue * faded(truth.diffusion.adc[name])))
            parts.append((freqs, amps * weight, 1.0 / (np.pi * t2), 0.0))
        water = self.scan.water.lines(engine, sequence, self.fractions, truth, gamma)
        if water is not None:
            _, water_t2 = self.scan.water.relaxation(self.fractions, truth)
            w_freqs, w_amps = water
            parts.append((w_freqs, w_amps * np.complex64(faded(truth.diffusion.water)),
                          1.0 / (np.pi * water_t2), 0.0))
        lipid = self.scan.lipids.lines(engine, sequence, self, truth, gamma)
        if lipid is not None:
            l_freqs, l_amps = lipid
            parts.append((l_freqs, l_amps * np.complex64(faded(truth.diffusion.lipids)),
                          1.0 / (np.pi * truth.lipids.t2_s), 0.0))
        mm = self.scan.mm.lines(engine, sequence, self.fractions, truth, self._f0_mhz())
        if mm is not None:
            # real positive amplitudes render absorptive as they are; the engine's
            # receiver factor compensates density matrix phases the MM lines never had
            freqs, amps, gauss = mm
            parts.append((ops.convert_to_tensor(freqs),
                          ops.convert_to_tensor(amps * faded(truth.diffusion.mm)),
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
            if isinstance(adc, Tissues):
                total = self.fractions.gm + self.fractions.wm + self.fractions.csf
                adc = adc.gm if total <= 0.0 else (
                    self.fractions.gm * adc.gm + self.fractions.wm * adc.wm
                    + self.fractions.csf * adc.csf) / total
            return float(np.exp(-b * adc))

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
            fid = fid + truth.noise.sigma * (
                rng.standard_normal(npoints) + 1j * rng.standard_normal(npoints)
            ).astype(np.complex64)
        return t, fid

    def _clean_fid(self, truth, npoints, dwell_s):
        """The noise-free voxel FID: components summed, B0 dephasing, the drawn spurious
        echo, and the drawn acquisition imperfections (frequency shift, zero-order
        phase, decaying eddy-current phase) applied."""
        freqs, amps, lw, gauss = self._flat_lines(truth)
        t, total = self.scan.engine.render_fid(freqs, amps, npoints=npoints, dwell_s=dwell_s,
                                               lw_hz=lw, gauss_hz=gauss)
        fid = np.asarray(ops.convert_to_numpy(total), dtype=np.complex64)
        grid = np.arange(npoints) * dwell_s
        mm_fid = self.scan.mm.fid(self.fractions, truth, self._f0_mhz(), grid,
                                  sequence=self.scan.sequence)
        if mm_fid is not None:
            fid = fid + mm_fid * np.float32(
                self._diffusion_factor(truth)(truth.diffusion.mm))
        if self.scan.field is not None:
            fid = fid * self._field_kernel(npoints, dwell_s)
        spur = truth.spurious
        if spur.amplitude > 0.0:
            # the AGNOSTIC-form out-of-voxel echo, scaled to this voxel's FID maximum
            # and entering before the receiver-side modulation like any received signal
            echo = np.exp(-spur.width_s2 * (grid - spur.tau_s) ** 2) * np.exp(
                1j * (2.0 * np.pi * spur.ppm * self._f0_mhz() * grid
                      + np.radians(spur.phase_deg)))
            fid = fid + spur.amplitude * np.abs(fid).max() * echo.astype(np.complex64)
        modulation = np.exp(1j * (2.0 * np.pi * truth.acquisition.shift_hz * grid
                                  + np.radians(truth.acquisition.phase_deg)
                                  + np.radians(truth.acquisition.eddy_deg)
                                  * np.exp(-grid / truth.acquisition.eddy_tau_s)))
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
            channels = coils.sensitivities(self.position_mm)[:, None] * fid[None, :]
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
            {"seed": seed, "params": truth.to_dict(), "tissue_fractions": dict(self.fractions._asdict())},
            "MRSandbox: the realized parameter tree of this acquisition.",
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
             "size_mm": list(self.size_mm) if isinstance(self.size_mm, tuple) else self.size_mm},
            "MRSandbox: the virtual brain and the voxel placement.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr)

    #*************#
    #   helpers   #
    #*************#
    def _flat_lines(self, truth):
        """All components as one pruned line list with per-line linewidths.

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
        # pad to a stable bucket size so every draw renders at the same shapes (the jax
        # op cache is keyed by shape; a fresh shape per draw would recompile every time)
        count = int(keep.sum())
        bucket = -(-count // 512) * 512
        freqs, amps, lw, gauss = freqs[keep], amps[keep], lw[keep], gauss[keep]
        pad = bucket - count
        return (ops.convert_to_tensor(np.pad(freqs, (0, pad))),
                ops.convert_to_tensor(np.pad(amps, (0, pad))),
                ops.convert_to_tensor(np.pad(lw, (0, pad), constant_values=1.0)),
                ops.convert_to_tensor(np.pad(gauss, (0, pad))))

    def _metabolite_lines(self, name, truth):
        """Unweighted lines of one metabolite for one truth, plain or through the
        compiled per-system path.

        The lines depend on the draw (relaxation, b1) but not on the voxel position,
        so they are cached per truth on the scan: one truth serving many voxels (MRSI,
        multiplex) simulates each metabolite once, and every voxel applies only its own
        weights. Compiled, the jitted (t1_s, t2_s, b1_scale) -> (freqs, amps) functions
        are cached on the engine, so drawing fresh values per sample replays compiled
        graphs instead of re-tracing the whole simulation.
        """
        cached = self.scan._lines
        if cached is None or cached[0] is not truth:
            cached = self.scan._lines = (truth, {})
        key = (name, self.compiled)
        if key in cached[1]:
            return cached[1][key]
        systems = SpinSystem.from_database(name, database=self.scan.database)
        if self.compiled:
            t1 = ops.convert_to_tensor(np.float32(truth.t1[name]))
            t2 = ops.convert_to_tensor(np.float32(truth.t2[name]))
            b1 = ops.convert_to_tensor(np.float32(truth.b1_scale))
            parts = [self.scan.engine.compiled(self.scan.sequence, s)(t1, t2, b1) for s in systems]
            lines = (ops.concatenate([f for f, _ in parts]),
                     ops.concatenate([a for _, a in parts]))
        else:
            relaxed = tuple(s.with_relaxation(t1_s=truth.t1[name], t2_s=truth.t2[name])
                            for s in systems)
            lines = self.scan.engine.lines(self.scan.sequence, *relaxed, b1_scale=truth.b1_scale)
        cached[1][key] = lines
        return lines

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
