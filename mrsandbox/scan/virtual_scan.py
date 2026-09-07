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
from mrsandbox.database.relaxation import WATER
from mrsandbox.engine import Engine
from mrsandbox.params import Normal, Params, PerTissue, Priors, Uniform
from mrsandbox.phantom.lipids import Skull
from mrsandbox.phantom.macromolecules import CoupledMM
from mrsandbox.phantom.water import Suppressed
from mrsandbox.spins import SpinSystem


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
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequence, params=None, database=None, water=None,
                 lipids=None, coils=None, field=None, mm=None):
        self.brain = brain
        self.scanner = scanner
        self.sequence = sequence
        self.database = database if database is not None else Database()
        self.engine = Engine(scanner)
        self.water = water if water is not None else Suppressed()
        self.lipids = lipids if lipids is not None else Skull()
        self.mm = mm if mm is not None else CoupledMM()
        self.coils = coils
        self.field = field
        self.params = self.default_params(self.database) | (params if params is not None else Params())
        self._compiled = {}

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
            # water relaxation comes from the cited WATER table and is weighted by each
            # voxel's composition at render time
            water=dict(residual_fraction=(1e-4, 3e-3),
                       t1_s=PerTissue(**WATER["t1_s"]),
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
            # and drift as a global shift, and an uncorrected zero-order phase
            acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0)),
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
            # ParameterizedMM alternative; the mm relaxation times are stated
            mm=dict(level_mm=(1.5, 4.0), kappa=(5.5, 9.5), t1_s=0.4, t2_s=0.03),
        )
        relaxation = Params(t2=t2, t1=Priors.literature_t1(names))
        return Params(conc=conc) | relaxation | nuisance

    #***********#
    #   voxel   #
    #***********#
    def voxel(self, position_mm, size_mm, compiled=False):
        """Place a cubic voxel and get its acquisition handle.

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
             oversampling=1, weighting=None):
        """Place an MRSI slice lattice and get its acquisition handle; see scan.mrsi.MRSI."""
        from mrsandbox.scan.mrsi import MRSI

        return MRSI(self, center_mm=center_mm, matrix=matrix, voxel_mm=voxel_mm,
                    oversampling=oversampling, weighting=weighting)

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
        self.size_mm = float(size_mm)
        self.fractions = scan.brain.fractions(position_mm, size_mm)
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
        parts = []
        for name, concentration in truth.conc.items():
            t2 = truth.t2[name]
            freqs, amps = self._metabolite_lines(name, truth.t1[name], t2, truth.b1_scale)
            weight = ops.convert_to_tensor(np.complex64(concentration * tissue))
            parts.append((freqs, amps * weight, 1.0 / (np.pi * t2), 0.0))
        water = self.scan.water.lines(engine, sequence, self.fractions, truth, gamma)
        if water is not None:
            _, water_t2 = self.scan.water.relaxation(self.fractions, truth)
            parts.append((*water, 1.0 / (np.pi * water_t2), 0.0))
        lipid = self.scan.lipids.lines(engine, sequence, self, truth, gamma)
        if lipid is not None:
            parts.append((*lipid, 1.0 / (np.pi * truth.lipids.t2_s), 0.0))
        mm = self.scan.mm.lines(engine, sequence, self.fractions, truth, self._f0_mhz())
        if mm is not None:
            # real positive amplitudes render absorptive as they are; the engine's
            # receiver factor compensates density matrix phases the MM lines never had
            freqs, amps, gauss = mm
            parts.append((ops.convert_to_tensor(freqs), ops.convert_to_tensor(amps),
                          0.0, ops.convert_to_tensor(gauss)))
        return parts

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
        """The noise-free voxel FID: components summed, B0 dephasing and the drawn
        acquisition imperfections (frequency shift, zero-order phase) applied."""
        freqs, amps, lw, gauss = self._flat_lines(truth)
        t, total = self.scan.engine.render_fid(freqs, amps, npoints=npoints, dwell_s=dwell_s,
                                               lw_hz=lw, gauss_hz=gauss)
        fid = np.asarray(ops.convert_to_numpy(total), dtype=np.complex64)
        grid = np.arange(npoints) * dwell_s
        mm_fid = self.scan.mm.fid(self.fractions, truth, self._f0_mhz(), grid)
        if mm_fid is not None:
            fid = fid + mm_fid
        if self.scan.field is not None:
            fid = fid * self._field_kernel(npoints, dwell_s)
        modulation = np.exp(1j * (2.0 * np.pi * truth.acquisition.shift_hz * grid
                                  + np.radians(truth.acquisition.phase_deg)))
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
            {"name": type(self.scan.sequence).__name__, **timings},
            "MRSandbox: the sequence and its timings in seconds.",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.scan.brain.name, "position_mm": list(self.position_mm), "size_mm": self.size_mm},
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

    def _metabolite_lines(self, name, t1_s, t2_s, b1_scale):
        """Lines of one metabolite, plain or through the compiled per-system path.

        Compiled, the jitted (t1_s, t2_s, b1_scale) -> (freqs, amps) functions are
        cached on the scan, so drawing fresh values per sample replays compiled graphs
        instead of re-tracing the whole simulation.
        """
        systems = SpinSystem.from_database(name, database=self.scan.database)
        if not self.compiled:
            relaxed = tuple(s.with_relaxation(t1_s=t1_s, t2_s=t2_s) for s in systems)
            return self.scan.engine.lines(self.scan.sequence, *relaxed, b1_scale=b1_scale)
        if name not in self.scan._compiled:
            self.scan._compiled[name] = tuple(
                self.scan.engine.compiled(self.scan.sequence, s) for s in systems
            )
        t1 = ops.convert_to_tensor(np.float32(t1_s))
        t2 = ops.convert_to_tensor(np.float32(t2_s))
        b1 = ops.convert_to_tensor(np.float32(b1_scale))
        lines = [fn(t1, t2, b1) for fn in self.scan._compiled[name]]
        return ops.concatenate([f for f, _ in lines]), ops.concatenate([a for _, a in lines])

    def _f0_mhz(self):
        return self.scan.database.gamma("1H") * self.scan.engine.b0_t
