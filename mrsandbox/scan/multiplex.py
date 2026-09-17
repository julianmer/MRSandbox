####################################################################################################
#                                          multiplex.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Multiplexed acquisition: several named conditions (edited ON and OFF, a TE series, any  #
#          set of sequences) acquired against ONE realized truth, stacked on the standard          #
#          NIfTI-MRS editing dimension. This is the object a closed acquisition loop reasons       #
#          over: what the conditions measure jointly about one subject.                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox.engine import Engine
from mrsandbox.scan.virtual_scan import VirtualScan
from mrsandbox.scan.voxel import draw, receiver_noise


#**************************************************************************************************#
#                                         Class Multiplex                                          #
#**************************************************************************************************#
#                                                                                                  #
# Named acquisition conditions sharing one brain, scanner, parameter tree, and truth draw.         #
#                                                                                                  #
#**************************************************************************************************#
class Multiplex:
    """Named acquisition conditions sharing one brain, scanner, parameter tree, and
    truth draw.

    Every condition is a full VirtualScan differing only in its sequence, so water,
    lipids, macromolecules, B0, and coils apply identically; one parameter draw serves
    all conditions (the subject does not change between sub-acquisitions), while noise
    is drawn per condition. `acquire` stacks the conditions on DIM_EDIT in their given
    order.

    Args:
        brain, scanner: As for VirtualScan.
        sequences: Mapping of condition name to Sequence (e.g. MegaPress.pair()).
        combinations: Optional mapping of combination name to per-condition weights
            (e.g. Hermes.combinations); recorded in the header and applied by combine().
        compiled: Run the conditions through the engine's jitted per-system graphs, as
            VirtualScan.voxel does: the first acquisition compiles every system of every
            condition (seconds), every further one replays. Worth it for a closed loop
            acquiring many times from one Multiplex; a one-shot acquisition is faster
            plain.
        **kwargs: Forwarded to every condition's VirtualScan (params, water, coils...).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequences, combinations=None, compiled=False, **kwargs):
        self.conditions = ()
        self.combinations = dict(combinations) if combinations else {}
        self.compiled = compiled
        self.brain = brain
        self.scanner = scanner
        # one engine for every condition: designs of one structure share its graphs
        self._kwargs = dict(kwargs, engine=Engine(scanner, isochromats=kwargs.pop("isochromats", 32)))
        self.scans = {}
        for name, sequence in sequences.items():
            self.add(name, sequence)
        self.params = self.scans[self.conditions[0]].params

    #*****************#
    #   conditions    #
    #*****************#
    def add(self, name, sequence):
        """Register one more condition (a loop's next candidate) on the shared
        engine, brain, scanner, and parameter tree."""
        self.scans[name] = VirtualScan(self.brain, self.scanner, sequence, **self._kwargs)
        self.conditions = self.conditions + (name,)

    #*****************#
    #   acquisition   #
    #*****************#
    def acquire(self, position_mm, size_mm, seed=None, npoints=2048, dwell_s=1.0 / 4000.0,
                conditions=None, step=0, averages=1):
        """Acquire conditions of one voxel and return NIfTI-MRS on DIM_EDIT.

        Returns data of shape (1, 1, 1, npoints, n_conditions); the header records the
        condition order, the shared realized truth, and the voxel placement. The
        subject is the seed's; `conditions` selects a subset in the given order
        (default all), `step` the noise stream of the seed (a loop acquiring one
        subject over many steps takes fresh noise each time), and every condition is
        the average of `averages` transients.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        conditions = tuple(self.conditions if conditions is None else conditions)
        seed, truth, rng = draw(self.params, seed, step)
        stack = []
        for name in conditions:
            voxel = self.scans[name].voxel(position_mm, size_mm, compiled=self.compiled)
            _, fid = voxel._clean_fid(truth, npoints, dwell_s)
            sigma = voxel.noise_sigma(truth, averages)
            if sigma > 0.0:
                fid = fid + receiver_noise(rng, sigma, npoints)
            stack.append(fid)
        data = np.stack(stack, axis=-1).reshape(1, 1, 1, npoints, len(conditions))

        voxel = self.scans[conditions[0]].voxel(position_mm, size_mm)
        hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
        hdr.set_dim_info(0, "DIM_EDIT")
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "step": int(step), "averages": int(averages), "params": truth.to_dict(),
             "tissue_fractions": dict(voxel.fractions._asdict())},
            "MRSandbox: the one realized parameter tree shared by every condition.",
        )
        record = {"names": list(conditions),
                  "sequences": [type(self.scans[n].sequence).__name__ for n in conditions]}
        if self.combinations:
            record["combinations"] = self.combinations
        hdr.set_user_def(
            "SimulationConditions",
            record,
            "MRSandbox: the multiplexed conditions on DIM_EDIT, in order.",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.brain.name, "position_mm": [float(x) for x in position_mm],
             "size_mm": list(size_mm) if hasattr(size_mm, "__len__") else float(size_mm)},
            "MRSandbox: the virtual brain and the voxel placement.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr)

    #*************#
    #   combine   #
    #*************#
    def combine(self, acquisition, name):
        """One named combination of an acquired condition stack (e.g. the Hadamard GABA
        difference): the weighted sum over DIM_EDIT, returned as a complex FID array."""
        weights = self.combinations[name]
        w = np.asarray([weights[c] for c in self.conditions], dtype=np.float32)
        return np.asarray(acquisition[:])[0, 0, 0] @ w

    #****************#
    #   transients   #
    #****************#
    def transients(self, position_mm, size_mm, schedule, tr_s=2.0, seed=None,
                   npoints=2048, dwell_s=1.0 / 4000.0, synthesis=None):
        """Acquire a schedule of transients and return NIfTI-MRS on DIM_DYN.

        The schedule names one condition per transient (a closed loop may interleave
        them arbitrarily); an integer repeats the first condition. One truth serves the
        whole run, while the scanner and the subject evolve across it through
        Augmentrum's TransientSynthesizer: frequency drift, autoregressive wander and
        breathing, a phase coupled to the frequency plus receiver jitter, amplitude
        wander, growing broadening, and sparse motion events, its 3 T envelope by
        default. Its four per-transient tracks (frequency, phase, amplitude, extra
        Lorentzian broadening) multiply every transient's clean FID as Augmentrum's own
        train factor does, a exp(-pi L t) exp(i (2 pi f t + phi)), and receiver noise is
        drawn per transient. The realized tracks are recorded in the header, so
        alignment algorithms can be scored against the truth.

        Args:
            position_mm, size_mm: The voxel placement.
            schedule: Condition names per transient, or a transient count.
            tr_s: Repetition time, the acquisition clock of the tracks.
            seed: As for acquire; the synthesizer's own seed derives from it.
            npoints, dwell_s: The readout grid.
            synthesis: Keyword arguments for Augmentrum's TransientSynthesizer (its
                ranges, e.g. drift_hz_per_min=0.0 and events_per_min=0.0 for a still
                scanner and subject); the transient count, TR, and seed are set here.
        """
        try:
            import augmentrum
            from augmentrum.augmentation import TransientSynthesizer
        except ImportError as error:
            raise ImportError("transients need Augmentrum (the mrsandbox[augment] extra; "
                              "its fsl-mrs dependency is not on PyPI, see pyproject.toml "
                              "for the install recipe)") from error
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        if isinstance(schedule, int):
            schedule = [self.conditions[0]] * schedule
        schedule = list(schedule)
        seed, truth, rng = draw(self.params, seed)

        clean = {}
        for name in set(schedule):
            voxel = self.scans[name].voxel(position_mm, size_mm, compiled=self.compiled)
            _, clean[name] = voxel._clean_fid(truth, npoints, dwell_s)

        n = len(schedule)
        synthesis = dict(synthesis or {})
        synth_seed = int(rng.integers(2 ** 31))
        synth = TransientSynthesizer(n_transients=n, tr_s=float(tr_s), seed=synth_seed,
                                     **synthesis)
        # the realized tracks, which Augmentrum applies through the same factor
        freq_hz, phase_rad, amplitude, broaden_hz = synth.tracks(synth.rng.numpy_rng())

        grid = np.arange(npoints) * dwell_s
        data = np.zeros((1, 1, 1, npoints, n), dtype=np.complex64)
        for k, name in enumerate(schedule):
            modulation = amplitude[k] * np.exp(-np.pi * broaden_hz[k] * grid) * np.exp(
                1j * (2.0 * np.pi * freq_hz[k] * grid + phase_rad[k]))
            fid = clean[name] * modulation
            sigma = voxel.noise_sigma(truth)
            if sigma > 0.0:
                fid = fid + receiver_noise(rng, sigma, npoints)
            data[0, 0, 0, :, k] = fid

        voxel = self.scans[schedule[0]].voxel(position_mm, size_mm)
        hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
        hdr.set_dim_info(0, "DIM_DYN")
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "params": truth.to_dict(),
             "tissue_fractions": dict(voxel.fractions._asdict())},
            "MRSandbox: the one realized parameter tree shared by every transient.",
        )
        hdr.set_user_def(
            "SimulationSchedule",
            {"conditions": schedule, "tr_s": float(tr_s),
             "synthesis": {"module": "augmentrum.augmentation.TransientSynthesizer",
                           "version": augmentrum.__version__, "seed": synth_seed,
                           **{key: (list(value) if isinstance(value, tuple) else value)
                              for key, value in synthesis.items()}},
             "frequency_hz": [float(x) for x in freq_hz],
             "phase_deg": [float(np.degrees(x)) for x in phase_rad],
             "amplitude": [float(x) for x in amplitude],
             "broadening_hz": [float(x) for x in broaden_hz]},
            "MRSandbox: per transient condition and the realized Augmentrum instability "
            "tracks.",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.brain.name, "position_mm": [float(x) for x in position_mm],
             "size_mm": list(size_mm) if hasattr(size_mm, "__len__") else float(size_mm)},
            "MRSandbox: the virtual brain and the voxel placement.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr)
