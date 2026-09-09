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

from mrsandbox.scan.virtual_scan import VirtualScan
from mrsandbox.scan.voxel import acquisition_seed, receiver_noise


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
        **kwargs: Forwarded to every condition's VirtualScan (params, water, coils...).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequences, combinations=None, **kwargs):
        self.conditions = tuple(sequences)
        self.combinations = dict(combinations) if combinations else {}
        self.scans = {
            name: VirtualScan(brain, scanner, sequence, **kwargs)
            for name, sequence in sequences.items()
        }
        first = self.scans[self.conditions[0]]
        self.params = first.params
        self.brain = brain

    #*****************#
    #   acquisition   #
    #*****************#
    def acquire(self, position_mm, size_mm, seed=None, npoints=2048, dwell_s=1.0 / 4000.0):
        """Acquire every condition of one voxel and return NIfTI-MRS on DIM_EDIT.

        Returns data of shape (1, 1, 1, npoints, n_conditions); the header records the
        condition order, the shared realized truth, and the voxel placement.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        seed = acquisition_seed(seed)
        rng = np.random.default_rng(seed)
        truth = self.params.sample(rng)
        stack = []
        for name in self.conditions:
            voxel = self.scans[name].voxel(position_mm, size_mm)
            _, fid = voxel._clean_fid(truth, npoints, dwell_s)
            if truth.noise.sigma > 0.0:
                fid = fid + receiver_noise(rng, truth.noise.sigma, npoints)
            stack.append(fid)
        data = np.stack(stack, axis=-1).reshape(1, 1, 1, npoints, len(self.conditions))

        voxel = self.scans[self.conditions[0]].voxel(position_mm, size_mm)
        hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
        hdr.set_dim_info(0, "DIM_EDIT")
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "params": truth.to_dict(),
             "tissue_fractions": dict(voxel.fractions._asdict())},
            "MRSandbox: the one realized parameter tree shared by every condition.",
        )
        record = {"names": list(self.conditions),
                  "sequences": [type(self.scans[n].sequence).__name__ for n in self.conditions]}
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
                   npoints=2048, dwell_s=1.0 / 4000.0):
        """Acquire a schedule of transients and return NIfTI-MRS on DIM_DYN.

        The schedule names one condition per transient (a closed loop may interleave
        them arbitrarily); an integer repeats the first condition. One truth serves the
        whole run, while the scanner and the subject evolve across it: the drawn drift
        rate accumulates over the acquisition clock (transient k starts at k times TR),
        shot-to-shot frequency and phase jitter and amplitude instability perturb every
        transient, and motion events (a Poisson process with the drawn rate) add step
        changes in frequency and phase from their transient onward. Receiver noise is
        drawn per transient. Every realized instability is recorded in the header, so
        alignment algorithms can be scored against the truth.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        if isinstance(schedule, int):
            schedule = [self.conditions[0]] * schedule
        schedule = list(schedule)
        seed = acquisition_seed(seed)
        rng = np.random.default_rng(seed)
        truth = self.params.sample(rng)
        knobs = truth.transients

        clean = {}
        for name in set(schedule):
            voxel = self.scans[name].voxel(position_mm, size_mm)
            _, clean[name] = voxel._clean_fid(truth, npoints, dwell_s)

        n = len(schedule)
        times_s = np.arange(n) * float(tr_s)
        drift_hz = knobs.drift_hz_per_min * times_s / 60.0
        jitter_hz = knobs.jitter_hz * rng.standard_normal(n)
        phase_rad = np.radians(knobs.phase_jitter_deg) * rng.standard_normal(n)
        amplitude = 1.0 + knobs.amplitude_sd * rng.standard_normal(n)
        n_events = rng.poisson(knobs.motion_rate_per_min * times_s[-1] / 60.0) if n > 1 else 0
        events = []
        for _ in range(n_events):
            at = int(rng.integers(1, n))
            shift = float(knobs.motion_shift_hz) * float(rng.choice((-1.0, 1.0)))
            phase = float(rng.uniform(-abs(knobs.motion_phase_deg), abs(knobs.motion_phase_deg)))
            events.append({"transient": at, "shift_hz": shift, "phase_deg": phase})
            drift_hz[at:] += shift
            phase_rad[at:] += np.radians(phase)

        grid = np.arange(npoints) * dwell_s
        data = np.zeros((1, 1, 1, npoints, n), dtype=np.complex64)
        for k, name in enumerate(schedule):
            modulation = amplitude[k] * np.exp(
                1j * (2.0 * np.pi * (drift_hz[k] + jitter_hz[k]) * grid + phase_rad[k])
            )
            fid = clean[name] * modulation
            if truth.noise.sigma > 0.0:
                fid = fid + receiver_noise(rng, truth.noise.sigma, npoints)
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
             "drift_hz": [float(x) for x in drift_hz],
             "jitter_hz": [float(x) for x in jitter_hz],
             "phase_deg": [float(np.degrees(x)) for x in phase_rad],
             "amplitude": [float(x) for x in amplitude],
             "motion_events": events},
            "MRSandbox: per transient condition and realized instability tracks.",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.brain.name, "position_mm": [float(x) for x in position_mm],
             "size_mm": list(size_mm) if hasattr(size_mm, "__len__") else float(size_mm)},
            "MRSandbox: the virtual brain and the voxel placement.",
        )
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr)
