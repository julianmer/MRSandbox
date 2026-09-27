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
from mrsandbox.scan.voxel import draw, nifti_mrs, receiver_noise


#**************************************************************************************************#
#                                         Class Multiplex                                          #
#**************************************************************************************************#
#                                                                                                  #
# Named acquisition conditions sharing one brain, scanner, parameter tree, and truth draw.         #
#                                                                                                  #
#**************************************************************************************************#
# the run's leaves of the parameter tree (scan.defaults) that are the synthesizer's arguments
RUN_KEYS = ("drift_hz_per_min", "ar_sigma_hz", "resp_freq_hz", "resp_amp_hz", "phase_jitter_deg", "amp_jitter_frac", "broaden_hz", "events_per_min")


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
        conditions = tuple(self.conditions if conditions is None else conditions)
        seed, truth, rng = draw(self.params, seed, step)
        stack = []
        for name in conditions:
            voxel = self.scans[name].voxel(position_mm, size_mm, compiled=self.compiled)
            _, fid = voxel._clean_fid(truth, npoints, dwell_s)
            stack.append(fid + self._noise(voxel, truth, rng, npoints, dwell_s, averages))
        data = np.stack(stack, axis=-1).reshape(1, 1, 1, npoints, len(conditions))
        hdr = self._header(conditions, position_mm, size_mm, seed, step, averages, truth)
        return nifti_mrs(data, dwell_s, hdr)

    def acquire_components(self, position_mm, size_mm, seed=None, npoints=2048,
                           dwell_s=1.0 / 4000.0, conditions=None, step=0, averages=1):
        """Acquire the conditions component by component and return NIfTI-MRS on
        DIM_EDIT then DIM_USER_0: the same subject, noise, header, and provenance as
        `acquire` with the same arguments, every condition split into the signal
        components named in the header's SimulationComponents (the union over the
        conditions in order of appearance, a component silent in a condition being
        zero there, and noise last), so the components sum to the acquisition
        exactly. The second file of a training pair (Voxel.acquire_components)."""
        conditions = tuple(self.conditions if conditions is None else conditions)
        seed, truth, rng = draw(self.params, seed, step)
        parts, names = [], []
        for name in conditions:
            voxel = self.scans[name].voxel(position_mm, size_mm, compiled=self.compiled)
            fids = voxel._component_fids(truth, npoints, dwell_s)
            names += [n for n in fids if n not in names]
            fids["noise"] = self._noise(voxel, truth, rng, npoints, dwell_s, averages)
            parts.append(fids)
        names.append("noise")
        zero = np.zeros(npoints, dtype=np.complex64)
        data = np.stack([np.stack([fids.get(n, zero) for n in names], axis=-1) for fids in parts],
                        axis=-2).reshape(1, 1, 1, npoints, len(conditions), len(names))
        hdr = self._header(conditions, position_mm, size_mm, seed, step, averages, truth)
        hdr.set_dim_info(1, "DIM_USER_0",
                         info="MRSandbox: the signal components named in SimulationComponents.")
        hdr.set_user_def("SimulationComponents", {"names": names},
                         "MRSandbox: the components on DIM_USER_0, in order; they sum to "
                         "the acquisition of the same seed and step.")
        return nifti_mrs(data, dwell_s, hdr)

    @staticmethod
    def _noise(voxel, truth, rng, npoints, dwell_s, averages):
        """One condition's receiver noise of `averages` transients, zeros when silent."""
        sigma = voxel.noise_sigma(truth, dwell_s, averages)
        return (receiver_noise(rng, sigma, npoints, voxel.scan.receiver) if sigma > 0.0
                else np.zeros(npoints, dtype=np.complex64))

    def _header(self, conditions, position_mm, size_mm, seed, step, averages, truth):
        """The header extension every acquisition of a condition stack carries."""
        from nifti_mrs.hdr_ext import Hdr_Ext

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
        return hdr

    #*************#
    #   combine   #
    #*************#
    def combine(self, acquisition, name):
        """One named combination of an acquired condition stack (e.g. the Hadamard GABA
        difference): the weighted sum over DIM_EDIT, returned as a complex array of
        the voxel's remaining dimensions (the FID, or the FID by the components of an
        `acquire_components` file)."""
        weights = self.combinations[name]
        w = np.asarray([weights[c] for c in self.conditions], dtype=np.float32)
        axis = 4 + acquisition.dim_tags.index("DIM_EDIT")
        return np.tensordot(np.asarray(acquisition[:])[0, 0, 0], w, axes=([axis - 3], [0]))

    #****************#
    #   transients   #
    #****************#
    def transients(self, position_mm, size_mm, schedule, tr_s=2.0, seed=None,
                   npoints=2048, dwell_s=1.0 / 4000.0, synthesis=None, displacement_mm=None,
                   motion_mm=None, rotation_deg=None, pivot_mm=(0.0, 0.0, 0.0), step=0,
                   frequency_start_hz=0.0):
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

        The spurious echo comes and goes over the run: the exam's echo (its timing,
        width, place and typical size are the geometry's, drawn once in the truth) is
        carried by the transients that spurious.rate picks and by every transient of a
        motion event (a head that moved spoils the crushing), each at its own size about
        the typical one (spurious.spread) and its own phase; the others are clean. The
        header records the size and phase per transient, so a processing step that
        rejects transients can be scored. A single acquisition (`acquire`) carries the
        exam's echo whole.

        Args:
            position_mm, size_mm: The voxel placement.
            schedule: Condition names per transient, or a transient count.
            tr_s: Repetition time, the acquisition clock of the tracks.
            seed: As for acquire; the synthesizer's own seed derives from it.
            npoints, dwell_s: The readout grid.
            synthesis: Keyword arguments for Augmentrum's TransientSynthesizer (its
                ranges, e.g. drift_hz_per_min=0.0 and events_per_min=0.0 for a still
                scanner and subject); the transient count, TR, and seed are set here.
            displacement_mm: Where the head is at every transient, (transients, 3) in
                mm from its planned place, or None for a head that stays. The voxel
                stays where the scanner put it, so a displaced transient is rendered
                at the planned position minus the displacement: its tissue fractions,
                water, fat, field, and transmit map are the moved head's, the truth is
                the one subject's, and the shim is still the one fitted for the planned
                place, which the move spoils (motion within a transient is not
                modelled). Recorded in the header.
            motion_mm: A (low, high) range in mm, or None: every motion event of the
                synthesizer (the ones that kick its frequency, phase, and amplitude
                tracks) also moves the head by a step of that size in a random
                direction, for the event's transients, and for good when the event is
                persistent. The range is the user's (stated: no measured distribution
                is typed); the steps add to `displacement_mm` and are drawn after
                everything else, so a run without them is unchanged.
            rotation_deg: How the head is turned at every transient, (transients, 3)
                angles about x, y, z in degrees (applied in that order) about
                `pivot_mm`, the phantom's origin unless given, before the displacement;
                None for a head that does not turn. The voxel's box stays aligned with
                the head's axes (small angles; stated).
            pivot_mm: The point the head turns about.
            frequency_start_hz: Where the scanner's frequency stands when the run starts, added to
                the whole frequency track: a scanner that drifted through earlier runs and was
                not centred again. Zero is a run that starts on resonance.
            step: Which noise stream of the seed this run takes (scan.voxel draw): one
                seed is one subject, and every further run of that subject takes another
                step for fresh noise and a fresh course of the scanner's instabilities.
        """
        try:
            import augmentrum
            from augmentrum.augmentation import TransientSynthesizer
        except ImportError as error:
            raise ImportError("transients need Augmentrum (the mrsandbox[augment] extra; "
                              "its fsl-mrs dependency is not on PyPI, see pyproject.toml "
                              "for the install recipe)") from error
        from nifti_mrs.hdr_ext import Hdr_Ext

        if isinstance(schedule, int):
            schedule = [self.conditions[0]] * schedule
        schedule = list(schedule)
        seed, truth, rng = draw(self.params, seed, step)

        n = len(schedule)
        # the run's leaves of the drawn truth are the synthesizer's arguments unless `synthesis` sets them
        drawn = {key: float(truth["run"][key]) for key in RUN_KEYS if "run" in truth and key in truth["run"]}
        synthesis = {**drawn, **dict(synthesis or {})}
        if motion_mm is None and "run" in truth and float(truth["run"].get("motion_mm", 0.0)) > 0.0:
            motion_mm = (float(truth["run"]["motion_mm"]), float(truth["run"]["motion_mm"]))
        synth_seed = int(rng.integers(2 ** 31))
        synth = TransientSynthesizer(n_transients=n, tr_s=float(tr_s), seed=synth_seed,
                                     **synthesis)
        # the realized tracks, which Augmentrum applies through the same factor
        freq_hz, phase_rad, amplitude, broaden_hz = synth.tracks(synth.rng.numpy_rng())
        freq_hz = freq_hz + float(frequency_start_hz)

        moved = (np.zeros((n, 3)) if displacement_mm is None
                 else np.asarray(displacement_mm, dtype=np.float64).reshape(n, 3).copy())
        if motion_mm is not None:
            if not hasattr(synth, "last_events_"):
                raise ImportError("motion_mm needs an Augmentrum whose TransientSynthesizer "
                                  "records its motion events (last_events_)")
            steps = np.random.default_rng([synth_seed, 1])
            for event in synth.last_events_:
                direction = steps.standard_normal(3)
                step = steps.uniform(*motion_mm) * direction / np.linalg.norm(direction)
                moved[event["start"]:n if event["persistent"] else event["stop"]] += step
        # where the voxel, fixed in the scanner, sits in the frame of the moved head:
        # the head went p -> R (p - pivot) + pivot + d, so the voxel reads the head at
        # R^T (p - pivot - d) + pivot
        planned = np.asarray(position_mm, dtype=np.float64)
        pivot = np.asarray(pivot_mm, dtype=np.float64)
        turned = (np.zeros((n, 3)) if rotation_deg is None
                  else np.asarray(rotation_deg, dtype=np.float64).reshape(n, 3))
        places = []
        for shift, angles in zip(moved, np.radians(turned)):
            rotation = np.eye(3)
            for axis, angle in enumerate(angles):
                c, s = np.cos(angle), np.sin(angle)
                i, j = [k for k in range(3) if k != axis]
                turn = np.eye(3)
                turn[i, i], turn[i, j], turn[j, i], turn[j, j] = c, -s, s, c
                rotation = turn @ rotation
            places.append(tuple(np.round(rotation.T @ (planned - pivot - shift) + pivot, 9)))
        clean, echo = {}, {}
        for name, place in set(zip(schedule, places)):
            still = np.allclose(place, planned)
            voxel = self.scans[name].voxel(place, size_mm, compiled=self.compiled,
                                           shim_at_mm=None if still else tuple(planned))
            parts = voxel._component_fids(truth, npoints, dwell_s)
            echo[name, place] = parts.pop("spurious", None)
            clean[name, place] = sum(parts.values())
        # which transients carry the exam's echo, at what size and phase (scan.defaults spurious)
        spur = truth.spurious
        ghosts = np.random.default_rng([synth_seed, 2])
        carries = ghosts.random(n) < spur.rate
        for event in getattr(synth, "last_events_", ()):
            carries[event["start"]:event["stop"]] = True
        size = np.where(carries, spur.amplitude * ghosts.uniform(1.0 - spur.spread, 1.0 + spur.spread, n), 0.0)
        turn = ghosts.uniform(0.0, 360.0, n)

        grid = np.arange(npoints) * dwell_s
        data = np.zeros((1, 1, 1, npoints, n), dtype=np.complex64)
        for k, name in enumerate(schedule):
            modulation = amplitude[k] * np.exp(-np.pi * broaden_hz[k] * grid) * np.exp(
                1j * (2.0 * np.pi * freq_hz[k] * grid + phase_rad[k]))
            fid = clean[name, places[k]]
            if size[k] > 0.0:      # the rendered echo, scaled to this transient's size and turned to its phase
                fid = fid + echo[name, places[k]] * (size[k] / spur.amplitude) * np.exp(
                    1j * np.radians(turn[k] - spur.phase_deg))
            fid = fid * modulation
            sigma = voxel.noise_sigma(truth, dwell_s)
            if sigma > 0.0:
                fid = fid + receiver_noise(rng, sigma, npoints, voxel.scan.receiver)
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
             "broadening_hz": [float(x) for x in broaden_hz],
             "displacement_mm": (None if displacement_mm is None and not np.any(moved)          # a head that never moved: none
                                 else moved.tolist()),
             "motion_mm": None if motion_mm is None else [float(x) for x in motion_mm],
             "rotation_deg": None if rotation_deg is None else turned.tolist(),
             "pivot_mm": [float(x) for x in pivot],
             "spurious": {"amplitude": [float(x) for x in size],
                          "phase_deg": [float(t) if c else None for t, c in zip(turn, carries)]}},
            "MRSandbox: per transient condition, the realized Augmentrum instability "
            "tracks, and the spurious echo's size and phase (0 and None where it is absent).",
        )
        hdr.set_user_def(
            "SimulationPhantom",
            {"brain": self.brain.name, "position_mm": [float(x) for x in position_mm],
             "size_mm": list(size_mm) if hasattr(size_mm, "__len__") else float(size_mm)},
            "MRSandbox: the virtual brain and the voxel placement.",
        )
        return nifti_mrs(data, dwell_s, hdr)
