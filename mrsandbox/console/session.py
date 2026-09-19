####################################################################################################
#                                            session.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The state of one console session, free of any interface: which scanner and subject are  #
#          in the room, where the voxel lies, the protocol card, and the scans that ran. The       #
#          console draws it and asks it to act; all physics stays in the package.                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import json
import pathlib

import numpy as np

from mrsandbox.hardware import CoilArray
from mrsandbox.hardware.scanners import PRESETS
from mrsandbox.params import Fixed, Params, Uniform
from mrsandbox.phantom import Brain, Phantom
from mrsandbox.scan import Multiplex, ScannerProfile
from mrsandbox.sequences import (Chess, CoherenceFilter, Delay, Gradient, Hercules, Hermes, LocalizedMegaPress, MegaPress,
                                 Press, Pulse, SemiLaser, Sequence, ShapedPulse, Steam, Vapor, Wet)
from mrsandbox.sequences.timeline import event_spans, gradient_waveforms

#**************#
#   subjects   #
#**************#
# who can lie on the table: a name for the picker and how to make them
SUBJECTS = {
    "Brain, ICBM152 average": lambda: Brain.load("icbm152"),
    "Phantom, SPECTRE sphere": lambda: Phantom.spectre(temperature_c=21.0),
    "Phantom, Braino sphere": lambda: Phantom.braino(temperature_c=21.0),
}

#***************#
#   sequences   #
#***************#
# the protocol library of the walking skeleton: a name and how a card's timings make the design
SEQUENCES = {
    "PRESS": lambda te_s, tr_s: Press(te1_s=min(0.011, te_s / 2.0), te2_s=te_s - min(0.011, te_s / 2.0), tr_s=tr_s),
    "STEAM": lambda te_s, tr_s: Steam(te_s=te_s, tm_s=0.010, tr_s=tr_s),
    "sLASER": lambda te_s, tr_s: SemiLaser(te_s=te_s, tr_s=tr_s),
}


#*************#
#   editing   #
#*************#
# the edited acquisitions of the library: the sub-experiments a card interleaves over its transients (made
# from the repetition time, the spectrometer frequency, and the voxel's size), how they combine, and the
# combination the console shows while it scans. The ideal editing bands are fast and right for the edited
# target, not for co-editing at the bands' flanks; the played schemes (20 ms Gaussian editing pulses and
# the localization's profiles) reproduce raw in vivo HERMES and HERCULES without a fit and take minutes
# (sequences.library HadamardEditing).
DIFFERENCE = {"difference": {"on": 1.0, "off": -1.0}, "sum": {"on": 1.0, "off": 1.0}}
EDITING = {
    "MEGA-PRESS GABA": (lambda tr_s, f0_mhz, size_mm: MegaPress.pair(tr_s=tr_s), DIFFERENCE, "difference"),
    "HERMES GABA GSH": (lambda tr_s, f0_mhz, size_mm: Hermes.scheme(tr_s=tr_s), Hermes.combinations, "gaba"),
    "MEGA-PRESS GABA, played": (lambda tr_s, f0_mhz, size_mm: LocalizedMegaPress.pair(f0_mhz, size_mm=size_mm, tr_s=tr_s),
                                DIFFERENCE, "difference"),
    "HERMES, played": (lambda tr_s, f0_mhz, size_mm: LocalizedMegaPress.scheme(Hermes, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                       Hermes.combinations, "gaba"),
    "HERCULES, played": (lambda tr_s, f0_mhz, size_mm: LocalizedMegaPress.scheme(Hercules, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                         Hercules.combinations, "gaba"),
}

#***********#
#   gears   #
#***********#
# which branches of the parameter tree each panel's gear opens
GEARS = {
    "subject": ("conc", "t1", "t2", "contrast", "diffusion", "mm", "downfield", "lipids", "water"),
    "scanner": ("noise", "lineshape", "b1_scale", "gradients", "acquisition", "spurious"),
}

#**********#
#   mrsi   #
#**********#
# the spectroscopic imaging card: one slice of matrix x matrix cells of the voxel's size about the planned
# place, PRESS at the card's timings, ideal boxes (scan.mrsi); and the lines its maps are read at, ppm windows
MRSI = "MRSI PRESS"
MAPS = {"NAA": (1.9, 2.1), "Cr": (2.95, 3.1), "Cho": (3.13, 3.28), "lipid": (0.9, 1.5)}

#************#
#   blocks   #
#************#
# the building blocks of a custom design: what the designer offers, the settings each takes with
# their defaults, and how a block becomes events
BLOCKS = {
    "hard pulse": (dict(flip_deg=90.0, phase_deg=0.0), lambda b: (Pulse(b["flip_deg"], phase_deg=b["phase_deg"]),)),
    "shaped pulse": (dict(shape="sinc", flip_deg=90.0, duration_ms=2.6, offset_ppm=3.0, phase_deg=0.0,
                          slice_axis="none", slice_hz_per_mm=0.0),
                     lambda b: (getattr(ShapedPulse, b["shape"])(
                         b["flip_deg"], b["duration_ms"] / 1000.0, offset_ppm=b["offset_ppm"], phase_deg=b["phase_deg"],
                         **({} if b["slice_axis"] == "none" else dict(gradient_hz_per_mm=b["slice_hz_per_mm"],
                                                                      gradient_axis=b["slice_axis"]))),)),
    "delay": (dict(ms=10.0), lambda b: (Delay(b["ms"] / 1000.0),)),
    "crusher": (dict(axis="x", cycles_per_mm=0.15), lambda b: (Gradient(b["cycles_per_mm"], 20.0, axis=b["axis"]),)),
    "coherence filter": (dict(order=-1), lambda b: (CoherenceFilter(int(b["order"])),)),
    "water suppression": (dict(scheme="VAPOR"),
                          lambda b: {"VAPOR": Vapor, "WET": Wet, "CHESS": Chess}[b["scheme"]]().events),
}
# a spin echo with played crushers, the design the designer opens with
STARTER = [("hard pulse", dict(flip_deg=90.0)), ("delay", dict(ms=15.0)), ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)), ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("delay", dict(ms=15.0))]


#**************************************************************************************************#
#                                          Class Protocol                                          #
#**************************************************************************************************#
#                                                                                                  #
# One protocol card: the sequence and what an operator sets on it.                                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass
class Protocol:
    """One protocol card: the sequence and what an operator sets on it."""

    sequence: str = "PRESS"
    te_ms: float = 30.0
    tr_ms: float = 2000.0
    averages: int = 64
    npoints: int = 2048
    bandwidth_hz: float = 4000.0
    matrix: int = 8

    @property
    def scan_time_s(self):
        """One repetition per transient, or per phase encode of a spectroscopic image."""
        return (int(self.matrix) ** 2 if self.sequence == "MRSI PRESS" else self.averages) * self.tr_ms / 1000.0


#**************************************************************************************************#
#                                            Class Exam                                            #
#**************************************************************************************************#
#                                                                                                  #
# One entry of the exam queue: a card with its voxel, and after it ran, its data.                  #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass
class Exam:
    """One entry of the exam queue: a card with its voxel, and after it ran, its data.

    Args:
        name: What the queue shows.
        card: The Protocol as it was when the exam was planned.
        position_mm, size_mm: The voxel as it was planned.
        status: "planned", "running", or "done".
        data: NIfTI-MRS with the transients apart (DIM_DYN), once it ran.
        ppm: The display's axis, 0.2 to 4.2 ppm.
        frames: The display as the scan comes in, [(transients so far, real spectrum)].
        records: What the header says of the calibration records in force.
        sequence: The design as it was when the exam was planned (an edited card's first
            sub-experiment).
        conditions, combinations, shown: The sub-experiments {name: design} interleaved over
            the transients, how they combine {name: {condition: weight}}, and the
            combination the display shows; one condition "scan" for an unedited card.
        spectra: After the run, {name: real spectrum} of every condition and combination.
        tracks: What the scanner and the subject did over the transients, {name: values}.
        truth: The drawn concentrations in mM, {metabolite: value}.
        maps, cells: Of a spectroscopic imaging exam, {line: (nx, ny) heights} and the cells' real
            spectra (nx, ny, points), each phased on its own tallest line.
    """

    name: str
    card: Protocol
    position_mm: tuple
    size_mm: float
    status: str = "planned"
    data: object = None
    ppm: object = None
    frames: list = dataclasses.field(default_factory=list)
    records: dict = dataclasses.field(default_factory=dict)
    sequence: object = None
    conditions: dict = dataclasses.field(default_factory=dict)
    combinations: dict = dataclasses.field(default_factory=dict)
    shown: str = "scan"
    spectra: dict = dataclasses.field(default_factory=dict)
    tracks: dict = dataclasses.field(default_factory=dict)
    truth: dict = dataclasses.field(default_factory=dict)
    maps: dict = dataclasses.field(default_factory=dict)
    cells: object = None


#**************************************************************************************************#
#                                          Class Session                                           #
#**************************************************************************************************#
#                                                                                                  #
# What is in the scanner room and what was scanned, without any interface.                         #
#                                                                                                  #
#**************************************************************************************************#
class Session:
    """What is in the scanner room and what was scanned, without any interface.

    Attributes:
        scanner: Name of the scanner preset on the floor.
        coil: Name of the receive array (a key of COILS).
        subject: Name of the subject on the table (a key of SUBJECTS).
        position_mm, size_mm: The voxel being planned, world millimetres.
        protocol: The protocol card being edited.
        queue: The exams, in the order they were planned.
        overrides: The operator's own settings of leaves, {path: value or (low, high)}.
    """

    # the receive arrays on the shelf
    COILS = {"one ideal channel": lambda: None, "head array, 32 channels": CoilArray.helmet,
             "ring, 8 channels": lambda: CoilArray.ring(8)}

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        self.scanner = "siemens_prisma_3t"
        self.coil = "one ideal channel"
        self.subject = "Brain, ICBM152 average"
        self.position_mm = [0.0, -45.0, 35.0]
        self.size_mm = 20.0
        self.protocol = Protocol()
        self.queue = []
        self.overrides = {}
        self.design = [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in STARTER]
        self._bodies = {}

    #**********#
    #   body   #
    #**********#
    def body(self):
        """The subject on the table, made once per name."""
        if self.subject not in self._bodies:
            self._bodies[self.subject] = SUBJECTS[self.subject]()
        return self._bodies[self.subject]

    @staticmethod
    def scanners():
        """What can stand on the floor: the package's presets, and the operator's own scanners,
        the profiles saved as JSON (scan.profile) in ./data/profiles and ~/MRSandbox/profiles."""
        folders = (pathlib.Path.cwd() / "data" / "profiles", pathlib.Path.home() / "MRSandbox" / "profiles")
        return sorted(PRESETS) + sorted(str(path) for folder in folders for path in folder.glob("*.json"))

    def profile(self):
        """The scanner on the floor with its records: a preset, or a saved profile."""
        return ScannerProfile.load(self.scanner) if self.scanner.endswith(".json") else ScannerProfile.preset(self.scanner)

    #**************#
    #   sequence   #
    #**************#
    def sequence(self, card=None):
        """The design of a card: one of the library's at the card's timings, or the
        operator's own, the blocks of `design` in order ("Custom")."""
        card = card if card is not None else self.protocol
        if card.sequence in EDITING:
            return next(iter(self.editing(card).values()))
        if card.sequence == MRSI:
            return SEQUENCES["PRESS"](card.te_ms / 1000.0, card.tr_ms / 1000.0)
        if card.sequence == "Custom":
            events = [event for kind, settings in self.design for event in BLOCKS[kind][1](settings)]
            return Sequence(*events, tr_s=card.tr_ms / 1000.0)
        return SEQUENCES[card.sequence](card.te_ms / 1000.0, card.tr_ms / 1000.0)

    def editing(self, card):
        """The sub-experiments of an edited card, {name: design}, for this room's field and voxel."""
        return EDITING[card.sequence][0](card.tr_ms / 1000.0, 42.5774785 * self.profile().scanner.b0_t, float(self.size_mm))

    def diagram(self, sequence=None, readout_ms=12.0):
        """A design as a pulse sequence diagram, times in ms: {"rf": [(t, height)] a trace
        with hard pulses as spikes of their flip over 180 and shaped pulses as their
        envelopes, "marks": [(t, text)] for what takes no time and has no height (coherence
        filters), "x", "y", "z": [(t, mT/m)] the gradients as the scanner would play them
        (sequences.timeline), "adc": (start, end)}."""
        sequence = sequence if sequence is not None else self.sequence()
        spans = event_spans(sequence)
        end_ms = 1000.0 * spans[-1][1] if spans else 0.0
        rf, marks = [(0.0, 0.0)], []
        for event, (start, stop) in zip(sequence.events, spans):
            start, stop = 1000.0 * start, 1000.0 * stop
            if isinstance(event, ShapedPulse):
                envelope = np.abs(np.asarray(event.envelope, dtype=float))
                height = 1.0 if event.flip_deg is None else float(event.flip_deg) / 180.0
                times = np.linspace(start, stop, envelope.size)
                rf += [(start, 0.0)] + [(float(a), float(height * b / envelope.max())) for a, b in zip(times, envelope)] + [(stop, 0.0)]
            elif isinstance(event, CoherenceFilter):
                marks.append((start, f"p = {event.orders if hasattr(event, 'orders') else ''}".strip()))
            elif hasattr(event, "flip_deg") and not isinstance(event, (Delay, Gradient)):
                rf += [(start - 0.05, 0.0), (start, float(event.flip_deg) / 180.0), (start + 0.05, 0.0)]
        rf.append((end_ms + readout_ms, 0.0))
        until_s = (end_ms + readout_ms) / 1000.0
        times, waves = gradient_waveforms(sequence, self.profile().scanner, 5e-5, until_s)
        lanes = {axis: [(float(a), float(b)) for a, b in zip(1000.0 * times[::4], waves.get(axis, np.zeros(times.size))[::4])]
                 for axis in "xyz"}
        return {"rf": rf, "marks": marks, **lanes, "adc": (end_ms, end_ms + readout_ms)}

    #************#
    #   leaves   #
    #************#
    def params(self):
        """The operator's own settings as a Params tree over the defaults: {path: value or
        (low, high)} of `overrides`, nested."""
        tree = {}
        for path, value in self.overrides.items():
            node = tree
            *branch, leaf = path.split(".")
            for key in branch:
                node = node.setdefault(key, {})
            node[leaf] = value
        return Params(tree)

    def leaves(self):
        """Every leaf of the parameter tree a scan of this room would draw from, as
        [(path, kind, numbers, changed)]: kind "fixed" with (value,), "range" with (low,
        high), or "other" with the prior's own text (a distribution, a tissue record, a
        map), which the console shows and does not edit."""
        sequence = self.sequence()
        found = []

        def walk(node, path):
            for key, value in node.items():
                here = path + (key,)
                if isinstance(value, Params):
                    walk(value, here)
                elif isinstance(value, Fixed) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                    found.append((".".join(here), "fixed", (float(value.value),), ".".join(here) in self.overrides))
                elif isinstance(value, Uniform):
                    found.append((".".join(here), "range", (float(value.low), float(value.high)), ".".join(here) in self.overrides))
                else:
                    found.append((".".join(here), "other", repr(value)[:60], False))

        walk(self.profile().scan(self.body(), sequence, params=self.params()).params, ())
        return found

    #************#
    #   planes   #
    #************#
    def planes(self):
        """The three planning planes through the voxel's centre as {"tra", "sag", "cor"}:
        each (image, rectangle), an 8-bit array drawn with its first row at the top
        (anterior up in the transverse plane, superior up in the other two) and the
        voxel's outline (left, top, width, height) in the image's pixels."""
        body = self.body()
        tissue = 0.85 * body.gm + 1.0 * body.wm + 0.35 * body.csf
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        i, j, k = (int(round(c)) for c in (np.asarray(self.position_mm) - origin) / step)
        nx, ny, nz = tissue.shape
        i, j, k = min(max(i, 0), nx - 1), min(max(j, 0), ny - 1), min(max(k, 0), nz - 1)
        half = 0.5 * self.size_mm / step
        if self.protocol.sequence == MRSI:          # the slice's whole grid in the plane, one cell thick
            half = half * np.array([self.protocol.matrix, self.protocol.matrix, 1.0])
        side = max(nx, ny, nz)

        def framed(plane, left, top, width, height):
            """A plane shaded to 8 bits and centred in a square of the largest grid side, all three
            planes then drawing alike; the voxel's outline moves with it."""
            shade = np.uint8(255 * np.clip(plane / max(float(tissue.max()), 1e-6), 0.0, 1.0))
            rows, columns = (side - shade.shape[0]) // 2, (side - shade.shape[1]) // 2
            square = np.zeros((side, side), dtype=np.uint8)
            square[rows:rows + shade.shape[0], columns:columns + shade.shape[1]] = shade
            return square, (left + columns, top + rows, width, height)

        return {
            "tra": framed(tissue[:, ::-1, k].T, i - half[0], ny - 1 - j - half[1], 2 * half[0], 2 * half[1]),
            "sag": framed(tissue[i, :, ::-1].T, j - half[1], nz - 1 - k - half[2], 2 * half[1], 2 * half[2]),
            "cor": framed(tissue[:, j, ::-1].T, i - half[0], nz - 1 - k - half[2], 2 * half[0], 2 * half[2]),
        }

    def place(self, plane, column, row):
        """Move the voxel to the point clicked in a plane, (column, row) in image pixels;
        the coordinate the plane does not show stays."""
        body = self.body()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = body.gm.shape
        side = max(nx, ny, nz)
        # the planes are centred in squares of the largest grid side (`planes`)
        px, py, pz = ((side - n) // 2 for n in (nx, ny, nz))
        index = {"tra": (column - px, ny - 1 - (row - py), None), "sag": (None, column - py, nz - 1 - (row - pz)),
                 "cor": (column - px, None, nz - 1 - (row - pz))}[plane]
        for axis, value in enumerate(index):
            if value is not None:
                self.position_mm[axis] = float(origin[axis] + step[axis] * value)

    #***********#
    #   scene   #
    #***********#
    def scene(self, points=5000):
        """What the room's 3D view draws, world millimetres: the subject's surface as a
        cloud of `points`, the receive loops as (centre, normal, radius), the voxel's box
        as (centre, size)."""
        body = self.body()
        inside = (body.gm + body.wm) > 0.5
        surface = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1)
                             & np.roll(inside, -1, 1) & np.roll(inside, 1, 2) & np.roll(inside, -1, 2))
        index = np.argwhere(surface)
        index = index[np.random.default_rng(0).choice(len(index), size=min(points, len(index)), replace=False)]
        cloud = index * np.diag(body.affine)[:3] + body.affine[:3, 3]
        coils = self.COILS[self.coil]()
        loops = [] if coils is None else [(tuple(loop.center_mm), tuple(loop.normal), float(loop.radius_mm)) for loop in coils.loops]
        return {"cloud": cloud.astype(np.float32), "loops": loops, "voxel": (tuple(self.position_mm), float(self.size_mm))}

    #*************#
    #   outlook   #
    #*************#
    def outlook(self):
        """What the card promises before it runs: tissue in the voxel, the scan time, the
        scanner's limits and why the design would not play on them (empty when it does),
        and the calibration records a scan would run under with where it departs."""
        card = self.protocol
        sequence = self.sequence()
        profile = self.profile()
        scan = profile.scan(self.body(), sequence, params=self.params())
        fractions = scan.voxel(tuple(self.position_mm), self.size_mm).fractions
        limits = profile.scanner
        return {"gm": float(fractions.gm), "wm": float(fractions.wm), "csf": float(fractions.csf),
                "scan_time_s": card.scan_time_s,
                "limits": (limits.peak_b1_ut, limits.max_grad_mt_m, limits.max_slew_t_m_s),
                "violations": tuple(sequence.violations(limits)),
                "records": {leaf: record.record(scan) for leaf, record in profile.calibrations().items()}}

    #***********#
    #   queue   #
    #***********#
    def plan(self):
        """Put the card with the voxel as they stand into the queue; returns the exam."""
        card = dataclasses.replace(self.protocol, averages=int(self.protocol.averages), npoints=int(self.protocol.npoints))
        timed = card.sequence in SEQUENCES or card.sequence == MRSI   # an edited or an own design has its own timings
        name = f"{len(self.queue) + 1} {card.sequence}" + (f" TE {card.te_ms:g}" if timed else "")
        exam = Exam(name=name, card=card, position_mm=tuple(self.position_mm), size_mm=float(self.size_mm),
                    records=self.outlook()["records"], sequence=self.sequence(card))
        if card.sequence in EDITING:
            _, exam.combinations, exam.shown = EDITING[card.sequence]
            exam.conditions = self.editing(card)
        else:
            exam.conditions, exam.combinations = {"scan": exam.sequence}, {"scan": {"scan": 1.0}}
        self.queue.append(exam)
        return exam

    def run(self, exam, seed=None):
        """Acquire an exam: every transient apart, with the instabilities of a real run
        (scan.Multiplex.transients: drift, breathing, motion), and the display as it
        builds up over the transients."""
        exam.status = "running"
        card, sequence = exam.card, exam.sequence
        profile = self.profile()
        coils = self.COILS[self.coil]()
        if card.sequence == MRSI:
            grid = profile.scan(self.body(), sequence, params=self.params()).mrsi(
                center_mm=exam.position_mm, matrix=(int(card.matrix), int(card.matrix)), voxel_mm=exam.size_mm)
            exam.data = grid.acquire(seed=seed, npoints=card.npoints, dwell_s=1.0 / card.bandwidth_hz)
            fids = np.asarray(exam.data[:])[:, :, 0]
            window = np.exp(-np.pi * 3.0 * np.arange(card.npoints) / card.bandwidth_hz)
            spectra = np.fft.fftshift(np.fft.fft(fids * window, 2 * card.npoints, axis=-1), axes=-1)
            f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
            ppm = np.fft.fftshift(np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz
            upfield = (ppm > 0.2) & (ppm < 4.2)
            exam.ppm, spectra = ppm[upfield], spectra[..., upfield]
            exam.maps = {name: np.abs(spectra[..., (exam.ppm > low) & (exam.ppm < high)]).max(axis=-1)
                         for name, (low, high) in MAPS.items()}
            top = np.take_along_axis(spectra, np.abs(spectra).argmax(axis=-1)[..., None], axis=-1)
            exam.cells = (spectra * np.exp(-1j * np.angle(top))).real
            centre = int(card.matrix) // 2
            exam.frames = [(1, exam.cells[centre, centre])]
            exam.spectra = {"centre cell": exam.cells[centre, centre]}
            exam.truth = {name: float(value) for name, value in exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]["conc"].items()
                          if isinstance(value, (int, float))}
            exam.status = "done"
            return exam
        room = Multiplex(self.body(), profile.scanner, exam.conditions, records=profile.records, params=self.params(),
                         coils=None if coils is None else coils.loaded().calibrated(exam.position_mm, exam.size_mm))
        names = list(exam.conditions)
        schedule = [names[k % len(names)] for k in range(card.averages)]        # interleaved, as an edited scan runs
        exam.data = room.transients(exam.position_mm, exam.size_mm, schedule, tr_s=card.tr_ms / 1000.0, seed=seed,
                                    npoints=card.npoints, dwell_s=1.0 / card.bandwidth_hz)
        fids = np.asarray(exam.data[:])[0, 0, 0]
        fids = fids if fids.ndim == 2 else fids.sum(axis=1)             # channels summed for the display alone
        # the display a console gives: light apodization, zero filled, phased on the tallest line upfield of water
        window = np.exp(-np.pi * 2.0 * np.arange(card.npoints) / card.bandwidth_hz)
        f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
        ppm = np.fft.fftshift(np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz
        upfield = (ppm > 0.2) & (ppm < 4.2)
        played = np.asarray(schedule)

        def combined(weights, count):
            """The spectrum of a combination over the first `count` transients: every condition's
            mean so far, weighted; None while a condition has not played yet."""
            total = 0.0
            for condition, weight in weights.items():
                mine = np.flatnonzero(played[:count] == condition)
                if mine.size == 0:
                    return None
                total = total + weight * fids[:, mine].mean(axis=1)
            return np.fft.fftshift(np.fft.fft(total * window, 2 * card.npoints))

        everything = combined({name: 1.0 for name in names}, card.averages)
        phase = np.exp(-1j * np.angle(everything[np.flatnonzero(upfield)[np.abs(everything[upfield]).argmax()]]))
        exam.ppm, exam.frames = ppm[upfield], []
        for count in sorted({max(len(names), round(card.averages * step / 12)) for step in range(1, 13)}):
            partial = combined(exam.combinations[exam.shown], count)
            if partial is not None:
                exam.frames.append((count, (partial * phase).real[upfield]))
        exam.spectra = {name: (combined(weights, card.averages) * phase).real[upfield]
                        for name, weights in {**{n: {n: 1.0} for n in names if len(names) > 1}, **exam.combinations}.items()}
        header = exam.data.hdr_ext.to_dict()
        exam.tracks = {key: header["SimulationSchedule"][key] for key in ("frequency_hz", "phase_deg", "amplitude")
                       if key in header.get("SimulationSchedule", {})}
        exam.truth = {name: float(value) for name, value in header["SimulationGroundTruth"]["params"]["conc"].items()
                      if isinstance(value, (int, float))}
        exam.status = "done"
        return exam

    #*************#
    #   library   #
    #*************#
    LIBRARY = pathlib.Path.home() / "MRSandbox" / "protocols"

    def protocols(self):
        """The names of the protocols the operator saved."""
        return sorted(path.stem for path in self.LIBRARY.glob("*.json"))

    def save_protocol(self, name):
        """Keep the card as it stands, with the voxel's size, the operator's own design, and
        the advanced settings, as one JSON file of the protocol library; returns the path."""
        self.LIBRARY.mkdir(parents=True, exist_ok=True)
        path = self.LIBRARY / f"{name}.json"
        path.write_text(json.dumps({"card": dataclasses.asdict(self.protocol), "size_mm": self.size_mm,
                                    "design": self.design, "overrides": self.overrides}, indent=2))
        return path

    def load_protocol(self, name):
        """Put a saved protocol on the card."""
        stored = json.loads((self.LIBRARY / f"{name}.json").read_text())
        for attribute, value in stored["card"].items():       # in place: the console's fields hold this card
            setattr(self.protocol, attribute, value)
        self.size_mm = float(stored["size_mm"])
        self.design = [(kind, dict(settings)) for kind, settings in stored["design"]]
        self.overrides = {path: tuple(value) if isinstance(value, list) else value for path, value in stored["overrides"].items()}

    #************#
    #   export   #
    #************#
    def export_seq(self, exam, folder):
        """Write an exam's design as a Pulseq file (io.pulseq.write_seq); a design with an
        idealization Pulseq cannot play (a coherence filter, an ideal selective pulse)
        says so."""
        from mrsandbox.io import write_seq

        folder = pathlib.Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (exam.name.replace(" ", "_").replace(".", "p") + ".seq")
        f0_mhz = 42.5774785 * self.profile().scanner.b0_t
        write_seq(exam.sequence, str(path), exam.card.npoints, 1.0 / exam.card.bandwidth_hz, f0_mhz)
        return path

    def export(self, exam, folder):
        """Write an exam's raw data, the transients apart, as NIfTI-MRS into `folder`;
        returns the path."""
        folder = pathlib.Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (exam.name.replace(" ", "_").replace(".", "p") + ".nii.gz")
        exam.data.save(path)
        return path
