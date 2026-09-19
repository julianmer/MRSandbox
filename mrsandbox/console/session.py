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
import copy
import dataclasses
import json
import pathlib

import numpy as np

from mrsandbox.hardware import CoilArray
from mrsandbox.hardware.coils import MappedCoilArray
from mrsandbox.hardware.scanners import PRESETS
from mrsandbox.params import Fixed, Mapped, Normal, Params, Uniform
from mrsandbox.phantom import Brain, FieldMap, Phantom, Volume
from mrsandbox.phantom.water import Unsuppressed
from mrsandbox.scan import Multiplex, ScannerProfile
from mrsandbox.sequences import (Chess, CoherenceFilter, Delay, Gradient, Hercules, Hermes, LocalizedMegaPress, LocalizedPress,
                                 LocalizedSemiLaser, LocalizedSteam, MegaPress,
                                 OuterVolumeSuppression, Press, SelectivePulse, Pulse, SemiLaser, Sequence, ShapedPulse, Steam, Vapor, Wet)
from mrsandbox.sequences.events import trapezoid
from mrsandbox.sequences.timeline import event_spans, gradient_waveforms

#**************#
#   subjects   #
#**************#
# who can lie on the table: a name for the picker and how to make them
SUBJECTS = {
    "Brain, ICBM152 average, 2 mm": lambda: Brain.load("icbm152"),
    "Brain, BigBrain-MR, 0.8 mm": lambda: Brain.load("bigbrain_mr"),
    "Head, QSM challenge, 0.64 mm": lambda: Brain.load("qsm_rc2"),
    "Phantom, SPECTRE sphere": lambda: Phantom.spectre(temperature_c=21.0),
    "Phantom, Braino sphere": lambda: Phantom.braino(temperature_c=21.0),
}
# what the planes show of a subject that has a relaxation map on its grid: R1 reads like a T1-weighted
# image; the others show their tissue maps
PICTURES = {
    "Brain, BigBrain-MR, 0.8 mm": lambda: (lambda t1: np.where(t1 > 0.0, 1.0 / np.maximum(t1, 0.5), 0.0))(
        Volume.load("bigbrain_mr/t1map_800um").data),
    "Head, QSM challenge, 0.64 mm": lambda: Volume.load("qsm_rc2/r1").data,          # skull and scalp with it
}

#***************#
#   sequences   #
#***************#
# the single-voxel cards: a name and how a card's timings, the spectrometer frequency, and the voxel's size make
# the design. The ideal ones select the voxel exactly and at once; the played ones play their slice-selective
# pulses under gradients, so the voxel has a profile and every shift its own displaced voxel (SHIFTED).
SEQUENCES = {
    "PRESS": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: Press(te1_s=min(0.011, te_s / 2.0), te2_s=te_s - min(0.011, te_s / 2.0), tr_s=tr_s),
    "STEAM": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: Steam(te_s=te_s, tm_s=0.010, tr_s=tr_s),
    "sLASER": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: SemiLaser(te_s=te_s, tr_s=tr_s),
    "PRESS, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: LocalizedPress(
        min(0.014 * fit.grow[180], te_s / 2.0), te_s - min(0.014 * fit.grow[180], te_s / 2.0), f0_mhz, size_mm=size_mm, tr_s=tr_s, axes=axes,
        excitation_s=0.0026 * fit.grow[90], refocusing_s=0.0065 * fit.grow[180]),
    "STEAM, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: fit(LocalizedSteam)(te_s, 0.010, f0_mhz, size_mm=size_mm, tr_s=tr_s),
    "sLASER, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: fit(LocalizedSemiLaser)(te_s, f0_mhz, size_mm=size_mm, tr_s=tr_s),
}
# a card's design with its pulses real (Protocol.real): shaped pulses under gradients, played in time, in place
# of ideal ones that act at once and select exactly. HERCULES has no ideal design and is always real.
REAL = {"PRESS": "PRESS, played", "STEAM": "STEAM, played", "sLASER": "sLASER, played",
        "MEGA-PRESS": "MEGA-PRESS, played", "HERMES GABA GSH": "HERMES, played"}
# the scanners by name: what a preset stands for (hardware.scanners, scan.calibrations)
SCANNERS = {"philips_mr7700_3t": "Philips MR 7700 3 T", "generic_1p5t": "Generic 1.5 T", "generic_3t": "Generic 3 T", "generic_7t": "Generic 7 T",
            "siemens_prisma_3t": "Siemens Prisma 3 T", "siemens_trio_3t": "Siemens Trio 3 T", "ge_mr750_3t": "GE MR750 3 T",
            "philips_elition_3t": "Philips Elition 3 T", "philips_achieva_3t": "Philips Achieva 3 T"}
# a scanner's other calibrations in the package, {scanner: {what it is: the preset that carries its records}}: the
# hardware is the same, the records were read under another coil
CALIBRATIONS = {"ge_mr750_3t": {"noise read with the 32 channel head array": "ge_mr750_32ch_3t"}}
# a run's instabilities as the operator sets them (Augmentrum's TransientSynthesizer, scan.Multiplex.transients):
# a name, the synthesizer's argument, and what switches it off; left alone, each is drawn from the 3 T envelope
RUN = {"frequency drift (Hz per minute)": ("drift_hz_per_min", 0.0), "slow frequency wander (Hz)": ("ar_sigma_hz", 0.0),
       "breathing (Hz)": ("resp_amp_hz", 0.0), "phase jitter (degrees)": ("phase_jitter_deg", 0.0),
       "amplitude jitter (fraction)": ("amp_jitter_frac", 0.0), "broadening over the run (Hz)": ("broaden_hz", 0.0),
       "motion events (per minute)": ("events_per_min", 0.0)}
# the measured B0 field a subject lies in unless the operator chooses another (phantom.FieldMap.PRESETS); a
# sphere of solution has none
FIELDS = {"Brain, ICBM152 average, 2 mm": "bigbrain_mr", "Brain, BigBrain-MR, 0.8 mm": "bigbrain_mr",
          "Head, QSM challenge, 0.64 mm": "qsm_rc2"}
# what the realism pop-up can switch off, each the leaves it sets to nothing: the console starts real
IDEALS = {
    "noise": {"noise.sigma": 0.0},
    "macromolecules": {"mm.level_mm": 0.0},
    "lipids": {"lipids.amplitude_mm": 0.0},
    "residual water": {"water.residual_fraction": 0.0},
    "frequency and phase errors": {"acquisition.shift_hz": 0.0, "acquisition.phase_deg": 0.0},
    "eddy currents": {"acquisition.eddy_std_rad": 0.0},
    "spurious echoes": {"spurious.amplitude": 0.0},
}
# what the planes can wash over the anatomy: the room's maps and the subject's tissues
OVERLAYS = ("anatomy", "B0 field", "B1+", "grey matter", "white matter", "CSF", "lipid layer")
LENT = "the QSM head's, carried over"      # a lipid layer for the brains that have no scalp
# the shifts whose displaced voxels the planes draw beside the planned one
SHIFTED = {"water": 4.65, "lipid": 1.3}

# what a timed card can play before its localization: the water suppression trains, and outer-volume
# suppression as six slabs of OVS_MM about the voxel's faces. The slabs play between the train and the
# excitation, where they saturate what the excitation would see; the train's last delay grows by them.
SUPPRESSION = {"VAPOR": Vapor, "WET": Wet, "CHESS": Chess}
# how the water comes about. The water is a spin system of its own, run through the card's sequence at its full amplitude
# (phantom.water Unsuppressed), so "off" is the whole water line and a train suppresses it as far as its physics does, by this
# scanner's B1 and the water's T1. ASSUMED is the package's shortcut: no train is played and the water is scaled to a drawn
# residual fraction (water.residual_fraction, 0.01 to 0.3 percent), fast, and the same whatever the card plays.
ASSUMED = "assumed residual"
# the branches of the parameter tree that are the scanner's state of one exam, drawn anew for every exam (Session.run)
STATE = ("acquisition", "spurious", "b1_scale")
OVS_MM = 30.0


#*************#
#   editing   #
#*************#
# the edited acquisitions of the library: the sub-experiments a card interleaves over its transients (made
# from the repetition time, the spectrometer frequency, and the voxel's size), how they combine, and the
# combination the console shows while it scans. The ideal editing bands are fast and right for the edited
# target, not for co-editing at the bands' flanks; the played schemes (20 ms Gaussian editing pulses and
# the localization's profiles) reproduce raw in vivo HERMES and HERCULES without a fit and take minutes
# (sequences.library HadamardEditing).
# where a MEGA editing pulse goes for a target, (ON, OFF) in ppm at the canonical TE of 68 ms: the card takes any
# pair, these fill it (GABA: Mescher 1998; GSH: Terpstra 2003; lactate: Star-Lack 1998)
TARGETS = {"GABA": (1.88, 7.5), "GSH": (4.56, 7.5), "lactate": (4.1, 7.5)}
DIFFERENCE = {"difference": {"on": 1.0, "off": -1.0}, "sum": {"on": 1.0, "off": 1.0}}
EDITING = {
    "MEGA-PRESS": (lambda card, tr_s, f0_mhz, size_mm, fit: MegaPress.pair(on_ppm=card.edit_on_ppm, off_ppm=card.edit_off_ppm, tr_s=tr_s),
                   DIFFERENCE, "difference"),
    "HERMES GABA GSH": (lambda card, tr_s, f0_mhz, size_mm, fit: Hermes.scheme(tr_s=tr_s), Hermes.combinations, "gaba"),
    "MEGA-PRESS, played": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).pair(
        f0_mhz, on_ppm=card.edit_on_ppm, off_ppm=card.edit_off_ppm, size_mm=size_mm, tr_s=tr_s), DIFFERENCE, "difference"),
    "HERMES, played": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).scheme(Hermes, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                       Hermes.combinations, "gaba"),
    "HERCULES": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).scheme(Hercules, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                         Hercules.combinations, "gaba"),
}

#***********#
#   gears   #
#***********#
# which branches of the parameter tree each panel's gear opens
GEARS = {
    "subject": ("conc", "t1", "t2", "contrast", "diffusion", "mm", "downfield", "lipids", "water"),
    "scanner": ("noise", "lineshape", "b1_scale", "gradients", "acquisition", "spurious"),
    # and the card's tabs: what of the simulation belongs to each
    "Geometry": ("lineshape", "b1_scale"),
    "Contrast": ("t1", "t2", "mm", "lipids", "water", "downfield"),
    "Acquisition": ("noise", "acquisition", "spurious", "gradients"),
    "Imaging": ("lipids", "lineshape"),
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
    "own pulse": (dict(flip_deg=90.0, duration_ms=3.0, envelope="0, 0.3, 0.8, 1, 0.8, 0.3, 0", offset_ppm=3.0, phase_deg=0.0),
                  lambda b: (ShapedPulse(b["flip_deg"], b["duration_ms"] / 1000.0,
                                         np.array([float(a) for a in b["envelope"].replace(";", ",").split(",") if a.strip()]),
                                         offset_ppm=b["offset_ppm"], phase_deg=b["phase_deg"]),)),
    "shaped pulse": (dict(shape="sinc", flip_deg=90.0, duration_ms=2.6, offset_ppm=3.0, phase_deg=0.0,
                          slice_axis="none", slice_hz_per_mm=0.0),
                     lambda b: (getattr(ShapedPulse, b["shape"])(
                         b["flip_deg"], b["duration_ms"] / 1000.0, offset_ppm=b["offset_ppm"], phase_deg=b["phase_deg"],
                         **({} if b["slice_axis"] == "none" else dict(gradient_hz_per_mm=b["slice_hz_per_mm"],
                                                                      gradient_axis=b["slice_axis"]))),)),
    "adiabatic pulse": (dict(duration_ms=6.0, bandwidth_hz=1500.0, peak_hz=1000.0, offset_ppm=4.65),
                        lambda b: (ShapedPulse.hyperbolic_secant(b["duration_ms"] / 1000.0, b["bandwidth_hz"], b["peak_hz"],
                                                                 offset_ppm=b["offset_ppm"]),)),
    # an event of the card's own sequence the designer took over as it is (Session.adopt): moved, copied, removed, not reshaped
    "as the card plays it": (dict(what="", event=None), lambda b: (b["event"],)),
    "delay": (dict(ms=10.0), lambda b: (Delay(b["ms"] / 1000.0),)),
    "crusher": (dict(axis="x", cycles_per_mm=0.15), lambda b: (Gradient(b["cycles_per_mm"], 20.0, axis=b["axis"]),)),
    "coherence filter": (dict(order=-1), lambda b: (CoherenceFilter(int(b["order"])),)),
    "water suppression": (dict(scheme="VAPOR"), lambda b: SUPPRESSION[b["scheme"]]().events),
}
# what a design can start from, in the designer's own blocks: a spin echo with played crushers (the design
# the designer opens with), and the library's ideal PRESS and STEAM, their coherence pathways kept by filters
STARTER = [("hard pulse", dict(flip_deg=90.0)), ("delay", dict(ms=15.0)), ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)), ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("delay", dict(ms=15.0))]
TEMPLATES = {
    "Spin echo, played crushers, TE 30": STARTER,
    "PRESS, TE 30": [("hard pulse", dict(flip_deg=90.0)), ("coherence filter", dict(order=-1)), ("delay", dict(ms=5.5)),
                     ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)), ("coherence filter", dict(order=1)), ("delay", dict(ms=15.0)),
                     ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)), ("coherence filter", dict(order=-1)), ("delay", dict(ms=9.5))],
    "sLASER, TE 30": [("hard pulse", dict(flip_deg=90.0)), ("coherence filter", dict(order=-1)), ("delay", dict(ms=0.75)),
                      ("adiabatic pulse", {}), ("coherence filter", dict(order=1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}), ("coherence filter", dict(order=-1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}), ("coherence filter", dict(order=1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}), ("coherence filter", dict(order=-1)), ("delay", dict(ms=0.75))],
    "STEAM, TE 20, TM 10": [("hard pulse", dict(flip_deg=90.0)), ("coherence filter", dict(order=1)), ("delay", dict(ms=10.0)),
                            ("hard pulse", dict(flip_deg=90.0)), ("coherence filter", dict(order=0)), ("delay", dict(ms=10.0)),
                            ("hard pulse", dict(flip_deg=90.0, phase_deg=180.0)), ("coherence filter", dict(order=-1)), ("delay", dict(ms=10.0))],
}


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
    matrix_y: int = 8
    slices: int = 1
    water_suppression: str = "VAPOR"
    ovs: bool = False
    real: bool = True
    water_reference: int = 1
    water_reference_first: bool = True
    axes: str = "xyz"
    edit_on_ppm: float = 1.88
    edit_off_ppm: float = 7.5
    # a spectroscopic image's encoding (scan.mrsi): how finely the subject is rendered under each cell (1: ideal boxes,
    # no bleed between cells), the k-space weighting and coverage, the slab's profile, and the readout
    mrsi_oversampling: int = 2
    mrsi_weighting: str = "none"
    mrsi_sampling: str = "full"
    mrsi_profile: str = "gaussian"
    mrsi_readout: str = "phase encoded"

    @property
    def scan_time_s(self):
        """One repetition per transient, or per encode of a spectroscopic image: every point of
        k-space that is covered when phase encoded (the corners left out by elliptical coverage),
        one line per repetition with an EPSI readout."""
        if self.sequence != "MRSI PRESS":
            return self.averages * self.tr_ms / 1000.0
        nx, ny, nz = int(self.matrix), int(self.matrix_y), int(self.slices)
        kx, ky = (np.arange(n) - (n - 1) / 2.0 for n in (nx, ny))
        covered = ((kx[:, None] / (nx / 2.0)) ** 2 + (ky[None, :] / (ny / 2.0)) ** 2 <= 1.0).sum() if self.mrsi_sampling == "elliptical" else nx * ny
        return (ny if self.mrsi_readout != "phase encoded" else int(covered)) * nz * self.tr_ms / 1000.0


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
        status: "planned", "running", "done", or "failed" (with `problem`, the reason).
        data: NIfTI-MRS with the transients apart (DIM_DYN), once it ran.
        ppm: The display's axis, 0.2 to 4.2 ppm.
        frames: The display as the scan comes in, [(transients so far, the real spectrum of their
            average, the last transient's own)], on `ppm` (-0.5 to 6.5); `window` is what the display
            opens on. `builds` holds the same build-up for every condition and combination.
        snr: The signal to noise ratio of what is shown, (of the average, of one transient alone), the
            tallest real point upfield over the noise of a signal-free stretch, no apodization.
        records: What the header says of the calibration records in force.
        sequence: The design as it was when the exam was planned (an edited card's first
            sub-experiment).
        conditions, combinations, shown: The sub-experiments {name: design} interleaved over
            the transients, how they combine {name: {condition: weight}}, and the
            combination the display shows; one condition "scan" for an unedited card.
        spectra: After the run, {name: real spectrum} of every condition and combination.
        tracks: What the scanner and the subject did over the transients, {name: values}.
        truth: The drawn concentrations in mM, {metabolite: value}.
        overrides, run: The operator's settings and the run's instabilities as they stood when the
            exam was planned; a water reference plays its card without water suppression.
        maps, cells: Of a spectroscopic imaging exam, {line: (nx, ny, nz) heights} and the cells' real
            spectra (nx, ny, nz, points), each phased on its own tallest line.
    """

    name: str
    card: Protocol
    position_mm: tuple
    size_mm: tuple
    status: str = "planned"
    problem: str = ""
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
    window: tuple = (0.2, 4.2)
    snr: tuple = ()
    builds: dict = dataclasses.field(default_factory=dict)
    overrides: dict = dataclasses.field(default_factory=dict)
    run: dict = dataclasses.field(default_factory=dict)


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
        position_mm, size_mm: The voxel being planned, its centre and its (x, y, z) edges, world mm.
        calibration: None for the scanner's own records, or the preset whose records stand in
            (CALIBRATIONS).
        seed: The subject on the table: every exam scans this one draw of the population, each
            with noise of its own; a new subject is a new seed.
        prescan, clock_s: The scanner's session. Its frequency drifts at one rate through all exams
            (`drift_hz_per_min`, drawn once unless the operator set it in the run's settings), and
            the clock runs on by every exam's duration. With `prescan` the frequency is centred
            before every exam, as scanners do by default; without it an exam starts where the
            drift has taken the scanner by then.
        align: Whether the display aligns every transient in frequency and phase to the run's mean
            before it averages, as a scanner's or a toolbox's preprocessing does; the export stays raw.
        run, motion_mm: A run's instabilities the operator set, {synthesizer argument: value}
            (RUN), and the step in mm a motion event moves the head by, (low, high) or None.
        view_mm: The point the three planes cut through, which the operator scrolls
            without moving the voxel.
        protocol: The protocol card being edited.
        queue: The exams, in the order they were planned.
        designs, condition, weights: The operator's own design. One sub-acquisition ("A") is a plain
            sequence. Several, each its own blocks, are interleaved over the transients as an
            edited scan is (scan.Multiplex), and `weights` says how they combine, {name of the
            combination: {sub-acquisition: weight}}: a difference, a Hadamard scheme, anything.
            `condition` is the one the designer shows, and `design` is its blocks.
        overrides: The operator's own settings of leaves, {path: value, (low, high), or the
            path of a NIfTI map in the leaf's units, which every voxel reads where it lies}.
        subjects: Who can lie on the table, SUBJECTS and the operator's own anatomies.
        field: The B0 field of the room: "uniform", a measured preset (phantom.FieldMap),
            or the path of the operator's own NIfTI map of offsets in Hz.
        transmit: The path of a NIfTI map of the relative B1+, or None for the nominal field.
        shim, shim_mm: The shim's order (phantom.FieldMap.SHIMS) and the edge of the shim box about
            the voxel's centre; None shims over the voxel itself.
    """

    # the receive arrays on the shelf: how each is made from its settings, and the settings it starts with (mm)
    COILS = {"one ideal channel": lambda: None,
             "head array": lambda loops_at_0_deg, loops_at_35_deg, loops_at_65_deg, loops_at_95_deg, loop_radius_mm: CoilArray.helmet(
                 rows=tuple((angle, int(count)) for angle, count in zip((0.0, 35.0, 65.0, 95.0), (
                     loops_at_0_deg, loops_at_35_deg, loops_at_65_deg, loops_at_95_deg)) if count >= 1),
                 loop_radius_mm=loop_radius_mm),
             "ring of loops": lambda channels, ring_radius_mm, loop_radius_mm, z_mm: CoilArray.ring(
                 int(channels), ring_radius_mm=ring_radius_mm, loop_radius_mm=loop_radius_mm, z_mm=z_mm),
             "surface loop": lambda x_mm, y_mm, z_mm, loop_radius_mm: CoilArray.surface(       # facing the head's centre
                 (x_mm, y_mm, z_mm), tuple(c - v for c, v in zip(CoilArray.HEAD_CENTER_MM, (x_mm, y_mm, z_mm))),
                 loop_radius_mm=loop_radius_mm)}
    COIL_SETTINGS = {"one ideal channel": {}, "head array": dict(loops_at_0_deg=1, loops_at_35_deg=6, loops_at_65_deg=10, loops_at_95_deg=15, loop_radius_mm=38.0),
                     "ring of loops": dict(channels=8, ring_radius_mm=110.0, loop_radius_mm=40.0, z_mm=0.0),
                     "surface loop": dict(x_mm=0.0, y_mm=-105.0, z_mm=35.0, loop_radius_mm=50.0)}

    #**********#
    #   init   #
    #**********#
    def __init__(self, subject="Head, QSM challenge, 0.64 mm"):
        self.scanner = "siemens_prisma_3t"
        self.coil = "one ideal channel"
        self.subject = subject
        self.position_mm = [0.0, -45.0, 35.0]
        self.view_mm = [0.0, -45.0, 35.0]
        self.size_mm = [20.0, 20.0, 20.0]
        self.calibration = None
        self.seed = int(np.random.default_rng().integers(2 ** 31))
        self.run_settings, self.motion_mm = {}, None
        self.align = True
        self.prescan, self.clock_s, self._drift = True, 0.0, None
        self.unfitted = ""
        self.protocol = Protocol()
        self.queue = []
        self.overrides = {}
        self.subjects, self.pictures, self.images = dict(SUBJECTS), dict(PICTURES), {}
        self.field = FIELDS.get(self.subject, "uniform")
        self.transmit = None
        self.shim, self.shim_mm = "linear", None
        self.overlay = "anatomy"
        self.coil_settings = {name: dict(settings) for name, settings in self.COIL_SETTINGS.items()}
        self.coils, self.own_noise = dict(self.COILS), None
        self.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in STARTER]}
        self.condition, self.weights = "A", {}
        self._bodies = {}
        try:
            self.body()
        except Exception:       # the head's files are placed by hand (resources.qsm): without them, the average brain
            self.subject = "Brain, ICBM152 average, 2 mm"
        self.field = FIELDS.get(self.subject, "uniform")

    #**********#
    #   body   #
    #**********#
    def body(self):
        """The subject on the table, made once per name."""
        if self.subject not in self._bodies:
            self._bodies[self.subject] = self.subjects[self.subject]()
        return self._bodies[self.subject]

    def picture(self):
        """The subject as the planes show it, 8 bits on the subject's grid, made once per name."""
        key = ("picture", self.subject)
        if key not in self._bodies:
            body = self.body()
            shown = self.pictures[self.subject]() if self.subject in self.pictures else 0.85 * body.gm + 1.0 * body.wm + 0.35 * body.csf
            # a relaxation map is windowed on its own middle (its edges hold outliers); an anatomical image of the operator's as
            # a scanner shows it, from nothing to its brightest tissue, fat, so that white matter stays a mid grey
            low, high = ((0.0, float(np.percentile(shown[shown > 0], 99.8))) if self.subject not in PICTURES and self.subject in self.pictures
                         else np.percentile(shown[shown > 0], (2.0, 95.0)) if self.subject in self.pictures else (0.0, float(shown.max())))
            self._bodies[key] = np.uint8(255 * np.clip((shown - low) / max(float(high - low), 1e-6), 0.0, 1.0) * (shown > 0))
        return self._bodies[key]

    def wash(self):
        """The overlay on the subject's grid with its window, (values, low, high, diverging), made
        once per choice; None for the anatomy alone or a map that is not on. A map of another
        grid is read at the nearest cell, NaN outside it."""
        body, fields = self.body(), self.fields()
        source = {"B0 field": fields["field"] and fields["field"].volume, "B1+": fields["transmit"], "grey matter": body.gm,
                  "white matter": body.wm, "CSF": body.csf, "lipid layer": getattr(body, "fat", None)}.get(self.overlay)
        if self.overlay == "lipid layer" and source is not None:       # a tissue of the subject's own grid, outside the brain
            source = source.data if source.data.shape[:3] == body.gm.shape else None
        if source is None:
            return None
        key = ("wash", self.subject, self.overlay, self.field, self.transmit)
        if key not in self._bodies:
            if isinstance(source, Volume):
                step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
                theirs = [(origin[a] + step[a] * np.arange(body.gm.shape[a]) - source.affine[a, 3]) / source.affine[a, a] for a in range(3)]
                inside = [((t > -0.5) & (t < source.shape[a] - 0.5)).reshape([-1 if b == a else 1 for b in range(3)])
                          for a, t in enumerate(theirs)]
                values = np.asarray(source.data[np.ix_(*[np.clip(np.round(t).astype(int), 0, source.shape[a] - 1)
                                                         for a, t in enumerate(theirs)])], dtype=np.float32)
                values = np.where(inside[0] & inside[1] & inside[2] & (body.gm + body.wm + body.csf > 0.05), values, np.nan)
                reach = float(np.nanpercentile(np.abs(values - (1.0 if self.overlay == "B1+" else 0.0)), 90.0))
                middle = 1.0 if self.overlay == "B1+" else 0.0
                self._bodies[key] = (values, middle - reach, middle + reach, True)
            else:
                self._bodies[key] = (np.asarray(source, dtype=np.float32), 0.0, 1.0, False)
        return self._bodies[key]

    @staticmethod
    def volume(path):
        """A NIfTI file as a phantom.Volume in the file's own world frame."""
        import nibabel as nib

        image = nib.load(str(path))
        return Volume(np.asarray(image.get_fdata(), dtype=np.float32), np.array(image.affine, dtype=np.float64),
                      name=pathlib.Path(path).name)

    def onto(self, path):
        """A map of the operator's own (B0, B1+, lipids, a parameter) brought onto the grid of the subject on the table
        by the coordinates the scanner gave both: maps of one session lie where the anatomy lies, whatever their grid
        and angulation. Returns the path of the resampled NIfTI beside the original. (A map of another session must be
        registered first: FSL flirt of its magnitude image onto the anatomy, then its transform on the map.)"""
        import nibabel as nib
        import nibabel.processing

        body, source = self.body(), nib.load(str(path))
        if source.shape[:3] == body.gm.shape and np.allclose(source.affine, body.affine, atol=1e-3):
            return pathlib.Path(path)
        target = (body.gm.shape, body.affine)
        moved = nibabel.processing.resample_from_to(nib.Nifti1Image(np.asarray(source.dataobj, dtype=np.float32), source.affine), target, order=1)
        out = pathlib.Path(str(path).replace(".nii", f"_on_{abs(hash(self.subject)) % 10 ** 6}.nii", 1))
        nib.save(moved, str(out))
        return out

    def lipids(self, path):
        """Give what lies on the table a lipid layer: the operator's own map of the fat-bearing tissue, 0 to 1, brought
        onto the anatomy's grid (`onto`); or, with LENT, the QSM challenge head's layer carried into the average brain's
        or BigBrain-MR's world (resources.qsm QSMHead.fat), which have no scalp of their own."""
        if path == LENT:
            from mrsandbox.resources.qsm import QSMHead

            space = {"Brain, ICBM152 average, 2 mm": "icbm152", "Brain, BigBrain-MR, 0.8 mm": "bigbrain_mr"}.get(self.subject)
            if space is None:
                raise ValueError("the QSM head's layer is carried into the average brain and BigBrain-MR only: another anatomy would need "
                                 "it registered")
            self.body().fat = QSMHead.fat(space)
        else:
            self.body().fat = Volume(self.volume(self.onto(path)).data, self.body().affine, name=pathlib.Path(path).name)
        for key in [k for k in self._bodies if isinstance(k, tuple) and k[0] in ("fat", "wash")]:
            self._bodies.pop(key)

    def anatomy(self, name, gm, wm, csf, picture=None, fat=None):
        """Put the operator's own subject on the shelf: three NIfTI probability maps on one grid
        (a segmentation of their own T1-weighted image), and that image on the same grid for
        the planes to show, and a map of the fat-bearing layer about the skull, where given."""
        maps = [self.volume(path) for path in (gm, wm, csf)]
        self.subjects[name] = lambda: Brain.from_maps(*(m.data for m in maps), maps[0].affine, name=name,
                                                      fat=None if fat is None else self.volume(fat).data)
        if picture is not None:
            self.pictures[name] = lambda: self.volume(picture).data
            self.images[name] = pathlib.Path(picture)

    def segment(self, t1, name, folder=None, series=None):
        """Put a subject on the shelf from their T1-weighted image alone (an MPRAGE, a 3D TFE), a NIfTI
        file or a folder of DICOM of any vendor (io.read_dicom) with the `series` to take from it:
        the image is brought onto an axis-aligned 1 mm grid in scanner coordinates (nibabel's conform),
        the neck cropped and the brain extracted (FSL robustfov, bet) and segmented into CSF, grey, and white matter partial volumes
        (FSL fast), which become the subject's tissue maps with the image as its picture, and the bright
        tissue beyond the bone its lipid layer. FSL must
        be installed (it is no dependency of the package); minutes. The files are kept in `folder`
        (~/MRSandbox/subjects/<name> unless given). Returns the folder."""
        import os
        import shutil
        import subprocess

        import nibabel as nib
        import nibabel.processing

        tools = {tool: shutil.which(tool) or shutil.which(tool, path=os.path.expandvars("$FSLDIR/bin") + os.pathsep + os.path.expanduser(
            "~/fsl/share/fsl/bin")) for tool in ("robustfov", "bet", "fast")}
        if None in tools.values():
            raise RuntimeError("FSL's robustfov, bet and fast are needed to segment an image (fsl.fmrib.ox.ac.uk); or give the three tissue maps")
        folder = pathlib.Path(folder) if folder is not None else pathlib.Path.home() / "MRSandbox" / "subjects" / name
        folder.mkdir(parents=True, exist_ok=True)
        if pathlib.Path(t1).is_dir():
            from mrsandbox.io import read_dicom

            found = read_dicom(t1, folder / "dicom")
            if series not in found:
                raise ValueError("which series is the T1-weighted image? " + ", ".join(found))
            t1 = found[series]
        nib.save(nibabel.processing.conform(nib.load(str(t1)), out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), orientation="RAS"),
                 str(folder / "t1.nii.gz"))
        environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get("FSLDIR", str(pathlib.Path(tools["bet"]).parents[3]))}
        # the neck first: a sagittal 3D image reaches far below the brain, and bet then takes neck and scalp for brain. The
        # cropped image only finds the brain; its mask is carried back onto the whole image's grid by their affines
        subprocess.run([tools["robustfov"], "-i", str(folder / "t1"), "-r", str(folder / "cropped")], check=True, env=environment, capture_output=True)
        subprocess.run([tools["bet"], str(folder / "cropped"), str(folder / "cropped_brain"), "-R", "-f", "0.4", "-m"], check=True, env=environment)
        whole, mask = nib.load(str(folder / "t1.nii.gz")), nib.load(str(folder / "cropped_brain_mask.nii.gz"))
        onto = nibabel.processing.resample_from_to(mask, whole, order=0)
        nib.save(nib.Nifti1Image(np.asarray(whole.dataobj) * (np.asarray(onto.dataobj) > 0), whole.affine), str(folder / "brain.nii.gz"))
        subprocess.run([tools["fast"], "-t", "1", "-n", "3", "-o", str(folder / "tissue"), str(folder / "brain")], check=True, env=environment)
        # the lipid layer: fat is the brightest tissue of a T1-weighted image, so what outside the brain is brighter than
        # white matter, and lies beyond the bone (phantom.Brain.scalp), is scalp fat (and marrow, which a voxel never meets)
        image, white = np.asarray(whole.dataobj, dtype=np.float32), np.asarray(nib.load(str(folder / "tissue_pve_2.nii.gz")).dataobj)
        inside = np.asarray(onto.dataobj) > 0
        layer = ((image > np.median(image[white > 0.9])) & ~inside).astype(np.float32)
        nib.save(nib.Nifti1Image(Brain.scalp(layer, inside.astype(np.float32), whole.affine), whole.affine), str(folder / "fat.nii.gz"))
        self.anatomy(name, folder / "tissue_pve_1.nii.gz", folder / "tissue_pve_2.nii.gz", folder / "tissue_pve_0.nii.gz",
                     picture=folder / "t1.nii.gz", fat=folder / "fat.nii.gz")
        return folder

    def home(self):
        """Bring the voxel and the view to the middle of the subject's brain where the planned place holds
        no tissue (a subject of the operator's own lies in their scanner's coordinates, not the atlas's)."""
        body = self.body()
        if sum(body.fractions(tuple(self.position_mm), tuple(self.size_mm))) < 0.2:
            tissue = np.asarray(body.gm) + np.asarray(body.wm)
            middle = [float((tissue.sum(axis=tuple(b for b in range(3) if b != a)) * np.arange(tissue.shape[a])).sum() / tissue.sum()) for a in range(3)]
            self.position_mm = [float(body.affine[a, 3] + body.affine[a, a] * middle[a]) for a in range(3)]
            self.view_mm = list(self.position_mm)

    def fields(self, card=None):
        """The room's B0 and B1+ as VirtualScan takes them, made once per choice, and the water as the card has it: at
        its full amplitude through the sequence, or the assumed residual (ASSUMED)."""
        key = (self.field, self.transmit)
        if key not in self._bodies:
            field = None if self.field == "uniform" else (
                FieldMap.load(self.field, b0_t=self.profile().scanner.b0_t) if self.field in FieldMap.PRESETS
                else FieldMap.from_map(self.volume(self.field).data, self.volume(self.field).affine, name=pathlib.Path(self.field).name))
            self._bodies[key] = dict(field=field, transmit=None if self.transmit is None else self.volume(self.transmit))
        field = self._bodies[key]["field"]
        if field is not None:           # the adjustment as it stands now: the order, and the box about the planned voxel
            field.shim = self.shim
            field.shim_box = None if self.shim_mm is None else (tuple(self.position_mm), float(self.shim_mm))
        card = self.protocol if card is None else card
        return {**self._bodies[key], **({} if card.water_suppression == ASSUMED else {"water": Unsuppressed()})}

    def idealize(self, name, ideal):
        """Switch one imperfection of IDEALS off (its leaves set to nothing) or back on (the leaves
        as their sources give them)."""
        for path, value in IDEALS[name].items():
            self.overrides.pop(path, None)
            if ideal:
                self.overrides[path] = value

    def whole(self, group, way, number=None):
        """Set a whole branch of the tree at once (every metabolite's T2, say): "as its source gives
        it", "set to" `number`, or "scale by" `number`, which keeps what tells the leaves apart."""
        for path, kind, numbers, _ in self.leaves(own=False):
            if path.split(".")[0] == group and kind != "other":
                self.overrides.pop(path, None)
                if way == "set to":
                    self.overrides[path] = float(number)
                elif way == "scale by":
                    self.overrides[path] = (tuple(float(number) * v for v in numbers) if kind == "range"
                                            else float(number) * numbers[0])

    def place_of(self, position_mm):
        """What the subject is like in a voxel of the planned size about `position_mm`: the tissue, the draw's
        concentrations and relaxation there, the field and the transmit field's means, the distance to the lipid layer.
        The side panel shows it for the planned voxel, and for the point the operator looks at with shift held."""
        kept = list(self.position_mm)
        self.position_mm = [float(v) for v in position_mm]
        try:
            scan = self.profile().scan(self.body(), self.sequence(), params=self.params(), **self.fields())
            voxel = scan.voxel(tuple(self.position_mm), tuple(self.size_mm))
            drawn, fields = voxel.sample(self.seed), self.fields()
            gm, wm = float(voxel.fractions.gm), float(voxel.fractions.wm)

            def seen(name):
                """The concentration the voxel shows: the draw's (that of an equal mix of grey and white matter) split by the
                drawn grey to white contrast r over the tissue that is there, CSF carrying none (Voxel.metabolite_weights)."""
                r = float(voxel.resolve(drawn["contrast"][name]))
                return float(voxel.resolve(drawn["conc"][name])) * (2.0 * r * gm + 2.0 * wm) / (1.0 + r)

            shown = [name for name in ("NAA", "Cr", "PCh", "Glu", "Gln", "mI", "GABA", "GSH", "Lac") if name in drawn["conc"]]
            found = {"tissue": {name: float(getattr(voxel.fractions, name)) for name in ("gm", "wm", "csf")},
                     "concentrations here (mM)": {name: seen(name) for name in shown},
                     "T1 (ms)": {name: 1000.0 * float(voxel.resolve(drawn["t1"][name])) for name in ("NAA", "Cr", "PCh") if name in drawn["t1"]},
                     "T2 (ms)": {name: 1000.0 * float(voxel.resolve(drawn["t2"][name])) for name in ("NAA", "Cr", "PCh") if name in drawn["t2"]}}
            if fields["field"] is not None:
                offsets = fields["field"].offsets(tuple(self.position_mm), tuple(self.size_mm), shim="none")
                found["B0 before the shim (Hz)"] = float(np.mean(offsets)) if len(offsets) else 0.0
                found["line width from B0 (Hz)"] = self.linewidth()
            if fields["transmit"] is not None:
                values, weights = fields["transmit"].box(tuple(self.position_mm), tuple(self.size_mm))
                found["B1+ (of nominal)"] = float((values * weights).sum() / max(weights.sum(), 1e-9))
            lipid = self.lipid_mm()
            if lipid is not None:
                found["lipid layer (mm away)"] = lipid[0]
            # the width a singlet will have: its natural Lorentzian 1 / (pi T2), and as Gaussian parts the microscopic
            # broadening drawn for this scanner and the field's spread over the voxel after the shim (Olivero's Voigt width)
            natural = 1.0 / (np.pi * float(voxel.resolve(drawn["t2"]["NAA"])))
            spread = np.hypot(float(drawn["lineshape"]["gauss_hz"]), max((found.get("line width from B0 (Hz)") or 1.0) - 1.0, 0.0))
            found["NAA's line width expected (Hz)"] = float(0.5346 * natural + np.sqrt(0.2166 * natural ** 2 + spread ** 2))
            found["of it, microscopic broadening (Hz)"] = float(drawn["lineshape"]["gauss_hz"])
            found["noise level (the scanner's calibration)"] = float(drawn["noise"]["sigma"])
            return found
        finally:
            self.position_mm = kept

    def lipid_mm(self):
        """How far the subject's lipid layer is from the planned voxel and from the voxel lipid
        signal is really selected in (its chemical shift displaced box), (planned, displaced) in
        mm, 0 where a box reaches into the layer; None for a subject without such a layer."""
        fat = getattr(self.body(), "fat", None)
        if fat is None:
            return None
        if ("fat", self.subject) not in self._bodies:       # the layer's cells as world points, thinned to a fifth of a million
            index = np.argwhere(fat.data[..., 0] > 0.5 if fat.data.ndim == 4 else fat.data > 0.5)
            index = index[:: max(1, len(index) // 200000)]
            self._bodies[("fat", self.subject)] = (index * np.diag(fat.affine)[:3] + fat.affine[:3, 3]).astype(np.float32)
        points, half = self._bodies[("fat", self.subject)], 0.5 * np.asarray(self.size_mm)
        if len(points) == 0:
            return None
        try:
            moved = np.asarray(self.displacement(SHIFTED["lipid"]))
        except Exception:
            moved = np.zeros(3)
        return tuple(float(np.linalg.norm(np.maximum(np.abs(points - centre) - half, 0.0), axis=1).min())
                     for centre in (np.asarray(self.position_mm), np.asarray(self.position_mm) + moved))

    def encoding(self):
        """Whether the scanner's gradients can play a spectroscopic image's encoding, as {"said": what it
        takes in words, "problems": (...)}; None for a card that is no image. Every moment is played
        as the shortest trapezoid under the scanner's gradient and slew limits (sequences.events
        trapezoid), all axes at once.
        Phase encoded: the largest step reaches k = 1 / (2 cell) along every encoded axis, and it must
        fit between the last pulse and the readout (the design's last delay).
        EPSI: within one spectral dwell the readout crosses k-space along x, 1 / cell, once and flies
        back (flyback) or once (bidirectional); the longest sweep the gradients need sets the widest
        spectral bandwidth the image can have, and the card's must lie under it. The engine itself
        plays these trajectories with ideal gradients (scan.mrsi: the sweep's timing is the model)."""
        card = self.protocol
        if card.sequence != MRSI:
            return None
        scanner = self.profile().scanner

        def lasts(cycles_per_mm):
            _, ramp, flat = trapezoid(cycles_per_mm * 1e3 / (42.5774785 * 1e6), scanner.max_grad_mt_m * 1e-3, scanner.max_slew_t_m_s)
            return 2.0 * ramp + flat

        free = next((float(event.t_s) for event in reversed(self.sequence().events) if isinstance(event, Delay)), 0.0)
        if card.mrsi_readout == "phase encoded":
            axes = [self.size_mm[0], self.size_mm[1]] + ([self.size_mm[2]] if card.slices > 1 else [])
            needs = max(lasts(0.5 / edge) for edge in axes)
            return {"said": f"phase encoding takes {1e3 * needs:.2f} ms of the {1e3 * free:.1f} ms before the readout",
                    "problems": () if needs <= free else (f"the phase encoding gradients need {1e3 * needs:.2f} ms, the design leaves "
                                                          f"{1e3 * free:.1f} ms before the readout: larger cells, or a longer TE",)}
        sweeps = 2 if card.mrsi_readout == "epsi" else 1
        widest = 1.0 / (sweeps * lasts(1.0 / self.size_mm[0]))
        axes = [self.size_mm[1]] + ([self.size_mm[2]] if card.slices > 1 else [])
        needs = max(lasts(0.5 / edge) for edge in axes)
        problems = () if card.bandwidth_hz <= widest else (
            f"a sweep across {int(card.matrix)} cells of {self.size_mm[0]:g} mm takes {1e3 / (sweeps * widest):.2f} ms on these gradients: the "
            f"spectral bandwidth can be {widest:.0f} Hz at most, the card asks {card.bandwidth_hz:g} Hz (larger cells, or fewer Hz)",)
        return {"said": f"EPSI up to {widest:.0f} Hz of spectral bandwidth; phase encoding along y takes {1e3 * needs:.2f} ms",
                "problems": problems + (() if needs <= free else (f"the phase encoding gradients need {1e3 * needs:.2f} ms of {1e3 * free:.1f} ms",))}

    def linewidth(self):
        """The width in Hz the B0 field gives a line of the planned voxel after the shim: the
        full width at half maximum of the voxel's field distribution under a 1 Hz Lorentzian
        (which keeps a perfect shim finite). None in a uniform field."""
        field = self.fields()["field"]
        if field is None:
            return None
        t = np.arange(8192) / 1000.0
        line = np.fft.fft(field.kernel(tuple(self.position_mm), tuple(self.size_mm), t) * np.exp(-np.pi * t), 8 * t.size).real
        above = np.flatnonzero(np.fft.fftshift(line) > 0.5 * line.max())    # between the outermost crossings: a coarse map's
        return float((above[-1] - above[0] + 1) * 1000.0 / (8 * t.size))    # cells give a comb of lines, one width

    def array(self):
        """The receive array on the subject, made from the operator's settings of it; None is the
        one ideal channel."""
        return self.coils[self.coil](**self.coil_settings.get(self.coil, {}))

    def coil_maps(self, path, name):
        """Put a receive array of the operator's own on the shelf: measured sensitivity maps, one NIfTI of
        (nx, ny, nz, channels), complex or real (hardware.coils MappedCoilArray), in the scanner's coordinates."""
        import nibabel as nib

        image = nib.load(str(path))
        maps = np.asarray(image.dataobj)
        if maps.ndim != 4:
            raise ValueError(f"sensitivity maps are one volume per channel, (nx, ny, nz, channels); this file is {maps.shape}")
        self.coils[name] = lambda: MappedCoilArray(maps, np.array(image.affine, dtype=np.float64), name=name)
        self.coil_settings[name] = {}
        self.coil = name

    def coil_noise(self, source):
        """Read the share of a channel's noise that is its own from a raw scan with the coils apart (NIfTI-MRS;
        scan.calibrate CovarianceReading, the FIDs' tails) against the array of loops chosen in the room; the
        exams then load the array with it in place of the package's 0.25. Returns the share."""
        from mrsandbox.scan.calibrate import CovarianceReading

        array = self.array()
        if array is None or not array.loops:
            raise ValueError("choose an array of loops first: its geometry is what the channels' shared noise is held against")
        self.own_noise = float(CovarianceReading.from_nifti(str(source)).own_noise(array))
        return self.own_noise

    def borrow(self, what, source="qsm_rc2"):
        """Lend an anatomy of the operator's own a map of one of the package's heads: the QSM challenge head's B0 field
        or lipid layer, or BigBrain-MR's field, carried over by registering that head's own anatomical image (its R1
        map) onto the operator's T1-weighted image, FSL flirt with twelve degrees of freedom, and playing the transform
        on the map. It is ANOTHER head's field and scalp in this head's shape: a fair stand-in where no map of the
        operator's exists, not a measurement of them. The transform is kept beside the anatomy's image and found again.
        Returns the path of the carried map, on the anatomy's grid."""
        import os
        import shutil
        import subprocess

        import nibabel as nib

        if self.subject not in self.images:
            raise ValueError("a map is carried over onto an anatomy of your own with its image; the package's heads have their own maps")
        flirt = shutil.which("flirt") or shutil.which("flirt", path=os.path.expandvars("$FSLDIR/bin") + os.pathsep + os.path.expanduser(
            "~/fsl/share/fsl/bin"))
        if flirt is None:
            raise RuntimeError("FSL's flirt is needed to carry a map over (fsl.fmrib.ox.ac.uk)")
        environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get("FSLDIR", str(pathlib.Path(flirt).parents[3]))}
        image = self.images[self.subject]
        folder = image.parent / "carried_over"
        folder.mkdir(exist_ok=True)
        transform = folder / f"{source}_to_this.mat"
        if not transform.exists():
            theirs = Volume.load("qsm_rc2/r1") if source == "qsm_rc2" else Volume.load("bigbrain_mr/t1map_800um")
            shown = theirs.data if source == "qsm_rc2" else np.where(theirs.data > 0.0, 1.0 / np.maximum(theirs.data, 0.5), 0.0)
            nib.save(nib.Nifti1Image(np.asarray(shown, dtype=np.float32), theirs.affine), str(folder / f"{source}_anatomy.nii.gz"))
            subprocess.run([flirt, "-in", str(folder / f"{source}_anatomy"), "-ref", str(image), "-omat", str(transform), "-dof", "12",
                            "-cost", "normmi", "-searchrx", "-30", "30", "-searchry", "-30", "30", "-searchrz", "-30", "30"],
                           check=True, env=environment)
        if what == "B0":
            field = FieldMap.load(source, b0_t=self.profile().scanner.b0_t)
            moving = nib.Nifti1Image(np.asarray(field.volume.data, dtype=np.float32), field.volume.affine)
        else:
            fat = Brain.load("qsm_rc2").fat
            moving = nib.Nifti1Image(np.asarray(fat.data, dtype=np.float32), fat.affine)
        nib.save(moving, str(folder / f"{source}_{what}.nii.gz"))
        out = folder / f"{source}_{what}_on_this.nii.gz"
        subprocess.run([flirt, "-in", str(folder / f"{source}_{what}"), "-ref", str(image), "-applyxfm", "-init", str(transform), "-out", str(out)],
                       check=True, env=environment)
        return self.onto(out)

    def register(self, path, reference=None):
        """Bring a map of ANOTHER session onto the subject on the table: FSL flirt, rigid (six degrees of freedom,
        normalized mutual information), of the map's own image onto the subject's picture, then resampled onto its
        grid. Right for a map that shows anatomy (a lipid or tissue map, a B1+ magnitude, an anatomical image); a
        field map in Hz shows none, and wants its MAGNITUDE image registered and the transform applied, which is
        not built. A map of the same session needs none of this (`onto`). `reference` is the image to register
        onto where it is not the picture of what lies on the table (an anatomy still being put together). Returns the
        registered NIfTI's path."""
        import os
        import shutil
        import subprocess

        import nibabel as nib

        flirt = shutil.which("flirt") or shutil.which("flirt", path=os.path.expandvars("$FSLDIR/bin") + os.pathsep + os.path.expanduser(
            "~/fsl/share/fsl/bin"))
        if flirt is None:
            raise RuntimeError("FSL's flirt is needed to register a map (fsl.fmrib.ox.ac.uk)")
        if reference is None:
            reference = pathlib.Path(path).parent / "registration_reference.nii.gz"
            nib.save(nib.Nifti1Image(self.picture().astype(np.float32), self.body().affine), str(reference))
        out = pathlib.Path(str(path).replace(".nii", "_registered.nii", 1))
        subprocess.run([flirt, "-in", str(path), "-ref", str(reference), "-out", str(out), "-dof", "6", "-cost", "normmi"], check=True,
                       env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get("FSLDIR", str(pathlib.Path(flirt).parents[3]))})
        return out if out.exists() else pathlib.Path(str(out) + ".gz")

    @staticmethod
    def scanners():
        """What can stand on the floor, {key: name}: the package's scanners as hardware. What
        calibrates one is chosen apart (`calibrations`)."""
        return dict(sorted(((name, label) for name, label in SCANNERS.items() if name in PRESETS), key=lambda item: item[1]))

    def calibrations(self):
        """What can calibrate the scanner on the floor beside its own records, {key: what it is}:
        the package's records of the same hardware read under another coil (CALIBRATIONS), and
        the operator's own profiles of this scanner, read from their scans and saved as JSON
        (scan.profile) in ./data/profiles or ~/MRSandbox/profiles, which stay on this machine."""
        found = {preset: what for what, preset in CALIBRATIONS.get(self.scanner, {}).items()}
        for folder in (pathlib.Path.cwd() / "data" / "profiles", pathlib.Path.home() / "MRSandbox" / "profiles"):
            for path in sorted(folder.glob("*.json")):
                if json.loads(path.read_text())["scanner"]["name"] == self.scanner:
                    found[str(path)] = "read from your own scans: " + path.stem.replace("_private", "").replace("_", " ")
        return found

    def profile(self):
        """The scanner on the floor with its records: a preset, or a saved profile."""
        if self.calibration in self.calibrations():
            return ScannerProfile.load(self.calibration) if self.calibration.endswith(".json") else ScannerProfile.preset(self.calibration)
        return ScannerProfile.preset(self.scanner)

    def calibrate(self, source, name, folder=None):
        """Calibrate the scanner on the floor from one raw acquisition of the operator's own
        (scan.profile ScannerProfile.read): `source` is NIfTI-MRS with the transients apart (with
        the coils apart too and an array chosen in the room, the array's own noise share is read as
        well), and the card, the voxel, and the room as they stand
        are its virtual twin, so set them as the scan was acquired first. The noise and the line
        broadening become records of a profile saved under `name` in ~/MRSandbox/profiles, which
        stays on this machine and is then this scanner's calibration; returns its path."""
        twin = self.profile().scan(self.body(), self.sequence(), params=self.params(), **self.fields()).voxel(
            tuple(self.position_mm), tuple(self.size_mm), compiled=True)
        folder = pathlib.Path(folder) if folder is not None else pathlib.Path.home() / "MRSandbox" / "profiles"
        folder.mkdir(parents=True, exist_ok=True)
        import nibabel as nib

        # a file with its coils apart, and an array in the room standing in for the coil's geometry: the share of the
        # channels' noise that is their own is read too, and the profile carries the array so loaded
        tags = json.loads(nib.load(str(source)).header.extensions[0].get_content())
        array = self.array() if "DIM_COIL" in (tags.get("dim_5"), tags.get("dim_6"), tags.get("dim_7")) else None
        ScannerProfile.read(str(source), twin, dataset=name, array=array).save(folder / f"{name}.json")
        self.calibration = str(folder / f"{name}.json")
        return folder / f"{name}.json"

    #**************#
    #   sequence   #
    #**************#
    def sequence(self, card=None):
        """The design of a card: one of the library's at the card's timings, or the
        operator's own, the blocks of `design` in order ("Custom")."""
        card = card if card is not None else self.protocol
        card = dataclasses.replace(card, sequence=REAL[card.sequence]) if card.real and card.sequence in REAL else card
        if card.sequence in EDITING:
            return next(iter(self.editing(card).values()))
        if card.sequence == MRSI:
            return SEQUENCES["PRESS"](card.te_ms / 1000.0, card.tr_ms / 1000.0, None, None, None, None)
        scanner = self.profile().scanner
        if card.sequence == "Custom":
            design = self.turned(self.own(card.tr_ms / 1000.0), card.axes)
        else:
            design = self.turned(self.fitted(lambda fit: SEQUENCES[card.sequence](
                card.te_ms / 1000.0, card.tr_ms / 1000.0, 42.5774785 * scanner.b0_t, self.edges(card.axes), fit, "xyz")), card.axes)
        for prelude in reversed(self.preludes(card).values()):
            design = prelude + design
        return design

    @property
    def design(self):
        return self.designs[self.condition]

    @design.setter
    def design(self, blocks):
        self.designs[self.condition] = blocks

    def edges(self, axes):
        """The voxel's edges in the order a design's first, second and third slice pulse will select them: the
        designs are made along x, y, z and then turned (`turned`), so the first pulse gets the edge of axes[0]."""
        return tuple(float(self.size_mm["xyz".index(axis)]) for axis in axes)

    @staticmethod
    def turned(design, axes):
        """The same design with its slice order as the operator wants it, "yzx" say: what it plays along x (its first
        slice pulse and that pulse's gradients), it plays along axes[0], and so on, for every design alike: PRESS and
        sLASER (the 90, then the two pairs), STEAM (its three 90s), the edited ones, and an own design of slice-selective
        pulses and crushers. Every event is copied once, so a pulse played four times stays one object and is computed once."""
        if axes == "xyz":
            return design
        copies = {}
        for event in design.events:
            if id(event) not in copies:
                copies[id(event)] = event = copy.copy(event)
                for name in ("gradient_axis", "axis"):
                    if getattr(event, name, None) in ("x", "y", "z"):
                        setattr(event, name, axes["xyz".index(getattr(event, name))])
        turned = copy.copy(design)
        turned.events = tuple(copies[id(event)] for event in design.events)
        return turned

    def fitted(self, build):
        """A real design fitted to the scanner's transmitter, as a vendor fits its pulses: `build(fit)`
        makes the design (or the sub-experiments of an edited one), and where its amplitude-modulated 90s
        or 180s ask for more B1 than the scanner has, they are played longer by that factor at the same
        time-bandwidth product, so narrower, and displace a shift more. `fit(cls)` is the design's class
        with its pulse durations grown; `fit.grow` the factors by flip angle. An adiabatic full passage
        whose peak exceeds the scanner's is played at the scanner's peak and longer by the square of the
        ratio, which keeps its adiabaticity (peak squared times duration over bandwidth) and with it the
        inversion: a Bloch simulation of the GOIA-WURST pair pulse gives Mz -0.991 in the slab as designed
        (15 uT, 4.5 ms), -0.980 at 13.5 uT unfitted, -0.993 fitted (5.56 ms), and -0.915 against -0.995 at
        11 uT. A design whose timings do not hold the longer pulses stays as it was and says so among the
        card's problems."""
        limit = 42.5774785 * self.profile().scanner.peak_b1_ut

        def fit(cls, grow=None):
            grow = fit.grow if grow is None else grow
            longer = {name: getattr(cls, name) * grow[flip] for name, flip in (("PULSE_S", 90), ("EXCITATION_S", 90), ("REFOCUSING_S", 180))
                      if hasattr(cls, name) and grow[flip] > 1.0}
            if fit.adiabatic and getattr(cls, "AFP_PEAK_HZ", 0.0) > limit:
                longer.update(AFP_PEAK_HZ=0.99 * limit, AFP_S=cls.AFP_S * (cls.AFP_PEAK_HZ / (0.99 * limit)) ** 2)
            return type(cls.__name__, (cls,), longer) if longer else cls

        fit.grow, fit.adiabatic, self.unfitted = {90: 1.0, 180: 1.0}, False, ""           # the probe is the design as it is typed
        probe = build(fit)
        for design in (probe.values() if isinstance(probe, dict) else (probe,)):
            for event in design.events:
                if isinstance(event, ShapedPulse) and event.peak_hz is not None and event.flip_deg is not None and float(event.flip_deg) in (90.0, 180.0):
                    fit.grow[int(event.flip_deg)] = max(fit.grow[int(event.flip_deg)], 1.02 * float(event.peak_hz) / limit)
        fit.grow = {flip: factor if factor > 1.02 else 1.0 for flip, factor in fit.grow.items()}
        fit.adiabatic = any(isinstance(event, ShapedPulse) and event.flip_deg is None and event.peak_hz is not None and float(event.peak_hz) > limit
                        for design in (probe.values() if isinstance(probe, dict) else (probe,)) for event in design.events)
        if max(fit.grow.values()) == 1.0 and not fit.adiabatic:
            return probe
        try:
            return build(fit)
        except ValueError as error:
            self.unfitted = (f"fitted to this scanner's {self.profile().scanner.peak_b1_ut:g} uT the 90s would last {fit.grow[90]:.2f} and the 180s "
                             f"{fit.grow[180]:.2f} times as long, which these timings do not hold ({error}): a longer TE may")
            return probe

    def own(self, tr_s=None, condition=None):
        """The operator's own design (the sub-acquisition shown, or the one named), its blocks in
        order, as a Sequence."""
        blocks = self.designs[self.condition if condition is None else condition]
        return Sequence(*(event for kind, settings in blocks for event in BLOCKS[kind][1](settings)), tr_s=tr_s)

    def phase_cycle(self, block, steps):
        """Make the design shown a phase cycle of `steps` sub-acquisitions (2 or 4): the pulse at `block`
        steps its phase through the circle, and the receiver follows as the pulse's role asks, so the
        wanted pathway adds and the others cancel over the cycle. An excitation (a flip up to 135
        degrees) takes the receiver with it, exp(-i phi); a refocusing pulse is EXORCYCLE (Bodenhausen,
        Freeman, Turner 1977), the receiver at exp(-2 i phi): +, -, +, - over 0, 90, 180, 270. The
        transients then take turns through the cycle, as on a scanner. Coherence filters of the
        design do the same job ideally: take them out to see the cycle work."""
        blocks = self.designs[self.condition]
        kind, settings = blocks[block]
        if "phase_deg" not in settings:
            raise ValueError("a phase cycle steps a pulse's phase: choose a pulse")
        order = 1 if float(settings.get("flip_deg", 90.0)) <= 135.0 else 2
        self.designs, weights = {}, {}
        for step in range(steps):
            phi = 360.0 * step / steps
            name = chr(65 + step)
            self.designs[name] = [(k, dict(v, phase_deg=float(v["phase_deg"]) + phi) if n == block else dict(v)) for n, (k, v) in enumerate(blocks)]
            weight = np.exp(-1j * order * np.radians(phi))
            weights[name] = complex(np.round(weight.real, 12), np.round(weight.imag, 12))
        self.condition, self.weights = "A", {"cycled": weights, "first step alone": {"A": 1.0}}

    def combinations(self):
        """How the own design's sub-acquisitions combine: the operator's `weights` where they name
        sub-acquisitions that exist, else their plain sum."""
        kept = {name: {c: (complex(w) if complex(w).imag else float(complex(w).real)) for c, w in mix.items() if c in self.designs and w}
                for name, mix in self.weights.items()}
        kept = {name: mix for name, mix in kept.items() if mix}
        return kept or {"sum": {name: 1.0 for name in self.designs}}

    def adopt(self):
        """Make the own design the sequence the card plays now, block by block, so the designer opens
        on what the diagram shows: ideal pulses, delays, filters, and crushers over the 20 mm
        ensemble become blocks one can set; every other event (a slice-selective or adiabatic pulse
        of a real design, a projection) comes over as the card plays it. Preludes stay the card's."""
        card = self.protocol
        skip = sum(len(prelude.events) for prelude in self.preludes(card).values()) if REAL.get(card.sequence, card.sequence) in (
            *SEQUENCES, *EDITING, "Custom") else 0
        design = []
        for event in self.sequence().events[skip:]:
            if type(event) is Pulse:
                design.append(("hard pulse", dict(flip_deg=float(event.flip_deg), phase_deg=float(event.phase_deg))))
            elif isinstance(event, Delay):
                design.append(("delay", dict(ms=1000.0 * float(event.t_s))))
            elif isinstance(event, CoherenceFilter) and len(event.orders) == 1:
                design.append(("coherence filter", dict(order=int(event.orders[0]))))
            elif isinstance(event, Gradient) and float(event.extent_mm) == 20.0:
                design.append(("crusher", dict(axis=event.axis, cycles_per_mm=float(event.moment_per_mm))))
            else:
                what = type(event).__name__ + (f", {float(event.flip_deg):.0f}°" if getattr(event, "flip_deg", None) is not None else "") + (
                    f", {1000.0 * float(event.duration_s):.1f} ms" if hasattr(event, "duration_s") else "") + (
                    f", selects along {event.gradient_axis}" if getattr(event, "gradient_hz_per_mm", 0.0) else "")
                design.append(("as the card plays it", dict(what=what, event=event)))
        self.design = design

    def preludes(self, card):
        """What a timed card plays before its localization, in order, {name: design}."""
        found = {}
        if card.water_suppression in SUPPRESSION:
            found[card.water_suppression] = SUPPRESSION[card.water_suppression]()
        if card.ovs:
            found["OVS"] = OuterVolumeSuppression([(axis, side * (0.5 * (edge + OVS_MM) + 2.0), OVS_MM)
                                                   for axis, edge in zip("xyz", self.size_mm) for side in (-1.0, 1.0)],
                                                  42.5774785 * self.profile().scanner.b0_t, extent_mm=float(max(self.size_mm)))
        return found

    def displacement(self, ppm):
        """Where the voxel of spins at `ppm` lies from the planned one, (x, y, z) in mm: under a
        slice gradient of G Hz per mm a pulse centred for its reference selects a shift d ppm away
        at -d f0 / G. Zero along an axis no played pulse selects (an ideal design: all three). The
        card's preludes do not count: an OVS slab's carrier places the slab, it is no reference."""
        f0_mhz = 42.5774785 * self.profile().scanner.b0_t
        moved = [0.0, 0.0, 0.0]
        card = self.protocol
        skip = sum(len(prelude.events) for prelude in self.preludes(card).values()) if REAL.get(card.sequence, card.sequence) in (
            *SEQUENCES, *EDITING, "Custom") else 0
        for event in self.sequence().events[skip:]:
            if isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
                shift = -(ppm - float(event.offset_ppm)) * f0_mhz / float(event.gradient_hz_per_mm)
                axis = "xyz".index(event.gradient_axis)
                moved[axis] = shift if abs(shift) > abs(moved[axis]) else moved[axis]
        return moved

    def editing(self, card):
        """The sub-experiments of an edited card, {name: design}, for this room's field and voxel."""
        card = dataclasses.replace(card, sequence=REAL[card.sequence]) if card.real and card.sequence in REAL else card
        conditions = self.fitted(lambda fit: EDITING[card.sequence][0](
            card, card.tr_ms / 1000.0, 42.5774785 * self.profile().scanner.b0_t, self.edges(card.axes), fit))
        conditions = {name: self.turned(design, card.axes) for name, design in conditions.items()}
        for prelude in reversed(self.preludes(card).values()):      # water suppression and OVS before every sub-experiment
            conditions = {name: prelude + design for name, design in conditions.items()}
        return conditions

    def diagram(self, sequence=None, readout_ms=12.0):
        """A design as a pulse sequence diagram, times in ms: {"rf": [(t, height)] a trace
        of the shaped pulses' envelopes (a real envelope keeps its lobes' signs), "bars":
        [(t, height, selective)] the ideal pulses, which take no time, their flip over 180 and whether
        they act on a band alone (a suppression or editing pulse), "pulses": [(t, text)] every
        pulse's centre and flip, "marks": [(t, text)] for what takes no time and has no
        height (coherence filters), "x", "y", "z": [(t, mT/m)] the gradients as the scanner
        would play them (sequences.timeline), "spans": [(start, end)] of every event, "crushers": [(start, end, axis)] the spans
        that are crushers and no slice selection, "groups": [(start, end, name)] the
        card's preludes (water suppression, OVS), "adc": (start, end)}."""
        preludes = self.preludes(self.protocol) if sequence is None and self.protocol.sequence in (*SEQUENCES, *EDITING, "Custom") else {}
        sequence = sequence if sequence is not None else self.sequence()
        spans = [list(span) for span in event_spans(sequence)]
        # a prelude without duration (an ideal CHESS: selective pulses and filters that take no time) is laid out
        # over a nominal stretch on paper, its events in order, and what follows moves on by it
        held, first = [], 0
        for name, prelude in preludes.items():
            last = first + len(prelude.events)
            if spans[last - 1][1] - spans[first][0] < 1e-9:
                for k in range(first, last):
                    spans[k] = [spans[k][0] + 0.030 * (k - first + 0.5) / (last - first)] * 2
                for k in range(last, len(spans)):
                    spans[k] = [spans[k][0] + 0.030, spans[k][1] + 0.030]
                held.append((spans[first][0] - 0.030 * 0.5 / (last - first), name))
            first = last
        end_ms = 1000.0 * spans[-1][1] if spans else 0.0
        rf, bars, pulses, marks, crushers, groups, first = [(0.0, 0.0)], [], [], [], [], [], 0
        for name, prelude in preludes.items():
            begin = next((1000.0 * at for at, which in held if which == name), 1000.0 * spans[first][0])
            groups.append((begin, begin + 30.0 if any(which == name for _, which in held) else 1000.0 * spans[first + len(prelude.events) - 1][1],
                           name if not any(which == name for _, which in held) else name + ", ideal"))
            first += len(prelude.events)
        for event, (start, stop) in zip(sequence.events, spans):
            start, stop = 1000.0 * start, 1000.0 * stop
            if isinstance(event, ShapedPulse):
                envelope = np.asarray(event.envelope)
                envelope = np.abs(envelope) if np.iscomplexobj(envelope) else envelope.astype(float)
                height = 1.0 if event.flip_deg is None else float(event.flip_deg) / 180.0
                times = np.linspace(start, stop, envelope.size)
                rf += [(start, 0.0)] + [(float(a), float(height * b / np.abs(envelope).max())) for a, b in zip(times, envelope)] + [(stop, 0.0)]
                pulses.append((0.5 * (start + stop), "" if event.flip_deg is None else f"{float(event.flip_deg):.0f}°"))
            elif isinstance(event, CoherenceFilter):
                marks.append((start, f"p = {event.orders if hasattr(event, 'orders') else ''}".strip()))
            elif isinstance(event, Gradient):
                crushers.append((start, stop, event.axis))
            elif hasattr(event, "flip_deg") and not isinstance(event, Delay):
                bars.append((start, float(event.flip_deg) / 180.0, isinstance(event, SelectivePulse)))   # no time in the engine
                pulses.append((start, f"{float(event.flip_deg):.0f}°"))
        rf.append((end_ms + readout_ms, 0.0))
        until_s = (end_ms + readout_ms) / 1000.0
        times, waves = gradient_waveforms(sequence, self.profile().scanner, 5e-5, until_s - 0.030 * len(held))
        for at, _ in held:
            times = np.where(times >= at, times + 0.030, times)
        lanes = {axis: [(float(a), float(b)) for a, b in zip(1000.0 * times[::4], waves.get(axis, np.zeros(times.size))[::4])]
                 for axis in "xyz"}
        return {"rf": rf, "bars": bars, "pulses": pulses, "marks": marks, **lanes, "crushers": crushers, "groups": groups,
                "adc": (end_ms, end_ms + readout_ms), "spans": [(1000.0 * a, 1000.0 * b) for a, b in spans]}

    #************#
    #   leaves   #
    #************#
    def params(self, overrides=None):
        """The operator's own settings as a Params tree over the defaults: {path: value or
        (low, high)} of `overrides` (the session's as they stand, unless given), nested."""
        tree = {}
        for path, value in (self.overrides if overrides is None else overrides).items():
            node = tree
            *branch, leaf = path.split(".")
            for key in branch:
                node = node.setdefault(key, {})
            node[leaf] = Mapped(self.volume(value)) if isinstance(value, str) else value
        return Params(tree)

    def leaves(self, own=True):
        """Every leaf of the parameter tree a scan of this room would draw from, as
        [(path, kind, numbers, changed)]: kind "fixed" with (value,), "range" with (low,
        high), "normal" with (mean, sd), or "other" with the prior's own text (a tissue
        record, a map), which the console shows and does not edit. Without `own` the
        tree is the one before the operator's settings: the priors as their sources give them."""
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
                elif isinstance(value, Normal):
                    found.append((".".join(here), "normal", (float(value.mean), float(value.sd)), ".".join(here) in self.overrides))
                else:
                    found.append((".".join(here), "other", repr(value)[:60], ".".join(here) in self.overrides))

        walk(self.profile().scan(self.body(), sequence, params=self.params() if own else Params({}), **self.fields()).params, ())
        return found

    #************#
    #   planes   #
    #************#
    def planes(self):
        """The three planning planes through the view point as {"tra", "sag", "cor"}: each
        (image, rectangle, cut, cross, shifted), an 8-bit array drawn with its first row at the top
        (anterior up in the transverse plane, superior up in the other two), the voxel's
        outline (left, top, width, height) in the image's pixels, whether this slice cuts
        the voxel, where the other two planes cross it (column, row), and the other
        outlines {name: (left, top, width, height)}: the displaced voxels of SHIFTED (none
        for an ideal design) and the "shim" box where the operator set one; and last the
        world corners of the square picture, [[top left, top right], [bottom left, bottom
        right]] in mm, for a view that hangs the three planes in space."""
        body, tissue, wash = self.body(), self.picture(), self.wash()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = tissue.shape
        i, j, k = (np.asarray(self.position_mm) - origin) / step          # the voxel's centre, the outline's
        a, b, c = (int(min(max(round(v), 0), n - 1)) for v, n in zip((np.asarray(self.view_mm) - origin) / step, (nx, ny, nz)))
        half = 0.5 * np.asarray(self.size_mm) / step
        if self.protocol.sequence == MRSI:          # the slice's whole grid in the plane, one cell thick
            half = half * np.array([self.protocol.matrix, self.protocol.matrix_y, self.protocol.slices])
        side = max(nx, ny, nz)

        try:
            moved = {name: np.asarray(self.displacement(ppm)) / step for name, ppm in SHIFTED.items()}
        except Exception:        # a card that is no design yet draws its voxel alone
            moved = {}
        moved = {name: by for name, by in moved.items() if np.any(by)}

        def framed(cutting, left, top, width, height, cut, cross, across, up):
            """A plane shaded to 8 bits and centred in a square of the largest grid side, all three
            planes then drawing alike; the voxel's outline and the cross move with it."""
            shade = cutting(tissue)
            if wash is not None:    # blue below and red above the middle of a map, sand for a tissue, the anatomy showing through
                values, low, high, diverging = wash
                values = cutting(values)
                if self.overlay == "B0 field":      # as the adjustment leaves it: the shim's own field taken off, everywhere
                    world = [cutting(np.broadcast_to((origin[n] + step[n] * np.arange(tissue.shape[n])).reshape(
                        [-1 if m == n else 1 for m in range(3)]), tissue.shape)) for n in range(3)]
                    values = values - self.fields()["field"].shim_field(
                        np.stack([w.ravel() for w in world], axis=1), tuple(self.position_mm), tuple(self.size_mm)).reshape(values.shape)
                level = np.nan_to_num(np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0), nan=0.5 if diverging else 0.0)
                strength = (0.75 * np.abs(2.0 * level - 1.0) if diverging else 0.7 * level)[..., None]
                colour = np.where((level > 0.5)[..., None], (225, 80, 60), (70, 130, 225)) if diverging else np.array((226, 165, 95))
                shade = np.uint8((1.0 - strength) * shade[..., None] + strength * colour)
            rows, columns = (side - shade.shape[0]) // 2, (side - shade.shape[1]) // 2
            square = np.zeros((side, side) + shade.shape[2:], dtype=np.uint8)
            square[rows:rows + shade.shape[0], columns:columns + shade.shape[1]] = shade
            # the picture's corners in the world: a column runs along `across`, a row down along `up`, the plane lies at the view
            normal = 3 - across - up
            corners = []
            for row in (0, side - 1):
                corners.append([])
                for column in (0, side - 1):
                    point = [0.0, 0.0, 0.0]
                    point[normal] = float(origin[normal] + step[normal] * (a, b, c)[normal])
                    point[across] = float(origin[across] + step[across] * (column - columns))
                    point[up] = float(origin[up] + step[up] * (tissue.shape[up] - 1 - (row - rows)))
                    corners[-1].append(point)
            return (square, (left + columns, top + rows, width, height), bool(cut), (cross[0] + columns, cross[1] + rows),
                    {**{name: (left + columns + by[across], top + rows - by[up], width, height) for name, by in moved.items()},
                     **({} if self.shim_mm is None else {"shim": (
                         left + columns + 0.5 * width - 0.5 * self.shim_mm / step[across],
                         top + rows + 0.5 * height - 0.5 * self.shim_mm / step[up],
                         self.shim_mm / step[across], self.shim_mm / step[up])})},
                    corners)

        return {
            "tra": framed(lambda v: v[:, ::-1, c].T, i - half[0], ny - 1 - j - half[1], 2 * half[0], 2 * half[1],
                          abs(c - k) <= half[2], (a, ny - 1 - b), 0, 1),
            "sag": framed(lambda v: v[a, :, ::-1].T, j - half[1], nz - 1 - k - half[2], 2 * half[1], 2 * half[2],
                          abs(a - i) <= half[0], (b, nz - 1 - c), 1, 2),
            "cor": framed(lambda v: v[:, b, ::-1].T, i - half[0], nz - 1 - k - half[2], 2 * half[0], 2 * half[2],
                          abs(b - j) <= half[1], (a, nz - 1 - c), 0, 2),
        }

    def scroll(self, plane, slices):
        """Leaf through a plane by `slices` of the subject's grid; the voxel stays."""
        body = self.body()
        axis = {"sag": 0, "cor": 1, "tra": 2}[plane]
        step, origin = np.diag(body.affine)[axis], body.affine[axis, 3]
        index = min(max(round((self.view_mm[axis] - origin) / step) + slices, 0), body.gm.shape[axis] - 1)
        self.view_mm[axis] = float(origin + step * index)

    def look(self, plane, column, row):
        """Bring the other two planes to the point pressed in this one, (column, row) in image pixels:
        the view moves through the subject, the voxel stays."""
        kept = list(self.position_mm)
        self.place(plane, column, row)
        self.view_mm, self.position_mm = list(self.position_mm), kept

    def place(self, plane, column, row):
        """Move the voxel to the point clicked in a plane, (column, row) in image pixels; the
        coordinate the plane does not show becomes the slice on view, as the click was on it."""
        body = self.body()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = body.gm.shape
        side = max(nx, ny, nz)
        # the planes are centred in squares of the largest grid side (`planes`)
        px, py, pz = ((side - n) // 2 for n in (nx, ny, nz))
        index = {"tra": (column - px, ny - 1 - (row - py), None), "sag": (None, column - py, nz - 1 - (row - pz)),
                 "cor": (column - px, None, nz - 1 - (row - pz))}[plane]
        for axis, value in enumerate(index):
            self.position_mm[axis] = self.view_mm[axis] if value is None else float(origin[axis] + step[axis] * value)

    #***********#
    #   scene   #
    #***********#
    def scene(self, points=5000):
        """What the room's 3D view draws, world millimetres: the subject's surface as a
        cloud of `points`, the receive loops as (centre, normal, radius), the voxel's box
        as (centre, size)."""
        body = self.body()
        if ("cloud", self.subject) not in self._bodies:          # the surface of a fine grid takes seconds: once per subject
            inside = (body.gm + body.wm) > 0.5
            surface = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1)
                                 & np.roll(inside, -1, 1) & np.roll(inside, 1, 2) & np.roll(inside, -1, 2))
            index = np.argwhere(surface)
            index = index[np.random.default_rng(0).choice(len(index), size=min(points, len(index)), replace=False)]
            self._bodies[("cloud", self.subject)] = (index * np.diag(body.affine)[:3] + body.affine[:3, 3]).astype(np.float32)
        coils = self.array()
        loops = [] if coils is None else [(tuple(loop.center_mm), tuple(loop.normal), float(loop.radius_mm)) for loop in coils.loops]
        self.lipid_mm()                                       # fills the layer's points where the subject has one
        return {"cloud": self._bodies[("cloud", self.subject)], "lipid": self._bodies.get(("fat", self.subject), np.empty((0, 3)))[::40],
                "loops": loops, "voxel": (tuple(self.position_mm), tuple(self.size_mm))}

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
        scan = profile.scan(self.body(), sequence, params=self.params(), **self.fields())
        fractions = scan.voxel(tuple(self.position_mm), tuple(self.size_mm)).fractions
        limits = profile.scanner
        drawn = scan.voxel(tuple(self.position_mm), tuple(self.size_mm)).sample(self.seed)
        return {"subject": {"concentrations (mM)": {name: float(drawn["conc"][name]) for name in ("NAA", "Cr", "PCh", "Glu", "Gln", "mI", "GABA", "GSH", "Lac")
                                                    if name in drawn["conc"] and isinstance(drawn["conc"][name], (int, float))},
                            "T2 (ms)": {name: 1000.0 * float(drawn["t2"][name]) for name in ("NAA", "Cr", "PCh") if name in drawn["t2"]
                                        and isinstance(drawn["t2"][name], (int, float))},
                            "line broadening (Hz)": float(drawn["lineshape"]["gauss_hz"]), "noise level": float(drawn["noise"]["sigma"])},
                "linewidth_hz": self.linewidth(), "lipid_mm": self.lipid_mm(),
                "displaced": {name: self.displacement(ppm) for name, ppm in SHIFTED.items()},
                "gm": float(fractions.gm), "wm": float(fractions.wm), "csf": float(fractions.csf),
                "scan_time_s": card.scan_time_s,
                "limits": (limits.peak_b1_ut, limits.max_grad_mt_m, limits.max_slew_t_m_s),
                "encoding": self.encoding(),
                # (a suppression train's selective pulses are ideal and take no time in the engine: the package's check then says
                # how much time they would need on this scanner, which is said with it)
                "violations": tuple(problem + (" (the suppression train's selective pulses take no time in the simulation; on this scanner "
                                               "they would need that much more)" if problem.startswith("the delays are") and self.preludes(card) else "")
                                    for problem in dict.fromkeys(sequence.violations(limits))) + ((self.unfitted,) if self.unfitted else ()) + (
                    self.encoding()["problems"] if self.encoding() else ()),
                "records": {leaf: record.record(scan) for leaf, record in profile.calibrations().items()}}

    #***********#
    #   queue   #
    #***********#
    def plan(self, queued=True, instead_of=None):
        """Put the card with the voxel as they stand into the queue; returns the exam. An exam
        that is not queued is a preview (the designer's look at a design's spectrum); with
        `instead_of` it takes that planned exam's place and number (an edited card)."""
        card = dataclasses.replace(self.protocol, averages=int(self.protocol.averages), npoints=int(self.protocol.npoints))
        timed = card.sequence in SEQUENCES or card.sequence == MRSI   # an edited or an own design has its own timings
        number = len(self.queue) + 1 if instead_of is None else self.queue.index(instead_of) + 1
        name = f"{number} {card.sequence}" + (f" TE {card.te_ms:g}" if timed else "")
        exam = Exam(name=name, card=card, position_mm=tuple(self.position_mm), size_mm=tuple(self.size_mm),
                    overrides=dict(self.overrides), run=dict(synthesis=dict(self.run_settings), motion_mm=self.motion_mm),
                    records=self.outlook()["records"], sequence=self.sequence(card))
        if card.sequence in EDITING:
            _, exam.combinations, exam.shown = EDITING[card.sequence]
            exam.conditions = self.editing(card)
        elif card.sequence == "Custom" and len(self.designs) > 1:       # the operator's own multiplexed design
            exam.conditions = {name: self.own(card.tr_ms / 1000.0, name) for name in self.designs}
            for prelude in reversed(self.preludes(card).values()):
                exam.conditions = {name: prelude + design for name, design in exam.conditions.items()}
            exam.combinations = self.combinations()
            exam.shown = next(iter(exam.combinations))
        else:
            exam.conditions, exam.combinations = {"scan": exam.sequence}, {"scan": {"scan": 1.0}}
        if instead_of is not None:
            self.queue[number - 1] = exam
        elif queued:
            self.queue.append(exam)
            if card.water_reference and card.sequence != MRSI:     # the same card without water suppression, a few transients
                bare = dataclasses.replace(card, averages=int(card.water_reference), water_suppression="off", water_reference=0)
                unsuppressed = (self.editing(bare) if card.sequence in EDITING else
                                {name: self.own(card.tr_ms / 1000.0, name) for name in self.designs} if card.sequence == "Custom" and len(self.designs) > 1
                                else {"scan": self.sequence(bare)})          # the same card WITHOUT its suppression train
                self.queue.append(dataclasses.replace(
                    exam, name=f"{number + 1} {card.sequence} water reference", frames=[], spectra={}, tracks={}, truth={},
                    card=bare, sequence=next(iter(unsuppressed.values())), conditions=unsuppressed,
                    overrides={**exam.overrides, "water.residual_fraction": 1.0, "downfield.presaturated": False}))
                if card.water_reference_first:          # the reference runs before its card, and the numbers follow the order
                    self.queue[-2], self.queue[-1] = self.queue[-1], self.queue[-2]
                    self.queue[-2].name, exam.name = f"{number} {card.sequence} water reference", exam.name.replace(f"{number} ", f"{number + 1} ", 1)
        return exam

    @property
    def drift_hz_per_min(self):
        """The rate the scanner's frequency drifts at in this session: the operator's, or one draw of
        Augmentrum's 3 T envelope, read off a still synthesizer's two transients a minute apart."""
        if "drift_hz_per_min" in self.run_settings:
            return float(self.run_settings["drift_hz_per_min"])
        if self._drift is None:
            from augmentrum.augmentation import TransientSynthesizer

            still = TransientSynthesizer(n_transients=2, tr_s=60.0, seed=self.seed, ar_sigma_hz=0.0, resp_amp_hz=0.0, events_per_min=0.0)
            self._drift = float(np.diff(still.tracks(still.rng.numpy_rng())[0])[0])
        return self._drift

    def showcase(self):
        """What the own design does to a spectrum, to show the sequence and not a result: two
        transients of a fixed subject (seed 0) in the average brain's parietal white and grey
        matter, a uniform field, and everything of IDEALS off (no noise, macromolecules, lipids,
        residual water, frequency and phase errors, eddy currents, spurious echoes), the run
        still. Returns (ppm, real spectrum) on 0.2 to 4.2 ppm."""
        clean = Session("Brain, ICBM152 average, 2 mm")
        clean.scanner, clean.field = self.scanner, "uniform"
        clean.designs, clean.condition, clean.weights = self.designs, self.condition, self.weights
        clean.protocol = dataclasses.replace(self.protocol, sequence="Custom", averages=2 * len(self.designs), water_reference=0, water_suppression="off", ovs=False)
        clean.overrides = {path: value for leaves in IDEALS.values() for path, value in leaves.items()}
        clean.run_settings = {argument: off for argument, off in RUN.values()}
        exam = clean.run(clean.plan(queued=False), seed=0)
        shown = (exam.ppm > 0.2) & (exam.ppm < 4.2)
        return exam.ppm[shown], exam.frames[-1][1][shown]

    def run(self, exam, seed=None):
        """Acquire an exam: every transient apart, with the instabilities of a real run
        (scan.Multiplex.transients: drift, breathing, motion, as the exam's `run` sets them),
        and the display as it builds up over the transients. The seed is the subject on the
        table unless given; every exam takes noise of its own (its place in the queue)."""
        seed = self.seed if seed is None else seed
        exam.status = "running"
        card, sequence = exam.card, exam.sequence
        profile = self.profile()
        coils = self.array()
        if card.sequence == MRSI:
            grid = profile.scan(self.body(), sequence, params=self.params(exam.overrides), **self.fields(card)).mrsi(
                center_mm=exam.position_mm, matrix=(int(card.matrix), int(card.matrix_y)) + ((int(card.slices),) if card.slices > 1 else ()),
                voxel_mm=exam.size_mm[0], thickness_mm=exam.size_mm[2],
                oversampling=int(card.mrsi_oversampling), weighting=None if card.mrsi_weighting == "none" else card.mrsi_weighting,
                sampling=None if card.mrsi_sampling == "full" else card.mrsi_sampling,
                profile=None if card.mrsi_profile == "box" else card.mrsi_profile,
                trajectory=None if card.mrsi_readout == "phase encoded" else card.mrsi_readout)
            exam.data = grid.acquire(seed=seed, npoints=card.npoints, dwell_s=1.0 / card.bandwidth_hz)
            fids = np.asarray(exam.data[:])                         # (nx, ny, nz, points)
            window = np.exp(-np.pi * 3.0 * np.arange(card.npoints) / card.bandwidth_hz)
            spectra = np.fft.fftshift(np.fft.fft(fids * window, 2 * card.npoints, axis=-1), axes=-1)
            f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
            ppm = 4.65 + np.fft.fftshift(np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz   # the file's frame: the centre at 4.65
            upfield = (ppm > 0.2) & (ppm < 4.2)
            exam.ppm, spectra = ppm[upfield], spectra[..., upfield]
            exam.maps = {name: np.abs(spectra[..., (exam.ppm > low) & (exam.ppm < high)]).max(axis=-1)
                         for name, (low, high) in MAPS.items()}
            top = np.take_along_axis(spectra, np.abs(spectra).argmax(axis=-1)[..., None], axis=-1)
            exam.cells = (spectra * np.exp(-1j * np.angle(top))).real
            centre = exam.cells[int(card.matrix) // 2, int(card.matrix_y) // 2, int(card.slices) // 2]
            exam.frames = [(1, centre, centre)]
            exam.spectra = {"centre cell": centre}
            exam.truth = {name: float(value) for name, value in exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]["conc"].items()
                          if isinstance(value, (int, float))}
            exam.status = "done"
            return exam
        # what is the SCANNER'S of this exam is drawn anew for every exam (its place in the queue), while the seed keeps the
        # person: the frequency offset and phase the acquisition starts with, its eddy currents, its spurious echoes, and
        # the transmit gain, by which a suppression train's residual differs from exam to exam. What the operator set stays.
        anew = int(np.random.SeedSequence([int(seed), self.queue.index(exam) + 1 if exam in self.queue else 0]).generate_state(1)[0])
        drawn = profile.scan(self.body(), sequence, params=self.params(exam.overrides), **self.fields(card)).params.sample(anew).to_dict()
        state = {}
        for group in STATE:             # a branch of leaves, or one leaf at the root (b1_scale)
            for path, value in ([(f"{group}.{leaf}", v) for leaf, v in drawn[group].items()] if isinstance(drawn.get(group), dict)
                                else [(group, drawn.get(group))]):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    state[path] = float(value)
        room = Multiplex(self.body(), profile.scanner, exam.conditions, records=profile.records, params=self.params({**state, **exam.overrides}),
                         **self.fields(card),
                         # the array's noise is tied to the scanner's record AT THIS VOXEL, so every place has the record's SNR: what
                         # an array gives by place (better under its loops) is not in the console yet (info/HANDOFF.md item 196)
                         coils=None if coils is None else (coils.loaded(**({} if self.own_noise is None else {"own_noise": self.own_noise}))
                                                           if coils.loops else coils).calibrated(exam.position_mm, exam.size_mm))
        names = list(exam.conditions)
        schedule = [names[k % len(names)] for k in range(card.averages)]        # interleaved, as an edited scan runs
        run = {**exam.run, "synthesis": {"drift_hz_per_min": self.drift_hz_per_min, **exam.run.get("synthesis", {})}}
        started = 0.0 if self.prescan else self.drift_hz_per_min * self.clock_s / 60.0
        if exam in self.queue:                                  # a preview leaves the scanner's clock alone
            self.clock_s += card.scan_time_s
        exam.data = room.transients(exam.position_mm, exam.size_mm, schedule, tr_s=card.tr_ms / 1000.0, seed=seed,
                                    step=self.queue.index(exam) + 1 if exam in self.queue else 0, frequency_start_hz=started, **run,
                                    npoints=card.npoints, dwell_s=1.0 / card.bandwidth_hz)
        fids = np.asarray(exam.data[:])[0, 0, 0]
        fids = fids[:, None] if fids.ndim == 1 else fids if fids.ndim == 2 else fids.sum(axis=1)   # one transient; channels summed for the display alone
        # the display a console gives: light apodization, zero filled, phased on the tallest line upfield of water
        window = np.exp(-np.pi * 2.0 * np.arange(card.npoints) / card.bandwidth_hz)
        f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
        ppm = 4.65 + np.fft.fftshift(np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz   # the file's frame: the centre at 4.65
        # a water reference shows its water line, every other exam the metabolites upfield of it
        upfield = (ppm > 3.4) & (ppm < 5.9) if exam.overrides.get("water.residual_fraction") == 1.0 else (ppm > 0.2) & (ppm < 4.2)
        played = np.asarray(schedule)
        if self.align and fids.shape[1] > 1:
            # frequency and phase of every transient against the mean of ITS OWN sub-acquisition (what tells the
            # sub-acquisitions apart, an editing pulse or a phase, must stay): the shift that best overlays the apodized
            # spectra within 20 Hz (to an eighth of a bin by zero filling), then the phase of what is left
            clock = np.arange(card.npoints) / card.bandwidth_hz
            fill = 8 * card.npoints
            reach = int(np.ceil(20.0 * fill / card.bandwidth_hz))
            means = {name: fids[:, played == name].mean(axis=1) for name in names}
            for k in range(fids.shape[1]):
                mean = means[played[k]]
                target = np.fft.fft(mean * window * np.exp(-np.pi * 4.0 * clock), fill)
                own = np.fft.fft(fids[:, k] * window * np.exp(-np.pi * 4.0 * clock), fill)
                overlay = [np.abs(np.vdot(np.roll(own, shift), target)) for shift in range(-reach, reach + 1)]
                moved = fids[:, k] * np.exp(2j * np.pi * (int(np.argmax(overlay)) - reach) * card.bandwidth_hz / fill * clock)
                fids[:, k] = moved * np.exp(1j * np.angle(np.vdot(moved, mean)))

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

        # one zero-order phase for the whole exam, read at the tallest line of the average of everything; no transient is
        # aligned in frequency or phase before it is averaged, so what drift and motion do to the average shows
        # (of an own multiplexed design the plain sum may be what the scheme cancels, a phase cycle's unwanted signal: there
        # the shown combination gives the phase; an edited scan's sum is its unedited spectrum)
        own = card.sequence == "Custom" and len(names) > 1
        everything = combined(exam.combinations[exam.shown] if own else {name: 1.0 for name in names}, card.averages)
        phase = np.exp(-1j * np.angle(everything[np.flatnonzero(upfield)[np.abs(everything[upfield]).argmax()]]))
        wide = (ppm > -0.5) & (ppm < 6.5)
        exam.ppm, exam.window = ppm[wide], (float(ppm[upfield].min()), float(ppm[upfield].max()))
        counts = sorted({int(round(c)) for c in np.linspace(1, card.averages, min(card.averages, 64))})
        views = {**{n: {n: 1.0} for n in names if len(names) > 1}, **exam.combinations}
        exam.builds = {name: [(count, (partial * phase).real[wide]) for count in counts
                              if (partial := combined(weights, count)) is not None] for name, weights in views.items()}
        # the display as it builds up: the transients so far, their average in the shown combination, and the last one alone
        exam.frames = [(count, average, (np.fft.fftshift(np.fft.fft(fids[:, count - 1] * window, 2 * card.npoints)) * phase).real[wide])
                       for count, average in exam.builds[exam.shown]]
        exam.spectra = {name: build[-1][1] for name, build in exam.builds.items()}
        # the consensus SNR of what is shown (Near et al. 2021): the tallest real point within 0.2 to 4.2 ppm (NAA in a brain)
        # over the standard deviation of a signal-free stretch, -4 to -1 ppm, after a linear detrend; of the average and of
        # one transient alone
        quiet = (ppm > -4.0) & (ppm < -1.0)

        def snr(fid):
            real = (np.fft.fftshift(np.fft.fft(fid, 2 * card.npoints)) * phase).real
            return float(real[upfield].max() / (real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])).std())

        weights = exam.combinations[exam.shown]
        exam.snr = (snr(sum(w * fids[:, played == c].mean(axis=1) for c, w in weights.items())),
                    snr(fids[:, np.flatnonzero(played == next(iter(weights)))[0]]))
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
        if any(isinstance(w, complex) for mix in self.weights.values() for w in mix.values()):
            raise ValueError("a phase-cycled design is not saved yet: its receiver phases are complex numbers")
        if any(kind == "as the card plays it" for blocks in self.designs.values() for kind, _ in blocks):
            raise ValueError("a design that holds events taken over from a card is not saved yet: they are objects, not settings")
        self.LIBRARY.mkdir(parents=True, exist_ok=True)
        path = self.LIBRARY / f"{name}.json"
        path.write_text(json.dumps({"card": dataclasses.asdict(self.protocol), "size_mm": list(self.size_mm),
                                    "designs": self.designs, "weights": self.weights, "overrides": self.overrides}, indent=2))
        return path

    def load_protocol(self, name):
        """Put a saved protocol on the card."""
        stored = json.loads((self.LIBRARY / f"{name}.json").read_text())
        for attribute, value in stored["card"].items():       # in place: the console's fields hold this card
            setattr(self.protocol, attribute, value)
        self.size_mm = [float(v) for v in (stored["size_mm"] if isinstance(stored["size_mm"], list) else [stored["size_mm"]] * 3)]
        stored_designs = stored.get("designs") or {"A": stored["design"]}           # a protocol saved before designs had sub-acquisitions
        self.designs = {name: [(kind, dict(settings)) for kind, settings in blocks] for name, blocks in stored_designs.items()}
        self.condition, self.weights = next(iter(self.designs)), stored.get("weights", {})
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
