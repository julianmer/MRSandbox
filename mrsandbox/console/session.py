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
import functools
import json
import pathlib
import time

import numpy as np
from keras import ops

from mrsandbox.backend import Backend
from mrsandbox.database import Database, concentrations
from mrsandbox.engine import Engine
from mrsandbox.hardware import CoilArray
from mrsandbox.hardware.arrays import PRESETS as ARRAY_PRESETS
from mrsandbox.hardware.coils import MappedCoilArray
from mrsandbox.hardware.scanners import PRESETS
from mrsandbox.params import Empirical, Fixed, Mapped, Normal, Params, Sourced, Uniform
from mrsandbox.params.values import Tissues
from mrsandbox.phantom import Brain, FieldMap, Phantom, Volume
from mrsandbox.phantom.volume import turning
from mrsandbox.phantom.macromolecules import CoupledMM, MeasuredMM
from mrsandbox.phantom.water import Unsuppressed, WaterSource
from mrsandbox.resources import Resources
from mrsandbox.resources.icbm import NIHPD_BANDS, nihpd_band
from mrsandbox.io import read_seq
from mrsandbox.scan.calibrate import CovarianceReading
from mrsandbox.scan.maps import MapReading
from mrsandbox.scan.run import MotionReading, PhysioReading
from mrsandbox.scan.calibrations import calibrations
from mrsandbox.scan.cohort import holding, package
from mrsandbox.scan import Multiplex, ScannerProfile
from mrsandbox.sequences import (Chess, CoherenceFilter, Cows, SiemensChess, Delay, Gradient,
                                 Hercules, Hermes, LocalizedMegaPress, LocalizedPress,
                                 LocalizedSemiLaser, LocalizedSteam, MegaPress,
                                 OuterVolumeSuppression, Press, PulseAcquire, SelectivePulse,
                                 Pulse, SemiLaser, Sequence, ShapedPulse, Steam, Vapor, Wet)
from mrsandbox.sequences.events import Projection, trapezoid
from mrsandbox.sequences.timeline import event_spans, gradient_waveforms
from mrsandbox.spins import SpinSystem

#**************#
#   subjects   #
#**************#
# who can lie on the table: a name for the picker and how to make them
SUBJECTS = {
    "Brain, ICBM152 average, 2 mm": lambda: Brain.load("icbm152"),
    # a child's tissue in the same frame, one template per age band in years
    **{f"Brain, NIHPD {band} years, 2 mm": (lambda band=band: Brain.load(f"nihpd_{band}"))
       for band in NIHPD_BANDS},
    "Brain, BigBrain-MR, 0.8 mm": lambda: Brain.load("bigbrain_mr"),
    "Head, QSM challenge, 0.64 mm": lambda: Brain.load("qsm_rc2"),
    "Head, Colin27 fuzzy, 0.5 mm": lambda: Brain.load("colin27"),
    # a round sphere on the planes, not 4 mm blocks
    "Phantom, SPECTRE sphere": lambda: Phantom.spectre(temperature_c=21.0, cell_mm=1.0),
    "Phantom, Braino sphere": lambda: Phantom.braino(temperature_c=21.0, cell_mm=1.0),
}
# what the planes show of a subject that has relaxation maps on its grid: a T1-weighted image made
# of them, so every anatomy reads alike, white matter a light grey, grey matter darker, CSF dark;
# the others show their tissue maps mixed to the same contrast (Session.picture). BigBrain-MR:
# 1 - exp(-TR / T1) at TR 250 ms, its T1 held to 0.9 to 5 s so that the map's outliers do not take
# the window. The QSM head: a spoiled gradient echo of TR 200 ms and TE 3 ms, M0 (1 - exp(-TR R1))
# exp(-TE R2*), so bone is dark and the scalp's fat bright and smooth (R1 alone is noise wherever M0
# is small: bone, air, the sinuses)
PICTURES = {
    "Brain, BigBrain-MR, 0.8 mm": lambda: (
        lambda t1: np.where(t1 > 0.0, 1.0 - np.exp(-0.25 / np.clip(t1, 0.9, 5.0)), 0.0))(
        Volume.load("bigbrain_mr/t1map_800um").data),
    "Head, QSM challenge, 0.64 mm": lambda: Volume.load("qsm_rc2/m0").data * (1.0 - np.exp(
        -0.2 * np.maximum(Volume.load("qsm_rc2/r1").data, 0.0))) * np.exp(
        -0.003 * np.maximum(Volume.load("qsm_rc2/r2star").data, 0.0)),
}
# where the voxel starts: medial parietal cortex, well inside the brain and clear of the scalp, in
# the atlas's coordinates and in the QSM head's own
PLACES = {"Head, QSM challenge, 0.64 mm": (0.0, -35.0, 20.0)}
PLACE = (0.0, -45.0, 35.0)

#***************#
#   sequences   #
#***************#
# the single-voxel cards: a name and how a card's timings, the spectrometer frequency, and the
# voxel's size make the design. The ideal ones select the voxel exactly and at once; the played
# ones play their slice-selective pulses under gradients, so the voxel has a profile and every
# shift its own displaced voxel (SHIFTED).
SEQUENCES = {
    "PRESS": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: Press(
        te1_s=min(0.011, te_s / 2.0), te2_s=te_s - min(0.011, te_s / 2.0), tr_s=tr_s),
    "STEAM": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: Steam(te_s=te_s, tm_s=0.010, tr_s=tr_s),
    "sLASER": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: SemiLaser(te_s=te_s, tr_s=tr_s),
    "PRESS, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: LocalizedPress(
        min(0.014 * fit.grow[180], te_s / 2.0), te_s - min(0.014 * fit.grow[180], te_s / 2.0),
        f0_mhz, size_mm=size_mm, tr_s=tr_s, axes=axes,
        excitation_s=0.0026 * fit.grow[90], refocusing_s=0.0065 * fit.grow[180]),
    "STEAM, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: fit(LocalizedSteam)(
        te_s, 0.010, f0_mhz, size_mm=size_mm, tr_s=tr_s),
    "sLASER, played": lambda te_s, tr_s, f0_mhz, size_mm, fit, axes: fit(LocalizedSemiLaser)(
        te_s, f0_mhz, size_mm=size_mm, tr_s=tr_s),
}
# a card's design with its pulses real (Protocol.real): shaped pulses under gradients, played in
# time, in place of ideal ones that act at once and select exactly. HERCULES has no ideal design
# and is always real.
REAL = {"PRESS": "PRESS, played", "STEAM": "STEAM, played", "sLASER": "sLASER, played",
        "MEGA-PRESS": "MEGA-PRESS, played", "HERMES GABA GSH": "HERMES, played"}
# the scanners by name: what a preset stands for (hardware.scanners, scan.calibrations), the
# nominal field in the name (the exact one is the scanner's)
SCANNERS = {
    "generic_1p5t": "Generic", "generic_3t": "Generic", "generic_7t": "Generic",
    **{name: f"""{ {'ge': 'GE', 'canon': 'Canon', 'united': 'United Imaging', 'bruker': 'Bruker'}
                  .get(preset.vendor, str(preset.vendor).title())} """
             f"{preset.model.replace('TrioTim', 'Trio')} "
             f"{min((1.5, 3.0, 5.0, 7.0, 9.4, 11.7), key=lambda f: abs(f - preset.b0_t)):g} T"
       for name, preset in PRESETS.items() if preset.vendor}}
# the package's cohorts by name: the dataset each holds; the rest of a cohort's label comes off its
# rows (Cohort.label)
COHORTS = {"big_gaba_trio": "Big GABA", "big_gaba_ge": "Big GABA", "big_gaba_philips": "Big GABA",
           "cows_prisma": "COWS ds006812", "fmrs_pain_achieva": "fMRS in pain",
           "ds006444_ge_mr750": "OpenNeuro ds006444", "ds005917_siemens_7t": "OpenNeuro ds005917",
           "neuromet_siemens_7t": "NeuroMET", "ds001168_siemens_7t": "OpenNeuro ds001168",
           "hmri_demo_prisma_3t": "hMRI demo", "uk7t_oxford_7t": "UK7T Oxford",
           "ds004611_prisma_3t": "OpenNeuro ds004611", "ixi_guys_1p5t": "IXI Guy's",
           "ixi_hh_3t": "IXI Hammersmith", "ixi_iop_1p5t": "IXI IOP",
           "radial_mese_prisma_3t": "Radial MESE"}
# a run's instabilities as the operator sets them (Augmentrum's TransientSynthesizer,
# scan.Multiplex.transients): a name, the synthesizer's argument, and what switches it off; left
# alone, each is drawn from the 3 T envelope
RUN = {"frequency drift (Hz per minute)": ("drift_hz_per_min", 0.0),
       "slow frequency wander (Hz)": ("ar_sigma_hz", 0.0),
       "breathing (Hz)": ("resp_amp_hz", 0.0), "phase jitter (degrees)": ("phase_jitter_deg", 0.0),
       "amplitude jitter (fraction)": ("amp_jitter_frac", 0.0),
       "broadening over the run (Hz)": ("broaden_hz", 0.0),
       "motion events (per minute)": ("events_per_min", 0.0)}
# what a calibration's twin holds still: no spurious echoes, no frequency or phase error, no leaked
# water (the readings are of these)
CALM = Params(spurious=dict(amplitude=0.0),
              acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
              oov=dict(fraction=0.0))
# whose the run's instabilities are: the scanner's (its frequency and its receiver over time) or the
# subject's (a body breathes and moves; a phantom does neither)
RUN_OF = {"scanner": ("drift_hz_per_min", "ar_sigma_hz", "phase_jitter_deg", "amp_jitter_frac",
                      "broaden_hz"),
          "subject": ("resp_freq_hz", "resp_amp_hz", "events_per_min", "motion_mm")}
# the measured B0 field a subject lies in unless the operator chooses another
# (phantom.FieldMap.PRESETS); a sphere of solution has none
FIELDS = {"Brain, ICBM152 average, 2 mm": "bigbrain_mr",
          **{f"Brain, NIHPD {band} years, 2 mm": "bigbrain_mr" for band in NIHPD_BANDS},
          "Brain, BigBrain-MR, 0.8 mm": "bigbrain_mr",
          "Head, Colin27 fuzzy, 0.5 mm": "bigbrain_mr",
          "Head, QSM challenge, 0.64 mm": "qsm_rc2_measured"}
# what the realism pop-up can switch off, each the leaves it sets to nothing: the console starts
# real
IDEALS = {
    "noise": {"noise.sigma": 0.0},
    # the measured shape's level, and the coupled systems'
    "macromolecules": {"mm.level_mm": 0.0, "mm.kappa": 0.0},
    "lipids": {"lipids.amplitude_mm": 0.0},
    "residual water": {"water.residual_fraction": 0.0, "oov.fraction": 0.0},
    "frequency and phase errors": {"acquisition.shift_hz": 0.0, "acquisition.phase_deg": 0.0},
    "eddy currents": {"acquisition.eddy_std_rad": 0.0, "acquisition.eddy_transient_rad": 0.0},
    "spurious echoes": {"spurious.amplitude": 0.0},
    # no bound pool: the saturation pulses take nothing through it (phantom.transfer)
    "magnetization transfer": {"mt.pool_fraction": 0.0},
}
# what the planes can wash over the anatomy: the room's maps and the subject's tissues
OVERLAYS = ("anatomy", "B0 field", "B1+", "grey matter", "white matter", "CSF", "lipid layer")
LENT = "the QSM head's, carried over"      # a lipid layer for the brains that have no scalp
# the shifts whose displaced voxels the planes draw beside the planned one
SHIFTED = {"water": 4.65, "lipid": 1.3}

# what a timed card can play before its localization: the water suppression trains, and
# outer-volume suppression as six slabs of the card's ovs_mm about the voxel's faces
# (Session.slabs). The slabs play between the train and the excitation, where they saturate what
# the excitation would see; the train's last delay grows by them.
SUPPRESSION = {"VAPOR": Vapor, "WET": Wet, "CHESS": Chess, "CHESS, Siemens svs_se": SiemensChess,
               "COWS(7;236)": Cows,
               "COWS(12;626)": lambda real=False: Cows("COWS(12;626)", real=real),
               "VAPOR7, Igwe 2025": lambda real=False: Cows("VAPOR7", real=real)}
# how the water comes about. The water is a spin system of its own, run through the card's
# sequence at its full amplitude (phantom.water Unsuppressed), so "off" is the whole water line and
# a train suppresses it as far as its physics does, by this scanner's B1 and the water's T1.
# ASSUMED is the package's shortcut: no train is played and the water is scaled to a drawn
# residual fraction (water.residual_fraction, 0.01 to 0.3 percent), fast, and the same whatever the
# card plays.
ASSUMED = "assumed residual"
# the branches of the parameter tree that are the scanner's state of one exam, drawn anew for every
# exam (Session.run)
STATE = ("acquisition", "spurious", "b1_scale", "water.adjusted_residual")
# how far the display processes an exam's transients (Session.display)
COMBINE, ALIGN, PROCESS = "as acquired", "aligned", "processed"


#*************#
#   editing   #
#*************#
# the edited acquisitions of the library: the sub-experiments a card interleaves over its
# transients (made from the repetition time, the spectrometer frequency, and the voxel's size), how
# they combine, and the combination the console shows while it scans. The ideal editing bands are
# fast and right for the edited target, not for co-editing at the bands' flanks; the played schemes
# (20 ms Gaussian editing pulses and the localization's profiles) hold the co-editing too and are
# slower (sequences.library HadamardEditing).
# where a MEGA editing pulse goes for a target, (ON, OFF) in ppm at the canonical TE of 68 ms: the
# card takes any pair, these fill it (GABA: Mescher 1998; GSH: Terpstra 2003; lactate: Star-Lack
# 1998)
TARGETS = {"GABA": (1.88, 7.5), "GSH": (4.56, 7.5), "lactate": (4.1, 7.5)}
DIFFERENCE = {"difference": {"on": 1.0, "off": -1.0}, "sum": {"on": 1.0, "off": 1.0}}
EDITING = {
    "MEGA-PRESS": (lambda card, tr_s, f0_mhz, size_mm, fit: MegaPress.pair(
                       on_ppm=card.edit_on_ppm, off_ppm=card.edit_off_ppm, tr_s=tr_s),
                   DIFFERENCE, "difference"),
    "HERMES GABA GSH": (lambda card, tr_s, f0_mhz, size_mm, fit: Hermes.scheme(tr_s=tr_s),
                        Hermes.combinations, "gaba"),
    "MEGA-PRESS, played": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).pair(
                               f0_mhz, on_ppm=card.edit_on_ppm, off_ppm=card.edit_off_ppm,
                               size_mm=size_mm, tr_s=tr_s),
                           DIFFERENCE, "difference"),
    "HERMES, played": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).scheme(
                           Hermes, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                       Hermes.combinations, "gaba"),
    "HERCULES": (lambda card, tr_s, f0_mhz, size_mm, fit: fit(LocalizedMegaPress).scheme(
                     Hercules, f0_mhz, edit_s=0.020, size_mm=size_mm, tr_s=tr_s),
                         Hercules.combinations, "gaba"),
}

#***********#
#   gears   #
#***********#
# which branches of the parameter tree each panel's gear opens
GEARS = {
    "subject": ("conc", "t1", "t2", "contrast", "diffusion", "mm", "downfield", "lipids", "water"),
    "scanner": ("noise", "lineshape", "b1_scale", "gradients", "acquisition", "spurious", "oov"),
    # and the card's tabs: what of the simulation belongs to each
    "Geometry": ("contrast", "lipids"),
    "Timing": ("t1", "t2", "mm", "lipids", "water", "downfield"),
    "Adjustments": ("lineshape", "b1_scale", "acquisition"),
    "Acquisition": ("noise", "acquisition", "spurious", "gradients"),
    "Imaging": ("lipids", "lineshape"),
}

#**********#
#   mrsi   #
#**********#
# the spectroscopic imaging card: one slice of matrix x matrix cells of the voxel's size about the
# planned place, PRESS at the card's timings, ideal boxes (scan.mrsi); and the lines its maps are
# read at, ppm windows
MRSI = "MRSI PRESS"
MAPS = {"NAA": (1.9, 2.1), "Cr": (2.95, 3.1), "Cho": (3.13, 3.28), "lipid": (0.9, 1.5)}

#************#
#   blocks   #
#************#
FIXED = "fixed pulse"       # a real pulse of the card, in the designer as it is
# the building blocks of a custom design: what the designer offers, the settings each takes with
# their defaults, and how a block becomes events
BLOCKS = {
    "hard pulse": (dict(flip_deg=90.0, phase_deg=0.0),
                   lambda b: (Pulse(b["flip_deg"], phase_deg=b["phase_deg"]),)),
    "own pulse": (dict(flip_deg=90.0, duration_ms=3.0, envelope="0, 0.3, 0.8, 1, 0.8, 0.3, 0",
                       offset_ppm=3.0, phase_deg=0.0),
                  lambda b: (ShapedPulse(
                      b["flip_deg"], b["duration_ms"] / 1000.0,
                      np.array([float(a) for a in b["envelope"].replace(";", ",").split(",")
                                if a.strip()]),
                      offset_ppm=b["offset_ppm"], phase_deg=b["phase_deg"]),)),
    # the shapes are ShapedPulse's own (sequences.events): a sinc of `lobes` zero crossings a side,
    # a Gaussian cut at `truncation` of its peak, the Hermite of 90 or 180 degrees, or an SLR of
    # `time_bandwidth` (excitation at 90 degrees, refocusing otherwise); a setting a shape does not
    # take is left alone
    "shaped pulse": (dict(shape="sinc", flip_deg=90.0, duration_ms=2.6, offset_ppm=3.0,
                          phase_deg=0.0, lobes=2.0, truncation=0.1,
                          time_bandwidth=4.0, slice_axis="none", slice_hz_per_mm=0.0),
                     lambda b: (shaped(b),)),
    "adiabatic pulse": (dict(duration_ms=6.0, bandwidth_hz=1500.0, peak_hz=1000.0,
                             offset_ppm=4.65),
                        lambda b: (ShapedPulse.hyperbolic_secant(
                            b["duration_ms"] / 1000.0, b["bandwidth_hz"], b["peak_hz"],
                            offset_ppm=b["offset_ppm"]),)),
    # an event of the card's own sequence the designer took over (Session.adopt): its shape stays
    # the card's, its flip, duration, carrier, phase, peak B1 and slice gradient are settings like
    # any block's
    FIXED: (dict(what="", event=None), lambda b: (adjusted(b),)),
    "delay": (dict(ms=10.0), lambda b: (Delay(b["ms"] / 1000.0),)),
    "crusher": (dict(axis="x", cycles_per_mm=0.15),
                lambda b: (Gradient(b["cycles_per_mm"], 20.0, axis=b["axis"]),)),
    "coherence filter": (dict(order=-1, cycles_per_mm=0.0),
                         lambda b: (CoherenceFilter(int(b["order"]), b.get("cycles_per_mm", 0.0)),)),
    "water suppression": (dict(scheme="COWS(7;236)", pulses="played"),
                          lambda b: SUPPRESSION[b["scheme"]](
                              real=b.get("pulses", "ideal") == "played").events),
}


#**********************#
#   a shaped pulse     #
#**********************#
def shaped(block):
    """The ShapedPulse a "shaped pulse" block describes, by its shape's own settings."""
    slice = {} if block["slice_axis"] == "none" else dict(
        gradient_hz_per_mm=block["slice_hz_per_mm"], gradient_axis=block["slice_axis"])
    flip, duration = block["flip_deg"], block["duration_ms"] / 1000.0
    if block["shape"] == "slr":
        return ShapedPulse.slr(flip, duration, block.get("time_bandwidth", 4.0),
                               "excitation" if flip == 90.0 else "refocusing",
                               offset_ppm=block["offset_ppm"], phase_deg=block["phase_deg"],
                               **slice)
    own = {"sinc": dict(lobes=int(block.get("lobes", 2))),
           "gaussian": dict(truncation=block.get("truncation", 0.1))}.get(block["shape"], {})
    return getattr(ShapedPulse, block["shape"])(flip, duration, offset_ppm=block["offset_ppm"],
                                                phase_deg=block["phase_deg"], **own, **slice)


#**********************#
#   an event in words   #
#**********************#
def named(event):
    """What a block of the card's says of its event: a pulse's flip or "adiabatic", duration and
    slice axis; a gradient's axis and span; a projection, which has nothing to set."""
    if isinstance(event, Gradient):
        return f"along {event.axis}, over {float(event.extent_mm):g} mm"
    if isinstance(event, Projection):
        return "ideal, nothing to set"
    return ("adiabatic" if isinstance(event, ShapedPulse) and event.flip_deg is None
            else f"{float(event.flip_deg):.0f}°" if getattr(event, "flip_deg", None) is not None
            else type(event).__name__) + (
        f", {1000.0 * float(event.duration_s):.1f} ms" if hasattr(event, "duration_s") else "") + (
        f", slice along {event.gradient_axis}" if getattr(event, "gradient_hz_per_mm", 0.0) else "")


#***********************#
#   an adopted pulse    #
#***********************#
def adjusted(block):
    """The event a fixed block took over from the card, with the scalar settings the designer put
    on it: its shape (the envelope, the sweep) stays the card's."""
    event = copy.copy(block["event"])
    for key, name in (("flip_deg", "flip_deg"), ("phase_deg", "phase_deg"),
                      ("offset_ppm", "offset_ppm"), ("peak_hz", "peak_hz"),
                      ("slice_hz_per_mm", "gradient_hz_per_mm"),
                      ("cycles_per_mm", "moment_per_mm")):
        if key in block:
            setattr(event, name, float(block[key]))
    if ("flip_deg" in block and event.peak_hz is not None
            and float(block["flip_deg"]) != float(block["event"].flip_deg)):
        # a flip set by hand governs the amplitude; the fit's peak B1 stood for the old flip
        event.peak_hz = None
    if "duration_ms" in block:
        event.duration_s = float(block["duration_ms"]) / 1000.0
    return event
# what a design can start from, in the designer's own blocks: a spin echo with played crushers (the
# design the designer opens with), and the library's ideal PRESS and STEAM, their coherence pathways
# kept by filters
STARTER = [("hard pulse", dict(flip_deg=90.0)), ("delay", dict(ms=15.0)),
           ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)),
           ("crusher", dict(axis="x", cycles_per_mm=0.15)),
           ("delay", dict(ms=15.0))]
TEMPLATES = {
    "Spin echo, played crushers, TE 30": STARTER,
    "PRESS, TE 30": [("hard pulse", dict(flip_deg=90.0)),
                     ("coherence filter", dict(order=-1)), ("delay", dict(ms=5.5)),
                     ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)),
                     ("coherence filter", dict(order=1)), ("delay", dict(ms=15.0)),
                     ("hard pulse", dict(flip_deg=180.0, phase_deg=90.0)),
                     ("coherence filter", dict(order=-1)), ("delay", dict(ms=9.5))],
    "sLASER, TE 30": [("hard pulse", dict(flip_deg=90.0)),
                      ("coherence filter", dict(order=-1)), ("delay", dict(ms=0.75)),
                      ("adiabatic pulse", {}),
                      ("coherence filter", dict(order=1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}),
                      ("coherence filter", dict(order=-1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}),
                      ("coherence filter", dict(order=1)), ("delay", dict(ms=1.5)),
                      ("adiabatic pulse", {}),
                      ("coherence filter", dict(order=-1)), ("delay", dict(ms=0.75))],
    "STEAM, TE 20, TM 10": [("hard pulse", dict(flip_deg=90.0)),
                            ("coherence filter", dict(order=1)), ("delay", dict(ms=10.0)),
                            ("hard pulse", dict(flip_deg=90.0)),
                            ("coherence filter", dict(order=0)), ("delay", dict(ms=10.0)),
                            ("hard pulse", dict(flip_deg=90.0, phase_deg=180.0)),
                            ("coherence filter", dict(order=-1)), ("delay", dict(ms=10.0))],
}


#**************#
#   datasets   #
#**************#
@functools.lru_cache(maxsize=None)
def _datasets():
    return {cohort.name: cohort.label(COHORTS.get(cohort.name, cohort.name))
            for cohort in package()}


#***************#
#   describe    #
#***************#
def _describe(value):
    """A Value as the console lists it, (kind, numbers): "fixed" with (value,), "range" with (low,
    high), "normal" with (mean, sd), "empirical" with (median, 5th, 95th, subjects), or "other"
    with the value's own text (a tissue record, a map); a Sourced leaf as its chosen source."""
    if isinstance(value, Sourced):
        value = value.current
    if isinstance(value, Empirical):    # a pooled leaf: one measured subject per draw
        return "empirical", (value.value, value.low, value.high, float(len(value.values)))
    if (isinstance(value, Fixed) and isinstance(value.value, (int, float))
            and not isinstance(value.value, bool)):
        return "fixed", (float(value.value),)
    if isinstance(value, Uniform):
        return "range", (float(value.low), float(value.high))
    if isinstance(value, Normal):
        return "normal", (float(value.mean), float(value.sd))
    return "other", repr(value)[:60]


# the makers of the receive arrays on the shelf (hardware.arrays), as the console names them
MAKERS = {"siemens": "Siemens", "ge": "GE", "philips": "Philips", "research": "Research"}


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
    water_suppression: str = "COWS(7;236)"
    ovs: bool = False
    ovs_mm: float = 30.0
    ovs_gap_mm: float = 10.0
    # the six slabs, a pair along every axis: on or off, a gap and thickness of their own (None:
    # the card's), and a tilt toward the first and the second of the other two axes in x, y, z
    # order, in degrees; replaced as a whole, never changed in place, so a planned exam keeps
    # the card it was planned with
    ovs_slabs: tuple = dataclasses.field(default_factory=lambda: tuple(
        {"axis": axis, "side": side, "on": True, "gap_mm": None, "mm": None,
         "tilt_deg": (0.0, 0.0)} for axis in "xyz" for side in (-1.0, 1.0)))
    real: bool = True
    water_reference: int = 1
    water_reference_first: bool = True
    axes: str = "xyz"
    edit_on_ppm: float = 1.88
    edit_off_ppm: float = 7.5
    # a spectroscopic image's encoding (scan.mrsi): how finely the subject is rendered under each
    # cell (1: ideal boxes, no bleed between cells), the k-space weighting and coverage, the slab's
    # profile, and the readout
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
        covered = (((kx[:, None] / (nx / 2.0)) ** 2 + (ky[None, :] / (ny / 2.0)) ** 2 <= 1.0).sum()
                   if self.mrsi_sampling == "elliptical" else nx * ny)
        return ((ny if self.mrsi_readout != "phase encoded" else int(covered)) * nz
                * self.tr_ms / 1000.0)


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
        ppm: The display's axis, the whole acquired band.
        frames: The display as the scan comes in, [(transients so far, the real spectrum of their
            average, the last transient's own)], on `ppm`; `window` is what the display
            opens on. `builds` holds the same build-up for every condition and combination.
        room: The room as it stood when the exam was planned (Session.ROOM: scanner, calibration,
            coil, anatomy, person, field, shim); the exam runs in it whatever stands in the room
            by then.
        schedule, transients, rejected: The sub-experiment every transient played; the kept
            transients' own spectra {view of one sub-experiment: [real spectrum]}, 32 at most;
            how many the processing left out as unlike the rest.
        snr: The signal to noise ratio of what is shown, (of the average, of one transient
            alone), the tallest real point upfield over the noise of a signal-free stretch, no
            apodization.
        records: What the header says of the calibration records in force.
        sequence: The design as it was when the exam was planned (an edited card's first
            sub-experiment).
        conditions, combinations, shown: The sub-experiments {name: design} interleaved over
            the transients, how they combine {name: {condition: weight}}, and the
            combination the display shows; one condition "scan" for an unedited card.
        spectra: After the run, {name: real spectrum} of every condition and combination.
        tracks: What the scanner and the subject did over the transients, {name: values}.
        echoes: The spurious echo's size in every transient, 0 where the transient came without it.
        adjustment: The factor the prescan put on the water suppression train's flips for this
            subject and voxel (Session.adjust), 1.0 where nothing was adjusted.
        truth: The drawn concentrations in mM, {metabolite: value}.
        overrides, run: The operator's settings and the run's instabilities as they stood when the
            exam was planned; a water reference plays its card without water suppression.
        maps, cells: Of a spectroscopic imaging exam, {line: (nx, ny, nz) heights} and the
            cells' real spectra (nx, ny, nz, points), each phased on its own tallest line.
    """

    name: str
    card: Protocol
    position_mm: tuple
    size_mm: tuple
    turn_deg: tuple = (0.0, 0.0, 0.0)
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
    echoes: list = dataclasses.field(default_factory=list)
    adjustment: float = 1.0
    truth: dict = dataclasses.field(default_factory=dict)
    maps: dict = dataclasses.field(default_factory=dict)
    cells: object = None
    window: tuple = (0.2, 4.2)
    snr: tuple = ()
    builds: dict = dataclasses.field(default_factory=dict)
    room: dict = dataclasses.field(default_factory=dict)
    schedule: list = dataclasses.field(default_factory=list)
    transients: dict = dataclasses.field(default_factory=dict)
    rejected: int = 0
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
        person: Who lies there, as the scanner registers them: {"age_y": years, "sex": "F" or
            "M", "health": "healthy" or the condition by name}; the measured rows a person
            carries (the water's relaxation, breathing, motion) are pooled over people like
            them (scan.cohort.pooled). Empty is a healthy adult; a cohort's subject on the
            floor is its own person.
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
        processing: How far the display processes the transients before it shows them
            (`display`): COMBINE, ALIGN, or PROCESS; the export stays raw.
        run: What the operator set of the run beyond the tree (empty: the tree's run leaves are
            drawn) (RUN), and the step in mm a motion event moves the head by, (low, high) or
            None.
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

    # the receive arrays on the shelf: how each is made from its settings, and the settings it
    # starts with (mm)
    COILS = {"one ideal channel": lambda: None,
             "head array": lambda loops_at_0_deg, loops_at_35_deg, loops_at_65_deg,
                                  loops_at_95_deg, loop_radius_mm: CoilArray.helmet(
                 rows=tuple((angle, int(count)) for angle, count in zip(
                     (0.0, 35.0, 65.0, 95.0),
                     (loops_at_0_deg, loops_at_35_deg, loops_at_65_deg, loops_at_95_deg))
                            if count >= 1),
                 loop_radius_mm=loop_radius_mm),
             "ring of loops": lambda channels, ring_radius_mm, loop_radius_mm, z_mm: CoilArray.ring(
                 int(channels), ring_radius_mm=ring_radius_mm, loop_radius_mm=loop_radius_mm,
                 z_mm=z_mm),
             # facing the head's centre
             "surface loop": lambda x_mm, y_mm, z_mm, loop_radius_mm: CoilArray.surface(
                 (x_mm, y_mm, z_mm),
                 tuple(c - v for c, v in zip(CoilArray.HEAD_CENTER_MM, (x_mm, y_mm, z_mm))),
                 loop_radius_mm=loop_radius_mm)}
    COIL_SETTINGS = {"one ideal channel": {},
                     "head array": dict(loops_at_0_deg=1, loops_at_35_deg=6, loops_at_65_deg=10,
                                        loops_at_95_deg=15, loop_radius_mm=38.0),
                     "ring of loops": dict(channels=8, ring_radius_mm=110.0, loop_radius_mm=40.0,
                                           z_mm=0.0),
                     "surface loop": dict(x_mm=0.0, y_mm=-105.0, z_mm=35.0, loop_radius_mm=50.0)}
    # the scanners' own arrays as published (hardware.arrays), by maker: no settings, what is cited
    # and what stated on the card
    ARRAYS = {f"{MAKERS[preset.vendor]} · {preset.label}": key
              for key, preset in ARRAY_PRESETS.items()}
    COILS = {**COILS, **{label: (lambda key=key: ARRAY_PRESETS[key].array())
                         for label, key in ARRAYS.items()}}
    COIL_SETTINGS = {**COIL_SETTINGS, **{label: {} for label in ARRAYS}}

    # what of the session is the room an exam is planned in and runs in
    ROOM = ("scanner", "field_t", "calibration", "coil", "subject", "person", "seed", "field",
            "transmit", "shim", "shim_mm")

    #**********#
    #   init   #
    #**********#
    def __init__(self, subject="Head, QSM challenge, 0.64 mm"):
        # a generic system at the field set in the top bar; specific ones in the scanner builder
        self.scanner = "generic_3t"
        self.field_t = None     # the field the scanner runs at; None is the preset's own
        self.coil = "one ideal channel"
        self.subject = subject
        self.person = {}
        self.position_mm = list(PLACES.get(subject, PLACE))
        self.view_mm = list(self.position_mm)
        self.size_mm = [20.0, 20.0, 20.0]
        # the voxel turned about x, then y, then z, in degrees (phantom.volume.turning)
        self.turn_deg = [0.0, 0.0, 0.0]
        self.calibration = None
        # {field of hardware.Scanner: value}: the operator's own limits (custom)
        self.hardware = {}
        self.files = {}         # {path: label}: scanner files chosen from anywhere on this machine
        self.reports = {}       # what the last folder gave, shown on the scanner card
        self.uploads = []       # every file read this session, (path, kind, what it gave)
        # the run's readings of the subject's own files (scan.run), records like a profile's
        self.run_records = ()
        # a package cohort on the floor: every exam scans one of its real subjects
        self.cohort = None
        self.seed = int(np.random.default_rng().integers(2 ** 31))
        # the last design built, by what it depends on (Session.sequence)
        self._sequence_memo = (None, None)
        # the last voxel's width from the field (Session.linewidth)
        self._linewidth_memo = (None, None)
        self.processing = ALIGN
        self.prescan, self.clock_s, self._drift = True, 0.0, None
        self.unfitted = ""
        self.protocol = Protocol()
        self.queue = []
        self.overrides = {}
        # {path: source}: literature or measured chosen per leaf (the custom is an override)
        self.choices = {}
        self.subjects, self.pictures, self.images = dict(SUBJECTS), dict(PICTURES), {}
        self.field = FIELDS.get(self.subject, "uniform")
        self.transmit = None
        self.shim, self.shim_mm = "linear", None
        self.overlay = "anatomy"
        self.coil_settings = {name: dict(settings) for name, settings in self.COIL_SETTINGS.items()}
        self.coils, self.own_noise = dict(self.COILS), None
        self.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings})
                              for kind, settings in STARTER]}
        self.condition, self.weights = "A", {}
        self._bodies = {}
        # what exams took here, by kind, and a whole-voxel design's cost per isochromat
        self._took, self._probes = {}, {}
        try:
            self.body()
        except Exception:
            # the head's files are placed by hand (resources.qsm): without them, the average brain
            self.subject = "Brain, ICBM152 average, 2 mm"
            self.position_mm, self.view_mm = list(PLACE), list(PLACE)
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
            shown = (self.pictures[self.subject]() if self.subject in self.pictures
                     else 1.0 * body.wm + 0.62 * body.gm + 0.12 * body.csf)
            # an anatomical image as a scanner shows it, from nothing to its brightest tissue (fat,
            # where there is a scalp), so that white matter stays a light grey; tissue maps mixed
            # to the same contrast, a little head room above white matter
            low, high = ((0.0, float(np.percentile(shown[shown > 0], 99.8)))
                         if self.subject in self.pictures else (0.0, 1.15 * float(shown.max())))
            self._bodies[key] = np.uint8(
                255 * np.clip((shown - low) / max(float(high - low), 1e-6), 0.0, 1.0) * (shown > 0))
        return self._bodies[key]

    def wash(self):
        """The overlay on the subject's grid with its window, (values, low, high, diverging), made
        once per choice; None for the anatomy alone or a map that is not on. A map of another
        grid is read at the nearest cell, NaN outside it."""
        body, fields = self.body(), self.fields()
        source = {"B0 field": fields["field"] and fields["field"].volume,
                  "B1+": fields["transmit"], "grey matter": body.gm, "white matter": body.wm,
                  "CSF": body.csf, "lipid layer": getattr(body, "fat", None)}.get(self.overlay)
        # a tissue of the subject's own grid, outside the brain
        if self.overlay == "lipid layer" and source is not None:
            source = source.data if source.data.shape[:3] == body.gm.shape else None
        if source is None:
            return None
        key = ("wash", self.subject, self.overlay, self.field, self.transmit)
        if key not in self._bodies:
            if isinstance(source, Volume):
                step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
                theirs = [(origin[a] + step[a] * np.arange(body.gm.shape[a]) - source.affine[a, 3])
                          / source.affine[a, a] for a in range(3)]
                inside = [((t > -0.5) & (t < source.shape[a] - 0.5)).reshape(
                    [-1 if b == a else 1 for b in range(3)]) for a, t in enumerate(theirs)]
                values = np.asarray(source.data[np.ix_(
                    *[np.clip(np.round(t).astype(int), 0, source.shape[a] - 1)
                      for a, t in enumerate(theirs)])], dtype=np.float32)
                values = np.where(inside[0] & inside[1] & inside[2]
                                  & (body.gm + body.wm + body.csf > 0.05), values, np.nan)
                reach = float(np.nanpercentile(
                    np.abs(values - (1.0 if self.overlay == "B1+" else 0.0)), 90.0))
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
        return Volume(np.asarray(image.get_fdata(), dtype=np.float32),
                      np.array(image.affine, dtype=np.float64),
                      name=pathlib.Path(path).name)

    def onto(self, path):
        """A map of the operator's own (B0, B1+, lipids, a parameter) brought onto the grid of the
        subject on the table by the coordinates the scanner gave both: maps of one session lie
        where the anatomy lies, whatever their grid and angulation. Returns the path of the
        resampled NIfTI beside the original. (A map of another session must be registered first:
        FSL flirt of its magnitude image onto the anatomy, then its transform on the map.)"""
        import nibabel as nib
        import nibabel.processing

        body, source = self.body(), nib.load(str(path))
        if source.shape[:3] == body.gm.shape and np.allclose(source.affine, body.affine, atol=1e-3):
            return pathlib.Path(path)
        target = (body.gm.shape, body.affine)
        moved = nibabel.processing.resample_from_to(
            nib.Nifti1Image(np.asarray(source.dataobj, dtype=np.float32), source.affine), target,
            order=1)
        out = pathlib.Path(
            str(path).replace(".nii", f"_on_{abs(hash(self.subject)) % 10 ** 6}.nii", 1))
        nib.save(moved, str(out))
        return out

    def lipids(self, path):
        """Give what lies on the table a lipid layer: the operator's own map of the fat-bearing
        tissue, 0 to 1, brought onto the anatomy's grid (`onto`); or, with LENT, the QSM challenge
        head's layer carried into the average brain's or BigBrain-MR's world (resources.qsm
        QSMHead.fat), which have no scalp of their own."""
        if path == LENT:
            from mrsandbox.resources.qsm import QSMHead

            space = {"Brain, ICBM152 average, 2 mm": "icbm152",
                     "Brain, BigBrain-MR, 0.8 mm": "bigbrain_mr"}.get(self.subject)
            if space is None:
                raise ValueError("the QSM head's layer is carried into the average brain and "
                                 "BigBrain-MR only: another anatomy would need it registered")
            self.body().fat = QSMHead.fat(space)
        else:
            self.body().fat = Volume(self.volume(self.onto(path)).data, self.body().affine,
                                     name=pathlib.Path(path).name)
        for key in [k for k in self._bodies if isinstance(k, tuple) and k[0] in ("fat", "wash")]:
            self._bodies.pop(key)

    def anatomy(self, name, gm, wm, csf, picture=None, fat=None):
        """Put the operator's own subject on the shelf: three NIfTI probability maps on one grid
        (a segmentation of their own T1-weighted image), and that image on the same grid for
        the planes to show, and a map of the fat-bearing layer about the skull, where given."""
        maps = [self.volume(path) for path in (gm, wm, csf)]
        self.subjects[name] = lambda: Brain.from_maps(
            *(m.data for m in maps), maps[0].affine, name=name,
            fat=None if fat is None else self.volume(fat).data)
        if picture is not None:
            self.pictures[name] = lambda: self.volume(picture).data
            self.images[name] = pathlib.Path(picture)

    def segment(self, t1, name, folder=None, series=None):
        """Put a subject on the shelf from their T1-weighted image alone (an MPRAGE, a 3D TFE), a
        NIfTI file or a folder of DICOM of any vendor (io.read_dicom) with the `series` to take
        from it: the image is brought onto an axis-aligned 1 mm grid in scanner coordinates about
        its own centre (resampled by its affine, so an oblique acquisition is set straight:
        nibabel's conform keeps the tilt), the neck cropped and the brain extracted (FSL robustfov,
        bet) and segmented into CSF, grey, and white matter partial volumes (FSL fast), which
        become the subject's tissue maps with the image as its picture, and the bright tissue
        beyond the bone its lipid layer. FSL must be installed (it is no dependency of the
        package); minutes. The files are kept in `folder` (~/MRSandbox/subjects/<name> unless
        given). Returns the folder."""
        import nibabel as nib

        from mrsandbox.scan import maps

        maps._fsl("robustfov", "bet", "fast")            # all three, before any work
        folder = (pathlib.Path(folder) if folder is not None
                  else pathlib.Path.home() / "MRSandbox" / "subjects" / name)
        folder.mkdir(parents=True, exist_ok=True)
        if pathlib.Path(t1).is_dir():
            from mrsandbox.io import read_dicom

            found = read_dicom(t1, folder / "dicom")
            if series not in found:
                raise ValueError("which series is the T1-weighted image? " + ", ".join(found))
            t1 = found[series]
        nib.save(maps.conform(nib.load(str(t1))), str(folder / "t1.nii.gz"))
        whole = nib.load(str(folder / "t1.nii.gz"))
        onto = maps.tissues(folder / "t1.nii.gz", folder)
        # the lipid layer: fat is the brightest tissue of a T1-weighted image, so what outside the
        # brain is brighter than white matter, and lies beyond the bone (phantom.Brain.scalp), is
        # scalp fat (and marrow, which a voxel never meets)
        image, white = (np.asarray(whole.dataobj, dtype=np.float32),
                        np.asarray(nib.load(str(folder / "tissue_pve_2.nii.gz")).dataobj))
        inside = np.asarray(onto.dataobj) > 0
        layer = ((image > np.median(image[white > 0.9])) & ~inside).astype(np.float32)
        nib.save(nib.Nifti1Image(Brain.scalp(layer, inside.astype(np.float32), whole.affine),
                                 whole.affine), str(folder / "fat.nii.gz"))
        self.anatomy(name, folder / "tissue_pve_1.nii.gz", folder / "tissue_pve_2.nii.gz",
                     folder / "tissue_pve_0.nii.gz",
                     picture=folder / "t1.nii.gz", fat=folder / "fat.nii.gz")
        return folder

    def home(self):
        """Bring the voxel and the view to the middle of the subject's brain where the planned
        place holds no tissue (a subject of the operator's own lies in their scanner's
        coordinates, not the atlas's)."""
        body = self.body()
        if sum(body.fractions(tuple(self.position_mm), tuple(self.size_mm), self.turn)) < 0.2:
            tissue = np.asarray(body.gm) + np.asarray(body.wm)
            middle = [float((tissue.sum(axis=tuple(b for b in range(3) if b != a))
                             * np.arange(tissue.shape[a])).sum() / tissue.sum()) for a in range(3)]
            self.position_mm = [float(body.affine[a, 3] + body.affine[a, a] * middle[a])
                                for a in range(3)]
            self.view_mm = list(self.position_mm)

    def fields(self, card=None):
        """The room's B0 and B1+ as VirtualScan takes them, made once per choice, the water as the
        card has it (at its full amplitude through the sequence, or the assumed residual, ASSUMED),
        and the macromolecules: the measured 3 T shape (MeasuredMM, its level read off raw data),
        which matches a raw spectrum's whole background, where the coupled systems fall short. An
        edited card and an own design play the coupled systems (CoupledMM), which run through the
        sequence: the measured shape carries no editing response, and no phase cycle or
        sub-acquisition weight acts on it."""
        key = (self.field, self.transmit)
        if key not in self._bodies:
            name = self.field
            # placed by hand, like the head
            if name == "qsm_rc2_measured" and not Resources.cached("qsm_rc2/total_field"):
                name = "qsm_rc2"
            field = None if name == "uniform" else (
                FieldMap.load(name, b0_t=self.profile().scanner.b0_t) if name in FieldMap.PRESETS
                else FieldMap.from_map(self.volume(name).data, self.volume(name).affine,
                                       name=pathlib.Path(name).name))
            self._bodies[key] = dict(field=field, transmit=None if self.transmit is None
                                     else self.volume(self.transmit))
        field = self._bodies[key]["field"]
        # the adjustment as it stands now: the order, and the box about the planned voxel
        if field is not None:
            field.shim = self.shim
            field.shim_box = (None if self.shim_mm is None
                              else (tuple(self.position_mm), float(self.shim_mm)))
            field.weights = self.content()
        card = self.protocol if card is None else card
        mm = self._bodies.setdefault("mm", {"measured": MeasuredMM(), "coupled": CoupledMM()})
        return {**self._bodies[key],
                "mm": mm["coupled" if card.sequence in EDITING or card.sequence == "Custom"
                         else "measured"],
                **({} if card.water_suppression == ASSUMED
                   else {"water": Unsuppressed(), "water_groups": 8})}

    def content(self):
        """What holds signal in the subject, per cell: the tissues' water content weighted by their
        fractions (a Volume on the subject's grid), for the field to weigh a voxel's cells by; None
        for a subject without tissue maps."""
        body = self.body()
        maps = getattr(body, "maps", None)
        if maps is None:
            return None
        if ("content", self.subject) not in self._bodies:
            content = sum(body.water_content[name] * maps.data[..., k]
                          for k, name in enumerate(("gm", "wm", "csf")))
            self._bodies[("content", self.subject)] = Volume(content.astype(np.float32),
                                                             maps.affine, name="content")
        return self._bodies[("content", self.subject)]

    def reset(self):
        """Everything the operator set about the simulation back to how the console starts: the
        parameters' own sources, the run drawn, the linear shim over the voxel, the prescan, the
        measured field, real pulses."""
        self.overrides = {}
        self.shim, self.shim_mm, self.prescan, self.processing = "linear", None, True, ALIGN
        self.field, self.protocol.real = FIELDS.get(self.subject, "uniform"), True

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
                    self.overrides[path] = (tuple(float(number) * v for v in numbers)
                                            if kind == "range" else float(number) * numbers[0])

    def place_of(self, position_mm):
        """What the subject is like in a voxel of the planned size about `position_mm`: the tissue,
        the draw's concentrations and relaxation there, the field and the transmit field's means,
        the distance to the lipid layer. The side panel shows it for the planned voxel, and for
        the point the operator looks at with shift held."""
        kept = list(self.position_mm)
        self.position_mm = [float(v) for v in position_mm]
        try:
            scan = self.twin()
            voxel = scan.voxel(tuple(self.position_mm), tuple(self.size_mm), turn=self.turn)
            drawn, fields = voxel.sample(self.seed), self.fields()
            gm, wm = float(voxel.fractions.gm), float(voxel.fractions.wm)

            def seen(name):
                """The concentration the voxel shows: the draw's (that of an equal mix of grey and
                white matter) split by the drawn grey to white contrast r over the tissue that is
                there, CSF carrying none (Voxel.metabolite_weights)."""
                r = float(voxel.resolve(drawn["contrast"][name]))
                return (float(voxel.resolve(drawn["conc"][name])) * (2.0 * r * gm + 2.0 * wm)
                        / (1.0 + r))

            # under the names a spectroscopist reads them by: choline is the two choline compounds
            # together
            shown = {"NAA": ("NAA",), "Cr": ("Cr",), "PCr": ("PCr",), "Cho": ("PCho", "GPC"),
                     "Ins": ("Ins",), "Glu": ("Glu",), "Gln": ("Gln",),
                     "GABA": ("GABA",), "GSH": ("GSH",), "Lac": ("Lac",)}
            found = {"tissue": {name: float(getattr(voxel.fractions, name))
                                for name in ("gm", "wm", "csf")},
                     "concentrations (mM)": {label: sum(seen(name) for name in names)
                                             for label, names in shown.items()
                                             if all(name in drawn["conc"] for name in names)},
                     "T1 (ms)": {label: 1000.0 * float(voxel.resolve(drawn["t1"][name]))
                                 for label, name in (("NAA", "NAA"), ("Cr", "Cr"), ("Cho", "PCho"))
                                 if name in drawn["t1"]},
                     "T2 (ms)": {label: 1000.0 * float(voxel.resolve(drawn["t2"][name]))
                                 for label, name in (("NAA", "NAA"), ("Cr", "Cr"), ("Cho", "PCho"))
                                 if name in drawn["t2"]}}
            field = {}
            if fields["field"] is not None:
                offsets = fields["field"].offsets(tuple(self.position_mm), tuple(self.size_mm),
                                                  shim="none", turn=self.turn)
                field["B0 offset, unshimmed (Hz)"] = (float(np.mean(offsets)) if len(offsets)
                                                      else 0.0)
                field["line width from B0 (Hz)"] = self.linewidth()
            if fields["transmit"] is not None:
                values, weights = fields["transmit"].box(tuple(self.position_mm),
                                                         tuple(self.size_mm), self.turn)
                field["B1+ (of nominal)"] = float((values * weights).sum()
                                                  / max(weights.sum(), 1e-9))
            lipid = self.lipid_mm()
            if lipid is not None:
                field["lipid layer (mm away)"] = lipid[0]
            # the width a singlet will have: its natural Lorentzian 1 / (pi T2), and as Gaussian
            # parts the microscopic broadening drawn for this scanner and the field's spread over
            # the voxel after the shim (Olivero's Voigt width)
            natural = 1.0 / (np.pi * float(voxel.resolve(drawn["t2"]["NAA"])))
            spread = np.hypot(float(drawn["lineshape"]["gauss_hz"]),
                              max((field.get("line width from B0 (Hz)") or 1.0) - 1.0, 0.0))
            field["line width of NAA (Hz)"] = float(
                0.5346 * natural + np.sqrt(0.2166 * natural ** 2 + spread ** 2))
            return {**found, "field and lines": field,
                    "scanner": {"broadening below the map's cells (Hz)":
                                float(drawn["lineshape"]["gauss_hz"]),
                                "noise level": float(drawn["noise"]["sigma"])}}
        finally:
            self.position_mm = kept

    def lipid_mm(self):
        """How far the subject's lipid layer is from the planned voxel and from the voxel lipid
        signal is really selected in (its chemical shift displaced box), (planned, displaced) in
        mm, 0 where a box reaches into the layer; None for a subject without such a layer."""
        fat = getattr(self.body(), "fat", None)
        if fat is None:
            return None
        # the layer's cells as world points, thinned to a fifth of a million
        if ("fat", self.subject) not in self._bodies:
            index = np.argwhere(fat.data[..., 0] > 0.5 if fat.data.ndim == 4 else fat.data > 0.5)
            index = index[:: max(1, len(index) // 200000)]
            self._bodies[("fat", self.subject)] = (index * np.diag(fat.affine)[:3]
                                                   + fat.affine[:3, 3]).astype(np.float32)
        points, half = self._bodies[("fat", self.subject)], 0.5 * np.asarray(self.size_mm)
        if len(points) == 0:
            return None
        try:
            moved = np.asarray(self.displacement(SHIFTED["lipid"]))
        except Exception:
            moved = np.zeros(3)
        # the boxes along their own axes
        turn = np.eye(3) if self.turn is None else self.turn
        return tuple(float(np.linalg.norm(np.maximum(np.abs((points - centre) @ turn) - half, 0.0),
                                          axis=1).min())
                     for centre in (np.asarray(self.position_mm),
                                    np.asarray(self.position_mm) + turn @ moved))

    def encoding(self):
        """Whether the scanner's gradients can play a spectroscopic image's encoding, as {"said":
        what it takes in words, "problems": (...)}; None for a card that is no image. Every moment
        is played as the shortest trapezoid under the scanner's gradient and slew limits
        (sequences.events trapezoid), all axes at once.
        Phase encoded: the largest step reaches k = 1 / (2 cell) along every encoded axis, and it
        must fit between the last pulse and the readout (the design's last delay).
        EPSI: within one spectral dwell the readout crosses k-space along x, 1 / cell, once and
        flies back (flyback) or once (bidirectional); the longest sweep the gradients need sets the
        widest spectral bandwidth the image can have, and the card's must lie under it. The engine
        itself plays these trajectories with ideal gradients (scan.mrsi: the sweep's timing is the
        model)."""
        card = self.protocol
        if card.sequence != MRSI:
            return None
        scanner = self.profile().scanner

        def lasts(cycles_per_mm):
            _, ramp, flat = trapezoid(cycles_per_mm * 1e3 / (42.5774785 * 1e6),
                                      scanner.max_grad_mt_m * 1e-3, scanner.max_slew_t_m_s)
            return 2.0 * ramp + flat

        free = next((float(event.t_s) for event in reversed(self.sequence().events)
                     if isinstance(event, Delay)), 0.0)
        if card.mrsi_readout == "phase encoded":
            axes = ([self.size_mm[0], self.size_mm[1]]
                    + ([self.size_mm[2]] if card.slices > 1 else []))
            needs = max(lasts(0.5 / edge) for edge in axes)
            return {"said": f"phase encoding takes {1e3 * needs:.2f} ms of the "
                            f"{1e3 * free:.1f} ms before the readout",
                    "problems": () if needs <= free else (
                        f"the phase encoding gradients need {1e3 * needs:.2f} ms, the design "
                        f"leaves {1e3 * free:.1f} ms before the readout: larger cells, or a "
                        "longer TE",)}
        sweeps = 2 if card.mrsi_readout == "epsi" else 1
        widest = 1.0 / (sweeps * lasts(1.0 / self.size_mm[0]))
        axes = [self.size_mm[1]] + ([self.size_mm[2]] if card.slices > 1 else [])
        needs = max(lasts(0.5 / edge) for edge in axes)
        problems = () if card.bandwidth_hz <= widest else (
            f"a sweep across {int(card.matrix)} cells of {self.size_mm[0]:g} mm takes "
            f"{1e3 / (sweeps * widest):.2f} ms on these gradients: the spectral bandwidth can be "
            f"{widest:.0f} Hz at most, the card asks {card.bandwidth_hz:g} Hz (larger cells, or "
            "fewer Hz)",)
        return {"said": f"EPSI up to {widest:.0f} Hz of spectral bandwidth; phase encoding "
                        f"along y takes {1e3 * needs:.2f} ms",
                "problems": problems + (() if needs <= free else (
                    f"the phase encoding gradients need {1e3 * needs:.2f} ms of "
                    f"{1e3 * free:.1f} ms",))}

    def linewidth(self):
        """The width in Hz the B0 field gives a line of the planned voxel after the shim: the
        full width at half maximum of the voxel's field distribution under a 1 Hz Lorentzian
        (which keeps a perfect shim finite). None in a uniform field."""
        field = self.fields()["field"]
        if field is None:
            return None
        # the same voxel in the same field: the same width
        key = (id(field), tuple(self.position_mm), tuple(self.size_mm), tuple(self.turn_deg),
               self.shim, self.shim_mm)
        if self._linewidth_memo[0] == key:
            return self._linewidth_memo[1]
        t = np.arange(8192) / 1000.0
        line = np.fft.fft(field.kernel(tuple(self.position_mm), tuple(self.size_mm), t,
                                       turn=self.turn)
                          * np.exp(-np.pi * t), 8 * t.size).real
        # between the outermost crossings: a coarse map's cells give a comb of lines, one width
        above = np.flatnonzero(np.fft.fftshift(line) > 0.5 * line.max())
        self._linewidth_memo = (key, float((above[-1] - above[0] + 1) * 1000.0 / (8 * t.size)))
        return self._linewidth_memo[1]

    def array(self):
        """The receive array on the subject, made from the operator's settings of it; None is the
        one ideal channel."""
        return self.coils[self.coil](**self.coil_settings.get(self.coil, {}))

    def own_share(self):
        """The share of a channel's noise that is its own, in force for the array on the floor, and
        where it comes from: the operator's own reading (coil_noise), else the package's rows
        pooled for the scanner at the array's channel count (scan.cohort), as (value, source,
        provenance): the channels' measured correlation turned into the share against this array
        (a share read against a stand-in geometry carries the stand-in), else a pooled share."""
        if self.own_noise is not None:
            return self.own_noise, "own", "read from your scan with the coils apart"
        array = self.array()
        served = (self.profile().calibrations() if array is None
                  else calibrations(self.profile().scanner, self.profile().records,
                                    channels=len(array)))
        correlation = served.get("coils.partner_correlation")
        # a mapped array has no geometry to take it against
        if array is not None and array.loops and correlation is not None:
            return (round(CovarianceReading.share(array, correlation.value), 2), "pooled",
                    f"the channels' correlation {correlation.value:.2f}, "
                    f"{correlation.source['dataset']}, {correlation.date}, "
                    f"against this array")
        record = served.get("coils.loaded.own_noise")
        if record is None:
            return 0.25, "literature", "the package's stated share"
        return float(record.value), "pooled", f"{record.source['dataset']}, {record.date}"

    def load_scanner(self, path):
        """A scanner file from anywhere on this machine (scan.profile JSON) as this scanner's
        calibration."""
        profile = ScannerProfile.load(path)
        self.files[str(path)] = f"your file: {pathlib.Path(path).stem.replace('_', ' ')}"
        if profile.scanner.name in PRESETS:
            self.scanner = profile.scanner.name
        self.calibration = str(path)

    def card_from(self, source):
        """The card and the scanner set from a NIfTI-MRS run's header, as the run was acquired, so
        the room is its twin before it is read: TE, TR, the averages (the transients apart), the
        points, the bandwidth, the sequence by its name (PRESS, STEAM, sLASER; MEGA and HERMES by
        theirs), the voxel's size from the file's grid, the field from the spectrometer frequency,
        the system where the vendor and model, in the header or the BIDS sidecar, name a preset
        (else the generic one at that field), and the head array where the run has 32 channels
        apart (the helmet the cohorts read such runs against). Returns what was set, {field:
        value}."""
        import nibabel as nib

        from mrsandbox.scan.calibrate import _raw
        from mrsandbox.scan.run import _sidecar

        x, dwell_s, f0_mhz, header = _raw(str(source))
        # the BIDS sidecar, by inheritance, says what a NIfTI-MRS header may not (a header may name
        # the station where its sidecar names the model)
        stem = pathlib.Path(source).name.split(".")[0]
        sidecar = _sidecar(pathlib.Path(source), f"_{stem.rsplit('_', 1)[-1]}.json")
        said = json.loads(sidecar.read_text()) if sidecar else {}
        image = nib.load(str(source))
        card, was = self.protocol, {}
        if header.get("EchoTime"):
            card.te_ms = was["te_ms"] = round(1000.0 * float(header["EchoTime"]), 2)
        if header.get("RepetitionTime"):
            card.tr_ms = was["tr_ms"] = round(1000.0 * float(header["RepetitionTime"]), 1)
        card.averages = was["averages"] = int(x.shape[2])
        card.npoints = was["npoints"] = int(x.shape[0])
        card.bandwidth_hz = was["bandwidth_hz"] = round(1.0 / dwell_s, 1)
        name = " ".join(str(header.get(key, ""))
                        for key in ("SequenceName", "ProtocolName")).lower()
        for words, sequence in (("hermes", "HERMES GABA GSH"), ("mega", "MEGA-PRESS"),
                                ("laser", "sLASER"), ("steam", "STEAM"), ("press", "PRESS"),
                                ("svs_se", "PRESS")):
            if words in name:
                card.sequence = was["sequence"] = sequence
                break
        size = [float(v) for v in image.header.get_zooms()[:3]]
        if all(1.0 < s < 100.0 for s in size):
            self.size_mm = was["size_mm"] = size
        self.field_t = was["field_t"] = round(f0_mhz / 42.5774785, 3)
        vendor = " ".join(str(told.get("Manufacturer", "")) for told in (header, said)).lower()
        model = " ".join(str(told.get("ManufacturersModelName", ""))
                         for told in (header, said)).lower().replace(" ", "")
        for key, preset in PRESETS.items():
            if (preset.vendor and preset.vendor in vendor and preset.model
                    and preset.model.lower().replace(" ", "") in model):
                self.scanner = was["scanner"] = key
                self.field_t = (None if abs(preset.b0_t - f0_mhz / 42.5774785) < 0.05
                                else self.field_t)
                break
        else:
            self.scanner = was["scanner"] = "generic_3t"
        # the array as acquired: the coil the header or sidecar names, where a preset is that coil;
        # a 32-channel run on an unnamed coil the helmet the cohorts read such runs against
        named = str(header.get("ReceiveCoilName") or said.get("ReceiveCoilName") or "")
        preset = next((label for label, key in self.ARRAYS.items()
                       if named in ARRAY_PRESETS[key].names and ARRAY_PRESETS[key].vendor in vendor
                       and ARRAY_PRESETS[key].channels == x.shape[1]), None) if named else None
        if preset is not None:
            self.coil = was["array"] = preset
        elif x.shape[1] == 32 and len(self.array() or ()) != 32:
            self.coil = was["array"] = "head array"
        self.calibration = None
        return was

    def replicate(self, folder):
        """A folder of scans read at once, whatever the scanner gave (io.folder): every file told
        apart by what it is, raw spectroscopy brought to NIfTI-MRS with spec2nii, and each kind
        used for what it can give: the first run with its water reference calibrates the scanner
        (`calibrate`), a T1-weighted image becomes the anatomy (`segment`), a field map the B0 map,
        a B1 map the transmit map, coil maps the array, a noise scan the channels' own noise share,
        a Pulseq file the design, a saved scanner its calibration. Returns the report, [(file,
        kind, what it gave or why not)]."""
        from mrsandbox.io import folder as scans

        folder = pathlib.Path(folder)
        work = pathlib.Path.home() / "MRSandbox" / "converted" / folder.name
        found = scans.survey(folder)
        report = []
        folder = folder if folder.is_dir() else folder.parent
        if any(item.kind == "DICOM series" for item in found):
            from mrsandbox.io import read_dicom

            found = [item for item in found if item.kind != "DICOM series"]
            try:
                series = read_dicom(folder, work / "dicom")
                report.append((f"{folder.name} (DICOM)", "DICOM series",
                               f"{len(series)} image series converted with dcm2niix: "
                               + ", ".join(sorted(series))))
                found += [item for item in scans.survey(work / "dicom")
                          if item.kind not in ("unknown",)]
            except Exception as error:
                report.append((f"{folder.name} (DICOM)", "DICOM series",
                               f"not converted: {error}"[:160]))
            spectra = scans.Found(folder, "MRS run", note="spectroscopy DICOM through spec2nii")
            if scans.convert(spectra, work / "dicom_mrs") is not None:
                found.append(spectra)
        for item in found:
            if (item.kind in ("MRS run", "MRS reference")
                    and item.path.suffix.lower() not in (".nii", ".gz")):
                scans.convert(item, work, f0_mhz=42.5774785 * self.profile().scanner.b0_t)
        used, later = set(), []
        for run, reference in scans.pairs(found):
            source = run.converted or run.path
            if run.converted is None and run.path.suffix.lower() not in (".nii", ".gz"):
                continue
            with_reference = (reference.converted or reference.path if reference is not None
                              else None)
            used.update({id(run), id(reference)})
            # a lab/raw's noise label, kept beside it when converted
            label = work / (run.path.stem + "_noise.npy")
            try:
                # the card and the scanner as the run was acquired: its twin
                was = self.card_from(source)
                self.calibrate(source, folder.name, reference=with_reference,
                               noise=np.load(label) if label.exists() else None)
            except Exception as error:
                report.append((run.path.name, run.kind, f"not read: {error}"[:160]))
                continue
            receiver, in_vitro = self.profile().receiver, hasattr(self.body(), "contents")
            report.append((run.path.name, run.kind, "the card and the scanner as acquired ("
                           + ", ".join(f"{k} {v}" for k, v in was.items()) + "); "
                           + ("in vitro, the hardware alone: the channels' noise correlation, "
                              if in_vitro else "the scanner's noise level and line width, ")
                           + "its drift and jitters, "
                           + ("no receiver filter in the band" if receiver is None else
                              f"its receiver filter ({receiver.span_dwells} dwells a side)") + (
                f", and with {reference.path.name} the residual water, the receiver delay, the "
                "eddy transient, the leaked water" if reference and not in_vitro else "")))
            break                   # one run calibrates; the others are listed as read
        for item in found:
            if id(item) in used:
                continue
            try:
                if item.kind == "T1w":
                    self.segment(str(item.path), folder.name)
                    self.subject = folder.name      # the new anatomy on the floor
                    self.position_mm = ([0.0, 0.0, 0.0] if hasattr(self.body(), "contents")
                                        else list(self.position_mm))
                    self.home()
                    report.append((item.path.name, item.kind,
                                   "the anatomy, segmented, on the floor"))
                elif item.kind == "B0":
                    self.field = str(self.onto(str(item.path)))
                    report.append((item.path.name, item.kind, "the B0 map"))
                elif item.kind == "B1+":
                    self.transmit = str(self.onto(str(item.path)))
                    report.append((item.path.name, item.kind, "the transmit map"))
                elif item.kind == "coils":
                    self.coil_maps(str(item.path), item.path.name.split(".")[0])
                    report.append((item.path.name, item.kind, "the receive array's maps"))
                elif item.kind == "noise scan":
                    self.own_noise = float(CovarianceReading.from_ismrmrd(
                        str(item.path)).own_noise(self.array() or CoilArray.helmet()))
                    report.append((item.path.name, item.kind,
                                   f"the channels' own noise share, {self.own_noise:.2f}"))
                elif item.kind == "Pulseq":
                    self.adopt_seq(item.path)
                    report.append((item.path.name, item.kind, "the design, block by block"))
                elif item.kind == "scanner file":
                    self.load_scanner(item.path)
                    report.append((item.path.name, item.kind, "the scanner's calibration"))
                elif item.kind == "physio":
                    reading = PhysioReading.from_bids(item.path)
                    self.run_records += tuple(r for r in (
                        reading.record_breathing(folder.name) if reading.resp_hz else None,
                        reading.record_heart(folder.name) if reading.cardiac_hz else None)
                        if r is not None)
                    report.append((item.path.name, item.kind,
                                   f"the subject's breathing at {60 * (reading.resp_hz or 0):.0f} "
                                   f"and heart at {60 * (reading.cardiac_hz or 0):.0f} a minute"))
                elif item.kind == "image series":
                    reading = MotionReading.from_series(item.path)
                    self.run_records += (reading.record_events(folder.name),
                                         reading.record_size(folder.name))
                    report.append((item.path.name, item.kind,
                                   f"the head's motion: {reading.events.size} events in "
                                   f"{reading.minutes:.1f} minutes"))
                elif item.kind in ("MRS run", "MRS reference"):
                    report.append((item.path.name, item.kind,
                                   "read but not used: one run calibrates"
                                   if item.converted or item.path.suffix.lower() in (".nii", ".gz")
                                   else item.note))
                elif item.kind in ("T1 map", "R1 map", "T2 map", "R2 map"):
                    # read on the anatomy, once a T1w of the folder is on the floor
                    later.append(item)
                else:
                    report.append((item.path.name, item.kind, item.note))
            except Exception as error:
                report.append((item.path.name, item.kind, f"not read: {error}"[:160]))
        for item in later:
            # a quantitative map of the subject on the floor, read per tissue on the anatomy's
            # tissues (scan.maps): its records the subject's own, the water's T1 or T2 per
            # tissue "yours"; a rate map (R1, R2) is inverted on reading
            try:
                import nibabel as nib
                import nibabel.processing

                maps = getattr(self.body(), "maps", None)
                if maps is None:
                    raise ValueError("the anatomy on the floor has no tissue maps to read it on")
                onto = nibabel.processing.resample_from_to(
                    nib.load(str(item.path)), (maps.data.shape[:3], maps.affine), order=1)
                data = np.asarray(onto.dataobj, dtype=np.float64)
                data = (np.divide(1.0, data, out=np.zeros_like(data), where=data > 0.0)
                        if item.kind.startswith("R") else data)
                which = "T1" if item.kind in ("T1 map", "R1 map") else "T2"
                sidecar = item.path.parent / (item.path.name.split(".")[0] + ".json")
                reading = MapReading.from_arrays(
                    f"water.{which.lower()}_s", data,
                    {tissue: maps.data[..., k] for k, tissue in enumerate(("gm", "wm", "csf"))},
                    json.loads(sidecar.read_text()) if sidecar.exists() else {})
                records = reading.records(folder.name)
                self.run_records += records
                report.append((item.path.name, item.kind,
                               f"the water's {which} per tissue: " + ", ".join(
                    f"{r.leaf.split('.')[-1]} {r.value:.2f} s" for r in records) if records
                    else "no tissue with enough of its own cells"))
            except Exception as error:
                report.append((item.path.name, item.kind, f"not read: {error}"[:160]))
        self.uploads += [(str(pathlib.Path(folder) / file) if not pathlib.Path(file).is_absolute()
                          else file, kind, gave) for file, kind, gave in report]
        return report

    def uploads_for(self, leaf):
        """The files read this session that can give `leaf`, [(path, kind, what it gave)]."""
        from mrsandbox.io.folder import reads

        return [(path, kind, gave) for path, kind, gave in self.uploads if kind in reads(leaf)]

    def where_from(self):
        """The room as it stands, one row per leaf of the tree: (path, kind, numbers, source, from
        what), the value as `leaves` describes it, the source chosen (literature, pooled, dataset,
        own, custom; None for a leaf without sources of its own, which the package stated or typed
        from a table), and what it came from: the citation, the pooled rows' datasets, the
        dataset's subject, and for the operator's own the files read this session that give
        it."""
        sources = self.sources()
        rows = []
        for path, kind, numbers, changed in self.leaves():
            told = sources.get(path)
            source = "custom" if changed else told["chosen"] if told else None
            said = ("set by the operator" if changed
                    else told["provenance"].get(source, "") if told else "")
            if source == "own":
                said = "; ".join([said] + [pathlib.Path(file).name
                                           for file, _, _ in self.uploads_for(path)])
            rows.append((path, kind, numbers, source, said))
        return rows

    def reset_room(self):
        """The room back to the package's: every setting, choice and hardware number of the
        operator's gone, the scanner generic on the package's pooled rows, no dataset, no file, no
        reading of the operator's."""
        self.reset()
        self.choices, self.hardware, self.files, self.own_noise = {}, {}, {}, None
        self.field_t, self.calibration, self.person = None, None, {}
        if self.cohort is not None:
            self.pick_cohort(None)
        self.scanner = "generic_3t"

    def save_scanner(self, name):
        """Save the scanner on the floor as one file, ~/MRSandbox/profiles/<name>.json
        (scan.profile), which then stands in its calibrations; returns the path."""
        folder = pathlib.Path.home() / "MRSandbox" / "profiles"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.json"
        self.profile().save(path)
        return path

    def coil_maps(self, path, name):
        """Put a receive array of the operator's own on the shelf: measured sensitivity maps, one
        NIfTI of (nx, ny, nz, channels), complex or real (hardware.coils MappedCoilArray), in the
        scanner's coordinates."""
        import nibabel as nib

        image = nib.load(str(path))
        maps = np.asarray(image.dataobj)
        if maps.ndim != 4:
            raise ValueError("sensitivity maps are one volume per channel, (nx, ny, nz, "
                             f"channels); this file is {maps.shape}")
        self.coils[name] = lambda: MappedCoilArray(maps, np.array(image.affine, dtype=np.float64),
                                                   name=name)
        self.coil_settings[name] = {}
        self.coil = name

    def coil_noise(self, source):
        """Read the share of a channel's noise that is its own from a raw scan with the coils apart
        (NIfTI-MRS; scan.calibrate CovarianceReading, the FIDs' tails) against the array of loops
        chosen in the room; the exams then load the array with it in place of the package's 0.25.
        Returns the share."""
        from mrsandbox.scan.calibrate import CovarianceReading

        array = self.array()
        if array is None or not array.loops:
            raise ValueError("choose an array of loops first: its geometry is what the channels' "
                             "shared noise is held against")
        self.own_noise = float(CovarianceReading.from_nifti(str(source)).own_noise(array))
        return self.own_noise

    def borrow(self, what, source="qsm_rc2"):
        """Lend an anatomy of the operator's own a map of one of the package's heads: the QSM
        challenge head's B0 field or lipid layer, or BigBrain-MR's field, carried over by
        registering that head's own anatomical image (its R1 map) onto the operator's T1-weighted
        image, FSL flirt with twelve degrees of freedom, and playing the transform on the map. It
        is ANOTHER head's field and scalp in this head's shape: a fair stand-in where no map of the
        operator's exists, not a measurement of them. The transform is kept beside the anatomy's
        image and found again. Returns the path of the carried map, on the anatomy's grid."""
        import os
        import shutil
        import subprocess

        import nibabel as nib

        if self.subject not in self.images:
            raise ValueError("a map is carried over onto an anatomy of your own with its image; "
                             "the package's heads have their own maps")
        flirt = shutil.which("flirt") or shutil.which(
            "flirt", path=os.path.expandvars("$FSLDIR/bin") + os.pathsep
            + os.path.expanduser("~/fsl/share/fsl/bin"))
        if flirt is None:
            raise RuntimeError("FSL's flirt is needed to carry a map over (fsl.fmrib.ox.ac.uk)")
        environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get(
            "FSLDIR", str(pathlib.Path(flirt).parents[3]))}
        image = self.images[self.subject]
        folder = image.parent / "carried_over"
        folder.mkdir(exist_ok=True)
        transform = folder / f"{source}_to_this.mat"
        if not transform.exists():
            theirs = (Volume.load("qsm_rc2/r1") if source == "qsm_rc2"
                      else Volume.load("bigbrain_mr/t1map_800um"))
            shown = (theirs.data if source == "qsm_rc2"
                     else np.where(theirs.data > 0.0, 1.0 / np.maximum(theirs.data, 0.5), 0.0))
            nib.save(nib.Nifti1Image(np.asarray(shown, dtype=np.float32), theirs.affine),
                     str(folder / f"{source}_anatomy.nii.gz"))
            subprocess.run([flirt, "-in", str(folder / f"{source}_anatomy"), "-ref", str(image),
                            "-omat", str(transform), "-dof", "12", "-cost", "normmi",
                            "-searchrx", "-30", "30", "-searchry", "-30", "30",
                            "-searchrz", "-30", "30"],
                           check=True, env=environment)
        if what == "B0":
            field = FieldMap.load(source, b0_t=self.profile().scanner.b0_t)
            moving = nib.Nifti1Image(np.asarray(field.volume.data, dtype=np.float32),
                                     field.volume.affine)
        else:
            fat = Brain.load("qsm_rc2").fat
            moving = nib.Nifti1Image(np.asarray(fat.data, dtype=np.float32), fat.affine)
        nib.save(moving, str(folder / f"{source}_{what}.nii.gz"))
        out = folder / f"{source}_{what}_on_this.nii.gz"
        subprocess.run([flirt, "-in", str(folder / f"{source}_{what}"), "-ref", str(image),
                        "-applyxfm", "-init", str(transform), "-out", str(out)],
                       check=True, env=environment)
        return self.onto(out)

    def register(self, path, reference=None):
        """Bring a map of ANOTHER session onto the subject on the table: FSL flirt, rigid (six
        degrees of freedom, normalized mutual information), of the map's own image onto the
        subject's picture, then resampled onto its grid. Right for a map that shows anatomy (a
        lipid or tissue map, a B1+ magnitude, an anatomical image); a field map in Hz shows none,
        and wants its MAGNITUDE image registered and the transform applied, which is not built. A
        map of the same session needs none of this (`onto`). `reference` is the image to register
        onto where it is not the picture of what lies on the table (an anatomy still being put
        together). Returns the registered NIfTI's path."""
        import os
        import shutil
        import subprocess

        import nibabel as nib

        flirt = shutil.which("flirt") or shutil.which(
            "flirt", path=os.path.expandvars("$FSLDIR/bin") + os.pathsep
            + os.path.expanduser("~/fsl/share/fsl/bin"))
        if flirt is None:
            raise RuntimeError("FSL's flirt is needed to register a map (fsl.fmrib.ox.ac.uk)")
        if reference is None:
            reference = pathlib.Path(path).parent / "registration_reference.nii.gz"
            nib.save(nib.Nifti1Image(self.picture().astype(np.float32), self.body().affine),
                     str(reference))
        out = pathlib.Path(str(path).replace(".nii", "_registered.nii", 1))
        subprocess.run([flirt, "-in", str(path), "-ref", str(reference), "-out", str(out),
                        "-dof", "6", "-cost", "normmi"], check=True,
                       env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get(
                           "FSLDIR", str(pathlib.Path(flirt).parents[3]))})
        return out if out.exists() else pathlib.Path(str(out) + ".gz")

    def scanners(self):
        """The top bar's scanners, {key: name}: the generic system first (its field the number
        beside it), then every specific system (`systems`)."""
        return {"generic_3t": "Generic", **self.systems()}

    @staticmethod
    def systems():
        """The specific systems the scanner builder offers, {key: name}, the package's presets
        with a vendor."""
        return dict(sorted(((name, label) for name, label in SCANNERS.items()
                            if name in PRESETS and PRESETS[name].vendor),
                           key=lambda item: item[1]))

    def calibrations(self):
        """What can calibrate the scanner on the floor in place of the package's pooled rows,
        {key: what it is}: one of the package's cohorts that holds rows of its systems (that
        dataset's rows alone, scan.cohort holding), and the operator's own profiles of this
        scanner, read from their scans and saved as JSON (scan.profile) in ./data/profiles or
        ~/MRSandbox/profiles, which stay on this machine."""
        found = {name: self.datasets()[name] for name in holding(PRESETS[self.scanner])}
        found.update(self.files)
        for folder in (pathlib.Path.cwd() / "data" / "profiles",
                       pathlib.Path.home() / "MRSandbox" / "profiles"):
            for path in sorted(folder.glob("*.json")):
                if json.loads(path.read_text())["scanner"]["name"] == self.scanner:
                    found[str(path)] = ("read from your own scans: "
                                        + path.stem.replace("_private", "").replace("_", " "))
        return found

    def member(self):
        """The subject of the cohort on the floor that the session's seed draws (scan.cohort), None
        without a cohort."""
        if self.cohort is None:
            return None
        return next(cohort for cohort in package() if cohort.name == self.cohort).draw(self.seed)

    def admit(self, age_y=None, sex=None, health=None):
        """Admit the person on the table (`person`) as a scanner's patient registration does: their
        age in years, sex ("F", "M") and health ("healthy" or the condition by name); what is
        not given is not asked, so nothing given is a healthy adult. A template brain on the
        table becomes the one of the person's age: a child's band, an adult's average."""
        self.person = {key: value for key, value in
                       dict(age_y=age_y, sex=sex, health=health).items() if value is not None}
        average = "Brain, ICBM152 average, 2 mm"
        if self.subject == average or self.subject.startswith("Brain, NIHPD "):
            band = nihpd_band(age_y)
            self.subject = average if band is None else f"Brain, NIHPD {band} years, 2 mm"

    def anatomies(self):
        """The subject picker's entries: every subject but the pediatric templates, which a
        registration picks by the person's age (the one on the table is listed)."""
        return [name for name in self.subjects
                if not name.startswith("Brain, NIHPD ") or name == self.subject]

    def arrays(self):
        """The receive array picker's entries, {key: label}: the generic arrays say so, as the
        generic scanner does; the vendors' arrays carry their maker."""
        return {key: key if key == "one ideal channel" or key in self.ARRAYS else f"Generic · {key}"
                for key in self.coils}

    def pick_cohort(self, name):
        """Put a package cohort on the floor, or none: its subject's scanner (where it is a
        preset), voxel place and size come with it, and every exam scans that subject's own
        readings, fixed; a new draw of the population (the seed) takes another subject. "None"
        puts back what stood on the floor before the cohort."""
        if name is None and self.cohort is not None:
            (self.scanner, self.field_t, self.calibration, self.position_mm, self.size_mm,
             self.view_mm) = self._before_cohort
        elif name is not None and self.cohort is None:
            self._before_cohort = (self.scanner, self.field_t, self.calibration,
                                   list(self.position_mm), list(self.size_mm), list(self.view_mm))
        self.cohort = name
        member = self.member()
        if member is not None:
            if member.profile.scanner.name in PRESETS:
                self.scanner = member.profile.scanner.name
            self.position_mm, self.size_mm = list(member.position_mm), list(member.size_mm)
            self.view_mm = list(self.position_mm)
            self.field_t, self.calibration = None, None

    @staticmethod
    def datasets():
        """The package's datasets, {cohort name: label}, every label the same shape (Cohort.label),
        made once."""
        return _datasets()

    @staticmethod
    def catalogue():
        """The measured database as the operator browses it, one row per cohort: its name, its
        label, the subjects, readings and failed readings it holds, the leaves it reads, where
        it comes from, and whether it is the operator's own (~/MRSandbox/cohorts)."""
        from mrsandbox.scan import cohort as cohorts

        rows = []
        for cohort in package():
            records = [r for m in cohort.members for r in m.profile.records]
            source = records[0].source if records else {}
            rows.append({"name": cohort.name,
                         "dataset": cohort.label(COHORTS.get(cohort.name, cohort.name)),
                         "subjects": len(cohort.members), "readings": len(records),
                         "failed": sum(1 for r in records if r.failed),
                         "leaves": ", ".join(sorted({r.leaf for r in records})),
                         "source": source.get("dataset") or "",
                         "citation": source.get("citation") or "", "url": source.get("url") or "",
                         "yours": not (cohorts.COHORTS / f"{cohort.name}.json").exists()})
        return rows

    def parameter(self, leaf):
        """One leaf of the measured database against age: every subject's reading that did not
        fail as {"dataset", "method", "subject", "age_y", "value", "share"}, the share that
        dataset takes in this room's pool (none where the room's conditions leave it out; a
        range of ages stands at its middle, an open one or none at None); and this room's
        literature and pooled value, each (5th, 50th, 95th percentile), None where the room
        has none."""
        profile, array, member = self.profile(), self.array(), self.member()
        record = calibrations(profile.scanner, profile.records,
                              channels=None if array is None else len(array),
                              subject=(member.traits if member is not None else self.person)
                              or None).get(leaf)
        share = record.measured.get("dataset_share", {}) if record is not None else {}
        points = []
        for cohort in package():
            for person in cohort.members:
                age = person.traits.get("age_y")
                if isinstance(age, tuple):
                    age = None if None in age else 0.5 * (age[0] + age[1])
                points += [{"dataset": cohort.name, "method": r.sequence.get("family") or "",
                            "subject": person.subject, "age_y": age, "value": float(r.value),
                            "share": share.get(cohort.name)}
                           for r in person.profile.records if r.leaf == leaf and not r.failed]
        sources = self.twin(params=Params({})).params.sources()
        path = leaf if leaf in sources else leaf.rpartition(".")[0]
        literature = sources[path].literature if path in sources else None
        if literature is not None and path != leaf:
            literature = getattr(literature, leaf.rpartition(".")[2])
        if literature is not None:
            rng = np.random.default_rng(0)
            literature = tuple(float(v) for v in np.percentile(
                [float(literature.sample(rng)) for _ in range(2000)], (5.0, 50.0, 95.0)))
        return {"points": points, "literature": literature,
                "pooled": None if record is None or "dataset_share" not in record.measured
                else (record.prior[0], record.value, record.prior[1])}

    def cohort_says(self):
        """One sentence on the dataset on the floor and what it sets, empty without one."""
        member = self.member()
        if member is None:
            return ""
        return (f"subject {member.subject}: its scanner, voxel and "
                f"{len(member.fixed().records)} readings; a new draw of the population "
                "takes another subject")

    def choose_all(self, paths, source):
        """Choose `source` for every leaf of `paths` that has it (the card's "for all lines"): the
        leaf's own setting goes, and "custom" keeps what is set."""
        sourced = self.twin(params=Params({})).params.sources()
        for path in paths:
            if (source in ("literature", "pooled", "mixed", "dataset", "own") and path in sourced
                    and source in sourced[path].available):
                self.overrides.pop(path, None)
                self.choices[path] = source

    def profile(self):
        """The scanner on the floor with its records: the cohort's subject with its own readings
        fixed, or the preset on the package's pooled rows, or calibrated by one cohort's rows, or
        a saved profile; at the operator's field where one is set (its limits and records stay,
        and a record names the departure)."""
        member = self.member()
        if member is not None:
            profile = member.fixed()
        elif self.calibration in self.calibrations():
            profile = (ScannerProfile.load(self.calibration) if self.calibration.endswith(".json")
                       else ScannerProfile.preset(self.scanner, cohort=self.calibration))
        else:
            profile = ScannerProfile.preset(self.scanner)
        if self.field_t is not None:
            profile = dataclasses.replace(profile, scanner=dataclasses.replace(
                profile.scanner, b0_t=float(self.field_t)))
        if self.hardware:
            profile = dataclasses.replace(profile, scanner=dataclasses.replace(
                profile.scanner, **self.hardware))
        return profile

    def calibrate(self, source, name, folder=None, reference=None, noise=None):
        """Calibrate the scanner on the floor from one raw acquisition of the operator's own
        (scan.profile ScannerProfile.read): `source` is NIfTI-MRS with the transients apart (with
        the coils apart too and an array chosen in the room, the array's own noise share is read as
        well), and the card, the voxel, and the room as they stand are its virtual twin, so set
        them as the scan was acquired first. The noise and the line broadening become records of a
        profile saved under `name` in ~/MRSandbox/profiles, which stays on this machine and is then
        this scanner's calibration; returns its path. With `reference`, the same voxel's water
        reference file, the residual water and the receiver delay are read as well; `noise` is
        the receiver's own noise samples (samples, coils), read where the FIDs still ring. A
        phantom on the table makes the run in vitro: the hardware alone is read (ScannerProfile
        .read)."""
        # the readers' twin plays ideal pulses, as the cohorts' did
        card = dataclasses.replace(self.protocol, real=False)
        twin = self.twin(card=card, params=self.params() | CALM).voxel(
            tuple(self.position_mm), tuple(self.size_mm), compiled=True, turn=self.turn)
        folder = (pathlib.Path(folder) if folder is not None
                  else pathlib.Path.home() / "MRSandbox" / "profiles")
        folder.mkdir(parents=True, exist_ok=True)
        import nibabel as nib

        # a file with its coils apart, and an array in the room standing in for the coil's
        # geometry: the share of the channels' noise that is their own is read too, and the
        # profile carries the array so loaded
        tags = json.loads(nib.load(str(source)).header.extensions[0].get_content())
        array = (self.array() if "DIM_COIL" in (tags.get("dim_5"), tags.get("dim_6"),
                                                tags.get("dim_7")) else None)
        in_vitro = hasattr(self.body(), "contents")
        ScannerProfile.read(str(source), twin, dataset=name, array=array,
                            load="a phantom" if in_vitro else "a head", in_vitro=in_vitro,
                            noise=noise, reference=None if reference is None else str(reference),
                            # a semi-LASER twin is fast enough to render three seeds
                            seeds=3 if "LASER" in card.sequence.upper() else None).save(
                                folder / f"{name}.json")
        self.calibration = str(folder / f"{name}.json")
        return folder / f"{name}.json"

    #**************#
    #   sequence   #
    #**************#
    def sequence(self, card=None):
        """The design of a card: one of the library's at the card's timings, or the
        operator's own, the blocks of `design` in order ("Custom")."""
        card = card if card is not None else self.protocol
        card = (dataclasses.replace(card, sequence=REAL[card.sequence])
                if card.real and card.sequence in REAL else card)
        if card.sequence in EDITING:
            return next(iter(self.editing(card).values()))
        if card.sequence == MRSI:
            return SEQUENCES["PRESS"](card.te_ms / 1000.0, card.tr_ms / 1000.0, None, None, None,
                                      None)
        scanner = self.profile().scanner
        # what the design depends on
        key = (dataclasses.astuple(card), self.condition, repr(self.designs), scanner.b0_t,
               scanner.peak_b1_ut)
        if self._sequence_memo[0] == key:
            return self._sequence_memo[1]
        self._sequence_memo = (key, self._sequence(card, scanner))
        return self._sequence_memo[1]

    def _sequence(self, card, scanner):
        if card.sequence == "Custom":
            design = self.turned(self.own(card.tr_ms / 1000.0), card.axes)
        else:
            design = self.turned(self.fitted(lambda fit: SEQUENCES[card.sequence](
                card.te_ms / 1000.0, card.tr_ms / 1000.0, 42.5774785 * scanner.b0_t,
                self.edges(card.axes), fit, "xyz")), card.axes)
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
        """The voxel's edges in the order a design's first, second and third slice pulse will
        select them: the designs are made along x, y, z and then turned (`turned`), so the first
        pulse gets the edge of axes[0]."""
        return tuple(float(self.size_mm["xyz".index(axis)]) for axis in axes)

    @staticmethod
    def turned(design, axes):
        """The same design with its slice order as the operator wants it, "yzx" say: what it plays
        along x (its first slice pulse and that pulse's gradients), it plays along axes[0], and so
        on, for every design alike: PRESS and sLASER (the 90, then the two pairs), STEAM (its three
        90s), the edited ones, and an own design of slice-selective pulses and crushers. Every
        event is copied once, so a pulse played four times stays one object and is computed
        once."""
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
        """A real design fitted to the scanner's transmitter, as a vendor fits its pulses:
        `build(fit)` makes the design (or the sub-experiments of an edited one), and where its
        amplitude-modulated 90s or 180s ask for more B1 than the scanner has, they are played
        longer by that factor at the same time-bandwidth product, so narrower, and displace a
        shift more. `fit(cls)` is the design's class with its pulse durations grown; `fit.grow`
        the factors by flip angle. An adiabatic full passage whose peak exceeds the scanner's is
        played at the scanner's peak and longer by the square of the ratio, which keeps its
        adiabaticity (peak squared times duration over bandwidth) and with it the inversion. A
        design whose timings do not hold the longer pulses stays as it was and says so among the
        card's problems."""
        limit = 42.5774785 * self.profile().scanner.peak_b1_ut

        def fit(cls, grow=None):
            grow = fit.grow if grow is None else grow
            longer = {name: getattr(cls, name) * grow[flip]
                      for name, flip in (("PULSE_S", 90), ("EXCITATION_S", 90),
                                         ("REFOCUSING_S", 180))
                      if hasattr(cls, name) and grow[flip] > 1.0}
            if fit.adiabatic and getattr(cls, "AFP_PEAK_HZ", 0.0) > limit:
                longer.update(AFP_PEAK_HZ=0.99 * limit,
                              AFP_S=cls.AFP_S * (cls.AFP_PEAK_HZ / (0.99 * limit)) ** 2)
            return type(cls.__name__, (cls,), longer) if longer else cls

        # the probe is the design as it is typed
        fit.grow, fit.adiabatic, self.unfitted = {90: 1.0, 180: 1.0}, False, ""
        probe = build(fit)
        for design in (probe.values() if isinstance(probe, dict) else (probe,)):
            for event in design.events:
                if (isinstance(event, ShapedPulse) and event.peak_hz is not None
                        and event.flip_deg is not None and float(event.flip_deg) in (90.0, 180.0)):
                    fit.grow[int(event.flip_deg)] = max(fit.grow[int(event.flip_deg)],
                                                        1.02 * float(event.peak_hz) / limit)
        fit.grow = {flip: factor if factor > 1.02 else 1.0 for flip, factor in fit.grow.items()}
        fit.adiabatic = any(isinstance(event, ShapedPulse) and event.flip_deg is None
                            and event.peak_hz is not None and float(event.peak_hz) > limit
                            for design in (probe.values() if isinstance(probe, dict) else (probe,))
                            for event in design.events)
        if max(fit.grow.values()) == 1.0 and not fit.adiabatic:
            return probe
        try:
            return build(fit)
        except ValueError as error:
            self.unfitted = (f"fitted to this scanner's {self.profile().scanner.peak_b1_ut:g} uT "
                             f"the 90s would last {fit.grow[90]:.2f} and the 180s "
                             f"{fit.grow[180]:.2f} times as long, which these timings do not hold "
                             f"({error}): a longer TE may")
            return probe

    def own(self, tr_s=None, condition=None):
        """The operator's own design (the sub-acquisition shown, or the one named), its blocks in
        order, as a Sequence."""
        blocks = self.designs[self.condition if condition is None else condition]
        return Sequence(*(event for kind, settings in blocks
                          for event in BLOCKS[kind][1](settings)), tr_s=tr_s)

    def phase_cycle(self, block, steps):
        """Make the design shown a phase cycle of `steps` sub-acquisitions (2 or 4): the pulse at
        `block` steps its phase through the circle, and the receiver follows as the pulse's role
        asks, so the wanted pathway adds and the others cancel over the cycle. An excitation (a
        flip up to 135 degrees) takes the receiver with it, exp(-i phi); a refocusing pulse is
        EXORCYCLE (Bodenhausen, Freeman, Turner 1977), the receiver at exp(-2 i phi): +, -, +, -
        over 0, 90, 180, 270. The transients then take turns through the cycle, as on a scanner.
        Coherence filters of the design do the same job ideally: take them out to see the cycle
        work."""
        blocks = self.designs[self.condition]
        kind, settings = blocks[block]
        if "phase_deg" not in settings:
            raise ValueError("a phase cycle steps a pulse's phase: choose a pulse")
        order = 1 if float(settings.get("flip_deg", 90.0)) <= 135.0 else 2
        self.designs, weights = {}, {}
        for step in range(steps):
            phi = 360.0 * step / steps
            name = chr(65 + step)
            self.designs[name] = [(k, dict(v, phase_deg=float(v["phase_deg"]) + phi)
                                   if n == block else dict(v)) for n, (k, v) in enumerate(blocks)]
            weight = np.exp(-1j * order * np.radians(phi))
            weights[name] = complex(np.round(weight.real, 12), np.round(weight.imag, 12))
        self.condition, self.weights = "A", {"cycled": weights, "first step alone": {"A": 1.0}}

    def combinations(self):
        """How the own design's sub-acquisitions combine: the operator's `weights` where they name
        sub-acquisitions that exist, else their plain sum."""
        kept = {name: {c: (complex(w) if complex(w).imag else float(complex(w).real))
                       for c, w in mix.items() if c in self.designs and w}
                for name, mix in self.weights.items()}
        kept = {name: mix for name, mix in kept.items() if mix}
        return kept or {"sum": {name: 1.0 for name in self.designs}}

    def adopt(self):
        """Make the own design the sequence the card plays now, block by block, so the designer
        opens on what the diagram shows: ideal pulses, delays, filters, and crushers over the 20 mm
        ensemble become blocks one can set; every other event (a slice-selective or adiabatic pulse
        of a real design, a projection) comes over as it is, a shaped pulse with its flip,
        duration, carrier, phase, peak B1 and slice gradient as settings (its shape stays). The
        card's own water suppression and OVS stay its switches; a design may hold a water
        suppression block of its own."""
        card = self.protocol
        skip = (sum(len(prelude.events) for prelude in self.preludes(card).values())
                if REAL.get(card.sequence, card.sequence) in (*SEQUENCES, *EDITING, "Custom")
                else 0)
        self.design = self._blocks(self.sequence().events[skip:])

    def adopt_seq(self, path):
        """Make the own design a Pulseq file's sequence (io.read_seq: the pulses, delays and
        gradients up to its first readout), block by block as `adopt` does; the card plays it as
        "Custom"."""
        self.design = self._blocks(read_seq(str(path)).events)
        self.protocol.sequence = "Custom"

    @staticmethod
    def _blocks(events):
        design = []
        for event in events:
            if type(event) is Pulse:
                design.append(("hard pulse", dict(flip_deg=float(event.flip_deg),
                                                  phase_deg=float(event.phase_deg))))
            elif isinstance(event, Delay):
                design.append(("delay", dict(ms=1000.0 * float(event.t_s))))
            elif isinstance(event, CoherenceFilter) and len(event.orders) == 1:
                design.append(("coherence filter", dict(order=int(event.orders[0]),
                                                        cycles_per_mm=float(event.crusher_per_mm))))
            elif isinstance(event, Gradient) and float(event.extent_mm) == 20.0:
                design.append(("crusher", dict(axis=event.axis,
                                               cycles_per_mm=float(event.moment_per_mm))))
            else:
                knobs = {}
                if isinstance(event, Gradient):
                    knobs = {"cycles_per_mm": float(event.moment_per_mm)}
                if isinstance(event, ShapedPulse):
                    # one amplitude knob: the flip where the pulse is flip calibrated, the peak B1
                    # where it is adiabatic
                    knobs = {key: float(value) for key, value in (
                        ("flip_deg", event.flip_deg),
                        ("duration_ms", 1000.0 * float(event.duration_s)),
                        ("offset_ppm", event.offset_ppm), ("phase_deg", event.phase_deg),
                        ("peak_hz", event.peak_hz if event.flip_deg is None else None),
                        ("slice_hz_per_mm",
                         event.gradient_hz_per_mm if event.gradient_hz_per_mm else None))
                        if value is not None}
                design.append((FIXED, dict(what=named(event), event=event, **knobs)))
        return design

    def slabs(self, card=None):
        """The outer-volume suppression slabs of a card that are on, [(axis, centre, thickness)]
        in mm from the voxel's centre: a pair along every axis, each `ovs_mm` thick and
        `ovs_gap_mm` clear of the voxel's face unless the slab has a gap or thickness of its own
        (`ovs_slabs`). The gap matters: the slabs are placed for water, and under their 3.7 kHz
        pulse (122 Hz per mm on a 30 mm slab) NAA's slab lies 2.8 mm nearer on one side, the
        lipids' 3.5 mm, with a transition band of about a millimetre more. A tilted slab carries
        its unit normal fourth, its axis tipped toward the other two by its tilt, the centre
        along the normal; the voxel then plays as a whole (Sequence.unfolded)."""
        card = self.protocol if card is None else card
        found = []
        for slab in card.ovs_slabs:
            if not slab["on"]:
                continue
            gap = card.ovs_gap_mm if slab["gap_mm"] is None else slab["gap_mm"]
            thickness = card.ovs_mm if slab["mm"] is None else slab["mm"]
            n = "xyz".index(slab["axis"])
            normal, others = np.eye(3)[n], [k for k in range(3) if k != n]
            # a protocol saved before slabs tilted has no tilt
            normal[others] = np.tan(np.radians(slab.get("tilt_deg", (0.0, 0.0))))
            found.append((slab["axis"], slab["side"] * (0.5 * (self.size_mm[n] + thickness) + gap),
                          float(thickness))
                         + ((tuple(normal / np.linalg.norm(normal)),) if normal[others].any()
                            else ()))
        return found

    def set_slab(self, axis, side, **settings):
        """One of the card's slabs changed ("on", "gap_mm", "mm", "tilt_deg"; None puts a gap or
        thickness back to the card's): the tuple is replaced, so a planned exam keeps its own."""
        self.protocol.ovs_slabs = tuple(
            {**slab, **settings} if (slab["axis"], slab["side"]) == (axis, side) else slab
            for slab in self.protocol.ovs_slabs)

    def at(self, plane, column, row):
        """The world point of (column, row) in a plane's picture, (x, y, z) in mm, the
        coordinate the plane does not show being the slice on view."""
        body = self.body()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = body.gm.shape
        side = max(nx, ny, nz)
        # the planes are centred in squares of the largest grid side (`planes`)
        px, py, pz = ((side - n) // 2 for n in (nx, ny, nz))
        index = {"tra": (column - px, ny - 1 - (row - py), None),
                 "sag": (None, column - py, nz - 1 - (row - pz)),
                 "cor": (column - px, None, nz - 1 - (row - pz))}[plane]
        return [self.view_mm[axis] if value is None else float(origin[axis] + step[axis] * value)
                for axis, value in enumerate(index)]

    def designed(self):
        """The own design as the card plays it: the card's preludes (its water suppression and
        OVS) first, then the design's blocks; what the designer draws."""
        design = self.own(self.protocol.tr_ms / 1000.0)
        for prelude in reversed(self.preludes(self.protocol).values()):
            design = prelude + design
        return design

    def preludes(self, card):
        """What a timed card plays before its localization, in order, {name: design}."""
        found = {}
        if card.water_suppression in SUPPRESSION:
            found[card.water_suppression] = SUPPRESSION[card.water_suppression](real=card.real)
        if card.ovs:
            scanner = self.profile().scanner
            # the slab pulse played longer where the transmitter is short of its peak
            probe = ShapedPulse.slr(90.0, 0.003, OuterVolumeSuppression.TIME_BANDWIDTH, "excitation")
            grow = max(1.0, probe.played_peak_hz / (42.5774785 * scanner.peak_b1_ut))
            found["OVS"] = OuterVolumeSuppression(self.slabs(card), 42.5774785 * scanner.b0_t,
                                                  duration_s=0.003 * grow,
                                                  extent_mm=float(max(self.size_mm)))
        return found

    @property
    def turn(self):
        """The voxel's rotation from its own frame to the world's, or None while it stands
        square."""
        return turning(self.turn_deg) if any(self.turn_deg) else None

    def displacement(self, ppm):
        """Where the voxel of spins at `ppm` lies from the planned one, along the voxel's own
        (x, y, z) axes in mm (turned with it, `turn`): under a
        slice gradient of G Hz per mm a pulse centred for its reference selects a shift d ppm away
        at -d f0 / G. Zero along an axis no played pulse selects (an ideal design: all three). The
        card's preludes do not count: an OVS slab's carrier places the slab, it is no reference."""
        f0_mhz = 42.5774785 * self.profile().scanner.b0_t
        moved = [0.0, 0.0, 0.0]
        card = self.protocol
        skip = (sum(len(prelude.events) for prelude in self.preludes(card).values())
                if REAL.get(card.sequence, card.sequence) in (*SEQUENCES, *EDITING, "Custom")
                else 0)
        for event in self.sequence().events[skip:]:
            if isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
                shift = -(ppm - float(event.offset_ppm)) * f0_mhz / float(event.gradient_hz_per_mm)
                axis = "xyz".index(event.gradient_axis)
                moved[axis] = shift if abs(shift) > abs(moved[axis]) else moved[axis]
        return moved

    def editing(self, card):
        """The sub-experiments of an edited card, {name: design}, for this room's field and
        voxel."""
        card = (dataclasses.replace(card, sequence=REAL[card.sequence])
                if card.real and card.sequence in REAL else card)
        conditions = self.fitted(lambda fit: EDITING[card.sequence][0](
            card, card.tr_ms / 1000.0, 42.5774785 * self.profile().scanner.b0_t,
            self.edges(card.axes), fit))
        conditions = {name: self.turned(design, card.axes) for name, design in conditions.items()}
        # water suppression and OVS before every sub-experiment
        for prelude in reversed(self.preludes(card).values()):
            conditions = {name: prelude + design for name, design in conditions.items()}
        return conditions

    def diagram(self, sequence=None, readout_ms=12.0, preludes=None):
        """A design as a pulse sequence diagram, times in ms: {"rf": [(t, height)] a trace
        of the shaped pulses' envelopes (a real envelope keeps its lobes' signs), "bars":
        [(t, height, selective)] the ideal pulses, which take no time, their flip over 180 and
        whether they act on a band alone (a suppression or editing pulse), "pulses": [(t, text)]
        every pulse's centre and flip, "marks": [(t, text)] for what takes no time and has no
        height (coherence filters), "x", "y", "z": [(t, mT/m)] the gradients as the scanner
        would play them (sequences.timeline), "spans": [(start, end)] of every event, "crushers":
        [(start, end, axis)] the spans that are crushers and no slice selection, "groups":
        [(start, end, name)] the card's preludes (water suppression, OVS), "adc": (start, end)}.
        `preludes` names those a given `sequence` opens with (the card's own where it is not
        given)."""
        if preludes is None:
            preludes = (self.preludes(self.protocol) if sequence is None
                        and self.protocol.sequence in (*SEQUENCES, *EDITING, "Custom") else {})
        sequence = sequence if sequence is not None else self.sequence()
        spans = [list(span) for span in event_spans(sequence)]
        # a prelude without duration (an ideal CHESS: selective pulses and filters that take no
        # time) is laid out over a nominal stretch on paper, its events in order, and what follows
        # moves on by it
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
            begin = next((1000.0 * at for at, which in held if which == name),
                         1000.0 * spans[first][0])
            groups.append((begin, begin + 30.0 if any(which == name for _, which in held)
                           else 1000.0 * spans[first + len(prelude.events) - 1][1],
                           name if not any(which == name for _, which in held)
                           else name + ", ideal"))
            first += len(prelude.events)
        for event, (start, stop) in zip(sequence.events, spans):
            start, stop = 1000.0 * start, 1000.0 * stop
            if isinstance(event, ShapedPulse):
                envelope = np.asarray(event.envelope)
                envelope = np.abs(envelope) if np.iscomplexobj(envelope) else envelope.astype(float)
                height = 1.0 if event.flip_deg is None else float(event.flip_deg) / 180.0
                times = np.linspace(start, stop, envelope.size)
                rf += ([(start, 0.0)] + [(float(a), float(height * b / np.abs(envelope).max()))
                                         for a, b in zip(times, envelope)] + [(stop, 0.0)])
                pulses.append((0.5 * (start + stop),
                               "" if event.flip_deg is None else f"{float(event.flip_deg):.0f}°"))
            elif isinstance(event, CoherenceFilter):
                marks.append((start,
                              f"p = {event.orders if hasattr(event, 'orders') else ''}".strip()))
            elif isinstance(event, Gradient):
                crushers.append((start, stop, event.axis))
            elif hasattr(event, "flip_deg") and not isinstance(event, Delay):
                bars.append((start, float(event.flip_deg) / 180.0,
                             isinstance(event, SelectivePulse)))   # no time in the engine
                pulses.append((start, f"{float(event.flip_deg):.0f}°"))
        rf.append((end_ms + readout_ms, 0.0))
        until_s = (end_ms + readout_ms) / 1000.0
        times, waves = gradient_waveforms(sequence, self.profile().scanner, 5e-5,
                                          until_s - 0.030 * len(held))
        for at, _ in held:
            times = np.where(times >= at, times + 0.030, times)
        lanes = {axis: [(float(a), float(b)) for a, b in zip(
                     1000.0 * times[::4], waves.get(axis, np.zeros(times.size))[::4])]
                 for axis in "xyz"}
        return {"rf": rf, "bars": bars, "pulses": pulses, "marks": marks, **lanes,
                "crushers": crushers, "groups": groups, "adc": (end_ms, end_ms + readout_ms),
                "spans": [(1000.0 * a, 1000.0 * b) for a, b in spans]}

    #************#
    #   leaves   #
    #************#
    def twin(self, sequence=None, params=None, card=None, profile=None):
        """A VirtualScan of the room as it stands: the profile on the floor (or `profile`), the
        body, the card's sequence (or `sequence`), the operator's settings (or `params`) and
        the card's fields, with the operator's source choices applied to its tree (`choices`:
        a leaf's literature or measured source in place of the default's)."""
        profile = profile if profile is not None else self.profile()
        if self.run_records:
            profile = dataclasses.replace(profile, records=profile.records + self.run_records)
        member = self.member()
        scan = profile.scan(self.body(), sequence if sequence is not None else self.sequence(card),
                            params=params if params is not None else self.params(),
                            subject=(member.traits if member is not None else self.person) or None,
                            **self.fields(card))
        if self.choices:
            sourced = scan.params.sources()
            # a leaf without that source keeps its own
            scan.params = scan.params.select(**{
                path: source for path, source in self.choices.items()
                if path in sourced and source in sourced[path].available})
        return scan

    def sources(self):
        """Every leaf of the tree that carries values per source, {path: {"chosen": source,
        "sources": {source: (kind, numbers)}, "provenance": {source: text}}} in the tree before the
        operator's settings, the choices applied (Session.leaves describes kind and numbers the
        same way)."""
        return {path: {"chosen": leaf.chosen,
                       "sources": {name: _describe(getattr(leaf, name)) for name in leaf.available},
                       "provenance": dict(leaf.provenance)}
                for path, leaf in self.twin(params=Params({})).params.sources().items()}

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
        high), "normal" with (mean, sd), "empirical" with (median, 5th, 95th, subjects), or
        "other" with the prior's own text (a tissue record, a map), which the console shows
        and does not edit. Without `own` the
        tree is the one before the operator's settings: the priors as their sources give them."""
        sequence = self.sequence()
        found = []

        def walk(node, path):
            for key, value in node.items():
                here = path + (key,)
                if isinstance(value, Params):
                    walk(value, here)
                else:
                    found.append((".".join(here), *_describe(value),
                                  ".".join(here) in self.overrides))

        walk(self.twin(sequence, params=self.params() if own else Params({})).params, ())
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
        for an ideal design), the "shim" box where the operator set one, and the "ovs ..."
        saturation slabs of a card that plays them; and last the world corners of the square
        picture, [[top left, top right], [bottom left, bottom right]] in mm, for a view that hangs
        the three planes in space."""
        body, tissue, wash = self.body(), self.picture(), self.wash()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = tissue.shape
        # the voxel's centre, the outline's
        i, j, k = (np.asarray(self.position_mm) - origin) / step
        a, b, c = (int(min(max(round(v), 0), n - 1))
                   for v, n in zip((np.asarray(self.view_mm) - origin) / step, (nx, ny, nz)))
        half = 0.5 * np.asarray(self.size_mm) / step
        if self.protocol.sequence == MRSI:  # the slice's whole grid in the plane, one cell thick
            half = half * np.array([self.protocol.matrix, self.protocol.matrix_y,
                                    self.protocol.slices])
        side = max(nx, ny, nz)

        try:
            moved_mm = {name: np.asarray(self.displacement(ppm)) for name, ppm in SHIFTED.items()}
        except Exception:        # a card that is no design yet draws its voxel alone
            moved_mm = {}
        moved_mm = {name: by for name, by in moved_mm.items() if np.any(by)}
        moved = {name: by / step for name, by in moved_mm.items()}
        # a turned voxel (a spectroscopic image's grid stands square) and what turns with it
        turned = self.turn is not None and self.protocol.sequence != MRSI
        facing = self.turn if turned else np.eye(3)

        def framed(cutting, left, top, width, height, cut, cross, across, up):
            """A plane shaded to 8 bits and centred in a square of the largest grid side, all three
            planes then drawing alike; the voxel's outline and the cross move with it."""
            shade = cutting(tissue)
            # blue below and red above the middle of a map, sand for a tissue, the anatomy showing
            # through
            if wash is not None:
                values, low, high, diverging = wash
                values = cutting(values)
                # as the adjustment leaves it: the shim's own field taken off, everywhere
                if self.overlay == "B0 field":
                    world = [cutting(np.broadcast_to(
                        (origin[n] + step[n] * np.arange(tissue.shape[n])).reshape(
                            [-1 if m == n else 1 for m in range(3)]), tissue.shape))
                             for n in range(3)]
                    values = values - self.fields()["field"].shim_field(
                        np.stack([w.ravel() for w in world], axis=1), tuple(self.position_mm),
                        tuple(self.size_mm)).reshape(values.shape)
                level = np.nan_to_num(np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0),
                                      nan=0.5 if diverging else 0.0)
                strength = (0.75 * np.abs(2.0 * level - 1.0) if diverging
                            else 0.7 * level)[..., None]
                colour = (np.where((level > 0.5)[..., None], (225, 80, 60), (70, 130, 225))
                          if diverging else np.array((226, 165, 95)))
                shade = np.uint8((1.0 - strength) * shade[..., None] + strength * colour)
            rows, columns = (side - shade.shape[0]) // 2, (side - shade.shape[1]) // 2
            square = np.zeros((side, side) + shade.shape[2:], dtype=np.uint8)
            square[rows:rows + shade.shape[0], columns:columns + shade.shape[1]] = shade
            # the picture's corners in the world: a column runs along `across`, a row down along
            # `up`, the plane lies at the view
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
            # what crosses this plane, as polygons of pixels cut out of the whole picture: a point
            # r, in mm from the voxel's centre, lies (u, v) along the plane's two axes and w, the
            # plane's own offset, along its normal, and every side (n, bound) keeps n . r >= bound
            middle = (left + columns + 0.5 * width, top + rows + 0.5 * height)
            w = float(origin[normal] + step[normal] * (a, b, c)[normal]) - self.position_mm[normal]

            def clipped(*sides):
                polygon = [(0.0, 0.0), (float(side), 0.0), (float(side), float(side)),
                           (0.0, float(side))]
                for n, bound in sides:
                    def level(x, y, n=n, bound=bound):
                        return (n[across] * (x - middle[0]) * step[across]
                                + n[up] * (middle[1] - y) * step[up] + n[normal] * w - bound)
                    kept = []
                    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                        here, there = level(*start), level(*end)
                        if here >= 0.0:
                            kept.append(start)
                        if (here >= 0.0) != (there >= 0.0):
                            t = here / (here - there)
                            kept.append((start[0] + t * (end[0] - start[0]),
                                         start[1] + t * (end[1] - start[1])))
                    polygon = kept
                return polygon

            # the saturation slabs: the strip between a slab's two faces, turned with the voxel
            bands = {}
            for axis, centre, thick, *tilted in (self.slabs() if self.protocol.ovs else ()):
                n = facing @ (np.asarray(tilted[0]) if tilted else np.eye(3)["xyz".index(axis)])
                if n[across] or n[up]:
                    polygon = clipped((n, centre - 0.5 * thick), (-n, -centre - 0.5 * thick))
                    if polygon:
                        bands[f"ovs {axis}{centre:+.0f}"] = polygon
            # a turned voxel and its displaced ones: the six sides of each box
            boxes = {}
            if turned:
                halves = 0.5 * np.asarray(self.size_mm)
                for name, by in {"voxel": np.zeros(3),
                                 **{name: facing @ d for name, d in moved_mm.items()}}.items():
                    boxes[name] = clipped(*((sign * facing[:, k],
                                             sign * (facing[:, k] @ by) - halves[k])
                                            for k in range(3) for sign in (1.0, -1.0)))
                cut = bool(boxes["voxel"])
            return (square, (left + columns, top + rows, width, height), bool(cut),
                    (cross[0] + columns, cross[1] + rows),
                    {**bands,
                     **(boxes if turned else {
                         name: (left + columns + by[across], top + rows - by[up], width, height)
                         for name, by in moved.items()}),
                     **({} if self.shim_mm is None else {"shim": (
                         left + columns + 0.5 * width - 0.5 * self.shim_mm / step[across],
                         top + rows + 0.5 * height - 0.5 * self.shim_mm / step[up],
                         self.shim_mm / step[across], self.shim_mm / step[up])})},
                    corners)

        return {
            "tra": framed(lambda v: v[:, ::-1, c].T, i - half[0], ny - 1 - j - half[1],
                          2 * half[0], 2 * half[1], abs(c - k) <= half[2], (a, ny - 1 - b), 0, 1),
            "sag": framed(lambda v: v[a, :, ::-1].T, j - half[1], nz - 1 - k - half[2],
                          2 * half[1], 2 * half[2], abs(a - i) <= half[0], (b, nz - 1 - c), 1, 2),
            "cor": framed(lambda v: v[:, b, ::-1].T, i - half[0], nz - 1 - k - half[2],
                          2 * half[0], 2 * half[2], abs(b - j) <= half[1], (a, nz - 1 - c), 0, 2),
        }

    def scroll(self, plane, slices):
        """Leaf through a plane by `slices` of the subject's grid; the voxel stays."""
        body = self.body()
        axis = {"sag": 0, "cor": 1, "tra": 2}[plane]
        step, origin = np.diag(body.affine)[axis], body.affine[axis, 3]
        index = min(max(round((self.view_mm[axis] - origin) / step) + slices, 0),
                    body.gm.shape[axis] - 1)
        self.view_mm[axis] = float(origin + step * index)

    def look(self, plane, column, row):
        """Bring the other two planes to the point pressed in this one, (column, row) in image
        pixels: the view moves through the subject, the voxel stays."""
        kept = list(self.position_mm)
        self.place(plane, column, row)
        self.view_mm, self.position_mm = list(self.position_mm), kept

    def place(self, plane, column, row):
        """Move the voxel to the point clicked in a plane, (column, row) in image pixels; the
        coordinate the plane does not show becomes the slice on view, as the click was on it."""
        self.position_mm = self.at(plane, column, row)

    #***********#
    #   scene   #
    #***********#
    def scene(self, points=5000):
        """What the room's 3D view draws, world millimetres: the subject's surface as a
        cloud of `points`, the receive loops as (centre, normal, radius), the voxel's box
        as (centre, size)."""
        body = self.body()
        # the surface of a fine grid takes seconds: once per subject
        if ("cloud", self.subject) not in self._bodies:
            inside = (body.gm + body.wm) > 0.5
            surface = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0)
                                 & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
                                 & np.roll(inside, 1, 2) & np.roll(inside, -1, 2))
            index = np.argwhere(surface)
            index = index[np.random.default_rng(0).choice(len(index), size=min(points, len(index)),
                                                          replace=False)]
            self._bodies[("cloud", self.subject)] = (index * np.diag(body.affine)[:3]
                                                     + body.affine[:3, 3]).astype(np.float32)
        coils = self.array()
        loops = [] if coils is None else [(tuple(loop.center_mm), tuple(loop.normal),
                                           float(loop.radius_mm)) for loop in coils.loops]
        self.lipid_mm()     # fills the layer's points where the subject has one
        return {"cloud": self._bodies[("cloud", self.subject)],
                "lipid": self._bodies.get(("fat", self.subject), np.empty((0, 3)))[::40],
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
        scan = self.twin(sequence, profile=profile)
        voxel = scan.voxel(tuple(self.position_mm), tuple(self.size_mm), turn=self.turn)
        fractions, limits, drawn = voxel.fractions, profile.scanner, voxel.sample(self.seed)
        return {"subject": {"concentrations (mM)": {
                                name: float(drawn["conc"][name])
                                for name in ("NAA", "Cr", "PCr", "PCho", "GPC", "Ins", "Glu",
                                             "Gln", "GABA", "GSH", "Lac")
                                if name in drawn["conc"]
                                and isinstance(drawn["conc"][name], (int, float))},
                            "T2 (ms)": {name: 1000.0 * float(drawn["t2"][name])
                                        for name in ("NAA", "Cr", "PCho") if name in drawn["t2"]
                                        and isinstance(drawn["t2"][name], (int, float))},
                            "line broadening (Hz)": float(drawn["lineshape"]["gauss_hz"]),
                            "noise level": float(drawn["noise"]["sigma"])},
                "linewidth_hz": self.linewidth(), "lipid_mm": self.lipid_mm(),
                "displaced": {name: self.displacement(ppm) for name, ppm in SHIFTED.items()},
                "gm": float(fractions.gm), "wm": float(fractions.wm), "csf": float(fractions.csf),
                "scan_time_s": card.scan_time_s,
                # what one repetition holds: the preludes, the localization, the readout; the
                # repetition time must
                "needs_s": float(event_spans(sequence)[-1][1]) + card.npoints / card.bandwidth_hz,
                "limits": (limits.peak_b1_ut, limits.max_grad_mt_m, limits.max_slew_t_m_s),
                "encoding": self.encoding(),
                # (a suppression train's selective pulses are ideal and take no time in the engine:
                # the package's check then says how much time they would need on this scanner,
                # which is said with it)
                "violations": tuple(
                    problem + (" (the suppression train's selective pulses take no time in the "
                               "simulation; on this scanner they would need that much more)"
                               if problem.startswith("the delays are") and self.preludes(card)
                               else "")
                    for problem in dict.fromkeys(sequence.violations(limits)))
                + ((self.unfitted,) if self.unfitted else ()) + (
                    (f"TR {card.tr_ms:g} ms is shorter than one repetition, "
                     f"""{1000.0 * (float(event_spans(sequence)[-1][1])
                                    + card.npoints / card.bandwidth_hz):.0f} ms"""
                     " with its water suppression, OVS and readout",)
                    if card.tr_ms / 1000.0 < float(event_spans(sequence)[-1][1])
                    + card.npoints / card.bandwidth_hz
                    else ()) + (
                    self.encoding()["problems"] if self.encoding() else ()),
                "records": {leaf: record.record(scan)
                            for leaf, record in profile.calibrations().items()}}

    #***********#
    #   queue   #
    #***********#
    def plan(self, queued=True, instead_of=None):
        """Put the card with the voxel as they stand into the queue; returns the exam. An exam
        that is not queued is a preview (the designer's look at a design's spectrum); with
        `instead_of` it takes that planned exam's place and number (an edited card)."""
        card = dataclasses.replace(self.protocol, averages=int(self.protocol.averages),
                                   npoints=int(self.protocol.npoints))
        # an edited or an own design has its own timings
        timed = card.sequence in SEQUENCES or card.sequence == MRSI
        number = len(self.queue) + 1 if instead_of is None else self.queue.index(instead_of) + 1
        name = f"{number} {card.sequence}" + (f" TE {card.te_ms:g}" if timed else "")
        exam = Exam(name=name, card=card, position_mm=tuple(self.position_mm),
                    size_mm=tuple(self.size_mm),
                    # a spectroscopic image's grid stands square
                    turn_deg=(0.0, 0.0, 0.0) if card.sequence == MRSI else tuple(self.turn_deg),
                    overrides=dict(self.overrides), run={},
                    room={name: getattr(self, name) for name in self.ROOM},
                    records=self.outlook()["records"], sequence=self.sequence(card))
        if card.sequence in EDITING:
            _, exam.combinations, exam.shown = EDITING[card.sequence]
            exam.conditions = self.editing(card)
        elif card.sequence == "Custom" and len(self.designs) > 1:
            # the operator's own multiplexed design
            exam.conditions = {name: self.own(card.tr_ms / 1000.0, name) for name in self.designs}
            for prelude in reversed(self.preludes(card).values()):
                exam.conditions = {name: prelude + design
                                   for name, design in exam.conditions.items()}
            exam.combinations = self.combinations()
            exam.shown = next(iter(exam.combinations))
        else:
            exam.conditions, exam.combinations = {"scan": exam.sequence}, {"scan": {"scan": 1.0}}
        if instead_of is not None:
            self.queue[number - 1] = exam
        elif queued:
            self.queue.append(exam)
            # the same card without water suppression, a few transients
            if card.water_reference and card.sequence != MRSI:
                bare = dataclasses.replace(card, averages=int(card.water_reference),
                                           water_suppression="off", water_reference=0)
                # the same card WITHOUT its suppression train
                unsuppressed = (self.editing(bare) if card.sequence in EDITING else
                                {name: self.own(card.tr_ms / 1000.0, name) for name in self.designs}
                                if card.sequence == "Custom" and len(self.designs) > 1
                                else {"scan": self.sequence(bare)})
                self.queue.append(dataclasses.replace(
                    exam, name=f"{number + 1} {card.sequence} water reference", frames=[],
                    spectra={}, tracks={}, truth={},
                    card=bare, sequence=next(iter(unsuppressed.values())), conditions=unsuppressed,
                    overrides={**exam.overrides, "water.residual_fraction": 1.0,
                               "downfield.presaturated": False}))
                # the reference runs before its card, and the numbers follow the order
                if card.water_reference_first:
                    self.queue[-2], self.queue[-1] = self.queue[-1], self.queue[-2]
                    self.queue[-2].name, exam.name = (
                        f"{number} {card.sequence} water reference",
                        exam.name.replace(f"{number} ", f"{number + 1} ", 1))
        return exam

    @property
    def drift_hz_per_min(self):
        """The rate the scanner's frequency drifts at in this session: the operator's, or one draw
        of Augmentrum's 3 T envelope, read off a still synthesizer's two transients a minute
        apart."""
        if ("run.drift_hz_per_min" in self.overrides
                and not isinstance(self.overrides["run.drift_hz_per_min"], (tuple, str))):
            return float(self.overrides["run.drift_hz_per_min"])
        if self._drift is None:
            self._drift = float(self.twin().params.sample(
                np.random.default_rng(self.seed))["run"]["drift_hz_per_min"])
        return self._drift

    def showcase(self):
        """What the own design does to a spectrum as the card plays it: two transients of the
        subject on the table (seed 0, the registered person) at the planned voxel, a uniform
        field, the card's water suppression and OVS before the design (the water and the scalp's
        lipids in, so what they leave shows, or what the design alone leaves where they are
        off), and the rest of IDEALS off (no noise, macromolecules, leaked water, frequency and
        phase errors, eddy currents, spurious echoes), the run still. Returns (ppm, real
        spectrum) on 0.2 to 4.2 ppm."""
        clean = Session("Brain, ICBM152 average, 2 mm")
        clean.subjects, clean.pictures, clean.images, clean._bodies = (
            self.subjects, self.pictures, self.images, self._bodies)
        clean.subject, clean.person = self.subject, dict(self.person)
        clean.position_mm, clean.size_mm = list(self.position_mm), list(self.size_mm)
        clean.scanner, clean.field = self.scanner, "uniform"
        clean.designs, clean.condition, clean.weights = self.designs, self.condition, self.weights
        clean.protocol = dataclasses.replace(self.protocol, sequence="Custom",
                                             averages=2 * len(self.designs), water_reference=0)
        clean.overrides = {path: value for name, leaves in IDEALS.items() if name != "lipids"
                           for path, value in leaves.items() if path != "water.residual_fraction"}
        clean.overrides.update({f"run.{argument}": off for argument, off in RUN.values()})
        exam = clean.run(clean.plan(queued=False), seed=0)
        shown = (exam.ppm > 0.2) & (exam.ppm < 4.2)
        return exam.ppm[shown], exam.frames[-1][1][shown]

    def run(self, exam, seed=None):
        """Acquire an exam: every transient apart, with the instabilities of a real run
        (scan.Multiplex.transients: drift, breathing, motion, as the exam's `run` sets them),
        and the display as it builds up over the transients. The seed is the subject on the
        table unless given; every exam takes noise of its own (its place in the queue)."""
        now = {name: getattr(self, name) for name in self.ROOM}
        # in the room the exam was planned in; what stands there now comes back after
        for name, value in exam.room.items():
            setattr(self, name, value)
        began = time.perf_counter()
        try:
            return self.acquire(exam, seed)
        finally:
            for name, value in now.items():
                setattr(self, name, value)
            if exam.status == "done":
                self._took[(exam.card.sequence, exam.card.real,
                            getattr(exam.sequence, "unfolded", False))] = (
                    time.perf_counter() - began)

    def compute_s(self, card=None):
        """About how long this computer takes to simulate a card, in seconds, or None until
        something tells: what the last exam of its kind took here. A card that plays the whole
        voxel at once (a tilted slab, Sequence.unfolded) is costed, its metabolites being nearly
        all of it: the design timed once on glutamate at two small ensembles (the difference
        leaves the fixed cost out), the time per isochromat scaled by every metabolite system's
        dimension cubed and the whole ensemble, and what the card takes untilted added."""
        card = self.protocol if card is None else card
        design = self.sequence(card)
        took = self._took.get((card.sequence, card.real, design.unfolded))
        if not design.unfolded or took is not None:
            return took
        engine = Engine(self.profile().scanner)
        key = design.structure()
        if key not in self._probes:
            probe, spent = SpinSystem.from_database("Glu"), []
            for side in (3, 6):
                began = time.perf_counter()
                engine.lines(design, *probe, isochromats=side)
                spent.append(time.perf_counter() - began)
            self._probes[key] = ((spent[1] - spent[0]) / (6 ** 3 - 3 ** 3),
                                 max(system.dim for system in probe))
        per, dim = self._probes[key]
        database = Database()
        systems = [system for name in database.metabolites() if name in concentrations.DE_GRAAF
                   for system in SpinSystem.from_database(name, database=database)]
        return (sum(per * (system.dim / dim) ** 3 for system in systems) * engine.isochromats ** 3
                + self._took.get((card.sequence, card.real, False), 0.0))

    def acquire(self, exam, seed=None):
        """`run` in the room as it stands."""
        seed = self.seed if seed is None else seed
        exam.status = "running"
        card, sequence = exam.card, exam.sequence
        profile = self.profile()
        coils = self.array()
        if card.sequence == MRSI:
            grid = self.twin(sequence, params=self.params(exam.overrides), card=card,
                             profile=profile).mrsi(
                center_mm=exam.position_mm,
                matrix=(int(card.matrix), int(card.matrix_y))
                + ((int(card.slices),) if card.slices > 1 else ()),
                voxel_mm=exam.size_mm[0], thickness_mm=exam.size_mm[2],
                oversampling=int(card.mrsi_oversampling),
                weighting=None if card.mrsi_weighting == "none" else card.mrsi_weighting,
                sampling=None if card.mrsi_sampling == "full" else card.mrsi_sampling,
                profile=None if card.mrsi_profile == "box" else card.mrsi_profile,
                trajectory=None if card.mrsi_readout == "phase encoded" else card.mrsi_readout)
            exam.data = grid.acquire(seed=seed, npoints=card.npoints,
                                     dwell_s=1.0 / card.bandwidth_hz)
            fids = np.asarray(exam.data[:])                         # (nx, ny, nz, points)
            window = np.exp(-np.pi * 3.0 * np.arange(card.npoints) / card.bandwidth_hz)
            spectra = np.fft.fftshift(np.fft.fft(fids * window, 2 * card.npoints, axis=-1), axes=-1)
            f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
            # the file's frame: the centre at 4.65
            ppm = 4.65 + np.fft.fftshift(
                np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz
            upfield = (ppm > 0.2) & (ppm < 4.2)
            exam.ppm, spectra = ppm[upfield], spectra[..., upfield]
            exam.maps = {name: np.abs(spectra[..., (exam.ppm > low)
                                               & (exam.ppm < high)]).max(axis=-1)
                         for name, (low, high) in MAPS.items()}
            top = np.take_along_axis(spectra, np.abs(spectra).argmax(axis=-1)[..., None], axis=-1)
            exam.cells = (spectra * np.exp(-1j * np.angle(top))).real
            centre = exam.cells[int(card.matrix) // 2, int(card.matrix_y) // 2,
                                int(card.slices) // 2]
            exam.frames = [(1, centre, centre)]
            exam.spectra = {"centre cell": centre}
            exam.truth = {name: float(value)
                          for name, value in exam.data.hdr_ext.to_dict()["SimulationGroundTruth"][
                              "params"]["conc"].items()
                          if isinstance(value, (int, float))}
            exam.status = "done"
            return exam
        # what is the SCANNER'S of this exam is drawn anew for every exam (its place in the queue),
        # while the seed keeps the person: the frequency offset and phase the acquisition starts
        # with, its eddy currents, its spurious echoes, and the transmit gain, by which a
        # suppression train's residual differs from exam to exam. What the operator set stays.
        anew = int(np.random.SeedSequence([int(seed), self.queue.index(exam) + 1
                                           if exam in self.queue else 0]).generate_state(1)[0])
        scan = self.twin(sequence, params=self.params(exam.overrides), card=card, profile=profile)
        truth = scan.params.sample(anew)
        drawn = truth.to_dict()
        state = {}
        # a branch of leaves, one leaf at the root (b1_scale), or one leaf by its path
        for group in STATE:
            node = drawn
            for key in group.split("."):
                node = node.get(key) if isinstance(node, dict) else None
            for path, value in ([(f"{group}.{leaf}", v) for leaf, v in node.items()]
                                if isinstance(node, dict) else [(group, node)]):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    state[path] = float(value)
        exam.adjustment = self.adjust(exam, scan.voxel(exam.position_mm, exam.size_mm,
                                                       turn=turning(exam.turn_deg)
                                                       if any(exam.turn_deg) else None),
                                      truth, state)
        room = Multiplex(self.body(), profile.scanner, exam.conditions, records=profile.records,
                         params=self.params({**state, **exam.overrides}),
                         # one engine per field for every exam: its graphs are built once, not
                         # once an exam
                         engine=self._bodies.setdefault(("engine", profile.scanner.b0_t),
                                                        Engine(profile.scanner)),
                         **self.fields(card),
                         # the array's noise is tied to the scanner's record AT THIS VOXEL, so
                         # every place has the record's SNR: what an array gives by place (better
                         # under its loops) is not in the console yet
                         coils=None if coils is None else (
                             coils.loaded(own_noise=self.own_share()[0]) if coils.loops
                             else coils).calibrated(exam.position_mm, exam.size_mm))
        names = list(exam.conditions)
        # interleaved, as an edited scan runs
        exam.schedule = [names[k % len(names)] for k in range(card.averages)]
        run = {**exam.run, "synthesis": {"drift_hz_per_min": self.drift_hz_per_min,
                                         **exam.run.get("synthesis", {})}}
        started = 0.0 if self.prescan else self.drift_hz_per_min * self.clock_s / 60.0
        if exam in self.queue:      # a preview leaves the scanner's clock alone
            self.clock_s += card.scan_time_s
        exam.data = room.transients(exam.position_mm, exam.size_mm, exam.schedule,
                                    tr_s=card.tr_ms / 1000.0, seed=seed, turn_deg=exam.turn_deg,
                                    step=self.queue.index(exam) + 1 if exam in self.queue else 0,
                                    frequency_start_hz=started, **run,
                                    npoints=card.npoints, dwell_s=1.0 / card.bandwidth_hz)
        self.display(exam)
        header = exam.data.hdr_ext.to_dict()
        exam.tracks = {key: header["SimulationSchedule"][key]
                       for key in ("frequency_hz", "phase_deg", "amplitude")
                       if key in header.get("SimulationSchedule", {})}
        exam.echoes = header["SimulationSchedule"].get("spurious", {}).get("amplitude", [])
        exam.truth = {name: float(value)
                      for name, value in header["SimulationGroundTruth"]["params"]["conc"].items()
                      if isinstance(value, (int, float))}
        exam.status = "done"
        return exam

    #****************#
    #   adjustment   #
    #****************#
    def adjust(self, exam, voxel, truth, state):
        """The scanner's water suppression adjustment for an exam, as a prescan finds it on the
        subject: the factor on the train's flips that leaves of the water what the scanner's own
        adjustment leaves: the drawn water.adjusted_residual (a record per scanner, read against
        the water references, signed by the residual's phase against the metabolites) of the full
        water's steady state at the excitation. The water is summed over the voxel's field groups
        (in or out of the pulses' band), its tissues (each with its own T1 and content) and the
        drawn transmit scale, in the closed form of the train's ideal pulses: a spin in the band
        keeps cos(flip) of its Mz, the crusher keeps Mz alone, the gaps and the rest of the TR
        recover with T1 (from the excitation to the next train), which is what the engine then
        plays. A played train the engine plays to its steady state instead, one water spin per
        tissue and field group in one batch: its pulses' own band and the T1 acting through them
        put the null where the closed form cannot follow. The exam's designs get the adjusted
        train. 1.0 without a prescan, without a train, or with the shortcut water."""
        card = exam.card
        if not self.prescan or card.water_suppression not in SUPPRESSION:
            return 1.0
        train = SUPPRESSION[card.water_suppression](real=card.real)
        n = len(train.events)
        if not train.adjusted or any(
                len(design.events) < n
                or not isinstance(design.events[0], (SelectivePulse, ShapedPulse))
                for design in exam.conditions.values()):
            return 1.0
        field = self.fields(card)["field"]
        groups = ([(mean, weight) for mean, weight, _ in field.groups(
                      exam.position_mm, exam.size_mm, np.zeros(1), 8,
                      spread_hz=float(voxel.resolve(truth.lineshape.gauss_hz)))]
                  if field is not None else [(0.0, 1.0)])
        shift, b1 = state.get("acquisition.shift_hz", 0.0), voxel.b1_scale(truth)
        content, fractions = self.body().water_content, voxel.fractions
        per_tissue = lambda leaf, name: (getattr(leaf, name) if isinstance(leaf, Tissues)
                                         else voxel.resolve(leaf))
        tissues = [(getattr(fractions, name) * content[name], per_tissue(truth.water.t1_s, name),
                    per_tissue(truth.water.t2_s, name))
                   for name in ("gm", "wm", "csf")
                   if getattr(fractions, name) * content[name] > 0.0]
        tr_s = card.tr_ms / 1000.0
        # the water recovers from the excitation to the next train
        rest = max(tr_s - sum(train.delays_s), 0.0)
        if train.pulse_s is None:
            factors = np.linspace(0.5, 1.5, 1001)
            water, full = np.zeros_like(factors), 0.0
            for share, t1, _ in tissues:
                # the reference: no train, the same TR
                full += share * (1.0 - np.exp(-(rest + sum(train.delays_s)) / t1))
                for mean, weight in groups:
                    inband = abs(mean - shift) <= train.bandwidth_hz / 2.0
                    mz = np.ones_like(factors)
                    for _ in range(8):                                    # to the steady state
                        for flip, delay in zip(train.flips_deg, train.delays_s):
                            if inband:
                                mz = mz * np.cos(np.radians(flip * factors * b1))
                            mz = 1.0 - (1.0 - mz) * np.exp(-delay / t1)
                        before = mz
                        # the excitation takes it all, then the TR's rest
                        mz = np.full_like(factors, 1.0 - np.exp(-rest / t1))
                    water = water + share * weight * before
        else:
            # the played train in the engine's spoiled steady state of the TR, every tissue and
            # field group a draw of one batch (T1, T2, the transmit scale, the offset); what stands
            # before the excitation over the full water's, weighted as the closed form weights its
            # tissues
            engine = voxel.scan.engine
            rows = [(t1, t2, share * weight, mean - shift) for share, t1, t2 in tissues
                    for mean, weight in groups]
            draws = [ops.convert_to_tensor(np.asarray(values, dtype=np.float32)) for values in (
                [[t1] for t1, _, _, _ in rows], [[t2] for _, t2, _, _ in rows], [b1] * len(rows),
                [offset for _, _, _, offset in rows])]
            system = WaterSource.system(voxel.scan.database.gamma("1H"))
            def steady(design):
                fn, _ = engine.compiled_batch(design, ((system,),))
                return np.asarray(ops.convert_to_numpy(
                    Backend.vmap(fn, in_axes=(0, 0, 0, 0))(*draws)[1]))
            reference = steady(PulseAcquire(tr_s=tr_s))
            shares = np.array([share * (1.0 - np.exp(-tr_s / t1)) for t1, _, share, _ in rows])
            factors = np.linspace(0.5, 1.5, 101)
            water = np.array([np.sum(shares * np.real(np.sum(
                steady(train.scaled(factor) + PulseAcquire(tr_s=tr_s)) * np.conj(reference),
                axis=1)) / np.sum(np.abs(reference) ** 2, axis=1)) for factor in factors])
            full = float(shares.sum())
        # the scanner's adjustment leaves its record's residual, not zero: land there, on the
        # record's side
        # the record is the residual's total at t = 0 as the raw reads it, the leaked water inside
        # it: the voxel's own residual lands at the record less the leak's share along the full
        # water (the leak's phase is drawn against the residual, whose side is the record's)
        record = float(state.get("water.adjusted_residual", 0.0))
        leak = (float(truth.oov.fraction) * np.cos(np.radians(float(truth.oov.phase_deg)))
                * np.sign(record))
        target = (record - leak) * full
        # the water crosses the target on both sides of the train's null and again where the flips
        # are far too low; a scanner's adjustment starts from the nominal flips and lands past the
        # nearest null (as the Trio's raw residuals show), so the first crossing above that null is
        # the scanner's; a record the train cannot reach (a played train relaxing through its
        # pulses nulls only so far) puts it at the null
        error = water - target
        magnitude = np.abs(water)
        dips = np.flatnonzero((magnitude[1:-1] <= magnitude[:-2])
                              & (magnitude[1:-1] <= magnitude[2:])) + 1
        null = dips[np.abs(factors[dips] - 1.0).argmin()] if dips.size else magnitude.argmin()
        crossings = np.flatnonzero(np.sign(error[:-1]) != np.sign(error[1:]))
        above = crossings[crossings >= null]
        if above.size:
            k = above[0]
            factor = float(factors[k if abs(error[k]) <= abs(error[k + 1]) else k + 1])
        else:
            factor = float(factors[null])
        tuned = train.scaled(factor)
        # the design with the tuned train in place of its own, its timings kept
        def adjusted(design):
            design = copy.copy(design)
            design.events = (*tuned.events, *design.events[n:])
            return design

        exam.conditions = {name: adjusted(design) for name, design in exam.conditions.items()}
        exam.sequence = adjusted(exam.sequence)
        return factor

    #*************#
    #   display   #
    #*************#
    def display(self, exam):
        """What the console shows of an exam that ran, made from its raw transients as `processing`
        stands, so a finished exam follows the switch and the export stays raw. The coils are
        summed; then
        COMBINE is the data as they came in: the transients averaged, no phase, no alignment, so
        the phase the acquisition started with and what drift and motion do to the average show;
        ALIGN is a rough cleaning: every transient aligned in frequency and phase to the others of
        ITS OWN sub-acquisition (what tells the sub-acquisitions apart, an editing pulse or a
        phase, must stay), Augmentrum's RawProcessor with FSL-MRS's registration, and one
        zero-order phase for the exam;
        PROCESS is the processed product of a toolbox, all Augmentrum's: aligned, the transients
        unlike the rest left out, the residual water taken out (HLSVD), and of a plain scan the
        spectrum shifted onto its reference peak and phased on creatine.
        Every level shows with 2 Hz of apodization, zero filled, over the whole acquired band.
        Fills `builds`, `frames`, `transients`, `spectra`, `snr`, `ppm` and `window`."""
        card, names = exam.card, list(exam.conditions)
        fids = np.array(exam.data[:])[0, 0, 0]
        # one transient; channels summed for the display alone
        fids = fids[:, None] if fids.ndim == 1 else fids if fids.ndim == 2 else fids.sum(axis=1)
        played = np.asarray(exam.schedule)
        f0_mhz = float(exam.data.hdr_ext["SpectrometerFrequency"][0])
        reference = exam.overrides.get("water.residual_fraction") == 1.0
        kept = np.ones(fids.shape[1], dtype=bool)
        processor = None
        if self.processing != COMBINE and not reference:
            from augmentrum.processing import RawProcessor

            processor = RawProcessor(coil=False, remove_outliers=self.processing == PROCESS,
                                     average=False, ecc=False, shift_ref=False,
                                     phase_correct=False)
            for name in names:
                mine = np.flatnonzero(played == name)
                if mine.size > 2:
                    moved, _ = processor.process_tensor(fids[:, mine].T[None, None, None, None],
                                                        sw_hz=card.bandwidth_hz, sf_mhz=f0_mhz,
                                                        dim_tags=["DIM_DYN", None, None])
                    fids[:, mine] = np.asarray(moved).reshape(mine.size, -1).T
                    if self.processing == PROCESS:
                        kept[mine] = np.asarray(processor.last_keep_mask_).ravel()
        window = np.exp(-np.pi * 2.0 * np.arange(card.npoints) / card.bandwidth_hz)
        # the file's frame: the centre at 4.65
        ppm = 4.65 + np.fft.fftshift(
            np.fft.fftfreq(2 * card.npoints, 1.0 / card.bandwidth_hz)) / f0_mhz
        # a water reference shows its water line, every other exam the metabolites upfield of it
        upfield = (ppm > 3.4) & (ppm < 5.9) if reference else (ppm > 0.2) & (ppm < 4.2)

        def combined(weights, count):
            """The FID of a combination over the first `count` transients: every condition's mean
            of what was kept so far, weighted; None while a condition has not played yet."""
            total = 0.0
            for condition, weight in weights.items():
                mine = np.flatnonzero((played[:count] == condition) & kept[:count])
                if mine.size == 0:
                    return None
                total = total + weight * fids[:, mine].mean(axis=1)
            return total

        def spectrum(fid):
            return np.fft.fftshift(np.fft.fft(fid * window, 2 * card.npoints))

        # one zero-order phase for the whole exam, read at the tallest line of the average of
        # everything (of an own multiplexed design the plain sum may be what the scheme cancels, a
        # phase cycle's unwanted signal: there the shown combination gives the phase; an edited
        # scan's sum is its unedited spectrum)
        own = card.sequence == "Custom" and len(names) > 1
        everything = spectrum(combined(exam.combinations[exam.shown] if own
                                       else {name: 1.0 for name in names}, fids.shape[1]))
        phase = 1.0 if self.processing == COMBINE and not reference else np.exp(
            -1j * np.angle(
                everything[np.flatnonzero(upfield)[np.abs(everything[upfield]).argmax()]]))
        # the whole band: the display opens on `window` and zooms out to it
        wide = np.ones(ppm.size, dtype=bool)
        exam.ppm, exam.window = ppm[wide], (float(ppm[upfield].min()), float(ppm[upfield].max()))
        counts = sorted({int(round(c))
                         for c in np.linspace(1, fids.shape[1], min(fids.shape[1], 64))})
        views = {**{n: {n: 1.0} for n in names if len(names) > 1}, **exam.combinations}
        exam.builds = {name: [(count, (spectrum(partial) * phase).real[wide]) for count in counts
                              if (partial := combined(weights, count)) is not None]
                       for name, weights in views.items()}
        # the residual water out of what stands at the end
        if self.processing == PROCESS and not reference:
            for name, weights in views.items():
                clean = processor.water_removal(combined(weights, fids.shape[1])[None, :],
                                                card.bandwidth_hz, f0_mhz)
                # a plain scan: onto its reference peak, phased on creatine (the conditions of an
                # edited one share the exam's phase)
                if len(names) == 1:
                    clean = processor.phase_correction(
                        processor.shift_to_reference(clean, card.bandwidth_hz, f0_mhz), None,
                        card.bandwidth_hz, f0_mhz)[0]
                    exam.builds[name][-1] = (exam.builds[name][-1][0],
                                             spectrum(np.asarray(clean)[0]).real[wide])
                else:
                    exam.builds[name][-1] = (exam.builds[name][-1][0],
                                             (spectrum(np.asarray(clean)[0]) * phase).real[wide])
        # every transient that was kept, coils combined, under the view it belongs to (a
        # combination of several has none); 32 at most, evenly through the run
        exam.transients = {name: [(spectrum(fids[:, k]) * phase).real[wide]
                                  for k in mine[:: max(1, -(-mine.size // 32))]]
                           for name, weights in views.items() if len(weights) == 1
                           for mine in (np.flatnonzero((played == next(iter(weights))) & kept),)}
        # the display as it builds up: the transients so far, their average in the shown
        # combination, and the last one alone
        exam.frames = [(count, average, (spectrum(fids[:, count - 1]) * phase).real[wide])
                       for count, average in exam.builds[exam.shown]]
        exam.spectra = {name: build[-1][1] for name, build in exam.builds.items()}
        # the consensus SNR of what is shown (Near et al. 2021): the tallest real point within 0.2
        # to 4.2 ppm (NAA in a brain) over the standard deviation of a signal-free stretch after a
        # linear detrend; of the average and of one transient alone. The stretch lies inside the
        # band and away from the wings of the broad lines: 9 to 12 ppm where the band reaches
        # 12.3, else -4 to -1 ppm kept 0.3 ppm inside the band's low edge, clear of where a
        # receiver's filter rolls off and the lipids' wings reach
        low, high = float(ppm.min()) + 0.3, float(ppm.max()) - 0.3
        quiet = ((ppm > 9.0) & (ppm < 12.0) if high >= 12.0
                 else (ppm > max(-4.0, low)) & (ppm < -1.0))
        if quiet.sum() < 32:    # a band too narrow for either: what lies beyond the amides
            quiet = (ppm > 6.5) & (ppm < high)

        def snr(fid):
            real = (np.fft.fftshift(np.fft.fft(fid, 2 * card.npoints)) * phase).real
            return float(real[upfield].max() / (real[quiet] - np.polyval(
                np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])).std())

        weights = exam.combinations[exam.shown]
        exam.snr = (snr(combined(weights, fids.shape[1])),
                    # one transient the processing kept
                    snr(fids[:, np.flatnonzero((played == next(iter(weights))) & kept)[0]]))
        exam.rejected = int((~kept).sum())

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
            raise ValueError("a phase-cycled design is not saved yet: its receiver phases are "
                             "complex numbers")
        if any(kind == FIXED for blocks in self.designs.values() for kind, _ in blocks):
            raise ValueError("a design that holds events taken over from a card is not saved yet: "
                             "they are objects, not settings")
        self.LIBRARY.mkdir(parents=True, exist_ok=True)
        path = self.LIBRARY / f"{name}.json"
        path.write_text(json.dumps({"card": dataclasses.asdict(self.protocol),
                                    "size_mm": list(self.size_mm),
                                    "designs": self.designs, "weights": self.weights,
                                    "overrides": self.overrides,
                                    "choices": self.choices}, indent=2))
        return path

    def load_protocol(self, name):
        """Put a saved protocol on the card."""
        stored = json.loads((self.LIBRARY / f"{name}.json").read_text())
        # in place: the console's fields hold this card
        for attribute, value in stored["card"].items():
            setattr(self.protocol, attribute, value)
        self.size_mm = [float(v) for v in (stored["size_mm"] if isinstance(stored["size_mm"], list)
                                           else [stored["size_mm"]] * 3)]
        # a protocol saved before designs had sub-acquisitions
        stored_designs = stored.get("designs") or {"A": stored["design"]}
        self.designs = {name: [(kind, dict(settings)) for kind, settings in blocks]
                        for name, blocks in stored_designs.items()}
        self.condition, self.weights = next(iter(self.designs)), stored.get("weights", {})
        self.overrides = {path: tuple(value) if isinstance(value, list) else value
                          for path, value in stored["overrides"].items()}
        self.choices = dict(stored.get("choices", {}))

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
