####################################################################################################
#                                              app.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual scanner console: a NiceGUI face over console.session, in a window of its    #
#          own. The room (scanner, coil, subject) in the top bar, the exam cards on the left,      #
#          planning on three planes one leafs through and a 3D view, the card's parameters and     #
#          its pulse sequence diagram below, scan control on the right. Usage:                     #
#          uvx --from "mrsandbox[console]" mrsandbox-console                                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import asyncio
import base64
import collections
import copy
import dataclasses
import io
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from nicegui import Client, background_tasks, run, ui
from PIL import Image

from mrsandbox.io.folder import reads
from mrsandbox.hardware.arrays import PRESETS as ARRAY_PRESETS
from mrsandbox.console.session import (ALIGN, ASSUMED, BLOCKS, COMBINE, EDITING, FIELDS, FIXED,
                                       GEARS, IDEALS, LENT, MRSI, OVERLAYS, PLACE, PLACES, PROCESS,
                                       REAL, RUN, RUN_OF, SCANNERS, TARGETS, adjusted, named,
                                       SEQUENCES, SUPPRESSION, TEMPLATES, Session)
from mrsandbox.hardware.scanners import PRESETS
from mrsandbox.io import read_dicom
from mrsandbox.phantom import FieldMap
from mrsandbox.database import Database
from mrsandbox.scan.cohort import stated, table

#*************#
#   palette   #
#*************#
# the logo's: sand and copper on charcoal
SAND, COPPER, GLOW, GROUND, PANEL, LINE, INK, DIM, GOOD, BAD = (
    "#E2A55F", "#B5622C", "#F5C77E", "#151210", "#1E1A17", "#3A322B", "#EADFD2", "#8E8378",
    "#7FA66B", "#D9653B")
STYLE = f"""
body {{ background: {GROUND}; color: {INK}; font-family: "IBM Plex Mono", ui-monospace, Menlo, \
monospace; font-size: 12px; }}
.nicegui-content {{ padding: 6px; gap: 6px; height: 100vh; flex-wrap: nowrap; }}
.pane {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 2px; padding: 8px; \
gap: 6px; }}
.label {{ color: {DIM}; letter-spacing: .12em; text-transform: uppercase; font-size: 10px; }}
.value {{ color: {INK}; font-variant-numeric: tabular-nums; }}
.port {{ position: relative; border: 1px solid {LINE}; background: #000; min-width: 0; \
overflow: hidden; }}
.port img {{ -webkit-user-drag: none; user-select: none; }}
.port .tag {{ position: absolute; color: {SAND}; font-size: 11px; pointer-events: none; \
z-index: 2; text-shadow: 0 0 3px #000; }}
.exam {{ border-left: 3px solid transparent; border-bottom: 1px solid {LINE}; padding: 6px 8px; \
cursor: pointer; display: flex;
         gap: 8px; align-items: center; width: 100%; }}
.exam:hover {{ background: #26211d; }}
.exam.chosen {{ border-left-color: {SAND}; background: #2b241e; color: #fff; }}
.tool {{ border: 1px solid {LINE}; border-radius: 2px; color: {DIM} !important; width: 96px; \
height: 46px; padding: 0; font-size: 10.5px;
         line-height: 1.1; }}
.tool .q-icon {{ font-size: 19px; }}
.chip {{ border: 1px solid {LINE}; border-radius: 2px; color: {INK} !important; width: 118px; \
height: 50px; padding: 0; font-size: 10.5px;
         line-height: 1.15; cursor: grab; }}
.chip.over {{ border-left: 4px solid {GLOW}; }}
.chip.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.chip.prelude {{ border-style: dashed; color: {DIM} !important; cursor: default; }}
.tool.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.info {{ display: grid; grid-template-columns: auto 1fr; column-gap: 12px; row-gap: 5px; \
align-content: start; white-space: nowrap; }}
.q-tab {{ min-height: 28px; padding: 0 12px; }} .q-tab__label {{ font-size: 11px; \
letter-spacing: .08em; }}
.q-splitter__separator {{ background: transparent !important; width: 6px !important; }} \
.q-splitter__separator:hover {{ background: {LINE} !important; }}
.q-splitter__panel {{ overflow: hidden; }}
.q-tab-panel {{ padding: 8px 2px 2px 2px; }} .q-tab-panels {{ background: transparent; }}
.q-field__native, .q-field__label, .q-select__dropdown-icon, .q-field__input {{ \
color: {INK} !important; }}
.q-field--outlined .q-field__control:before {{ border-color: {LINE} !important; }}
"""
MM = 1.0 / 40.0  # the 3D view's units per millimetre
BLANK = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
         "DwAChwGA60e6kgAAAABJRU5ErkJggg==")
# the logo has one home, the repository's assets/; an installed package carries that file in its
# environment's share/mrsandbox (pyproject.toml, data-files)
LOGO = next((path for path in (pathlib.Path(__file__).parents[2] / "assets" / "logo" / "logo.png",
                               pathlib.Path(sys.prefix) / "share" / "mrsandbox" / "logo.png")
             if path.exists()), None)
CHOICES = {"shape": ["sinc", "gaussian", "hermite", "slr"], "slice_axis": ["none", "x", "y", "z"],
           "axis": ["x", "y", "z"], "scheme": list(SUPPRESSION), "pulses": ["played", "ideal"]}
# the designer's blocks in plain words: an icon, what the block is, and its settings with their
# units
BLOCK_ICONS = {FIXED: "push_pin", "hard pulse": "bolt", "adiabatic pulse": "all_inclusive",
               "own pulse": "gesture", "shaped pulse": "waves", "delay": "hourglass_empty",
               "crusher": "signal_cellular_alt", "coherence filter": "filter_alt",
               "water suppression": "water_drop"}
BLOCK_NOTES = {
    FIXED: "A block of the card's own sequence, as the card plays it. A pulse keeps the card's "
           "shape; its flip, duration, carrier, phase, peak B1 and slice gradient are yours to "
           "set. A gradient keeps its span; its dephasing is yours. A projection is ideal.",
    "hard pulse": "An ideal pulse of no duration. It turns every spin alike, whatever its "
                  "frequency.",
    "own pulse": "A pulse of your own shape. Type its envelope or load it from a file; the flip "
                 "angle scales it.",
    "shaped pulse": "A sinc, Gaussian, Hermite or SLR pulse, played in time, its shape's own "
                    "settings beside it. With a slice gradient it selects a slab along an axis.",
    "adiabatic pulse": "A hyperbolic secant full passage. It inverts over its whole bandwidth "
                       "whatever the B1, above a threshold; play them in pairs.",
    "delay": "Free evolution. The echo time is the sum of the delays and the pulses between "
             "excitation and readout.",
    "crusher": "A gradient pulse that dephases what is transverse. A pair about a 180 keeps the "
               "echo and spoils the rest.",
    "coherence filter": "An ideal crusher. It keeps the one coherence order given and removes "
                        "every other, without time.",
    "water suppression": "A train of selective pulses on the water line before the excitation: "
                         "VAPOR, WET, CHESS or COWS, its pulses played in time or ideal.",
}


#****************#
#   block name   #
#****************#
def block_name(kind, settings):
    """A block's kind in the designer's words: the card's own blocks by what they are."""
    if kind != FIXED:
        return kind
    event = type(settings["event"]).__name__
    return "card's " + {"ShapedPulse": "pulse"}.get(event, event.lower())


CAPTIONS = {"flip_deg": "flip angle (°)", "phase_deg": "phase (°)",
            "duration_ms": "duration (ms)", "offset_ppm": "centre (ppm)", "shape": "shape",
            "slice_axis": "slice along",
            "slice_hz_per_mm": "slice gradient (Hz/mm)", "ms": "duration (ms)",
            "axis": "along", "cycles_per_mm": "dephasing (cycles/mm)",
            "order": "coherence order kept", "scheme": "scheme",
            "envelope": "envelope, amplitudes apart by commas", "pulses": "pulses",
            "lobes": "sinc lobes a side", "truncation": "Gaussian cut (fraction of the peak)",
            "time_bandwidth": "time-bandwidth product (SLR)", "bandwidth_hz": "bandwidth (Hz)",
            "peak_hz": "peak B1 (Hz)"}
# the calibration records in plain words
RECORDS = {"noise.sigma": "Noise level (the simulation's unit: one transient, 8 mL, 3 T)",
           "lineshape.gauss_hz": "Line broadening (Hz)",
           "conc.Cr": "Creatine level the amplitudes are scaled by (mM)",
           "coils.loaded.own_noise": "Share of a channel's noise that is its own",
           "mm.level_mm": "Macromolecule level"}
# the settings' groups by name, and the ways a parameter can be set
GROUPS = {"conc": "Concentrations (mM)", "t1": "T1 of the metabolites (s)",
          "t2": "T2 of the metabolites (s)", "contrast": "Grey to white matter concentration ratio",
          "diffusion": "Diffusion", "water": "Water", "lipids": "Lipids",
          "mm": "Macromolecules", "downfield": "Downfield peaks", "lineshape": "Line shape (Hz)",
          "noise": "Noise", "b1_scale": "B1 scale",
          "acquisition": "Acquisition: shifts, phases, eddy currents",
          "gradients": "Gradient eddy currents", "spurious": "Spurious echoes",
          "oov": "Water outside the voxel, leaked",
          "run": "The run: drift, breathing, jitter, motion"}
WAYS = ["as its source gives it", "set to", "vary between",
        "from a map (a NIfTI on the anatomy's grid)"]
# what the water suppression leaves: the scanner's, though the leaf lives under water (the leaked
# water's oov is the scanner's group)
RESIDUAL = ("water.adjusted_residual",)
# the sources a value can come from, the same word and colour on every chip; the tree's "own" says
# "yours"
SOURCE_WORDS = {"literature": "literature", "stated": "stated", "pooled": "pooled",
                "mixed": "mixed", "dataset": "dataset", "own": "yours", "custom": "custom"}
SOURCE_SAYS = {"literature": "a published number, or the manufacturer's",
               "stated": "a range the package assumed, nothing published behind it",
               "pooled": "the package's measured rows, pooled for this room",
               "mixed": "the literature's studies and the measured datasets, each an equal "
                        "voice",
               "dataset": "one real subject's own reading, from the dataset on the floor",
               "own": "measured from a scan you uploaded", "custom": "a number or range you set"}
SOURCE_COLORS = {"literature": "#7FA6D9", "stated": "#8C8C8C", "pooled": "#7FA66B",
                 "mixed": "#6FB5AE", "dataset": "#C9A0DC", "own": "#E2A55F", "custom": "#E8E2D6"}
# the datasets of a scatter, in this order and at most three (told apart under colour blindness on
# the panel); the literature's blue and the pooled green stay with the reference lines
CATEGORY = ("#C98500", "#9085E9", "#D55181")
SAID = "sources: " + " · ".join(f"{SOURCE_WORDS[s]} = {SOURCE_SAYS[s]}" for s in SOURCE_WORDS)
# the leaves whose ranges the package assumed, and why (scan.defaults says it at each); every other
# plain leaf is typed from a table of the literature (de Graaf, Gudmundson, Deelchand 2018, the
# consensus MM, Ozdemir, the water tables)
STATED = {"gradients": "the eddy terms of a design's gradients: ranges set so the strong end "
                       "distorts a line visibly, the weak end is clean",
          "spurious": "spurious echoes: stated ranges",
          "acquisition.shift_hz": "a stated spread of the centre frequency",
          "acquisition.phase_deg": "a stated spread of the zero-order phase",
          "acquisition.eddy_std_rad": "a stated spread of the eddy phase over the FID",
          "acquisition.eddy_lp_cut_hz": "its stated bandwidth",
          "acquisition.eddy_seed": "a seed",
          "acquisition.eddy_transient_tau_s": "a stated time constant of the transient",
          "lipids": "subcutaneous fat: water-like proton density and stated relaxation",
          "water.residual_fraction": "the shortcut's residual, stated",
          "oov.t2star_s": "typed from the Trio readings of the leaked water, not pooled",
          "oov.offset_hz": "typed from the Trio readings of the leaked water",
          "oov.phase_deg": "typed from the Trio readings of the leaked water",
          "b1_scale": "a stated calibration scatter of the transmitter"}


def plain_word(path):
    """The source word of a leaf that carries no sources of its own: stated where the package
    assumed its range, literature where a table gives it."""
    return "stated" if path in STATED or path.split(".")[0] in STATED else "literature"


def plain_why(path):
    return (STATED.get(path) or STATED.get(path.split(".")[0])
            or "typed from a table of the literature (scan.defaults names it)")
# the scanner's leaves in words
NAMES = {"noise.sigma": "noise level (one transient, 8 mL, 3 T)",
         "lineshape.gauss_hz": "line broadening, the shim (Hz)",
         "acquisition.delay_s": "receiver delay (s)",
         "acquisition.eddy_transient_rad": "eddy phase transient at the start (rad)",
         "acquisition.eddy_transient_tau_s": "its time constant (s)",
         "acquisition.shift_hz": "frequency offset (Hz)",
         "acquisition.phase_deg": "zero-order phase (deg)",
         "acquisition.eddy_std_rad": "eddy phase over the FID (rad, std)",
         "acquisition.eddy_lp_cut_hz": "its bandwidth (Hz)", "acquisition.eddy_seed": "its seed",
         "water.adjusted_residual": "residual water the suppression leaves (of the full water)",
         "oov.fraction": "water leaked from outside the voxel (of the full water)",
         "b1_scale": "transmit scale (of nominal)", "spurious.amplitude": "spurious echo size",
         "run.drift_hz_per_min": "frequency drift over the run (Hz per minute)",
         "run.ar_sigma_hz": "slow frequency wander (Hz)",
         "run.resp_freq_hz": "breathing rate (Hz)",
         "run.resp_amp_hz": "breathing's field modulation (Hz)",
         "run.phase_jitter_deg": "phase jitter (degrees)",
         "run.amp_jitter_frac": "amplitude jitter (fraction)",
         "run.broaden_hz": "broadening over the run (Hz)",
         "run.events_per_min": "motion events (per minute)",
         "run.motion_mm": "a motion event moves the head by (mm; zero: the events kick frequency, "
                          "phase and amplitude alone)",
         "water.t1_s": "the water's T1 per tissue (s)"}
SHOWN = {"fixed": "{0:g}", "range": "{0:g} to {1:g}", "normal": "{0:.4g} ± {1:.2g}",
         "empirical": "{0:.4g}, one of {3:.0f} measured subjects ({1:.3g} to {2:.3g})"}
OWN = "your own file ..."      # on top of every drop-down of stored maps
# the package's stored maps by one name each, the same on every line
STORED = {"qsm_rc2": "QSM head", "bigbrain_mr": "BigBrain-MR", LENT: "QSM head"}
# the displaced voxels' colours on the planes (session.SHIFTED)
SHIFT_COLOURS = {"water": "#7FB2D9", "lipid": "#D98BB0", "shim": "#E6D84A"}
SLAB = "#6FC3B2"                # the outer-volume suppression slabs
# the planning view's keys (a field being typed in keeps its keys)
KEYS = "+ − 0: zoom the plane under the mouse · v: to the voxel · s: shifts · f: flip · o: overlay"
# the planes' edges as a console letters them (top, bottom, left, right), and the axis each plane
# leafs along
EDGES = {"tra": ("A", "P", "L", "R", 2, "z"), "sag": ("H", "F", "P", "A", 0, "x"),
         "cor": ("H", "F", "L", "R", 1, "y")}

#**********#
#   mark   #
#**********#
# the logo as its file has it, its own transparency kept, trimmed to what is drawn: small for the
# bar, and large for the loading screen at three device pixels per shown one, so that it stays
# sharp on a dense display
MARK = MARK_LARGE = None
if LOGO is not None:
    drawn = Image.open(LOGO).convert("RGBA")
    drawn = drawn.crop(drawn.getchannel("A").getbbox())
    sizes = {}
    for height in (100, 450):
        buffer = io.BytesIO()
        drawn.resize((round(height * drawn.width / drawn.height), height), Image.LANCZOS).save(
            buffer, format="PNG")
        sizes[height] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    MARK, MARK_LARGE = sizes[100], sizes[450]


#*************#
#   diagram   #
#*************#
def lanes(drawn, width=470.0, lane=62.0, mark=None):
    """A pulse sequence diagram (Session.diagram) drawn the way the literature draws one
    (Landheer et al. 2020, Figure 1; Tkac et al. 1999, Figure 1 for a suppression train), as
    SVG: the RF with its lobes and flips (an ideal pulse a bar of fixed width, a 90 half as
    tall as a 180), the three gradients with slice selection open and crushers filled, an
    ideal crusher (a coherence filter) dashed on all three, the signal, the intervals between
    the localization's pulses on top. A prelude that outlasts the localization (a
    suppression train) is drawn on a time axis of its own, compressed into a framed stretch.
    `mark` (start, end, shift) lights a span in ms up, the designer's chosen block, moved right
    by `shift` drawing units (a coherence filter is drawn after its instant, not on it)."""
    names = ("RF", "Gx", "Gy", "Gz", "Signal")
    left, right, top, fold, tall = 50.0, 12.0, 30.0, 0.24 * width, 2.0 * lane
    start, stop = drawn["adc"]
    folds = [(a, b) for a, b, _ in drawn["groups"]]
    scale = ((width - left - right - fold * len(folds))
             / max(stop - sum(b - a for a, b in folds), 1e-9))

    def x(t):
        at, since = left, 0.0
        for a, b in folds:
            if t < a:
                break
            at, since = at + scale * (a - since) + fold * min((t - a) / max(b - a, 1e-9), 1.0), b
            if t <= b:
                return at
        return at + scale * (t - since)

    def folded(t):
        return any(a <= t < b - 1e-6 for a, b in folds)

    def path(trace, middle, reach):
        parts = [f"{'M' if k == 0 else 'L'}{x(t):.1f},{middle - reach * value:.1f}"
                 for k, (t, value) in enumerate(trace)]
        return " ".join(parts)

    middles = [top + 0.72 * tall] + [top + tall + lane * (k + 0.5) for k in range(4)]
    height = top + tall + 4 * lane + 24.0
    peak = max([abs(v) for axis in "xyz" for _, v in drawn[axis]] + [1e-9])
    reach = 0.62 * tall / max([abs(v) for _, v in drawn["rf"]] + [h for _, h, _ in drawn["bars"]]
                              + [1e-9])
    out = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" style="display: block" '
           f'font-family="inherit" font-size="10">']
    if mark is not None:
        out.append(f'<rect x="{x(mark[0]) - 5 + mark[2]:.1f}" y="{top - 4}" '
                   f'width="{x(mark[1]) - x(mark[0]) + 10:.1f}" height="{tall + 4 * lane + 8}" '
                   f'rx="3" fill="{GLOW}" fill-opacity="0.1" stroke="{GLOW}" '
                   f'stroke-opacity="0.5" />')
    for (a, b, name) in drawn["groups"]:
        out.append(f'<rect x="{x(a) + 1:.1f}" y="{top - 2}" width="{fold - 2:.1f}" '
                   f'height="{tall + 3 * lane + 4}" rx="2" fill="{GLOW}" '
                   f'fill-opacity="0.04" stroke="{LINE}" stroke-dasharray="3 2" />'
                   f'<text x="{x(a) + fold / 2:.1f}" y="{top - 8}" fill="{GLOW}" '
                   f'text-anchor="middle" font-size="9">{name}'
                   + ("" if name.endswith("ideal") else f" · {b - a:.0f} ms") + "</text>")
    for name, middle in zip(names, middles):
        out.append(f'<text x="6" y="{middle + 3:.1f}" fill="{DIM}">{name}</text>'
                   f'<line x1="{left}" x2="{width - right}" y1="{middle:.1f}" y2="{middle:.1f}" '
                   f'stroke="{LINE}" />')
    out.append(f'<path d="{path(drawn["rf"], middles[0], reach)}" fill="{GLOW}" '
               f'fill-opacity="0.22" stroke="{GLOW}" stroke-width="1.2" stroke-linejoin="round" />')
    for t, tallness, selective in drawn["bars"]:
        # a band-selective pulse (water suppression, editing) is drawn as a Gaussian, as in Tkac
        # 1999 and Mescher 1998
        if selective:
            bell = " ".join(f"L{x(t) + 9.0 * u:.1f},"
                            f"{middles[0] - reach * tallness * np.exp(-4.5 * u * u):.1f}"
                            for u in np.linspace(-1.0, 1.0, 21))
            out.append(f'<path d="M{x(t) - 9:.1f},{middles[0]:.1f} {bell} '
                       f'L{x(t) + 9:.1f},{middles[0]:.1f}" fill="{GLOW}" fill-opacity="0.3" '
                       f'stroke="{GLOW}" stroke-width="1.1" />')
            continue
        out.append(f'<rect x="{x(t) - 3:.1f}" y="{middles[0] - reach * tallness:.1f}" width="6" '
                   f'height="{reach * tallness:.1f}" fill="{GLOW}" '
                   f'fill-opacity="{0.35 if tallness < 0.75 else 0.85}" stroke="{GLOW}" />')
    for k, axis in enumerate("xyz"):
        traced = path(drawn[axis], middles[k + 1], 0.42 * lane / peak)
        spans = [(a, b) for a, b, which in drawn["crushers"] if which == axis]
        out.append(f'<clipPath id="crush{axis}">' + "".join(
            f'<rect x="{x(a):.1f}" y="{middles[k + 1] - lane / 2}" width="{x(b) - x(a):.1f}" '
            f'height="{lane}" />' for a, b in spans)
            + f'</clipPath><path d="{traced}" fill="{SAND}" fill-opacity="0.08" stroke="{SAND}" '
              f'stroke-width="1.1" />'
              f'<path d="{traced}" fill="{COPPER}" stroke="{SAND}" stroke-width="1.1" '
              f'clip-path="url(#crush{axis})" />')
        # an ideal crusher right after its pulse, the same on every axis
        for t, _ in drawn["marks"]:
            at, base = x(t) + 5.0, middles[k + 1]
            out.append(f'<path d="M{at:.1f},{base:.1f} l2,-{0.3 * lane:.1f} l5,0 '
                       f'l2,{0.3 * lane:.1f}" fill="{COPPER}" fill-opacity="0.25" '
                       f'stroke="{SAND}" stroke-width="0.9" stroke-dasharray="2 1.5" />')
    # the flips below the RF lane, the intervals between the localization's pulses on top
    last = -1e9
    for t, text in drawn["pulses"]:
        if text and x(t) - last > 22.0:
            out.append(f'<text x="{x(t):.1f}" y="{middles[0] + 0.27 * tall:.1f}" fill="{INK}" '
                       f'text-anchor="middle" font-size="10">{text}</text>')
            last = x(t)
    centres = [t for t, _ in drawn["pulses"] if not folded(t)] + [start]
    for a, b in zip(centres[:-1], centres[1:]):
        out.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="14" y2="14" stroke="{DIM}" />'
                   + "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="10" y2="18" '
                             f'stroke="{DIM}" />' for t in (a, b))
                   + (f'<text x="{0.5 * (x(a) + x(b)):.1f}" y="10" fill="{INK}" '
                      f'text-anchor="middle" font-size="9">{b - a:.1f}</text>'
                      if x(b) - x(a) > 30.0 else ""))
    out.append(f'<text x="{0.5 * (x(start) + x(stop)):.1f}" y="17" fill="{GOOD}" '
               f'text-anchor="middle">DAQ</text>')
    wave = [(start + (stop - start) * u, np.exp(-3.5 * u) * np.cos(2.0 * np.pi * 11.0 * u))
            for u in np.linspace(0.0, 1.0, 160)]
    out.append(f'<path d="{path(wave, middles[4], 0.42 * lane)}" fill="none" stroke="{GOOD}" '
               f'stroke-width="1.1" />')
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)
                if (stop - sum(b - a for a, b in folds)) / s <= 10)
    begin = folds[-1][1] if folds else 0.0      # the localization's clock starts after the preludes
    for t in np.arange(0.0, stop - begin + 1e-9, step):
        out.append(f'<line x1="{x(begin + t):.1f}" x2="{x(begin + t):.1f}" y1="{height - 22}" '
                   f'y2="{height - 18}" stroke="{DIM}" />'
                   f'<text x="{x(begin + t):.1f}" y="{height - 7}" fill="{DIM}" '
                   f'text-anchor="middle" font-size="9">{t:g}</text>')
    out.append(f'<text x="6" y="{height - 7}" fill="{DIM}" font-size="9">ms</text></svg>')
    return "".join(out)


#*****************#
#   choose file   #
#*****************#
def choose_folder(prompt):
    """The operating system's folder dialog, as `choose_file`; the chosen path or None."""
    if sys.platform == "darwin":
        command = ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")']
    else:
        command = [sys.executable, "-c",
                   "import tkinter, tkinter.filedialog as d; r = tkinter.Tk(); r.withdraw(); "
                   f"r.attributes('-topmost', True); print(d.askdirectory(title={prompt!r}))"]
    chosen = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    return pathlib.Path(chosen) if chosen else None


def choose_file(prompt):
    """The operating system's own file dialog, as a process of its own, and the chosen path or
    None. The console runs on the operator's machine, so its server can ask; the embedded window's
    dialog (pywebview through NiceGUI) returns empty on macOS and is not relied on. macOS:
    AppleScript's choose file; elsewhere: Tk's, in a Python of its own, because Tk wants a main
    thread."""
    if sys.platform == "darwin":
        command = ["osascript", "-e", f'POSIX path of (choose file with prompt "{prompt}")']
    else:
        command = [sys.executable, "-c",
                   "import tkinter, tkinter.filedialog as d; r = tkinter.Tk(); r.withdraw(); "
                   f"r.attributes('-topmost', True); print(d.askopenfilename(title={prompt!r}))"]
    chosen = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    return pathlib.Path(chosen) if chosen else None


# the spectrum display's chart, the console's and the designer's
SPECTRUM = {
    "backgroundColor": "transparent", "animation": False,
    "title": {"text": "no scan yet",
              "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
    "grid": {"left": 12, "right": 26, "top": 28, "bottom": 44},
    "xAxis": {"type": "value", "inverse": True, "min": -0.5, "max": 6.5, "name": "ppm",
              "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM},
              "splitLine": {"show": False}},
    "yAxis": {"type": "value", "show": False},
    # the wheel zooms along ppm and a drag moves the spectrum both ways, the slider below shows
    # where one is in the acquired band; the height has the slider on the right and stays while one
    # zooms along ppm (the page opens it on what the exam's window shows). The points outside the
    # window stay (filterMode none), so nothing rescales by itself
    "dataZoom": [{"type": "inside", "xAxisIndex": 0, "startValue": 0.2, "endValue": 4.2,
                  "filterMode": "none"},
                 {"type": "slider", "xAxisIndex": 0, "startValue": 0.2, "endValue": 4.2,
                  "height": 14, "bottom": 2, "filterMode": "none",
                  "borderColor": LINE, "fillerColor": "rgba(226,165,95,0.15)",
                  "textStyle": {"color": DIM}},
                 {"type": "inside", "yAxisIndex": 0, "filterMode": "none",
                  "zoomOnMouseWheel": False, "moveOnMouseMove": True},
                 {"type": "slider", "yAxisIndex": 0, "width": 12, "right": 2,
                  "filterMode": "none", "showDetail": False,
                  "borderColor": LINE, "fillerColor": "rgba(226,165,95,0.15)"}],
    # the average, and behind it in grey a transient (the page adds one such series per transient
    # it shows)
    "series": [{"type": "line", "showSymbol": False, "data": [],
                "lineStyle": {"color": SAND, "width": 1.3}, "z": 3,
                "areaStyle": {"color": COPPER, "opacity": 0.12}},
               {"type": "line", "showSymbol": False, "data": [],
                "lineStyle": {"color": DIM, "width": 0.5, "opacity": 0.55}, "silent": True,
                "sampling": "lttb"}],
}


#**********#
#   page   #
#**********#
@ui.page("/")
async def console(client: Client):
    """One operator's console; every browser tab is a session of its own. The page answers at once
    with a loading screen, and the room (the subject's maps, the first planes and the card's outlook
    take seconds) is made behind it."""
    ui.add_head_html(f"<style>{STYLE}</style>")
    ui.dark_mode(True)
    # a decimal point whatever the machine's region: the native input takes the lang
    ui.number.default_props("lang=en-US")
    ui.colors(primary=COPPER, accent=SAND, positive=GOOD, negative=BAD)
    with ui.element("div").style(f"position: fixed; inset: 0; z-index: 9999; "
                                 f"background: {GROUND}; display: flex; flex-direction: column; "
                                 "align-items: center; justify-content: center; "
                                 "gap: 18px") as loading:
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" style="height: 150px" src="{MARK_LARGE}">',
                    sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 22px; letter-spacing: .14em")
        ui.spinner("bars", size="34px", color=SAND)
    await client.connected()
    session = await run.io_bound(Session)
    if session is None:         # the page was left while the room was being made
        return
    (views, shown, hung, letters, crosses, texts, tools, pickers, origins, levels, places, edges,
     chosen) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, [], [],
                {"exam": None, "box": None, "stop": False, "shifts": True, "snr": None,
                 "editing": None, "flipped": False, "ready": False, "dragging": False,
                 "bands": {}, "slab": None,
                 "looking": False, "every": True, "cell": None, "grid": None,
                 "zoom": {"tra": 1.0, "sag": 1.0, "cor": 1.0}, "over": "tra", "slabs": True})

    def room(loops_only=False):
        """Rebuild the 3D view: the three planes hung where they cut the subject (paint fills them,
        with the overlay they carry), the receive loops, the voxel's box. With `loops_only` (an
        array switch) the loops alone are replaced and the planes' textures stay where they are."""
        drawn = session.scene()
        if loops_only:
            for ring in chosen.get("rings", ()):
                ring.delete()
        else:
            view.clear()
            hung.clear()
            shown.clear()
        chosen["rings"] = []
        with view:
            if not loops_only:
                for name, (_, _, _, _, _, corners) in session.planes().items():
                    hung[name] = ui.scene.texture(BLANK, (np.asarray(corners) * MM).tolist())
            for centre, normal, radius in drawn["loops"]:
                n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
                a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
                a = a / np.linalg.norm(a)
                chosen["rings"].append(ui.scene.ring(0.92 * radius * MM, radius * MM, 40).material(
                    COPPER, 0.85).move(*(np.asarray(centre) * MM)).rotate_R(
                    np.stack([a, np.cross(n, a), n], axis=1).tolist()))
            if loops_only:
                return
            centre, size = drawn["voxel"]
            # the saturation slabs, 160 mm wide, their thickness along their normals (a box's z
            # turned onto the normal, as the loops are)
            for axis, offset, thick, *tilted in (session.slabs() if session.protocol.ovs
                                                 and chosen["slabs"] else ()):
                n = np.asarray(tilted[0]) if tilted else np.eye(3)["xyz".index(axis)]
                a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
                a = a / np.linalg.norm(a)
                ui.scene.box(160.0 * MM, 160.0 * MM, thick * MM).material(SLAB, 0.18).move(
                    *((np.asarray(centre) + offset * n) * MM)).rotate_R(
                    np.stack([a, np.cross(n, a), n], axis=1).tolist())
            chosen["box"] = ui.scene.box(*(edge * MM for edge in size), wireframe=True).material(
                GLOW).move(*(np.asarray(centre) * MM))

    async def estimate():
        """The card's compute time beside its scan time, found off the page's thread (a tilted
        slab's design is timed once, a few seconds)."""
        took = await run.io_bound(session.compute_s)
        texts["compute"].text = ("not timed yet" if took is None
                                 else f"{took / 3600.0:.1f} h" if took >= 3600.0
                                 else f"{took / 60.0:.0f} min" if took >= 90.0 else f"{took:.0f} s")

    def paint():
        """Redraw the planes (the slice on view, the voxel's outline where the slice cuts it, the
        other planes' cross, a spectroscopic image's grid) and the 3D box: what a drag needs."""
        card = session.protocol
        counts = ((int(card.matrix), int(card.matrix_y), int(card.slices)) if card.sequence == MRSI
                  else (1, 1, 1))
        for name, (image, (left, top, width, height), cut, (column, row), shifted,
                   corners) in session.planes().items():
            stamp = hash(image.tobytes())
            if shown.get(name) != stamp:  # a moved voxel changes the outlines, not the picture
                shown[name] = stamp
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                views[name].set_source("data:image/png;base64,"
                                       + base64.b64encode(buffer.getvalue()).decode())
                # the same picture in space, see-through where there is nothing
                colour = np.asarray(Image.fromarray(image).convert("RGBA")).copy()
                colour[..., 3] = np.where(colour[..., :3].max(axis=2) > 10, 235, 0)
                buffer = io.BytesIO()
                Image.fromarray(colour).save(buffer, format="PNG")
                hung[name].set_url("data:image/png;base64,"
                                   + base64.b64encode(buffer.getvalue()).decode())
                hung[name].set_coordinates((np.asarray(corners) * MM).tolist())
            side = image.shape[0]
            across, down = {"tra": (counts[0], counts[1]), "sag": (counts[1], counts[2]),
                            "cor": (counts[0], counts[2])}[name]
            grid = "".join(f'<line x1="{left + width * n / across:.1f}" '
                           f'x2="{left + width * n / across:.1f}" y1="{top:.1f}" '
                           f'y2="{top + height:.1f}" />' for n in range(1, across)) + "".join(
                f'<line y1="{top + height * n / down:.1f}" y2="{top + height * n / down:.1f}" '
                f'x1="{left:.1f}" x2="{left + width:.1f}" />'
                for n in range(1, down))
            pen = side / 115.0  # strokes as wide on a fine grid as on the 2 mm one
            # a spectroscopic image that ran lays the chosen line's map over its cells in the
            # transverse plane, the slice on view; the pressed cell is outlined, and Live shows its
            # spectrum
            painted, exam = "", chosen["exam"]
            if name == "tra":
                chosen["grid"] = None
                if exam is not None and exam.maps and cut:
                    heights = exam.maps.get(looks.value, next(iter(exam.maps.values())))
                    nz = heights.shape[2]
                    layer = int(np.clip(round((session.view_mm[2] - exam.position_mm[2])
                                              / exam.size_mm[2] + (nz - 1) / 2.0), 0, nz - 1))
                    chosen["grid"] = (left, top, width, height, layer)
                    painted = "".join(
                        f'<rect x="{left + width * i / across:.1f}" '
                        f'y="{top + height * (down - 1 - j) / down:.1f}" '
                        f'width="{width / across:.1f}" height="{height / down:.1f}" fill="{GLOW}" '
                        f'fill-opacity="'
                        f'{0.75 * heights[i, j, layer] / max(float(heights[..., layer].max()), 1e-12):.2f}" '
                        + (f'stroke="#fff" stroke-width="{0.6 * pen:.2f}" />'
                           if chosen["cell"] == (i, j, layer) else 'stroke="none" />')
                        for i in range(across) for j in range(down))
            chosen["bands"][name] = {which: points for which, points in shifted.items()
                                     if which.startswith("ovs") and chosen["slabs"]}
            views[name].content = (painted +
                f'<g stroke="{SAND}" stroke-width="{0.3 * pen:.2f}" opacity="0.45">'
                f'<line x1="{column}" x2="{column}" y1="0" y2="{side}" />'
                f'<line y1="{row}" y2="{row}" x1="0" x2="{side}" /></g>'
                f'<g fill="none" stroke="{GLOW}" stroke-width="{0.35 * pen:.2f}" opacity="0.7">'
                f'{grid}</g>'
                + "".join('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                          + f'" fill="{SLAB}" fill-opacity="0.22" stroke="{SLAB}" '
                          f'stroke-width="{0.3 * pen:.2f}" />'
                          for which, points in chosen["bands"][name].items())
                + "".join(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'fill="none" stroke="{SHIFT_COLOURS[which]}" '
                          f'stroke-width="{0.5 * pen:.2f}" opacity="{0.9 if cut else 0.4}" />'
                          for which, (x, y, w, h) in (
                              (which, box) for which, box in shifted.items()
                              if not which.startswith("ovs"))
                          if chosen["shifts"] or which == "shim")
                + f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
                  f'fill="none" stroke="{GLOW}" '
                + (f'stroke-width="{0.8 * pen:.2f}" />' if cut else
                   f'stroke-width="{0.5 * pen:.2f}" '
                   f'stroke-dasharray="{2 * pen:.1f} {1.5 * pen:.1f}" opacity="0.55" />'))
            axis, letter = EDGES[name][4:]
            texts[name].text = f"{name.upper()}  {letter} {session.view_mm[axis]:+.0f} mm" + (
                f"  ×{chosen['zoom'][name]:.2g}" if chosen["zoom"][name] != 1.0 else "")
            # a zoom holds the point the planes cross at
            origins[name] = f"{100.0 * column / side:.1f}% {100.0 * row / side:.1f}%"
        chosen["box"].move(*(np.asarray(session.position_mm) * MM))
        # a flipped view mirrors the pictures left to right, and the letters with them
        for name, labels in letters.items():
            mirrored = chosen["flipped"] and name in ("tra", "cor")
            views[name].style(f"transform-origin: {origins[name]}; transform: scale("
                              f"{-chosen['zoom'][name] if mirrored else chosen['zoom'][name]:g}, "
                              f"{chosen['zoom'][name]:g})")
            top, bottom, left, right = EDGES[name][:4]
            labels[2].text, labels[3].text = (right, left) if mirrored else (left, right)
            crosses[name].content = (
                f'<svg width="54" height="54" viewBox="0 0 54 54" font-size="9" '
                f'font-family="inherit"><g stroke="{SAND}" fill="{SAND}">'
                f'<line x1="20" y1="34" x2="46" y2="34" /><line x1="20" y1="34" x2="20" y2="8" />'
                f'<path d="M46,34 l-5,-3 v6 z" /><path d="M20,8 l-3,5 h6 z" /></g>'
                f'<text x="48" y="46" fill="{SAND}" text-anchor="end">{labels[3].text}</text>'
                f'<text x="6" y="12" fill="{SAND}">{top}</text></svg>')
        # the overlays there are for this room: a B0 or B1+ map only where one is on (maps), a lipid
        # layer where the subject has one
        there = [name for name in OVERLAYS if {
            "B0 field": session.field != "uniform", "B1+": session.transmit is not None,
            "lipid layer": getattr(session.body(), "fat", None) is not None}.get(name, True)]
        if chosen.get("overlays") != there:
            chosen["overlays"] = there
            session.overlay = session.overlay if session.overlay in there else "anatomy"
            overlays.clear()
            with overlays:
                for name in there:
                    ui.menu_item(name,
                                 lambda name=name: (setattr(session, "overlay", name), draw()))
        tools["overlay"].text = session.overlay
        wash = session.wash()
        texts["window"].text = ("" if wash is None or not wash[3] else
                                f"blue {wash[1]:+.3g} · red {wash[2]:+.3g}"
                                + (" Hz" if session.overlay == "B0 field" else "")
                                ) if session.overlay != "anatomy" else ""
        if session.overlay in ("B0 field", "B1+") and wash is None:
            texts["window"].text = "no such map on: choose one in maps"
        for axis, field in enumerate(places):
            field.value = round(session.position_mm[axis], 1)

    def draw():
        """Redraw everything the card and the room decide: the planes, the diagram, the outlook."""
        # a bound control speaks its first value while the page is still being built
        if not chosen["ready"]:
            return
        paint()
        card = session.protocol
        try:
            diagram.content = lanes(session.diagram())
            look = session.outlook()
        except Exception as error:
            # a card that is no design yet (a TE its pulses do not fit): say why, keep the rest
            texts["verdict"].text = str(error)[:160]
            texts["verdict"].style(f"color: {BAD}")
            return
        minutes, seconds = divmod(int(round(look["scan_time_s"])), 60)
        texts["name"].text = (card.sequence if chosen["editing"] is None
                              else f"editing {chosen['editing'].name}  →  {card.sequence}")
        accept.text = "Accept" if chosen["editing"] is None else "Accept changes"
        discard.set_visibility(chosen["editing"] is not None)
        texts["time"].text = texts["time2"].text = f"{minutes:02d}:{seconds:02d}"
        if chosen.get("estimate") is None or chosen["estimate"].done():
            chosen["estimate"] = background_tasks.create(estimate())
        texts["timing"].text = f"{card.tr_ms:g} / {card.te_ms:g} ms"
        texts["needs"].text = (f"{1000.0 * look['needs_s']:.0f} ms of the TR "
                               f"(suppression, OVS, sequence, readout)")
        texts["needs"].style(f"color: {BAD if 1000.0 * look['needs_s'] > card.tr_ms else INK}")
        texts["volume"].text = f"{np.prod(session.size_mm) / 1000.0:.1f} mL"
        # against the card as it was taken from the browser: the voxel's volume and the root of the
        # averages alone
        signal = np.prod(session.size_mm) * np.sqrt(1 if card.sequence == MRSI else card.averages)
        chosen["snr"] = signal if chosen["snr"] is None else chosen["snr"]
        texts["snr"].text = f"{100.0 * signal / chosen['snr']:.0f} %  (volume, averages)"
        texts["displaced"].text = "\n".join(
            f"{which} " + " ".join(f"{v:+.1f}" for v in moved) + " mm"
            for which, moved in look["displaced"].items()) if any(
            any(moved) for moved in look["displaced"].values()) else (
            "none, the localization is ideal")
        texts["linewidth"].text = ("uniform field, none" if look["linewidth_hz"] is None else
                                   f"{look['linewidth_hz']:.1f} Hz on a 1 Hz line, "
                                   f"{session.shim} shim")
        texts["lipid"].text = ("this anatomy has no lipid layer" if look["lipid_mm"] is None else
                               f"{look['lipid_mm'][0]:.0f} mm from the voxel, "
                               f"{look['lipid_mm'][1]:.0f} mm from where lipid is selected")
        texts["lipid"].style(
            f"color: "
            f"{BAD if look['lipid_mm'] is not None and min(look['lipid_mm']) < 1.0 else INK}")
        texts["tissue"].text = f"GM {look['gm']:.2f}  WM {look['wm']:.2f}  CSF {look['csf']:.2f}"
        texts["encoding"].text = (look["encoding"]["said"] if look["encoding"]
                                  else "a single voxel has none")
        texts["resolution"].text = f"{card.bandwidth_hz / card.npoints:.2f} Hz / point"
        texts["readout"].text = f"{1000.0 * card.npoints / card.bandwidth_hz:.0f} ms"
        texts["limits"].text = "B1 {:g} uT · G {:g} mT/m · slew {:g} T/m/s".format(*look["limits"])
        texts["verdict"].text = ("plays on this scanner" if not look["violations"]
                                 else "; ".join(look["violations"]))
        texts["verdict"].style(f"color: {GOOD if not look['violations'] else BAD}")
        records.clear()
        # the draw of the population that is scanned, at the planned voxel: what the exams will scan
        with records:
            where = session.view_mm if chosen["looking"] else session.position_mm
            panel.text = "Where you look (shift)" if chosen["looking"] else "This voxel"
            # a group under its heading: short names two to a row
            for group, values in session.place_of(where).items():
                ui.label(group).classes("label").style(
                    f"width: 100%; margin-top: 8px; padding-top: 5px; "
                    f"border-top: 1px solid {LINE}; color: {SAND}")
                short = all(len(name) <= 5 for name in values)
                with ui.element("div").classes("info w-full").style(
                        f"grid-template-columns: {'auto 1fr auto 1fr' if short else '1fr auto'}; "
                        "column-gap: 8px; white-space: normal"):
                    for name, value in values.items():
                        ui.label(name).style(f"color: {DIM}")
                        ui.label(f"{value:.2f}" if abs(value) < 10.0 else f"{value:.1f}"
                                 if abs(value) < 100.0 else f"{value:.0f}").classes("value").style(
                            "text-align: right")
        for name, tool in tools.items():
            on = {"ovs": card.ovs, "shifts": chosen["shifts"], "slabs": chosen["slabs"],
                  "flip": chosen["flipped"], "overlay": session.overlay != "anatomy",
                  "real": card.real or card.sequence == "HERCULES"}.get(
                name, card.water_suppression in SUPPRESSION)
            tool.classes(add="on" if on else "", remove="" if on else "on")
        tools["water"].text = "water " + card.water_suppression
        tools["water"].props(
            f"icon="
            f"{'water_drop' if card.water_suppression in SUPPRESSION else 'format_color_reset'}")

    def listing():
        """Redraw the exam cards."""
        rows.clear()
        with rows:
            for exam in session.queue:
                minutes, seconds = divmod(int(round(exam.card.scan_time_s)), 60)
                with ui.element("div").classes(
                        "exam" + (" chosen" if exam is chosen["exam"] else "")).on(
                        "click", lambda _, exam=exam: choose(exam)).mark(f"card {exam.name}"):
                    state = ui.icon({"planned": "edit" if exam is chosen["editing"]
                                     else "radio_button_unchecked", "running": "autorenew",
                                     "done": "check_circle", "failed": "error"}[exam.status],
                                    size="16px").style(
                        f"color: "
                        f"{dict(planned=DIM, running=SAND, done=GOOD, failed=BAD)[exam.status]}")
                    if exam.problem:
                        state.tooltip(exam.problem[:200])
                    with ui.column().style("flex: 1; gap: 0; min-width: 0"):
                        ui.label(exam.name)
                        # the field the exam ran at
                        field = (exam.room.get("field_t")
                                 or PRESETS[exam.room.get("scanner", session.scanner)].b0_t)
                        ui.label(f"{SCANNERS.get(exam.room.get('scanner'), '')} at {field:g} T · "
                                 f"{str(exam.room.get('subject', '')).split(',')[0]}").style(
                            f"color: {DIM}; font-size: 10px; white-space: nowrap; "
                            f"overflow: hidden; text-overflow: ellipsis")
                    ui.label(f"{minutes:02d}:{seconds:02d}").classes("value").style(f"color: {DIM}")
                    ui.icon("content_copy", size="14px").style(f"color: {DIM}").tooltip(
                        "duplicate as a new planned card").on(
                        "click.stop", lambda _, exam=exam: (
                            choose(exam), chosen.__setitem__("editing", None), plan()))
                    if exam.status == "planned":
                        ui.icon("close", size="14px").style(f"color: {DIM}").tooltip("remove").on(
                            "click.stop", lambda _, exam=exam: (
                                session.queue.remove(exam),
                                chosen.__setitem__("editing", None if chosen["editing"] is exam
                                                   else chosen["editing"]),
                                chosen.__setitem__("exam", None if chosen["exam"] is exam
                                                   else chosen["exam"]),
                                listing(), draw()))

    def frame(exam, index):
        """The Live display at one step of the build-up: the average so far in the combination or
        condition chosen and, grey behind it, the transient that just came in; at the end every
        transient that was kept, coils combined (the "every transient" switch), or the chosen cell
        of a spectroscopic image."""
        name = looks.value if looks.value in exam.builds else exam.shown
        last = index in (-1, len(exam.frames) - 1)
        count, real = (exam.builds[name][min(index, len(exam.builds[name]) - 1)] if exam.builds
                       else exam.frames[index][:2])
        behind = (exam.transients.get(name, []) if last and chosen["every"]
                  else [exam.frames[index][2]] if name == exam.shown and len(exam.frames) > 1
                  else [])
        if exam.cells is not None and chosen["cell"] is not None:
            real = exam.cells[chosen["cell"]]
        spectrum.options["series"] = [{**SPECTRUM["series"][0], "data": [
            [float(p), float(s)] for p, s in zip(exam.ppm, real)]}] + [
            {**SPECTRUM["series"][1], "data": [[float(p), float(s)]
                                               for p, s in zip(exam.ppm, single)]}
            for single in behind]
        # what the exam opens on: its window along ppm, and the height of what that window shows
        if index == 0 or last:
            zooms = spectrum.options["dataZoom"]
            for zoom in zooms[:2]:
                zoom["startValue"], zoom["endValue"] = exam.window
            spectrum.options["xAxis"].update(min=float(exam.ppm.min()), max=float(exam.ppm.max()))
            everything = [v for series in spectrum.options["series"] for _, v in series["data"]]
            inside = [v for series in spectrum.options["series"] for p, v in series["data"]
                      if exam.window[0] <= p <= exam.window[1]]
            # the slider's whole reach: the water line too
            spectrum.options["yAxis"].update(min=min(everything), max=max(everything))
            room = 0.06 * max(max(inside) - min(inside), 1e-12)
            for zoom in zooms[2:]:
                zoom["startValue"], zoom["endValue"] = min(inside) - room, max(inside) + room
        spectrum.options["title"]["text"] = (
            f"{exam.name}   ·   cell {chosen['cell'][0] + 1}, {chosen['cell'][1] + 1}"
            + (f", slice {chosen['cell'][2] + 1}" if exam.cells.shape[2] > 1 else "")
            if exam.cells is not None and chosen["cell"] is not None else
            f"{exam.name}   ·   press a cell on the transverse plane"
            if exam.cells is not None else
            f"{exam.name}   ·   transient {count} of {exam.card.averages}" + (
                f"   ·   SNR {exam.snr[0]:.0f}, of one transient {exam.snr[1]:.0f}" + (
                    f"   ·   {exam.rejected} left out as unlike the rest" if exam.rejected else "")
                if exam.snr and last else ""))
        spectrum.update()
        progress.value = count / max(exam.card.averages, 1)
        texts["count"].text = f"{count} / {exam.card.averages}"

    async def reprocess(level):
        """The display's processing (Session.display): kept for the next exams, and the chosen exam
        shown anew."""
        session.processing = level
        for name, button in levels.items():
            button.classes(add="on" if name == level else "", remove="" if name == level else "on")
        exam = chosen["exam"]
        if exam is not None and exam.schedule:
            await run.io_bound(session.display, exam)
            frame(exam, -1)

    def blank(said):
        """The Live display without a spectrum: a card that has not run yet."""
        spectrum.options["series"] = [{**SPECTRUM["series"][0], "data": []}]
        spectrum.options["title"]["text"] = said
        spectrum.update()
        progress.value = 0.0
        texts["count"].text = f"0 / {session.protocol.averages}"

    def choose(exam):
        """Select an exam: its voxel returns to the planes, its card to the tabs, its spectrum to
        the display. A planned exam is then being edited: Accept puts the card back in its place."""
        chosen["exam"], chosen["editing"], chosen["cell"] = (
            exam, exam if exam.status == "planned" else None, None)
        looks.set_options(list(exam.builds or exam.maps),
                          value=exam.shown if exam.builds else next(iter(exam.maps), None))
        for field in dataclasses.fields(exam.card):  # in place: the tabs' fields hold this card
            setattr(session.protocol, field.name, getattr(exam.card, field.name))
        session.position_mm, session.view_mm, session.size_mm = (
            list(exam.position_mm), list(exam.position_mm), list(exam.size_mm))
        for axis, field in enumerate(edges):
            field.value = exam.size_mm[axis]
        for name, value in exam.room.items():  # the room it was planned in comes back with it
            setattr(session, name, value)
        for name in ("scanner", "coil", "subject"):
            pickers[name].set_value(getattr(session, name))
        pickers["field_t"].set_value(session.profile().scanner.b0_t)
        frame(exam, -1) if exam.frames else blank(f"{exam.name}   ·   {exam.status}")
        listing()
        room()
        draw()

    def plan():
        chosen["exam"] = session.plan(instead_of=chosen["editing"])
        chosen["editing"] = None
        blank(f"{chosen['exam'].name}   ·   planned")
        listing()
        draw()

    async def scan_queue():
        """Run every planned exam in order; each shows as its transients come in. Stop holds
        the queue after the exam that is running, which the engine acquires in one piece."""
        start.disable()
        chosen["stop"] = False
        for exam in [e for e in session.queue if e.status == "planned"]:
            if chosen["stop"]:
                break
            chosen["exam"], exam.status = exam, "running"
            status.text = f"scanning {exam.name}"
            shows.value = "Live"
            progress.props("indeterminate")
            listing()
            began = time.perf_counter()
            try:
                await run.io_bound(session.run, exam)
            except Exception as error:
                # a card the engine cannot play: the queue goes on, the card says why
                exam.status, exam.problem = "failed", str(error)
                progress.props(remove="indeterminate")
                listing()
                continue
            texts["took"].text = (f"{exam.name}: simulated in {time.perf_counter() - began:.1f} s, "
                                  f"a scanner takes {exam.card.scan_time_s:.0f} s")
            progress.props(remove="indeterminate")
            looks.set_options(list(exam.builds or exam.maps),
                              value=exam.shown if exam.builds else next(iter(exam.maps), None))
            for index in range(len(exam.frames)):
                frame(exam, index)
                await asyncio.sleep(0.03)
            listing()
            paint()                     # a spectroscopic image lays its map over the planned grid
        status.text = "idle"
        start.enable()

    def export():
        """Export the chosen exam: where to, in which format, and where the file then is."""
        exam = chosen["exam"]
        if exam is None or exam.data is None:
            ui.notify("Select an exam that has run (press its card)", type="warning")
            return

        def write():
            try:
                path = session.export(exam, folder.value)
            except Exception as error:
                done.text = f"Not exported: {error}"[:200]
                return
            done.text = f"Written: {path}"
            reveal.set_visibility(True)
            reveal.on("click", lambda: subprocess.run(
                ["open", "-R", str(path)] if sys.platform == "darwin" else
                ["xdg-open", str(path.parent)], check=False))

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 640px"):
            ui.label(f"Export · {exam.name}").classes("label")
            folder = ui.input("folder", value=str(pathlib.Path.home() / "MRSandbox" / "exports")
                              ).props("dense outlined dark").classes("w-full")
            ui.select(["NIfTI-MRS, the transients apart, with the simulation's truth in the "
                       "header"],
                      value=("NIfTI-MRS, the transients apart, with the simulation's truth in the "
                             "header"), label="format").props(
                "dense outlined dark").classes("w-full").tooltip("more formats are to come")
            done = ui.label("").classes("value").style("white-space: normal; word-break: break-all")
            with ui.row().classes("w-full"):
                reveal = ui.button("Show in its folder", icon="folder_open").props(
                    "outline dense no-caps")
                reveal.set_visibility(False)
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense no-caps")
                ui.button("Export", icon="save_alt", on_click=write).mark("write the export").props(
                    "unelevated dense no-caps")
        dialog.open()

    def chip(source, provenance=""):
        """The source of a value as one coloured word."""
        return ui.label(SOURCE_WORDS[source]).style(
            f"color: {SOURCE_COLORS[source]}; width: 74px; font-size: 12px; letter-spacing: .04em"
        ).tooltip(SOURCE_SAYS[source] + (f": {provenance}" if provenance else ""))

    def row(name, tip=None):
        """One line of a card: the name on the left, the rest as the caller fills it."""
        holder = ui.row().classes("w-full items-center no-wrap").style(
            f"gap: 10px; min-height: 40px; border-bottom: 1px solid {LINE}")
        with holder:
            label = ui.label(name).style("width: 250px; font-size: 13px")
            if tip:
                label.tooltip(tip)
        return holder

    def line(holder, path, priors, sources, after=None):
        """One parameter's line: its name, the value in force, its source as a word, and one select
        that chooses the source or sets the value (a number, a range, a map). `after` runs after a
        change."""
        kind, numbers = priors[path]
        own = session.overrides.get(path)
        info = sources.get(path)
        chosen = (("custom" if own is not None else info["chosen"]) if info
                  else ("custom" if own is not None else plain_word(path)))
        provenance = (info["provenance"].get(chosen, "") if info
                      else (plain_why(path) if own is None else ""))
        middle = (numbers[0] if kind in ("fixed", "normal", "empirical")
                  else 0.5 * (numbers[0] + numbers[1]) if kind == "range" else 0.0)
        spread = (numbers if kind == "range" else numbers[1:3] if kind == "empirical"
                  else (numbers[0] - 2.0 * numbers[1], numbers[0] + 2.0 * numbers[1])
                  if kind == "normal" else (0.9 * middle, 1.1 * middle))
        available = set(info["sources"]) if info else {plain_word(path)}
        options = {**{name: SOURCE_WORDS[name] for name in SOURCE_WORDS if name != "custom"},
                   **({"set": "set to", "vary": "vary between", "map": "from a map"}
                      if kind != "other" else {"map": "from a map"})}
        way = (("map" if isinstance(own, str) else "vary" if isinstance(own, tuple) else "set")
               if own is not None else chosen)

        def why_not(name):
            """Why a line cannot take a source, in words."""
            kinds = reads(path)
            if name == "own":
                return "no scan of yours read for it yet: " + (
                    f"a {' or a '.join(kinds)} gives it" if kinds
                    else "no reader exists for this line yet")
            if name == "pooled":
                return "no dataset the package read gives it yet" + (
                    f" (a {' or a '.join(kinds)} would)" if kinds
                    else " and no reader exists for it")
            if name == "dataset":
                return ("no dataset on the floor (the scanner card's dataset line), or it holds no "
                        "reading of this line")
            if name == "literature":
                return "nothing published gives it: the package's range is stated"
            return "the package assumed no range here: it is typed from the literature"

        async def provide():
            """A file of yours for this line: one read this session that gives it, or a new one."""
            with ui.dialog() as picker, ui.card().classes("pane").style("min-width: 640px"):
                ui.label(f"yours, for {NAMES.get(path, path)}").classes("label")
                kinds = reads(path)
                ui.label(("read from a " + " or a ".join(kinds)) if kinds
                         else "no reader exists for this line yet").classes("label").style(
                    "text-transform: none; letter-spacing: 0")
                for file, kind_, gave in session.uploads_for(path):
                    with ui.row().classes("w-full items-center no-wrap").style("gap: 10px"):
                        ui.label(pathlib.Path(file).name).classes("value").style("width: 300px")
                        ui.label(f"{kind_}: {gave}").classes("label").style(
                            "text-transform: none; letter-spacing: 0; flex: 1")
                if kinds:
                    async def chosen_file():
                        chosen_path = await run.io_bound(
                            choose_file, f"A {' or a '.join(kinds)} for {NAMES.get(path, path)}")
                        if chosen_path is None:
                            return
                        report = await run.io_bound(session.replicate, chosen_path)
                        ui.notify("; ".join(f"{file}: {gave}" for file, _, gave in report),
                                  multi_line=True)
                        picker.close()
                        redraw()
                    ui.button("choose a file", icon="folder_open", on_click=chosen_file).props(
                        "outline dense no-caps").style("width: 150px")
                ui.button("Close", on_click=picker.close).props("flat dense no-caps")
            picker.open()

        def redraw():
            line(holder, path, priors, sources, after)
            if after is not None:
                after()

        def switch(event):
            if event.value in SOURCE_WORDS and event.value not in available:
                if event.value == "own":
                    asyncio.ensure_future(provide())
                else:
                    ui.notify(f"{SOURCE_WORDS[event.value]}: {why_not(event.value)}",
                              type="warning", multi_line=True)
                line(holder, path, priors, sources, after)  # the select back to what holds
                return
            session.overrides.pop(path, None)
            if event.value in SOURCE_WORDS and info:
                session.choices[path] = event.value
            elif event.value == "set":
                session.overrides[path] = float(middle)
            elif event.value == "vary":
                session.overrides[path] = (float(min(spread)), float(max(spread)))
            if event.value != "map":
                redraw()

        def put(index):
            def handle(event):
                if event.value is None:
                    return
                now = session.overrides[path]
                if isinstance(now, tuple):
                    values = list(now)
                    values[index] = float(event.value)
                    session.overrides[path] = (min(values), max(values))
                else:
                    session.overrides[path] = float(event.value)
                if after is not None:
                    after()
            return handle

        async def taken(event):
            # onto the subject's grid, by the scanner's coordinates
            session.overrides[path] = str(session.onto(await keep(event)))
            redraw()

        holder.clear()
        with holder:
            ui.label(NAMES.get(path, path.split(".", 1)[-1])).style(
                f"width: 250px; font-size: 13px; color: {SAND if own is not None else INK}")
            ui.label(SHOWN[kind].format(*numbers) if kind != "other" else str(numbers)[:40]
                     ).classes("value").style("width: 260px")
            chip(chosen, provenance)
            ui.select(options, value=way if way in options else chosen, on_change=switch).props(
                "dense outlined dark options-dense").style("width: 150px").tooltip(
                "\n".join(f"{SOURCE_WORDS[name]}: "
                          f"{SHOWN.get(k, '{}').format(*n) if k != 'other' else n}; "
                          f"{info['provenance'].get(name, '')}"
                          for name, (k, n) in info["sources"].items()) if info
                else "set to a number, vary between two, or read a map")
            if way == "map":
                if isinstance(own, str):
                    ui.label(pathlib.Path(own).name).classes("value")
                ui.upload(label="a NIfTI in this parameter's units", auto_upload=True,
                          on_upload=taken).props(
                    "flat dense dark accept=.nii,.gz").style("width: 240px")
            elif own is not None:
                for index, number in enumerate(own if isinstance(own, tuple) else (own,)):
                    ui.number(value=number, format="%.6g", on_change=put(index)).props(
                        "dense outlined dark").style("width: 110px")
                    if index == 0 and isinstance(own, tuple):
                        ui.label("to").classes("label")

    def lines(body, priors, sources, after=None, groups=True):
        """The lines of `priors` in `body`: with `groups`, one line per group carrying the group's
        source (a select for all its lines) and its members under it, opened on click; without, the
        lines alone. Returns build(wanted), which refills them (a search opens what it finds)."""
        def build(wanted=""):
            body.clear()
            found = {}
            for path in priors:
                if wanted.lower() in path.lower() or wanted.lower() in NAMES.get(path, "").lower():
                    found.setdefault(path.split(".")[0] if groups else "", []).append(path)
            with body:
                for group, paths in found.items():
                    if not groups:
                        for path in paths:
                            line(ui.row().classes("w-full items-center no-wrap").style(
                                f"gap: 10px; min-height: 40px; border-bottom: 1px solid {LINE}"),
                                 path, priors, sources, after)
                        continue
                    words = {(sources[path]["chosen"] if path in sources else plain_word(path))
                             if session.overrides.get(path) is None else "custom"
                             for path in paths}
                    common = next(iter(words)) if len(words) == 1 else None
                    offered = {name: SOURCE_WORDS[name] for name in SOURCE_WORDS
                               if name != "custom"}

                    def for_all(event, paths=paths):
                        if event.value not in ("literature", "stated", "pooled", "dataset", "own"):
                            return
                        taking = [path for path in paths if event.value in (
                            set(sources[path]["sources"]) if path in sources
                            else {plain_word(path)})]
                        for path in taking:
                            session.overrides.pop(path, None)
                        session.choose_all(taking, event.value)
                        if len(taking) < len(paths):
                            ui.notify(f"{SOURCE_WORDS[event.value]}: {len(taking)} of {len(paths)} "
                                      f"lines have it; the others keep theirs",
                                      type="warning" if not taking else "info")
                        build(wanted)
                        draw()
                        if after is not None:
                            after()

                    def fill(event, paths=paths, done=[]):
                        if event.value and not done:
                            done.append(True)
                            with event.sender:
                                for path in paths:
                                    line(ui.row().classes("w-full items-center no-wrap").style(
                                        f"gap: 10px; min-height: 40px; "
                                        f"border-bottom: 1px solid {LINE}; margin-left: 20px"),
                                         path, priors, sources, after)

                    with ui.column().classes("w-full").style("gap: 0"):
                        with ui.row().classes("w-full items-center no-wrap").style(
                                f"gap: 10px; min-height: 40px; border-bottom: 1px solid {LINE}"):
                            fold = ui.expansion(f"{GROUPS.get(group, group)}   ·   {len(paths)}",
                                                on_value_change=fill).props("dense dark").style(
                                "width: 520px")
                            chip(common or "custom",
                                 "" if common else "the lines differ: open the group")
                            if not common:
                                ui.label("mixed").classes("label")
                            ui.select(offered, value=common if common in offered else None,
                                      label="for all lines", on_change=for_all).props(
                                "dense outlined dark options-dense").style("width: 150px")
                        if wanted:
                            fold.value = True

        return build

    def gear(panel=None):
        """Every setting of the simulation as lines, grouped (the top bar's "all settings"); the
        search narrows them. Each piece's own card holds its own lines the same way."""
        priors = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves(own=False)
                  if panel is None or path.split(".")[0] in GEARS[panel]}
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1000px; max-width: 1140px"):
            with ui.row().classes("w-full items-center"):
                ui.label("All settings").classes("label")
                ui.space()
                ui.input(placeholder="search, e.g. NAA or t2",
                         on_change=lambda e: build(e.value or "")).props(
                    "dense outlined dark clearable debounce=300").style("width: 260px")
                ui.button("Reset all", icon="restart_alt", on_click=lambda: (
                    session.overrides.clear(), session.choices.clear(), build(), draw())).props(
                    "flat dense no-caps")
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props(
                    "unelevated dense no-caps")
            ui.label(SAID).classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.scroll_area().style("height: 620px; width: 100%"):
                body = ui.column().classes("w-full").style("gap: 0")
            build = lines(body, priors, session.sources())
            build()
        dialog.open()

    def designer():
        """The sequence designer: the design as a strip of blocks in the order they play, over its
        diagram; a block is chosen on the strip, set in the inspector beside the palette, and new
        blocks go in after it. The diagram marks the chosen block, and the design's spectrum at
        the planned voxel is one press away."""
        state, switches = {"at": 0}, {}

        def refresh():
            """Redraw the diagram with the chosen block marked, the strip, the inspector, and the
            card's three switches as they stand."""
            card = session.protocol
            for name, on in (("real", card.real), ("ovs", card.ovs),
                             ("water", card.water_suppression in SUPPRESSION)):
                switches[name].classes(add="on" if on else "", remove="" if on else "on")
            switches["water"].text = "water " + card.water_suppression
            strip.clear()
            with strip:
                for index, (kind, settings) in enumerate(session.design):
                    brief = next((f"{value:g}" + unit
                                  for key, unit in (("flip_deg", "°"), ("ms", " ms"),
                                                    ("cycles_per_mm", " /mm"), ("order", ""))
                                  if (value := settings.get(key)) is not None),
                                 settings.get("scheme", ""))
                    chip = ui.button(f"{index + 1} · {block_name(kind, settings)} {brief}",
                                     icon=BLOCK_ICONS[kind],
                                     on_click=lambda _, i=index: (
                                         state.__setitem__("at", i), refresh())).props(
                        "flat dense no-caps stack draggable").classes(
                        "chip" + (" on" if index == state["at"] else ""))
                    chip.on("dragstart", lambda _, i=index: state.__setitem__("dragged", i))
                    chip.on("dragover.prevent", lambda: None)
                    # the bar where the block will land
                    chip.on("dragenter", lambda _, chip=chip: chip.classes(add="over"))
                    chip.on("dragleave", lambda _, chip=chip: chip.classes(remove="over"))
                    chip.on("drop", lambda _, i=index: drop(i))
                    with chip, ui.context_menu().props("dark"):
                        ui.menu_item("Copy", lambda i=index: state.__setitem__(
                            "copied", (session.design[i][0], dict(session.design[i][1]))))
                        ui.menu_item("Paste after this", lambda i=index: "copied" in state and (
                            state.__setitem__("at", i),
                            place(state["copied"][0], dict(state["copied"][1]))))
                        ui.menu_item("Remove",
                                     lambda i=index: (state.__setitem__("at", i), remove()))
            inspector.clear()
            with inspector:
                if session.design:
                    kind, settings = session.design[state["at"]]
                    with ui.row().classes("w-full items-center"):
                        ui.icon(BLOCK_ICONS[kind], size="20px").style(f"color: {GLOW}")
                        ui.label(f"Block {state['at'] + 1} · {block_name(kind, settings)}").style(
                            f"color: {SAND}")
                        ui.space()
                        if kind in ("hard pulse", "shaped pulse"):
                            ui.button(
                                "make it " + ("ideal" if kind == "shaped pulse" else "shaped"),
                                icon="swap_horiz", on_click=lambda: (
                                    reshape(state["at"], kind == "hard pulse"), refresh())
                            ).props("outline dense no-caps")
                        for icon, tip, act in (
                                ("arrow_back", "earlier", lambda: move(-1)),
                                ("arrow_forward", "later", lambda: move(+1)),
                                ("content_copy", "copy",
                                 lambda: state.__setitem__("copied", (kind, dict(settings)))),
                                ("content_paste", "paste the copied block after this one",
                                 lambda: "copied" in state and place(
                                     state["copied"][0], dict(state["copied"][1]))),
                                ("delete", "remove", remove)):
                            ui.button(icon=icon, on_click=act).props(
                                "flat dense round").tooltip(tip)
                    ui.label(BLOCK_NOTES[kind]).classes("label").style(
                        "text-transform: none; letter-spacing: 0")
                    with ui.row().classes("w-full").style("gap: 8px"):
                        for key, value in settings.items():
                            def put(event, key=key, number=not isinstance(value, str)):
                                if event.value is not None:
                                    settings[key] = float(event.value) if number else event.value
                                    if kind == FIXED:  # the block's words follow its settings
                                        settings["what"] = named(adjusted(settings))
                                    redraw()
                            # an event taken over from the card: said, not set
                            if key in ("what", "event"):
                                if key == "what":
                                    ui.label(value).classes("value")
                            elif key in CHOICES:
                                ui.select(CHOICES[key], value=value, label=CAPTIONS[key],
                                          on_change=put).props("dense outlined dark").style(
                                    "width: 170px")
                            elif isinstance(value, str):
                                ui.input(CAPTIONS[key], value=value, on_change=put).props(
                                    "dense outlined dark debounce=400").style("width: 100%")
                                ui.upload(label="or a text file of amplitudes", auto_upload=True,
                                          on_upload=envelope).props(
                                    "flat dense dark accept=.txt,.csv").style("width: 280px")
                            else:
                                ui.number(CAPTIONS[key], value=value, format="%.6g",
                                          on_change=put).props("dense outlined dark").style(
                                    "width: 170px")
            redraw()

        def redraw():
            try:
                # the design after the card's preludes, as the card plays it
                preludes = session.preludes(session.protocol)
                drawn = session.diagram(session.designed(), preludes=preludes)
                counts = sum(len(design.events) for design in preludes.values()) + np.cumsum(
                    [0] + [len(BLOCKS[kind][1](settings)) for kind, settings in session.design])
                spans = (drawn["spans"][counts[state["at"]]:counts[state["at"] + 1]]
                         if session.design else [])
                preview.content = lanes(drawn, width=820.0, lane=44.0, mark=(
                    spans[0][0], spans[-1][1],
                    9.5 if session.design[state["at"]][0] == "coherence filter" else 0.0)
                    if spans else None)
                problem.text = (f"{len(session.design)} blocks · the readout opens at "
                                f"{drawn['adc'][0]:.1f} ms")
                problem.style(f"color: {DIM}")
            except Exception as error:
                # a half-typed block is not a design yet: say why and keep the last diagram
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")

        def parts():
            """The design's sub-acquisitions and how they combine: one is a plain sequence; several
            are interleaved over the transients as an edited scan is, and the weights say what is
            shown."""
            def another():
                name = next(chr(65 + k) for k in range(26) if chr(65 + k) not in session.designs)
                session.designs[name] = [(kind, dict(settings))
                                         for kind, settings in session.design]
                session.condition = name
                parts()
                refresh()

            def cycle(steps):
                """The chosen pulse phase cycled over `steps` sub-acquisitions; a block that is
                no pulse is said so, not raised."""
                try:
                    session.phase_cycle(state["at"], steps)
                except ValueError as error:
                    ui.notify(str(error), type="warning")
                    return
                parts()
                refresh()

            def without():
                session.designs.pop(session.condition)
                session.condition = next(iter(session.designs))
                state["at"] = 0
                parts()
                refresh()

            def scheme(rows):
                names = list(session.designs)
                session.weights = {label: dict(zip(names, signs)) for label, signs in rows}
                parts()

            several.clear()
            with several:
                with ui.row().classes("w-full items-center").style("gap: 6px"):
                    ui.label("sub-acquisitions").classes("label")
                    for name in session.designs:
                        ui.button(name, on_click=lambda _, name=name: (
                            setattr(session, "condition", name), state.__setitem__("at", 0),
                            parts(), refresh())).props("flat dense no-caps").classes(
                            "tool" + (" on" if name == session.condition else "")).style(
                            "width: 44px; height: 30px")
                    ui.button("another, a copy of this one", icon="add", on_click=another).props(
                        "flat dense no-caps")
                    if session.design and "phase_deg" in session.design[state["at"]][1]:
                        for steps in (2, 4):
                            ui.button(f"phase cycle the chosen pulse, {steps} steps", icon="cached",
                                      on_click=lambda _, steps=steps: cycle(steps)).props(
                                "flat dense no-caps").tooltip(
                                "the pulse's phase steps over the transients and the receiver "
                                "follows, so the wanted pathway adds and the rest cancels")
                    if len(session.designs) > 1:
                        ui.button("remove this one", icon="remove", on_click=without).props(
                            "flat dense no-caps")
                        ui.space()
                        if len(session.designs) == 2:
                            ui.button("difference", on_click=lambda: scheme(
                                (("difference", (1.0, -1.0)), ("sum", (1.0, 1.0))))).props(
                                "outline dense no-caps")
                        if len(session.designs) == 4:
                            ui.button("Hadamard", on_click=lambda: scheme((
                                ("sum", (1.0, 1.0, 1.0, 1.0)), ("A+B-C-D", (1.0, 1.0, -1.0, -1.0)),
                                ("A-B+C-D", (1.0, -1.0, 1.0, -1.0)),
                                ("A-B-C+D", (1.0, -1.0, -1.0, 1.0))))).props(
                                "outline dense no-caps")
                        ui.button("a combination", icon="add", on_click=lambda: (
                            session.weights.__setitem__(f"combination {len(session.weights) + 1}",
                                                        {name: 1.0 for name in session.designs}),
                            parts())).props("flat dense no-caps")
                for label, mix in (session.weights.items() if len(session.designs) > 1 else ()):
                    with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                        ui.label(label).style(f"width: 160px; color: {SAND}")
                        for name in session.designs:
                            # a receiver phase of a cycle: said, not typed
                            if isinstance(mix.get(name, 0.0), complex):
                                ui.label(f"{name}  {np.degrees(np.angle(mix[name])):+.0f}°"
                                         ).classes("value").style("width: 84px")
                                continue
                            ui.number(name, value=mix.get(name, 0.0), step=1.0, format="%.3g",
                                      on_change=lambda e, mix=mix, name=name: (
                                          e.value is not None
                                          and mix.__setitem__(name, float(e.value)))).props(
                                "dense outlined dark").style("width: 84px")
                        ui.button(icon="close", on_click=lambda _, label=label: (
                            session.weights.pop(label), parts())).props("flat dense round size=sm")
                if len(session.designs) > 1:
                    ui.label("The sub-acquisitions take turns over the transients. The first "
                             "combination is what Live shows; Review and the show select have "
                             "them all, and every sub-acquisition alone.").classes("label").style(
                        "text-transform: none; letter-spacing: 0")

        def drop(onto):
            """A dragged block takes the place of the one it is dropped on."""
            dragged = state.pop("dragged", None)
            if dragged is not None and dragged != onto:
                session.design.insert(onto, session.design.pop(dragged))
                state["at"] = onto
                refresh()

        def flip(name):
            """One of the card's switches pressed in the designer. Real pulses also turns the
            design: the card's own sequence is taken over again as the card now plays it, an own
            design's pulses become shaped, or ideal again."""
            toggle(name)
            if name == "real":
                if session.protocol.sequence != "Custom":
                    session.adopt()
                    state["at"] = min(state["at"], len(session.design) - 1)
                else:
                    for index in range(len(session.design)):
                        reshape(index, session.protocol.real)
            refresh()

        def reshape(index, shaped):
            """An ideal pulse becomes a shaped one of the same flip and phase, or back."""
            kind, settings = session.design[index]
            if kind in ("hard pulse", "shaped pulse") and (kind == "shaped pulse") != shaped:
                other = "shaped pulse" if shaped else "hard pulse"
                session.design[index] = (other, {**BLOCKS[other][0],
                                                 "flip_deg": settings["flip_deg"],
                                                 "phase_deg": settings["phase_deg"]})

        def place(kind, settings):
            """A new block goes in after the chosen one and becomes the chosen one."""
            state["at"] = min(state["at"] + 1, len(session.design))
            session.design.insert(state["at"], (kind, settings))
            refresh()

        def move(step):
            here, there = state["at"], state["at"] + step
            if 0 <= there < len(session.design):
                session.design[here], session.design[there] = (session.design[there],
                                                               session.design[here])
                state["at"] = there
                refresh()

        def remove():
            session.design.pop(state["at"])
            state["at"] = max(min(state["at"], len(session.design) - 1), 0)
            refresh()

        async def envelope(event):
            session.design[state["at"]][1]["envelope"] = ", ".join(
                (await event.file.text()).replace(",", " ").split())
            refresh()

        async def look():
            """What the design does to a spectrum (Session.showcase): the subject on the table at
            the planned voxel, a uniform field, no noise, the card's water suppression and OVS as
            the strip shows them."""
            busy.set_visibility(True)
            try:
                ppm, real = await run.io_bound(session.showcase)
            except Exception as error:
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")
                return
            finally:
                busy.set_visibility(False)
            outcome.options["series"][0]["data"] = [[float(a), float(b)] for a, b in zip(ppm, real)]
            outcome.update()

        def use():
            """The card plays the design, after the water suppression and OVS the strip shows."""
            keys.active, planning.active, session.protocol.sequence = False, True, "Custom"
            dialog.close()
            draw()

        def key(event):
            """Delete removes the chosen block; ctrl or cmd with C copies it, with V pastes after
            it."""
            if not event.action.keydown or not session.design:
                return
            if event.key in ("Delete", "Backspace"):
                remove()
            elif (event.modifiers.ctrl or event.modifiers.meta) and event.key == "c":
                state["copied"] = (session.design[state["at"]][0],
                                   dict(session.design[state["at"]][1]))
            elif ((event.modifiers.ctrl or event.modifiers.meta) and event.key == "v"
                  and "copied" in state):
                place(state["copied"][0], dict(state["copied"][1]))

        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1300px; max-width: 1600px"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label("Sequence designer").style(
                    f"color: {SAND}; font-size: 15px; white-space: nowrap")
                # its pulses as the card has them, real or ideal
                ui.select(list(TEMPLATES), label="start from", on_change=lambda e: e.value and (
                    setattr(session, "design", [(kind, {**BLOCKS[kind][0], **settings})
                                                for kind, settings in TEMPLATES[e.value]]),
                    [reshape(i, True) for i in range(len(session.design)) if session.protocol.real],
                    state.__setitem__("at", 0), refresh())).props("dense outlined dark").style(
                    "width: 240px").tooltip(
                    "a template's pulses are shaped where the card plays real pulses, ideal where "
                    "it plays ideal ones")

                async def seq_file():
                    """A Pulseq file becomes the design, block by block, up to its first readout."""
                    path = await run.io_bound(choose_file, "A Pulseq .seq file")
                    if path is None:
                        return
                    try:
                        await run.io_bound(session.adopt_seq, path)
                    except Exception as error:
                        ui.notify(f"Not read: {error}"[:200], type="warning", multi_line=True)
                        return
                    state["at"] = 0
                    refresh()

                ui.button("or a Pulseq file", icon="folder_open", on_click=seq_file).props(
                    "outline dense no-caps").style("width: 150px").tooltip(
                    "the file's pulses, delays and gradients up to its first readout become "
                    "blocks; the card plays it as Custom")
                # the card's switches, here too: the water suppression steps through its schemes
                for name, caption, icon, tip in (
                        ("real", "real pulses", "waves", "the design's pulses and the card's: "
                                                         "shaped, played in time, or ideal"),
                        ("water", "water", "water_drop", "the card's water suppression scheme, "
                                                         "played before the design"),
                        ("ovs", "OVS", "crop_free", "the card's outer-volume suppression, played "
                                                    "before the design")):
                    switches[name] = ui.button(caption, icon=icon, on_click=lambda _, name=name: (
                        flip(name))).props("flat dense no-caps").classes("tool").style(
                        "flex: none").tooltip(tip)
                problem = ui.label("").classes("value").style("white-space: nowrap")
                ui.space()
                ui.button("Close", on_click=lambda: (
                    setattr(keys, "active", False), setattr(planning, "active", True),
                    dialog.close())).props("flat dense no-caps").tooltip(
                    "the design is kept; the card plays what it played")
                ui.button("Use this design", icon="check", on_click=use).props(
                    "unelevated dense no-caps").style("flex: none").tooltip(
                    "the card plays this design, after its water suppression and OVS")
                keys = ui.keyboard(on_key=key, ignore=["input", "select", "textarea"])
            with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                preview = ui.html("", sanitize=False).style("flex: 3; min-width: 0")
                with ui.element("div").style("flex: 2; min-width: 0; position: relative"):
                    outcome = ui.echart({**copy.deepcopy(SPECTRUM), "title": {
                        **SPECTRUM["title"], "text": "the spectrum this design gives"}}).style(
                        "height: 340px").classes("w-full")
                    busy = ui.spinner(size="56px", color=SAND).style(
                        "position: absolute; top: 42%; left: 46%")
                    busy.set_visibility(False)
                    ui.button("Spectrum of this design", icon="insights", on_click=look).props(
                        "outline dense no-caps").style(
                        "position: absolute; bottom: 72px; right: 12px")
            several = ui.column().classes("w-full").style("gap: 4px")
            ui.label("The design, in the order it plays, after the card's water suppression and "
                     "OVS · press a block to choose it · drag one onto another to move it "
                     "there").classes("label")
            strip = ui.row().classes("w-full").style("gap: 4px")
            with ui.row().classes("w-full no-wrap").style(
                    "gap: 10px; align-items: stretch; flex: 1"):
                with ui.column().classes("pane").style("width: 430px"):
                    ui.label("Add after the chosen block").classes("label")
                    # a fixed pulse is only ever taken over from the card
                    for kind in (k for k in BLOCKS if k != FIXED):
                        with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                            ui.button(kind, icon=BLOCK_ICONS[kind], on_click=lambda _, kind=kind: (
                                place(kind, dict(BLOCKS[kind][0])))).props(
                                "outline dense no-caps align=left").style(
                                "width: 190px; height: 34px; flex: none")
                            ui.label(BLOCK_NOTES[kind].split(".")[0]).classes("label").style(
                                "text-transform: none; letter-spacing: 0")
                inspector = ui.column().classes("pane").style("flex: 1; min-width: 0")
            parts()
            refresh()
        planning.active = False  # the planning view's keys rest while blocks are typed and moved
        dialog.open()

    def review():
        """The Review room of the chosen exam: every condition and combination, what the scanner
        and the subject did over the transients, and the truth no scanner has."""
        exam = chosen["exam"]
        if exam is None or not exam.spectra:
            ui.notify("Select an exam that has run")
            return
        axis = {"type": "value", "axisLine": {"lineStyle": {"color": LINE}},
                "axisLabel": {"color": DIM}, "splitLine": {"show": False}}
        legend = {"textStyle": {"color": INK}, "top": 0}
        palette = [SAND, GLOW, COPPER, GOOD, "#9FB7C9", "#C9A0B8", BAD]
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1100px; max-width: 1400px"):
            with ui.row().classes("w-full items-center"):
                ui.label(f"Review · {exam.name}").classes("label")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense")
            if exam.maps:
                def painted(name):
                    grid = exam.maps[name][:, :, int(layer.value)]
                    return {"backgroundColor": "transparent", "animation": False, "tooltip": {},
                            "grid": {"left": 30, "right": 12, "top": 10, "bottom": 30},
                            "xAxis": {"type": "category", "data": list(range(grid.shape[0])),
                                      "axisLabel": {"color": DIM}},
                            "yAxis": {"type": "category", "data": list(range(grid.shape[1])),
                                      "axisLabel": {"color": DIM}},
                            "visualMap": {"min": 0, "max": float(grid.max()), "show": False,
                                          "inRange": {
                                              "color": ["#000000", COPPER, SAND, "#FFF2D8"]}},
                            "series": [{"type": "heatmap",
                                        "data": [[i, j, float(grid[i, j])]
                                                 for i in range(grid.shape[0])
                                                 for j in range(grid.shape[1])]}]}

                def cell(event):
                    i, j = int(event.data[0]), int(event.data[1])
                    one.options["series"][0]["data"] = [
                        [float(p), float(v)]
                        for p, v in zip(exam.ppm, exam.cells[i, j, int(layer.value)])]
                    one.options["title"]["text"] = f"cell {i}, {j}, slice {int(layer.value)}"
                    one.update()

                with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                    with ui.column().style("width: 420px"):
                        with ui.row().classes("w-full no-wrap"):
                            which = ui.select(list(exam.maps), value="NAA",
                                              label="map · click a cell").props(
                                "dense outlined dark").style("flex: 1")
                            layer = ui.select(list(range(exam.cells.shape[2])),
                                              value=exam.cells.shape[2] // 2, label="slice").props(
                                "dense outlined dark").style("width: 90px")
                        image = ui.echart(painted("NAA"), on_point_click=cell).style(
                            "height: 400px").classes("w-full")
                        for control in (which, layer):
                            control.on_value_change(lambda: (
                                image.options.clear(), image.options.update(painted(which.value)),
                                image.update()))
                    one = ui.echart({"backgroundColor": "transparent", "animation": False,
                                     "title": {"text": "centre cell",
                                               "textStyle": {"color": DIM, "fontSize": 11,
                                                             "fontWeight": "normal"}},
                                     "grid": {"left": 12, "right": 12, "top": 28, "bottom": 28},
                                     "xAxis": {**axis, "inverse": True, "min": 0.2, "max": 4.2,
                                               "name": "ppm"},
                                     "yAxis": {"type": "value", "show": False, "scale": True},
                                     "series": [{"type": "line", "showSymbol": False,
                                                 "lineStyle": {"color": SAND, "width": 1.3},
                                                 "data": [[float(p), float(v)] for p, v in zip(
                                                     exam.ppm, exam.spectra["centre cell"])]}]
                                     }).style("height: 440px; flex: 1")
                dialog.open()
                return
            ui.echart({"backgroundColor": "transparent", "animation": False, "legend": legend,
                       "color": palette,
                       "grid": {"left": 12, "right": 12, "top": 34, "bottom": 28},
                       "tooltip": {"trigger": "axis"},
                       "xAxis": {**axis, "inverse": True, "min": -0.5, "max": 6.5, "name": "ppm"},
                       "dataZoom": [{"type": "inside", "startValue": exam.window[0],
                                     "endValue": exam.window[1], "filterMode": "filter"}],
                       "yAxis": {"type": "value", "show": False, "scale": True},
                       "series": [{"type": "line", "name": name, "showSymbol": False,
                                   "lineStyle": {"width": 1.2},
                                   "data": [[float(p), float(v)] for p, v in zip(exam.ppm, real)]}
                                  for name, real in exam.spectra.items()]}).style(
                "height: 300px").classes("w-full")
            # one transient at a time under the mean, as the display's processing left them (32 at
            # most, evenly through the run)
            singles = next((t for t in exam.transients.values() if t), [])
            if singles:
                def one(k):
                    return [[float(p), float(v)] for p, v in zip(exam.ppm, singles[int(k) - 1])]

                mean = exam.spectra[next(name for name, t in exam.transients.items() if t)]
                with ui.row().classes("w-full items-center no-wrap").style("gap: 12px"):
                    ui.label("transient").classes("label")
                    step = ui.slider(min=1, max=len(singles), value=1, step=1).props(
                        "label color=accent").style("flex: 1")
                through = ui.echart({"backgroundColor": "transparent", "animation": False,
                                     "grid": {"left": 12, "right": 12, "top": 8, "bottom": 28},
                                     "xAxis": {**axis, "inverse": True, "min": exam.window[0],
                                               "max": exam.window[1], "name": "ppm"},
                                     "yAxis": {"type": "value", "show": False, "scale": True},
                                     "series": [{"type": "line", "showSymbol": False,
                                                 "lineStyle": {"color": DIM, "width": 0.8},
                                                 "data": one(1)},
                                                {"type": "line", "showSymbol": False, "z": 3,
                                                 "lineStyle": {"color": SAND, "width": 1.3},
                                                 "data": [[float(p), float(v)] for p, v in zip(
                                                     exam.ppm, mean)]}]}).style(
                    "height: 220px").classes("w-full")
                step.on_value_change(lambda e: (
                    through.options["series"][0].__setitem__("data", one(e.value)),
                    through.update()))
            with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch"):
                with ui.column().style("flex: 2; min-width: 0"):
                    ui.label("Over the transients · frequency, phase and amplitude as they ran"
                             ).classes("label")
                    ui.echart({"backgroundColor": "transparent", "animation": False,
                               "legend": legend, "color": palette,
                               "grid": {"left": 40, "right": 12, "top": 34, "bottom": 28},
                               "tooltip": {"trigger": "axis"},
                               "xAxis": {**axis, "name": "transient"},
                               "yAxis": {**axis, "scale": True},
                               "series": [{"type": "line", "name": name, "showSymbol": False,
                                           "data": [[k + 1, float(v)]
                                                    for k, v in enumerate(values)]}
                                          for name, values in exam.tracks.items()]
                               # the transients that carry the spurious echo, marked on the
                               # frequency track
                               + [{"type": "scatter", "name": "spurious echo", "symbolSize": 8,
                                   "itemStyle": {"color": BAD},
                                   "data": [[k + 1, float(f)] for k, (f, size) in enumerate(zip(
                                       exam.tracks.get("frequency_hz", []), exam.echoes))
                                            if size > 0.0]}]}).style("height: 230px").classes(
                        "w-full")
                with ui.column().style("flex: 1; min-width: 0"):
                    ui.label("Ground truth · mM").classes("label")
                    with ui.scroll_area().style("height: 230px"):
                        for name, value in sorted(exam.truth.items(), key=lambda item: -item[1]):
                            with ui.row().classes("w-full").style("justify-content: space-between"):
                                ui.label(name)
                                ui.label(f"{value:.2f}").classes("value")
        dialog.open()

    def export_seq():
        exam = chosen["exam"]
        if exam is None:
            ui.notify("Select an exam")
            return
        try:
            ui.notify(f"Exported "
                      f"{session.export_seq(exam, pathlib.Path.home() / 'MRSandbox' / 'exports')}")
        except Exception as error:  # an idealization Pulseq cannot play: the design says which
            ui.notify(f"Not exported: {error}", type="warning", multi_line=True)

    def field_changed(event):
        """The field the scanner runs at; back on the preset's own number it is the preset's again.
        A value the picker was set to by a scanner switch changes nothing and does not redraw."""
        if event.value is None:
            return
        before = session.field_t
        session.field_t = (None if abs(float(event.value) - PRESETS[session.scanner].b0_t) < 1e-6
                           else float(event.value))
        if session.field_t != before:
            draw()

    def changed(attribute, target=None):
        """A handler that writes a control's value to the session (or the card) and redraws."""
        def handle(event):
            if event.value is None:
                return
            before = getattr(session.protocol if target == "card" else session, attribute)
            setattr(session.protocol if target == "card" else session, attribute, event.value)
            if attribute == "scanner":                  # a scanner comes with its own field
                session.field_t = None
                pickers["field_t"].set_value(PRESETS[session.scanner].b0_t)
            if attribute == "subject":
                try:
                    session.body()
                except Exception as error:
                    # a subject whose files are placed by hand and are not there
                    ui.notify(f"{event.value} is not on this machine: {error}"[:200],
                              type="warning", multi_line=True)
                    session.subject = before
                    event.sender.value = before
                    return
                session.position_mm = ([0.0, 0.0, 0.0] if hasattr(session.body(), "contents")
                                       else list(PLACES.get(session.subject, PLACE)))
                session.view_mm = list(session.position_mm)
                session.home()
                # the operator's own map stays
                if session.field in FieldMap.PRESETS or session.field == "uniform":
                    session.field = FIELDS.get(session.subject, "uniform")
            if attribute == "subject":
                room()
            elif attribute == "coil":
                room(loops_only=True)
            draw()
        return handle

    async def keep(event):
        """An uploaded file, kept where the session finds it again (~/MRSandbox/maps); returns the
        path."""
        folder = pathlib.Path.home() / "MRSandbox" / "maps"
        folder.mkdir(parents=True, exist_ok=True)
        await event.file.save(folder / event.file.name)
        return folder / event.file.name

    def maps(said="", values=True):
        """The brain built in one place: the anatomy and its maps, line by line (the anatomy, a
        T1-weighted image, its three tissues, and its B0, B1+ and lipid maps; every line says what
        is loaded now and offers what the package has stored, with "your own file ..." on top: the
        system's file dialog, NIfTI or DICOM of any vendor; what is chosen waits, and "Apply" puts
        it all in force at once: a new anatomy with own tissue maps is used as it is, one without
        them is segmented first, and every map is brought onto the anatomy's grid; "Register" on a
        line is for a file of another session), and with `values` the subject's values, each with
        the source it comes from."""
        # the operator's files by line, the lines' drop-downs, what each shows
        own, selects, shown_now = {}, {}, {}

        async def picked(what, value):
            """A stored choice waits as it is; "your own file ..." asks for the file, and a DICOM
            file gives its series."""
            # the line showing the file that was just staged on it: not a new choice
            if value == own.get(what):
                return
            if value != OWN:
                own.pop(what, None)
                shown_now[what] = value
                return
            try:
                chosen_file = await run.io_bound(
                    choose_file, f"The {what} file: NIfTI, or any DICOM file of the series")
                if chosen_file is None:
                    # back to what it showed, so that the line can be chosen again
                    selects[what].set_value(shown_now[what])
                    return
                if not chosen_file.name.endswith((".nii", ".nii.gz")):
                    told.text = "Reading the DICOM"
                    chosen_file = max((await run.io_bound(read_dicom, chosen_file)).values(),
                                      key=lambda path: path.stat().st_size)
                stage(what, chosen_file)
            except Exception as error:
                told.text = f"Not taken: {error}"[:260]

        def stage(what, path):
            own[what] = shown_now[what] = str(path)
            selects[what].options[str(path)] = pathlib.Path(path).name
            selects[what].update()
            selects[what].set_value(str(path))
            told.text = f"{pathlib.Path(path).name} waits as the {what}: Apply puts it in force"

        async def register(what):
            """A file of the operator's own is registered to the anatomy; a stored map of one of the
            package's heads is carried over onto an anatomy of the operator's own, by registering
            that head onto it (Session.borrow)."""
            stored = shown_now[what] if shown_now[what] in STORED else None
            if what not in own and (stored is None or what not in ("B0", "lipid")):
                told.text = ("Choose your own file, or a stored map of another head, on this line "
                             "first")
                return
            if what not in own and session.subject not in session.images:
                told.text = ("This anatomy is the package's own and has its maps where they "
                             "belong. Put an anatomy of your own in force first (Apply), then "
                             "carry the stored map over onto it")
                return
            working.set_visibility(True)
            try:
                if what in own:
                    stage(what, await run.io_bound(session.register, own[what], own.get("anatomy")))
                else:
                    told.text = (f"Registering the {STORED[stored]} onto your anatomy and carrying "
                                 f"its {what} map over: some minutes the first time")
                    stage(what, await run.io_bound(
                        session.borrow, what,
                        "bigbrain_mr" if stored == "bigbrain_mr" else "qsm_rc2"))
                told.text = f"The {what} is registered to the anatomy and waits for Apply"
            except Exception as error:
                told.text = f"Not registered: {error}"[:260]
            finally:
                working.set_visibility(False)

        async def apply():
            """Everything that waits, in the order it depends on: the anatomy first, then the maps
            onto it."""
            working.set_visibility(True)
            try:
                if "anatomy" in own or {"GM", "WM", "CSF"} <= set(own):
                    label = name.value or pathlib.Path(
                        own.get("anatomy", own.get("GM", "anatomy"))).name.split(".")[0]
                    if {"GM", "WM", "CSF"} <= set(own):
                        session.anatomy(label, own["GM"], own["WM"], own["CSF"],
                                        picture=own.get("anatomy"))
                    else:
                        told.text = ("Cropping the neck, extracting the brain and segmenting it "
                                     "with FSL: about two minutes")
                        await run.io_bound(session.segment, own["anatomy"], label)
                    pickers["subject"].set_options(session.anatomies(), value=label)
                elif shown_now["anatomy"] != session.subject:
                    pickers["subject"].set_value(shown_now["anatomy"])
                for what in ("B0", "B1+", "lipid"):
                    value = own.get(what, shown_now[what])
                    if what in own:
                        value = str(session.onto(own[what]))
                    if what == "B0":
                        session.field = value
                    elif what == "B1+":
                        session.transmit = None if value == "nominal" else value
                    elif value == LENT or what in own:
                        session.lipids(value)
            except Exception as error:
                told.text = f"Not applied: {error}"[:260]
                return
            finally:
                working.set_visibility(False)
            dialog.close()
            room()
            draw()
            # the pop-up stays, showing what is loaded now, for the next change
            maps(
                f"In force: {session.subject}, B0 "
                f"{STORED.get(session.field, pathlib.Path(session.field).name)}, B1+ "
                f"{'nominal' if session.transmit is None else pathlib.Path(session.transmit).name}",
                values)

        fat = getattr(session.body(), "fat", None)
        entries = (("anatomy", "anatomy (T1-weighted)", session.anatomies(), session.subject),
                 ("GM", "grey matter", [f"of {session.subject}"], f"of {session.subject}"),
                 ("WM", "white matter", [f"of {session.subject}"], f"of {session.subject}"),
                 ("CSF", "CSF", [f"of {session.subject}"], f"of {session.subject}"),
                 ("B0", "B0 field (Hz)", ["uniform", *FieldMap.PRESETS], session.field),
                 ("B1+", "B1+ (of nominal)", ["nominal"], session.transmit or "nominal"),
                 ("lipid", "lipid layer (0 to 1)", ["its own" if fat is not None else "none", LENT],
                  "its own" if fat is not None else "none"))
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 900px; max-width: 1060px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Anatomy" if values else "Anatomy and maps").classes("label")
                ui.space()
                working = ui.spinner(size="26px", color=SAND)
                working.set_visibility(False)
                if values:
                    ui.input(placeholder="search a value, e.g. NAA or t2",
                             on_change=lambda e: build(e.value or "")).props(
                        "dense outlined dark clearable debounce=300").style("width: 240px")
            if values:
                ui.label("Anatomy and maps").classes("label").style(
                    f"border-top: 1px solid {LINE}; padding-top: 8px; width: 100%")
            told = ui.label(said).classes("value").style(
                f"white-space: normal; color: {GOOD if said else INK}")
            for what, caption, stored, now in entries:
                shown_now[what] = now
                with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                    ui.label(caption).style("width: 170px")
                    selects[what] = ui.select(
                        {OWN: OWN, **{option: STORED.get(option, option) for option in stored},
                         **({now: pathlib.Path(now).name} if now not in stored else {})},
                        value=now, on_change=lambda e, what=what: picked(what, e.value)).mark(
                        f"line {what}").props("dense outlined dark options-dense").style("flex: 1")
                    if what != "anatomy":
                        ui.button("Register", icon="my_location",
                                  on_click=lambda _, what=what: register(what)).props(
                            "outline dense no-caps").style("width: 130px").tooltip(
                            "register a file of another session onto the anatomy (FSL flirt), or "
                            "carry a stored head's map over onto your own anatomy")
                    else:  # the same as Apply does for a new anatomy without tissue maps, at once
                        ui.button("Segment", icon="psychology", on_click=apply).props(
                            "outline dense no-caps").style("width: 130px").tooltip(
                            "segment your image into tissues (FSL, about two minutes) and "
                            "apply the lines")
            ui.switch("the B0 map was measured after my own shim: apply none",
                      value=session.shim == "none", on_change=lambda e: (
                          setattr(session, "shim", "none" if e.value else "linear"))).props(
                "dense color=accent")
            ui.label("A new anatomy with your own three tissue maps is used as it is. Without them "
                     "it is segmented on Apply (FSL: the neck cropped, the brain extracted, grey "
                     "matter, white matter and CSF; the bright tissue beyond the bone becomes the "
                     "lipid layer): about two minutes. T1, T2, concentrations and every other "
                     "parameter take a map at their line in the settings (the gears); coil "
                     "sensitivities belong to the coils.").classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                name = ui.input("name of a new anatomy (the file's, if empty)").props(
                    "dense outlined dark").style("flex: 1")
                ui.button("Close", on_click=lambda: (dialog.close(), draw())).props(
                    "flat dense no-caps")
                ui.button("Apply", icon="check", on_click=apply).mark("apply the maps").props(
                    "unelevated dense no-caps").style("width: 120px")
            if values:
                ui.label(SAID).classes("label").style(
                    f"margin-top: 10px; border-top: 1px solid {LINE}; padding-top: 8px; "
                    f"text-transform: none; letter-spacing: 0; white-space: normal")
                # a body breathes and moves; a phantom does not
                body_moves = not hasattr(session.body(), "contents")
                priors = {path: (kind, numbers)
                          for path, kind, numbers, _ in session.leaves(own=False)
                          if (path.split(".")[0] in GEARS["subject"] and path not in RESIDUAL)
                          or (body_moves and path in {f"run.{key}" for key in RUN_OF["subject"]})}
                with ui.scroll_area().style(
                        "height: calc(100vh - 640px); min-height: 220px; width: 100%"):
                    body = ui.column().classes("w-full").style("gap: 0")
                build = lines(body, priors, session.sources())
                build()
        dialog.open()

    def switches():
        """One list of switches: what is real about the scan and what happens over a run, each on or
        off; the numbers behind them live in the cards (all settings, the scanner, the anatomy)."""
        def pulses(event):
            session.protocol.real = event.value
            draw()

        def field(event):
            session.field = FIELDS.get(session.subject, "uniform") if event.value else "uniform"
            draw()

        def running(key, off):
            def handle(event):
                session.overrides.pop(f"run.{key}", None)
                if not event.value:
                    session.overrides[f"run.{key}"] = off
                draw()
            return handle

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 700px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Switches · on is real, off is ideal or still").classes("label")
                ui.space()
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
            ui.label("the scan").classes("label").style(
                f"border-top: 1px solid {LINE}; padding-top: 6px; width: 100%")
            for caption, note, value, handler in (
                    ("real pulses",
                     "shaped pulses under gradients, played in time: a voxel profile, displaced "
                     "shifts, a slower simulation",
                     session.protocol.real, pulses),
                    ("B0 field",
                     "the measured field of the head, shimmed as the Geometry tab says (the B0 "
                     "overlay shows it after the shim)",
                     session.field != "uniform", field)):
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.switch(caption, value=value, on_change=handler).props(
                        "dense color=accent").style("width: 230px")
                    ui.label(note).classes("label").style("text-transform: none; letter-spacing: 0")
            for name, leaves in IDEALS.items():
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.switch(name, value=not all(session.overrides.get(path) == value
                                                  for path, value in leaves.items()),
                              on_change=lambda e, name=name: (
                                  session.idealize(name, not e.value), draw())).props(
                        "dense color=accent").style("width: 230px")
                    ui.label("off sets " + ", ".join(leaves) + " to 0" + (
                        "; on, the measured 3 T shape, coupled systems on edited cards"
                        if name == "macromolecules" else "")).classes(
                        "label").style("text-transform: none; letter-spacing: 0")
            ui.label("the run").classes("label").style(
                f"border-top: 1px solid {LINE}; padding-top: 6px; margin-top: 6px; width: 100%")
            ui.label(f"This session the scanner drifts at {session.drift_hz_per_min:+.2f} Hz per "
                     f"minute; {session.clock_s / 60.0:.1f} minutes "
                     "have been scanned. On: drawn from the 3 T envelope, or the number set in the "
                     "cards; off: still.").classes("label").style(
                "text-transform: none; letter-spacing: 0")
            for caption, (key, off) in RUN.items():
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.switch(caption, value=session.overrides.get(f"run.{key}") != off,
                              on_change=running(key, off)).props("dense color=accent").style(
                        "width: 230px")
                    ui.label("the scanner's" if key in RUN_OF["scanner"]
                             else "the subject's; a phantom has none").classes("label").style(
                        "text-transform: none; letter-spacing: 0")
            ui.label("Every number stays open in the cards: the scanner's over a run on the "
                     "scanner card, the subject's on the anatomy card.").classes(
                "label").style("text-transform: none; letter-spacing: 0")
        dialog.open()

    def scanner_builder():
        """The scanner as one list of lines: what stands on the floor (the system, a dataset, your
        calibration file), the hardware's numbers, the measured behaviour, each line with its value,
        its source and one select; then the scans of yours that read the lines, and the scanner
        saved."""
        held, status = session.reports, {}
        sources = session.sources()
        priors = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves(own=False)
                  if path.split(".")[0] in GEARS["scanner"] or path in RESIDUAL
                  or path in {f"run.{key}" for key in RUN_OF["scanner"]}}
        lined = [path for path in ("noise.sigma", "lineshape.gauss_hz", "acquisition.delay_s",
                                   "acquisition.eddy_transient_rad", "water.adjusted_residual",
                                   "oov.fraction", "spurious.amplitude", "b1_scale",
                                   *(f"run.{key}" for key in RUN_OF["scanner"])) if path in priors]

        def reopen():
            dialog.close()
            draw()
            scanner_builder()

        def system_picked(event):
            session.scanner = event.value
            session.field_t, session.calibration = None, None
            pickers["scanner"].set_options(session.scanners(), value=session.scanner)
            pickers["field_t"].set_value(PRESETS[session.scanner].b0_t)
            reopen()

        def dataset_picked(event):
            session.pick_cohort(event.value or None)
            pickers["scanner"].set_options(session.scanners(), value=session.scanner)
            pickers["field_t"].set_value(session.profile().scanner.b0_t)
            session.home()
            room()
            reopen()

        def for_all(event):
            if event.value in ("literature", "pooled", "mixed", "dataset", "own"):
                session.choose_all(lined, event.value)
                reopen()

        def departures():
            told.clear()
            with told:
                said = [f"{NAMES.get(leaf, leaf)}: " + "; ".join(
                            f"{what} {' → '.join(str(v) for v in pair)}"
                            for what, pair in record["extrapolated"].items())
                        for leaf, record in session.outlook()["records"].items()
                        if record["extrapolated"]]
                ui.label(("your setup differs from the measurements in: " + " | ".join(said))
                         if said else "your setup matches the measurements").style(
                    f"color: {BAD if said else GOOD}; font-size: 12px; white-space: normal")

        async def reference(event):
            held["reference"] = await keep(event)
            status["reference"].text = f"kept: {pathlib.Path(held['reference']).name}; now the run"

        async def read(event):
            if not named.value:
                status["scan"].text = "give the calibration a name first"
                return
            busy.set_visibility(True)
            try:
                path = await run.io_bound(session.calibrate, await keep(event), named.value, None,
                                          held.get("reference"))
            except Exception as error:
                status["scan"].text = f"not read: {error}"[:200]
                return
            finally:
                busy.set_visibility(False)
            ui.notify(f"This scanner is now calibrated by {path.name}")
            reopen()

        async def load_file():
            path = await run.io_bound(choose_file, "A scanner file, JSON as the package saves it")
            if path is None:
                return
            try:
                session.load_scanner(path)
            except Exception as error:
                ui.notify(f"Not read: {error}"[:200], type="warning", multi_line=True)
                return
            pickers["scanner"].set_options(session.scanners(), value=session.scanner)
            reopen()

        async def load_folder():
            path = await run.io_bound(
                choose_folder, "A folder of scans, any vendor's files: spectra with their water, "
                               "images, maps, noise scans")
            if path is None:
                return
            status["folder"].text = ("reading the folder: raw spectra converted, the scanner read "
                                     "from the first run, an anatomy segmented (some minutes)")
            try:
                held["report"] = await run.io_bound(session.replicate, path)
            except Exception as error:
                status["folder"].text = f"not read: {error}"[:200]
                return
            pickers["scanner"].set_options(session.scanners(), value=session.scanner)
            pickers["subject"].set_options(session.anatomies(), value=session.subject)
            pickers["coil"].set_options(session.arrays(), value=session.coil)
            room()
            reopen()

        def save(event=None):
            if not saved_as.value:
                status["save"].text = "give the file a name first"
                return
            status["save"].text = f"saved as {session.save_scanner(saved_as.value)}"

        limits = session.profile().scanner
        generic = "generic_3t"
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1000px; max-width: 1140px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Scanner").classes("label")
                ui.space()
                ui.select({name: SOURCE_WORDS[name]
                           for name in ("literature", "pooled", "mixed", "dataset", "own")
                           if any(name in sources[path]["sources"]
                                  for path in lined if path in sources)},
                          label="for all lines", on_change=for_all).props(
                    "dense outlined dark options-dense").style("width: 150px")
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props(
                    "unelevated dense no-caps")
            ui.label(SAID).classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.scroll_area().style("height: 660px; width: 100%"):
                with ui.column().classes("w-full").style("gap: 0"):
                    # what stands on the floor
                    with row("system"):
                        ui.select({generic: f"generic, at the field set ({limits.b0_t:g} T)",
                                   **session.systems()},
                                  value=session.scanner if session.scanner in session.systems()
                                  else generic, on_change=system_picked).mark("system").props(
                            "dense outlined dark options-dense").style("width: 260px")
                        chip("literature", limits.cited)
                        ui.label(limits.cited).classes("label").style(
                            "text-transform: none; letter-spacing: 0; white-space: normal; "
                            "flex: 1")
                    with row("dataset", "one real subject of a package dataset on the floor: its "
                                        "readings become a source on every line, its scanner and "
                                        "voxel come with it"):
                        ui.select({"": "none", **session.datasets()}, value=session.cohort or "",
                                  on_change=dataset_picked).mark("dataset").props(
                            "dense outlined dark options-dense").style("width: 480px")
                        if session.cohort:
                            chip("dataset", session.cohort_says())
                            ui.label(session.cohort_says()).classes("label").style(
                                "text-transform: none; letter-spacing: 0; white-space: normal; "
                                "flex: 1")
                    with row("your scanner file", "a scanner saved here, or read from your scans "
                                                  "below, or a file from anywhere: its readings "
                                                  "become a source on every line"):
                        files = {key: what for key, what in session.calibrations().items()
                                 if key.endswith(".json")}
                        ui.select({"": "none", **files},
                                  value=session.calibration if session.calibration in files else "",
                                  on_change=lambda e: (
                                      setattr(session, "calibration", e.value or None),
                                      reopen())).props(
                            "dense outlined dark options-dense").style("width: 380px")
                        ui.button("choose a file", icon="folder_open",
                                  on_click=lambda: load_file()).props(
                            "outline dense no-caps").style("width: 150px")
                        if session.calibration in files:
                            chip("own", session.calibration)
                    with row("a folder of scans",
                             "any vendor's files, read at once: raw spectra (Siemens twix and rda, "
                             "GE P-files, Philips SDAT and data/list, Bruker) through spec2nii, "
                             "NIfTI-MRS, T1-weighted images, field maps, B1 maps, coil maps, "
                             "ISMRMRD noise scans, Pulseq files, saved scanners; each used for "
                             "what it gives"):
                        ui.button("choose a folder", icon="folder_open",
                                  on_click=lambda: load_folder()).props(
                            "outline dense no-caps").style("width: 150px")
                        status["folder"] = ui.label("").classes("label").style(
                            "text-transform: none; letter-spacing: 0; white-space: normal; "
                            "flex: 1")
                    for file, kind, gave in held.get("report", ()):
                        with row("    " + file):
                            ui.label(kind).classes("label").style("width: 120px")
                            ui.label(gave).classes("value").style(
                                "white-space: normal; font-size: 12px; flex: 1")
                    if held.get("report"):
                        # the report as a table: every parameter the folder set, its value and the
                        # file it came from, and the rest counted by the source each keeps
                        table = session.where_from()
                        for path, kind, numbers, source, said in table:
                            if source == "own":
                                with row("    " + NAMES.get(path, path)):
                                    # a leaf per tissue (a T1 map's) says the tissues' readings, as
                                    # the file's line does
                                    ui.label(SHOWN[kind].format(*numbers) if kind != "other"
                                             else ", ".join(
                                        f"{r.leaf.split('.')[-1]} {r.value:.3g}"
                                        for r in session.run_records
                                        if r.leaf.startswith(path + "."))
                                        or str(numbers)[:40]).classes("value").style("width: 260px")
                                    chip("own", said)
                        kept = collections.Counter(SOURCE_WORDS[source or plain_word(path)]
                                                   for path, _, _, source, _ in table
                                                   if source != "own")
                        with row("    what the folder did not give",
                                 "every other line keeps its source; the lines of the cards "
                                 "say which"):
                            ui.label(", ".join(f"{count} {word}"
                                               for word, count in kept.most_common())).classes(
                                "value").style("flex: 1")
                    # the hardware, the manufacturer's numbers
                    for name, field, unit in (("field", "b0_t", "T"),
                                              ("peak B1", "peak_b1_ut", "uT"),
                                              ("gradient amplitude", "max_grad_mt_m", "mT/m"),
                                              ("slew rate", "max_slew_t_m_s", "T/m/s")):
                        with row(name, "the manufacturer's number; set your own here (the top "
                                       "bar's field is the same setting)"):
                            ui.number(value=getattr(limits, field), format="%.6g", suffix=unit,
                                      on_change=lambda e, field=field: e.value is not None and (
                                          setattr(session, "field_t", float(e.value))
                                          if field == "b0_t"
                                          else session.hardware.__setitem__(field, float(e.value)),
                                          pickers["field_t"].set_value(float(e.value))
                                          if field == "b0_t" else None, draw(), departures())
                                      ).props("dense outlined dark").style("width: 150px")
                            own_number = ((session.field_t is not None) if field == "b0_t"
                                          else field in session.hardware)
                            chip("custom" if own_number else "literature",
                                 "set by you" if own_number else limits.cited)
                            if own_number:
                                ui.button("the manufacturer's", icon="restart_alt",
                                          on_click=lambda field=field: (
                                              setattr(session, "field_t", None) if field == "b0_t"
                                              else session.hardware.pop(field, None),
                                              reopen())).props("flat dense no-caps")
                    receiver = session.profile().receiver
                    with row("receiver"):
                        ui.label("acquires at the card's bandwidth" + (
                            f", oversampled {receiver.oversampling} times, a filter of "
                            f"{receiver.span_dwells} dwells"
                            if receiver is not None
                            else ", flat: the raw data of this system show no filter")).classes(
                            "value").style("width: 520px")
                        chip("literature", "hardware.receiver, read off the vendor's raw noise")
                    # the measured behaviour
                    body = ui.column().classes("w-full").style("gap: 0")
                    build = lines(body, {path: priors[path] for path in lined}, sources,
                                  after=departures, groups=False)
                    build()
                    told = ui.column().classes("w-full").style("gap: 0; padding: 6px 0")
                    departures()
                    # your scans
                    with row("your scans", "raw NIfTI-MRS with the transients apart (spec2nii "
                                           "writes it); set the card, the voxel and the room as "
                                           "the scan was acquired first: they are its twin"):
                        named = ui.input("name of the calibration").props(
                            "dense outlined dark").style("width: 260px")
                        busy = ui.spinner(size="24px", color=SAND)
                        busy.set_visibility(False)
                    for key, what, gives in (
                            ("scan", "the water-suppressed run",
                             "gives the noise level, the line width; with the coils apart and an "
                             "array in the room, the channels' own noise share"),
                            ("reference",
                             "its water reference (optional): the same voxel unsuppressed, given "
                             "before the run",
                             "gives, with the run, the residual water, the receiver delay, the "
                             "eddy phase transient, the leaked water")):
                        with row("    " + what):
                            ui.upload(label=".nii.gz", auto_upload=True,
                                      on_upload=reference if key == "reference" else read).props(
                                "flat dense dark accept=.nii,.gz").style("width: 150px")
                            ui.label(gives).classes("label").style(
                                "text-transform: none; letter-spacing: 0; white-space: normal; "
                                "width: 420px")
                            status[key] = ui.label("").classes("value").style(
                                "white-space: normal; font-size: 12px")
                    with row("save the scanner as a file",
                             "~/MRSandbox/profiles; it then appears under "
                             "\"your calibration file\""):
                        saved_as = ui.input("name of the file").props("dense outlined dark").style(
                            "width: 260px")
                        ui.button("Save", icon="save", on_click=save).props(
                            "outline dense no-caps").style("width: 120px")
                        status["save"] = ui.label("").classes("label").style(
                            "text-transform: none; letter-spacing: 0; white-space: normal")
        dialog.open()

    def coils_builder():
        """The receive array as one list of lines: the array, its loops, the sensitivity maps, the
        channels' own noise share with its source, the channel covariance; each line says what feeds
        it."""
        settings = session.coil_settings[session.coil]

        def reopen():
            dialog.close()
            room()
            draw()
            coils_builder()

        async def own(kind):
            picked_file = await run.io_bound(
                choose_file, "Sensitivity maps, one NIfTI of all channels" if kind == "maps" else
                "A raw scan with the coils apart, NIfTI-MRS")
            if picked_file is None:
                said.text = "no file chosen"
                return
            try:
                if kind == "maps":
                    session.coil_maps(str(picked_file), picked_file.name.split(".")[0])
                    pickers["coil"].set_options(session.arrays(), value=session.coil)
                else:
                    await run.io_bound(session.coil_noise, str(picked_file))
            except Exception as error:
                said.text = f"not taken: {error}"[:260]
                return
            reopen()

        share, source, provenance = session.own_share()
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1000px; max-width: 1140px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Receive array").classes("label")
                ui.space()
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props(
                    "unelevated dense no-caps")
            ui.label(SAID).classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.column().classes("w-full").style("gap: 0"):
                with row("array", "the loops the room shows on the head; one ideal channel has "
                                  "uniform sensitivity and the scanner's noise"):
                    ui.select(session.arrays(), value=session.coil, on_change=lambda e: (
                        setattr(session, "coil", e.value), pickers["coil"].set_value(e.value),
                        reopen())).props("dense outlined dark options-dense").style("width: 260px")
                    published = (ARRAY_PRESETS[session.ARRAYS[session.coil]]
                                 if session.coil in session.ARRAYS else None)
                    chip("own" if session.coil not in session.COILS else "literature",
                         published.cited if published
                         else "a stand-in geometry of the package" if session.coil in session.COILS
                         else "your sensitivity maps")
                # a scanner's own array: what its sources say, what the package assumed
                if published:
                    for what, said_, word in (("published", published.cited, "literature"),
                                              ("stated", published.stated, "stated")):
                        with row("    " + what):
                            ui.label(said_).classes("label").style(
                                "text-transform: none; letter-spacing: 0; white-space: normal; "
                                "flex: 1")
                            chip(word, said_)
                for key, value in settings.items():
                    with row("    " + key.replace("_mm", " (mm)").replace("_", " ")):
                        ui.number(value=value, format="%.6g", on_change=lambda e, key=key: (
                            e.value is not None
                            and (settings.__setitem__(key, float(e.value)), room(), draw()))).props(
                            "dense outlined dark").style("width: 110px")
                        chip("custom"
                             if value != session.COIL_SETTINGS.get(session.coil, {}).get(key)
                             else "literature", "the package's stand-in geometry")
                if settings:
                    with row(""):
                        ui.button("back to the package's geometry", icon="restart_alt",
                                  on_click=lambda: (
                                      settings.clear(),
                                      settings.update(session.COIL_SETTINGS.get(session.coil, {})),
                                      reopen())).props("flat dense no-caps")
                with row("sensitivity maps",
                         "one NIfTI of (nx, ny, nz, channels), complex or real, in the scanner's "
                         "coordinates: the array is then yours"):
                    ui.button("choose a file", icon="folder_open",
                              on_click=lambda: own("maps")).props(
                        "outline dense no-caps").style("width: 150px")
                    chip("own" if session.coil not in session.COILS else "literature", "")
                with row("share of a channel's noise that is its own"):
                    ui.label(f"{share:.2f}").classes("value").style("width: 260px")
                    chip(source, provenance)
                    if source == "own":
                        ui.button("the package's", icon="restart_alt", on_click=lambda: (
                            setattr(session, "own_noise", None), reopen())).props(
                            "flat dense no-caps")
                    ui.label(provenance).classes("label").style(
                        "text-transform: none; letter-spacing: 0; white-space: normal; flex: 1")
                with row("a scan with the coils apart",
                         "raw NIfTI-MRS with DIM_COIL: the channels' own share of the noise is "
                         "read from the FIDs' tails"):
                    ui.button("choose a file", icon="folder_open",
                              on_click=lambda: own("noise")).props(
                        "outline dense no-caps").style("width: 150px")
                    ui.label("gives the share above").classes("label").style(
                        "text-transform: none; letter-spacing: 0")
                said = ui.label("").classes("value").style("white-space: normal")
        dialog.open()

    def browse():
        """The protocol browser: what can go on the card, by family, and the operator's saved
        protocols."""
        def pick(name):
            was, session.protocol.sequence = session.protocol.sequence, name
            # a protocol's own number of transients: an edited scan subtracts, and needs some 320
            # (Big GABA: 320 at TR 2 s, eleven minutes); a spectroscopic image takes one per point
            # of k-space
            session.protocol.averages = 320 if name in EDITING else 1 if name == MRSI else 64
            # a spectroscopic image is read shorter and narrower than a single voxel, and has no
            # reference exam
            if name == MRSI:
                (session.protocol.npoints, session.protocol.bandwidth_hz,
                 session.protocol.water_reference) = 1024, 2000.0, 0
            elif was == MRSI:  # and a single voxel after it has its readout and its reference back
                (session.protocol.npoints, session.protocol.bandwidth_hz,
                 session.protocol.water_reference) = 2048, 4000.0, 1
            chosen["snr"] = None
            dialog.close()
            draw()
            if name == "Custom":
                designer()

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 860px"):
            ui.label("Protocols").classes("label")
            with ui.row().classes("w-full no-wrap").style("gap: 14px; align-items: flex-start"):
                for family, names in (("Single voxel",
                                       [n for n in SEQUENCES if n not in REAL.values()]),
                                      ("Edited", [n for n in EDITING if n not in REAL.values()]),
                                      ("Spectroscopic imaging", [MRSI]),
                                      ("Own design", ["Custom"])):
                    with ui.column().style("gap: 4px; flex: 1"):
                        ui.label(family).classes("label")
                        for name in names:
                            ui.button(name, on_click=lambda _, name=name: pick(name)).mark(
                                f"protocol {name}").props(
                                "outline dense no-caps align=left").classes("w-full")
                with ui.column().style("gap: 4px; flex: 1"):
                    ui.label("Saved").classes("label")
                    for name in session.protocols():
                        ui.button(name, icon="bookmark", on_click=lambda _, name=name: (
                            session.load_protocol(name), dialog.close(), draw())).props(
                            "outline dense no-caps align=left").classes("w-full")
        dialog.open()

    def drag(plane):
        """A plane's mouse: a press or a drag moves the voxel, the release settles the card."""
        def handle(event):
            kind, pressed = event.args.get("mouse_event_type"), event.args.get("buttons") == 1
            grid, exam = chosen["grid"], chosen["exam"]
            if (kind == "mousedown" and plane == "tra" and grid is not None
                    and not event.args.get("shiftKey")):
                i = int((event.args.get("image_x", 0.0) - grid[0])
                        // (grid[2] / exam.cells.shape[0]))
                j = exam.cells.shape[1] - 1 - int((event.args.get("image_y", 0.0) - grid[1])
                                                  // (grid[3] / exam.cells.shape[1]))
                # inside the grid: the cell, and the voxel stays
                if 0 <= i < exam.cells.shape[0] and 0 <= j < exam.cells.shape[1]:
                    chosen["cell"] = (i, j, grid[4])
                    shows.value = "Live"
                    frame(exam, -1)
                    paint()
                    return
            x, y = event.args.get("image_x", 0.0), event.args.get("image_y", 0.0)
            here = np.asarray(session.at(plane, x, y)) - np.asarray(session.position_mm)
            if kind == "mousedown" and not event.args.get("shiftKey"):
                # a press inside a saturation slab's band (a convex polygon: the point on one
                # side of every edge) takes the slab along, if its axis lies in this plane
                chosen["slab"] = next((
                    (which[4], 1.0 if which[5] == "+" else -1.0)
                    for which, points in chosen["bands"].get(plane, {}).items()
                    if "xyz".index(which[4]) != EDGES[plane][4]
                    and len({np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))
                             for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])}
                            - {0.0}) == 1), None)
                if chosen["slab"] is not None:
                    # near the voxel the press moves the slab; toward an end of the band it turns
                    axis, side = chosen["slab"]
                    k = "xyz".index(axis)
                    j = 3 - k - EDGES[plane][4]
                    normal = next(np.asarray(tilted[0]) if tilted else np.eye(3)[k]
                                  for a, c, _, *tilted in session.slabs()
                                  if a == axis and c * side > 0)
                    along = abs(here[j] * normal[k] - here[k] * normal[j]) / np.hypot(
                        normal[k], normal[j])
                    chosen["slab"] = (axis, side, along > 1.5 * max(session.size_mm))
            if chosen["slab"] is not None and (kind == "mousedown"
                                              or (kind == "mousemove" and pressed)):
                axis, side, turning = chosen["slab"]
                k = "xyz".index(axis)
                slab = next(s for s in session.protocol.ovs_slabs
                            if (s["axis"], s["side"]) == (axis, side))
                thick = session.protocol.ovs_mm if slab["mm"] is None else slab["mm"]
                centre, normal = next((c, np.asarray(tilted[0]) if tilted else np.eye(3)[k])
                                      for a, c, _, *tilted in session.slabs()
                                      if a == axis and c * side > 0)
                if turning:
                    # the slab keeps its place along its normal and its tilt out of this plane, and
                    # turns in it until its middle runs through the mouse: the tilt toward this
                    # plane's other axis that solves normal . point = centre, nearest the last one
                    j = 3 - k - EDGES[plane][4]
                    others = [m for m in range(3) if m != k]
                    tilt = list(slab.get("tilt_deg", (0.0, 0.0)))
                    out = np.tan(np.radians(tilt[others.index(EDGES[plane][4])]))
                    angles = np.radians(np.linspace(-60.0, 60.0, 241))
                    miss = ((here[k] + np.tan(angles) * here[j] + out * here[EDGES[plane][4]])
                            / np.sqrt(1.0 + np.tan(angles) ** 2 + out ** 2) - centre)
                    roots = angles[:-1][np.sign(miss[:-1]) != np.sign(miss[1:])]
                    if roots.size:
                        last = np.radians(tilt[others.index(j)])
                        tilt[others.index(j)] = round(float(np.degrees(
                            roots[np.argmin(np.abs(roots - last))])), 1)
                        session.set_slab(axis, side, tilt_deg=tuple(tilt))
                else:
                    # the band's middle follows the mouse along the slab's normal; the gap is what
                    # is left between the voxel's face and the slab's inner face
                    gap = side * float(normal @ here) - 0.5 * session.size_mm[k] - 0.5 * thick
                    session.set_slab(axis, side, gap_mm=float(round(max(gap, 0.0))))
                chosen["dragging"] = True
                paint()
                return
            if chosen["slab"] is not None and chosen["dragging"]:
                chosen["slab"], chosen["dragging"] = None, False
                room()
                draw()
                return
            if kind == "mousedown" or (kind == "mousemove" and pressed):
                # with shift the press looks around (the other planes come to this point), without
                # it it places the voxel
                # the side panel then tells of the place looked at
                chosen["looking"] = bool(event.args.get("shiftKey"))
                (session.look if chosen["looking"] else session.place)(
                    plane, event.args.get("image_x", 0.0), event.args.get("image_y", 0.0))
                chosen["dragging"] = True
                paint()
            elif chosen["dragging"]:
                # the release, or the first move after it: the throttle may drop the release itself
                chosen["dragging"] = False
                draw()
        return handle

    def zoom(plane, factor):
        """Zoom one plane about the point the planes cross at, from half the picture's size (room
        about the head) to eight times; None is back to the whole picture."""
        chosen["zoom"][plane] = (1.0 if factor is None
                                 else float(np.clip(chosen["zoom"][plane] * factor, 0.5, 8.0)))
        paint()

    def key(event):
        """The planning view's keys (KEYS)."""
        if not event.action.keydown or event.modifiers.ctrl or event.modifiers.meta:
            return
        if event.key in ("+", "=", "-", "0"):         # the plane the mouse is over
            zoom(chosen["over"], {"+": 1.5, "=": 1.5, "-": 1.0 / 1.5, "0": None}[event.key.name])
        elif event.key == "v":
            session.view_mm = list(session.position_mm)
            paint()
        elif event.key in ("s", "f"):
            which = "shifts" if event.key == "s" else "flipped"
            chosen[which] = not chosen[which]
            draw()
        elif event.key == "o":
            session.overlay = chosen["overlays"][
                (chosen["overlays"].index(session.overlay) + 1) % len(chosen["overlays"])]
            draw()

    def toggle(name):
        """The toolbar's switches: the water suppression steps through its schemes, OVS is on or
        off."""
        card = session.protocol
        if name == "ovs":
            card.ovs = not card.ovs
            room()
        elif name == "real":
            card.real = not card.real
        else:
            schemes = [*SUPPRESSION, "off", ASSUMED]
            card.water_suppression = schemes[
                (schemes.index(card.water_suppression) + 1) % len(schemes)]
        draw()

    def slabs():
        """The card's six saturation slabs one by one: on or off, a gap and thickness of its own
        in place of the card's, and a tilt toward the other two axes; a band on the planes drags
        its slab too, near its middle along, near its ends round."""
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 720px"):
            ui.label("OVS slabs").classes("label")
            ui.label("A pair along every axis, placed for water about the voxel's faces. A gap "
                     "or thickness left empty is the card's. A tilt turns a slab toward the "
                     "other two axes; the whole voxel is then simulated at once, exact and "
                     "slow. Drag a slab's band on a plane to move it, near its ends to turn "
                     "it.").style(f"color: {DIM}")

            def set_slab(axis, side, **settings):
                session.set_slab(axis, side, **settings)
                room()
                draw()

            with ui.grid(columns=5).classes("w-full items-center").style("gap: 6px 12px"):
                for caption in ("slab", "on", "gap to the voxel (mm)", "thickness (mm)",
                                "tilt (°)"):
                    ui.label(caption).classes("label")
                for slab in session.protocol.ovs_slabs:
                    axis, side = slab["axis"], slab["side"]
                    sign = "+" if side > 0 else "−"
                    ui.label(f"{axis} {sign}").classes("value")
                    ui.switch(value=slab["on"], on_change=lambda e, a=axis, s=side: set_slab(
                        a, s, on=bool(e.value))).mark(f"slab {axis}{'+' if side > 0 else '-'}")
                    ui.number(value=slab["gap_mm"], min=0.0, step=1.0, format="%.6g",
                              placeholder=f"{session.protocol.ovs_gap_mm:g}",
                              on_change=lambda e, a=axis, s=side: set_slab(
                                  a, s, gap_mm=None if e.value is None else float(e.value))
                              ).props("dense outlined dark clearable").style("width: 150px")
                    ui.number(value=slab["mm"], min=5.0, step=5.0, format="%.6g",
                              placeholder=f"{session.protocol.ovs_mm:g}",
                              on_change=lambda e, a=axis, s=side: set_slab(
                                  a, s, mm=None if e.value is None else float(e.value))
                              ).props("dense outlined dark clearable").style("width: 150px")
                    with ui.row().classes("no-wrap").style("gap: 4px"):
                        for place, other in enumerate(o for o in "xyz" if o != axis):
                            # the other tilt as the slab holds it when this one changes
                            ui.number(f"to {other}", value=slab.get("tilt_deg", (0.0, 0.0))[place],
                                      min=-60.0, max=60.0, step=1.0, format="%.6g",
                                      on_change=lambda e, a=axis, s=side, p=place: set_slab(
                                          a, s, tilt_deg=tuple(
                                              float(e.value or 0.0) if q == p else t
                                              for q, t in enumerate(next(
                                                  x for x in session.protocol.ovs_slabs
                                                  if (x["axis"], x["side"]) == (a, s)).get(
                                                  "tilt_deg", (0.0, 0.0)))))).props(
                                "dense outlined dark").style("width: 72px").mark(
                                f"tilt {axis}{'+' if side > 0 else '-'} {other}")
            with ui.row().classes("w-full"):
                ui.button("all alike", on_click=lambda: (
                    [set_slab(s["axis"], s["side"], on=True, gap_mm=None, mm=None,
                              tilt_deg=(0.0, 0.0))
                     for s in session.protocol.ovs_slabs], dialog.close(), slabs())).props(
                    "flat dense no-caps").tooltip(
                    "every slab on, untilted, at the card's gap and thickness")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense no-caps")
        dialog.open()

    def patient():
        """The person on the table, as a scanner's registration takes them: age, sex, health;
        the rows a person carries (the water's relaxation, breathing, motion) are pooled over
        people like them, a healthy adult where nothing is given."""
        person = session.person
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 420px"):
            ui.label("Patient registration").classes("label")
            age = ui.number("age (years)", value=person.get("age_y"), min=0, max=120,
                            step=1).props("dense outlined dark clearable").classes(
                "w-full").tooltip("people of ten years either side are pooled; empty is any "
                                  "adult")
            sex = ui.select({None: "not given", "F": "female", "M": "male"},
                            value=person.get("sex"), label="sex").props(
                "dense outlined dark").classes("w-full")
            health = ui.select(sorted({"healthy", *stated("health")}),
                               value=person.get("health", "healthy"), label="health",
                               with_input=True, new_value_mode="add-unique").props(
                "dense outlined dark").classes("w-full").tooltip(
                "the conditions the measured database holds; a condition it holds no rows of "
                "keeps the literature")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense no-caps")

                def admit():
                    before = session.subject
                    session.admit(age_y=age.value, sex=sex.value,
                                  health=None if health.value == "healthy" else health.value)
                    dialog.close()
                    if session.subject != before:    # the template of the person's age
                        pickers["subject"].set_options(session.anatomies(), value=session.subject)
                        room()
                    draw()
                    ui.notify("Registered: " + (", ".join(
                        f"{value:g} years" if key == "age_y" else str(value)
                        for key, value in session.person.items()) or "a healthy adult"))
                ui.button("Register", icon="how_to_reg", on_click=admit).mark(
                    "register the patient").props("unelevated dense no-caps")
        dialog.open()

    def database():
        """What the simulation draws from, to browse: the measured datasets, each cohort's
        subjects with their traits and readings, and the typed literature with its sources."""
        with ui.dialog() as dialog, ui.card().classes("pane").style(
                "min-width: 1100px; max-width: 96vw"):
            ui.label("Database").classes("label")
            with ui.tabs().props("dense no-caps align=left") as tabs:
                measured, literature = ui.tab("Measured datasets"), ui.tab("Literature")
                parameters = ui.tab("Parameters")
            with ui.tab_panels(tabs, value=measured).classes("w-full").style(
                    "background: transparent"):
                with ui.tab_panel(measured):
                    frame = table()
                    frame["field"] = (2.0 * frame["field_t"]).round() / 2.0
                    labels = {r["name"]: r["dataset"] + (" (yours)" if r["yours"] else "")
                              for r in Session.catalogue()}
                    cited = {r["name"]: f"{r['source']}; {r['citation']}".strip("; ")
                             for r in Session.catalogue()}
                    ui.label("every reading of the measured database, a row each: narrow it by the "
                             "facets (the counts are what each choice would leave), by a query in "
                             "pandas syntax, or in the columns' headers; a row opens its subject"
                             ).classes("value").style("white-space: normal")
                    facets, chosen_now = {}, {"rows": frame}
                    names = {"dataset": "dataset", "leaf": "leaf", "field": "field (T)",
                             "vendor": "vendor", "method": "method", "sex": "sex",
                             "health": "health"}
                    with ui.row().classes("w-full no-wrap items-start").style("gap: 14px"):
                        with ui.column().style("width: 260px; gap: 4px"):
                            for key, label in names.items():
                                facets[key] = ui.select({}, multiple=True, label=label,
                                                        on_change=lambda: narrow()).props(
                                    "dense outlined dark use-chips clearable").classes(
                                    "w-full").mark(f"facet {key}")
                            ui.label("age (years)").classes("label").style(
                                "margin-bottom: 30px")
                            ages = ui.range(min=0, max=90, step=1,
                                            value={"min": 0, "max": 90}).props(
                                "label-always dark").on("change", lambda: narrow()).mark(
                                "facet age")
                            failed = ui.switch("failed readings too", value=False,
                                               on_change=lambda: narrow()).mark("facet failed")
                        with ui.column().classes("col").style("min-width: 0"):
                            with ui.row().classes("w-full items-center no-wrap"):
                                query = ui.input(
                                    "query (pandas): leaf == 'noise.sigma' and age_low >= 40",
                                    on_change=lambda: narrow()).props(
                                    "dense outlined dark clearable debounce=600").classes(
                                    "col").mark("database query")
                                ui.button(icon="download", on_click=lambda: ui.download.content(
                                    chosen_now["rows"].drop(columns="field").to_csv(index=False),
                                    "mrsandbox_database.csv")).props(
                                    "flat dense round").tooltip("these rows as CSV")
                            counted = ui.label().classes("value").mark("database count")
                            columns = ("dataset", "subject", "age", "sex", "health", "leaf",
                                       "value", "method", "field_t", "vendor", "model", "failed")
                            wide = {"dataset": 150, "leaf": 190, "method": 170, "model": 130}
                            grid = ui.aggrid({
                                "columnDefs": [
                                    {"field": key, "headerName": key.replace("_t", " (T)"),
                                     "minWidth": wide.get(key, 70),
                                     "filter": ("agNumberColumnFilter"
                                                if key in ("value", "field_t")
                                                else "agTextColumnFilter")}
                                    for key in columns],
                                "defaultColDef": {"sortable": True, "resizable": True,
                                                  "floatingFilter": True},
                                "rowData": []},
                                theme="balham").style("height: 520px").classes("w-full").mark(
                                "database grid")
                            grid.on("cellClicked", lambda event: subject(event.args["data"]))

                    def narrow():
                        rows = frame if failed.value else frame[frame["failed"] == ""]
                        low, high = ages.value["min"], ages.value["max"]
                        if (low, high) != (0, 90):
                            rows = rows[(rows["age_low"] >= low)
                                        & (rows["age_high"].fillna(np.inf) <= high)]
                        if query.value:
                            try:
                                rows = rows.query(query.value)
                                query.props(remove="error")
                            except Exception as error:                     # noqa: BLE001
                                query.props(f'error error-message="{str(error)[:80]}"')
                        picked = {key: facet.value or [] for key, facet in facets.items()}
                        # each facet's counts under every other facet's choice
                        for key, facet in facets.items():
                            others = rows
                            for other, values in picked.items():
                                if other != key and values:
                                    others = others[others[other].isin(values)]
                            counts = others[key].dropna().value_counts()
                            named = labels if key == "dataset" else {}
                            options = {value: f"{named.get(value, value)} ({counts.get(value, 0)})"
                                       for value in sorted(set(counts.index) | set(picked[key]),
                                                           key=str)}
                            facet.set_options(options, value=picked[key])
                        for key, values in picked.items():
                            if values and key != "leaf":
                                rows = rows[rows[key].isin(values)]
                        # the people the filter keeps, whatever leaf: the Parameters chart's
                        chosen_now["people"] = set(zip(rows["dataset"], rows["subject"]))
                        if picked["leaf"]:
                            rows = rows[rows["leaf"].isin(picked["leaf"])]
                        chosen_now["rows"] = rows
                        people = rows[["dataset", "subject"]].drop_duplicates().shape[0]
                        counted.text = (f"{len(rows)} readings of {people} subjects in "
                                        f"{rows['dataset'].nunique()} datasets")
                        shown = rows.assign(age=[
                            "" if np.isnan(lo) else f"{lo:g}" if lo == hi else
                            f"{lo:g} to {'' if np.isnan(hi) else f'{hi:g}'}".strip()
                            for lo, hi in zip(rows["age_low"], rows["age_high"])])
                        grid.options["rowData"] = [
                            {key: (None if isinstance(value, float) and np.isnan(value)
                                   else value) for key, value in row.items()}
                            for row in shown[list(columns)].to_dict("records")]
                        grid.update()

                    def subject(row):
                        """One subject's card: who, and every reading with its conditions."""
                        mine = frame[(frame["dataset"] == row["dataset"])
                                     & (frame["subject"] == row["subject"])]
                        first = mine.iloc[0]
                        with ui.dialog() as card, ui.card().classes("pane").style(
                                "min-width: 1200px; max-width: 94vw"):
                            ui.label(f"{row['subject']} · {labels.get(row['dataset'])}").classes(
                                "label")
                            ui.label(", ".join(str(v) for v in (
                                row["age"] and f"{row['age']} years", first["sex"],
                                first["health"]) if v and v == v)
                                + (f" (taken: {first['taken']})" if first["taken"] else "")
                            ).classes("value")
                            ui.label(cited.get(row["dataset"], "")).classes("value").style(
                                "white-space: normal")
                            ui.table(columns=[{"name": k, "label": k, "field": k, "align": "left"}
                                              for k in ("leaf", "value", "low", "high", "method",
                                                        "sequence", "field_t", "model",
                                                        "channels", "date", "failed")],
                                     rows=mine.astype(object).where(mine.notna(), "").to_dict(
                                         "records"), row_key="leaf").props(
                                "dense flat dark").classes("w-full").mark("subject card")
                            with ui.row().classes("w-full"):
                                ui.space()
                                ui.button("Close", on_click=card.close).props("flat dense no-caps")
                        card.open()
                    narrow()
                with ui.tab_panel(literature):
                    cards = Database.cards()
                    names = {f"{c['module']} · {c['name']}": c for c in cards}
                    ui.label(f"{len(cards)} typed tables, each with its source").classes("value")
                    card = ui.column().classes("w-full")

                    def read(name):
                        card.clear()
                        if name not in names:
                            return
                        with card:
                            for field in ("source", "data", "units"):
                                if names[name][field]:
                                    with ui.row().classes("w-full no-wrap"):
                                        ui.label(field).classes("label").style("width: 60px")
                                        ui.label(names[name][field]).classes("value").style(
                                            "white-space: normal")
                            ui.code(names[name]["text"], language="python").classes("w-full")
                    ui.select(list(names), label="table", with_input=True,
                              on_change=lambda event: read(event.value)).props(
                        "dense outlined dark").style("width: 420px").mark("literature table")
                with ui.tab_panel(parameters):
                    ui.label("every subject's reading of one leaf against age: in colour the "
                             "datasets this room's pool draws on, in grey the rows its "
                             "conditions (field, method, person, system) leave out; the "
                             "literature and this room's pooled value beside, each its 5th to "
                             "95th percentile").classes("value").style("white-space: normal")
                    ui.select(sorted(frame["leaf"].unique()),
                              label="leaf", with_input=True,
                              on_change=lambda event: plot(event.value)).props(
                        "dense outlined dark").style("width: 320px").mark("parameter leaf")
                    plotted = ui.column().classes("w-full")

                    def plot(leaf):
                        plotted.clear()
                        if not leaf:
                            return
                        seen = session.parameter(leaf)
                        everyone = len(seen["points"])
                        # the people the Measured tab's filter keeps
                        seen["points"] = [p for p in seen["points"]
                                          if (p["dataset"], p["subject"]) in chosen_now["people"]]
                        shares = {p["dataset"]: p["share"] for p in seen["points"] if p["share"]}
                        # a pool of more than three datasets folds into one series
                        named = (sorted(shares, key=lambda d: (-shares[d], d))
                                 if len(shares) <= len(CATEGORY) else [])
                        folded = ("other datasets of this room's pool" if named
                                  else "the datasets of this room's pool")
                        groups = {}
                        for p in seen["points"]:
                            if p["age_y"] is None:
                                continue
                            key = (f"{p['dataset']} · {p['method']}" if p["dataset"] in named
                                   else folded if p["share"]
                                   else "outside this room's conditions")
                            groups.setdefault(key, []).append(
                                {"name": f"{p['subject']} · {p['dataset']} · {p['method']}",
                                 "value": [p["age_y"], p["value"]]})
                        methods = {p["dataset"]: p["method"] for p in seen["points"]}
                        colours = {f"{d} · {methods[d]}": c for d, c in zip(named, CATEGORY)}

                        def style(key):
                            if key in colours:
                                return {"color": colours[key]}
                            if key == folded:
                                return {"color": INK}
                            return {"color": "transparent", "borderColor": DIM, "borderWidth": 1}
                        series = [{"type": "scatter", "name": key, "symbolSize": 8,
                                   "itemStyle": {"borderColor": PANEL, "borderWidth": 1,
                                                 **style(key)},
                                   "data": data} for key, data in sorted(
                            groups.items(), key=lambda kv: (kv[0] not in colours, kv[0]))]
                        for source, band in (("literature", seen["literature"]),
                                             ("pooled", seen["pooled"])):
                            if band is None:
                                continue
                            said = "this room's pool" if source == "pooled" else source
                            series.append({
                                "type": "scatter", "name": said, "data": [],
                                "itemStyle": {"color": SOURCE_COLORS[source]},
                                "markArea": {"silent": True, "itemStyle": {
                                    "color": SOURCE_COLORS[source], "opacity": 0.12},
                                    "data": [[{"yAxis": band[0]}, {"yAxis": band[2]}]]},
                                "markLine": {"silent": True, "symbol": "none", "data": [
                                    {"yAxis": band[1]}], "lineStyle": {
                                    "color": SOURCE_COLORS[source], "width": 2, "type": "solid"},
                                    "label": {"formatter": f"{said} {band[1]:.3g}",
                                              "color": INK, "position": "insideEndTop"}}})
                        axis = {"axisLine": {"lineStyle": {"color": LINE}},
                                "axisLabel": {"color": DIM}, "nameTextStyle": {"color": DIM},
                                "splitLine": {"show": False}}
                        unaged = sum(1 for p in seen["points"] if p["age_y"] is None)
                        # the value axis spans the readings and both bands
                        values = [p["value"] for p in seen["points"]] + [
                            v for band in (seen["literature"], seen["pooled"]) if band
                            for v in (band[0], band[2])]
                        pad = 0.05 * ((max(values) - min(values)) or abs(max(values)) or 1.0)
                        with plotted:
                            ui.echart({"backgroundColor": "transparent", "animation": False,
                                       "legend": {"textStyle": {"color": INK}, "top": 0},
                                       "grid": {"left": 64, "right": 24, "top": 56, "bottom": 40},
                                       "tooltip": {"trigger": "item",
                                                   "formatter": "{a}<br/>{b}<br/>age, value: {c}"},
                                       "xAxis": {**axis, "type": "value", "min": 0,
                                                 "name": "age (years)", "nameLocation": "middle",
                                                 "nameGap": 26},
                                       "yAxis": {**axis, "type": "value", "name": leaf,
                                                 "min": float(f"{min(values) - pad:.3g}"),
                                                 "max": float(f"{max(values) + pad:.3g}")},
                                       "series": series}).style("height: 460px").classes(
                                "w-full").mark("parameter chart")
                            ui.label(f"{len(seen['points'])} readings (of {everyone}: the "
                                     f"Measured tab's filter); {unaged} of them state no age (or "
                                     "only an open range) and are not drawn").classes("value")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense no-caps")
        dialog.open()

    # the top bar: what is in the room
    with ui.row().classes("w-full items-center pane no-wrap").style("gap: 14px; padding: 4px 10px"):
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" height="40" style="display: block; height: 40px" '
                    f'src="{MARK}">', sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 16px; letter-spacing: .1em")
        for attribute, caption, options, width in (
                ("scanner", "scanner", session.scanners(), 230),
                ("coil", "receive array", session.arrays(), 215),
                ("subject", "anatomy or phantom", session.anatomies(), 240)):
            pickers[attribute] = ui.select(options, value=getattr(session, attribute),
                                           label=caption, on_change=changed(attribute)).props(
                "dense outlined dark").style(f"min-width: {width}px")
            if attribute == "scanner":
                pickers["field_t"] = ui.number(label="field (T)",
                                               value=session.profile().scanner.b0_t, min=0.5,
                                               max=14.0, step=0.5, precision=3,
                                               on_change=field_changed).props(
                    "dense outlined dark").style("width: 92px").tooltip(
                    "the field this scanner runs at: its limits and records stay (a record names "
                    "the departure), the priors take the nearest typed field, 1.5, 3 or 7 T")
            if attribute in GEARS or attribute == "coil":
                ui.button(icon="settings", on_click={"scanner": scanner_builder,
                                                     "subject": lambda: maps(),
                                                     "coil": coils_builder}[attribute]
                          ).mark(f"{attribute} settings").props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip({
                        "subject": "the parameters and ranges of the anatomy or phantom",
                        "coil": "the receive array: its loops, your own sensitivity maps, the "
                                "noise its channels share"}.get(
                    attribute, f"the {attribute}'s parameters and ranges"))
        ui.button(icon="tune", on_click=lambda: gear()).mark("all settings").props(
            "flat dense round size=sm").style(f"color: {GLOW}").tooltip(
            "all settings: every parameter of the simulation as lines, grouped, each with its "
            "source")
        ui.button(icon="restart_alt", on_click=lambda: (
            session.reset_room(),
            pickers["scanner"].set_options(session.scanners(), value=session.scanner),
            pickers["field_t"].set_value(session.profile().scanner.b0_t), room(), draw(),
            ui.notify("The room is the package's again"))).mark("reset").props(
            "flat dense round size=sm").style(f"color: {GLOW}").tooltip(
            "reset: the room back to the package's, a generic scanner on the pooled rows, no "
            "setting, choice, dataset or file of yours")
        ui.button(icon="person_add_alt", on_click=lambda: (
            setattr(session, "seed", int(np.random.default_rng().integers(2 ** 31))),
            ui.notify("A new draw of the population is scanned from here on"))).props(
            "flat dense round size=sm").style(f"color: {GLOW}; margin-left: -10px").tooltip(
            "every exam scans the same draw of the population; this takes a new one")
        ui.button(icon="storage", on_click=database).mark("database").props(
            "flat dense round size=sm").style(f"color: {GLOW}; margin-left: -10px").tooltip(
            "the database: the measured datasets and their subjects, the literature and its "
            "sources")
        ui.button(icon="badge", on_click=patient).mark("patient").props(
            "flat dense round size=sm").style(f"color: {GLOW}; margin-left: -10px").tooltip(
            "the patient: age, sex and health, which the measured rows they carry are pooled by")
        ui.space()
        status = ui.label("idle").classes("label")

    # three panes whose widths the operator drags: the exam cards, the planning, scan control
    with ui.splitter(value=244, limits=(160, 520)).props("unit=px").classes("w-full").style(
            "flex: 1; min-height: 0") as left:
        with left.before:
            # the exam cards
            with ui.column().classes("pane").style("width: 100%; height: 100%"):
                ui.label("Exam cards").classes("label")
                with ui.scroll_area().style("flex: 1; width: 100%"):
                    rows = ui.column().classes("w-full").style("gap: 0")
                ui.button("New scan", icon="add", on_click=browse).props("outline no-caps").classes(
                    "w-full")
                ui.label("Protocol library").classes("label").style("margin-top: 4px")
                saved = ui.select(session.protocols(), label="name", with_input=True,
                                  new_value_mode="add-unique").props(
                    "dense outlined dark").classes("w-full")
                with ui.row().classes("w-full no-wrap").style("gap: 6px"):
                    ui.button("Save", icon="save", on_click=lambda: saved.value and (
                        session.save_protocol(saved.value),
                        saved.set_options(session.protocols(), value=saved.value),
                        ui.notify(f"Saved {saved.value}"))).props("outline dense no-caps").style(
                        "flex: 1")
                    ui.button("Load", icon="folder_open",
                              on_click=lambda: saved.value in session.protocols() and (
                                  session.load_protocol(saved.value), draw(),
                                  ui.notify(f"Loaded {saved.value}"))).props(
                        "outline dense no-caps").style("flex: 1")

        with left.after:
            with ui.splitter(value=244, limits=(200, 520)).props("reverse unit=px").classes(
                    "w-full").style("height: 100%") as right:
                with right.before:
                    # planning: the tools, three planes one leafs through and the room in 3D, the
                    # card below
                    with ui.column().style(
                            "width: 100%; height: 100%; min-width: 0; gap: 6px; padding: 0 6px"):
                        with ui.row().classes("w-full items-center pane no-wrap").style(
                                "padding: 4px 8px"):
                            # what changes the view alone ...
                            ui.label("view").classes("label").style(
                                "writing-mode: vertical-rl; transform: rotate(180deg)")
                            ui.button("to the voxel", icon="center_focus_strong", on_click=lambda: (
                                setattr(session, "view_mm", list(session.position_mm)), paint())
                            ).props("flat dense no-caps stack").classes("tool").tooltip(
                                "bring the three planes back to the voxel (v)")
                            tools["shifts"] = ui.button(
                                "shifts", icon="flip_to_back", on_click=lambda: (
                                    chosen.__setitem__("shifts", not chosen["shifts"]), draw())
                            ).props("flat dense no-caps stack").classes("tool on").tooltip(
                                "where water (blue) and lipid (pink) are really selected (s); "
                                "display only")
                            tools["flip"] = ui.button(
                                "L on the right", icon="flip", on_click=lambda: (
                                    chosen.__setitem__("flipped", not chosen["flipped"]), draw())
                            ).props("flat dense no-caps stack").classes("tool").tooltip(
                                "mirror the pictures: left on the right, as a radiologist reads "
                                "them (f)")
                            tools["slabs"] = ui.button(
                                "OVS slabs", icon="border_outer", on_click=lambda: (
                                    chosen.__setitem__("slabs", not chosen["slabs"]), room(),
                                    draw())
                            ).props("flat dense no-caps stack").classes("tool on").tooltip(
                                "draw the saturation slabs of a card that plays OVS, on the planes "
                                "and in the room; display only")
                            tools["overlay"] = ui.button(session.overlay, icon="opacity").props(
                                "flat dense no-caps stack icon-right=arrow_drop_down").classes(
                                "tool").tooltip(
                                "a drop-down: wash a map over the anatomy (o: the next one)")
                            with tools["overlay"]:
                                overlays = ui.menu().props("dark")
                            texts["window"] = ui.label("").classes("label").style(
                                "text-transform: none; letter-spacing: 0")
                            ui.space()
                            # ... and what changes the simulated room and its scans
                            ui.label("sim").classes("label").style(
                                "writing-mode: vertical-rl; transform: rotate(180deg)")
                            ui.button("switches", icon="toggle_on", on_click=switches).props(
                                "flat dense no-caps stack").classes("tool").tooltip(
                                "what is real about the scan and what happens over a run, each on "
                                "or off; the numbers live in the cards")
                            ui.icon("keyboard", size="20px").style(f"color: {DIM}").tooltip(
                                "scroll: leaf through · press or drag: place the voxel · with "
                                "shift: look around · " + KEYS)
                        with ui.element("div").classes("w-full").style(
                                "display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px"):
                            for name in ("tra", "sag", "cor"):
                                with ui.element("div").classes("port"):
                                    views[name] = ui.interactive_image(
                                        events=["mousedown", "mousemove", "mouseup"], cross=False
                                    ).classes("w-full")
                                    # throttled: an unthrottled stream of moves queues more repaints
                                    # than the window can show, and the console then seems deaf to
                                    # the next clicks
                                    views[name].on("mouse", drag(name), throttle=0.06)
                                    views[name].on("wheel.prevent", lambda e, plane=name: (
                                        session.scroll(plane, 1 if e.args["deltaY"] < 0 else -1),
                                        paint()), args=["deltaY"], throttle=0.04)
                                    views[name].on("mouseenter", lambda _, plane=name: (
                                        chosen.__setitem__("over", plane)))
                                    # this plane's own zoom
                                    with ui.row().style("position: absolute; top: 2px; right: 4px; "
                                                        "z-index: 3; gap: 0"):
                                        for icon, factor in (("remove", 1.0 / 1.5),
                                                             ("crop_free", None), ("add", 1.5)):
                                            ui.button(icon=icon, on_click=lambda _, plane=name,
                                                      factor=factor: zoom(plane, factor)).props(
                                                "flat dense round size=xs").style(f"color: {SAND}")
                                    texts[name] = ui.label("").classes("tag").style(
                                        "top: 4px; left: 6px")
                                    letters[name] = [
                                        ui.label(letter).classes("tag").style(place)
                                        for letter, place in zip(EDGES[name][:4], (
                                            "top: 4px; left: 50%", "bottom: 4px; left: 50%",
                                            "top: 48%; left: 6px", "top: 48%; right: 6px"))]
                                    # the axes' cross, lower right: where the picture's right and
                                    # its top point in the subject
                                    crosses[name] = ui.html("", sanitize=False).classes(
                                        "tag").style("right: 6px; bottom: 6px")
                        with ui.splitter(value=590, limits=(320, 1100)).props("unit=px").classes(
                                "w-full").style("flex: 1; min-height: 0") as below:
                            with below.before:
                                # the card's parameters by tab, what follows from them beside
                                with ui.column().classes("pane").style(
                                        "width: 100%; height: 100%; min-height: 0; "
                                        "overflow: hidden"):
                                    card = session.protocol
                                    with ui.row().classes("w-full items-center no-wrap"):
                                        with ui.tabs().props("dense align=left").style(
                                                f"color: {DIM}; flex: 1") as tabs:
                                            for name in ("Geometry", "Timing", "Acquisition",
                                                         "Adjustments"):
                                                ui.tab(name)
                                            ui.tab("Imaging").bind_visibility_from(
                                                card, "sequence",
                                                backward=lambda name: name == MRSI)
                                        ui.button(icon="settings", on_click=lambda: gear(tabs.value)
                                                  ).props("flat dense round size=sm").style(
                                            f"color: {GLOW}").tooltip(
                                            "the advanced settings behind this tab")
                                    with ui.row().classes("w-full no-wrap").style(
                                            "gap: 12px; flex: 1; min-height: 0"):
                                        with ui.tab_panels(tabs, value="Geometry").props(
                                                "keep-alive").style(
                                                "width: 215px; flex: none; overflow-y: auto; "
                                                "max-height: calc(100vh - 650px)"):
                                            with ui.tab_panel("Geometry"):
                                                for axis, caption in enumerate((
                                                        "x, to the right (mm)",
                                                        "y, to the front (mm)",
                                                        "z, to the head (mm)")):
                                                    places.append(ui.number(
                                                        caption, value=session.position_mm[axis],
                                                        step=1.0, on_change=lambda e, axis=axis: (
                                                            e.value is not None and abs(
                                                                e.value - session.position_mm[axis]
                                                            ) > 0.05 and (
                                                                session.position_mm.__setitem__(
                                                                    axis, float(e.value)),
                                                                draw()))).props(
                                                        "dense outlined dark").classes("w-full"))
                                                with ui.row().classes("w-full no-wrap").style(
                                                        "gap: 4px"):
                                                    for axis, caption in enumerate(
                                                            ("size x", "size y", "size z")):
                                                        edges.append(ui.number(
                                                            caption, value=session.size_mm[axis],
                                                            step=5.0, min=3.0,
                                                            on_change=lambda e, axis=axis: (
                                                                e.value is not None
                                                                and e.value != session.size_mm[axis]
                                                                and (session.size_mm.__setitem__(
                                                                         axis, float(e.value)),
                                                                     room(), draw()))).props(
                                                            "dense outlined dark").style(
                                                            "flex: 1; min-width: 0"))
                                                ui.select({axes: ", ".join(axes)
                                                           for axes in ("xyz", "xzy", "yxz", "yzx",
                                                                        "zxy", "zyx")},
                                                          label="slice order (real pulses)",
                                                          on_change=changed("axes", "card")
                                                          ).bind_value(card, "axes").props(
                                                    "dense outlined dark options-dense").classes(
                                                    "w-full").tooltip(
                                                    "the axes the first, second and third slice "
                                                    "pulse select along; the narrowest pulse "
                                                    "displaces a shift the most")
                                                ui.number("OVS gap to the voxel (mm)", step=1.0,
                                                          min=0.0,
                                                          on_change=changed("ovs_gap_mm", "card")
                                                          ).bind_value(card, "ovs_gap_mm").props(
                                                    "dense outlined dark").classes(
                                                    "w-full").bind_visibility_from(
                                                    card, "ovs").tooltip(
                                                    "the slabs are placed for water; a "
                                                    "metabolite's slab lies some mm nearer, so a "
                                                    "small gap costs signal")
                                                ui.number("OVS slab thickness (mm)", step=5.0,
                                                          min=5.0,
                                                          on_change=changed("ovs_mm", "card")
                                                          ).bind_value(card, "ovs_mm").props(
                                                    "dense outlined dark").classes(
                                                    "w-full").bind_visibility_from(
                                                    card, "ovs").tooltip(
                                                    "six saturation slabs, a pair along every axis")
                                                ui.button("OVS slabs one by one", icon="tune",
                                                          on_click=lambda: slabs()).mark(
                                                    "slabs").props(
                                                    "outline dense no-caps").classes(
                                                    "w-full").bind_visibility_from(card, "ovs")
                                            # what a scanner adjusts before it scans
                                            with ui.tab_panel("Adjustments"):
                                                ui.select(list(FieldMap.SHIMS), value=session.shim,
                                                          label="shim", on_change=changed("shim")
                                                          ).props("dense outlined dark").classes(
                                                    "w-full")
                                                ui.number("shim box (mm)", value=session.shim_mm,
                                                          step=5.0, min=5.0,
                                                          on_change=lambda e: (
                                                              setattr(session, "shim_mm", e.value),
                                                              draw())).props(
                                                    "dense outlined dark clearable").classes(
                                                    "w-full").tooltip(
                                                    "the yellow box about the voxel the shim is "
                                                    "fitted over; empty, the shim is fitted over "
                                                    "the voxel itself")
                                                ui.switch("centre the frequency before every exam",
                                                          value=session.prescan,
                                                          on_change=changed("prescan")).props(
                                                    "dense color=accent").tooltip(
                                                    "on, as scanners do: every exam starts on "
                                                    "resonance. off: it starts where the "
                                                    "session's drift has taken the scanner")
                                            with ui.tab_panel("Imaging"):
                                                with ui.row().classes("w-full no-wrap").style(
                                                        "gap: 4px"):
                                                    for attribute, caption, low in (
                                                            ("matrix", "cells x", 2),
                                                            ("matrix_y", "cells y", 2),
                                                            ("slices", "slices z", 1)):
                                                        ui.number(
                                                            caption, step=1, min=low,
                                                            on_change=changed(attribute, "card")
                                                        ).bind_value(card, attribute).props(
                                                            "dense outlined dark").style(
                                                            "flex: 1; min-width: 0")
                                                for attribute, caption, options, tip in (
                                                        ("mrsi_oversampling",
                                                         "rendering under a cell", [1, 2, 3],
                                                         "1: ideal boxes. More: every cell sees "
                                                         "its neighbours through the point spread "
                                                         "function, as a real image does"),
                                                        ("mrsi_weighting", "k-space weighting",
                                                         ["none", "hamming"],
                                                         "hamming: less bleed, wider cells"),
                                                        ("mrsi_sampling", "k-space coverage",
                                                         ["full", "elliptical"],
                                                         "elliptical: the corners left out, a "
                                                         "shorter scan"),
                                                        ("mrsi_profile", "slab profile",
                                                         ["box", "gaussian"],
                                                         "how the slab is excited through its "
                                                         "thickness"),
                                                        ("mrsi_readout", "readout",
                                                         ["phase encoded", "epsi",
                                                          "epsi-bidirectional"],
                                                         "EPSI reads a line of k-space per "
                                                         "repetition: a matrix times shorter scan, "
                                                         "its own artefacts")):
                                                    ui.select(options, label=caption,
                                                              on_change=changed(attribute, "card")
                                                              ).bind_value(card, attribute).props(
                                                        "dense outlined dark options-dense"
                                                    ).classes("w-full").tooltip(tip)
                                            for tab, fields in (
                                                    ("Timing", (
                                                        ("te_ms", "TE (ms)", 1.0),
                                                        ("tr_ms", "TR (ms)", 100.0))),
                                                    ("Acquisition", (
                                                        ("averages", "averages", 1),
                                                        ("npoints", "points", 512),
                                                        ("bandwidth_hz", "bandwidth (Hz)", 500.0),
                                                        ("water_reference",
                                                         "water reference transients", 1)))):
                                                with ui.tab_panel(tab):
                                                    if tab == "Timing":
                                                        ui.button(on_click=browse).bind_text_from(
                                                            card, "sequence").props(
                                                            "outline no-caps icon-right=expand_more"
                                                        ).classes("w-full")
                                                    if tab == "Acquisition":
                                                        ui.switch("reference before the scan",
                                                                  on_change=changed(
                                                                      "water_reference_first",
                                                                      "card")).bind_value(
                                                            card, "water_reference_first").props(
                                                            "dense color=accent").tooltip(
                                                            "the water reference is an exam of "
                                                            "its own: before its card, or after it "
                                                            "(off)")
                                                    if tab == "Timing":
                                                        with ui.column().classes("w-full").style(
                                                                "gap: 4px").bind_visibility_from(
                                                                card, "sequence",
                                                                backward=lambda name: (
                                                                    name == "MEGA-PRESS")):
                                                            ui.select(
                                                                list(TARGETS),
                                                                label="editing target",
                                                                on_change=lambda e: e.value and (
                                                                    setattr(card, "edit_on_ppm",
                                                                            TARGETS[e.value][0]),
                                                                    setattr(card, "edit_off_ppm",
                                                                            TARGETS[e.value][1]),
                                                                    draw())).props(
                                                                "dense outlined dark").classes(
                                                                "w-full").tooltip(
                                                                "fills the two frequencies below; "
                                                                "any pair can be typed")
                                                            for attribute, caption in (
                                                                    ("edit_on_ppm",
                                                                     "editing pulse ON (ppm)"),
                                                                    ("edit_off_ppm",
                                                                     "editing pulse OFF (ppm)")):
                                                                ui.number(caption, step=0.01,
                                                                          format="%.2f",
                                                                          on_change=changed(
                                                                              attribute, "card")
                                                                          ).bind_value(
                                                                    card, attribute).props(
                                                                    "dense outlined dark").classes(
                                                                    "w-full")
                                                    for attribute, caption, step in fields:
                                                        ui.number(
                                                            caption, value=getattr(card, attribute),
                                                            step=step,
                                                            on_change=changed(attribute, "card")
                                                        ).bind_value(card, attribute).props(
                                                            "dense outlined dark").classes("w-full")
                                        with ui.element("div").classes("info").style(
                                                f"flex: 1; min-height: 0; "
                                                f"max-height: calc(100vh - 650px); "
                                                f"overflow-y: auto; border: 1px solid {LINE}; "
                                                f"border-radius: 2px; padding: 8px 10px; "
                                                f"background: {GROUND}"):
                                            ui.label("Overview").classes("label").style(
                                                f"grid-column: 1 / -1; color: {SAND}")
                                            for what, caption in (
                                                    ("time", "scan duration"),
                                                    ("timing", "TR / TE"),
                                                    ("needs", "one repetition"),
                                                    ("volume", "voxel volume"),
                                                    ("snr", "relative SNR"), ("tissue", "tissue"),
                                                    ("displaced", "displaced voxels"),
                                                    ("lipid", "lipid layer"),
                                                    ("linewidth", "line width from B0"),
                                                    ("encoding", "image encoding"),
                                                    ("resolution", "spectral resolution"),
                                                    ("readout", "readout duration"),
                                                    ("limits", "hardware")):
                                                ui.label(caption).classes("label")
                                                # not `key`: the keyboard's handler below
                                                texts[what] = ui.label("").classes("value").style(
                                                    "white-space: pre-line")
                            with below.after:
                                # what the card plays and what came of it
                                with ui.column().classes("pane").style(
                                        "width: 100%; height: 100%; min-width: 0"):
                                    with ui.row().classes("w-full items-center no-wrap"):
                                        with ui.tabs().props("dense align=left").style(
                                                f"color: {DIM}; flex: 1") as shows:
                                            ui.tab("Sequence")
                                            ui.tab("Live")
                                        ui.button("Open in the designer", icon="architecture",
                                                  on_click=lambda: (
                                                      session.protocol.sequence != "Custom"
                                                      and session.adopt(), designer())).mark(
                                            "design").props("outline dense no-caps").tooltip(
                                            "this sequence block by block")
                                    with ui.tab_panels(shows, value="Sequence").props(
                                            "keep-alive").classes("w-full").style("flex: 1"):
                                        with ui.tab_panel("Sequence"):
                                            with ui.row().classes("w-full no-wrap").style(
                                                    "gap: 8px"):
                                                with ui.column().style("gap: 4px; flex: none"):
                                                    tools["real"] = ui.button(
                                                        "real pulses", icon="waves",
                                                        on_click=lambda: toggle("real")).props(
                                                        "flat dense no-caps stack").classes(
                                                        "tool").tooltip(
                                                        "on: shaped pulses under gradients, played "
                                                        "in time (slower). off: ideal pulses that "
                                                        "select the voxel exactly")
                                                    tools["water"] = ui.button(
                                                        "water off", icon="water_drop",
                                                        on_click=lambda: toggle("water")).props(
                                                        "flat dense no-caps stack").classes(
                                                        "tool").tooltip(
                                                        "VAPOR, WET, CHESS: the train is played on "
                                                        "the full water. off: the whole water "
                                                        "line. assumed residual: a drawn residual, "
                                                        "fast")
                                                    tools["ovs"] = ui.button(
                                                        "OVS", icon="border_outer",
                                                        on_click=lambda: toggle("ovs")).mark(
                                                        "ovs").props(
                                                        "flat dense no-caps stack").classes(
                                                        "tool").tooltip(
                                                        "outer-volume suppression: six saturation "
                                                        "slabs about the voxel")
                                                diagram = ui.html("", sanitize=False).style(
                                                    "flex: 1; min-width: 0")
                                        with ui.tab_panel("Live"):
                                            with ui.row().classes("w-full items-center no-wrap"):
                                                looks = ui.select(
                                                    [], label="show", on_change=lambda: (
                                                        chosen["exam"] is not None
                                                        and chosen["exam"].frames
                                                        and (frame(chosen["exam"], -1), paint()))
                                                ).props("dense outlined dark options-dense").style(
                                                    "width: 200px")
                                                for level, said in (
                                                        (COMBINE,
                                                         "as the transients came in: averaged, no "
                                                         "phase, no alignment"),
                                                        (ALIGN,
                                                         "a rough cleaning: every transient "
                                                         "aligned in frequency and phase, "
                                                         "one phase"),
                                                        (PROCESS,
                                                         "fully processed: aligned, outliers out, "
                                                         "water removed, referenced, phased")):
                                                    levels[level] = ui.button(
                                                        level, on_click=lambda _, level=level: (
                                                            reprocess(level))).mark(
                                                        f"level {level}").props(
                                                        "flat dense no-caps").classes(
                                                        "tool"
                                                        + (" on" if level == session.processing
                                                           else "")).style(
                                                        "width: 92px; height: 30px").tooltip(said)
                                                ui.switch("every transient", value=True,
                                                          on_change=lambda e: (
                                                              chosen.__setitem__("every", e.value),
                                                              chosen["exam"] is not None
                                                              and chosen["exam"].frames
                                                              and frame(chosen["exam"], -1))).props(
                                                    "dense color=accent").tooltip(
                                                    "grey behind the average: every transient "
                                                    "that was kept, coils summed (32 at most)")
                                            spectrum = ui.echart(copy.deepcopy(SPECTRUM)).style(
                                                "height: 300px").classes("w-full")
                        # the card in one line, and its way into the queue
                        with ui.row().classes("w-full items-center pane no-wrap").style(
                                "gap: 18px; padding: 4px 10px"):
                            texts["name"] = ui.label("").style(f"color: {SAND}")
                            ui.label("duration").classes("label")
                            texts["time2"] = ui.label("").classes("value")
                            ui.label("computing").classes("label").tooltip(
                                "about how long this computer takes to simulate the card: what "
                                "the last exam of its kind took here, or for a tilted slab (the "
                                "whole voxel at once) its design timed once and scaled")
                            texts["compute"] = ui.label("").classes("value")
                            texts["verdict"] = ui.label("").classes("value").style("flex: 1")
                            discard = ui.button("Discard", on_click=lambda: (
                                chosen.__setitem__("editing", None), listing(), draw())).props(
                                "flat dense no-caps")
                            accept = ui.button("Accept", icon="playlist_add", on_click=plan).props(
                                "unelevated dense no-caps")

                with right.after:
                    # scan control, and what the scan runs under
                    with ui.column().classes("pane").style("width: 100%; height: 100%"):
                        ui.label("Scan control").classes("label")
                        progress = ui.linear_progress(value=0.0, show_value=False, size="8px",
                                                      color=SAND)
                        with ui.row().classes("w-full items-center"):
                            texts["count"] = ui.label("0 / 0").classes("value")
                            ui.space()
                            ui.button(icon="stop",
                                      on_click=lambda: chosen.__setitem__("stop", True)).props(
                                "outline dense round").tooltip(
                                "hold the queue after the exam that is running")
                            start = ui.button("Start", icon="play_arrow",
                                              on_click=scan_queue).props("unelevated dense no-caps")
                        texts["took"] = ui.label("").classes("label").style(
                            "text-transform: none; letter-spacing: 0")
                        ui.button("Review", icon="insights", on_click=review).props(
                            "outline dense no-caps align=left").classes("w-full")
                        ui.button("Export raw transients", icon="save_alt", on_click=export).props(
                            "outline dense no-caps align=left").classes("w-full")
                        ui.button("Export Pulseq .seq", icon="code", on_click=export_seq).props(
                            "outline dense no-caps align=left").classes("w-full")
                        with ui.element("div").classes("port w-full"):
                            view = ui.scene(width=226, height=226, grid=False,
                                            background_color="#000000")
                            view.move_camera(x=5.2, y=-5.2, z=3.6, look_at_x=0, look_at_y=0,
                                             look_at_z=0)
                            ui.label("the room · drag to turn").classes("tag").style(
                                "top: 4px; left: 6px")
                        panel = ui.label("This voxel").classes("label").tooltip(
                            "the truth of the person scanned; a new person: the person icon"
                        ).classes("label").style("margin-top: 6px")
                        with ui.scroll_area().style("flex: 1; width: 100%"):
                            records = ui.column().classes("w-full").style("gap: 1px")
    planning = ui.keyboard(on_key=key, ignore=["input", "select", "textarea"])
    chosen["ready"] = True
    room()
    draw()
    # not deleted: a window closed while the room was being made has no page to delete from
    loading.set_visibility(False)


#**********#
#   main   #
#**********#
def main():
    """Start the console in a window of its own; with MRSANDBOX_CONSOLE_PORT set it serves a
    browser at that port instead (MRSANDBOX_CONSOLE_HEADLESS: without opening one)."""
    import keras

    if keras.backend.backend() == "jax":
        # the engine's kernels are compiled once per machine, not once per launch
        import jax

        jax.config.update("jax_compilation_cache_dir",
                          str(pathlib.Path.home() / ".cache" / "mrsandbox" / "jax"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    if "MRSANDBOX_CONSOLE_PORT" in os.environ:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False,
               show="MRSANDBOX_CONSOLE_HEADLESS" not in os.environ,
               port=int(os.environ["MRSANDBOX_CONSOLE_PORT"]))
    else:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False, native=True,
               window_size=(1680, 1020))


if __name__ in {"__main__", "__mp_main__"}:
    main()
