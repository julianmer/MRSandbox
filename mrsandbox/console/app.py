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
import copy
import dataclasses
import io
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from nicegui import Client, run, ui
from PIL import Image

from mrsandbox.console.session import BLOCKS, EDITING, FIELDS, GEARS, IDEALS, LENT, MRSI, OVERLAYS, REAL, RUN, TARGETS, SEQUENCES, SUPPRESSION, TEMPLATES, Session
from mrsandbox.io import read_dicom
from mrsandbox.phantom import FieldMap

#*************#
#   palette   #
#*************#
# the logo's: sand and copper on charcoal
SAND, COPPER, GLOW, GROUND, PANEL, LINE, INK, DIM, GOOD, BAD = (
    "#E2A55F", "#B5622C", "#F5C77E", "#151210", "#1E1A17", "#3A322B", "#EADFD2", "#8E8378", "#7FA66B", "#D9653B")
STYLE = f"""
body {{ background: {GROUND}; color: {INK}; font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 12px; }}
.nicegui-content {{ padding: 6px; gap: 6px; height: 100vh; flex-wrap: nowrap; }}
.pane {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 2px; padding: 8px; gap: 6px; }}
.label {{ color: {DIM}; letter-spacing: .12em; text-transform: uppercase; font-size: 10px; }}
.value {{ color: {INK}; font-variant-numeric: tabular-nums; }}
.port {{ position: relative; border: 1px solid {LINE}; background: #000; min-width: 0; overflow: hidden; }}
.port img {{ -webkit-user-drag: none; user-select: none; }}
.port .tag {{ position: absolute; color: {SAND}; font-size: 11px; pointer-events: none; z-index: 2; text-shadow: 0 0 3px #000; }}
.exam {{ border-left: 3px solid transparent; border-bottom: 1px solid {LINE}; padding: 6px 8px; cursor: pointer; display: flex;
         gap: 8px; align-items: center; width: 100%; }}
.exam:hover {{ background: #26211d; }}
.exam.chosen {{ border-left-color: {SAND}; background: #2b241e; color: #fff; }}
.tool {{ border: 1px solid {LINE}; border-radius: 2px; color: {DIM} !important; width: 96px; height: 46px; padding: 0; font-size: 10.5px;
         line-height: 1.1; }}
.tool .q-icon {{ font-size: 19px; }}
.chip {{ border: 1px solid {LINE}; border-radius: 2px; color: {INK} !important; width: 118px; height: 50px; padding: 0; font-size: 10.5px;
         line-height: 1.15; cursor: grab; }}
.chip.over {{ border-left: 4px solid {GLOW}; }}
.chip.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.tool.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.info {{ display: grid; grid-template-columns: auto 1fr; column-gap: 12px; row-gap: 5px; align-content: start; white-space: nowrap; }}
.q-tab {{ min-height: 28px; padding: 0 12px; }} .q-tab__label {{ font-size: 11px; letter-spacing: .08em; }}
.q-tab-panel {{ padding: 8px 2px 2px 2px; }} .q-tab-panels {{ background: transparent; }}
.q-field__native, .q-field__label, .q-select__dropdown-icon, .q-field__input {{ color: {INK} !important; }}
.q-field--outlined .q-field__control:before {{ border-color: {LINE} !important; }}
"""
MM = 1.0 / 40.0  # the 3D view's units per millimetre
BLANK = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
# the logo has one home, the repository's assets/; an installed package carries that file in its
# environment's share/mrsandbox (pyproject.toml, data-files)
LOGO = next((path for path in (pathlib.Path(__file__).parents[2] / "assets" / "logo" / "logo.png",
                               pathlib.Path(sys.prefix) / "share" / "mrsandbox" / "logo.png") if path.exists()), None)
CHOICES = {"shape": ["sinc", "gaussian", "hermite"], "slice_axis": ["none", "x", "y", "z"], "axis": ["x", "y", "z"],
           "scheme": list(SUPPRESSION)}
# the designer's blocks in plain words: an icon, what the block is, and its settings with their units
BLOCK_ICONS = {"as the card plays it": "lock", "hard pulse": "bolt", "adiabatic pulse": "all_inclusive", "own pulse": "gesture", "shaped pulse": "waves", "delay": "hourglass_empty",
               "crusher": "signal_cellular_alt", "coherence filter": "filter_alt", "water suppression": "water_drop"}
BLOCK_NOTES = {
    "as the card plays it": "An event of the card's own sequence, taken over as it is: move, copy or remove it. Its shape is the card's.",
    "hard pulse": "An ideal pulse of no duration. It turns every spin alike, whatever its frequency.",
    "own pulse": "A pulse of your own shape. Type its envelope or load it from a file; the flip angle scales it.",
    "shaped pulse": "A sinc, Gaussian or Hermite pulse, played in time. With a slice gradient it selects a slab along an axis.",
    "adiabatic pulse": "A hyperbolic secant full passage. It inverts over its whole bandwidth whatever the B1, above a threshold; play them in pairs.",
    "delay": "Free evolution. The echo time is the sum of the delays and the pulses between excitation and readout.",
    "crusher": "A gradient pulse that dephases what is transverse. A pair about a 180 keeps the echo and spoils the rest.",
    "coherence filter": "An ideal crusher. It keeps the one coherence order given and removes every other, without time.",
    "water suppression": "A train of selective pulses on the water line before the excitation: VAPOR, WET or CHESS.",
}
CAPTIONS = {"flip_deg": "flip angle (°)", "phase_deg": "phase (°)", "duration_ms": "duration (ms)", "offset_ppm": "centre (ppm)",
            "shape": "shape", "slice_axis": "slice along", "slice_hz_per_mm": "slice gradient (Hz/mm)", "ms": "duration (ms)",
            "axis": "along", "cycles_per_mm": "dephasing (cycles/mm)", "order": "coherence order kept", "scheme": "scheme",
            "envelope": "envelope, amplitudes apart by commas",
            "bandwidth_hz": "bandwidth (Hz)", "peak_hz": "peak B1 (Hz)"}
# the calibration records in plain words
RECORDS = {"noise.sigma": "Noise level (the simulation's unit: one transient, 8 mL, 3 T)", "lineshape.gauss_hz": "Line broadening (Hz)",
           "conc.Cr": "Creatine level the amplitudes are scaled by (mM)", "coils.loaded.own_noise": "Share of a channel's noise that is its own",
           "mm.level_mm": "Macromolecule level"}
# the settings' groups by name, and the ways a parameter can be set
GROUPS = {"conc": "Concentrations (mM)", "t1": "T1 of the metabolites (s)", "t2": "T2 of the metabolites (s)",
          "contrast": "Grey to white matter concentration ratio", "diffusion": "Diffusion", "water": "Water", "lipids": "Lipids",
          "mm": "Macromolecules", "downfield": "Downfield peaks", "lineshape": "Line shape (Hz)", "noise": "Noise",
          "b1_scale": "B1 scale", "acquisition": "Acquisition: shifts, phases, eddy currents", "gradients": "Gradient eddy currents",
          "spurious": "Spurious echoes"}
WAYS = ["as its source gives it", "set to", "vary between", "from a map"]
OWN = "your own file ..."      # on top of every drop-down of stored maps
# the displaced voxels' colours on the planes (session.SHIFTED)
SHIFT_COLOURS = {"water": "#7FB2D9", "lipid": "#D98BB0", "shim": "#E6D84A"}
# the planes' edges as a console letters them (top, bottom, left, right), and the axis each plane leafs along
EDGES = {"tra": ("A", "P", "L", "R", 2, "z"), "sag": ("H", "F", "P", "A", 0, "x"), "cor": ("H", "F", "L", "R", 1, "y")}

#**********#
#   mark   #
#**********#
# the logo as its file has it, its own transparency kept, trimmed to what is drawn: small for the bar, and large
# for the loading screen at three device pixels per shown one, so that it stays sharp on a dense display
MARK = MARK_LARGE = None
if LOGO is not None:
    drawn = Image.open(LOGO).convert("RGBA")
    drawn = drawn.crop(drawn.getchannel("A").getbbox())
    sizes = {}
    for height in (100, 450):
        buffer = io.BytesIO()
        drawn.resize((round(height * drawn.width / drawn.height), height), Image.LANCZOS).save(buffer, format="PNG")
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
    `mark` (start, end) in ms lights a span up, the designer's chosen block."""
    names = ("RF", "Gx", "Gy", "Gz", "Signal")
    left, right, top, fold, tall = 50.0, 12.0, 30.0, 0.24 * width, 2.0 * lane
    start, stop = drawn["adc"]
    folds = [(a, b) for a, b, _ in drawn["groups"]]
    scale = (width - left - right - fold * len(folds)) / max(stop - sum(b - a for a, b in folds), 1e-9)

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
        parts = [f"{'M' if k == 0 else 'L'}{x(t):.1f},{middle - reach * value:.1f}" for k, (t, value) in enumerate(trace)]
        return " ".join(parts)

    middles = [top + 0.72 * tall] + [top + tall + lane * (k + 0.5) for k in range(4)]
    height = top + tall + 4 * lane + 24.0
    peak = max([abs(v) for axis in "xyz" for _, v in drawn[axis]] + [1e-9])
    reach = 0.62 * tall / max([abs(v) for _, v in drawn["rf"]] + [h for _, h, _ in drawn["bars"]] + [1e-9])
    out = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" style="display: block" font-family="inherit" font-size="10">']
    if mark is not None:
        out.append(f'<rect x="{x(mark[0]) - 5:.1f}" y="{top - 4}" width="{x(mark[1]) - x(mark[0]) + 10:.1f}" height="{tall + 4 * lane + 8}" rx="3" '
                   f'fill="{GLOW}" fill-opacity="0.1" stroke="{GLOW}" stroke-opacity="0.5" />')
    for (a, b, name) in drawn["groups"]:
        out.append(f'<rect x="{x(a) + 1:.1f}" y="{top - 2}" width="{fold - 2:.1f}" height="{tall + 3 * lane + 4}" rx="2" fill="{GLOW}" '
                   f'fill-opacity="0.04" stroke="{LINE}" stroke-dasharray="3 2" />'
                   f'<text x="{x(a) + fold / 2:.1f}" y="{top - 8}" fill="{GLOW}" text-anchor="middle" font-size="9">{name}'
                   + ("" if name.endswith("ideal") else f" · {b - a:.0f} ms") + "</text>")
    for name, middle in zip(names, middles):
        out.append(f'<text x="6" y="{middle + 3:.1f}" fill="{DIM}">{name}</text>'
                   f'<line x1="{left}" x2="{width - right}" y1="{middle:.1f}" y2="{middle:.1f}" stroke="{LINE}" />')
    out.append(f'<path d="{path(drawn["rf"], middles[0], reach)}" fill="{GLOW}" fill-opacity="0.22" stroke="{GLOW}" '
               f'stroke-width="1.2" stroke-linejoin="round" />')
    for t, tallness, selective in drawn["bars"]:
        if selective:       # a band-selective pulse (water suppression, editing) is drawn as a Gaussian, as in Tkac 1999 and Mescher 1998
            bell = " ".join(f"L{x(t) + 9.0 * u:.1f},{middles[0] - reach * tallness * np.exp(-4.5 * u * u):.1f}" for u in np.linspace(-1.0, 1.0, 21))
            out.append(f'<path d="M{x(t) - 9:.1f},{middles[0]:.1f} {bell} L{x(t) + 9:.1f},{middles[0]:.1f}" fill="{GLOW}" fill-opacity="0.3" '
                       f'stroke="{GLOW}" stroke-width="1.1" />')
            continue
        out.append(f'<rect x="{x(t) - 3:.1f}" y="{middles[0] - reach * tallness:.1f}" width="6" height="{reach * tallness:.1f}" fill="{GLOW}" '
                   f'fill-opacity="{0.35 if tallness < 0.75 else 0.85}" stroke="{GLOW}" />')
    for k, axis in enumerate("xyz"):
        traced = path(drawn[axis], middles[k + 1], 0.42 * lane / peak)
        spans = [(a, b) for a, b, which in drawn["crushers"] if which == axis]
        out.append(f'<clipPath id="crush{axis}">' + "".join(
            f'<rect x="{x(a):.1f}" y="{middles[k + 1] - lane / 2}" width="{x(b) - x(a):.1f}" height="{lane}" />' for a, b in spans)
            + f'</clipPath><path d="{traced}" fill="{SAND}" fill-opacity="0.08" stroke="{SAND}" stroke-width="1.1" />'
              f'<path d="{traced}" fill="{COPPER}" stroke="{SAND}" stroke-width="1.1" clip-path="url(#crush{axis})" />')
        for t, _ in drawn["marks"]:     # an ideal crusher right after its pulse, the same on every axis
            at, base = x(t) + 5.0, middles[k + 1]
            out.append(f'<path d="M{at:.1f},{base:.1f} l2,-{0.3 * lane:.1f} l5,0 l2,{0.3 * lane:.1f}" fill="{COPPER}" fill-opacity="0.25" '
                       f'stroke="{SAND}" stroke-width="0.9" stroke-dasharray="2 1.5" />')
    # the flips below the RF lane, the intervals between the localization's pulses on top
    last = -1e9
    for t, text in drawn["pulses"]:
        if text and x(t) - last > 22.0:
            out.append(f'<text x="{x(t):.1f}" y="{middles[0] + 0.27 * tall:.1f}" fill="{INK}" text-anchor="middle" font-size="10">{text}</text>')
            last = x(t)
    centres = [t for t, _ in drawn["pulses"] if not folded(t)] + [start]
    for a, b in zip(centres[:-1], centres[1:]):
        out.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="14" y2="14" stroke="{DIM}" />'
                   + "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="10" y2="18" stroke="{DIM}" />' for t in (a, b))
                   + (f'<text x="{0.5 * (x(a) + x(b)):.1f}" y="10" fill="{INK}" text-anchor="middle" font-size="9">{b - a:.1f}</text>'
                      if x(b) - x(a) > 30.0 else ""))
    out.append(f'<text x="{0.5 * (x(start) + x(stop)):.1f}" y="17" fill="{GOOD}" text-anchor="middle">DAQ</text>')
    wave = [(start + (stop - start) * u, np.exp(-3.5 * u) * np.cos(2.0 * np.pi * 11.0 * u)) for u in np.linspace(0.0, 1.0, 160)]
    out.append(f'<path d="{path(wave, middles[4], 0.42 * lane)}" fill="none" stroke="{GOOD}" stroke-width="1.1" />')
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000) if (stop - sum(b - a for a, b in folds)) / s <= 10)
    begin = folds[-1][1] if folds else 0.0      # the localization's clock starts after the preludes
    for t in np.arange(0.0, stop - begin + 1e-9, step):
        out.append(f'<line x1="{x(begin + t):.1f}" x2="{x(begin + t):.1f}" y1="{height - 22}" y2="{height - 18}" stroke="{DIM}" />'
                   f'<text x="{x(begin + t):.1f}" y="{height - 7}" fill="{DIM}" text-anchor="middle" font-size="9">{t:g}</text>')
    out.append(f'<text x="6" y="{height - 7}" fill="{DIM}" font-size="9">ms</text></svg>')
    return "".join(out)


#*****************#
#   choose file   #
#*****************#
def choose_file(prompt):
    """The operating system's own file dialog, as a process of its own, and the chosen path or None. The console
    runs on the operator's machine, so its server can ask; the embedded window's dialog (pywebview through
    NiceGUI) came back empty at once on macOS here and is not relied on. macOS: AppleScript's choose file;
    elsewhere: Tk's, in a Python of its own, because Tk wants a main thread."""
    if sys.platform == "darwin":
        command = ["osascript", "-e", f'POSIX path of (choose file with prompt "{prompt}")']
    else:
        command = [sys.executable, "-c", "import tkinter, tkinter.filedialog as d; r = tkinter.Tk(); r.withdraw(); "
                                         f"r.attributes('-topmost', True); print(d.askopenfilename(title={prompt!r}))"]
    chosen = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    return pathlib.Path(chosen) if chosen else None


# the spectrum display's chart, the console's and the designer's
SPECTRUM = {
    "backgroundColor": "transparent", "animation": False,
    "title": {"text": "no scan yet", "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
    "grid": {"left": 12, "right": 12, "top": 28, "bottom": 44},
    "xAxis": {"type": "value", "inverse": True, "min": -0.5, "max": 6.5, "name": "ppm",
              "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}},
    "yAxis": {"type": "value", "show": False, "scale": True},
    # the wheel zooms and a drag pans along ppm; the slider below shows where one is in -0.5 to 6.5
    "dataZoom": [{"type": "inside", "xAxisIndex": 0, "startValue": 0.2, "endValue": 4.2, "filterMode": "none"},
                 {"type": "slider", "xAxisIndex": 0, "startValue": 0.2, "endValue": 4.2, "height": 14, "bottom": 2, "filterMode": "none",
                  "borderColor": LINE, "fillerColor": "rgba(226,165,95,0.15)", "textStyle": {"color": DIM}}],
    "series": [{"type": "line", "showSymbol": False, "data": [], "lineStyle": {"color": SAND, "width": 1.3},
                "areaStyle": {"color": COPPER, "opacity": 0.12}},
               {"type": "line", "showSymbol": False, "data": [], "lineStyle": {"color": DIM, "width": 0.7}, "silent": True}],
}


#**********#
#   page   #
#**********#
@ui.page("/")
async def console(client: Client):
    """One operator's console; every browser tab is a session of its own. The page answers at once with a
    loading screen, and the room (the subject's maps, the first planes and the card's outlook take seconds) is
    made behind it."""
    ui.add_head_html(f"<style>{STYLE}</style>")
    ui.dark_mode(True)
    ui.colors(primary=COPPER, accent=SAND, positive=GOOD, negative=BAD)
    with ui.element("div").style(f"position: fixed; inset: 0; z-index: 9999; background: {GROUND}; display: flex; flex-direction: column; "
                                 "align-items: center; justify-content: center; gap: 18px") as loading:
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" style="height: 150px" src="{MARK_LARGE}">', sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 22px; letter-spacing: .14em")
        ui.spinner("bars", size="34px", color=SAND)
    await client.connected()
    session = await run.io_bound(Session)
    views, shown, hung, letters, crosses, texts, tools, pickers, places, edges, chosen = {}, {}, {}, {}, {}, {}, {}, {}, [], [], {"exam": None, "box": None, "stop": False, "shifts": True, "snr": None, "editing": None, "flipped": False, "ready": False, "dragging": False, "looking": False}

    def room():
        """Rebuild the 3D view: the three planes hung where they cut the subject (paint fills them,
        with the overlay they carry), the receive loops, the voxel's box."""
        drawn = session.scene()
        view.clear()
        hung.clear()
        shown.clear()
        with view:
            for name, (_, _, _, _, _, corners) in session.planes().items():
                hung[name] = ui.scene.texture(BLANK, (np.asarray(corners) * MM).tolist())
            for centre, normal, radius in drawn["loops"]:
                n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
                a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
                a = a / np.linalg.norm(a)
                ui.scene.ring(0.92 * radius * MM, radius * MM, 40).material(COPPER, 0.85).move(
                    *(np.asarray(centre) * MM)).rotate_R(np.stack([a, np.cross(n, a), n], axis=1).tolist())
            centre, size = drawn["voxel"]
            chosen["box"] = ui.scene.box(*(edge * MM for edge in size), wireframe=True).material(GLOW).move(
                *(np.asarray(centre) * MM))

    def paint():
        """Redraw the planes (the slice on view, the voxel's outline where the slice cuts it, the
        other planes' cross, a spectroscopic image's grid) and the 3D box: what a drag needs."""
        card = session.protocol
        counts = (int(card.matrix), int(card.matrix_y), int(card.slices)) if card.sequence == MRSI else (1, 1, 1)
        for name, (image, (left, top, width, height), cut, (column, row), shifted, corners) in session.planes().items():
            stamp = hash(image.tobytes())
            if shown.get(name) != stamp:                            # a moved voxel changes the outlines, not the picture
                shown[name] = stamp
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                views[name].set_source("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
                # the same picture in space, see-through where there is nothing
                colour = np.asarray(Image.fromarray(image).convert("RGBA")).copy()
                colour[..., 3] = np.where(colour[..., :3].max(axis=2) > 10, 235, 0)
                buffer = io.BytesIO()
                Image.fromarray(colour).save(buffer, format="PNG")
                hung[name].set_url("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
                hung[name].set_coordinates((np.asarray(corners) * MM).tolist())
            side = image.shape[0]
            across, down = {"tra": (counts[0], counts[1]), "sag": (counts[1], counts[2]), "cor": (counts[0], counts[2])}[name]
            grid = "".join(f'<line x1="{left + width * n / across:.1f}" x2="{left + width * n / across:.1f}" y1="{top:.1f}" '
                           f'y2="{top + height:.1f}" />' for n in range(1, across)) + "".join(
                f'<line y1="{top + height * n / down:.1f}" y2="{top + height * n / down:.1f}" x1="{left:.1f}" x2="{left + width:.1f}" />'
                for n in range(1, down))
            pen = side / 115.0                                      # strokes as wide on a fine grid as on the 2 mm one
            views[name].content = (
                f'<g stroke="{SAND}" stroke-width="{0.3 * pen:.2f}" opacity="0.45"><line x1="{column}" x2="{column}" y1="0" y2="{side}" />'
                f'<line y1="{row}" y2="{row}" x1="0" x2="{side}" /></g>'
                f'<g fill="none" stroke="{GLOW}" stroke-width="{0.35 * pen:.2f}" opacity="0.7">{grid}</g>'
                + "".join(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
                          f'stroke="{SHIFT_COLOURS[which]}" stroke-width="{0.5 * pen:.2f}" opacity="{0.9 if cut else 0.4}" />'
                          for which, (x, y, w, h) in shifted.items() if chosen["shifts"] or which == "shim")
                + f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" fill="none" stroke="{GLOW}" '
                + (f'stroke-width="{0.8 * pen:.2f}" />' if cut else
                   f'stroke-width="{0.5 * pen:.2f}" stroke-dasharray="{2 * pen:.1f} {1.5 * pen:.1f}" opacity="0.55" />'))
            axis, letter = EDGES[name][4:]
            texts[name].text = f"{name.upper()}  {letter} {session.view_mm[axis]:+.0f} mm"
        chosen["box"].move(*(np.asarray(session.position_mm) * MM))
        for name, labels in letters.items():       # a flipped view mirrors the pictures left to right, and the letters with them
            mirrored = chosen["flipped"] and name in ("tra", "cor")
            views[name].style("transform: scaleX(-1)" if mirrored else "transform: none")
            top, bottom, left, right = EDGES[name][:4]
            labels[2].text, labels[3].text = (right, left) if mirrored else (left, right)
            crosses[name].content = (
                f'<svg width="54" height="54" viewBox="0 0 54 54" font-size="9" font-family="inherit"><g stroke="{SAND}" fill="{SAND}">'
                f'<line x1="20" y1="34" x2="46" y2="34" /><line x1="20" y1="34" x2="20" y2="8" />'
                f'<path d="M46,34 l-5,-3 v6 z" /><path d="M20,8 l-3,5 h6 z" /></g>'
                f'<text x="48" y="46" fill="{SAND}" text-anchor="end">{labels[3].text}</text><text x="6" y="12" fill="{SAND}">{top}</text></svg>')
        # the overlays there are for this room: a B0 or B1+ map only where one is on (maps), a lipid layer where the subject has one
        there = [name for name in OVERLAYS if {"B0 field": session.field != "uniform", "B1+": session.transmit is not None,
                                               "lipid layer": getattr(session.body(), "fat", None) is not None}.get(name, True)]
        if list(overlays.options) != there:
            session.overlay = session.overlay if session.overlay in there else "anatomy"
            overlays.set_options(there, value=session.overlay)
        wash = session.wash()
        texts["window"].text = ("" if wash is None or not wash[3] else f"blue {wash[1]:+.3g} · red {wash[2]:+.3g}"
                                + (" Hz" if session.overlay == "B0 field" else "")) if session.overlay != "anatomy" else ""
        if session.overlay in ("B0 field", "B1+") and wash is None:
            texts["window"].text = "no such map on: choose one in maps"
        for axis, field in enumerate(places):
            field.value = round(session.position_mm[axis], 1)

    def draw():
        """Redraw everything the card and the room decide: the planes, the diagram, the outlook."""
        if not chosen["ready"]:         # a bound control speaks its first value while the page is still being built
            return
        paint()
        card = session.protocol
        try:
            diagram.content = lanes(session.diagram())
            look = session.outlook()
        except Exception as error:  # a card that is no design yet (a TE its pulses do not fit): say why, keep the rest
            texts["verdict"].text = str(error)[:160]
            texts["verdict"].style(f"color: {BAD}")
            return
        minutes, seconds = divmod(int(round(look["scan_time_s"])), 60)
        texts["name"].text = card.sequence if chosen["editing"] is None else f"editing {chosen['editing'].name}  →  {card.sequence}"
        accept.text = "Accept" if chosen["editing"] is None else "Accept changes"
        discard.set_visibility(chosen["editing"] is not None)
        texts["time"].text = texts["time2"].text = f"{minutes:02d}:{seconds:02d}"
        texts["timing"].text = f"{card.tr_ms:g} / {card.te_ms:g} ms"
        texts["volume"].text = f"{np.prod(session.size_mm) / 1000.0:.1f} mL"
        # against the card as it was taken from the browser: the voxel's volume and the root of the averages alone
        signal = np.prod(session.size_mm) * np.sqrt(1 if card.sequence == MRSI else card.averages)
        chosen["snr"] = signal if chosen["snr"] is None else chosen["snr"]
        texts["snr"].text = f"{100.0 * signal / chosen['snr']:.0f} %  (volume, averages)"
        texts["displaced"].text = "\n".join(
            f"{which} " + " ".join(f"{v:+.1f}" for v in moved) + " mm" for which, moved in look["displaced"].items()) if any(
            any(moved) for moved in look["displaced"].values()) else "none, the localization is ideal"
        texts["linewidth"].text = ("uniform field, none" if look["linewidth_hz"] is None else
                                   f"{look['linewidth_hz']:.1f} Hz on a 1 Hz line, {session.shim} shim")
        texts["lipid"].text = ("this anatomy has no lipid layer" if look["lipid_mm"] is None else
                               f"{look['lipid_mm'][0]:.0f} mm from the voxel, {look['lipid_mm'][1]:.0f} mm from where lipid is selected")
        texts["lipid"].style(f"color: {BAD if look['lipid_mm'] is not None and min(look['lipid_mm']) < 1.0 else INK}")
        texts["tissue"].text = f"GM {look['gm']:.2f}  WM {look['wm']:.2f}  CSF {look['csf']:.2f}"
        texts["encoding"].text = look["encoding"]["said"] if look["encoding"] else "a single voxel has none"
        texts["resolution"].text = f"{card.bandwidth_hz / card.npoints:.2f} Hz / point"
        texts["readout"].text = f"{1000.0 * card.npoints / card.bandwidth_hz:.0f} ms"
        texts["limits"].text = "B1 {:g} uT · G {:g} mT/m · slew {:g} T/m/s".format(*look["limits"])
        texts["verdict"].text = "plays on this scanner" if not look["violations"] else "; ".join(look["violations"])
        texts["verdict"].style(f"color: {GOOD if not look['violations'] else BAD}")
        records.clear()
        with records:       # the draw of the population that is scanned, at the planned voxel: what the exams will scan
            where = session.view_mm if chosen["looking"] else session.position_mm
            panel.text = "Where you look (shift)" if chosen["looking"] else "This voxel"
            for group, values in session.place_of(where).items():
                if isinstance(values, dict):
                    ui.label(group).style(f"margin-top: 4px; color: {DIM}; font-size: 11px")
                    with ui.element("div").classes("info").style("grid-template-columns: auto auto auto auto; column-gap: 8px"):
                        for name, value in values.items():
                            ui.label(name).style(f"color: {DIM}")
                            ui.label(f"{value:.1f}" if value < 100.0 else f"{value:.0f}").classes("value")
                else:
                    with ui.row().classes("w-full").style("justify-content: space-between; margin-top: 4px"):
                        ui.label(group).style(f"color: {DIM}; font-size: 11px")
                        ui.label(f"{values:.1f}").classes("value")
        for name, tool in tools.items():
            on = {"ovs": card.ovs, "shifts": chosen["shifts"], "flip": chosen["flipped"], "real": card.real or card.sequence == "HERCULES"}.get(
                name, card.water_suppression != "off")
            tool.classes(add="on" if on else "", remove="" if on else "on")
        tools["water"].text = "water " + card.water_suppression
        tools["water"].props(f"icon={'format_color_reset' if card.water_suppression == 'off' else 'water_drop'}")

    def listing():
        """Redraw the exam cards."""
        rows.clear()
        with rows:
            for exam in session.queue:
                minutes, seconds = divmod(int(round(exam.card.scan_time_s)), 60)
                with ui.element("div").classes("exam" + (" chosen" if exam is chosen["exam"] else "")).on(
                        "click", lambda _, exam=exam: choose(exam)).mark(f"card {exam.name}"):
                    state = ui.icon({"planned": "edit" if exam is chosen["editing"] else "radio_button_unchecked", "running": "autorenew",
                                     "done": "check_circle", "failed": "error"}[exam.status], size="16px").style(
                        f"color: {dict(planned=DIM, running=SAND, done=GOOD, failed=BAD)[exam.status]}")
                    if exam.problem:
                        state.tooltip(exam.problem[:200])
                    ui.label(exam.name).style("flex: 1")
                    ui.label(f"{minutes:02d}:{seconds:02d}").classes("value").style(f"color: {DIM}")
                    ui.icon("content_copy", size="14px").style(f"color: {DIM}").tooltip("duplicate as a new planned card").on(
                        "click.stop", lambda _, exam=exam: (choose(exam), chosen.__setitem__("editing", None), plan()))
                    if exam.status == "planned":
                        ui.icon("close", size="14px").style(f"color: {DIM}").tooltip("remove").on("click.stop", lambda _, exam=exam: (
                            session.queue.remove(exam), chosen.__setitem__("editing", None if chosen["editing"] is exam else chosen["editing"]),
                            chosen.__setitem__("exam", None if chosen["exam"] is exam else chosen["exam"]), listing(), draw()))

    def frame(exam, index):
        """The Live display at one step of the build-up: the average so far in the combination or condition
        chosen, and, grey behind it, the transient that just came in."""
        name = looks.value if looks.value in exam.builds else exam.shown
        count, real = exam.builds[name][min(index, len(exam.builds[name]) - 1)] if exam.builds else exam.frames[index][:2]
        single = exam.frames[index][2] if name == exam.shown and len(exam.frames) > 1 else []
        spectrum.options["series"][0]["data"] = [[float(p), float(s)] for p, s in zip(exam.ppm, real)]
        spectrum.options["series"][1]["data"] = [[float(p), float(s)] for p, s in zip(exam.ppm, single)]
        if index in (0, -1) or index == len(exam.frames) - 1:        # the window the exam opens on; the operator's zoom stays in between
            for zoom in spectrum.options["dataZoom"]:
                zoom["startValue"], zoom["endValue"] = exam.window
        spectrum.options["title"]["text"] = f"{exam.name}   ·   transient {count} of {exam.card.averages}" + (
            f"   ·   SNR {exam.snr[0]:.0f}, of one transient {exam.snr[1]:.0f}" if exam.snr and index in (-1, len(exam.frames) - 1) else "")
        spectrum.update()
        progress.value = count / max(exam.card.averages, 1)
        texts["count"].text = f"{count} / {exam.card.averages}"

    def choose(exam):
        """Select an exam: its voxel returns to the planes, its card to the tabs, its spectrum to the
        display. A planned exam is then being edited: Accept puts the card back in its place."""
        chosen["exam"], chosen["editing"] = exam, exam if exam.status == "planned" else None
        looks.set_options(list(exam.builds), value=exam.shown if exam.builds else None)
        for field in dataclasses.fields(exam.card):                       # in place: the tabs' fields hold this card
            setattr(session.protocol, field.name, getattr(exam.card, field.name))
        session.position_mm, session.view_mm, session.size_mm = list(exam.position_mm), list(exam.position_mm), list(exam.size_mm)
        for axis, field in enumerate(edges):
            field.value = exam.size_mm[axis]
        if exam.frames:
            frame(exam, -1)
        listing()
        room()
        draw()

    def plan():
        chosen["exam"] = session.plan(instead_of=chosen["editing"])
        chosen["editing"] = None
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
            except Exception as error:  # a card the engine cannot play: the queue goes on, the card says why
                exam.status, exam.problem = "failed", str(error)
                progress.props(remove="indeterminate")
                listing()
                continue
            texts["took"].text = f"{exam.name}: simulated in {time.perf_counter() - began:.1f} s, a scanner takes {exam.card.scan_time_s:.0f} s"
            progress.props(remove="indeterminate")
            looks.set_options(list(exam.builds), value=exam.shown if exam.builds else None)
            for index in range(len(exam.frames)):
                frame(exam, index)
                await asyncio.sleep(0.03)
            listing()
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
            reveal.on("click", lambda: subprocess.run(["open", "-R", str(path)] if sys.platform == "darwin" else
                                                      ["xdg-open", str(path.parent)], check=False))

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 640px"):
            ui.label(f"Export · {exam.name}").classes("label")
            folder = ui.input("folder", value=str(pathlib.Path.home() / "MRSandbox" / "exports")).props("dense outlined dark").classes("w-full")
            ui.select(["NIfTI-MRS, the transients apart, with the simulation's truth in the header"], value=(
                "NIfTI-MRS, the transients apart, with the simulation's truth in the header"), label="format").props(
                "dense outlined dark").classes("w-full").tooltip("more formats are to come")
            done = ui.label("").classes("value").style("white-space: normal; word-break: break-all")
            with ui.row().classes("w-full"):
                reveal = ui.button("Show in its folder", icon="folder_open").props("outline dense no-caps")
                reveal.set_visibility(False)
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense no-caps")
                ui.button("Export", icon="save_alt", on_click=write).mark("write the export").props("unelevated dense no-caps")
        dialog.open()

    def gear(panel=None):
        """The settings of the simulation, a line per parameter: as its source gives it (a value,
        a range or a spread over the population, from which every exam draws), or the operator's
        own: set to a number, varied between two, or read from a map. A panel's gear opens its
        own branches (GEARS), the top bar's the whole tree; the search narrows either."""
        priors = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves(own=False)
                  if panel is None or path.split(".")[0] in GEARS[panel]}

        def line(holder, path):
            """One parameter's line for the way it is set now."""
            kind, numbers = priors[path]
            own = session.overrides.get(path)
            way = WAYS[0] if own is None else WAYS[3] if isinstance(own, str) else WAYS[2] if isinstance(own, tuple) else WAYS[1]
            middle = numbers[0] if kind in ("fixed", "normal") else 0.5 * (numbers[0] + numbers[1]) if kind == "range" else 0.0
            spread = (numbers if kind == "range" else (numbers[0] - 2.0 * numbers[1], numbers[0] + 2.0 * numbers[1]) if kind == "normal"
                      else (0.9 * middle, 1.1 * middle))

            def switch(event):
                session.overrides.pop(path, None)
                if event.value == WAYS[1]:
                    session.overrides[path] = float(middle)
                elif event.value == WAYS[2]:
                    session.overrides[path] = (float(min(spread)), float(max(spread)))
                line(holder, path)
                draw()

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
                    draw()
                return handle

            async def taken(event):
                session.overrides[path] = str(session.onto(await keep(event)))     # onto the subject's grid, by the scanner's coordinates
                line(holder, path)
                draw()

            holder.clear()
            with holder:
                ui.label(path.split(".", 1)[-1]).style(f"width: 190px; color: {SAND if own is not None else INK}")
                ui.select(WAYS if kind != "other" else [WAYS[0], WAYS[3]], value=way, on_change=switch).props(
                    "dense outlined dark options-dense").style("width: 190px")
                if way == WAYS[0]:
                    ui.label({"fixed": "{:g}", "range": "{:g} to {:g}", "normal": "{:.4g} ± {:.2g}"}[kind].format(*numbers)
                             if kind != "other" else numbers).classes("value").style(f"color: {DIM}")
                elif way == WAYS[3]:
                    if own is not None:
                        ui.label(pathlib.Path(own).name).classes("value")
                    ui.upload(label="a NIfTI map in this parameter's units", auto_upload=True, on_upload=taken).props(
                        "flat dense dark accept=.nii,.gz").style("width: 300px")
                else:
                    for index, number in enumerate(own if isinstance(own, tuple) else (own,)):
                        ui.number(value=number, format="%.6g", on_change=put(index)).props("dense outlined dark").style("width: 120px")
                        if index == 0 and isinstance(own, tuple):
                            ui.label("and").classes("label")

        def build(wanted=""):
            """The groups, each filled when it first opens; a search fills and opens what it finds."""
            body.clear()
            groups = {}
            for path in priors:
                if wanted.lower() in path.lower():
                    groups.setdefault(path.split(".")[0], []).append(path)
            with body:
                for group, paths in groups.items():
                    def fill(event, group=group, paths=paths, done=[]):
                        if event.value and not done:
                            done.append(True)
                            with event.sender:
                                if any(priors[path][0] != "other" for path in paths) and len(paths) > 1:
                                    with ui.row().classes("w-full items-center no-wrap").style(f"gap: 10px; border-bottom: 1px solid {LINE}"):
                                        ui.label("the whole group").style(f"width: 190px; color: {GLOW}")
                                        way = ui.select(["as its source gives it", "set to", "scale by"], value="as its source gives it").props(
                                            "dense outlined dark options-dense").style("width: 190px")
                                        number = ui.number(value=1.0, format="%.6g").props("dense outlined dark").style("width: 120px")
                                        ui.button("Apply", on_click=lambda: (session.whole(group, way.value, number.value), build(wanted), draw())
                                                  ).props("outline dense no-caps")
                                for path in paths:
                                    line(ui.row().classes("w-full items-center no-wrap").style("gap: 10px; min-height: 40px"), path)

                    changed_here = sum(path in session.overrides for path in paths)
                    fold = ui.expansion(f"{GROUPS.get(group, group)}   ·   {len(paths)}" + (f"   ·   {changed_here} your own" if changed_here else ""),
                                        on_value_change=fill).classes("w-full").props("dense dark")
                    if wanted:
                        fold.value = True

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 860px; max-width: 1000px"):
            with ui.row().classes("w-full items-center"):
                ui.label(f"Settings · {dict(subject='anatomy or phantom').get(panel, panel) if panel else 'everything'}").classes("label")
                ui.space()
                ui.input(placeholder="search, e.g. NAA or t2", on_change=lambda e: build(e.value or "")).props(
                    "dense outlined dark clearable debounce=300").style("width: 260px")
                ui.button("Reset all", icon="restart_alt", on_click=lambda: (session.overrides.clear(), build(), draw())).props(
                    "flat dense no-caps")
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
            ui.label("Every exam draws its values from these lines. Left as they are, they are the population's, each from its source. "
                     "Set one to a number to decide it yourself, or let it vary between two.").classes("label").style(
                "text-transform: none; letter-spacing: 0")
            with ui.scroll_area().style("height: 580px; width: 100%"):
                body = ui.column().classes("w-full").style("gap: 0")
            build()
        dialog.open()

    def designer():
        """The sequence designer: the design as a strip of blocks in the order they play, over its
        diagram; a block is chosen on the strip, set in the inspector beside the palette, and new
        blocks go in after it. The diagram marks the chosen block, and the design's spectrum at
        the planned voxel is one press away."""
        state = {"at": 0}

        def refresh():
            """Redraw the diagram with the chosen block marked, the strip, and the inspector."""
            strip.clear()
            with strip:
                for index, (kind, settings) in enumerate(session.design):
                    brief = next((f"{value:g}" + unit for key, unit in (("flip_deg", "°"), ("ms", " ms"), ("cycles_per_mm", " /mm"), ("order", ""))
                                  if (value := settings.get(key)) is not None), settings.get("scheme", ""))
                    chip = ui.button(f"{index + 1} · {kind} {brief}", icon=BLOCK_ICONS[kind], on_click=lambda _, i=index: (
                        state.__setitem__("at", i), refresh())).props("flat dense no-caps stack draggable").classes(
                        "chip" + (" on" if index == state["at"] else ""))
                    chip.on("dragstart", lambda _, i=index: state.__setitem__("dragged", i))
                    chip.on("dragover.prevent", lambda: None)
                    chip.on("dragenter", lambda _, chip=chip: chip.classes(add="over"))      # the bar where the block will land
                    chip.on("dragleave", lambda _, chip=chip: chip.classes(remove="over"))
                    chip.on("drop", lambda _, i=index: drop(i))
                    with chip, ui.context_menu().props("dark"):
                        ui.menu_item("Copy", lambda i=index: state.__setitem__("copied", (session.design[i][0], dict(session.design[i][1]))))
                        ui.menu_item("Paste after this", lambda i=index: "copied" in state and (
                            state.__setitem__("at", i), place(state["copied"][0], dict(state["copied"][1]))))
                        ui.menu_item("Remove", lambda i=index: (state.__setitem__("at", i), remove()))
            inspector.clear()
            with inspector:
                if session.design:
                    kind, settings = session.design[state["at"]]
                    with ui.row().classes("w-full items-center"):
                        ui.icon(BLOCK_ICONS[kind], size="20px").style(f"color: {GLOW}")
                        ui.label(f"Block {state['at'] + 1} · {kind}").style(f"color: {SAND}")
                        ui.space()
                        if kind in ("hard pulse", "shaped pulse"):
                            ui.button("make it " + ("ideal" if kind == "shaped pulse" else "shaped"), icon="swap_horiz", on_click=lambda: (
                                reshape(state["at"], kind == "hard pulse"), refresh())).props("outline dense no-caps")
                        for icon, tip, act in (("arrow_back", "earlier", lambda: move(-1)), ("arrow_forward", "later", lambda: move(+1)),
                                               ("content_copy", "copy", lambda: state.__setitem__("copied", (kind, dict(settings)))),
                                               ("content_paste", "paste the copied block after this one", lambda: "copied" in state and place(
                                                   state["copied"][0], dict(state["copied"][1]))),
                                               ("delete", "remove", remove)):
                            ui.button(icon=icon, on_click=act).props("flat dense round").tooltip(tip)
                    ui.label(BLOCK_NOTES[kind]).classes("label").style("text-transform: none; letter-spacing: 0")
                    with ui.row().classes("w-full").style("gap: 8px"):
                        for key, value in settings.items():
                            def put(event, key=key, number=not isinstance(value, str)):
                                if event.value is not None:
                                    settings[key] = float(event.value) if number else event.value
                                    redraw()
                            if key in ("what", "event"):       # an event taken over from the card: said, not set
                                if key == "what":
                                    ui.label(value).classes("value")
                            elif key in CHOICES:
                                ui.select(CHOICES[key], value=value, label=CAPTIONS[key], on_change=put).props("dense outlined dark").style("width: 170px")
                            elif isinstance(value, str):
                                ui.input(CAPTIONS[key], value=value, on_change=put).props("dense outlined dark debounce=400").style("width: 100%")
                                ui.upload(label="or a text file of amplitudes", auto_upload=True, on_upload=envelope).props(
                                    "flat dense dark accept=.txt,.csv").style("width: 280px")
                            else:
                                ui.number(CAPTIONS[key], value=value, format="%.6g", on_change=put).props("dense outlined dark").style("width: 170px")
            redraw()

        def redraw():
            try:
                drawn = session.diagram(session.own())             # the design itself, whatever the card plays
                counts = np.cumsum([0] + [len(BLOCKS[kind][1](settings)) for kind, settings in session.design])
                spans = drawn["spans"][counts[state["at"]]:counts[state["at"] + 1]] if session.design else []
                preview.content = lanes(drawn, width=820.0, lane=44.0, mark=(spans[0][0], spans[-1][1]) if spans else None)
                problem.text = f"{len(session.design)} blocks · the readout opens at {drawn['adc'][0]:.1f} ms"
                problem.style(f"color: {DIM}")
            except Exception as error:  # a half-typed block is not a design yet: say why and keep the last diagram
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")

        def parts():
            """The design's sub-acquisitions and how they combine: one is a plain sequence; several are
            interleaved over the transients as an edited scan is, and the weights say what is shown."""
            def another():
                name = next(chr(65 + k) for k in range(26) if chr(65 + k) not in session.designs)
                session.designs[name] = [(kind, dict(settings)) for kind, settings in session.design]
                session.condition = name
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
                        ui.button(name, on_click=lambda _, name=name: (setattr(session, "condition", name), state.__setitem__("at", 0), parts(),
                                                                       refresh())).props("flat dense no-caps").classes(
                            "tool" + (" on" if name == session.condition else "")).style("width: 44px; height: 30px")
                    ui.button("another, a copy of this one", icon="add", on_click=another).props("flat dense no-caps")
                    if session.design and "phase_deg" in session.design[state["at"]][1]:
                        for steps in (2, 4):
                            ui.button(f"phase cycle the chosen pulse, {steps} steps", icon="cached", on_click=lambda _, steps=steps: (
                                session.phase_cycle(state["at"], steps), parts(), refresh())).props("flat dense no-caps").tooltip(
                                "the chosen pulse steps its phase through the circle over the transients and the receiver follows (an excitation "
                                "takes it along, a refocusing pulse is EXORCYCLE), so the wanted pathway adds and the rest cancels. Coherence "
                                "filters do this ideally: take them out of the design to see the cycle work")
                    if len(session.designs) > 1:
                        ui.button("remove this one", icon="remove", on_click=without).props("flat dense no-caps")
                        ui.space()
                        if len(session.designs) == 2:
                            ui.button("difference", on_click=lambda: scheme((("difference", (1.0, -1.0)), ("sum", (1.0, 1.0))))).props(
                                "outline dense no-caps")
                        if len(session.designs) == 4:
                            ui.button("Hadamard", on_click=lambda: scheme((("sum", (1.0, 1.0, 1.0, 1.0)), ("A+B-C-D", (1.0, 1.0, -1.0, -1.0)), (
                                "A-B+C-D", (1.0, -1.0, 1.0, -1.0)), ("A-B-C+D", (1.0, -1.0, -1.0, 1.0))))).props("outline dense no-caps")
                        ui.button("a combination", icon="add", on_click=lambda: (session.weights.__setitem__(
                            f"combination {len(session.weights) + 1}", {name: 1.0 for name in session.designs}), parts())).props("flat dense no-caps")
                for label, mix in (session.weights.items() if len(session.designs) > 1 else ()):
                    with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                        ui.label(label).style(f"width: 160px; color: {SAND}")
                        for name in session.designs:
                            if isinstance(mix.get(name, 0.0), complex):     # a receiver phase of a cycle: said, not typed
                                ui.label(f"{name}  {np.degrees(np.angle(mix[name])):+.0f}°").classes("value").style("width: 84px")
                                continue
                            ui.number(name, value=mix.get(name, 0.0), step=1.0, format="%.3g", on_change=lambda e, mix=mix, name=name: (
                                e.value is not None and mix.__setitem__(name, float(e.value)))).props("dense outlined dark").style("width: 84px")
                        ui.button(icon="close", on_click=lambda _, label=label: (session.weights.pop(label), parts())).props("flat dense round size=sm")
                if len(session.designs) > 1:
                    ui.label("The sub-acquisitions take turns over the transients. The first combination is what Live shows; Review and the "
                             "show select have them all, and every sub-acquisition alone.").classes("label").style(
                        "text-transform: none; letter-spacing: 0")

        def drop(onto):
            """A dragged block takes the place of the one it is dropped on."""
            dragged = state.pop("dragged", None)
            if dragged is not None and dragged != onto:
                session.design.insert(onto, session.design.pop(dragged))
                state["at"] = onto
                refresh()

        def reshape(index, shaped):
            """An ideal pulse becomes a shaped one of the same flip and phase, or back."""
            kind, settings = session.design[index]
            if kind in ("hard pulse", "shaped pulse") and (kind == "shaped pulse") != shaped:
                other = "shaped pulse" if shaped else "hard pulse"
                session.design[index] = (other, {**BLOCKS[other][0], "flip_deg": settings["flip_deg"], "phase_deg": settings["phase_deg"]})

        def place(kind, settings):
            """A new block goes in after the chosen one and becomes the chosen one."""
            state["at"] = min(state["at"] + 1, len(session.design))
            session.design.insert(state["at"], (kind, settings))
            refresh()

        def move(step):
            here, there = state["at"], state["at"] + step
            if 0 <= there < len(session.design):
                session.design[here], session.design[there] = session.design[there], session.design[here]
                state["at"] = there
                refresh()

        def remove():
            session.design.pop(state["at"])
            state["at"] = max(min(state["at"], len(session.design) - 1), 0)
            refresh()

        async def envelope(event):
            session.design[state["at"]][1]["envelope"] = ", ".join((await event.file.text()).replace(",", " ").split())
            refresh()

        async def look():
            """What the design does to a spectrum (Session.showcase): a fixed subject in normal tissue,
            a uniform field, no noise, no water, nothing but the sequence."""
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

        def key(event):
            """Delete removes the chosen block; ctrl or cmd with C copies it, with V pastes after it."""
            if not event.action.keydown or not session.design:
                return
            if event.key in ("Delete", "Backspace"):
                remove()
            elif (event.modifiers.ctrl or event.modifiers.meta) and event.key == "c":
                state["copied"] = (session.design[state["at"]][0], dict(session.design[state["at"]][1]))
            elif (event.modifiers.ctrl or event.modifiers.meta) and event.key == "v" and "copied" in state:
                place(state["copied"][0], dict(state["copied"][1]))

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 1300px; max-width: 1500px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Sequence designer").style(f"color: {SAND}; font-size: 15px")
                ui.select(list(TEMPLATES), label="start from", on_change=lambda e: e.value and (
                    setattr(session, "design", [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in TEMPLATES[e.value]]),
                    state.__setitem__("at", 0), refresh())).props("dense outlined dark").style("width: 300px")
                problem = ui.label("").classes("value")
                ui.space()
                ui.button("all pulses ideal", icon="bolt", on_click=lambda: ([reshape(i, False) for i in range(len(session.design))], refresh())
                          ).props("outline dense no-caps")
                ui.button("all pulses shaped", icon="waves", on_click=lambda: ([reshape(i, True) for i in range(len(session.design))], refresh())
                          ).props("outline dense no-caps")
                ui.button("Spectrum of this design", icon="insights", on_click=look).props("outline dense no-caps")
                ui.button("Close", on_click=lambda: (setattr(keys, "active", False), dialog.close())).props("flat dense no-caps").tooltip(
                    "the design is kept; the card plays what it played")
                ui.button("Use this design", icon="check", on_click=lambda: (
                    setattr(keys, "active", False), setattr(session.protocol, "sequence", "Custom"), dialog.close(), draw())).props(
                    "unelevated dense no-caps").tooltip("the card plays this design (Custom); water suppression and OVS stay the card's switches")
                keys = ui.keyboard(on_key=key, ignore=["input", "select", "textarea"])
            with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                preview = ui.html("", sanitize=False).style("flex: 3; min-width: 0")
                with ui.element("div").style("flex: 2; min-width: 0; position: relative"):
                    outcome = ui.echart({**copy.deepcopy(SPECTRUM), "title": {**SPECTRUM["title"], "text": "the spectrum this design gives"}}).style(
                        "height: 340px").classes("w-full")
                    busy = ui.spinner(size="56px", color=SAND).style("position: absolute; top: 42%; left: 46%")
                    busy.set_visibility(False)
            several = ui.column().classes("w-full").style("gap: 4px")
            ui.label("The design, in the order it plays · press a block to choose it · drag one onto another to move it there").classes("label")
            strip = ui.row().classes("w-full").style("gap: 4px")
            with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch; flex: 1"):
                with ui.column().classes("pane").style("width: 430px"):
                    ui.label("Add after the chosen block").classes("label")
                    for kind in (k for k in BLOCKS if k != "as the card plays it"):
                        with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                            ui.button(kind, icon=BLOCK_ICONS[kind], on_click=lambda _, kind=kind: place(kind, dict(BLOCKS[kind][0]))).props(
                                "outline dense no-caps align=left").style("width: 190px; height: 34px; flex: none")
                            ui.label(BLOCK_NOTES[kind].split(".")[0]).classes("label").style("text-transform: none; letter-spacing: 0")
                inspector = ui.column().classes("pane").style("flex: 1; min-width: 0")
            parts()
            refresh()
        dialog.open()

    def review():
        """The Review room of the chosen exam: every condition and combination, what the scanner
        and the subject did over the transients, and the truth no scanner has."""
        exam = chosen["exam"]
        if exam is None or not exam.spectra:
            ui.notify("Select an exam that has run")
            return
        axis = {"type": "value", "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}}
        legend = {"textStyle": {"color": INK}, "top": 0}
        palette = [SAND, GLOW, COPPER, GOOD, "#9FB7C9", "#C9A0B8", BAD]
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 1100px; max-width: 1400px"):
            with ui.row().classes("w-full items-center"):
                ui.label(f"Review · {exam.name}").classes("label")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat dense")
            if exam.maps:
                def painted(name):
                    grid = exam.maps[name][:, :, int(layer.value)]
                    return {"backgroundColor": "transparent", "animation": False, "tooltip": {},
                            "grid": {"left": 30, "right": 12, "top": 10, "bottom": 30},
                            "xAxis": {"type": "category", "data": list(range(grid.shape[0])), "axisLabel": {"color": DIM}},
                            "yAxis": {"type": "category", "data": list(range(grid.shape[1])), "axisLabel": {"color": DIM}},
                            "visualMap": {"min": 0, "max": float(grid.max()), "show": False,
                                          "inRange": {"color": ["#000000", COPPER, SAND, "#FFF2D8"]}},
                            "series": [{"type": "heatmap", "data": [[i, j, float(grid[i, j])] for i in range(grid.shape[0])
                                                                    for j in range(grid.shape[1])]}]}

                def cell(event):
                    i, j = int(event.data[0]), int(event.data[1])
                    one.options["series"][0]["data"] = [[float(p), float(v)] for p, v in zip(exam.ppm, exam.cells[i, j, int(layer.value)])]
                    one.options["title"]["text"] = f"cell {i}, {j}, slice {int(layer.value)}"
                    one.update()

                with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                    with ui.column().style("width: 420px"):
                        with ui.row().classes("w-full no-wrap"):
                            which = ui.select(list(exam.maps), value="NAA", label="map · click a cell").props("dense outlined dark").style("flex: 1")
                            layer = ui.select(list(range(exam.cells.shape[2])), value=exam.cells.shape[2] // 2, label="slice").props(
                                "dense outlined dark").style("width: 90px")
                        image = ui.echart(painted("NAA"), on_point_click=cell).style("height: 400px").classes("w-full")
                        for control in (which, layer):
                            control.on_value_change(lambda: (image.options.clear(), image.options.update(painted(which.value)), image.update()))
                    one = ui.echart({"backgroundColor": "transparent", "animation": False,
                                     "title": {"text": "centre cell", "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
                                     "grid": {"left": 12, "right": 12, "top": 28, "bottom": 28},
                                     "xAxis": {**axis, "inverse": True, "min": 0.2, "max": 4.2, "name": "ppm"},
                                     "yAxis": {"type": "value", "show": False, "scale": True},
                                     "series": [{"type": "line", "showSymbol": False, "lineStyle": {"color": SAND, "width": 1.3},
                                                 "data": [[float(p), float(v)] for p, v in zip(exam.ppm, exam.spectra["centre cell"])]}]
                                     }).style("height: 440px; flex: 1")
                dialog.open()
                return
            ui.echart({"backgroundColor": "transparent", "animation": False, "legend": legend, "color": palette,
                       "grid": {"left": 12, "right": 12, "top": 34, "bottom": 28}, "tooltip": {"trigger": "axis"},
                       "xAxis": {**axis, "inverse": True, "min": -0.5, "max": 6.5, "name": "ppm"},
                       "dataZoom": [{"type": "inside", "startValue": exam.window[0], "endValue": exam.window[1], "filterMode": "none"}],
                       "yAxis": {"type": "value", "show": False, "scale": True},
                       "series": [{"type": "line", "name": name, "showSymbol": False, "lineStyle": {"width": 1.2},
                                   "data": [[float(p), float(v)] for p, v in zip(exam.ppm, real)]}
                                  for name, real in exam.spectra.items()]}).style("height: 300px").classes("w-full")
            with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch"):
                with ui.column().style("flex: 2; min-width: 0"):
                    ui.label("Over the transients · frequency, phase and amplitude as they ran").classes("label")
                    ui.echart({"backgroundColor": "transparent", "animation": False, "legend": legend, "color": palette,
                               "grid": {"left": 40, "right": 12, "top": 34, "bottom": 28}, "tooltip": {"trigger": "axis"},
                               "xAxis": {**axis, "name": "transient"}, "yAxis": {**axis, "scale": True},
                               "series": [{"type": "line", "name": name, "showSymbol": False,
                                           "data": [[k + 1, float(v)] for k, v in enumerate(values)]}
                                          for name, values in exam.tracks.items()]}).style("height: 230px").classes("w-full")
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
            ui.notify(f"Exported {session.export_seq(exam, pathlib.Path.home() / 'MRSandbox' / 'exports')}")
        except Exception as error:  # an idealization Pulseq cannot play: the design says which
            ui.notify(f"Not exported: {error}", type="warning", multi_line=True)

    def changed(attribute, target=None):
        """A handler that writes a control's value to the session (or the card) and redraws."""
        def handle(event):
            if event.value is None:
                return
            before = getattr(session.protocol if target == "card" else session, attribute)
            setattr(session.protocol if target == "card" else session, attribute, event.value)
            if attribute == "subject":
                try:
                    session.body()
                except Exception as error:  # a subject whose files are placed by hand and are not there
                    ui.notify(f"{event.value} is not on this machine: {error}"[:200], type="warning", multi_line=True)
                    session.subject = before
                    event.sender.value = before
                    return
                session.position_mm = [0.0, 0.0, 0.0] if hasattr(session.body(), "contents") else [0.0, -45.0, 35.0]
                session.view_mm = list(session.position_mm)
                session.home()
                if session.field in FieldMap.PRESETS or session.field == "uniform":       # the operator's own map stays
                    session.field = FIELDS.get(session.subject, "uniform")
            if attribute in ("subject", "coil"):
                room()
            draw()
        return handle

    async def keep(event):
        """An uploaded file, kept where the session finds it again (~/MRSandbox/maps); returns the path."""
        folder = pathlib.Path.home() / "MRSandbox" / "maps"
        folder.mkdir(parents=True, exist_ok=True)
        await event.file.save(folder / event.file.name)
        return folder / event.file.name

    def maps():
        """The anatomy and its maps, line by line: the anatomy (a T1-weighted image), its three tissues, and its B0, B1+ and
        lipid maps. Every line says what is loaded now and offers what the package has stored, with "your own file ..." on
        top (the system's file dialog; NIfTI, or DICOM of any vendor). What is chosen waits; "Apply" at the end puts it all
        in force at once: a new anatomy with own tissue maps is used as it is, one without them is segmented first, and
        every map is brought onto the anatomy's grid. "Register" on a line is for a file of another session."""
        own, selects, shown_now = {}, {}, {}          # the operator's files by line, the lines' drop-downs, what each shows

        async def picked(what, value):
            """A stored choice waits as it is; "your own file ..." asks for the file, and a DICOM file gives its series."""
            if value == own.get(what):      # the line showing the file that was just staged on it: not a new choice
                return
            if value != OWN:
                own.pop(what, None)
                shown_now[what] = value
                return
            try:
                chosen_file = await run.io_bound(choose_file, f"The {what} file: NIfTI, or any DICOM file of the series")
                if chosen_file is None:
                    selects[what].set_value(shown_now[what])       # back to what it showed, so that the line can be chosen again
                    return
                if not chosen_file.name.endswith((".nii", ".nii.gz")):
                    told.text = "Reading the DICOM"
                    chosen_file = max((await run.io_bound(read_dicom, chosen_file)).values(), key=lambda path: path.stat().st_size)
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
            if what not in own:
                told.text = "Choose your own file on this line first: what the package has stored lies where it should already"
                return
            working.set_visibility(True)
            try:
                stage(what, await run.io_bound(session.register, own[what], own.get("anatomy")))
                told.text = f"The {what} is registered to the anatomy and waits for Apply"
            except Exception as error:
                told.text = f"Not registered: {error}"[:260]
            finally:
                working.set_visibility(False)

        async def apply():
            """Everything that waits, in the order it depends on: the anatomy first, then the maps onto it."""
            working.set_visibility(True)
            try:
                if "anatomy" in own or {"GM", "WM", "CSF"} <= set(own):
                    label = name.value or pathlib.Path(own.get("anatomy", own.get("GM", "anatomy"))).name.split(".")[0]
                    if {"GM", "WM", "CSF"} <= set(own):
                        session.anatomy(label, own["GM"], own["WM"], own["CSF"], picture=own.get("anatomy"))
                    else:
                        told.text = "Cropping the neck, extracting the brain and segmenting it with FSL: about two minutes"
                        await run.io_bound(session.segment, own["anatomy"], label)
                    pickers["subject"].set_options(list(session.subjects), value=label)
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

        fat = getattr(session.body(), "fat", None)
        lines = (("anatomy", "anatomy (T1-weighted)", list(session.subjects), session.subject),
                 ("GM", "grey matter", [f"of {session.subject}"], f"of {session.subject}"),
                 ("WM", "white matter", [f"of {session.subject}"], f"of {session.subject}"),
                 ("CSF", "CSF", [f"of {session.subject}"], f"of {session.subject}"),
                 ("B0", "B0 field (Hz)", ["uniform", *FieldMap.PRESETS], session.field),
                 ("B1+", "B1+ (of nominal)", ["nominal"], session.transmit or "nominal"),
                 ("lipid", "lipid layer (0 to 1)", ["its own" if fat is not None else "none", LENT], "its own" if fat is not None else "none"))
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 820px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Anatomy and maps").style(f"color: {SAND}; font-size: 15px")
                ui.space()
                working = ui.spinner(size="26px", color=SAND)
                working.set_visibility(False)
            told = ui.label("").classes("value").style("white-space: normal")
            for what, caption, stored, now in lines:
                shown_now[what] = now
                with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                    ui.label(caption).style("width: 170px")
                    selects[what] = ui.select({OWN: OWN, **{option: option for option in stored}, **({now: pathlib.Path(now).name} if now not in stored else {})},
                                              value=now, on_change=lambda e, what=what: picked(what, e.value)).mark(f"line {what}").props(
                        "dense outlined dark options-dense").style("flex: 1")
                    if what != "anatomy":
                        ui.button("Register", icon="my_location", on_click=lambda _, what=what: register(what)).props("outline dense no-caps").style(
                            "width: 130px").tooltip(
                            "for a file of ANOTHER session: FSL flirt, rigid, onto the anatomy. A file of the same session lies where the "
                            "anatomy lies already. A field map in Hz shows no anatomy: register its magnitude image instead")
                    else:       # the same as Apply does for a new anatomy without tissue maps, at once
                        ui.button("Segment", icon="psychology", on_click=apply).props("outline dense no-caps").style("width: 130px").tooltip(
                            "your own image into grey matter, white matter and CSF (FSL, about two minutes), with everything else "
                            "that waits on the lines")
            ui.switch("the B0 map was measured after my own shim: apply none", value=session.shim == "none", on_change=lambda e: (
                setattr(session, "shim", "none" if e.value else "linear"))).props("dense color=accent")
            ui.label("A new anatomy with your own three tissue maps is used as it is. Without them it is segmented on Apply (FSL: the neck "
                     "cropped, the brain extracted, grey matter, white matter and CSF; the bright tissue beyond the bone becomes the lipid "
                     "layer): about two minutes. T1, T2, concentrations and every other parameter take a map at their line in the "
                     "settings (the gears); coil sensitivities belong to the coils.").classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                name = ui.input("name of a new anatomy (the file's, if empty)").props("dense outlined dark").style("flex: 1")
                ui.button("Cancel", on_click=dialog.close).props("flat dense no-caps")
                ui.button("Apply", icon="check", on_click=apply).mark("apply the maps").props("unelevated dense no-caps").style("width: 120px")
        dialog.open()

    def realism():
        """What is real about the scan, each a switch: the console starts with everything real, and
        what is switched off is ideal (pulses that act at once, a uniform field, no noise, ...)."""
        def pulses(event):
            session.protocol.real = event.value
            draw()

        def field(event):
            session.field = FIELDS.get(session.subject, "uniform") if event.value else "uniform"
            draw()

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 640px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Realism · on is real, off is ideal").classes("label")
                ui.space()
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
            for caption, note, value, handler in (
                    ("real pulses", "shaped pulses under gradients, played in time: a voxel profile, displaced shifts, a slower simulation",
                     session.protocol.real, pulses),
                    ("B0 field", "the measured field of the head, shimmed as the Geometry tab says (the B0 overlay shows it after the shim)",
                     session.field != "uniform", field)):
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.switch(caption, value=value, on_change=handler).props("dense color=accent").style("width: 230px")
                    ui.label(note).classes("label").style("text-transform: none; letter-spacing: 0")
            for name, leaves in IDEALS.items():
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.switch(name, value=not all(session.overrides.get(path) == value for path, value in leaves.items()),
                              on_change=lambda e, name=name: (session.idealize(name, not e.value), draw())).props(
                        "dense color=accent").style("width: 230px")
                    ui.label("off sets " + ", ".join(leaves) + " to 0").classes("label").style("text-transform: none; letter-spacing: 0")
            ui.label("Drift, breathing and motion over the transients are part of every run (Review shows them). "
                     "Every single value stays open in the settings.").classes("label").style("text-transform: none; letter-spacing: 0")
        dialog.open()

    def run_settings():
        """What happens over the transients of every run: each instability drawn from the 3 T
        envelope (scan.Multiplex.transients), off, or set by the operator; and how far a motion
        event moves the head."""
        ways = ["drawn, as at 3 T", "off", "set to"]

        def put(argument, off, way, number):
            session.run_settings.pop(argument, None)
            if way.value == ways[1]:
                session.run_settings[argument] = off
            elif way.value == ways[2] and number.value is not None:
                session.run_settings[argument] = float(number.value)

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 680px"):
            with ui.row().classes("w-full items-center"):
                ui.label("The run · what happens over the transients").classes("label")
                ui.space()
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
            ui.switch("centre the frequency before every exam (prescan)", value=session.prescan, on_change=lambda e: setattr(
                session, "prescan", e.value)).props("dense color=accent").tooltip(
                "the scanner drifts at one rate through the whole session and its clock runs on with every exam. On, as scanners do by "
                "default: every exam starts on resonance. Off: an exam starts where the drift has taken the scanner by then, so the "
                "order of the exams matters (a water reference before or after its scan)")
            ui.label(f"This session the scanner drifts at {session.drift_hz_per_min:+.2f} Hz per minute; {session.clock_s / 60.0:.1f} minutes "
                     "have been scanned.").classes("label").style("text-transform: none; letter-spacing: 0")
            for caption, (argument, off) in RUN.items():
                now = session.run_settings.get(argument)
                with ui.row().classes("w-full items-center no-wrap").style("gap: 10px"):
                    ui.label(caption).style("width: 250px")
                    way = ui.select(ways, value=ways[0] if now is None else ways[1] if now == off else ways[2]).props(
                        "dense outlined dark options-dense").style("width: 170px")
                    number = ui.number(value=now if now not in (None, off) else None, format="%.4g").props("dense outlined dark").style(
                        "width: 110px").bind_visibility_from(way, "value", value=ways[2])
                    way.on_value_change(lambda _, a=argument, o=off, w=way, n=number: put(a, o, w, n))
                    number.on_value_change(lambda _, a=argument, o=off, w=way, n=number: put(a, o, w, n))
            with ui.row().classes("w-full items-center no-wrap").style("gap: 10px"):
                ui.label("a motion event moves the head by (mm)").style("width: 250px")
                low = ui.number("from", value=None if session.motion_mm is None else session.motion_mm[0], min=0.0).props(
                    "dense outlined dark clearable").style("width: 110px")
                high = ui.number("to", value=None if session.motion_mm is None else session.motion_mm[1], min=0.0).props(
                    "dense outlined dark clearable").style("width: 110px")
                for field in (low, high):
                    field.on_value_change(lambda: setattr(session, "motion_mm", None if low.value is None or high.value is None
                                                          else (float(min(low.value, high.value)), float(max(low.value, high.value)))))
            ui.label("Empty: the head stays, the events kick the frequency, the phase and the amplitude alone. A moved head is scanned "
                     "where it then lies, under the shim of the planned place. Review shows the tracks of every exam. A phase cycle is "
                     "set in the designer (a pulse's phase stepped over the transients, the receiver following); the cards keep their "
                     "pathways by crushers and filters.").classes("label").style(
                "text-transform: none; letter-spacing: 0")
        dialog.open()

    def calibration():
        """What the scanner is calibrated by: the records in force with their sources, and, where
        the package has records of the same hardware read under another coil, the choice of them."""
        async def read(event):
            if not named.value:
                said.text = "Give the calibration a name first"
                return
            busy.set_visibility(True)
            try:
                path = await run.io_bound(session.calibrate, await keep(event), named.value)
            except Exception as error:
                said.text = f"Not read: {error}"[:260]
                return
            finally:
                busy.set_visibility(False)
            dialog.close()
            draw()
            calibration()
            ui.notify(f"This scanner is now calibrated by {path.name}")

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 720px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Calibration · the records this scanner's noise, line shape and levels come from").classes("label")
                ui.space()
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props("unelevated dense no-caps")
            others = session.calibrations()
            if others:
                ui.radio({None: "the package's records for this scanner", **others}, value=session.calibration if session.calibration in others else None,
                         on_change=lambda e: (setattr(session, "calibration", e.value), dialog.close(), draw(), calibration())).props("dense color=accent")
            for leaf, record in session.outlook()["records"].items():
                measured = record.get("scanner") or {}
                with ui.column().classes("w-full").style(f"gap: 2px; border-top: 1px solid {LINE}; padding-top: 6px"):
                    ui.label(f"{RECORDS.get(leaf, leaf)}: {record['value']:g}").style(f"color: {SAND}")
                    ui.label(f"measured in: {record['source']}" + (f", on a {measured.get('vendor', '')} {measured.get('model', '')} "
                             f"at {measured.get('field_t', '')} T" if measured else "") + (f", with {record['sequence']}" if record.get("sequence") else "")
                             ).classes("label").style("text-transform: none; letter-spacing: 0; white-space: normal")
                    if record["extrapolated"]:
                        ui.label("your setup differs from that measurement in: " + "; ".join(
                            f"{what} ({' → '.join(str(v) for v in pair)})" if isinstance(pair, (list, tuple)) else str(what)
                            for what, pair in (record["extrapolated"].items() if isinstance(record["extrapolated"], dict)
                                               else ((w, None) for w in record["extrapolated"])))).style(f"color: {BAD}")
                    else:
                        ui.label("your setup matches that measurement").style(f"color: {GOOD}")
            ui.label("Calibrate this scanner from a scan of your own").classes("label").style("margin-top: 8px")
            ui.label("First set the card, the voxel and the room as your scan was acquired: they are its virtual twin. Then give raw "
                     "NIfTI-MRS with the transients apart (spec2nii writes it). The noise level and the line broadening are read from "
                     "it and kept as this scanner's calibration, on this machine only.").classes("label").style(
                "text-transform: none; letter-spacing: 0; white-space: normal")
            with ui.row().classes("w-full items-center no-wrap"):
                named = ui.input("name of the calibration").props("dense outlined dark").style("width: 240px")
                ui.upload(label="your scan, .nii.gz", auto_upload=True, on_upload=lambda e: read(e)).props(
                    "flat dense dark accept=.nii,.gz").style("width: 260px")
                busy = ui.spinner(size="28px", color=SAND)
                busy.set_visibility(False)
            said = ui.label("").classes("value").style("white-space: normal")
        dialog.open()

    def coil_gear():
        """The receive array's own settings: how many loops, how large, where."""
        settings = session.coil_settings[session.coil]
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 420px"):
            ui.label(f"Receive array · {session.coil}").classes("label")
            if not settings:
                ui.label("One ideal channel has nothing to set: uniform sensitivity, the scanner's noise.").classes("value")
            for key, value in settings.items():
                ui.number(key.replace("_mm", " (mm)").replace("_", " "), value=value, format="%.6g", on_change=lambda e, key=key: (
                    e.value is not None and (settings.__setitem__(key, float(e.value)), room(), draw()))).props("dense outlined dark").classes("w-full")
            ui.button("Back to the defaults", icon="restart_alt", on_click=lambda: (
                settings.clear(), settings.update(session.COIL_SETTINGS.get(session.coil, {})), dialog.close(), room(), draw(), coil_gear())).props(
                "flat dense no-caps")
            async def own(kind):
                """The operator's own: measured sensitivity maps (the array is then theirs), or a raw scan with the coils apart,
                from which the channels' own share of the noise is read."""
                picked_file = await run.io_bound(choose_file, "Sensitivity maps, one NIfTI of all channels" if kind == "maps" else
                                                 "A raw scan with the coils apart, NIfTI-MRS")
                if picked_file is None:
                    said.text = "No file chosen"
                    return
                files = [str(picked_file)]
                try:
                    if kind == "maps":
                        session.coil_maps(files[0], pathlib.Path(files[0]).name.split(".")[0])
                        pickers["coil"].set_options(list(session.coils), value=session.coil)
                        dialog.close()
                        room()
                        draw()
                    else:
                        said.text = f"The channels' own share of the noise reads {await run.io_bound(session.coil_noise, files[0]):.2f} (the package's: 0.25)"
                except Exception as error:
                    said.text = f"Not taken: {error}"[:260]

            ui.label("Your own").classes("label").style("margin-top: 6px")
            with ui.row().classes("w-full").style("gap: 6px"):
                ui.button("Sensitivity maps", icon="folder_open", on_click=lambda: own("maps")).props("outline dense no-caps").style("width: 200px").tooltip(
                    "one NIfTI of (nx, ny, nz, channels), complex or real, in the scanner's coordinates: the array is then yours")
                ui.button("A scan with the coils apart", icon="folder_open", on_click=lambda: own("noise")).props("outline dense no-caps").style(
                    "width: 240px").tooltip("raw NIfTI-MRS: the share of a channel's noise that is its own is read from the FIDs' tails, against "
                                            "the array of loops chosen here")
            said = ui.label("" if session.own_noise is None else f"own noise share {session.own_noise:.2f}, read from your scan").classes("value").style(
                "white-space: normal")
            ui.label("The loops show in the 3D view. Their sensitivities and noise follow from their geometry and the load in them.").classes(
                "label").style("text-transform: none; letter-spacing: 0")
            ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
        dialog.open()

    def browse():
        """The protocol browser: what can go on the card, by family, and the operator's saved protocols."""
        def pick(name):
            session.protocol.sequence = name
            if name == MRSI:            # a spectroscopic image is read shorter and narrower than a single voxel, and has no reference exam
                session.protocol.npoints, session.protocol.bandwidth_hz, session.protocol.water_reference = 1024, 2000.0, 0
            chosen["snr"] = None
            dialog.close()
            draw()
            if name == "Custom":
                designer()

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 860px"):
            ui.label("Protocols").classes("label")
            with ui.row().classes("w-full no-wrap").style("gap: 14px; align-items: flex-start"):
                for family, names in (("Single voxel", [n for n in SEQUENCES if n not in REAL.values()]),
                                      ("Edited", [n for n in EDITING if n not in REAL.values()]),
                                      ("Spectroscopic imaging", [MRSI]), ("Own design", ["Custom"])):
                    with ui.column().style("gap: 4px; flex: 1"):
                        ui.label(family).classes("label")
                        for name in names:
                            ui.button(name, on_click=lambda _, name=name: pick(name)).mark(f"protocol {name}").props(
                                "outline dense no-caps align=left").classes("w-full")
                with ui.column().style("gap: 4px; flex: 1"):
                    ui.label("Saved").classes("label")
                    for name in session.protocols():
                        ui.button(name, icon="bookmark", on_click=lambda _, name=name: (
                            session.load_protocol(name), dialog.close(), draw())).props("outline dense no-caps align=left").classes("w-full")
        dialog.open()

    def drag(plane):
        """A plane's mouse: a press or a drag moves the voxel, the release settles the card."""
        def handle(event):
            kind, pressed = event.args.get("mouse_event_type"), event.args.get("buttons") == 1
            if kind == "mousedown" or (kind == "mousemove" and pressed):
                # with shift the press looks around (the other planes come to this point), without it it places the voxel
                chosen["looking"] = bool(event.args.get("shiftKey"))      # the side panel then tells of the place looked at
                (session.look if chosen["looking"] else session.place)(plane, event.args.get("image_x", 0.0), event.args.get("image_y", 0.0))
                chosen["dragging"] = True
                paint()
            elif chosen["dragging"]:        # the release, or the first move after it: the throttle may drop the release itself
                chosen["dragging"] = False
                draw()
        return handle

    def toggle(name):
        """The toolbar's switches: the water suppression steps through its schemes, OVS is on or off."""
        card = session.protocol
        if name == "ovs":
            card.ovs = not card.ovs
        elif name == "real":
            card.real = not card.real
        else:
            schemes = ["off"] + list(SUPPRESSION)
            card.water_suppression = schemes[(schemes.index(card.water_suppression) + 1) % len(schemes)]
        draw()

    # the top bar: what is in the room
    with ui.row().classes("w-full items-center pane no-wrap").style("gap: 14px; padding: 4px 10px"):
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" height="40" style="display: block; height: 40px" src="{MARK}">', sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 16px; letter-spacing: .1em")
        for attribute, caption, options, width in (("scanner", "scanner", session.scanners(), 230),
                                                   ("coil", "receive array", list(session.coils), 215),
                                                   ("subject", "anatomy or phantom", list(session.subjects), 240)):
            pickers[attribute] = ui.select(options, value=getattr(session, attribute), label=caption, on_change=changed(attribute)).props(
                "dense outlined dark").style(f"min-width: {width}px")
            if attribute == "scanner":
                ui.button(icon="verified", on_click=calibration).mark("calibration").props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip("what this scanner is calibrated by")
            if attribute in GEARS or attribute == "coil":
                ui.button(icon="settings", on_click=(lambda _, panel=attribute: gear(panel)) if attribute in GEARS else coil_gear).props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip({"subject": "the parameters and ranges of the anatomy or phantom"}.get(attribute, f"the {attribute}'s parameters and ranges"))
        ui.button(icon="person_add_alt", on_click=lambda: (setattr(session, "seed", int(np.random.default_rng().integers(2 ** 31))),
                                                           ui.notify("A new draw of the population is scanned from here on"))).props(
            "flat dense round size=sm").style(f"color: {GLOW}; margin-left: -10px").tooltip(
            "every exam scans the same draw of the population; this takes a new one")
        ui.space()
        ui.button("all settings", icon="tune", on_click=lambda: gear()).props("flat dense no-caps").style(f"color: {GLOW}")
        status = ui.label("idle").classes("label")

    with ui.row().classes("w-full no-wrap").style("gap: 6px; align-items: stretch; flex: 1; min-height: 0"):
        # the exam cards
        with ui.column().classes("pane").style("width: 244px"):
            ui.label("Exam cards").classes("label")
            with ui.scroll_area().style("flex: 1; width: 100%"):
                rows = ui.column().classes("w-full").style("gap: 0")
            ui.button("New scan", icon="add", on_click=browse).props("outline no-caps").classes("w-full")
            ui.label("Protocol library").classes("label").style("margin-top: 4px")
            saved = ui.select(session.protocols(), label="name", with_input=True, new_value_mode="add-unique").props(
                "dense outlined dark").classes("w-full")
            with ui.row().classes("w-full no-wrap").style("gap: 6px"):
                ui.button("Save", icon="save", on_click=lambda: saved.value and (
                    session.save_protocol(saved.value), saved.set_options(session.protocols(), value=saved.value),
                    ui.notify(f"Saved {saved.value}"))).props("outline dense no-caps").style("flex: 1")
                ui.button("Load", icon="folder_open", on_click=lambda: saved.value in session.protocols() and (
                    session.load_protocol(saved.value), draw(), ui.notify(f"Loaded {saved.value}"))).props(
                    "outline dense no-caps").style("flex: 1")

        # planning: the tools, three planes one leafs through and the room in 3D, the card below
        with ui.column().style("flex: 1; min-width: 0; gap: 6px"):
            with ui.row().classes("w-full items-center pane no-wrap").style("padding: 4px 8px"):
                ui.button("to the voxel", icon="center_focus_strong", on_click=lambda: (
                    setattr(session, "view_mm", list(session.position_mm)), paint())).props("flat dense no-caps stack").classes("tool").tooltip(
                    "bring the three planes back to the voxel")
                ui.button("realism", icon="auto_fix_high", on_click=realism).props("flat dense no-caps stack").classes("tool").tooltip(
                    "everything real about the scan as switches: pulses, the B0 field, noise, macromolecules, lipids, eddy currents, ...")
                tools["shifts"] = ui.button("shifts", icon="flip_to_back", on_click=lambda: (
                    chosen.__setitem__("shifts", not chosen["shifts"]), draw())).props("flat dense no-caps stack").classes("tool on").tooltip(
                    "display only: draws where water (blue) and lipid (pink) are really selected. With real pulses the displacement is "
                    "always in the signal; this button does not switch it off (ideal pulses do)")
                tools["flip"] = ui.button("L on the right", icon="flip", on_click=lambda: (
                    chosen.__setitem__("flipped", not chosen["flipped"]), draw())).props("flat dense no-caps stack").classes("tool").tooltip(
                    "off: left is on the picture's left (neurological). on: mirrored, left on the right, as a radiologist "
                    "reads it. The letters and the axes' crosses follow")
                ui.button("the run", icon="timeline", on_click=run_settings).props("flat dense no-caps stack").classes("tool").tooltip(
                    "what happens over the transients: drift, breathing, jitter, motion")
                ui.button("coils", icon="settings_input_antenna", on_click=coil_gear).props("flat dense no-caps stack").classes("tool").tooltip(
                    "the receive array: its loops, your own sensitivity maps, the noise its channels share")
                ui.button("maps", icon="layers", on_click=maps).props("flat dense no-caps stack").classes("tool").tooltip(
                    "load your own: the anatomy from a T1-weighted image, B0, B1+ and lipid maps; or use the measured ones")
                overlays = ui.select(list(OVERLAYS), value=session.overlay, on_change=lambda e: e.value and (
                    setattr(session, "overlay", e.value), paint())).props(
                    "dense outlined dark options-dense").style("width: 150px").tooltip(
                    "wash a map over the anatomy: the B0 field and B1+ (on in maps), or a tissue")
                texts["window"] = ui.label("").classes("label").style("text-transform: none; letter-spacing: 0")
                ui.space()
                ui.label("scroll: leaf through · press or drag: place the voxel · with shift: look around").classes("label")
            with ui.element("div").classes("w-full").style("display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px"):
                for name in ("tra", "sag", "cor"):
                    with ui.element("div").classes("port"):
                        views[name] = ui.interactive_image(events=["mousedown", "mousemove", "mouseup"], cross=False).classes("w-full")
                        # throttled: an unthrottled stream of moves queues more repaints than the window can show, and
                        # the console then seems deaf to the next clicks
                        views[name].on("mouse", drag(name), throttle=0.06)
                        views[name].on("wheel.prevent", lambda e, plane=name: (
                            session.scroll(plane, 1 if e.args["deltaY"] < 0 else -1), paint()), args=["deltaY"], throttle=0.04)
                        texts[name] = ui.label("").classes("tag").style("top: 4px; left: 6px")
                        letters[name] = [ui.label(letter).classes("tag").style(place) for letter, place in zip(
                            EDGES[name][:4], ("top: 4px; left: 50%", "bottom: 4px; left: 50%", "top: 48%; left: 6px", "top: 48%; right: 6px"))]
                        # the axes' cross, lower right: where the picture's right and its top point in the subject
                        crosses[name] = ui.html("", sanitize=False).classes("tag").style("right: 6px; bottom: 6px")
            with ui.row().classes("w-full no-wrap").style("gap: 6px; flex: 1; min-height: 0; align-items: stretch"):
                # the card's parameters by tab, what follows from them beside
                with ui.column().classes("pane").style("width: 590px; flex: none; min-height: 0; overflow: hidden"):
                    card = session.protocol
                    with ui.row().classes("w-full items-center no-wrap"):
                        with ui.tabs().props("dense align=left").style(f"color: {DIM}; flex: 1") as tabs:
                            for name in ("Geometry", "Contrast", "Acquisition"):
                                ui.tab(name)
                            ui.tab("Imaging").bind_visibility_from(card, "sequence", backward=lambda name: name == MRSI)
                        ui.button(icon="settings", on_click=lambda: gear(tabs.value)).props("flat dense round size=sm").style(
                            f"color: {GLOW}").tooltip("the advanced settings behind this tab")
                    with ui.row().classes("w-full no-wrap").style("gap: 12px; flex: 1; min-height: 0"):
                        with ui.tab_panels(tabs, value="Geometry").props("keep-alive").style("width: 215px; flex: none; overflow-y: auto; max-height: calc(100vh - 650px)"):
                            with ui.tab_panel("Geometry"):
                                for axis, caption in enumerate(("x, to the right (mm)", "y, to the front (mm)", "z, to the head (mm)")):
                                    places.append(ui.number(caption, value=session.position_mm[axis], step=1.0, on_change=lambda e, axis=axis: (
                                        e.value is not None and abs(e.value - session.position_mm[axis]) > 0.05 and (
                                            session.position_mm.__setitem__(axis, float(e.value)), draw()))).props(
                                        "dense outlined dark").classes("w-full"))
                                with ui.row().classes("w-full no-wrap").style("gap: 4px"):
                                    for axis, caption in enumerate(("size x", "size y", "size z")):
                                        edges.append(ui.number(caption, value=session.size_mm[axis], step=5.0, min=3.0, on_change=lambda e, axis=axis: (
                                            e.value is not None and e.value != session.size_mm[axis] and (
                                                session.size_mm.__setitem__(axis, float(e.value)), room(), draw()))).props(
                                            "dense outlined dark").style("flex: 1; min-width: 0"))
                                ui.select({axes: f"90 along {axes[0]}, 180s along {axes[1]} and {axes[2]}" for axes in (
                                    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx")}, label="slice order (real PRESS)", on_change=changed("axes", "card")
                                          ).bind_value(card, "axes").props("dense outlined dark options-dense").classes("w-full").tooltip(
                                    "the narrow 180s displace a shift the most: the order decides along which two axes")
                                ui.select(list(FieldMap.SHIMS), value=session.shim, label="shim", on_change=changed("shim")).props(
                                    "dense outlined dark").classes("w-full")
                                ui.number("shim box (mm)", value=session.shim_mm, step=5.0, min=5.0,
                                          on_change=lambda e: (setattr(session, "shim_mm", e.value), draw())).props(
                                    "dense outlined dark clearable").classes("w-full").tooltip(
                                    "the yellow box about the voxel the shim is fitted over; empty, the shim is fitted over the voxel itself")
                            with ui.tab_panel("Imaging"):
                                with ui.row().classes("w-full no-wrap").style("gap: 4px"):
                                    for attribute, caption, low in (("matrix", "cells x", 2), ("matrix_y", "cells y", 2), ("slices", "slices z", 1)):
                                        ui.number(caption, step=1, min=low, on_change=changed(attribute, "card")).bind_value(card, attribute).props(
                                            "dense outlined dark").style("flex: 1; min-width: 0")
                                for attribute, caption, options, tip in (
                                        ("mrsi_oversampling", "rendering under a cell", [1, 2, 3], "1: ideal boxes. More: every cell sees its neighbours "
                                         "through the point spread function, as a real image does"),
                                        ("mrsi_weighting", "k-space weighting", ["none", "hamming"], "hamming: less bleed, wider cells"),
                                        ("mrsi_sampling", "k-space coverage", ["full", "elliptical"], "elliptical: the corners left out, a shorter scan"),
                                        ("mrsi_profile", "slab profile", ["box", "gaussian"], "how the slab is excited through its thickness"),
                                        ("mrsi_readout", "readout", ["phase encoded", "epsi", "epsi-bidirectional"], "EPSI reads a line of "
                                         "k-space per repetition: a matrix times shorter scan, its own artefacts")):
                                    ui.select(options, label=caption, on_change=changed(attribute, "card")).bind_value(card, attribute).props(
                                        "dense outlined dark options-dense").classes("w-full").tooltip(tip)
                            for tab, fields in (("Contrast", (("te_ms", "TE (ms)", 1.0), ("tr_ms", "TR (ms)", 100.0))),
                                                ("Acquisition", (("averages", "averages", 1), ("npoints", "points", 512),
                                                                 ("bandwidth_hz", "bandwidth (Hz)", 500.0),
                                                                 ("water_reference", "water reference transients", 1)))):
                                with ui.tab_panel(tab):
                                    if tab == "Contrast":
                                        ui.button(on_click=browse).bind_text_from(card, "sequence").props(
                                            "outline no-caps icon-right=expand_more").classes("w-full")
                                    if tab == "Acquisition":
                                        ui.switch("reference before the scan", on_change=changed("water_reference_first", "card")).bind_value(
                                            card, "water_reference_first").props("dense color=accent").tooltip(
                                            "the water reference is an exam of its own: before its card, or after it (off)")
                                    if tab == "Contrast":
                                        with ui.column().classes("w-full").style("gap: 4px").bind_visibility_from(
                                                card, "sequence", backward=lambda name: name == "MEGA-PRESS"):
                                            ui.select(list(TARGETS), label="editing target", on_change=lambda e: e.value and (
                                                setattr(card, "edit_on_ppm", TARGETS[e.value][0]), setattr(card, "edit_off_ppm", TARGETS[e.value][1]),
                                                draw())).props("dense outlined dark").classes("w-full").tooltip(
                                                "fills the two frequencies below; any pair can be typed")
                                            for attribute, caption in (("edit_on_ppm", "editing pulse ON (ppm)"), ("edit_off_ppm", "editing pulse OFF (ppm)")):
                                                ui.number(caption, step=0.01, format="%.2f", on_change=changed(attribute, "card")).bind_value(
                                                    card, attribute).props("dense outlined dark").classes("w-full")
                                    for attribute, caption, step in fields:
                                        ui.number(caption, value=getattr(card, attribute), step=step, on_change=changed(attribute, "card")
                                                  ).bind_value(card, attribute).props("dense outlined dark").classes("w-full")
                        with ui.element("div").classes("info").style(
                                f"flex: 1; min-height: 0; max-height: calc(100vh - 650px); overflow-y: auto; border: 1px solid {LINE}; border-radius: 2px; padding: 8px 10px; background: {GROUND}"):
                            ui.label("Overview").classes("label").style(f"grid-column: 1 / -1; color: {SAND}")
                            for key, caption in (("time", "scan duration"), ("timing", "TR / TE"), ("volume", "voxel volume"), ("snr", "relative SNR"),
                                                 ("tissue", "tissue"), ("displaced", "displaced voxels"), ("lipid", "lipid layer"), ("linewidth", "line width from B0"), ("encoding", "image encoding"), ("resolution", "spectral resolution"), ("readout", "readout duration"),
                                                 ("limits", "hardware")):
                                ui.label(caption).classes("label")
                                texts[key] = ui.label("").classes("value").style("white-space: pre-line")
                # what the card plays and what came of it
                with ui.column().classes("pane").style("flex: 1; min-width: 0"):
                    with ui.tabs().props("dense align=left").style(f"color: {DIM}") as shows:
                        ui.tab("Sequence")
                        ui.tab("Live")
                    with ui.tab_panels(shows, value="Sequence").props("keep-alive").classes("w-full").style("flex: 1"):
                        with ui.tab_panel("Sequence"):
                            with ui.row().classes("w-full no-wrap").style("gap: 8px"):
                                with ui.column().style("gap: 4px; flex: none"):
                                    tools["real"] = ui.button("real pulses", icon="waves", on_click=lambda: toggle("real")).props(
                                        "flat dense no-caps stack").classes("tool").tooltip(
                                        "off: ideal pulses that act at once and select the voxel exactly (fast). on: shaped pulses under gradients, played in "
                                        "time, so the voxel has a profile, shifts displace, and the scan takes longer to simulate")
                                    tools["water"] = ui.button("water off", icon="water_drop", on_click=lambda: toggle("water")).props(
                                        "flat dense no-caps stack").classes("tool").tooltip("water suppression before the localization: off, VAPOR, WET, CHESS")
                                    tools["ovs"] = ui.button("OVS", icon="border_outer", on_click=lambda: toggle("ovs")).props(
                                        "flat dense no-caps stack").classes("tool").tooltip("outer-volume suppression: six saturation slabs about the voxel")
                                    ui.button("design", icon="architecture", on_click=lambda: (session.protocol.sequence != "Custom" and session.adopt(), designer())).props("flat dense no-caps stack").classes("tool").tooltip(
                                        "the sequence designer: your own design in blocks. The card keeps playing what it plays until you use the design")
                                diagram = ui.html("", sanitize=False).style("flex: 1; min-width: 0")
                        with ui.tab_panel("Live"):
                            with ui.row().classes("w-full items-center no-wrap"):
                                looks = ui.select([], label="show", on_change=lambda: chosen["exam"] is not None and chosen["exam"].frames
                                                  and frame(chosen["exam"], -1)).props("dense outlined dark options-dense").style("width: 200px")
                                ui.switch("align transients", value=session.align, on_change=lambda e: setattr(session, "align", e.value)).props(
                                    "dense color=accent").tooltip(
                                    "before the display averages, every transient is aligned in frequency and phase to the run's mean, as a "
                                    "scanner's or a toolbox's preprocessing does; off shows what drift and motion do. Applies to the next "
                                    "exams; the export is always raw. Display: 2 Hz apodization, zero filled, one zero-order phase per exam. "
                                    "Wheel zooms, drag pans, grey is the transient that just came in")
                            spectrum = ui.echart(copy.deepcopy(SPECTRUM)).style("height: 300px").classes("w-full")
            # the card in one line, and its way into the queue
            with ui.row().classes("w-full items-center pane no-wrap").style("gap: 18px; padding: 4px 10px"):
                texts["name"] = ui.label("").style(f"color: {SAND}")
                ui.label("duration").classes("label")
                texts["time2"] = ui.label("").classes("value")
                texts["verdict"] = ui.label("").classes("value").style("flex: 1")
                discard = ui.button("Discard", on_click=lambda: (chosen.__setitem__("editing", None), listing(), draw())).props(
                    "flat dense no-caps")
                accept = ui.button("Accept", icon="playlist_add", on_click=plan).props("unelevated dense no-caps")

        # scan control, and what the scan runs under
        with ui.column().classes("pane").style("width: 244px"):
            ui.label("Scan control").classes("label")
            progress = ui.linear_progress(value=0.0, show_value=False, size="8px", color=SAND)
            with ui.row().classes("w-full items-center"):
                texts["count"] = ui.label("0 / 0").classes("value")
                ui.space()
                ui.button(icon="stop", on_click=lambda: chosen.__setitem__("stop", True)).props("outline dense round").tooltip(
                    "hold the queue after the exam that is running")
                start = ui.button("Start", icon="play_arrow", on_click=scan_queue).props("unelevated dense no-caps")
            texts["took"] = ui.label("").classes("label").style("text-transform: none; letter-spacing: 0")
            ui.button("Review", icon="insights", on_click=review).props("outline dense no-caps align=left").classes("w-full")
            ui.button("Export raw transients", icon="save_alt", on_click=export).props("outline dense no-caps align=left").classes("w-full")
            ui.button("Export Pulseq .seq", icon="code", on_click=export_seq).props("outline dense no-caps align=left").classes("w-full")
            with ui.element("div").classes("port w-full"):
                view = ui.scene(width=226, height=226, grid=False, background_color="#000000")
                view.move_camera(x=6.5, y=-6.5, z=4.5, look_at_x=0, look_at_y=0, look_at_z=0)
                ui.label("the room · drag to turn").classes("tag").style("top: 4px; left: 6px")
            panel = ui.label("This voxel").classes("label").tooltip(
                "one draw of the population, scanned by every exam until a new one is taken (the person icon); "
                "the truth a real scanner never shows").classes("label").style("margin-top: 6px")
            with ui.scroll_area().style("flex: 1; width: 100%"):
                records = ui.column().classes("w-full").style("gap: 1px")
    chosen["ready"] = True
    room()
    draw()
    loading.set_visibility(False)       # (not deleted: a window closed while the room was being made has no page to delete from)


#**********#
#   main   #
#**********#
def main():
    """Start the console in a window of its own; with MRSANDBOX_CONSOLE_PORT set it serves a
    browser at that port instead (MRSANDBOX_CONSOLE_HEADLESS: without opening one)."""
    if "MRSANDBOX_CONSOLE_PORT" in os.environ:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False,
               show="MRSANDBOX_CONSOLE_HEADLESS" not in os.environ, port=int(os.environ["MRSANDBOX_CONSOLE_PORT"]))
    else:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False, native=True, window_size=(1680, 1020))


if __name__ in {"__main__", "__mp_main__"}:
    main()
