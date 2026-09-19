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
from nicegui import run, ui
from PIL import Image, ImageFilter

from mrsandbox.console.session import BLOCKS, EDITING, FIELDS, GEARS, IDEALS, MRSI, OVERLAYS, REAL, RUN, TARGETS, SEQUENCES, SUPPRESSION, TEMPLATES, Session
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
.chip.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.tool.on {{ border-color: {SAND}; color: {GLOW} !important; background: #2b241e; }}
.info {{ display: grid; grid-template-columns: auto 1fr; column-gap: 12px; row-gap: 5px; align-content: start; white-space: nowrap; }}
.q-tab {{ min-height: 28px; padding: 0 12px; }} .q-tab__label {{ font-size: 11px; letter-spacing: .08em; }}
.q-tab-panel {{ padding: 8px 2px 2px 2px; }} .q-tab-panels {{ background: transparent; }}
.q-field__native, .q-field__label, .q-select__dropdown-icon, .q-field__input {{ color: {INK} !important; }}
.q-field--outlined .q-field__control:before {{ border-color: {LINE} !important; }}
"""
MM = 1.0 / 40.0  # the 3D view's units per millimetre
# the logo has one home, the repository's assets/; an installed package carries that file in its
# environment's share/mrsandbox (pyproject.toml, data-files)
LOGO = next((path for path in (pathlib.Path(__file__).parents[2] / "assets" / "logo" / "logo.png",
                               pathlib.Path(sys.prefix) / "share" / "mrsandbox" / "logo.png") if path.exists()), None)
CHOICES = {"shape": ["sinc", "gaussian", "hermite"], "slice_axis": ["none", "x", "y", "z"], "axis": ["x", "y", "z"],
           "scheme": list(SUPPRESSION)}
# the designer's blocks in plain words: an icon, what the block is, and its settings with their units
BLOCK_ICONS = {"hard pulse": "bolt", "adiabatic pulse": "all_inclusive", "own pulse": "gesture", "shaped pulse": "waves", "delay": "hourglass_empty",
               "crusher": "signal_cellular_alt", "coherence filter": "filter_alt", "water suppression": "water_drop"}
BLOCK_NOTES = {
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
# the settings' groups by name, and the ways a parameter can be set
GROUPS = {"conc": "Concentrations (mM)", "t1": "T1 of the metabolites (s)", "t2": "T2 of the metabolites (s)",
          "contrast": "Grey to white matter concentration ratio", "diffusion": "Diffusion", "water": "Water", "lipids": "Lipids",
          "mm": "Macromolecules", "downfield": "Downfield peaks", "lineshape": "Line shape (Hz)", "noise": "Noise",
          "b1_scale": "B1 scale", "acquisition": "Acquisition: shifts, phases, eddy currents", "gradients": "Gradient eddy currents",
          "spurious": "Spurious echoes"}
WAYS = ["as its source gives it", "set to", "vary between", "from a map"]
# the displaced voxels' colours on the planes (session.SHIFTED)
SHIFT_COLOURS = {"water": "#7FB2D9", "lipid": "#D98BB0", "shim": "#E6D84A"}
# the planes' edges as a console letters them (top, bottom, left, right), and the axis each plane leafs along
EDGES = {"tra": ("A", "P", "L", "R", 2, "z"), "sag": ("H", "F", "P", "A", 0, "x"), "cor": ("H", "F", "L", "R", 1, "y")}

#**********#
#   mark   #
#**********#
# the logo for the bar, its studio backdrop cut away once at the start: the sand and the tray are grainy, the
# backdrop and the glow behind the peak are smooth, so what is smooth and joined to the border goes
MARK = None
if LOGO is not None:
    full = Image.open(LOGO).convert("RGB").crop((60, 200, 964, 800))
    small = full.resize((452, 300), Image.LANCZOS)
    tone = np.asarray(full.convert("L")).astype(float)
    rough = np.abs(tone - np.asarray(full.convert("L").filter(ImageFilter.BoxBlur(3))).astype(float))
    grain = np.asarray(Image.fromarray(np.uint8(np.clip(8.0 * rough, 0, 255))).filter(ImageFilter.BoxBlur(6)).resize((452, 300)))
    smooth = (grain < 8 * 2.2) & (np.asarray(small).max(axis=2) > 40)
    backdrop = np.zeros(smooth.shape, dtype=bool)
    backdrop[0, :] = backdrop[-1, :] = backdrop[:, 0] = backdrop[:, -1] = True
    backdrop &= smooth
    while True:
        grown = backdrop.copy()
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown |= np.roll(backdrop, shift, (0, 1)) & smooth
        if (grown == backdrop).all():
            break
        backdrop = grown
    small.putalpha(Image.fromarray(np.uint8(255 * ~backdrop)).filter(ImageFilter.GaussianBlur(1.2)))
    buffer = io.BytesIO()
    small.resize((151, 100), Image.LANCZOS).save(buffer, format="PNG")
    MARK = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


#*************#
#   diagram   #
#*************#
def lanes(drawn, width=470.0, lane=46.0, mark=None):
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
    reach = 0.62 * tall / max([abs(v) for _, v in drawn["rf"]] + [h for _, h in drawn["bars"]] + [1e-9])
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
    for t, tallness in drawn["bars"]:
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
def console():
    """One operator's console; every browser tab is a session of its own."""
    session = Session()
    ui.add_head_html(f"<style>{STYLE}</style>")
    ui.dark_mode(True)
    ui.colors(primary=COPPER, accent=SAND, positive=GOOD, negative=BAD)
    views, shown, texts, tools, pickers, places, edges, chosen = {}, {}, {}, {}, {}, [], [], {"exam": None, "box": None, "stop": False, "shifts": True, "snr": None, "editing": None}

    def room():
        """Rebuild the 3D view: the subject's surface, the receive loops, the voxel's box."""
        drawn = session.scene()
        view.clear()
        with view:
            cloud = drawn["cloud"] * MM
            shade = 0.45 + 0.55 * (cloud[:, 2] - cloud[:, 2].min()) / max(float(np.ptp(cloud[:, 2])), 1e-6)
            ui.scene.point_cloud(cloud.tolist(), (shade[:, None] * np.array([0.89, 0.65, 0.37])).tolist(), point_size=0.035)
            if len(drawn["lipid"]):                                # the lipid layer about the brain, to judge a voxel near it
                ui.scene.point_cloud((drawn["lipid"] * MM).tolist(), [[0.85, 0.55, 0.69]] * len(drawn["lipid"]), point_size=0.03)
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
        cells = int(session.protocol.matrix) if session.protocol.sequence == MRSI else 1
        for name, (image, (left, top, width, height), cut, (column, row), shifted) in session.planes().items():
            stamp = hash(image.tobytes())
            if shown.get(name) != stamp:                            # a moved voxel changes the outlines, not the picture
                shown[name] = stamp
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                views[name].set_source("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            side = image.shape[0]
            grid = "".join(f'<line x1="{left + width * n / cells:.1f}" x2="{left + width * n / cells:.1f}" y1="{top:.1f}" '
                           f'y2="{top + height:.1f}" /><line y1="{top + height * n / cells:.1f}" y2="{top + height * n / cells:.1f}" '
                           f'x1="{left:.1f}" x2="{left + width:.1f}" />' for n in range(1, cells))
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
        wash = session.wash()
        texts["window"].text = ("" if wash is None or not wash[3] else f"blue {wash[1]:+.3g} · red {wash[2]:+.3g}"
                                + (" Hz" if session.overlay == "B0 field" else "")) if session.overlay != "anatomy" else ""
        if session.overlay in ("B0 field", "B1+") and wash is None:
            texts["window"].text = "no such map on: choose one in maps"
        for axis, field in enumerate(places):
            field.value = round(session.position_mm[axis], 1)

    def draw():
        """Redraw everything the card and the room decide: the planes, the diagram, the outlook."""
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
        texts["lipid"].text = ("this subject has no lipid layer" if look["lipid_mm"] is None else
                               f"{look['lipid_mm'][0]:.0f} mm from the voxel, {look['lipid_mm'][1]:.0f} mm from where lipid is selected")
        texts["lipid"].style(f"color: {BAD if look['lipid_mm'] is not None and min(look['lipid_mm']) < 1.0 else INK}")
        texts["tissue"].text = f"GM {look['gm']:.2f}  WM {look['wm']:.2f}  CSF {look['csf']:.2f}"
        texts["resolution"].text = f"{card.bandwidth_hz / card.npoints:.2f} Hz / point"
        texts["readout"].text = f"{1000.0 * card.npoints / card.bandwidth_hz:.0f} ms"
        texts["limits"].text = "B1 {:g} uT · G {:g} mT/m · slew {:g} T/m/s".format(*look["limits"])
        texts["verdict"].text = "plays on this scanner" if not look["violations"] else "; ".join(look["violations"])
        texts["verdict"].style(f"color: {GOOD if not look['violations'] else BAD}")
        records.clear()
        with records:
            for leaf, record in look["records"].items():
                ui.label(f"{leaf}  {record['value']:g}").classes("value")
                ui.label(record["source"][:44] + (f" · departs in {', '.join(record['extrapolated'])}" if record["extrapolated"] else "")
                         ).classes("label").style(f"text-transform: none; letter-spacing: 0; color: {BAD if record['extrapolated'] else DIM}")
        for name, tool in tools.items():
            on = {"ovs": card.ovs, "shifts": chosen["shifts"], "real": card.real or card.sequence == "HERCULES"}.get(
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
        spectrum.options["title"]["text"] = f"{exam.name}   ·   transient {count} of {exam.card.averages}"
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
                session.overrides[path] = str(await keep(event))
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
                ui.label(f"Settings · {panel or 'everything'}").classes("label")
                ui.space()
                ui.input(placeholder="search, e.g. NAA or t2", on_change=lambda e: build(e.value or "")).props(
                    "dense outlined dark clearable debounce=300").style("width: 260px")
                ui.button("Reset all", icon="restart_alt", on_click=lambda: (session.overrides.clear(), build(), draw())).props(
                    "flat dense no-caps")
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
            ui.label("Every exam draws its values from these lines. Left as they are, they are the population's, each from its source. "
                     "Set one to a number for a subject you decide on, or let it vary between two.").classes("label").style(
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
                    chip.on("drop", lambda _, i=index: drop(i))
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
                            if key in CHOICES:
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
                drawn = session.diagram()
                counts = np.cumsum([0] + [len(BLOCKS[kind][1](settings)) for kind, settings in session.design])
                spans = drawn["spans"][counts[state["at"]]:counts[state["at"] + 1]] if session.design else []
                preview.content = lanes(drawn, width=820.0, lane=44.0, mark=(spans[0][0], spans[-1][1]) if spans else None)
                problem.text = f"{len(session.design)} blocks · the readout opens at {drawn['adc'][0]:.1f} ms"
                problem.style(f"color: {DIM}")
            except Exception as error:  # a half-typed block is not a design yet: say why and keep the last diagram
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")

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
            """The design's spectrum at the planned voxel, eight transients, beside its diagram."""
            busy.set_visibility(True)
            try:
                exam = session.plan(queued=False)
                exam.card = dataclasses.replace(exam.card, averages=8)
                await run.io_bound(session.run, exam)
            except Exception as error:
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")
                return
            finally:
                busy.set_visibility(False)
            outcome.options["series"][0]["data"] = [[float(a), float(b)] for a, b in zip(exam.ppm, exam.frames[-1][1])]
            outcome.update()

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
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props("unelevated dense no-caps")
            with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                preview = ui.html("", sanitize=False).style("flex: 3; min-width: 0")
                with ui.element("div").style("flex: 2; min-width: 0; position: relative"):
                    outcome = ui.echart({**copy.deepcopy(SPECTRUM), "title": {**SPECTRUM["title"], "text": "the spectrum this design gives"}}).style(
                        "height: 340px").classes("w-full")
                    busy = ui.spinner(size="56px", color=SAND).style("position: absolute; top: 42%; left: 46%")
                    busy.set_visibility(False)
            ui.label("The design, in the order it plays · press a block to choose it · drag one onto another to move it there").classes("label")
            strip = ui.row().classes("w-full").style("gap: 4px")
            with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch; flex: 1"):
                with ui.column().classes("pane").style("width: 430px"):
                    ui.label("Add after the chosen block").classes("label")
                    for kind in BLOCKS:
                        with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                            ui.button(kind, icon=BLOCK_ICONS[kind], on_click=lambda _, kind=kind: place(kind, dict(BLOCKS[kind][0]))).props(
                                "outline dense no-caps align=left").style("width: 190px")
                            ui.label(BLOCK_NOTES[kind].split(".")[0]).classes("label").style("text-transform: none; letter-spacing: 0")
                inspector = ui.column().classes("pane").style("flex: 1; min-width: 0")
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
                    grid = exam.maps[name]
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
                    one.options["series"][0]["data"] = [[float(p), float(v)] for p, v in zip(exam.ppm, exam.cells[i, j])]
                    one.options["title"]["text"] = f"cell {i}, {j}"
                    one.update()

                with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                    with ui.column().style("width: 420px"):
                        which = ui.select(list(exam.maps), value="NAA", label="map · click a cell").props("dense outlined dark").classes("w-full")
                        image = ui.echart(painted("NAA"), on_point_click=cell).style("height: 400px").classes("w-full")
                        which.on_value_change(lambda e: (image.options.clear(), image.options.update(painted(e.value)), image.update()))
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
                    ui.label("Over the transients · what the scanner and the subject did").classes("label")
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
        """The room's maps: the B0 field uniform, measured, or the operator's own, their B1+, and
        their own anatomy as three probability maps; all NIfTI, in one world frame."""
        async def field(event):
            session.field = str(await keep(event))
            chosen_field.set_options(["uniform", *FieldMap.PRESETS, session.field], value=session.field)

        async def transmit(event):
            session.transmit = str(await keep(event))
            shown.text = event.file.name

        async def tissue(event, which):
            parts[which] = await keep(event)

        def shelve():
            if len(parts) < 3 or not name.value:
                ui.notify("A name and the three maps, GM, WM, CSF")
                return
            session.anatomy(name.value, parts["GM"], parts["WM"], parts["CSF"])
            pickers["subject"].set_options(list(session.subjects), value=name.value)
            dialog.close()

        parts = {}
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 760px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Maps of the room · NIfTI, world millimetres").classes("label")
                ui.space()
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props("unelevated dense no-caps")
            ui.label("B0 field, offsets in Hz").classes("label")
            with ui.row().classes("w-full items-center no-wrap"):
                chosen_field = ui.select(list(dict.fromkeys(["uniform", *FieldMap.PRESETS, session.field])), value=session.field,
                                         on_change=lambda e: setattr(session, "field", e.value)).props("dense outlined dark").style("flex: 1")
                ui.upload(label="your own", auto_upload=True, on_upload=field).props("flat dense dark accept=.nii,.gz").style("width: 220px")
            ui.switch("this map was measured after my own shim: apply none", value=session.shim == "none", on_change=lambda e: setattr(
                session, "shim", "none" if e.value else "linear")).props("dense color=accent")
            ui.label("B1+, relative to nominal").classes("label")
            with ui.row().classes("w-full items-center no-wrap"):
                shown = ui.label("nominal everywhere" if session.transmit is None else pathlib.Path(session.transmit).name).classes(
                    "value").style("flex: 1")
                ui.button("nominal", on_click=lambda: (setattr(session, "transmit", None), setattr(shown, "text", "nominal everywhere"))).props(
                    "flat dense no-caps")
                ui.upload(label="your own", auto_upload=True, on_upload=transmit).props("flat dense dark accept=.nii,.gz").style("width: 220px")
            ui.label("Your own anatomy · probability maps of one grid, a segmentation of your T1-weighted image").classes("label")
            with ui.row().classes("w-full items-center no-wrap"):
                name = ui.input("name").props("dense outlined dark").style("width: 170px")
                for which in ("GM", "WM", "CSF"):
                    ui.upload(label=which, auto_upload=True, on_upload=lambda e, which=which: tissue(e, which)).props(
                        "flat dense dark accept=.nii,.gz").style("width: 150px")
                ui.button("On the shelf", icon="person_add", on_click=shelve).props("outline dense no-caps")
            ui.label("T1, T2, concentrations and every other parameter take a map at their line in the settings (the gears).").classes(
                "label").style("text-transform: none; letter-spacing: 0")
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
                ui.label("The run · what the scanner and the subject do over the transients").classes("label")
                ui.space()
                ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
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
                     "where it then lies, under the shim of the planned place. Review shows the tracks of every exam. Phase cycling is "
                     "not played: the engine keeps the wanted coherence pathway by crushers and filters.").classes("label").style(
                "text-transform: none; letter-spacing: 0")
        dialog.open()

    def calibration():
        """What the scanner is calibrated by: the records in force with their sources, and, where
        the package has records of the same hardware read under another coil, the choice of them."""
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
                with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                    ui.label(f"{leaf}  {record['value']:g}").classes("value").style("width: 200px")
                    ui.label(record["source"] + (f" · departs in {', '.join(record['extrapolated'])}" if record["extrapolated"] else "")).classes(
                        "label").style("text-transform: none; letter-spacing: 0")
            ui.label("Records read from your own scans (ScannerProfile.read in Python, saved as JSON to ~/MRSandbox/profiles or "
                     "./data/profiles) show here under their scanner, on this machine only. Reading them here from an uploaded scan is "
                     "not in the console yet.").classes("label").style("text-transform: none; letter-spacing: 0")
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
            ui.label("The loops show in the 3D view. Their sensitivities and noise follow from their geometry and the subject's load.").classes(
                "label").style("text-transform: none; letter-spacing: 0")
            ui.button("Done", on_click=dialog.close).props("unelevated dense no-caps")
        dialog.open()

    def browse():
        """The protocol browser: what can go on the card, by family, and the operator's saved protocols."""
        def pick(name):
            session.protocol.sequence = name
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
                            ui.button(name, on_click=lambda _, name=name: pick(name)).props("outline dense no-caps align=left").classes("w-full")
                with ui.column().style("gap: 4px; flex: 1"):
                    ui.label("Saved").classes("label")
                    for name in session.protocols():
                        ui.button(name, icon="bookmark", on_click=lambda _, name=name: (
                            session.load_protocol(name), dialog.close(), draw())).props("outline dense no-caps align=left").classes("w-full")
        dialog.open()

    def drag(plane):
        """A plane's mouse: a press or a drag moves the voxel, the release settles the card."""
        def handle(event):
            kind = event.args.get("mouse_event_type")
            if kind == "mouseup":
                draw()
            elif kind == "mousedown" or event.args.get("buttons") == 1:
                session.place(plane, event.args.get("image_x", 0.0), event.args.get("image_y", 0.0))
                paint()
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
                                                   ("coil", "receive array", list(session.COILS), 215),
                                                   ("subject", "on the table", list(session.subjects), 240)):
            pickers[attribute] = ui.select(options, value=getattr(session, attribute), label=caption, on_change=changed(attribute)).props(
                "dense outlined dark").style(f"min-width: {width}px")
            if attribute == "scanner":
                ui.button(icon="verified", on_click=calibration).mark("calibration").props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip("what this scanner is calibrated by")
            if attribute in GEARS or attribute == "coil":
                ui.button(icon="settings", on_click=(lambda _, panel=attribute: gear(panel)) if attribute in GEARS else coil_gear).props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip(f"the {attribute}'s parameters and ranges")
        ui.button(icon="person_add_alt", on_click=lambda: (setattr(session, "seed", int(np.random.default_rng().integers(2 ** 31))),
                                                           ui.notify("A new subject of the population lies on the table"))).props(
            "flat dense round size=sm").style(f"color: {GLOW}; margin-left: -10px").tooltip(
            "every exam scans the same subject, one draw of the population; this puts a new one on the table")
        ui.button("maps", icon="layers", on_click=maps).props("flat dense no-caps").style(f"color: {GLOW}; margin-left: -8px").tooltip(
            "your own anatomy, B0 and B1+ maps, or the measured ones")
        ui.space()
        ui.button("all settings", icon="tune", on_click=lambda: gear()).props("flat dense no-caps").style(f"color: {GLOW}")
        status = ui.label("idle").classes("label")

    with ui.row().classes("w-full no-wrap").style("gap: 6px; align-items: stretch; flex: 1; min-height: 0"):
        # the exam cards
        with ui.column().classes("pane").style("width: 250px"):
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
                ui.button("the run", icon="timeline", on_click=run_settings).props("flat dense no-caps stack").classes("tool").tooltip(
                    "what happens over the transients: drift, breathing, jitter, motion")
                ui.select(list(OVERLAYS), value=session.overlay, on_change=lambda e: (setattr(session, "overlay", e.value), paint())).props(
                    "dense outlined dark options-dense").style("width: 150px").tooltip(
                    "wash a map over the anatomy: the B0 field and B1+ (on in maps), or a tissue")
                texts["window"] = ui.label("").classes("label").style("text-transform: none; letter-spacing: 0")
                ui.space()
                ui.label("scroll: leaf through · press or drag: place the voxel").classes("label")
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
                        for letter, place in zip(EDGES[name][:4], ("top: 4px; left: 50%", "bottom: 4px; left: 50%",
                                                                   "top: 48%; left: 6px", "top: 48%; right: 6px")):
                            ui.label(letter).classes("tag").style(place)
            with ui.row().classes("w-full no-wrap").style("gap: 6px; flex: 1; min-height: 0; align-items: stretch"):
                # the card's parameters by tab, what follows from them beside
                with ui.column().classes("pane").style("width: 590px; flex: none"):
                    card = session.protocol
                    with ui.tabs().props("dense align=left").style(f"color: {DIM}") as tabs:
                        for name in ("Geometry", "Contrast", "Acquisition", "Imaging"):
                            ui.tab(name)
                    with ui.row().classes("w-full no-wrap").style("gap: 12px; flex: 1; min-height: 0"):
                        with ui.tab_panels(tabs, value="Geometry").props("keep-alive").style("width: 215px; flex: none; overflow-y: auto"):
                            with ui.tab_panel("Geometry"):
                                for axis, caption in enumerate(("x, to the right (mm)", "y, to the front (mm)", "z, to the head (mm)")):
                                    places.append(ui.number(caption, value=session.position_mm[axis], step=1.0, on_change=lambda e, axis=axis: (
                                        e.value is not None and abs(e.value - session.position_mm[axis]) > 0.05 and (
                                            session.position_mm.__setitem__(axis, float(e.value)), draw()))).props(
                                        "dense outlined dark").classes("w-full"))
                                ui.label("voxel size (mm)").classes("label")
                                with ui.row().classes("w-full no-wrap").style("gap: 4px"):
                                    for axis, caption in enumerate(("x", "y", "z")):
                                        edges.append(ui.number(caption, value=session.size_mm[axis], step=5.0, min=3.0, on_change=lambda e, axis=axis: (
                                            e.value is not None and e.value != session.size_mm[axis] and (
                                                session.size_mm.__setitem__(axis, float(e.value)), room(), draw()))).props(
                                            "dense outlined dark").style("flex: 1; min-width: 0"))
                                ui.select(list(FieldMap.SHIMS), value=session.shim, label="shim", on_change=changed("shim")).props(
                                    "dense outlined dark").classes("w-full")
                                ui.number("shim box (mm)", value=session.shim_mm, step=5.0, min=5.0,
                                          on_change=lambda e: (setattr(session, "shim_mm", e.value), draw())).props(
                                    "dense outlined dark clearable").classes("w-full").tooltip(
                                    "the yellow box about the voxel the shim is fitted over; empty, the shim is fitted over the voxel itself")
                            with ui.tab_panel("Imaging"):
                                ui.label("A spectroscopic image (MRSI PRESS). Its volume selection is ideal; its encoding is as set here.").classes(
                                    "label").style("text-transform: none; letter-spacing: 0; white-space: normal")
                                ui.number("matrix", value=card.matrix, step=2, min=2, on_change=changed("matrix", "card")).bind_value(
                                    card, "matrix").props("dense outlined dark").classes("w-full")
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
                                                                 ("water_reference", "water reference transients", 2)))):
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
                                f"flex: 1; border: 1px solid {LINE}; border-radius: 2px; padding: 8px 10px; background: {GROUND}"):
                            ui.label("Overview").classes("label").style(f"grid-column: 1 / -1; color: {SAND}")
                            for key, caption in (("time", "scan duration"), ("timing", "TR / TE"), ("volume", "voxel volume"), ("snr", "relative SNR"),
                                                 ("tissue", "tissue"), ("displaced", "displaced voxels"), ("lipid", "lipid layer"), ("linewidth", "line width from B0"), ("resolution", "spectral resolution"), ("readout", "readout duration"),
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
                                    ui.button("design", icon="architecture", on_click=lambda: (
                                        setattr(session.protocol, "sequence", "Custom"), designer())).props("flat dense no-caps stack").classes("tool").tooltip(
                                        "the sequence designer")
                                diagram = ui.html("", sanitize=False).style("flex: 1; min-width: 0")
                        with ui.tab_panel("Live"):
                            with ui.row().classes("w-full items-center no-wrap"):
                                looks = ui.select([], label="show", on_change=lambda: chosen["exam"] is not None and chosen["exam"].frames
                                                  and frame(chosen["exam"], -1)).props("dense outlined dark options-dense").style("width: 200px")
                                ui.label("wheel: zoom · drag: pan · as displayed: 2 Hz apodization, zero filled, ONE zero-order phase for the "
                                         "exam, no transient aligned before averaging; grey: the transient that just came in").classes(
                                    "label").style("text-transform: none; letter-spacing: 0")
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
        with ui.column().classes("pane").style("width: 290px"):
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
                view = ui.scene(width=272, height=272, grid=False, background_color="#000000")
                view.move_camera(x=6.5, y=-6.5, z=4.5, look_at_x=0, look_at_y=0, look_at_z=0)
                ui.label("the room · drag to turn").classes("tag").style("top: 4px; left: 6px")
            ui.label("Calibration records in force").classes("label").style("margin-top: 6px")
            with ui.scroll_area().style("flex: 1; width: 100%"):
                records = ui.column().classes("w-full").style("gap: 1px")
    room()
    draw()


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
