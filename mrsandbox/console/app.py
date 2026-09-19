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
import sys
import time

import numpy as np
from nicegui import run, ui
from PIL import Image, ImageFilter

from mrsandbox.console.session import BLOCKS, EDITING, GEARS, MRSI, SEQUENCES, SUPPRESSION, TEMPLATES, Session
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
.port .tag {{ position: absolute; color: {SAND}; font-size: 11px; pointer-events: none; z-index: 2; text-shadow: 0 0 3px #000; }}
.exam {{ border-left: 3px solid transparent; border-bottom: 1px solid {LINE}; padding: 6px 8px; cursor: pointer; display: flex;
         gap: 8px; align-items: center; width: 100%; }}
.exam:hover {{ background: #26211d; }}
.exam.chosen {{ border-left-color: {SAND}; background: #2b241e; color: #fff; }}
.tool {{ border: 1px solid {LINE}; border-radius: 2px; color: {DIM} !important; min-height: 30px; padding: 0 8px; font-size: 11px; }}
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
BLOCK_ICONS = {"hard pulse": "bolt", "own pulse": "gesture", "shaped pulse": "waves", "delay": "hourglass_empty",
               "crusher": "signal_cellular_alt", "coherence filter": "filter_alt", "water suppression": "water_drop"}
BLOCK_NOTES = {
    "hard pulse": "An ideal pulse of no duration. It turns every spin alike, whatever its frequency.",
    "own pulse": "A pulse of your own shape. Type its envelope or load it from a file; the flip angle scales it.",
    "shaped pulse": "A sinc, Gaussian or Hermite pulse, played in time. With a slice gradient it selects a slab along an axis.",
    "delay": "Free evolution. The echo time is the sum of the delays and the pulses between excitation and readout.",
    "crusher": "A gradient pulse that dephases what is transverse. A pair about a 180 keeps the echo and spoils the rest.",
    "coherence filter": "An ideal crusher. It keeps the one coherence order given and removes every other, without time.",
    "water suppression": "A train of selective pulses on the water line before the excitation: VAPOR, WET or CHESS.",
}
CAPTIONS = {"flip_deg": "flip angle (°)", "phase_deg": "phase (°)", "duration_ms": "duration (ms)", "offset_ppm": "centre (ppm)",
            "shape": "shape", "slice_axis": "slice along", "slice_hz_per_mm": "slice gradient (Hz/mm)", "ms": "duration (ms)",
            "axis": "along", "cycles_per_mm": "dephasing (cycles/mm)", "order": "coherence order kept", "scheme": "scheme",
            "envelope": "envelope, amplitudes apart by commas"}
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
def lanes(drawn, width=580.0, lane=62.0, mark=None):
    """A pulse sequence diagram (Session.diagram) drawn the way the literature draws one
    (Landheer et al. 2020, Figure 1), as SVG: the RF with its lobes and flips, the three
    gradients with slice selection open and crushers filled, the signal, the intervals
    between the pulses on top. A prelude that outlasts the localization (a suppression
    train) folds into a box, the time axis broken there. `mark` (start, end) in ms
    lights a span up, the designer's chosen block."""
    names = ("RF", "Gx", "Gy", "Gz", "Signal")
    left, right, top, fold = 56.0, 14.0, 26.0, 86.0
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

    def hidden(t):   # an ideal pulse's bar reaches 0.15 ms before its time (Session.diagram)
        return any(a - 0.2 <= t < b - 0.2 for a, b in folds)

    def path(trace, middle, reach):
        """A trace as path data about its baseline; what a fold hides breaks the path, and every
        piece starts and ends on the baseline so that its fill closes along it."""
        parts, last = [], None
        for t, value in trace:
            if hidden(t):
                if last is not None:
                    parts.append(f"L{last:.1f},{middle:.1f}")
                last = None
                continue
            parts.append((f"M{x(t):.1f},{middle:.1f} " if last is None else "") + f"L{x(t):.1f},{middle - reach * value:.1f}")
            last = x(t)
        return " ".join(parts) + (f" L{last:.1f},{middle:.1f}" if last is not None else "")

    middles = [top + lane * (k + 0.5) for k in range(5)]
    height = top + 5 * lane + 24.0
    peak = max([abs(v) for axis in "xyz" for _, v in drawn[axis]] + [1e-9])
    tall = max([abs(v) for _, v in drawn["rf"]] + [1e-9])
    out = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" style="display: block; max-height: 100%" '
           f'font-family="inherit" font-size="10">']
    if mark is not None:
        out.append(f'<rect x="{x(mark[0]) - 4:.1f}" y="{top - 4}" width="{x(mark[1]) - x(mark[0]) + 8:.1f}" height="{5 * lane + 8}" rx="3" '
                   f'fill="{GLOW}" fill-opacity="0.1" stroke="{GLOW}" stroke-opacity="0.5" />')
    for name, middle in zip(names, middles):
        out.append(f'<text x="6" y="{middle + 3:.1f}" fill="{DIM}">{name}</text>'
                   f'<line x1="{left}" x2="{width - right}" y1="{middle:.1f}" y2="{middle:.1f}" stroke="{LINE}" />')
    for (a, b, name) in drawn["groups"]:
        out.append(f'<rect x="{x(a) + 3:.1f}" y="{top + 3}" width="{fold - 6}" height="{lane - 6}" rx="2" fill="{PANEL}" '
                   f'stroke="{GLOW}" stroke-dasharray="3 2" />'
                   f'<text x="{x(a) + fold / 2:.1f}" y="{middles[0] - 1:.1f}" fill="{GLOW}" text-anchor="middle">{name}</text>'
                   f'<text x="{x(a) + fold / 2:.1f}" y="{middles[0] + 11:.1f}" fill="{DIM}" text-anchor="middle" font-size="9">'
                   f'{b - a:.0f} ms</text>'
                   f'<rect x="{x(a):.1f}" y="{top + lane}" width="{fold}" height="{4 * lane}" fill="{GLOW}" opacity="0.04" />')
    out.append(f'<path d="{path(drawn["rf"], middles[0], 0.42 * lane / tall)}" fill="{GLOW}" fill-opacity="0.2" stroke="{GLOW}" '
               f'stroke-width="1.2" stroke-linejoin="round" />')
    for k, axis in enumerate("xyz"):
        traced = path(drawn[axis], middles[k + 1], 0.42 * lane / peak)
        spans = [(a, b) for a, b, which in drawn["crushers"] if which == axis and not hidden(0.5 * (a + b))]
        out.append(f'<clipPath id="crush{axis}">' + "".join(
            f'<rect x="{x(a):.1f}" y="{middles[k + 1] - lane / 2}" width="{x(b) - x(a):.1f}" height="{lane}" />' for a, b in spans)
            + f'</clipPath><path d="{traced}" fill="{SAND}" fill-opacity="0.08" stroke="{SAND}" stroke-width="1.1" />'
              f'<path d="{traced}" fill="{COPPER}" stroke="{SAND}" stroke-width="1.1" clip-path="url(#crush{axis})" />')
    for t, _ in drawn["marks"]:
        if not hidden(t):
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top + 2}" y2="{top + 8}" stroke="{DIM}" />')
    # the pulses' flips below the RF lane, the intervals between their centres on top
    centres, last = [t for t, _ in drawn["pulses"] if not hidden(t)] + [start], -1e9
    for t, text in drawn["pulses"]:
        if not hidden(t) and x(t) - last > 24.0:
            out.append(f'<text x="{x(t):.1f}" y="{middles[0] + lane / 2 + 1:.1f}" fill="{DIM}" text-anchor="middle" font-size="9">{text}</text>')
            last = x(t)
    for a, b in zip(centres[:-1], centres[1:]):
        out.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="12" y2="12" stroke="{DIM}" />'
                   + "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="16" stroke="{DIM}" />' for t in (a, b))
                   + (f'<text x="{0.5 * (x(a) + x(b)):.1f}" y="9" fill="{INK}" text-anchor="middle" font-size="9">{b - a:.1f}</text>'
                      if x(b) - x(a) > 34.0 else ""))
    out.append(f'<text x="{0.5 * (x(start) + x(stop)):.1f}" y="15" fill="{GOOD}" text-anchor="middle">DAQ</text>')
    wave = [(start + (stop - start) * u, np.exp(-3.5 * u) * np.cos(2.0 * np.pi * 11.0 * u)) for u in np.linspace(0.0, 1.0, 160)]
    out.append(f'<path d="{path(wave, middles[4], 0.42 * lane)}" fill="none" stroke="{GOOD}" stroke-width="1.1" />')
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000) if stop / s <= 12)
    for t in np.arange(0.0, stop + 1e-9, step):
        if not hidden(t):
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{height - 22}" y2="{height - 18}" stroke="{DIM}" />'
                       f'<text x="{x(t):.1f}" y="{height - 7}" fill="{DIM}" text-anchor="middle" font-size="9">{t:g}</text>')
    out.append(f'<text x="6" y="{height - 7}" fill="{DIM}" font-size="9">ms</text></svg>')
    return "".join(out)


# the spectrum display's chart, the console's and the designer's
SPECTRUM = {
    "backgroundColor": "transparent", "animation": False,
    "title": {"text": "no scan yet", "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
    "grid": {"left": 12, "right": 12, "top": 28, "bottom": 28},
    "xAxis": {"type": "value", "inverse": True, "min": 0.2, "max": 4.2, "name": "ppm",
              "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}},
    "yAxis": {"type": "value", "show": False, "scale": True},
    "series": [{"type": "line", "showSymbol": False, "data": [], "lineStyle": {"color": SAND, "width": 1.3},
                "areaStyle": {"color": COPPER, "opacity": 0.12}}],
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
    views, texts, tools, pickers, places, chosen = {}, {}, {}, {}, [], {"exam": None, "box": None, "stop": False, "shifts": True, "snr": None}

    def room():
        """Rebuild the 3D view: the subject's surface, the receive loops, the voxel's box."""
        drawn = session.scene()
        view.clear()
        with view:
            cloud = drawn["cloud"] * MM
            shade = 0.45 + 0.55 * (cloud[:, 2] - cloud[:, 2].min()) / max(float(np.ptp(cloud[:, 2])), 1e-6)
            ui.scene.point_cloud(cloud.tolist(), (shade[:, None] * np.array([0.89, 0.65, 0.37])).tolist(), point_size=0.035)
            for centre, normal, radius in drawn["loops"]:
                n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
                a = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
                a = a / np.linalg.norm(a)
                ui.scene.ring(0.92 * radius * MM, radius * MM, 40).material(COPPER, 0.85).move(
                    *(np.asarray(centre) * MM)).rotate_R(np.stack([a, np.cross(n, a), n], axis=1).tolist())
            centre, size = drawn["voxel"]
            chosen["box"] = ui.scene.box(size * MM, size * MM, size * MM, wireframe=True).material(GLOW).move(
                *(np.asarray(centre) * MM))

    def paint():
        """Redraw the planes (the slice on view, the voxel's outline where the slice cuts it, the
        other planes' cross, a spectroscopic image's grid) and the 3D box: what a drag needs."""
        cells = int(session.protocol.matrix) if session.protocol.sequence == MRSI else 1
        for name, (image, (left, top, width, height), cut, (column, row), shifted) in session.planes().items():
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
        texts["name"].text = card.sequence
        texts["time"].text = texts["time2"].text = f"{minutes:02d}:{seconds:02d}"
        texts["timing"].text = f"{card.tr_ms:g} / {card.te_ms:g} ms"
        texts["volume"].text = f"{session.size_mm ** 3 / 1000.0:.1f} mL"
        # against the card as it was taken from the browser: the voxel's volume and the root of the averages alone
        signal = session.size_mm ** 3 * np.sqrt(1 if card.sequence == MRSI else card.averages)
        chosen["snr"] = signal if chosen["snr"] is None else chosen["snr"]
        texts["snr"].text = f"{100.0 * signal / chosen['snr']:.0f} %  (volume, averages)"
        texts["displaced"].text = "\n".join(
            f"{which} " + " ".join(f"{v:+.1f}" for v in moved) + " mm" for which, moved in look["displaced"].items()) if any(
            any(moved) for moved in look["displaced"].values()) else "none, the localization is ideal"
        texts["linewidth"].text = ("uniform field, none" if look["linewidth_hz"] is None else
                                   f"{look['linewidth_hz']:.1f} Hz on a 1 Hz line, {session.shim} shim")
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
            on = card.ovs if name == "ovs" else chosen["shifts"] if name == "shifts" else card.water_suppression != "off"
            tool.classes(add="on" if on else "", remove="" if on else "on")
        tools["water"].text = "water " + card.water_suppression

    def listing():
        """Redraw the exam cards."""
        rows.clear()
        with rows:
            for exam in session.queue:
                minutes, seconds = divmod(int(round(exam.card.scan_time_s)), 60)
                with ui.element("div").classes("exam" + (" chosen" if exam is chosen["exam"] else "")).on(
                        "click", lambda _, exam=exam: choose(exam)):
                    ui.icon({"planned": "radio_button_unchecked", "running": "autorenew", "done": "check_circle"}[exam.status],
                            size="16px").style(f"color: {dict(planned=DIM, running=SAND, done=GOOD)[exam.status]}")
                    ui.label(exam.name).style("flex: 1")
                    ui.label(f"{minutes:02d}:{seconds:02d}").classes("value").style(f"color: {DIM}")

    def frame(exam, index):
        count, real = exam.frames[index]
        spectrum.options["series"][0]["data"] = [[float(p), float(s)] for p, s in zip(exam.ppm, real)]
        spectrum.options["title"]["text"] = f"{exam.name}   ·   transient {count} of {exam.card.averages}"
        spectrum.update()
        progress.value = count / max(exam.card.averages, 1)
        texts["count"].text = f"{count} / {exam.card.averages}"

    def choose(exam):
        """Select an exam: its voxel returns to the planes, its spectrum to the display."""
        chosen["exam"] = exam
        session.position_mm, session.view_mm, session.size_mm = list(exam.position_mm), list(exam.position_mm), exam.size_mm
        if exam.frames:
            frame(exam, -1)
        listing()
        room()
        draw()

    def plan():
        chosen["exam"] = session.plan()
        listing()

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
            await run.io_bound(session.run, exam)
            texts["took"].text = f"{exam.name}: simulated in {time.perf_counter() - began:.1f} s, a scanner takes {exam.card.scan_time_s:.0f} s"
            progress.props(remove="indeterminate")
            for index in range(len(exam.frames)):
                frame(exam, index)
                await asyncio.sleep(0.03)
            listing()
        status.text = "idle"
        start.enable()

    def export():
        exam = chosen["exam"]
        if exam is None or exam.data is None:
            ui.notify("Select an exam that has run")
            return
        ui.notify(f"Exported {session.export(exam, pathlib.Path.home() / 'MRSandbox' / 'exports')}")

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
                    def fill(event, paths=paths, done=[]):
                        if event.value and not done:
                            done.append(True)
                            with event.sender:
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
                    ui.button(f"{index + 1}  {kind}  {brief}", icon=BLOCK_ICONS[kind], on_click=lambda _, i=index: (
                        state.__setitem__("at", i), refresh())).props("flat dense no-caps").classes(
                        "tool" + (" on" if index == state["at"] else ""))
            inspector.clear()
            with inspector:
                if session.design:
                    kind, settings = session.design[state["at"]]
                    with ui.row().classes("w-full items-center"):
                        ui.icon(BLOCK_ICONS[kind], size="20px").style(f"color: {GLOW}")
                        ui.label(f"Block {state['at'] + 1} · {kind}").style(f"color: {SAND}")
                        ui.space()
                        for icon, tip, act in (("arrow_back", "earlier", lambda: move(-1)), ("arrow_forward", "later", lambda: move(+1)),
                                               ("content_copy", "duplicate", lambda: place(kind, dict(settings))),
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
                preview.content = lanes(drawn, width=900.0, lane=50.0, mark=(spans[0][0], spans[-1][1]) if spans else None)
                problem.text = f"{len(session.design)} blocks · the readout opens at {drawn['adc'][0]:.1f} ms"
                problem.style(f"color: {DIM}")
            except Exception as error:  # a half-typed block is not a design yet: say why and keep the last diagram
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")

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
            try:
                exam = session.plan(queued=False)
                exam.card = dataclasses.replace(exam.card, averages=8)
                await run.io_bound(session.run, exam, 0)
            except Exception as error:
                problem.text = str(error)[:160]
                problem.style(f"color: {BAD}")
                return
            outcome.options["series"][0]["data"] = [[float(a), float(b)] for a, b in zip(exam.ppm, exam.frames[-1][1])]
            outcome.update()

        with ui.dialog().props("maximized") as dialog, ui.card().classes("pane"):
            with ui.row().classes("w-full items-center"):
                ui.label("Sequence designer").style(f"color: {SAND}; font-size: 15px")
                ui.select(list(TEMPLATES), label="start from", on_change=lambda e: e.value and (
                    setattr(session, "design", [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in TEMPLATES[e.value]]),
                    state.__setitem__("at", 0), refresh())).props("dense outlined dark").style("width: 300px")
                problem = ui.label("").classes("value")
                ui.space()
                ui.button("Spectrum of this design", icon="insights", on_click=look).props("outline dense no-caps")
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props("unelevated dense no-caps")
            with ui.row().classes("w-full no-wrap").style("gap: 10px"):
                preview = ui.html("", sanitize=False).style("flex: 3; min-width: 0")
                outcome = ui.echart({**copy.deepcopy(SPECTRUM), "title": {**SPECTRUM["title"], "text": "the spectrum this design gives"}}).style(
                    "flex: 2; min-width: 0; height: 300px")
            ui.label("The design, in the order it plays · press a block to choose it").classes("label")
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
                       "xAxis": {**axis, "inverse": True, "min": 0.2, "max": 4.2, "name": "ppm"},
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
            if attribute in ("subject", "coil", "size_mm"):
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
                for family, names in (("Single voxel", list(SEQUENCES)), ("Edited", list(EDITING)),
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
            if event.type == "mouseup":
                draw()
            elif event.type == "mousedown" or event.buttons == 1:
                session.place(plane, event.image_x, event.image_y)
                paint()
        return handle

    def toggle(name):
        """The toolbar's switches: the water suppression steps through its schemes, OVS is on or off."""
        card = session.protocol
        if name == "ovs":
            card.ovs = not card.ovs
        else:
            schemes = ["off"] + list(SUPPRESSION)
            card.water_suppression = schemes[(schemes.index(card.water_suppression) + 1) % len(schemes)]
        draw()

    # the top bar: what is in the room
    with ui.row().classes("w-full items-center pane no-wrap").style("gap: 14px; padding: 4px 10px"):
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" height="40" style="display: block; height: 40px" src="{MARK}">', sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 16px; letter-spacing: .1em")
        for attribute, caption, options, width in (("scanner", "scanner", {name: pathlib.Path(name).stem for name in session.scanners()}, 230),
                                                   ("coil", "receive array", list(session.COILS), 215),
                                                   ("subject", "on the table", list(session.subjects), 240)):
            pickers[attribute] = ui.select(options, value=getattr(session, attribute), label=caption, on_change=changed(attribute)).props(
                "dense outlined dark").style(f"min-width: {width}px")
            if attribute in GEARS:
                ui.button(icon="settings", on_click=lambda _, panel=attribute: gear(panel)).props("flat dense round size=sm").style(
                    f"color: {GLOW}; margin-left: -10px").tooltip(f"the {attribute}'s parameters and ranges")
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
                    setattr(session, "view_mm", list(session.position_mm)), paint())).props("flat dense no-caps").classes("tool").tooltip(
                    "bring the three planes back to the voxel")
                tools["water"] = ui.button("water off", icon="water_drop", on_click=lambda: toggle("water")).props(
                    "flat dense no-caps").classes("tool").tooltip("water suppression before the localization: off, VAPOR, WET, CHESS")
                tools["ovs"] = ui.button("OVS", icon="border_outer", on_click=lambda: toggle("ovs")).props(
                    "flat dense no-caps").classes("tool").tooltip("outer-volume suppression: six saturation slabs about the voxel")
                tools["shifts"] = ui.button("shifts", icon="flip_to_back", on_click=lambda: (
                    chosen.__setitem__("shifts", not chosen["shifts"]), draw())).props("flat dense no-caps").classes("tool on").tooltip(
                    "the displaced voxels of water (blue) and lipid (pink) under a played localization")
                ui.button("design", icon="architecture", on_click=lambda: (
                    setattr(session.protocol, "sequence", "Custom"), designer())).props("flat dense no-caps").classes("tool").tooltip(
                    "the sequence designer")
                ui.button("sequence", icon="settings", on_click=lambda: gear("scanner")).props("flat dense no-caps").classes("tool")
                ui.space()
                ui.label("scroll a plane to leaf through it · press or drag to place the voxel · drag the 3D view to turn it").classes("label")
            with ui.element("div").classes("w-full").style("display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px"):
                for name in ("tra", "sag", "cor"):
                    with ui.element("div").classes("port"):
                        views[name] = ui.interactive_image(on_mouse=drag(name), events=["mousedown", "mousemove", "mouseup"],
                                                           cross=False).classes("w-full")
                        views[name].on("wheel.prevent", lambda e, plane=name: (
                            session.scroll(plane, 1 if e.args["deltaY"] < 0 else -1), paint()), args=["deltaY"], throttle=0.04)
                        texts[name] = ui.label("").classes("tag").style("top: 4px; left: 6px")
                        for letter, place in zip(EDGES[name][:4], ("top: 4px; left: 50%", "bottom: 4px; left: 50%",
                                                                   "top: 48%; left: 6px", "top: 48%; right: 6px")):
                            ui.label(letter).classes("tag").style(place)
                with ui.element("div").classes("port").style("aspect-ratio: 1"):
                    view = ui.scene(width=400, height=400, grid=False, background_color="#000000").style("width: 100%; height: 100%")
                    view.move_camera(x=6.5, y=-6.5, z=4.5, look_at_x=0, look_at_y=0, look_at_z=0)
                    ui.label("3D").classes("tag").style("top: 4px; left: 6px")
            with ui.row().classes("w-full no-wrap").style("gap: 6px; flex: 1; min-height: 0; align-items: stretch"):
                # the card's parameters by tab, what follows from them beside
                with ui.column().classes("pane").style("width: 590px; flex: none"):
                    card = session.protocol
                    with ui.tabs().props("dense align=left").style(f"color: {DIM}") as tabs:
                        for name in ("Geometry", "Contrast", "Acquisition"):
                            ui.tab(name)
                    with ui.row().classes("w-full no-wrap").style("gap: 12px; flex: 1"):
                        with ui.tab_panels(tabs, value="Geometry").props("keep-alive").style("width: 190px; flex: none"):
                            with ui.tab_panel("Geometry"):
                                for axis, caption in enumerate(("x, to the right (mm)", "y, to the front (mm)", "z, to the head (mm)")):
                                    places.append(ui.number(caption, value=session.position_mm[axis], step=1.0, on_change=lambda e, axis=axis: (
                                        e.value is not None and abs(e.value - session.position_mm[axis]) > 0.05 and (
                                            session.position_mm.__setitem__(axis, float(e.value)), draw()))).props(
                                        "dense outlined dark").classes("w-full"))
                                ui.number("voxel size (mm)", value=session.size_mm, step=5.0, min=5.0, on_change=changed("size_mm")).props(
                                    "dense outlined dark").classes("w-full")
                                ui.select(list(FieldMap.SHIMS), value=session.shim, label="shim", on_change=changed("shim")).props(
                                    "dense outlined dark").classes("w-full")
                                ui.number("shim box (mm), empty: the voxel", value=session.shim_mm, step=5.0, min=5.0,
                                          on_change=lambda e: (setattr(session, "shim_mm", e.value), draw())).props(
                                    "dense outlined dark clearable").classes("w-full").tooltip(
                                    "the yellow box about the voxel the shim is fitted over; it acts where a B0 map is on (maps)")
                                ui.number("matrix (MRSI)", value=card.matrix, step=2, on_change=changed("matrix", "card")).bind_value(
                                    card, "matrix").props("dense outlined dark").classes("w-full")
                            for tab, fields in (("Contrast", (("te_ms", "TE (ms)", 1.0), ("tr_ms", "TR (ms)", 100.0))),
                                                ("Acquisition", (("averages", "averages", 1), ("npoints", "points", 512),
                                                                 ("bandwidth_hz", "bandwidth (Hz)", 500.0)))):
                                with ui.tab_panel(tab):
                                    if tab == "Contrast":
                                        ui.button(on_click=browse).bind_text_from(card, "sequence").props(
                                            "outline no-caps icon-right=expand_more").classes("w-full")
                                    for attribute, caption, step in fields:
                                        ui.number(caption, value=getattr(card, attribute), step=step, on_change=changed(attribute, "card")
                                                  ).bind_value(card, attribute).props("dense outlined dark").classes("w-full")
                        with ui.element("div").classes("info").style("flex: 1"):
                            for key, caption in (("time", "scan duration"), ("timing", "TR / TE"), ("volume", "voxel volume"), ("snr", "relative SNR"),
                                                 ("tissue", "tissue"), ("displaced", "displaced voxels"), ("linewidth", "line width from B0"), ("resolution", "spectral resolution"), ("readout", "readout duration"),
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
                            diagram = ui.html("", sanitize=False).classes("w-full")
                        with ui.tab_panel("Live"):
                            spectrum = ui.echart(copy.deepcopy(SPECTRUM)).style("height: 250px").classes("w-full")
            # the card in one line, and its way into the queue
            with ui.row().classes("w-full items-center pane no-wrap").style("gap: 18px; padding: 4px 10px"):
                texts["name"] = ui.label("").style(f"color: {SAND}")
                ui.label("duration").classes("label")
                texts["time2"] = ui.label("").classes("value")
                texts["verdict"] = ui.label("").classes("value").style("flex: 1")
                ui.button("Accept", icon="playlist_add", on_click=plan).props("unelevated dense no-caps")

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
