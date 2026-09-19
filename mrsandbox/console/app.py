####################################################################################################
#                                              app.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual scanner console: a NiceGUI face over console.session. The room (scanner,    #
#          coil, subject) in the top bar, planning on three planes and a 3D view, protocol cards   #
#          in an exam queue, the spectrum building up over the transients, the records in force,   #
#          export of the raw transients. Usage:                                                    #
#          uvx --from "mrsandbox[console]" mrsandbox-console                                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import asyncio
import base64
import io
import os
import pathlib

import numpy as np
from nicegui import run, ui
from PIL import Image

from mrsandbox.console.session import BLOCKS, SEQUENCES, SUBJECTS, Session

#*************#
#   palette   #
#*************#
# the logo's: sand and copper on charcoal
SAND, COPPER, GLOW, GROUND, PANEL, LINE, INK, DIM, GOOD, BAD = (
    "#E2A55F", "#B5622C", "#F5C77E", "#151210", "#1E1A17", "#3A322B", "#EADFD2", "#8E8378", "#7FA66B", "#D9653B")
STYLE = f"""
body {{ background: {GROUND}; color: {INK}; font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; }}
.pane {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 3px; padding: 10px; gap: 8px; }}
.label {{ color: {DIM}; letter-spacing: .12em; text-transform: uppercase; font-size: 11px; }}
.value {{ color: {INK}; font-variant-numeric: tabular-nums; }}
.plane {{ border: 1px solid {LINE}; background: #000; }}
.exam {{ border: 1px solid {LINE}; padding: 5px 8px; cursor: pointer; display: flex; justify-content: space-between; width: 100%; }}
.exam.chosen {{ border-color: {SAND}; color: #fff; }}
.q-field__native, .q-field__label, .q-select__dropdown-icon {{ color: {INK} !important; }}
.q-field--outlined .q-field__control:before {{ border-color: {LINE} !important; }}
"""
MM = 1.0 / 40.0  # the 3D view's units per millimetre
CHOICES = {"shape": ["sinc", "gaussian", "hermite"], "slice_axis": ["none", "x", "y", "z"], "axis": ["x", "y", "z"],
           "scheme": ["VAPOR", "WET", "CHESS"]}


#*************#
#   diagram   #
#*************#
def lanes(drawn):
    """The chart of a pulse sequence diagram (Session.diagram): RF, the three gradients, and the
    readout as five lanes over one time axis in ms."""
    names = ("RF", "Gx", "Gy", "Gz", "ADC")
    start, stop = drawn["adc"]
    traces = (drawn["rf"], drawn["x"], drawn["y"], drawn["z"], [(0.0, 0.0), (start, 0.0), (start, 1.0), (stop, 1.0), (stop, 0.0)])
    colours = (GLOW, SAND, SAND, SAND, GOOD)
    grids = [{"left": 44, "right": 14, "top": f"{6 + 18.5 * k}%", "height": "14%"} for k in range(5)]
    return {
        "backgroundColor": "transparent", "animation": False, "grid": grids,
        "xAxis": [{"type": "value", "gridIndex": k, "min": 0, "max": round(stop, 1), "show": k == 4, "name": "ms" if k == 4 else "",
                   "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}} for k in range(5)],
        "yAxis": [{"type": "value", "gridIndex": k, "name": names[k], "nameLocation": "middle", "nameRotate": 0, "nameGap": 24,
                   "nameTextStyle": {"color": DIM, "fontSize": 10}, "axisLabel": {"show": False}, "axisTick": {"show": False},
                   "axisLine": {"show": False}, "splitLine": {"show": False}, "scale": False} for k in range(5)],
        "series": [{"type": "line", "xAxisIndex": k, "yAxisIndex": k, "showSymbol": False, "data": [list(p) for p in traces[k]],
                    "lineStyle": {"color": colours[k], "width": 1.2}, "areaStyle": {"color": colours[k], "opacity": 0.18},
                    **({"markLine": {"symbol": "none", "silent": True, "lineStyle": {"color": DIM, "type": "dotted"},
                                     "label": {"show": False}, "data": [{"xAxis": t} for t, _ in drawn["marks"]]}} if k == 0 else {})}
                   for k in range(5)],
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
    views, texts, chosen = {}, {}, {"exam": None, "box": None}

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

    def draw():
        """Redraw the planes with the voxel's outline, the 3D box, and the card's outlook."""
        for name, (image, (left, top, width, height)) in session.planes().items():
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            views[name].set_source("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            views[name].content = (f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
                                   f'fill="none" stroke="{GLOW}" stroke-width="1" />')
        chosen["box"].move(*(np.asarray(session.position_mm) * MM))
        diagram.options.clear()
        diagram.options.update(lanes(session.diagram()))
        diagram.update()
        look = session.outlook()
        texts["position"].text = "  ".join(f"{v:+.0f}" for v in session.position_mm) + " mm"
        texts["tissue"].text = f"GM {look['gm']:.2f}  WM {look['wm']:.2f}  CSF {look['csf']:.2f}"
        minutes, seconds = divmod(int(round(look["scan_time_s"])), 60)
        texts["time"].text = f"{minutes}:{seconds:02d}"
        texts["limits"].text = "B1 {:g} uT · G {:g} mT/m · slew {:g} T/m/s".format(*look["limits"])
        texts["verdict"].text = "plays on this scanner" if not look["violations"] else "; ".join(look["violations"])
        texts["verdict"].style(f"color: {GOOD if not look['violations'] else BAD}")
        texts["records"].text = "records   " + "   ·   ".join(
            f"{leaf} {record['value']:g} ({record['source'][:28]})"
            + (f" departs in {', '.join(record['extrapolated'])}" if record["extrapolated"] else "")
            for leaf, record in look["records"].items())

    def listing():
        """Redraw the exam queue."""
        rows.clear()
        with rows:
            for exam in session.queue:
                minutes, seconds = divmod(int(round(exam.card.scan_time_s)), 60)
                with ui.element("div").classes("exam" + (" chosen" if exam is chosen["exam"] else "")).on(
                        "click", lambda _, exam=exam: choose(exam)):
                    ui.label(exam.name)
                    ui.label({"planned": f"{minutes}:{seconds:02d}", "running": "scanning", "done": "done"}[exam.status]).style(
                        f"color: {SAND if exam.status == 'running' else DIM}")

    def frame(exam, index):
        count, real = exam.frames[index]
        spectrum.options["series"][0]["data"] = [[float(p), float(s)] for p, s in zip(exam.ppm, real)]
        spectrum.options["title"]["text"] = f"{exam.name}   ·   transient {count} of {exam.card.averages}"
        spectrum.update()

    def choose(exam):
        """Select an exam: its voxel returns to the planes, its spectrum to the display."""
        chosen["exam"] = exam
        session.position_mm, session.size_mm = list(exam.position_mm), exam.size_mm
        if exam.frames:
            frame(exam, -1)
        listing()
        room()
        draw()

    def plan():
        chosen["exam"] = session.plan()
        listing()

    async def scan_queue():
        """Run every planned exam in order; each shows as its transients come in."""
        button.disable()
        for exam in [e for e in session.queue if e.status == "planned"]:
            chosen["exam"], exam.status = exam, "running"
            status.text = f"scanning {exam.name}"
            listing()
            await run.io_bound(session.run, exam)
            for index in range(len(exam.frames)):
                frame(exam, index)
                await asyncio.sleep(0.03)
            listing()
        status.text = "idle"
        button.enable()

    def export():
        exam = chosen["exam"]
        if exam is None or exam.data is None:
            ui.notify("Select an exam that has run")
            return
        ui.notify(f"Exported {session.export(exam, pathlib.Path.home() / 'MRSandbox' / 'exports')}")

    def gear():
        """The advanced settings: every leaf of the parameter tree with its prior, a number or
        a range open to the operator, the rest shown as it is."""
        def put(path, kind, index):
            def handle(event):
                if event.value is None:
                    return
                now = dict((p, n) for p, k, n, _ in session.leaves() if k == kind).get(path)
                values = list(now if now is not None else (0.0,) * (1 if kind == "fixed" else 2))
                values[index] = float(event.value)
                session.overrides[path] = values[0] if kind == "fixed" else (min(values), max(values))
                draw()
            return handle

        groups = {}
        for path, kind, numbers, touched in session.leaves():
            groups.setdefault(path.split(".")[0], []).append((path, kind, numbers, touched))
        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 720px; max-width: 900px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Advanced settings · the parameter tree").classes("label")
                ui.space()
                ui.button("Reset all to their sources", on_click=lambda: (session.overrides.clear(), dialog.close(), draw())).props(
                    "flat dense")
            with ui.scroll_area().style("height: 560px; width: 100%"):
                for group, rows in groups.items():
                    with ui.expansion(f"{group}   ({len(rows)})").classes("w-full").props("dense dark"):
                        for path, kind, numbers, touched in rows:
                            with ui.row().classes("w-full items-center no-wrap").style("gap: 10px"):
                                ui.label(path).style(f"width: 250px; color: {SAND if touched else INK}")
                                if kind == "other":
                                    ui.label(numbers).classes("label").style("text-transform: none; letter-spacing: 0")
                                else:
                                    for index, number in enumerate(numbers):
                                        ui.number(value=number, format="%.6g", on_change=put(path, kind, index)).props(
                                            "dense outlined dark").style("width: 130px")
                                    ui.label("fixed" if kind == "fixed" else "drawn between").classes("label")
        dialog.open()

    def designer():
        """The sequence designer: the operator's own design as blocks in order, each with its
        settings, and its diagram as it grows."""
        def rebuild():
            listing_of_blocks.clear()
            with listing_of_blocks:
                for index, (kind, settings) in enumerate(session.design):
                    with ui.row().classes("w-full items-center no-wrap").style("gap: 8px"):
                        ui.label(f"{index + 1}").classes("label").style("width: 18px")
                        ui.label(kind).style(f"width: 140px; color: {SAND}")
                        for key, value in settings.items():
                            if key in CHOICES:
                                ui.select(CHOICES[key], value=value, label=key, on_change=lambda e, s=settings, k=key: (
                                    s.__setitem__(k, e.value), refresh())).props("dense outlined dark").style("width: 120px")
                            else:
                                ui.number(key, value=value, format="%.6g", on_change=lambda e, s=settings, k=key: (
                                    e.value is not None and s.__setitem__(k, float(e.value)), refresh())).props(
                                    "dense outlined dark").style("width: 120px")
                        ui.space()
                        ui.button("↑", on_click=lambda _, i=index: move(i, -1)).props("flat dense")
                        ui.button("↓", on_click=lambda _, i=index: move(i, +1)).props("flat dense")
                        ui.button("✕", on_click=lambda _, i=index: (session.design.pop(i), rebuild(), refresh())).props("flat dense")

        def move(index, step):
            other = index + step
            if 0 <= other < len(session.design):
                session.design[index], session.design[other] = session.design[other], session.design[index]
                rebuild()
                refresh()

        def refresh():
            try:
                preview.options.clear()
                preview.options.update(lanes(session.diagram()))
                preview.update()
                problem.text = ""
            except Exception as error:  # a half-typed block is not a design yet: say why and keep the last diagram
                problem.text = str(error)[:140]

        with ui.dialog() as dialog, ui.card().classes("pane").style("min-width: 1100px; max-width: 1400px"):
            with ui.row().classes("w-full items-center"):
                ui.label("Sequence designer · blocks play in order, top to bottom").classes("label")
                ui.space()
                new_kind = ui.select(list(BLOCKS), value="delay").props("dense outlined dark").style("width: 190px")
                ui.button("Add block", on_click=lambda: (session.design.append((new_kind.value, dict(BLOCKS[new_kind.value][0]))),
                                                         rebuild(), refresh())).props("outline dense")
                ui.button("Done", on_click=lambda: (dialog.close(), draw())).props("unelevated dense")
            preview = ui.echart(lanes(session.diagram())).style("height: 300px").classes("w-full")
            problem = ui.label("").style(f"color: {BAD}")
            with ui.scroll_area().style("height: 300px; width: 100%"):
                listing_of_blocks = ui.column().classes("w-full").style("gap: 4px")
            rebuild()
        dialog.open()

    def changed(attribute, target=None):
        """A handler that writes a control's value to the session (or the card) and redraws."""
        def handle(event):
            if event.value is None:
                return
            setattr(session.protocol if target == "card" else session, attribute, event.value)
            if attribute == "subject":
                session.position_mm = [0.0, 0.0, 0.0] if hasattr(session.body(), "contents") else [0.0, -45.0, 35.0]
            if attribute in ("subject", "coil", "size_mm"):
                room()
            draw()
        return handle

    # the status bar: what is in the room
    with ui.row().classes("w-full items-center pane").style("gap: 16px"):
        # the logo, scaled once for the bar (the file is the repository's 1.7 MB original)
        logo = Image.open(pathlib.Path(__file__).with_name("logo.png")).convert("RGB")
        logo = logo.crop((60, 200, 964, 800)).resize((144, 96), Image.LANCZOS)
        buffer = io.BytesIO()
        logo.save(buffer, format="PNG")
        ui.html(f'<img alt="MRSandbox" width="72" height="48" style="border-radius: 2px; display: block" '
                f'src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">')
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 17px; letter-spacing: .08em")
        for attribute, caption, options, width in (("scanner", "scanner", {name: pathlib.Path(name).stem for name in session.scanners()}, 230),
                                                   ("coil", "receive array", list(session.COILS), 215),
                                                   ("subject", "on the table", list(SUBJECTS), 240)):
            ui.select(options, value=getattr(session, attribute), label=caption, on_change=changed(attribute)).props(
                "dense outlined dark").style(f"min-width: {width}px")
        ui.space()
        ui.button("⚙ advanced", on_click=gear).props("flat dense").style(f"color: {GLOW}")
        status = ui.label("idle").classes("label")

    with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch"):
        # the exam queue
        with ui.column().classes("pane").style("width: 240px"):
            ui.label("Exam queue").classes("label")
            rows = ui.column().classes("w-full").style("gap: 4px")
            ui.button("Add card to queue", on_click=plan).props("outline").classes("w-full")
            button = ui.button("Run queue", on_click=scan_queue).props("unelevated").classes("w-full")
            ui.button("Export raw transients", on_click=export).props("outline").classes("w-full")
            ui.label("Outlook").classes("label").style("margin-top: 10px")
            for key, caption in (("position", "voxel at"), ("tissue", "tissue"), ("time", "scan time")):
                ui.label(caption).classes("label")
                texts[key] = ui.label("").classes("value")

        # planning: three planes and the room in 3D, the spectrum below
        with ui.column().classes("pane").style("flex: 1; min-width: 0"):
            ui.label("Planning · click a plane to move the voxel · drag the 3D view to turn it").classes("label")
            with ui.grid(columns=4).classes("w-full").style("gap: 8px"):
                for name, caption in (("tra", "TRA"), ("sag", "SAG"), ("cor", "COR")):
                    with ui.column().style("gap: 2px; min-width: 0"):
                        views[name] = ui.interactive_image(
                            on_mouse=lambda e, plane=name: (session.place(plane, e.image_x, e.image_y), draw()),
                            events=["mousedown"], cross=False).classes("plane w-full")
                        ui.label(caption).classes("label")
                with ui.column().style("gap: 2px; min-width: 0"):
                    view = ui.scene(width=300, height=300, grid=False, background_color="#000000").classes("plane")
                    view.move_camera(x=6.5, y=-6.5, z=4.5, look_at_x=0, look_at_y=0, look_at_z=0)
                    ui.label("3D").classes("label")
            spectrum = ui.echart({
                "backgroundColor": "transparent", "animation": False,
                "title": {"text": "no scan yet", "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
                "grid": {"left": 12, "right": 12, "top": 28, "bottom": 28},
                "xAxis": {"type": "value", "inverse": True, "min": 0.2, "max": 4.2, "name": "ppm",
                          "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}},
                "yAxis": {"type": "value", "show": False, "scale": True},
                "series": [{"type": "line", "showSymbol": False, "data": [], "lineStyle": {"color": SAND, "width": 1.3},
                            "areaStyle": {"color": COPPER, "opacity": 0.12}}],
            }).style("height: 220px").classes("w-full")
            ui.label("Sequence · RF, gradients as this scanner plays them, readout").classes("label")
            diagram = ui.echart(lanes(session.diagram())).style("height: 230px").classes("w-full")

        # the protocol card and the hardware it must play on
        with ui.column().classes("pane").style("width: 270px"):
            ui.label("Protocol card").classes("label")
            card = session.protocol
            ui.select(list(SEQUENCES) + ["Custom"], value=card.sequence, label="sequence", on_change=changed("sequence", "card")).props(
                "dense outlined dark").classes("w-full")
            for attribute, caption, step in (("te_ms", "TE (ms)", 1.0), ("tr_ms", "TR (ms)", 100.0), ("averages", "averages", 1),
                                             ("npoints", "points", 512), ("bandwidth_hz", "bandwidth (Hz)", 500.0)):
                ui.number(caption, value=getattr(card, attribute), step=step, on_change=changed(attribute, "card")).props(
                    "dense outlined dark").classes("w-full")
            ui.number("voxel size (mm)", value=session.size_mm, step=5.0, min=5.0, on_change=changed("size_mm")).props(
                "dense outlined dark").classes("w-full")
            ui.button("Design a sequence", on_click=lambda: (setattr(card, "sequence", "Custom"), designer())).props(
                "outline").classes("w-full")
            ui.label("Hardware limits").classes("label").style("margin-top: 6px")
            texts["limits"] = ui.label("").classes("value")
            texts["verdict"] = ui.label("").classes("value")

    # the records in force and where a scan of this room departs from them
    with ui.row().classes("w-full pane"):
        texts["records"] = ui.label("").classes("label").style("text-transform: none; letter-spacing: .02em")
    room()
    draw()


#**********#
#   main   #
#**********#
def main():
    """Start the console and open it in the browser."""
    ui.run(title="MRSandbox Console", favicon=pathlib.Path(__file__).with_name("logo.png"), dark=True, reload=False,
           show="MRSANDBOX_CONSOLE_HEADLESS" not in os.environ, port=int(os.environ.get("MRSANDBOX_CONSOLE_PORT", "8642")))


if __name__ in {"__main__", "__mp_main__"}:
    main()
