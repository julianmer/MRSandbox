####################################################################################################
#                                              app.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual scanner console: a NiceGUI face over console.session. Stage one, the        #
#          walking skeleton: pick the scanner and the subject, plan the voxel on three planes,     #
#          set a protocol card, run, see the spectrum, export. Usage:                              #
#          uvx --from "mrsandbox[console]" mrsandbox-console                                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import base64
import io
import os
import pathlib

from nicegui import run, ui
from PIL import Image

from mrsandbox.console.session import SEQUENCES, SUBJECTS, Session

#*************#
#   palette   #
#*************#
# the logo's: sand and copper on charcoal
SAND, COPPER, GLOW, GROUND, PANEL, LINE, INK, DIM = (
    "#E2A55F", "#B5622C", "#F5C77E", "#151210", "#1E1A17", "#3A322B", "#EADFD2", "#8E8378")
STYLE = f"""
body {{ background: {GROUND}; color: {INK}; font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 13px; }}
.pane {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 3px; padding: 10px; gap: 8px; }}
.label {{ color: {DIM}; letter-spacing: .12em; text-transform: uppercase; font-size: 11px; }}
.value {{ color: {INK}; font-variant-numeric: tabular-nums; }}
.plane {{ border: 1px solid {LINE}; background: #000; }}
.q-field__native, .q-field__label, .q-select__dropdown-icon {{ color: {INK} !important; }}
.q-field--outlined .q-field__control:before {{ border-color: {LINE} !important; }}
"""


#**********#
#   page   #
#**********#
@ui.page("/")
def console():
    """One operator's console; every browser tab is a session of its own."""
    session = Session()
    ui.add_head_html(f"<style>{STYLE}</style>")
    ui.dark_mode(True)
    ui.colors(primary=COPPER, accent=SAND, positive="#7FA66B", negative="#D9653B")
    views, outlook_labels = {}, {}

    def draw():
        """Redraw the planes with the voxel's outline, the position, and the card's outlook."""
        for name, (image, (left, top, width, height)) in session.planes().items():
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            views[name].set_source("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            views[name].content = (f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
                                   f'fill="none" stroke="{GLOW}" stroke-width="1" />')
        look = session.outlook()
        outlook_labels["position"].text = "  ".join(f"{v:+.0f}" for v in session.position_mm) + " mm"
        outlook_labels["tissue"].text = f"GM {look['gm']:.2f}  WM {look['wm']:.2f}  CSF {look['csf']:.2f}"
        minutes, seconds = divmod(int(round(look["scan_time_s"])), 60)
        outlook_labels["time"].text = f"{minutes}:{seconds:02d}"
        outlook_labels["limits"].text = "plays on this scanner" if not look["violations"] else "; ".join(look["violations"])
        outlook_labels["limits"].style(f"color: {'#7FA66B' if not look['violations'] else '#D9653B'}")

    def show(scan):
        spectrum.options["series"][0]["data"] = [[float(p), float(s)] for p, s in zip(scan["ppm"], scan["spectrum"])]
        spectrum.options["title"]["text"] = scan["name"]
        spectrum.update()

    async def acquire():
        button.disable()
        status.text = "scanning"
        scan = await run.io_bound(session.run)
        queue.options = [s["name"] for s in session.scans]
        queue.value = scan["name"]
        show(scan)
        status.text = "idle"
        button.enable()

    def export():
        chosen = next((s for s in session.scans if s["name"] == queue.value), None)
        if chosen is None:
            ui.notify("Nothing scanned yet")
            return
        ui.notify(f"Exported {session.export(chosen, pathlib.Path.home() / 'MRSandbox' / 'exports')}")

    def changed(attribute, target=None):
        """A handler that writes a control's value to the session (or the card) and redraws."""
        def handle(event):
            if event.value is None:
                return
            setattr(session.protocol if target == "card" else session, attribute, event.value)
            if attribute == "subject":
                body = session.body()
                session.position_mm = [0.0, 0.0, 0.0] if hasattr(body, "contents") else [0.0, -45.0, 35.0]
            draw()
        return handle

    # the status bar: what is in the room
    with ui.row().classes("w-full items-center pane").style("gap: 18px"):
        ui.image(pathlib.Path(__file__).with_name("logo.png")).style("width: 66px; height: 45px; border-radius: 2px")
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 17px; letter-spacing: .08em")
        ui.select(session.scanners(), value=session.scanner, label="scanner", on_change=changed("scanner")).props(
            "dense outlined dark").style("min-width: 210px")
        ui.select(list(SUBJECTS), value=session.subject, label="on the table", on_change=changed("subject")).props(
            "dense outlined dark").style("min-width: 240px")
        ui.space()
        status = ui.label("idle").classes("label")

    with ui.row().classes("w-full no-wrap").style("gap: 10px; align-items: stretch"):
        # the exam queue
        with ui.column().classes("pane").style("width: 230px"):
            ui.label("Exam queue").classes("label")
            queue = ui.select([], label="scans").props("dense outlined dark").classes("w-full")
            queue.on_value_change(lambda e: [show(s) for s in session.scans if s["name"] == e.value])
            button = ui.button("Run", on_click=acquire).props("unelevated").classes("w-full")
            ui.button("Export NIfTI-MRS", on_click=export).props("outline").classes("w-full")
            ui.label("Outlook").classes("label").style("margin-top: 10px")
            for key, caption in (("position", "voxel at"), ("tissue", "tissue"), ("time", "scan time"), ("limits", "hardware")):
                ui.label(caption).classes("label")
                outlook_labels[key] = ui.label("").classes("value")

        # planning: three planes and the spectrum
        with ui.column().classes("pane").style("flex: 1; min-width: 0"):
            ui.label("Planning · click a plane to move the voxel").classes("label")
            with ui.row().classes("w-full no-wrap").style("gap: 8px"):
                for name, caption in (("tra", "TRA"), ("sag", "SAG"), ("cor", "COR")):
                    with ui.column().style("flex: 1; gap: 2px; min-width: 0"):
                        views[name] = ui.interactive_image(
                            on_mouse=lambda e, plane=name: (session.place(plane, e.image_x, e.image_y), draw()),
                            events=["mousedown"], cross=False).classes("plane w-full")
                        ui.label(caption).classes("label")
            spectrum = ui.echart({
                "backgroundColor": "transparent", "animation": False,
                "title": {"text": "no scan yet", "textStyle": {"color": DIM, "fontSize": 11, "fontWeight": "normal"}},
                "grid": {"left": 12, "right": 12, "top": 28, "bottom": 28},
                "xAxis": {"type": "value", "inverse": True, "min": 0.2, "max": 4.2, "name": "ppm",
                          "axisLine": {"lineStyle": {"color": LINE}}, "axisLabel": {"color": DIM}, "splitLine": {"show": False}},
                "yAxis": {"type": "value", "show": False, "scale": True},
                "series": [{"type": "line", "showSymbol": False, "data": [], "lineStyle": {"color": SAND, "width": 1.3},
                            "areaStyle": {"color": COPPER, "opacity": 0.12}}],
            }).style("height: 230px").classes("w-full")

        # the protocol card
        with ui.column().classes("pane").style("width: 270px"):
            ui.label("Protocol card").classes("label")
            card = session.protocol
            ui.select(list(SEQUENCES), value=card.sequence, label="sequence", on_change=changed("sequence", "card")).props(
                "dense outlined dark").classes("w-full")
            for attribute, caption, step in (("te_ms", "TE (ms)", 1.0), ("tr_ms", "TR (ms)", 100.0), ("averages", "averages", 1),
                                             ("npoints", "points", 512), ("bandwidth_hz", "bandwidth (Hz)", 500.0)):
                ui.number(caption, value=getattr(card, attribute), step=step, on_change=changed(attribute, "card")).props(
                    "dense outlined dark").classes("w-full")
            ui.number("voxel size (mm)", value=session.size_mm, step=5.0, min=5.0, on_change=changed("size_mm")).props(
                "dense outlined dark").classes("w-full")
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
