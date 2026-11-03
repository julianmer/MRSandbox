####################################################################################################
#                                              app.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual scanner console in a window of its own: the window opens at once on a       #
#          loading screen while the console's page (console.page, the engine with it) loads behind #
#          it, and the simulations run in a worker process (console.worker). Light on purpose: it  #
#          imports neither. Usage: uvx --from "mrsandbox[console]" mrsandbox-console               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import base64
import importlib
import io
import multiprocessing
import os
import pathlib
import subprocess
import sys

from nicegui import Client, run, ui
from PIL import Image

from mrsandbox.console.worker import WORKER, started, working

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
.pick {{ fill: transparent; cursor: pointer; }} .pick:hover {{ fill: {GLOW}; fill-opacity: 0.12; }}
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
# the logo has one home, the repository's assets/; an installed package carries that file in its
# environment's share/mrsandbox (pyproject.toml, data-files)
LOGO = next((path for path in (pathlib.Path(__file__).parents[2] / "assets" / "logo" / "logo.png",
                               pathlib.Path(sys.prefix) / "share" / "mrsandbox" / "logo.png")
             if path.exists()), None)


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


#**********#
#   page   #
#**********#
@ui.page("/")
async def console(client: Client):
    """One operator's console; every browser tab is a session of its own. The page answers at once
    with the loading screen, and the console's page (console.page: the engine with it, seconds to
    import the first time) loads behind it and makes the room."""
    ui.add_head_html(f"<style>{STYLE}</style>")
    ui.dark_mode(True)
    with ui.element("div").style(f"position: fixed; inset: 0; z-index: 9999; "
                                 f"background: {GROUND}; display: flex; flex-direction: column; "
                                 "align-items: center; justify-content: center; "
                                 "gap: 18px") as loading:
        if MARK is not None:
            ui.html(f'<img alt="MRSandbox" style="height: 150px" src="{MARK_LARGE}">',
                    sanitize=False)
        ui.label("MRSandbox").style(f"color: {SAND}; font-size: 22px; letter-spacing: .14em")
        ui.spinner("bars", size="34px", color=SAND)
    # the loading screen goes out now; the rest follows over the page's connection
    await client.connected()
    page = await run.io_bound(importlib.import_module, "mrsandbox.console.page")
    # the page's own work (the room's outlook, a preview) compiles into the machine's cache too
    working()
    await page.console(client, loading)


#**********#
#   main   #
#**********#
def main():
    """Start the console in a window of its own; with MRSANDBOX_CONSOLE_PORT set it serves a
    browser at that port instead (MRSANDBOX_CONSOLE_HEADLESS: without opening one).

    `mrsandbox-console --install-shortcut` instead puts the console among this computer's
    applications with the logo, started from this Python environment: an app in ~/Applications
    on macOS, a menu entry in ~/.local/share/applications on Linux, a Start menu shortcut on
    Windows. Run it again after moving the environment; an environment that disappears (uvx's)
    leaves a shortcut that starts nothing. `mrsandbox-console --warm` builds the library's
    designs' graphs into this machine's cache in the terminal (what a launch does in the
    background, console.worker warm) and returns."""
    if sys.argv[1:] == ["--warm"]:
        from mrsandbox.console.session import Session
        from mrsandbox.console.worker import warm, working

        working()
        built = warm(Session(), said=print)
        print(f"{len(built)} families built; the rest were in the cache already")
        return
    if sys.argv[1:] == ["--install-shortcut"]:
        name, logo = "MRSandbox Console", Image.open(LOGO)
        if sys.platform == "darwin":
            import plistlib

            bundle = pathlib.Path.home() / "Applications" / f"{name}.app"
            (bundle / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
            (bundle / "Contents" / "Resources").mkdir(exist_ok=True)
            launcher = bundle / "Contents" / "MacOS" / "console"
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m mrsandbox.console.app\n')
            launcher.chmod(0o755)
            logo.save(bundle / "Contents" / "Resources" / "logo.icns")
            with open(bundle / "Contents" / "Info.plist", "wb") as plist:
                plistlib.dump({"CFBundleName": name, "CFBundleDisplayName": name,
                               "CFBundleIdentifier": "org.mrsandbox.console",
                               "CFBundleExecutable": "console", "CFBundleIconFile": "logo.icns",
                               "CFBundlePackageType": "APPL", "NSHighResolutionCapable": True},
                              plist)
            # Finder reads a bundle's icon again once the bundle has changed
            os.utime(bundle)
            print(f"{name} is in {bundle.parent}: open it from Launchpad or Spotlight, or drag it "
                  "to the Dock")
        elif sys.platform == "win32":
            pythonw = pathlib.Path(sys.executable).with_name("pythonw.exe")
            icon = pathlib.Path(os.environ["LOCALAPPDATA"]) / "MRSandbox" / "console.ico"
            icon.parent.mkdir(parents=True, exist_ok=True)
            logo.save(icon, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
            link = (pathlib.Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu"
                    / "Programs" / f"{name}.lnk")
            quoted = lambda path: "'" + str(path).replace("'", "''") + "'"
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
                            f"{quoted(link)}); $s.TargetPath = "
                            f"{quoted(pythonw if pythonw.exists() else sys.executable)}; "
                            "$s.Arguments = '-m mrsandbox.console.app'; "
                            f"$s.IconLocation = {quoted(icon)}; $s.Save()"], check=True)
            print(f"{name} is in the Start menu ({link}); pin it to the taskbar from there")
        else:
            share = pathlib.Path.home() / ".local" / "share"
            icon = share / "icons" / "mrsandbox-console.png"
            icon.parent.mkdir(parents=True, exist_ok=True)
            logo.resize((256, 256)).save(icon)
            entry = share / "applications" / "mrsandbox-console.desktop"
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(f'[Desktop Entry]\nType=Application\nName={name}\n'
                             'Comment=The MRSandbox virtual MR scanner\n'
                             f'Exec="{sys.executable}" -m mrsandbox.console.app\nIcon={icon}\n'
                             'Terminal=false\nCategories=Science;Education;\n')
            print(f"{name} is in the applications menu ({entry})")
        return
    # the simulations in a process of their own, so a compile never holds the page; a child process
    # (the worker, the native window) runs this module again and starts none
    if multiprocessing.parent_process() is None:
        WORKER["pool"] = started()
    if "MRSANDBOX_CONSOLE_PORT" in os.environ:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False,
               show="MRSANDBOX_CONSOLE_HEADLESS" not in os.environ,
               port=int(os.environ["MRSANDBOX_CONSOLE_PORT"]))
    else:
        ui.run(title="MRSandbox Console", favicon=LOGO, dark=True, reload=False, native=True,
               window_size=(1680, 1020))


if __name__ in {"__main__", "__mp_main__"}:
    main()
