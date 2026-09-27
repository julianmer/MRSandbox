####################################################################################################
#                                           test_app.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gate of console.app, the console's face: a simulated operator (nicegui.testing) opens   #
#          the page and every pop-up, and sets what they offer, without a browser.                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import asyncio
import importlib

import pytest

pytest.importorskip("nicegui")


#**************************************************************************************************#
#                                          Class TestApp                                           #
#**************************************************************************************************#
@pytest.mark.render
class TestApp:

    def test_an_operator_opens_every_pop_up(self, tmp_path):
        import nibabel as nib
        import numpy as np
        from nicegui import ui

        from mrsandbox import Brain
        from nicegui.testing import user_simulation

        async def visit():
            async with user_simulation() as user:
                import mrsandbox.console.app as app

                importlib.reload(app)                                   # the page registers with this simulation
                await user.open("/")
                await user.should_see("Exam cards", retries=200)          # behind the loading screen the room is made
                assert all(callable(handler) for keyboard in user.find(kind=ui.keyboard).elements for handler in keyboard._key_handlers)   # the shortcuts' handler is a function, not a shadowed name
                user.find(marker="all settings").click()
                await user.should_see("sources:")
                for fold in user.find(kind=ui.expansion).elements:      # a group fills when it opens
                    fold.value = True
                await user.should_see("literature")                       # a chip on an opened line
                next(e for e in user.find(kind=ui.select).elements                  # a leaf that can vary (a tissue's or a model's cannot)
                     if e.value == "literature" and "vary" in e.options).value = "vary"
                await user.should_see("to")
                user.find("Done").click()
                user.find(marker="design").click()
                await user.should_see("the readout opens at")
                user.find("crusher").click()                            # a block from the palette, after the chosen one
                await user.should_see("16 blocks")                      # the card's real PRESS came over, 15 blocks, and one more
                await user.should_see("1 · fixed pulse 90°")                # its slice-selective 90, kept as the card plays it
                user.find("all pulses ideal").click()
                user.find("another, a copy of this one").click()        # a second sub-acquisition, and how the two combine
                await user.should_see("remove this one")
                user.find("difference").click()
                await user.should_see("The sub-acquisitions take turns")
                user.find("Use this design").click()                    # the card plays it from here on, not before
                user.find(marker="subject settings").click()             # the anatomy's wheel: its card, the maps lines on top
                await user.should_see("Anatomy and maps")
                # a uniform field from the stored ones, and an anatomy of the operator's own: a file chosen on a line waits
                # there (the line then SHOWING it is no new choice: that once dropped the file), and one Apply at the end
                # makes the anatomy of the three tissue maps and puts the field in force
                brain, chosen = Brain.load("icbm152"), {}
                for line, data in (("anatomy", brain.wm), ("GM", brain.gm), ("WM", brain.wm), ("CSF", brain.csf)):
                    chosen[line] = tmp_path / f"{line}.nii.gz"
                    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), brain.affine), chosen[line])
                next(iter(user.find(marker="line B0").elements)).value = "uniform"
                for line in ("anatomy", "GM", "WM", "CSF"):             # by their markers: a simulated page keeps no order
                    app.choose_file = lambda prompt, line=line: chosen[line]
                    next(iter(user.find(marker=f"line {line}").elements)).value = "your own file ..."
                    await asyncio.sleep(0.4)                            # the line's choice is taken in a task of its own
                next(e for e in user.find(kind=ui.input).elements if "new anatomy" in str(e.props.get("label", ""))).value = "my own"
                user.find(marker="apply the maps").click()              # by its marker: the settings have an Apply of their own
                await user.should_see("uniform field, none", retries=100)
                await user.should_see("In force: my own, B0 uniform, B1+ nominal")     # the pop-up stays and says what holds now
                assert any(getattr(e, "value", None) == "my own" for e in user.find(kind=ui.select).elements)
                user.find("New scan").click()
                await user.should_see("Single voxel")
                user.find(marker="protocol PRESS").click()                              # real pulses are the default: the displaced voxels in words
                await user.should_see("water -")                              # the displaced voxels in mm
                user.find(kind=ui.button, content="real pulses").click()   # by kind: a click takes the first match, and a tooltip names it too
                await user.should_see("none, the localization is ideal")
                user.find("switches").click()                            # realism and the run as one list of switches
                await user.should_see("on is real, off is ideal or still")
                await user.should_see("breathing (Hz)")
                user.find("Done").click()
                user.find(marker="scanner settings").click()            # the scanner card: one list of lines
                await user.should_see("your scanner file")
                await user.should_see("one of")                          # a pooled leaf: one measured subject per draw
                await user.should_see("your setup")                      # the departures, redrawn after every change
                source = next(e for e in user.find(kind=ui.select).elements if e.value == "literature" and "pooled" in e.options)
                source.value = "pooled"                                  # the line width read on the package's cohorts in place of the stated range
                await user.should_see("your scans")
                await user.should_see("save the scanner as a file")
                user.find("Done").click()
                user.find(marker="coil settings").click()                # the receive array card
                await user.should_see("share of a channel's noise that is its own")
                await user.should_see("sensitivity maps")
                nova = "GE · 32-ch Head (Nova 3832016)"                   # a scanner's own array, as published (item 246)
                [e for e in user.find(kind=ui.select).elements if nova in e.options][-1].value = nova          # the card's select, not the top bar's
                await user.should_see("Nova Medical's user manual")
                await user.should_see("the mean of the two pitches")     # what the package stated
                await user.should_see("against this array")              # the pooled correlation turned into this array's share
                [e for e in user.find(kind=ui.select).elements if nova in e.options][-1].value = "one ideal channel"
                user.find("Done").click()
                user.find(marker="subject settings").click()             # the anatomy card: the maps lines, then the values in groups
                await user.should_see("Anatomy and maps")
                await user.should_see("for all lines")
                user.find("Close").click()
                user.find(marker="scanner settings").click()
                next(iter(user.find(marker="dataset").elements)).value = "big_gaba_trio"        # a dataset's subject on the floor
                await user.should_see("Big GABA · Siemens Trio 3 T · PRESS · 36 subjects")
                await user.should_see("subject S")
                next(iter(user.find(marker="dataset").elements)).value = ""
                await user.should_see("generic, at the field set")                              # "none" put the room back
                user.find("Done").click()
                user.find("New scan").click()
                user.find(marker="protocol STEAM").click()
                next(e for e in user.find(kind=ui.number).elements if e.props.get("label") == "averages").value = 4
                user.find("Accept").click()
                await user.should_see("2 STEAM TE 30")                 # the water reference runs first, the default
                user.find("Start").click()
                await user.should_see("simulated in", retries=400)
                await user.should_see("4 / 4", retries=300)                # after the reference, which runs first
                # the display's processing acts on the exam that ran: shown anew, every kept transient grey behind the average
                user.find(marker="level as acquired").click()
                await asyncio.sleep(1.0)
                live = next(e for e in user.find(kind=ui.echart).elements if "transient 4 of 4" in e.options["title"]["text"])
                assert len(live.options["series"]) == 1 + 4 and live.options["yAxis"]["max"] > live.options["yAxis"]["min"]
                user.find("Adjustments").click()                        # the shim and the prescan have a tab of their own
                await user.should_see("centre the frequency before every exam")
                user.find("Export raw transients").click()              # the exam that just ran, into a folder of the test's own
                await user.should_see("Export · 2 STEAM TE 30")
                next(e for e in user.find(kind=ui.input).elements if e.props.get("label") == "folder").value = str(tmp_path)
                user.find(marker="write the export").click()
                await user.should_see("Written:")
                assert list(tmp_path.glob("*.nii.gz"))
                user.find("Close").click()
                # a spectroscopic image comes with one transient per point of k-space, and its cells are there to be pressed
                user.find("New scan").click()
                user.find(marker="protocol MRSI PRESS").click()
                await user.should_see("phase encoding takes")                # (the card's one transient shows in a bound field, stale here)
                user.find("New scan").click()
                user.find(marker="protocol MEGA-PRESS").click()          # an edited scan with the transients its difference needs
                await user.should_see("10:40")
                user.find("New scan").click()
                user.find(marker="protocol sLASER").click()
                user.find("Accept").click()
                user.find(marker="card 4 sLASER TE 30").click()             # a planned card opens for editing, in its place
                await user.should_see("Accept changes")
                next(e for e in user.find(kind=ui.number).elements if e.props.get("label") == "TE (ms)").value = 40
                user.find("Accept changes").click()
                await user.should_see("4 sLASER TE 40")
                await user.should_not_see("Accept changes")

        asyncio.run(visit())
