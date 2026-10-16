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

                importlib.reload(app)  # the page registers with this simulation
                await user.open("/")
                # behind the loading screen the room is made
                await user.should_see("Exam cards", retries=600)
                # the shortcuts' handler is a function, not a shadowed name
                assert all(callable(handler) for keyboard in user.find(kind=ui.keyboard).elements
                           for handler in keyboard._key_handlers)
                user.find(marker="all settings").click()
                await user.should_see("sources:")
                # the population beside the lines, every subject at first
                await user.should_see("subjects in")
                for fold in user.find(kind=ui.expansion).elements:  # a group fills when it opens
                    fold.value = True
                await user.should_see("literature")                       # a chip on an opened line
                # a leaf that can vary (a tissue's or a model's cannot)
                next(e for e in user.find(kind=ui.select).elements
                     if e.value == "literature" and "vary" in e.options).value = "vary"
                await user.should_see("to")
                user.find("Done").click()
                # the card's OVS on, and its slabs one by one: the x+ slab off
                user.find(marker="ovs").click()
                user.find("Geometry").click()
                await user.should_see(marker="slabs")  # shown once the binding catches up
                user.find(marker="slabs").click()
                await user.should_see("OVS slabs")
                user.find(marker="slab x+").click()
                assert user.find(marker="slab x+").elements.pop().value is False
                # the x- slab tilted toward y: the whole voxel at once, its compute time costed
                user.find(marker="tilt x- y").elements.pop().value = 20
                await user.should_see("computing")
                user.find("Close").click()
                # the voxel turned about z, and square again for the exams below
                turn = user.find(marker="turn z").elements.pop()
                turn.value = 30
                await user.should_see(marker="turn z")
                turn.value = 0
                user.find(marker="ovs").click()  # off again: the exams below play no slabs
                user.find(marker="design").click()
                await user.should_see("the readout opens at")
                user.find("crusher").click()  # a block from the palette, after the chosen one
                # the card's real PRESS came over, 15 blocks, and one more
                await user.should_see("16 blocks")
                # its slice-selective 90, kept as the card plays it
                await user.should_see("1 · card's pulse 90°")
                # a second sub-acquisition, and how the two combine
                user.find("another, a copy of this one").click()
                await user.should_see("remove this one")
                user.find("difference").click()
                await user.should_see("The sub-acquisitions take turns")
                user.find("Use this design").click()  # the card plays it from here on, not before
                # the anatomy's wheel: its card, the maps lines on top
                user.find(marker="subject settings").click()
                await user.should_see("Anatomy and maps")
                # a uniform field from the stored ones, and an anatomy of the operator's own: a file
                # chosen on a line waits there (the line then showing it is no new choice), and one
                # Apply at the end makes the anatomy of the three tissue maps and puts the field in
                # force
                brain, chosen = Brain.load("icbm152"), {}
                for line, data in (("anatomy", brain.wm), ("GM", brain.gm), ("WM", brain.wm),
                                   ("CSF", brain.csf)):
                    chosen[line] = tmp_path / f"{line}.nii.gz"
                    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), brain.affine),
                             chosen[line])
                next(iter(user.find(marker="line B0").elements)).value = "uniform"
                # by their markers: a simulated page keeps no order
                for line in ("anatomy", "GM", "WM", "CSF"):
                    app.choose_file = lambda prompt, line=line: chosen[line]
                    next(iter(user.find(marker=f"line {line}").elements)).value = (
                        "your own file ...")
                    await asyncio.sleep(0.4)  # the line's choice is taken in a task of its own
                next(e for e in user.find(kind=ui.input).elements
                     if "new anatomy" in str(e.props.get("label", ""))).value = "my own"
                # by its marker: the settings have an Apply of their own
                user.find(marker="apply the maps").click()
                await user.should_see("uniform field, none", retries=100)
                # the pop-up stays and says what holds now
                await user.should_see("In force: my own, B0 uniform, B1+ nominal")
                assert any(getattr(e, "value", None) == "my own"
                           for e in user.find(kind=ui.select).elements)
                user.find("New scan").click()
                await user.should_see("Single voxel")
                # real pulses are the default: the displaced voxels in words
                user.find(marker="protocol PRESS").click()
                await user.should_see("water -")  # the displaced voxels in mm
                # by kind: a click takes the first match, and a tooltip names it too
                user.find(kind=ui.button, content="real pulses").click()
                await user.should_see("none, the localization is ideal")
                user.find("switches").click()  # realism and the run as one list of switches
                await user.should_see("on is real, off is ideal or still")
                await user.should_see("breathing (Hz)")
                user.find("Done").click()
                user.find(marker="scanner settings").click()  # the scanner card: one list of lines
                await user.should_see("your scanner file")
                await user.should_see("one of")  # a pooled leaf: one measured subject per draw
                await user.should_see("your setup")  # the departures, redrawn after every change
                source = next(e for e in user.find(kind=ui.select).elements
                              if e.value == "literature" and "pooled" in e.options)
                # the line width read on the package's cohorts in place of the stated range
                source.value = "pooled"
                await user.should_see("your scans")
                await user.should_see("save the scanner as a file")
                user.find("Done").click()
                user.find(marker="coil settings").click()                # the receive array card
                await user.should_see("share of a channel's noise that is its own")
                await user.should_see("sensitivity maps")
                nova = "GE · 32-ch Head (Nova 3832016)"  # a scanner's own array, as published
                # the card's select, not the top bar's (labelled; the found elements are a set)
                next(e for e in user.find(kind=ui.select).elements
                     if nova in e.options and e.props.get("label") != "receive array").value = nova
                await user.should_see("Nova Medical's user manual")
                await user.should_see("the mean of the two pitches")     # what the package stated
                # the pooled correlation turned into this array's share
                await user.should_see("against this array")
                [e for e in user.find(kind=ui.select).elements
                 if nova in e.options][-1].value = "one ideal channel"
                user.find("Done").click()
                # the anatomy card: the maps lines, then the values in groups
                user.find(marker="subject settings").click()
                await user.should_see("Anatomy and maps")
                await user.should_see("for all lines")
                user.find("Close").click()
                user.find(marker="scanner settings").click()
                # a dataset's subject on the floor
                next(iter(user.find(marker="dataset").elements)).value = "big_gaba_trio"
                await user.should_see("Big GABA · Siemens Trio 3 T · PRESS · 36 subjects")
                await user.should_see("subject S")
                next(iter(user.find(marker="dataset").elements)).value = ""
                await user.should_see("generic, at the field set")  # "none" put the room back
                user.find("Done").click()
                user.find("New scan").click()
                user.find(marker="protocol STEAM").click()
                next(e for e in user.find(kind=ui.number).elements
                     if e.props.get("label") == "averages").value = 4
                user.find("Accept").click()
                # the water reference runs first, the default
                await user.should_see("2 STEAM TE 30")
                user.find("Start").click()
                await user.should_see("simulated in", retries=400)
                await user.should_see("4 / 4", retries=300)  # after the reference, which runs first
                # the display's processing acts on the exam that ran: shown anew, every kept
                # transient grey behind the average
                user.find(marker="level as acquired").click()
                await asyncio.sleep(1.0)
                live = next(e for e in user.find(kind=ui.echart).elements
                            if "transient 4 of 4" in e.options.get("title", {}).get("text", ""))
                assert (len(live.options["series"]) == 1 + 4
                        and live.options["yAxis"]["max"] > live.options["yAxis"]["min"])
                # the shim and the prescan have a tab of their own
                user.find("Adjustments").click()
                await user.should_see("centre the frequency before every exam")
                # the exam that just ran, into a folder of the test's own
                user.find("Export raw transients").click()
                await user.should_see("Export · 2 STEAM TE 30")
                next(e for e in user.find(kind=ui.input).elements
                     if e.props.get("label") == "folder").value = str(tmp_path)
                user.find(marker="write the export").click()
                await user.should_see("Written:")
                assert list(tmp_path.glob("*.nii.gz"))
                user.find("Close").click()
                # a spectroscopic image comes with one transient per point of k-space, and its cells
                # are there to be pressed
                user.find("New scan").click()
                user.find(marker="protocol MRSI PRESS").click()
                # the card's one transient shows in a bound field, stale here
                await user.should_see("phase encoding takes")
                user.find("New scan").click()
                # an edited scan with the transients its difference needs
                user.find(marker="protocol MEGA-PRESS").click()
                await user.should_see("10:40")
                user.find("New scan").click()
                user.find(marker="protocol sLASER").click()
                user.find("Accept").click()
                # a planned card opens for editing, in its place
                user.find(marker="card 4 sLASER TE 30").click()
                await user.should_see("Accept changes")
                next(e for e in user.find(kind=ui.number).elements
                     if e.props.get("label") == "TE (ms)").value = 40
                user.find("Accept changes").click()
                await user.should_see("4 sLASER TE 40")
                await user.should_not_see("Accept changes")
                # the database to browse: every reading, narrowed by a facet and by a query
                user.find(marker="database").click()
                await user.should_see("Measured datasets")
                # these rows as the room's population: every pooled value comes from them
                user.find(marker="population adopt").click()
                await user.should_see("pooled values come from every dataset; a healthy adult")
                user.find(marker="facet dataset").elements.pop().value = ["big_gaba_trio"]
                await user.should_see("of 36 subjects in 1 datasets", retries=20)
                user.find(marker="database query").elements.pop().value = "leaf == 'noise.sigma'"
                await user.should_see("readings of 35 subjects", retries=20)
                grid = user.find(marker="database grid").elements.pop()
                assert {row["leaf"] for row in grid.options["rowData"]} == {"noise.sigma"}
                user.find("Literature").click()
                user.find(marker="literature table").elements.pop().value = "relaxation · CORBIN"
                await user.should_see("Whole-brain T2 mapping", retries=20)
                # one leaf against age, the literature and this room's pool beside
                user.find("Parameters").click()
                user.find(marker="parameter leaf").elements.pop().value = "water.t2_s.wm"
                await user.should_see("state no age", retries=20)
                names = [s["name"] for s in
                         user.find(marker="parameter chart").elements.pop().options["series"]]
                assert "literature" in names and "this room's pool" in names

        asyncio.run(visit())
