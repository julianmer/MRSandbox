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
                user.find("all settings").click()
                await user.should_see("Every exam draws")
                for fold in user.find(kind=ui.expansion).elements:      # a group fills when it opens
                    fold.value = True
                await user.should_see("as its source gives it")
                next(e for e in user.find(kind=ui.select).elements if e.value == "as its source gives it").value = "vary between"
                await user.should_see("and")
                user.find("Done").click()
                user.find("design").click()
                await user.should_see("the readout opens at")
                user.find("crusher").click()                            # a block from the palette, after the chosen one
                await user.should_see("16 blocks")                      # the card's real PRESS came over, 15 blocks, and one more
                await user.should_see("1 · as the card plays it")       # its slice-selective 90 as the card plays it
                user.find("all pulses ideal").click()
                user.find("another, a copy of this one").click()        # a second sub-acquisition, and how the two combine
                await user.should_see("remove this one")
                user.find("difference").click()
                await user.should_see("The sub-acquisitions take turns")
                user.find("Use this design").click()                    # the card plays it from here on, not before
                user.find(kind=ui.button, content="maps").click()      # a tooltip names the maps too
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
                assert any(getattr(e, "value", None) == "my own" for e in user.find(kind=ui.select).elements)
                user.find("New scan").click()
                await user.should_see("Single voxel")
                user.find(marker="protocol PRESS").click()                              # real pulses are the default: the displaced voxels in words
                await user.should_see("water -1.8 -4.4 -4.4 mm")
                user.find(kind=ui.button, content="real pulses").click()   # by kind: a click takes the first match, and a tooltip names it too
                await user.should_see("none, the localization is ideal")
                user.find("realism").click()
                await user.should_see("on is real, off is ideal")
                user.find("Done").click()
                user.find("New scan").click()
                user.find(marker="protocol STEAM").click()
                next(e for e in user.find(kind=ui.number).elements if e.props.get("label") == "averages").value = 4
                user.find("Accept").click()
                await user.should_see("2 STEAM TE 30")                 # the water reference runs first, the default
                user.find("Start").click()
                await user.should_see("simulated in", retries=400)
                await user.should_see("4 / 4", retries=300)                # after the reference, which runs first
                user.find("Export raw transients").click()              # the exam that just ran, into a folder of the test's own
                await user.should_see("Export · 2 STEAM TE 30")
                next(e for e in user.find(kind=ui.input).elements if e.props.get("label") == "folder").value = str(tmp_path)
                user.find(marker="write the export").click()
                await user.should_see("Written:")
                assert list(tmp_path.glob("*.nii.gz"))
                user.find("Close").click()
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
