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

    def test_an_operator_opens_every_pop_up(self):
        from nicegui import ui
        from nicegui.testing import user_simulation

        async def visit():
            async with user_simulation() as user:
                import mrsandbox.console.app as app

                importlib.reload(app)                                   # the page registers with this simulation
                await user.open("/")
                await user.should_see("Exam cards")
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
                await user.should_see("7 blocks")
                user.find("maps").click()
                await user.should_see("B0 field")
                user.find("New scan").click()
                await user.should_see("Single voxel")
                user.find("PRESS, played").click()                      # a card from the browser, its displaced voxels in words
                await user.should_see("water -1.8 -4.4 -4.4 mm")
                user.find("New scan").click()
                user.find("STEAM").click()
                next(e for e in user.find(kind=ui.number).elements if e.props.get("label") == "averages").value = 4
                user.find("Accept").click()
                await user.should_see("1 STEAM TE 30")
                user.find("Start").click()
                await user.should_see("simulated in", retries=100)
                await user.should_see("4 / 4")

        asyncio.run(visit())
