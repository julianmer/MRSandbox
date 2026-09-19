####################################################################################################
#                                         test_session.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gates of console.session, the console's state without its interface: the planning       #
#          planes and the voxel on them, the card's outlook, a scan, an export.                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.console.session import Session


#**************************************************************************************************#
#                                        Class TestSession                                         #
#**************************************************************************************************#
@pytest.mark.render
class TestSession:

    def test_the_planes_are_squares_and_a_click_on_the_voxel_leaves_it_there(self):
        session = Session()
        planes = session.planes()
        assert {image.shape for image, _ in planes.values()} == {(115, 115)}
        assert all(image.max() > 200 and image.dtype == np.uint8 for image, _ in planes.values())
        before = list(session.position_mm)
        for name, (_, (left, top, width, height)) in planes.items():
            session.place(name, left + width / 2.0, top + height / 2.0)
        assert np.allclose(session.position_mm, before, atol=1.0)          # half a 2 mm cell
        height = session.position_mm[2]
        session.place("tra", 80.0, 40.0)                                   # a transverse click leaves the height
        assert session.position_mm[2] == height and session.position_mm[0] != before[0]

    def test_the_card_promises_and_the_scan_delivers(self, tmp_path):
        session = Session()
        session.protocol.averages, session.protocol.npoints = 4, 1024
        look = session.outlook()
        assert look["gm"] + look["wm"] + look["csf"] == pytest.approx(1.0, abs=0.02)
        assert look["scan_time_s"] == 8.0 and look["violations"] == ()
        scan = session.run(seed=1)
        assert scan["name"] == "1 PRESS TE 30" and len(session.scans) == 1
        assert abs(scan["ppm"][scan["spectrum"].argmax()] - 2.01) < 0.03    # phased on NAA, the tallest line
        path = session.export(scan, tmp_path)
        assert path.exists() and path.name == "1_PRESS_TE_30.nii.gz"

    def test_a_phantom_lies_on_the_table_as_a_brain_does(self):
        session = Session()
        session.subject, session.position_mm = "Phantom, SPECTRE sphere", [0.0, 0.0, 0.0]
        assert session.outlook()["gm"] == pytest.approx(1.0)
        assert {image.shape[0] == image.shape[1] for image, _ in session.planes().values()} == {True}
