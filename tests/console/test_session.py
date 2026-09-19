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

    def test_the_card_promises_and_the_exam_delivers(self, tmp_path):
        session = Session()
        session.protocol.averages, session.protocol.npoints = 6, 1024
        look = session.outlook()
        assert look["gm"] + look["wm"] + look["csf"] == pytest.approx(1.0, abs=0.02)
        assert look["scan_time_s"] == 12.0 and look["violations"] == () and look["limits"] == (25.0, 80.0, 200.0)
        assert look["records"]["noise.sigma"]["source"].startswith("COWS")     # the Prisma's own record
        exam = session.plan()
        assert exam.name == "1 PRESS TE 30" and exam.status == "planned" and session.queue == [exam]
        session.position_mm = [20.0, 0.0, 30.0]                                 # the exam keeps the voxel it was planned with
        session.run(exam, seed=1)
        assert exam.status == "done" and exam.position_mm == (0.0, -45.0, 35.0)
        assert exam.data.shape[-1] == 6                                         # the transients apart, as a scanner exports them
        assert [count for count, _ in exam.frames][-1] == 6 and len(exam.frames) <= 12
        assert abs(exam.ppm[exam.frames[-1][1].argmax()] - 2.01) < 0.03         # phased on NAA, the tallest line
        path = session.export(exam, tmp_path)
        assert path.exists() and path.name == "1_PRESS_TE_30.nii.gz"

    def test_a_phantom_and_an_array_are_in_the_room_as_a_brain_is(self):
        session = Session()
        session.subject, session.position_mm, session.coil = "Phantom, SPECTRE sphere", [0.0, 0.0, 0.0], "head array, 32 channels"
        assert session.outlook()["gm"] == pytest.approx(1.0)
        assert {image.shape[0] == image.shape[1] for image, _ in session.planes().values()} == {True}
        drawn = session.scene(points=500)
        assert drawn["cloud"].shape == (500, 3) and len(drawn["loops"]) == 32 and drawn["voxel"] == ((0.0, 0.0, 0.0), 20.0)
        assert np.allclose(np.linalg.norm(drawn["cloud"], axis=1), 90.0, atol=5.0)  # the sphere's surface

    def test_the_gear_opens_every_leaf_and_a_setting_reaches_the_scan(self):
        session = Session()
        leaves = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves()}
        assert leaves["noise.sigma"] == ("range", (3.7, 6.9)) and leaves["water.temperature_c"] == ("fixed", (37.0,))
        assert leaves["water.t1_s"][0] == "other"                               # a tissue record, shown and not edited
        session.overrides.update({"conc.NAA": 20.0, "noise.sigma": (1.0, 1.5)})
        changed = {path: numbers for path, _, numbers, touched in session.leaves() if touched}
        assert changed == {"conc.NAA": (20.0,), "noise.sigma": (1.0, 1.5)}
        session.protocol.averages, session.protocol.npoints = 2, 512
        exam = session.run(session.plan(), seed=1)
        assert exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]["conc"]["NAA"] == 20.0
