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

ICBM = "Brain, ICBM152 average, 2 mm"  # the subject every machine has; the console's own default is placed by hand


#**************************************************************************************************#
#                                        Class TestSession                                         #
#**************************************************************************************************#
@pytest.mark.render
class TestSession:

    def test_the_planes_are_squares_and_a_click_on_the_voxel_leaves_it_there(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        planes = session.planes()
        assert {image.shape for image, *_ in planes.values()} == {(115, 115)}
        assert all(image.max() > 200 and image.dtype == np.uint8 for image, *_ in planes.values())
        before = list(session.position_mm)
        for name, (_, (left, top, width, height), cut, *_) in planes.items():
            assert cut
            session.place(name, left + width / 2.0, top + height / 2.0)
        assert np.allclose(session.position_mm, before, atol=1.0)          # half a 2 mm cell
        height = session.position_mm[2]
        session.place("tra", 80.0, 40.0)                                   # a transverse click leaves the height
        assert session.position_mm[2] == height and session.position_mm[0] != before[0]

    def test_leafing_through_a_plane_leaves_the_voxel_and_a_click_places_it_on_the_slice(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        before = list(session.position_mm)
        session.scroll("tra", 20)
        image, _, cut, *_ = session.planes()["tra"]
        assert session.position_mm == before and session.view_mm[2] > before[2] + 30.0 and not cut
        session.scroll("tra", 10 ** 6)                                     # the grid's last slice, no further
        assert session.view_mm[2] < 200.0
        session.scroll("tra", -10 ** 6)
        session.scroll("tra", 45)
        session.place("tra", 60.0, 60.0)                                   # a click is on the slice on view
        assert session.position_mm[2] == session.view_mm[2] and session.planes()["tra"][2]

    def test_a_played_localization_displaces_water_and_lipid_and_an_ideal_one_does_not(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.real = False                                      # ideal pulses select exactly
        assert session.displacement(4.65) == [0.0, 0.0, 0.0] and session.planes()["tra"][4] == {}
        session.protocol.real = True                                       # the default
        water, lipid = np.asarray(session.displacement(4.65)), np.asarray(session.displacement(1.3))
        assert np.all(water < 0.0) and np.all(lipid > 0.0)                 # either side of the 3 ppm the slabs are centred for
        assert abs(water[1]) > 2.0 * abs(water[0])                         # the narrow 180s displace more than the 90
        assert np.allclose(water / lipid, (4.65 - 3.0) / (1.3 - 3.0))
        _, (left, top, _, _), _, _, shifted, _ = session.planes()["tra"]
        assert shifted["water"][0] < left and shifted["water"][1] > top    # towards lower x and lower y (rows run down)
        session.protocol.sequence = "sLASER"                               # broadband adiabatic pairs: a tenth of it
        assert abs(session.displacement(4.65)[1]) < 0.15 * abs(water[1])

    def test_the_shim_and_its_box_set_the_line_width_a_b0_map_gives(self, tmp_path):
        import nibabel as nib

        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        assert session.field == "bigbrain_mr" and 1.0 < session.linewidth() < 5.0   # the measured field, shimmed, by default
        session.field = "uniform"
        assert session.linewidth() is None                                 # a uniform field broadens nothing
        body = session.body()
        x, y = ((np.arange(n) * np.diag(body.affine)[axis] + body.affine[axis, 3]) for axis, n in enumerate(body.gm.shape[:2]))
        saddle = 0.4 * x[:, None, None] + 0.02 * (x[:, None, None] ** 2 - y[None, :, None] ** 2)   # Hz: a slope and a harmonic saddle
        nib.save(nib.Nifti1Image(np.broadcast_to(saddle, body.gm.shape).astype(np.float32), body.affine), tmp_path / "b0.nii.gz")
        session.field, widths = str(tmp_path / "b0.nii.gz"), {}
        for shim in ("none", "linear", "quadratic"):
            session.shim = shim
            widths[shim] = session.linewidth()
        assert widths["none"] > 3.0 and widths["none"] > 1.5 * widths["linear"] and widths["linear"] > 1.2 * widths["quadratic"]
        assert widths["quadratic"] == pytest.approx(1.0, abs=0.15)         # the saddle is gone, the 1 Hz line stays
        session.shim, session.shim_mm = "linear", 60.0
        assert session.planes()["tra"][4]["shim"][2] == pytest.approx(60.0 / np.diag(body.affine)[0])
        assert session.linewidth() == pytest.approx(widths["linear"], rel=0.05)    # a saddle has no slope: the same fit over a wider box

    def test_one_subject_lies_on_the_table_and_a_water_reference_follows_its_card(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.real, session.protocol.averages, session.protocol.npoints, session.protocol.water_reference = False, 4, 512, 2
        session.size_mm = [30.0, 20.0, 15.0]                                # an anisotropic voxel
        session.run_settings = {"events_per_min": 0.0}
        session.plan()
        assert [exam.name for exam in session.queue] == ["1 PRESS water reference", "2 PRESS TE 30"]   # the reference first, the default
        water, scan = (session.run(exam) for exam in session.queue)
        assert scan.truth == water.truth                                    # the same draw of the population
        assert abs(water.ppm[np.argmax(water.frames[-1][1])] - 4.65) < 0.1 and water.card.averages == 2
        assert water.frames[-1][1].max() > 5.0 * scan.frames[-1][1].max()   # unsuppressed
        again = session.run(session.plan(queued=False))
        session.queue.append(again)
        assert not np.allclose(session.run(again).frames[-1][1], scan.frames[-1][1])   # its own noise at another place in the queue

    def test_the_card_plays_water_suppression_and_ovs_before_the_localization(self):
        from mrsandbox import Vapor

        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        bare = len(session.sequence().events)
        session.protocol.water_suppression, session.protocol.ovs = "VAPOR", True
        assert len(session.sequence().events) == bare + len(Vapor().events) + 12     # six slabs, a filter after each
        drawn = session.diagram()
        assert [name for _, _, name in drawn["groups"]] == ["VAPOR", "OVS"]
        assert drawn["groups"][0][1] == pytest.approx(drawn["groups"][1][0]) and drawn["adc"][0] > 30.0

    def test_the_operators_own_maps_go_through_a_scan(self, tmp_path):
        import nibabel as nib

        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 4, 512, False   # the maps are the point
        body = session.body()
        for name, data in (("gm", body.gm), ("wm", body.wm), ("csf", body.csf), ("b0", np.full(body.gm.shape, 7.0)),
                           ("b1", np.full(body.gm.shape, 0.9)), ("t1", np.full(body.gm.shape, 1.23))):
            nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), body.affine), tmp_path / f"{name}.nii.gz")
        before = session.outlook()["gm"]
        session.anatomy("mine", *(tmp_path / f"{name}.nii.gz" for name in ("gm", "wm", "csf")))
        session.subject, session.field, session.transmit = "mine", str(tmp_path / "b0.nii.gz"), str(tmp_path / "b1.nii.gz")
        session.overrides["t1.NAA"] = str(tmp_path / "t1.nii.gz")                  # a leaf read from a map
        assert session.outlook()["gm"] == pytest.approx(before)
        assert ("t1.NAA", "other") in {(path, kind) for path, kind, _, touched in session.leaves() if touched}
        exam = session.run(session.plan(), seed=1)
        assert exam.status == "done" and np.isfinite(exam.frames[-1][1]).all()

    def test_the_card_promises_and_the_exam_delivers(self, tmp_path):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
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
        assert [count for count, *_ in exam.frames] == [1, 2, 3, 4, 5, 6]          # every transient as it comes in, with its average
        assert exam.frames[0][2].shape == exam.frames[0][1].shape == exam.ppm.shape
        upfield = (exam.ppm > exam.window[0]) & (exam.ppm < exam.window[1])      # the display keeps -0.5 to 6.5, and opens on this
        assert abs(exam.ppm[upfield][exam.frames[-1][1][upfield].argmax()] - 2.01) < 0.03   # phased on NAA, the tallest line
        path = session.export(exam, tmp_path)
        assert path.exists() and path.name == "1_PRESS_TE_30.nii.gz"

    def test_a_phantom_and_an_array_are_in_the_room_as_a_brain_is(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.subject, session.position_mm, session.coil = "Phantom, SPECTRE sphere", [0.0, 0.0, 0.0], "head array"
        assert session.outlook()["gm"] == pytest.approx(1.0)
        assert {image.shape[0] == image.shape[1] for image, *_ in session.planes().values()} == {True}
        drawn = session.scene(points=500)
        assert drawn["cloud"].shape == (500, 3) and len(drawn["loops"]) == 32 and drawn["voxel"] == ((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
        assert np.allclose(np.linalg.norm(drawn["cloud"], axis=1), 90.0, atol=5.0)  # the sphere's surface

    def test_the_gear_opens_every_leaf_and_a_setting_reaches_the_scan(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        leaves = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves()}
        assert leaves["noise.sigma"] == ("range", (3.7, 6.9)) and leaves["water.temperature_c"] == ("fixed", (37.0,))
        assert leaves["water.t1_s"][0] == "other"                               # a tissue record, shown and not edited
        session.overrides.update({"conc.NAA": 20.0, "noise.sigma": (1.0, 1.5)})
        changed = {path: numbers for path, _, numbers, touched in session.leaves() if touched}
        assert changed == {"conc.NAA": (20.0,), "noise.sigma": (1.0, 1.5)}
        session.protocol.averages, session.protocol.npoints = 2, 512
        exam = session.run(session.plan(), seed=1)
        assert exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]["conc"]["NAA"] == 20.0

    def test_a_design_of_blocks_draws_as_a_diagram_and_scans(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.real = False
        drawn = session.diagram()                                               # the card's ideal PRESS, TE 30
        assert [text for _, text in drawn["pulses"]] == ["90°", "180°", "180°"]
        assert np.allclose([time for time, _ in drawn["pulses"]], [0.0, 5.5, 20.5], atol=0.6) and drawn["adc"][0] == pytest.approx(30.0)
        assert len(drawn["marks"]) == 3 and max(abs(g) for _, g in drawn["x"]) == 0.0   # filters, no gradient played
        session.protocol.sequence = "Custom"                                    # the starter: a spin echo with crushers
        drawn = session.diagram()
        assert drawn["adc"][0] == pytest.approx(30.0) and max(abs(g) for _, g in drawn["x"]) > 5.0
        assert max(abs(g) for _, g in drawn["y"]) == 0.0
        session.design.insert(0, ("water suppression", {"scheme": "VAPOR"}))
        from mrsandbox import Vapor

        assert len(session.sequence().events) == len(session.design) - 1 + len(Vapor().events)   # the module's own train
        session.design.pop(0)
        session.protocol.averages, session.protocol.npoints = 2, 1024
        exam = session.run(session.plan(), seed=1)
        assert exam.name == "1 Custom" and exam.status == "done"
        upfield = (exam.ppm > exam.window[0]) & (exam.ppm < exam.window[1])
        assert abs(exam.ppm[upfield][exam.frames[-1][1][upfield].argmax()] - 2.01) < 0.05

    def test_an_edited_card_interleaves_its_sub_experiments_and_reviews_them(self, tmp_path, monkeypatch):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.sequence, session.protocol.averages, session.protocol.npoints, session.size_mm = "MEGA-PRESS", 8, 1024, [30.0] * 3
        session.protocol.real = False                                           # the interleaving is the point
        exam = session.run(session.plan(), seed=1)
        assert exam.name == "1 MEGA-PRESS" and list(exam.conditions) == ["on", "off"] and exam.shown == "difference"
        assert set(exam.spectra) == {"on", "off", "difference", "sum"} and exam.data.shape[-1] == 8
        difference, near = exam.spectra["difference"], lambda lo, hi: (exam.ppm > lo) & (exam.ppm < hi)
        assert difference[near(1.9, 2.1)].min() < 0.0 < difference[near(2.9, 3.1)].max()   # NAA goes with the ON pulse, GABA stays
        assert len(exam.tracks["frequency_hz"]) == 8 and 0.5 < exam.truth["GABA"] < 3.0
        # the library keeps a card with its design and settings; Pulseq takes a design it can play
        monkeypatch.setattr(Session, "LIBRARY", tmp_path / "protocols")
        session.protocol.sequence, session.overrides["conc.NAA"] = "Custom", 15.0
        session.save_protocol("echo")
        session.design, session.overrides = [], {}
        session.load_protocol("echo")
        assert session.protocols() == ["echo"] and len(session.design) == 6 and session.overrides == {"conc.NAA": 15.0}
        assert session.export_seq(session.plan(), tmp_path).read_text().startswith("# Pulseq")
        with pytest.raises(Exception, match="(?i)ideal|filter|coherence"):
            session.export_seq(exam, tmp_path)                                  # MEGA's ideal bands do not play on hardware

    def test_a_spectroscopic_image_is_a_card_too(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.sequence, session.protocol.npoints, session.protocol.bandwidth_hz = "MRSI PRESS", 512, 2000.0
        session.protocol.matrix, session.size_mm, session.position_mm = 6, [14.0] * 3, [0.0, -10.0, 35.0]
        assert session.outlook()["scan_time_s"] == 72.0                         # a repetition per phase encode
        _, (_, _, width, height), *_ = session.planes()["tra"]
        assert width == height == pytest.approx(6 * 14.0 / 2.0)                 # the whole grid on the plane, 2 mm cells
        exam = session.run(session.plan(), seed=1)
        assert exam.data.shape[:3] == (6, 6, 1) and exam.cells.shape[:2] == (6, 6) and set(exam.maps) == {"NAA", "Cr", "Cho", "lipid"}
        assert exam.maps["NAA"][3, 3] > 0.0 and exam.maps["NAA"].shape == (6, 6)
        assert abs(exam.ppm[exam.cells[3, 3].argmax()] - 2.01) < 0.05           # a cell in the brain shows NAA
