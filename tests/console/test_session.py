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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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

    def test_the_background_is_the_measured_shape_and_coupled_systems_on_edited_cards(self):
        from mrsandbox.phantom.macromolecules import CoupledMM, MeasuredMM

        session = Session(ICBM)
        assert isinstance(session.fields()["mm"], MeasuredMM)
        for own in ("MEGA-PRESS", "Custom"):                      # co-editing; a design's own cycle and weights
            session.protocol.sequence = own
            assert isinstance(session.fields()["mm"], CoupledMM)
        session.idealize("macromolecules", True)                     # off silences both sources
        assert session.overrides["mm.level_mm"] == 0.0 and session.overrides["mm.kappa"] == 0.0

    def test_the_snr_reads_its_noise_inside_the_band_and_from_a_kept_transient(self):
        # at 2 kHz the band ends at -3.5 ppm: the noise comes from 9 to 12 ppm, beyond the amides and away from the
        # lipids' wings (read at the edge it came out three times low against the raw runs, item 215); and the single
        # transient's SNR at the processed level is one the processing kept, not a rejected, zeroed one
        from mrsandbox.console.session import COMBINE, PROCESS

        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        session.protocol.averages, session.protocol.npoints, session.protocol.bandwidth_hz, session.protocol.real = 8, 1024, 2000.0, False
        exam = session.run(session.plan(), seed=5)
        session.processing = COMBINE                 # as acquired: no phase, no alignment, so the reading is reproducible
        session.display(exam)
        ppm = exam.ppm
        real = np.fft.fftshift(np.fft.fft(np.asarray(exam.data[:])[0, 0, 0].mean(axis=1), 2 * 1024)).real   # no apodization
        quiet, upfield = (ppm > 9.0) & (ppm < 12.0), (ppm > 0.2) & (ppm < 4.2)
        floor = real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])
        assert exam.snr[0] == pytest.approx(real[upfield].max() / floor.std(), rel=1e-6)
        session.processing = PROCESS
        session.display(exam)
        assert np.isfinite(exam.snr[1]) and exam.snr[1] > 0.0

    def test_the_prescan_adjusts_the_water_suppression_to_the_subject(self):
        # the scanner's water suppression adjustment: the train's flips scaled to leave the least water at the
        # excitation over the voxel's tissues, field groups and transmit scale; the room plays the adjusted train
        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.water_suppression = 0, "CHESS, Siemens svs_se"
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 4, 1024, False
        session.scanner = "siemens_trio_3t"
        exam = session.run(session.plan(), seed=3)
        assert 0.7 < exam.adjustment < 1.3 and exam.adjustment != 1.0
        assert exam.sequence.events[0].flip_deg == pytest.approx(90.0 * exam.adjustment)
        # and it leaves what the Trio's record says it leaves, read the record's own way (HLSVD water of the mean FID
        # at t = 0 over the dry maximum): 0.7 to 1.1 on Big GABA's Trio runs (scan.calibrations, item 217)
        from fsl_mrs.utils.preproc.remove import hlsvd

        mean = np.asarray(exam.data[:])[0, 0, 0].mean(axis=1).astype(np.complex128)
        dry = hlsvd(mean, 1.0 / session.protocol.bandwidth_hz, float(exam.data.hdr_ext["SpectrometerFrequency"][0]) * 1e6,
                    (-0.25, 0.25), limitUnits="ppm")
        assert 0.3 < abs(mean[0] - dry[0]) / np.abs(dry).max() < 3.0
        assert abs(np.angle((mean[0] - dry[0]) / dry[0])) > np.pi / 2       # inverted, as the record's sign says of the Trio
        session.prescan = False
        assert session.run(session.plan(), seed=3).adjustment == 1.0

    def test_a_played_localization_displaces_water_and_lipid_and_an_ideal_one_does_not(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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

    def test_the_scanner_drifts_through_the_session_and_a_prescan_centres_it(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.protocol.real, session.protocol.averages, session.protocol.npoints = False, 30, 256
        session.run_settings = {"ar_sigma_hz": 0.0, "resp_amp_hz": 0.0, "events_per_min": 0.0}     # the drift alone
        session.prescan = False
        first, second = (session.run(session.plan()).tracks["frequency_hz"] for _ in range(2))
        rate = session.drift_hz_per_min
        assert 0.05 <= abs(rate) <= 1.5 and session.clock_s == 120.0
        assert first[0] == 0.0 and second[0] == pytest.approx(rate * 1.0)             # a minute later, a minute of drift on
        assert second[-1] - second[0] == pytest.approx(first[-1] - first[0])          # the same rate all session
        session.prescan = True
        assert session.run(session.plan()).tracks["frequency_hz"][0] == 0.0
        session.protocol.axes = "zxy"                                                 # and the real PRESS orders its axes
        session.protocol.real = True
        assert np.argmin(np.abs(session.displacement(4.65))) == 2                     # the 90, the broad one, selects along z now

    def test_a_scan_of_the_console_calibrates_a_scanner_back_to_what_was_drawn(self, tmp_path):
        import json

        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.field, session.protocol.real, session.protocol.averages, session.seed = "uniform", False, 32, 4
        session.run_settings = {"drift_hz_per_min": 0.0, "ar_sigma_hz": 0.0, "resp_amp_hz": 0.0, "broaden_hz": 0.0, "events_per_min": 0.0}
        exam = session.run(session.plan())
        drawn = exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]
        profile = json.loads(session.calibrate(session.export(exam, tmp_path), "mine", folder=tmp_path).read_text())
        read = {record["leaf"]: record["value"] for record in profile["records"]}
        # the file is in the standard's frame (the centre at 4.65 ppm), so the reading finds NAA where a scanner's data has it
        # one subject's NAA level against the twin's median subject: within 15 percent over five subjects tried, the width
        # being read first and held in the twin (it was 40 percent off when the twin had the population's widths)
        assert read["noise.sigma"] == pytest.approx(drawn["noise"]["sigma"], rel=0.2)
        assert read["lineshape.gauss_hz"] == pytest.approx(drawn["lineshape"]["gauss_hz"], rel=0.2)
        assert profile["scanner"]["name"] == session.scanner

    def test_an_own_design_of_two_sub_acquisitions_is_multiplexed_as_its_weights_say(self):
        from mrsandbox.console.session import BLOCKS, TEMPLATES

        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.protocol.sequence, session.protocol.averages, session.protocol.npoints = "Custom", 8, 512
        press = [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in TEMPLATES["PRESS, TE 30"]]
        session.designs = {"A": press, "B": [(kind, dict(settings)) for kind, settings in press]}
        session.designs["B"][0][1]["phase_deg"] = 180.0                             # the same echo, excited the other way round
        assert session.combinations() == {"sum": {"A": 1.0, "B": 1.0}}              # no weights yet: the plain sum
        session.weights = {"A minus B": {"A": 1.0, "B": -1.0}, "A plus B": {"A": 1.0, "B": 1.0}, "of nothing": {"C": 1.0}}
        exam = session.run(session.plan(), seed=1)
        assert list(exam.conditions) == ["A", "B"] and exam.shown == "A minus B" and "of nothing" not in exam.combinations
        assert exam.data.hdr_ext.to_dict()["SimulationSchedule"]["conditions"][:4] == ["A", "B", "A", "B"]
        upfield = (exam.ppm > 1.8) & (exam.ppm < 2.2)
        tall, flat = (np.abs(exam.spectra[name][upfield]).max() for name in ("A minus B", "A plus B"))
        assert tall > 10.0 * flat                                                   # opposite signs add in the difference
        assert session.showcase()[1].shape == session.showcase()[0].shape

    def test_real_designs_are_fitted_to_the_scanners_transmitter(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"                    # the localization alone: a train has its own timing to say
        prisma = {name: None for name in ("PRESS", "STEAM")}
        for name in prisma:
            session.protocol.sequence = name
            prisma[name] = np.abs(session.displacement(4.65))
        session.scanner = "philips_achieva_3t"                                     # 13.5 uT against the Prisma's 25
        for name in prisma:
            session.protocol.sequence = name
            look = session.outlook()
            assert look["violations"] == ()                                        # longer pulses at the same time-bandwidth product
            assert np.all(np.abs(session.displacement(4.65)) >= prisma[name]) and np.abs(session.displacement(4.65)).max() > prisma[name].max()
        session.protocol.sequence = "sLASER"                                       # TE 30 does not hold the longer pulses: it says so
        assert "a longer TE may" in session.outlook()["violations"][-1]
        session.protocol.te_ms = 40.0                                              # the adiabatic pairs at the scanner's peak, longer
        assert session.outlook()["violations"] == ()
        pairs = [event for event in session.sequence().events if getattr(event, "flip_deg", 0) is None and hasattr(event, "peak_hz")]
        assert len(pairs) == 4 and all(float(event.peak_hz) <= 42.5774785 * 13.5 for event in pairs)
        assert float(pairs[0].duration_s) == pytest.approx(0.0045 * (15.0 / (0.99 * 13.5)) ** 2)   # peak squared times duration held

    def test_a_phase_cycle_cancels_what_an_imperfect_refocusing_pulse_leaves(self):
        from mrsandbox.console.session import ALIGN, BLOCKS

        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.protocol.sequence, session.protocol.averages, session.protocol.npoints, session.field = "Custom", 8, 1024, "uniform"
        # nothing of the scanner's own state, which is drawn anew for every exam: the exams below differ by their designs alone
        session.overrides = {"noise.sigma": 0.0, "acquisition.shift_hz": 0.0, "acquisition.phase_deg": 0.0, "acquisition.eddy_std_rad": 0.0,
                             "spurious.amplitude": 0.0, "b1_scale": 1.0}
        session.run_settings = {"drift_hz_per_min": 0.0, "ar_sigma_hz": 0.0, "resp_amp_hz": 0.0, "phase_jitter_deg": 0.0,
                                "amp_jitter_frac": 0.0, "broaden_hz": 0.0, "events_per_min": 0.0}
        # a spin echo without any filter or crusher, its refocusing pulse 40 degrees short: the 140 leaves a free induction
        # decay of its own and unrefocused signal beside the echo
        echo = [("hard pulse", dict(flip_deg=90.0)), ("delay", dict(ms=15.0)), ("hard pulse", dict(flip_deg=140.0, phase_deg=90.0)), ("delay", dict(ms=15.0))]
        session.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in echo]}
        session.phase_cycle(2, 4)
        assert list(session.designs) == ["A", "B", "C", "D"] and [b[2][1]["phase_deg"] for b in session.designs.values()] == [90.0, 180.0, 270.0, 360.0]
        assert [complex(w) for w in session.weights["cycled"].values()] == [1, -1, 1, -1]        # EXORCYCLE for a refocusing pulse
        exam = session.run(session.plan(), seed=2)
        naa = (exam.ppm > 1.9) & (exam.ppm < 2.1)
        session.processing = ALIGN
        cycled, alone = (session.run(session.plan(), seed=2).spectra[name] for name in ("cycled", "first step alone"))
        # what the cycle keeps is the echo alone: a clean NAA line, where one step carries the unrefocused signal's phase on it
        assert np.abs(cycled[naa]).max() > 0.0 and not np.allclose(cycled / np.abs(cycled).max(), alone / np.abs(alone).max(), atol=0.05)
        # and it is the pathway an ideal crusher keeps: the same echo with coherence filters in place of the cycle
        filtered = [echo[0], ("coherence filter", dict(order=1)), echo[1], echo[2], ("coherence filter", dict(order=-1)), echo[3]]
        session.designs, session.condition, session.weights = {"A": [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in filtered]}, "A", {}
        ideal = session.run(session.plan(), seed=2).frames[-1][1]                   # one sub-acquisition: a plain scan
        lines = (exam.ppm > 1.8) & (exam.ppm < 3.3)                               # NAA, creatine, choline: the uncoupled lines
        assert np.corrcoef(cycled[lines], ideal[lines])[0, 1] > 0.9995 and np.corrcoef(alone[lines], ideal[lines])[0, 1] < 0.995
        assert cycled[naa].max() / 4.0 == pytest.approx(ideal[naa].max(), rel=0.03)  # four steps add to four times the echo
        session.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in echo]}
        session.phase_cycle(0, 2)                                                   # an excitation takes the receiver with it
        assert [complex(w) for w in session.weights["cycled"].values()] == [1, -1]

    def test_an_images_encoding_is_held_against_the_scanners_gradients(self):
        session = Session(ICBM)
        assert session.encoding() is None and session.outlook()["encoding"] is None     # a single voxel encodes nothing
        session.protocol.sequence, session.protocol.bandwidth_hz, session.size_mm = "MRSI PRESS", 2000.0, [10.0, 10.0, 15.0]
        assert session.encoding()["problems"] == () and "phase encoding takes 0.15 ms" in session.encoding()["said"]
        session.protocol.mrsi_readout = "epsi"                                          # a sweep and its flyback per spectral dwell
        flyback = session.encoding()
        session.protocol.mrsi_readout = "epsi-bidirectional"                            # one sweep per dwell: twice the bandwidth
        assert "4614 Hz" in session.encoding()["said"] and "2307 Hz" in flyback["said"] and flyback["problems"] == ()
        session.protocol.mrsi_readout, session.size_mm = "epsi", [5.0, 5.0, 15.0]        # finer cells, a longer sweep
        assert "1631 Hz at most" in session.outlook()["violations"][-1]
        session.scanner = "generic_7t"                                                  # stronger and no faster gradients: the slew decides
        assert "1631 Hz" in session.encoding()["said"]

    def test_the_person_stays_the_scanners_state_is_drawn_anew_and_the_water_is_physics(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.real, session.protocol.averages, session.protocol.npoints, session.seed = False, 4, 512, 4
        session.run_settings = {"events_per_min": 0.0}
        default = session.protocol.water_suppression
        assert default == "COWS(7;236)"                                             # the console's default: the train is played
        heights, truths = {}, []
        for scheme in (default, default, "off", "assumed residual"):
            session.protocol.water_suppression = scheme
            exam = session.run(session.plan())
            water, naa = (exam.ppm > 4.4) & (exam.ppm < 4.9), (exam.ppm > 1.9) & (exam.ppm < 2.1)
            heights.setdefault(scheme, []).append(np.abs(exam.frames[-1][1][water]).max() / exam.frames[-1][1][naa].max())
            truths.append(exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"])
        assert len({truth["conc"]["NAA"] for truth in truths}) == 1                 # one person through all four exams
        assert len({truth["acquisition"]["shift_hz"] for truth in truths}) == 4     # the scanner's state anew every time
        assert heights[default][0] != heights[default][1]                           # so a train's residual differs from exam to exam
        assert heights["off"][0] > 5.0 * max(heights[default])                      # without a train, the whole water line

    def test_the_card_plays_water_suppression_and_ovs_before_the_localization(self):
        from mrsandbox import Vapor

        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
        session.protocol.water_suppression = "off"
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
        session.subject, session.position_mm, session.coil = "Phantom, SPECTRE sphere", [0.0, 0.0, 0.0], "head array"
        assert session.outlook()["gm"] == pytest.approx(1.0)
        assert {image.shape[0] == image.shape[1] for image, *_ in session.planes().values()} == {True}
        drawn = session.scene(points=500)
        assert drawn["cloud"].shape == (500, 3) and len(drawn["loops"]) == 32 and drawn["voxel"] == ((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
        assert np.allclose(np.linalg.norm(drawn["cloud"], axis=1), 90.0, atol=5.0)  # the sphere's surface

    def test_the_gear_opens_every_leaf_and_a_setting_reaches_the_scan(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0       # one card, one exam; the reference has a test of its own
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
        session.protocol.sequence, session.protocol.averages, session.protocol.npoints, session.size_mm = "MEGA-PRESS", 8, 1024, [30.0] * 3
        session.protocol.real = False                                           # the interleaving is the point
        exam = session.run(session.plan(), seed=1)
        assert exam.name == "1 MEGA-PRESS" and list(exam.conditions) == ["on", "off"] and exam.shown == "difference"
        assert set(exam.spectra) == {"on", "off", "difference", "sum"} and exam.data.shape[-1] == 8
        difference, near = exam.spectra["difference"], lambda lo, hi: (exam.ppm > lo) & (exam.ppm < hi)
        assert difference[near(1.9, 2.1)].min() < 0.0 < difference[near(2.9, 3.1)].max()   # NAA goes with the ON pulse, GABA stays
        assert len(exam.tracks["frequency_hz"]) == len(exam.echoes) == 8 and 0.5 < exam.truth["GABA"] < 3.0
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
        session.protocol.water_suppression = "assumed residual"    # the package's shortcut: no train played, as these gates were written
        session.protocol.sequence, session.protocol.npoints, session.protocol.bandwidth_hz = "MRSI PRESS", 512, 2000.0
        session.protocol.matrix, session.protocol.matrix_y, session.size_mm, session.position_mm = 6, 6, [14.0] * 3, [0.0, -10.0, 35.0]
        assert session.outlook()["scan_time_s"] == 72.0                         # a repetition per phase encode
        _, (_, _, width, height), *_ = session.planes()["tra"]
        assert width == height == pytest.approx(6 * 14.0 / 2.0)                 # the whole grid on the plane, 2 mm cells
        exam = session.run(session.plan(), seed=1)
        assert exam.data.shape[:3] == (6, 6, 1) and exam.cells.shape[:3] == (6, 6, 1) and set(exam.maps) == {"NAA", "Cr", "Cho", "lipid"}
        assert exam.maps["NAA"][3, 3, 0] > 0.0 and exam.maps["NAA"].shape == (6, 6, 1)
        assert abs(exam.ppm[exam.cells[3, 3, 0].argmax()] - 2.01) < 0.05           # a cell in the brain shows NAA

    def test_the_scanner_runs_at_the_operators_field(self):
        # the field beside the scanner: the preset's name, limits and records stay, the spins precess at the new field,
        # the exam keeps the field in its room, and the preset's own number is the preset again
        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 2, 512, False
        session.field_t = 1.5
        assert session.profile().scanner.name == "siemens_prisma_3t" and session.profile().scanner.b0_t == 1.5
        exam = session.run(session.plan(), seed=3)
        assert float(exam.data.hdr_ext["SpectrometerFrequency"][0]) == pytest.approx(42.5774785 * 1.5, rel=1e-3)
        assert exam.room["field_t"] == 1.5
        session.field_t = None
        assert session.profile().scanner.b0_t == 3.0

