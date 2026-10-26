####################################################################################################
#                                          test_session.py                                         #
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
import os
import pathlib
import shutil

import numpy as np
import pytest

from mrsandbox.console.session import Session

# the subject every machine has; the console's own default is placed by hand
ICBM = "Brain, ICBM152 average, 2 mm"


#**************************************************************************************************#
#                                         Class TestSession                                        #
#**************************************************************************************************#
@pytest.mark.render
class TestSession:

    def test_the_planes_are_squares_and_a_click_on_the_voxel_leaves_it_there(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        planes = session.planes()
        assert {image.shape for image, *_ in planes.values()} == {(115, 115)}
        assert all(image.max() > 200 and image.dtype == np.uint8 for image, *_ in planes.values())
        before = list(session.position_mm)
        for name, (_, (left, top, width, height), cut, *_) in planes.items():
            assert cut
            session.place(name, left + width / 2.0, top + height / 2.0)
        assert np.allclose(session.position_mm, before, atol=1.0)          # half a 2 mm cell
        height = session.position_mm[2]
        session.place("tra", 80.0, 40.0)  # a transverse click leaves the height
        assert session.position_mm[2] == height and session.position_mm[0] != before[0]

    def test_leafing_through_a_plane_leaves_the_voxel_and_a_click_places_it_on_the_slice(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        before = list(session.position_mm)
        session.scroll("tra", 20)
        image, _, cut, *_ = session.planes()["tra"]
        assert session.position_mm == before and session.view_mm[2] > before[2] + 30.0 and not cut
        session.scroll("tra", 10 ** 6)  # the grid's last slice, no further
        assert session.view_mm[2] < 200.0
        session.scroll("tra", -10 ** 6)
        session.scroll("tra", 45)
        session.place("tra", 60.0, 60.0)  # a click is on the slice on view
        assert session.position_mm[2] == session.view_mm[2] and session.planes()["tra"][2]

    def test_the_background_is_the_measured_shape_and_coupled_systems_on_edited_cards(self):
        from mrsandbox.phantom.macromolecules import CoupledMM, MeasuredMM

        session = Session(ICBM)
        assert isinstance(session.fields()["mm"], MeasuredMM)
        for own in ("MEGA-PRESS", "Custom"):  # co-editing; a design's own cycle and weights
            session.protocol.sequence = own
            assert isinstance(session.fields()["mm"], CoupledMM)
        session.idealize("macromolecules", True)                     # off silences both sources
        assert session.overrides["mm.level_mm"] == 0.0 and session.overrides["mm.kappa"] == 0.0

    def test_the_snr_reads_its_noise_inside_the_band_and_from_a_kept_transient(self):
        # at 2 kHz the band ends at -3.5 ppm: the noise comes from 9 to 12 ppm, beyond the amides
        # and away from the lipids' wings; and the single transient's SNR at the processed level is
        # one the processing kept, not a rejected, zeroed one
        from mrsandbox.console.session import COMBINE, PROCESS

        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        (session.protocol.averages, session.protocol.npoints, session.protocol.bandwidth_hz,
         session.protocol.real) = 8, 1024, 2000.0, False
        exam = session.run(session.plan(), seed=5)
        # as acquired: no phase, no alignment, so the reading is reproducible
        session.processing = COMBINE
        session.display(exam)
        ppm = exam.ppm
        # no apodization
        real = np.fft.fftshift(np.fft.fft(np.asarray(exam.data[:])[0, 0, 0].mean(axis=1),
                                          2 * 1024)).real
        # the signal is the tallest of the metabolite lines, not the residual water's wing
        quiet, lines = (ppm > 9.0) & (ppm < 12.0), (ppm > 1.8) & (ppm < 3.4)
        floor = real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])
        assert exam.snr[0] == pytest.approx(real[lines].max() / floor.std(), rel=1e-6)
        session.processing = PROCESS
        session.display(exam)
        assert np.isfinite(exam.snr[1]) and exam.snr[1] > 0.0

    def test_the_processed_transients_lie_under_their_average(self):
        # processed, the average is moved onto its reference peak, phased on creatine and loses its
        # residual water: the transients behind it are turned and cleaned the same way, so their
        # mean, apodized as the average is, lies on it over the metabolite lines (they stood 0.74 of
        # NAA apart when only the average was turned)
        from mrsandbox.console.session import PROCESS

        session = Session(ICBM)
        session.seed = 1
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        session.protocol.averages, session.protocol.npoints = 8, 1024
        exam = session.run(session.plan(), seed=5)
        session.processing = PROCESS
        session.display(exam)
        n, bandwidth = session.protocol.npoints, session.protocol.bandwidth_hz
        # the mean of the transients' real spectra, apodized by 2 Hz as the average is shown
        fid = np.fft.ifft(np.fft.ifftshift(np.mean(exam.transients[exam.shown], axis=0)))
        fid[:n] *= np.exp(-np.pi * 2.0 * np.arange(n) / bandwidth)
        mean = np.fft.fftshift(np.fft.fft(fid)).real
        average = exam.spectra[exam.shown]
        lines = (exam.ppm > 1.8) & (exam.ppm < 3.4)
        assert np.abs(mean - average)[lines].max() < 0.2 * np.abs(average[lines]).max()

    def test_an_exams_basis_set_exports_as_fsl_mrs_reads_it(self, tmp_path):
        # the design without its preludes, every metabolite the room simulates, one JSON each
        import json

        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.real = 0, False
        session.protocol.npoints = 512
        exam = session.plan()
        folder = session.export_basis(exam, tmp_path)
        written = sorted(path.stem for path in folder.glob("*.json"))
        assert "NAA" in written and len(written) >= 20
        basis = json.loads((folder / "NAA.json").read_text())["basis"]
        assert len(basis["basis_re"]) == 512 and basis["basis_name"] == "NAA"
        assert abs(basis["basis_dwell"] - 1.0 / session.protocol.bandwidth_hz) < 1e-12

    def test_the_window_brightens_and_narrows_the_picture(self):
        # the planes' window and level: a lower level brightens, a narrower width adds contrast;
        # None is the picture's own
        session = Session(ICBM)
        own = session.picture().astype(float)
        session.window = (0.3, 1.0)
        bright = session.picture().astype(float)
        session.window = (0.5, 0.4)
        narrow = session.picture().astype(float)
        session.window = None
        inside = own > 0
        assert bright[inside].mean() > own[inside].mean()
        assert narrow[inside].std() > own[inside].std()
        assert np.array_equal(session.picture(), own.astype(np.uint8))

    def test_held_readings_play_the_records_at_their_values(self, tmp_path):
        # a replica: every record of the scanner at its value, where a draw takes the record's
        # prior a quarter either side (a preset's pooled rows are priors, not records: a saved
        # profile with its own record stands in)
        import dataclasses

        from mrsandbox import Scanner
        from mrsandbox.scan import ScannerProfile
        from mrsandbox.scan.calibrations import calibrations

        record = dataclasses.replace(
            calibrations("siemens_prisma_3t")["noise.sigma"], value=8.0, prior=(6.0, 10.0),
            source={"dataset": "my scanner", "citation": None, "url": None})
        ScannerProfile(Scanner.preset("siemens_prisma_3t"), records=(record,)).save(
            tmp_path / "my_prisma.json")
        session = Session(ICBM)
        session.load_scanner(tmp_path / "my_prisma.json")
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 2, 1024, False

        def sigma():
            truth = session.run(session.plan(), seed=3).data.hdr_ext.to_dict()
            return float(truth["SimulationGroundTruth"]["params"]["noise"]["sigma"])

        drawn = sigma()
        session.hold_records = True
        assert sigma() == pytest.approx(record.value) and drawn != pytest.approx(record.value)
        assert session.queue[-1].room["hold_records"]

    def test_the_prescan_adjusts_the_water_suppression_to_the_subject(self):
        # the scanner's water suppression adjustment: the train's flips scaled to leave the least
        # water at the excitation over the voxel's tissues, field groups and transmit scale; the
        # room plays the adjusted train
        session = Session(ICBM)
        (session.protocol.water_reference,
         session.protocol.water_suppression) = 0, "CHESS, Siemens svs_se"
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 4, 1024, False
        session.scanner = "siemens_trio_3t"
        exam = session.run(session.plan(), seed=3)
        # the Trio's pooled residual is a tenth of a percent: a train scaled well down
        assert 0.4 < exam.adjustment < 1.3 and exam.adjustment != 1.0
        assert exam.sequence.events[0].flip_deg == pytest.approx(90.0 * exam.adjustment)
        # and it leaves what the Trio's record says it leaves, read the record's own way
        # (scan.calibrate WaterReading: the exponentials within 0.4 ppm of the carrier in the
        # mean FID at t = 0, over the maximum of the FID without them)
        from mrsandbox.scan.calibrate import _exponentials

        def residual(data):
            mean = np.asarray(data[:])[0, 0, 0].mean(axis=1).astype(np.complex128)
            dwell_s = 1.0 / session.protocol.bandwidth_hz
            freqs, damping, amplitudes = _exponentials(mean[:1024], dwell_s)
            water = ((np.abs(freqs) < 0.4 * float(data.hdr_ext["SpectrometerFrequency"][0]))
                     & (damping < 0.0))
            t = np.arange(mean.size) * dwell_s
            dry = mean - sum(a * np.exp((2j * np.pi * f + d) * t) for f, d, a in
                             zip(freqs[water], damping[water], amplitudes[water]))
            return amplitudes[water].sum() / np.abs(dry).max(), dry[0]

        water, start = residual(exam.data)
        assert 0.3 < abs(water) < 3.0
        # inverted, as the record's sign says of the Trio
        assert abs(np.angle(water / start)) > np.pi / 2
        session.prescan = False
        assert session.run(session.plan(), seed=3).adjustment == 1.0
        # the train played real: its Gaussians' own band and the relaxation through them put its
        # null a tenth above the ideal one and leave more there than the record says, so the
        # prescan searches the engine's steady state and lands on that null
        session.prescan, session.protocol.real = True, True
        played = session.run(session.plan(), seed=3)
        assert exam.adjustment + 0.05 < played.adjustment < 1.5
        water, start = residual(played.data)
        assert 0.3 < abs(water) < 3.0
        assert abs(np.angle(water / start)) > np.pi / 2

    def test_a_played_localization_displaces_water_and_lipid_and_an_ideal_one_does_not(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        session.protocol.real = False  # ideal pulses select exactly
        assert session.displacement(4.65) == [0.0, 0.0, 0.0] and session.planes()["tra"][4] == {}
        session.protocol.real = True                                       # the default
        water, lipid = np.asarray(session.displacement(4.65)), np.asarray(session.displacement(1.3))
        # either side of the 3 ppm the slabs are centred for
        assert np.all(water < 0.0) and np.all(lipid > 0.0)
        assert abs(water[1]) > 2.0 * abs(water[0])  # the narrow 180s displace more than the 90
        assert np.allclose(water / lipid, (4.65 - 3.0) / (1.3 - 3.0))
        _, (left, top, _, _), _, _, shifted, _ = session.planes()["tra"]
        # towards lower x and lower y (rows run down)
        assert shifted["water"][0] < left and shifted["water"][1] > top
        session.protocol.sequence = "sLASER"  # broadband adiabatic pairs: a tenth of it
        assert abs(session.displacement(4.65)[1]) < 0.15 * abs(water[1])

    def test_the_shim_and_its_box_set_the_line_width_a_b0_map_gives(self, tmp_path):
        import nibabel as nib

        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        # the measured field, shimmed, by default
        assert session.field == "bigbrain_mr" and 1.0 <= session.linewidth() < 5.0
        session.field = "uniform"
        assert session.linewidth() is None  # a uniform field broadens nothing
        body = session.body()
        x, y = ((np.arange(n) * np.diag(body.affine)[axis] + body.affine[axis, 3])
                for axis, n in enumerate(body.gm.shape[:2]))
        # Hz: a slope and a harmonic saddle
        saddle = 0.4 * x[:, None, None] + 0.02 * (x[:, None, None] ** 2 - y[None, :, None] ** 2)
        nib.save(nib.Nifti1Image(np.broadcast_to(saddle, body.gm.shape).astype(np.float32),
                                 body.affine), tmp_path / "b0.nii.gz")
        session.field, widths = str(tmp_path / "b0.nii.gz"), {}
        for shim in ("none", "linear", "quadratic"):
            session.shim = shim
            widths[shim] = session.linewidth()
        assert (widths["none"] > 3.0 and widths["none"] > 1.5 * widths["linear"]
                and widths["linear"] > 1.2 * widths["quadratic"])
        # the saddle is gone, the 1 Hz line stays
        assert widths["quadratic"] == pytest.approx(1.0, abs=0.15)
        session.shim, session.shim_mm = "linear", 60.0
        assert session.planes()["tra"][4]["shim"][2] == pytest.approx(
            60.0 / np.diag(body.affine)[0])
        # a saddle has no slope: the same fit over a wider box
        assert session.linewidth() == pytest.approx(widths["linear"], rel=0.05)

    def test_one_subject_lies_on_the_table_and_a_water_reference_follows_its_card(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        (session.protocol.real, session.protocol.averages, session.protocol.npoints,
         session.protocol.water_reference) = False, 4, 512, 2
        session.size_mm = [30.0, 20.0, 15.0]                                # an anisotropic voxel
        session.overrides.update({"run.events_per_min": 0.0})
        session.plan()
        # the reference first, the default
        assert [exam.name for exam in session.queue] == ["1 PRESS water reference", "2 PRESS TE 30"]
        water, scan = (session.run(exam) for exam in session.queue)
        assert scan.truth == water.truth  # the same draw of the population
        assert (abs(water.ppm[np.argmax(water.frames[-1][1])] - 4.65) < 0.1
                and water.card.averages == 2)
        assert water.frames[-1][1].max() > 5.0 * scan.frames[-1][1].max()   # unsuppressed
        again = session.run(session.plan(queued=False))
        session.queue.append(again)
        # its own noise at another place in the queue
        assert not np.allclose(session.run(again).frames[-1][1], scan.frames[-1][1])

    def test_one_engine_serves_every_exam_and_the_physics_is_the_same(self):
        # the engine of a field is kept across exams, so its graphs are built once; an exam on the
        # kept engine equals the one that built it, a preview at the same place and seed
        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.protocol.real, session.protocol.averages, session.protocol.npoints = False, 4, 512
        first = session.run(session.plan(queued=False), seed=3)
        engine = session._bodies[("engine", session.profile().scanner.b0_t)]
        kept = session.run(session.plan(queued=False), seed=3)
        assert (session._bodies[("engine", session.profile().scanner.b0_t)] is engine
                and engine._compiled)
        assert np.array_equal(np.asarray(kept.data[:]), np.asarray(first.data[:]))

    def test_the_scanner_drifts_through_the_session_and_a_prescan_centres_it(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        session.protocol.real, session.protocol.averages, session.protocol.npoints = False, 30, 256
        # the drift alone, at a stated rate (a drawn one may lie near zero, as some measured runs
        # drift)
        session.overrides.update({"run.drift_hz_per_min": 0.4, "run.ar_sigma_hz": 0.0,
                                  "run.resp_amp_hz": 0.0, "run.events_per_min": 0.0})
        session.prescan = False
        first, second = (session.run(session.plan()).tracks["frequency_hz"] for _ in range(2))
        rate = session.drift_hz_per_min
        assert rate == 0.4 and session.clock_s == 120.0
        # a minute later, a minute of drift on
        assert first[0] == 0.0 and second[0] == pytest.approx(rate * 1.0)
        # the same rate all session
        assert second[-1] - second[0] == pytest.approx(first[-1] - first[0])
        session.prescan = True
        assert session.run(session.plan()).tracks["frequency_hz"][0] == 0.0
        session.protocol.axes = "zxy"  # and the real PRESS orders its axes
        session.protocol.real = True
        # the 90, the broad one, selects along z now
        assert np.argmin(np.abs(session.displacement(4.65))) == 2

    def test_a_scan_of_the_console_calibrates_a_scanner_back_to_what_was_drawn(self, tmp_path):
        import json

        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        (session.field, session.protocol.real, session.protocol.averages,
         session.seed) = "uniform", False, 32, 4
        # still, and no water leaking in from outside the voxel: the calibration's twin holds
        # none (CALM), so a leak would read as the scanner's noise and width
        session.overrides.update({"run.drift_hz_per_min": 0.0, "run.ar_sigma_hz": 0.0,
                                  "run.resp_amp_hz": 0.0, "run.broaden_hz": 0.0,
                                  "run.events_per_min": 0.0, "oov.fraction": 0.0})
        session.protocol.water_reference = 2  # its water reference exam runs first
        # the file is in the standard's frame (the centre at 4.65 ppm), so the reading finds NAA
        # where a scanner's data has it; one subject's NAA level against the twin's median subject,
        # the width being read first and held in the twin: one subject reads off by as much as
        # subjects differ, so five subjects, the median of read over drawn
        ratios, first = {"noise.sigma": [], "lineshape.gauss_hz": []}, None
        for seed in range(4, 9):
            session.seed, session.queue = seed, []
            session.plan()
            reference, exam = (session.run(planned) for planned in session.queue)
            drawn = exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]
            profile = json.loads(session.calibrate(
                session.export(exam, tmp_path / str(seed)), "mine", folder=tmp_path,
                reference=session.export(reference, tmp_path / str(seed))).read_text())
            records = {record["leaf"]: record for record in profile["records"]}
            truth = {"noise.sigma": drawn["noise"]["sigma"],
                     "lineshape.gauss_hz": drawn["lineshape"]["gauss_hz"]}
            for leaf in ratios:
                if not records[leaf]["failed"]:     # a reading its own rule failed is not a ratio
                    ratios[leaf].append(records[leaf]["value"] / truth[leaf])
            first = first or (profile, {leaf: r["value"] for leaf, r in records.items()})
        for leaf, found in ratios.items():
            assert len(found) >= 4 and np.median(found) == pytest.approx(1.0, abs=0.2), leaf
        profile, read = first
        assert profile["scanner"]["name"] == session.scanner
        # with the reference: the residual the shortcut water left, upright, and a receiver delay
        # record come too (the delay's number is gated on a voxel's own FID in
        # tests/scan/test_profile.py; on a console exam the shortcut water's wing bends creatine's
        # phase)
        assert 0.0 < read["water.adjusted_residual"] < 0.05 and "acquisition.delay_s" in read

    def test_a_filter_block_carries_the_crusher_moment_it_stands_for(self):
        # the ideal crusher of an own design weights by diffusion as the kick it stands for
        # would: a pair of 9.7 cycles/mm about a 180 gives (2 pi m)^2 times the time between
        # them, and a library editing design shown as blocks keeps its pairs' moments
        import math

        from mrsandbox import LocalizedMegaPress
        from mrsandbox.console.session import BLOCKS, TEMPLATES
        from mrsandbox.sequences.diffusion import b_value_ms_um2

        session = Session(ICBM)
        session.protocol.sequence = "Custom"
        press = [(kind, {**BLOCKS[kind][0], **settings})
                 for kind, settings in TEMPLATES["PRESS, TE 30"]]
        kinds = [kind for kind, _ in press]
        assert kinds.count("coherence filter") >= 2 and BLOCKS["coherence filter"][0] == {
            "order": -1, "cycles_per_mm": 0.0}
        first = kinds.index("coherence filter")
        second = kinds.index("coherence filter", first + 1)
        press[first][1]["cycles_per_mm"] = press[second][1]["cycles_per_mm"] = 9.7
        session.designs, session.condition = {"A": press}, "A"
        between = sum(settings["ms"] / 1000.0
                      for kind, settings in press[first + 1:second] if kind == "delay")
        assert between > 0.0
        expected = (2.0 * math.pi * 9.7e3) ** 2 * between * 1e-9
        assert abs(b_value_ms_um2(session.sequence()) - expected) < 1e-9 * expected
        blocks = Session._blocks(LocalizedMegaPress(7.5, 127.7, size_mm=30.0).events)
        moments = [settings["cycles_per_mm"] for kind, settings in blocks if kind == "coherence filter"]
        assert sorted(moments)[0] == -9.7 and sorted(moments)[-1] == 9.7 and moments.count(0.0) == 5

    def test_an_own_design_of_two_sub_acquisitions_is_multiplexed_as_its_weights_say(self):
        from mrsandbox.console.session import BLOCKS, TEMPLATES

        session = Session(ICBM)
        # a room of its own: a session draws one, and its drift between the alternating
        # sub-acquisitions (0.64 Hz a minute in one) spoils the cancellation the weights are read on
        session.seed = 1
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        (session.protocol.sequence, session.protocol.averages,
         session.protocol.npoints) = "Custom", 8, 512
        press = [(kind, {**BLOCKS[kind][0], **settings})
                 for kind, settings in TEMPLATES["PRESS, TE 30"]]
        session.designs = {"A": press, "B": [(kind, dict(settings)) for kind, settings in press]}
        # the same echo, excited the other way round
        session.designs["B"][0][1]["phase_deg"] = 180.0
        # no weights yet: the plain sum
        assert session.combinations() == {"sum": {"A": 1.0, "B": 1.0}}
        session.weights = {"A minus B": {"A": 1.0, "B": -1.0}, "A plus B": {"A": 1.0, "B": 1.0},
                           "of nothing": {"C": 1.0}}
        exam = session.run(session.plan(), seed=1)
        assert (list(exam.conditions) == ["A", "B"] and exam.shown == "A minus B"
                and "of nothing" not in exam.combinations)
        assert (exam.data.hdr_ext.to_dict()["SimulationSchedule"]["conditions"][:4]
                == ["A", "B", "A", "B"])
        upfield = (exam.ppm > 1.8) & (exam.ppm < 2.2)
        tall, flat = (np.abs(exam.spectra[name][upfield]).max()
                      for name in ("A minus B", "A plus B"))
        assert tall > 10.0 * flat  # opposite signs add in the difference
        assert session.showcase()[1].shape == session.showcase()[0].shape

    def test_the_designers_preview_renders_the_planned_voxel(self):
        session = Session(ICBM)
        session.protocol.npoints, session.protocol.water_suppression = 512, "off"
        session.protocol.ovs = False

        def bands():
            ppm, real = session.showcase()
            return (float(np.abs(real[(ppm > 0.9) & (ppm < 1.5)]).max()),
                    float(np.abs(real[(ppm > 1.9) & (ppm < 2.1)]).max()))
        # a voxel at the vertex, in the scalp: the lipids stand over NAA
        session.position_mm = [0.0, -20.0, 90.0]
        lipids, naa = bands()
        assert lipids > 3.0 * naa > 0.0
        # beyond the head: nothing
        session.position_mm = [0.0, 0.0, 300.0]
        assert bands() == (0.0, 0.0)

    def test_the_designers_perfect_spectrum_has_no_broadening(self):
        session = Session(ICBM)
        # the water suppressed as assumed (no train played), so NAA stands clear of its wing
        session.protocol.npoints, session.protocol.water_suppression = 2048, "assumed residual"
        session.protocol.ovs = False

        def naa(perfect):
            ppm, real = session.showcase(perfect)
            line = real[(ppm > 1.9) & (ppm < 2.1)] - np.median(real[(ppm > 1.7) & (ppm < 2.3)])
            return line.max(), int(np.sum(line > line.max() / 2))
        (top, width), (sharp, narrow) = naa(False), naa(True)
        # the same signal in fewer points: taller and narrower; the session's own display untouched
        assert sharp > 2.0 * top and narrow < width and session.apodization_hz == 2.0

    def test_the_designer_shows_the_cards_preludes_before_the_design(self):
        session = Session(ICBM)
        session.adopt()
        session.protocol.water_suppression, session.protocol.ovs = "WET", True
        preludes = session.preludes(session.protocol)
        own = session.own()
        designed = session.designed()
        # the card's water suppression and OVS first, then the design's blocks
        assert list(preludes) == ["WET", "OVS"] and len(designed.events) == (
            sum(len(p.events) for p in preludes.values()) + len(own.events))
        drawn = session.diagram(designed, preludes=preludes)
        assert [name for _, _, name in drawn["groups"]][:1] == ["WET"]
        # removed on the card, gone from the designer
        session.protocol.water_suppression, session.protocol.ovs = "off", False
        assert len(session.designed().events) == len(own.events)

    def test_real_designs_are_fitted_to_the_scanners_transmitter(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        # the localization alone: a train has its own timing to say
        session.protocol.water_suppression = "assumed residual"
        prisma = {name: None for name in ("PRESS", "STEAM")}
        for name in prisma:
            session.protocol.sequence = name
            prisma[name] = np.abs(session.displacement(4.65))
        session.scanner = "philips_achieva_3t"  # 13.5 uT against the Prisma's 25
        for name in prisma:
            session.protocol.sequence = name
            look = session.outlook()
            # longer pulses at the same time-bandwidth product
            assert look["violations"] == ()
            assert (np.all(np.abs(session.displacement(4.65)) >= prisma[name])
                    and np.abs(session.displacement(4.65)).max() > prisma[name].max())
        # TE 30 does not hold the longer pulses: it says so
        session.protocol.sequence = "sLASER"
        assert "a longer TE may" in session.outlook()["violations"][-1]
        session.protocol.te_ms = 40.0  # the adiabatic pairs at the scanner's peak, longer
        assert session.outlook()["violations"] == ()
        pairs = [event for event in session.sequence().events
                 if getattr(event, "flip_deg", 0) is None and hasattr(event, "peak_hz")]
        assert len(pairs) == 4 and all(float(event.peak_hz) <= 42.5774785 * 13.5 for event in pairs)
        # peak squared times duration held
        assert float(pairs[0].duration_s) == pytest.approx(0.0045 * (15.0 / (0.99 * 13.5)) ** 2)

    def test_a_phase_cycle_cancels_what_an_imperfect_refocusing_pulse_leaves(self):
        from mrsandbox.console.session import ALIGN, BLOCKS

        session = Session(ICBM)
        session.protocol.water_reference = 0
        session.protocol.water_suppression = "assumed residual"
        (session.protocol.sequence, session.protocol.averages, session.protocol.npoints,
         session.field) = "Custom", 8, 1024, "uniform"
        # nothing of the scanner's own state, which is drawn anew for every exam: the exams below
        # differ by their designs alone
        session.overrides = {"noise.sigma": 0.0, "acquisition.shift_hz": 0.0,
                             "acquisition.phase_deg": 0.0, "acquisition.eddy_std_rad": 0.0,
                             "spurious.amplitude": 0.0, "b1_scale": 1.0,
                             "acquisition.delay_s": 0.0, "acquisition.eddy_transient_rad": 0.0,
                             "oov.fraction": 0.0}  # drawn anew per run: held still here
        session.overrides.update({"run.drift_hz_per_min": 0.0, "run.ar_sigma_hz": 0.0,
                                  "run.resp_amp_hz": 0.0, "run.phase_jitter_deg": 0.0,
                                  "run.amp_jitter_frac": 0.0, "run.broaden_hz": 0.0,
                                  "run.events_per_min": 0.0})
        # a spin echo without any filter or crusher, its refocusing pulse 40 degrees short: the 140
        # leaves a free induction decay of its own and unrefocused signal beside the echo
        echo = [("hard pulse", dict(flip_deg=90.0)), ("delay", dict(ms=15.0)),
                ("hard pulse", dict(flip_deg=140.0, phase_deg=90.0)), ("delay", dict(ms=15.0))]
        session.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings})
                                 for kind, settings in echo]}
        session.phase_cycle(2, 4)
        assert (list(session.designs) == ["A", "B", "C", "D"]
                and [b[2][1]["phase_deg"] for b in session.designs.values()]
                == [90.0, 180.0, 270.0, 360.0])
        # EXORCYCLE for a refocusing pulse
        assert [complex(w) for w in session.weights["cycled"].values()] == [1, -1, 1, -1]
        exam = session.run(session.plan(), seed=2)
        naa = (exam.ppm > 1.9) & (exam.ppm < 2.1)
        session.processing = ALIGN
        cycled, alone = (session.run(session.plan(), seed=2).spectra[name]
                         for name in ("cycled", "first step alone"))
        # what the cycle keeps is the echo alone: a clean NAA line, where one step carries the
        # unrefocused signal's phase on it
        assert (np.abs(cycled[naa]).max() > 0.0
                and not np.allclose(cycled / np.abs(cycled).max(), alone / np.abs(alone).max(),
                                    atol=0.05))
        # and it is the pathway an ideal crusher keeps: the same echo with coherence filters in
        # place of the cycle
        filtered = [echo[0], ("coherence filter", dict(order=1)), echo[1], echo[2],
                    ("coherence filter", dict(order=-1)), echo[3]]
        session.designs, session.condition, session.weights = (
            {"A": [(kind, {**BLOCKS[kind][0], **settings}) for kind, settings in filtered]},
            "A", {})
        # one sub-acquisition: a plain scan
        ideal = session.run(session.plan(), seed=2).frames[-1][1]
        lines = (exam.ppm > 1.8) & (exam.ppm < 3.3)  # NAA, creatine, choline: the uncoupled lines
        assert (np.corrcoef(cycled[lines], ideal[lines])[0, 1] > 0.9995
                and np.corrcoef(alone[lines], ideal[lines])[0, 1] < 0.995)
        # four steps add to four times the echo
        assert cycled[naa].max() / 4.0 == pytest.approx(ideal[naa].max(), rel=0.03)
        session.designs = {"A": [(kind, {**BLOCKS[kind][0], **settings})
                                 for kind, settings in echo]}
        session.phase_cycle(0, 2)  # an excitation takes the receiver with it
        assert [complex(w) for w in session.weights["cycled"].values()] == [1, -1]

    def test_an_images_encoding_is_held_against_the_scanners_gradients(self):
        session = Session(ICBM)
        # a single voxel encodes nothing
        assert session.encoding() is None and session.outlook()["encoding"] is None
        (session.protocol.sequence, session.protocol.bandwidth_hz,
         session.size_mm) = "MRSI PRESS", 2000.0, [10.0, 10.0, 15.0]
        assert (session.encoding()["problems"] == ()
                and "phase encoding takes 0.15 ms" in session.encoding()["said"])
        session.protocol.mrsi_readout = "epsi"  # a sweep and its flyback per spectral dwell
        flyback = session.encoding()
        # one sweep per dwell: twice the bandwidth
        session.protocol.mrsi_readout = "epsi-bidirectional"
        assert ("4614 Hz" in session.encoding()["said"] and "2307 Hz" in flyback["said"]
                and flyback["problems"] == ())
        # finer cells, a longer sweep
        session.protocol.mrsi_readout, session.size_mm = "epsi", [5.0, 5.0, 15.0]
        assert "1631 Hz at most" in session.outlook()["violations"][-1]
        session.scanner = "generic_7t"  # stronger and no faster gradients: the slew decides
        assert "1631 Hz" in session.encoding()["said"]

    def test_a_registered_child_lies_in_the_template_of_their_age(self):
        session = Session(ICBM)
        session.admit(age_y=9)
        assert session.subject == "Brain, NIHPD 7-11 years, 2 mm"
        assert session.body().name == "nihpd_7-11" and session.body().fractions(
            session.position_mm, session.size_mm).gm > 0.0
        session.admit(age_y=3)                      # under the youngest band: that band
        assert session.subject == "Brain, NIHPD 4.5-8.5 years, 2 mm"
        session.admit(age_y=40)                     # an adult: the average brain
        assert session.subject == ICBM
        session.admit()
        assert session.subject == ICBM
        # a head that is not a template stays
        session.subject = "Brain, BigBrain-MR, 0.8 mm"
        session.admit(age_y=9)
        assert session.subject == "Brain, BigBrain-MR, 0.8 mm"

    def test_the_person_stays_the_scanners_state_is_drawn_anew_and_the_water_is_physics(self):
        session = Session(ICBM)
        session.protocol.water_reference = 0
        (session.protocol.real, session.protocol.averages, session.protocol.npoints,
         session.seed) = False, 4, 512, 4
        # the train's work alone: no water leaking in from outside the voxel, which no train
        # suppresses
        session.overrides.update({"run.events_per_min": 0.0, "oov.fraction": 0.0})
        default = session.protocol.water_suppression
        assert default == "COWS(7;236)"  # the console's default: the train is played
        heights, truths = {}, []
        for scheme in (default, default, "off", "assumed residual"):
            session.protocol.water_suppression = scheme
            exam = session.run(session.plan())
            water, naa = (exam.ppm > 4.4) & (exam.ppm < 4.9), (exam.ppm > 1.9) & (exam.ppm < 2.1)
            heights.setdefault(scheme, []).append(np.abs(exam.frames[-1][1][water]).max()
                                                  / exam.frames[-1][1][naa].max())
            truths.append(exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"])
        # one person through all four exams
        assert len({truth["conc"]["NAA"] for truth in truths}) == 1
        # the scanner's state anew every time
        assert len({truth["acquisition"]["shift_hz"] for truth in truths}) == 4
        # so a train's residual differs from exam to exam
        assert heights[default][0] != heights[default][1]
        # without a train, the whole water line: the residual goes with the scanner's state, so
        # over five people and their exams, the median of unsuppressed over suppressed
        over = [heights["off"][0] / max(heights[default])]
        for seed in range(5, 9):
            session.seed, found = seed, {}
            for scheme in (default, "off"):
                session.protocol.water_suppression = scheme
                exam = session.run(session.plan())
                water = (exam.ppm > 4.4) & (exam.ppm < 4.9)
                naa = (exam.ppm > 1.9) & (exam.ppm < 2.1)
                found[scheme] = (np.abs(exam.frames[-1][1][water]).max()
                                 / exam.frames[-1][1][naa].max())
            over.append(found["off"] / found[default])
        assert np.median(over) > 5.0

    def test_the_card_plays_water_suppression_and_ovs_before_the_localization(self):
        from mrsandbox import Vapor

        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        session.protocol.water_suppression = "off"
        bare = len(session.sequence().events)
        session.protocol.water_suppression, session.protocol.ovs = "VAPOR", True
        # six slabs, a filter after each
        assert len(session.sequence().events) == bare + len(Vapor().events) + 12
        drawn = session.diagram()
        assert [name for _, _, name in drawn["groups"]] == ["VAPOR", "OVS"]
        assert (drawn["groups"][0][1] == pytest.approx(drawn["groups"][1][0])
                and drawn["adc"][0] > 30.0)

    def test_the_slabs_are_placed_one_by_one_and_kept_with_the_protocol(self, tmp_path):
        session = Session(ICBM)
        session.protocol.ovs = True
        assert len(session.slabs()) == 6
        planned = session.plan(queued=False)         # a planned exam keeps the card's six
        # one slab off, one with a gap and thickness of its own: 25 clear of a 20 mm voxel's face
        session.set_slab("x", 1.0, on=False)
        session.set_slab("z", -1.0, gap_mm=25.0, mm=40.0)
        slabs = dict(((axis, centre > 0), (centre, thick)) for axis, centre, thick in session.slabs())
        assert len(slabs) == 5 and ("x", True) not in slabs
        assert slabs[("z", False)] == (-(10.0 + 40.0 / 2.0 + 25.0), 40.0)
        assert len(session.slabs(planned.card)) == 6
        # a slab tilted 30 degrees toward y: its normal, and the whole voxel played at once; in the
        # transverse plane its band leans (a polygon whose columns move down the rows)
        session.set_slab("x", -1.0, tilt_deg=(30.0, 0.0))
        axis, centre, thick, normal = session.slabs()[0]
        assert np.allclose(normal, (np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0))
        assert session.sequence().unfolded and not session.sequence(planned.card).unfolded
        band = session.planes()["tra"][4][f"ovs x{centre:+.0f}"]
        top, bottom = (np.mean([x for x, y in band if y == row]) for row in (0.0, 115.0))
        assert abs(top - bottom) > 10.0
        # dragging a band: the point of a plane's picture in millimetres, and the same place
        column, row = 80, 70
        before = list(session.position_mm)
        session.place("cor", column, row)
        assert session.at("cor", column, row) == session.position_mm != before
        # the slabs travel with the saved protocol
        session.LIBRARY = tmp_path
        session.save_protocol("slabs")
        fresh = Session(ICBM)
        fresh.LIBRARY = tmp_path
        fresh.load_protocol("slabs")
        assert fresh.protocol.ovs and len(fresh.slabs()) == 5 and len(fresh.slabs()[0]) == 4
        fresh.set_slab("x", 1.0, on=True)
        fresh.set_slab("x", -1.0, tilt_deg=(0.0, 0.0))
        assert len(fresh.slabs()) == 6 and not fresh.sequence().unfolded

    def test_a_turned_voxel_reads_the_subject_as_it_lies_and_the_exam_keeps_it(self):
        session = Session(ICBM)
        # a voxel long along x turned a quarter about z holds what one long along y holds
        session.size_mm = [30.0, 10.0, 10.0]
        session.turn_deg = [0.0, 0.0, 90.0]
        body = session.body()
        turned = body.fractions(tuple(session.position_mm), (30.0, 10.0, 10.0), session.turn)
        square = body.fractions(tuple(session.position_mm), (10.0, 30.0, 10.0))
        assert np.allclose(turned, square, atol=0.01)
        # in the transverse plane its outline is the turned box: 10 mm across, 30 mm up (2 mm cells)
        outline = session.planes()["tra"][4]["voxel"]
        columns, rows = (np.ptp([point[k] for point in outline]) for k in (0, 1))
        assert abs(columns - 5.0) < 0.6 and abs(rows - 15.0) < 0.6
        # the exam plays it so and says so
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        (session.protocol.averages, session.protocol.npoints, session.protocol.real) = 2, 512, False
        exam = session.run(session.plan(), seed=3)
        header = exam.data.hdr_ext.to_dict()
        assert exam.turn_deg == (0.0, 0.0, 90.0)
        assert header["SimulationPhantom"]["turn_deg"] == [0.0, 0.0, 90.0]
        assert np.allclose(list(header["SimulationGroundTruth"]["tissue_fractions"].values()),
                           turned, atol=1e-6)

    def test_the_operators_own_maps_go_through_a_scan(self, tmp_path):
        import nibabel as nib

        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        # the maps are the point
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 4, 512, False
        body = session.body()
        for name, data in (("gm", body.gm), ("wm", body.wm), ("csf", body.csf),
                           ("b0", np.full(body.gm.shape, 7.0)), ("b1", np.full(body.gm.shape, 0.9)),
                           ("t1", np.full(body.gm.shape, 1.23))):
            nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), body.affine),
                     tmp_path / f"{name}.nii.gz")
        before = session.outlook()["gm"]
        session.anatomy("mine", *(tmp_path / f"{name}.nii.gz" for name in ("gm", "wm", "csf")))
        session.subject, session.field, session.transmit = (
            "mine", str(tmp_path / "b0.nii.gz"), str(tmp_path / "b1.nii.gz"))
        session.overrides["t1.NAA"] = str(tmp_path / "t1.nii.gz")  # a leaf read from a map
        assert session.outlook()["gm"] == pytest.approx(before)
        assert ("t1.NAA", "other") in {(path, kind) for path, kind, _, touched in session.leaves()
                                       if touched}
        exam = session.run(session.plan(), seed=1)
        assert exam.status == "done" and np.isfinite(exam.frames[-1][1]).all()

    def test_one_dataset_can_calibrate_the_scanner_on_the_floor(self):
        # the scanner is hardware; its calibration is chosen apart: the package's pooled rows, or
        # one cohort's alone
        session = Session(ICBM)
        session.scanner = "siemens_trio_3t"
        assert session.calibrations()["big_gaba_trio"] == (
            "Big GABA · Siemens Trio 3 T · PRESS · 36 subjects")
        assert (session.profile().records == ()
                and session.outlook()["records"]["noise.sigma"]["source"].startswith("pooled over"))
        session.calibration = "big_gaba_trio"
        assert ({r.leaf for r in session.profile().records}
                and all(r.source["dataset"].endswith(": big_gaba_trio")
                        for r in session.profile().records))
        assert session.outlook()["records"]["noise.sigma"]["source"].endswith(": big_gaba_trio")

    def test_the_hardware_numbers_and_a_folder_are_the_operators_to_set(self, tmp_path):
        # the manufacturer's numbers are the scanner's until set; a scanner file from anywhere
        # calibrates; an empty folder says what it lacks
        session = Session(ICBM)
        assert session.profile().scanner.max_slew_t_m_s == 200.0 and len(session.systems()) >= 25
        session.hardware["max_slew_t_m_s"] = 120.0
        assert (session.profile().scanner.max_slew_t_m_s == 120.0
                and session.outlook()["limits"][2] == 120.0)
        path = tmp_path / "my_slow_scanner.json"
        session.profile().save(path)
        fresh = Session(ICBM)
        fresh.load_scanner(path)
        assert (fresh.calibration == str(path) and str(path) in fresh.calibrations()
                and fresh.profile().scanner.max_slew_t_m_s == 120.0)
        report = fresh.replicate(tmp_path)  # the folder holds the saved scanner alone
        assert report == [("my_slow_scanner.json", "scanner file", "the scanner's calibration")]

    def test_a_pulseq_file_becomes_the_design(self, tmp_path):
        # the sequence builder's upload: a .seq file's events up to its first readout as blocks,
        # the card on Custom
        import numpy as np
        import pypulseq

        system = pypulseq.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
        seq = pypulseq.Sequence(system=system)
        # within a 25 uT peak B1
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system))
        seq.add_block(pypulseq.make_delay(0.01))
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi, duration=2e-3, system=system))
        seq.add_block(pypulseq.make_delay(0.01))
        seq.add_block(pypulseq.make_adc(num_samples=512, dwell=2.5e-4, system=system))
        seq.write(str(tmp_path / "echo.seq"))
        session = Session(ICBM)
        session.adopt_seq(tmp_path / "echo.seq")
        kinds = [kind for kind, _ in session.design]
        assert session.protocol.sequence == "Custom" and len(kinds) >= 3 and "delay" in kinds
        assert session.sequence().events and session.outlook()["violations"] == ()

    def test_a_cohort_on_the_floor_scans_one_of_its_subjects(self):
        # a package cohort picked: its subject's scanner, voxel place and size come with it, every
        # record of the scan is that subject's own reading fixed, and a new draw of the population
        # takes another subject
        session = Session(ICBM)
        session.pick_cohort("big_gaba_trio")
        assert (session.scanner == "siemens_trio_3t" and session.position_mm == [0.0, -35.0, 20.0]
                and session.size_mm == [30.0, 30.0, 30.0])
        # a subject whose noise reading did not fail
        while "noise.sigma" not in {r.leaf for r in session.member().fixed().records}:
            session.seed += 1
        member = session.member()
        assert (member.subject.startswith("S")
                and all(record.prior == (record.value, record.value)
                        for record in session.profile().records))
        assert not any(record.failed for record in session.profile().records)
        leaves = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves()}
        assert leaves["noise.sigma"] == ("fixed", (next(r.value for r in member.profile.records
                                                        if r.leaf == "noise.sigma"),))
        subjects = set()
        for seed in range(12):
            session.seed = seed
            subjects.add(session.member().subject)
        assert len(subjects) > 1
        session.pick_cohort(None)
        assert session.member() is None and session.profile().records == ()
        # a new draw of a cohort scanned in several regions moves the voxel to the new subject's
        session.pick_cohort("cows_prisma")
        places = set()
        for seed in range(12):
            session.seed = seed
            member = session.member()
            assert (session.position_mm == list(member.position_mm)
                    and session.size_mm == list(member.size_mm))
            places.add(tuple(session.position_mm))
        assert len(places) > 1
        # one subject picked by name stays on the floor, place and all, until a new draw
        from mrsandbox.scan.cohort import package

        cohort = next(c for c in package() if c.name == "cows_prisma")
        wanted = next(m for m in cohort.members if "Occipital" in m.subject)
        session.pick_member(wanted.subject)
        assert (session.member().subject == wanted.subject
                and session.position_mm == list(wanted.position_mm))
        # what the exam draws at that voxel: the subject's own readings say so, the rest its source
        drawn = {path: (value, origin) for path, value, origin in session.drawn()}
        assert drawn["noise.sigma"][1] == "dataset" and drawn["noise.sigma"][0] == next(
            r.value for r in wanted.profile.records if r.leaf == "noise.sigma")
        assert drawn["conc.NAA"][1] in ("literature", "pooled", "mixed", "stated") and drawn["conc.NAA"][0] > 0
        session.seed += 1
        assert session.member_name is None

    def test_a_population_narrows_every_pooled_value(self):
        # the measured rows every pooled value comes from: every row by default, then the datasets
        # named and the subjects a query keeps; a leaf the population holds no row of keeps the
        # literature, and the room says what it draws from
        import re

        session = Session(ICBM)

        def subjects(path):
            said = session.sources()[path]["provenance"]["pooled"]
            return int(re.search(r"pooled over (\d+) subjects", said).group(1))

        everyone = subjects("noise.sigma")
        session.population = {"datasets": ("cows_prisma",), "query": ""}
        cows = subjects("noise.sigma")
        assert 0 < cows < everyone and "cows_prisma" in session.sources()["noise.sigma"]["provenance"]["pooled"]
        session.population = {"datasets": ("cows_prisma",), "query": "age_low >= 40"}
        assert 0 < subjects("noise.sigma") < cows
        assert all(name == "cows_prisma" for name, _ in session.population_within())
        assert session.population_says().startswith("pooled values come from cows_prisma where age_low >= 40")
        # a cohort of maps alone has no noise rows: the leaf keeps the literature
        session.population = {"datasets": ("ds004611_prisma_3t",), "query": ""}
        assert "noise.sigma" not in session.sources()
        session.population = {"datasets": (), "query": ""}
        assert subjects("noise.sigma") == everyone

    def test_the_overview_narrows_by_its_charts_and_shows_what_is_drawn(self):
        # every chart's choice is a condition the rows meet (an age range too, which as the person
        # alone would give way), nothing chosen is everyone again, and the draws of a pooled leaf
        # follow the population
        import re

        session = Session(ICBM)

        def subjects(path):
            said = session.sources()[path]["provenance"]["pooled"]
            return int(re.search(r"pooled over (\d+) subjects", said).group(1))

        everyone, wide = subjects("noise.sigma"), session.draws()["noise.sigma"]
        session.narrow(datasets=("cows_prisma",))
        cows = subjects("noise.sigma")
        session.narrow(datasets=("cows_prisma",), ages=(40, 90))
        assert 0 < subjects("noise.sigma") < cows < everyone
        # the charts hand numpy's values
        session.narrow(vendors=(np.str_("ge"),), fields=(np.float64(3.0),))
        assert 0 < subjects("noise.sigma") < everyone and "np." not in session.population["query"]
        session.narrow(datasets=("cows_prisma",), ages=(40, 90))
        assert session.person == {"age_y": (40.0, 90.0)}
        narrow = session.draws()
        assert narrow["noise.sigma"].shape == (300,) and narrow["water.t1_s"].shape == (300, 3)
        assert np.ptp(narrow["noise.sigma"]) < np.ptp(wide)
        session.narrow()
        assert subjects("noise.sigma") == everyone and session.person == {}

    def test_a_dataset_is_the_card_on_new_draws_with_their_truth(self, tmp_path):
        # each subject a new draw, its exam a NIfTI-MRS file, its drawn values a row of the table;
        # the session's own draw comes back
        import pandas as pd

        session = Session(ICBM)
        session.seed = 7
        session.protocol.averages, session.protocol.npoints = 2, 512
        session.protocol.water_reference, session.protocol.real = 0, False
        table = session.make_dataset(tmp_path, 2, seed=100)
        rows = pd.read_csv(table)
        assert list(rows["seed"]) == [100, 101] and session.seed == 7
        assert all((tmp_path / name).exists() for name in rows["file"])
        assert rows["conc.NAA"].nunique() == 2 and "noise.sigma" in rows

    def test_the_arrays_noise_share_comes_from_the_pooled_rows(self):
        # the share of a channel's noise that is its own: the package's rows of the channels'
        # correlation for the scanner at the array's channel count, turned into the share against
        # the array on the floor, until the operator reads one off a scan of their own
        from mrsandbox.scan.calibrate import CovarianceReading
        from mrsandbox.scan.calibrations import calibrations

        session = Session(ICBM)
        session.coil = "head array"  # 32 loops: a share to take
        share, source, provenance = session.own_share()
        correlation = calibrations("generic_3t", channels=32)["coils.partner_correlation"]
        assert (source == "pooled" and provenance.startswith("the channels' correlation")
                and provenance.endswith("against this array"))
        assert share == round(CovarianceReading.share(session.array(), correlation.value), 2)
        # an array of the operator's own maps has no loops to take the correlation against: the
        # pooled share serves
        from mrsandbox.hardware.coils import MappedCoilArray

        session.coils["my maps"] = lambda: MappedCoilArray(
            np.ones((4, 4, 4, 32), dtype=np.complex64), np.eye(4), name="my maps")
        session.coil = "my maps"
        assert session.own_share()[1] == "pooled"
        session.own_noise = 0.4
        assert session.own_share() == (0.4, "own", "read from your scan with the coils apart")

    def test_a_source_chosen_for_a_leaf_reaches_the_scan(self, tmp_path):
        # every Sourced leaf lists its sources with where each comes from; a choice of literature
        # or measured is applied to every scan of the session (Session.twin) and saved with the
        # protocol; an override makes the leaf custom
        session = Session(ICBM)
        sources = session.sources()
        assert (sources["lineshape.gauss_hz"]["chosen"] == "literature"
                and set(sources["lineshape.gauss_hz"]["sources"]) == {"literature", "pooled",
                                                                       "mixed"})
        assert (sources["noise.sigma"]["chosen"] == "pooled"
                and sources["noise.sigma"]["provenance"]["pooled"].startswith("pooled over"))
        assert (sources["lineshape.gauss_hz"]["sources"]["literature"] == ("range", (2.0, 6.0))
                and sources["lineshape.gauss_hz"]["sources"]["pooled"][0] == "empirical")
        session.choices["lineshape.gauss_hz"] = "pooled"
        leaves = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves()}
        assert (leaves["lineshape.gauss_hz"][0] == "empirical"
                and session.sources()["lineshape.gauss_hz"]["chosen"] == "pooled")
        assert session.twin().params.lineshape.gauss_hz.chosen == "pooled"
        session.LIBRARY = tmp_path
        session.save_protocol("chosen")
        fresh = Session(ICBM)
        fresh.LIBRARY = tmp_path
        fresh.load_protocol("chosen")
        assert fresh.choices == {"lineshape.gauss_hz": "pooled"}

    def test_the_card_promises_and_the_exam_delivers(self, tmp_path):
        session = Session(ICBM)
        # these gates assume the Prisma (the console opens on a generic 3 T)
        session.scanner = "siemens_prisma_3t"
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        session.protocol.averages, session.protocol.npoints = 6, 1024
        look = session.outlook()
        assert look["gm"] + look["wm"] + look["csf"] == pytest.approx(1.0, abs=0.02)
        assert (look["scan_time_s"] == 12.0 and look["violations"] == ()
                and look["limits"] == (25.0, 80.0, 200.0))
        # the Prisma's own rows, pooled
        assert look["records"]["noise.sigma"]["source"].endswith("cows_prisma")
        exam = session.plan()
        assert exam.name == "1 PRESS TE 30" and exam.status == "planned" and session.queue == [exam]
        session.position_mm = [20.0, 0.0, 30.0]  # the exam keeps the voxel it was planned with
        session.run(exam, seed=1)
        assert exam.status == "done" and exam.position_mm == (0.0, -45.0, 35.0)
        assert exam.data.shape[-1] == 6  # the transients apart, as a scanner exports them
        # every transient as it comes in, with its average
        assert [count for count, *_ in exam.frames] == [1, 2, 3, 4, 5, 6]
        assert exam.frames[0][2].shape == exam.frames[0][1].shape == exam.ppm.shape
        # the display keeps -0.5 to 6.5, and opens on this
        upfield = (exam.ppm > exam.window[0]) & (exam.ppm < exam.window[1])
        # phased on NAA, the tallest line
        assert abs(exam.ppm[upfield][exam.frames[-1][1][upfield].argmax()] - 2.01) < 0.03
        path = session.export(exam, tmp_path)
        assert path.exists() and path.name == "1_PRESS_TE_30.nii.gz"

    def test_a_phantom_and_an_array_are_in_the_room_as_a_brain_is(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        session.subject, session.position_mm, session.coil = (
            "Phantom, SPECTRE sphere", [0.0, 0.0, 0.0], "head array")
        assert session.outlook()["gm"] == pytest.approx(1.0)
        assert {image.shape[0] == image.shape[1]
                for image, *_ in session.planes().values()} == {True}
        drawn = session.scene(points=500)
        assert (drawn["cloud"].shape == (500, 3) and len(drawn["loops"]) == 32
                and drawn["voxel"] == ((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)))
        # the sphere's surface
        assert np.allclose(np.linalg.norm(drawn["cloud"], axis=1), 90.0, atol=5.0)

    def test_the_gear_opens_every_leaf_and_a_setting_reaches_the_scan(self):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        leaves = {path: (kind, numbers) for path, kind, numbers, _ in session.leaves()}
        from mrsandbox.scan.calibrations import calibrations

        sigma = calibrations("generic_3t")["noise.sigma"]
        assert leaves["noise.sigma"][0] == "empirical"  # the pooled rows, one per draw
        assert leaves["noise.sigma"][1] == pytest.approx(
            (sigma.value, *sigma.prior, float(sigma.measured["subjects"])))
        assert leaves["water.temperature_c"] == ("fixed", (37.0,))
        assert leaves["water.t1_s"][0] == "other"  # a tissue record, shown and not edited
        session.overrides.update({"conc.NAA": 20.0, "noise.sigma": (1.0, 1.5)})
        changed = {path: numbers for path, _, numbers, touched in session.leaves() if touched}
        assert changed == {"conc.NAA": (20.0,), "noise.sigma": (1.0, 1.5)}
        session.protocol.averages, session.protocol.npoints = 2, 512
        exam = session.run(session.plan(), seed=1)
        assert exam.data.hdr_ext.to_dict()["SimulationGroundTruth"]["params"]["conc"]["NAA"] == 20.0

    def test_a_design_of_blocks_draws_as_a_diagram_and_scans(self):
        session = Session(ICBM)
        # these gates assume the Prisma: the generic 3 T pools every vendor's leaked water
        session.scanner = "siemens_prisma_3t"
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        session.protocol.real = False
        drawn = session.diagram()  # the card's ideal PRESS, TE 30
        assert [text for _, text in drawn["pulses"]] == ["90°", "180°", "180°"]
        assert (np.allclose([time for time, _ in drawn["pulses"]], [0.0, 5.5, 20.5], atol=0.6)
                and drawn["adc"][0] == pytest.approx(30.0))
        # filters (one after the 90, one on each side of every 180), no gradient played
        assert len(drawn["marks"]) == 5 and max(abs(g) for _, g in drawn["x"]) == 0.0
        session.protocol.sequence = "Custom"  # the starter: a spin echo with crushers
        drawn = session.diagram()
        assert drawn["adc"][0] == pytest.approx(30.0) and max(abs(g) for _, g in drawn["x"]) > 5.0
        assert max(abs(g) for _, g in drawn["y"]) == 0.0
        session.design.insert(0, ("water suppression", {"scheme": "VAPOR"}))
        from mrsandbox import Vapor

        # the module's own train
        assert len(session.sequence().events) == len(session.design) - 1 + len(Vapor().events)
        session.design.pop(0)
        session.protocol.averages, session.protocol.npoints = 2, 1024
        exam = session.run(session.plan(), seed=1)
        assert exam.name == "1 Custom" and exam.status == "done"
        upfield = (exam.ppm > exam.window[0]) & (exam.ppm < exam.window[1])
        assert abs(exam.ppm[upfield][exam.frames[-1][1][upfield].argmax()] - 2.01) < 0.05

    def test_an_edited_card_interleaves_its_sub_experiments_and_reviews_them(
            self, tmp_path, monkeypatch):
        session = Session(ICBM)
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        (session.protocol.sequence, session.protocol.averages, session.protocol.npoints,
         session.size_mm) = "MEGA-PRESS", 8, 1024, [30.0] * 3
        session.protocol.real = False  # the interleaving is the point
        exam = session.run(session.plan(), seed=1)
        assert (exam.name == "1 MEGA-PRESS" and list(exam.conditions) == ["on", "off"]
                and exam.shown == "difference")
        assert set(exam.spectra) == {"on", "off", "difference", "sum"} and exam.data.shape[-1] == 8
        difference, near = (exam.spectra["difference"],
                            lambda lo, hi: (exam.ppm > lo) & (exam.ppm < hi))
        # NAA goes with the ON pulse, GABA stays
        assert difference[near(1.9, 2.1)].min() < 0.0 < difference[near(2.9, 3.1)].max()
        assert (len(exam.tracks["frequency_hz"]) == len(exam.echoes) == 8
                and 0.5 < exam.truth["GABA"] < 3.0)
        # the library keeps a card with its design and settings; Pulseq takes a design it can play
        monkeypatch.setattr(Session, "LIBRARY", tmp_path / "protocols")
        session.protocol.sequence, session.overrides["conc.NAA"] = "Custom", 15.0
        session.save_protocol("echo")
        session.design, session.overrides = [], {}
        session.load_protocol("echo")
        assert (session.protocols() == ["echo"] and len(session.design) == 6
                and session.overrides == {"conc.NAA": 15.0})
        assert session.export_seq(session.plan(), tmp_path).read_text().startswith("# Pulseq")
        with pytest.raises(Exception, match="(?i)ideal|filter|coherence"):
            session.export_seq(exam, tmp_path)  # MEGA's ideal bands do not play on hardware

    def test_a_spectroscopic_image_is_a_card_too(self):
        session = Session(ICBM)
        # these gates assume the Prisma: the generic 3 T pools every vendor's leaked water
        session.scanner = "siemens_prisma_3t"
        # one card, one exam; the reference has a test of its own
        session.protocol.water_reference = 0
        # the package's shortcut: no train played, which these gates assume
        session.protocol.water_suppression = "assumed residual"
        (session.protocol.sequence, session.protocol.npoints,
         session.protocol.bandwidth_hz) = "MRSI PRESS", 512, 2000.0
        (session.protocol.matrix, session.protocol.matrix_y, session.size_mm,
         session.position_mm) = 6, 6, [14.0] * 3, [0.0, -10.0, 35.0]
        assert session.outlook()["scan_time_s"] == 72.0  # a repetition per phase encode
        _, (_, _, width, height), *_ = session.planes()["tra"]
        # the whole grid on the plane, 2 mm cells
        assert width == height == pytest.approx(6 * 14.0 / 2.0)
        exam = session.run(session.plan(), seed=1)
        assert (exam.data.shape[:3] == (6, 6, 1) and exam.cells.shape[:3] == (6, 6, 1)
                and set(exam.maps) == {"NAA", "Cr", "Cho", "lipid"})
        assert exam.maps["NAA"][3, 3, 0] > 0.0 and exam.maps["NAA"].shape == (6, 6, 1)
        # a cell in the brain shows NAA
        assert abs(exam.ppm[exam.cells[3, 3, 0].argmax()] - 2.01) < 0.05

    def test_the_scanner_runs_at_the_operators_field(self):
        # the field beside the scanner: the preset's name, limits and records stay, the spins
        # precess at the new field, the exam keeps the field in its room, and the preset's own
        # number is the preset again
        session = Session(ICBM)
        session.protocol.water_reference, session.protocol.water_suppression = 0, "assumed residual"
        session.protocol.averages, session.protocol.npoints, session.protocol.real = 2, 512, False
        session.field_t = 1.5
        assert (session.profile().scanner.name == "generic_3t"
                and session.profile().scanner.b0_t == 1.5)
        exam = session.run(session.plan(), seed=3)
        assert float(exam.data.hdr_ext["SpectrometerFrequency"][0]) == pytest.approx(
            42.5774785 * 1.5, rel=1e-3)
        assert exam.room["field_t"] == 1.5
        session.field_t = None
        assert session.profile().scanner.b0_t == 3.0  # the generic system's nominal field


    @pytest.mark.slow
    @pytest.mark.skipif(not any(pathlib.Path(p).joinpath("fast").exists() for p in (
        os.path.expandvars("$FSLDIR/bin"), os.path.expanduser("~/fsl/share/fsl/bin")))
                        and shutil.which("fast") is None,
                        reason="FSL segments the image")
    def test_an_oblique_t1_weighted_image_lies_straight_on_the_floor(self, tmp_path):
        # an MPM or a sagittal MPRAGE is acquired tilted (the hMRI demo subject by 30 degrees about
        # x): the anatomy is set straight on the scanner's axes, where the planes cut it, and the
        # image stays where the head lay
        import nibabel as nib

        body = Session(ICBM).body()
        image = (350.0 * np.asarray(body.csf) + 650.0 * np.asarray(body.gm)
                 + 1000.0 * np.asarray(body.wm))
        turn = np.radians(30.0)
        tilt = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, np.cos(turn), -np.sin(turn), 0.0],
                         [0.0, np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 0.0, 1.0]])
        nib.save(nib.Nifti1Image(image.astype(np.float32), tilt @ body.affine),
                 tmp_path / "t1w.nii.gz")
        session = Session(ICBM)
        session.segment(tmp_path / "t1w.nii.gz", "tilted", folder=tmp_path / "tilted")
        session.subject = "tilted"
        straight = session.body().affine
        assert np.allclose(straight[:3, :3], np.eye(3))

        def centre(values, affine):  # the image's centre of mass in the scanner's mm
            values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
            cells = np.stack(np.meshgrid(*(np.arange(n) for n in values.shape), indexing="ij"),
                             axis=-1).reshape(-1, 3)
            return affine[:3, :3] @ (cells.T @ values.ravel()) / values.sum() + affine[:3, 3]

        picture = nib.load(tmp_path / "tilted" / "t1.nii.gz")
        assert np.allclose(centre(picture.dataobj, picture.affine),
                           centre(image, tilt @ body.affine), atol=1.0)
