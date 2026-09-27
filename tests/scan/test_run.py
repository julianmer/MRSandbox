####################################################################################################
#                                            test_run.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: The breathing, the heart and the head's motion read off a run's recordings.             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pathlib

import numpy as np
import pytest

from mrsandbox.scan.run import MotionReading, PhysioReading, RunReading

SUBJECT = pathlib.Path("data/openneuro_ds001168/sub-01/ses-1")


#**************************************************************************************************#
#                                        Class TestPhysio                                          #
#**************************************************************************************************#
class TestPhysio:

    def test_the_rates_are_the_traces_dominant_frequencies(self):
        t = np.arange(0.0, 300.0, 0.01)
        traces = {"respiratory": np.sin(2.0 * np.pi * 0.25 * t) + 0.3 * t / 300.0, "cardiac": np.sin(2.0 * np.pi * 1.1 * t) + 0.2 * np.sin(2.0 * np.pi * 0.25 * t)}
        reading = PhysioReading.from_arrays(traces, 100.0, {"Manufacturer": "Siemens", "MagneticFieldStrength": 7})
        assert abs(reading.resp_hz - 0.25) < 0.005 and abs(reading.cardiac_hz - 1.1) < 0.005 and reading.duration_s == 300.0
        breathing = reading.record_breathing("a belt")
        assert breathing.leaf == "run.resp_freq_hz" and breathing.carried_by == "subject" and breathing.measured["per_minute"] == 15.0
        assert breathing.scanner["vendor"] == "siemens" and breathing.prior == (0.1875, 0.3125)
        assert PhysioReading.from_arrays({"cardiac": traces["cardiac"]}, 100.0).resp_hz is None

    @pytest.mark.skipif(not (SUBJECT / "func").exists(), reason="ds001168's subject 01 is not on this machine")
    def test_a_bids_recording_reads_as_the_subject_breathed(self):
        reading = PhysioReading.from_bids(SUBJECT / "func" / "sub-01_ses-1_task-rest_acq-prefrontal_physio.tsv.gz")
        assert 0.25 < reading.resp_hz < 0.4 and 0.8 < reading.cardiac_hz < 1.2 and reading.duration_s == 600.0
        assert reading.header.get("RepetitionTime") == 4.0                                  # the scan's sidecar by inheritance


#**************************************************************************************************#
#                                        Class TestMotion                                          #
#**************************************************************************************************#
class TestMotion:

    def test_events_are_the_steps_past_half_a_millimetre(self):
        translations = np.zeros((61, 3))
        translations[20:, 0] = 0.8                                                          # one step of 0.8 mm
        rotations = np.zeros((61, 3))
        rotations[40:, 1] = 0.02                                                            # one turn: 50 mm * 0.02 rad = 1 mm at the cortex
        reading = MotionReading.from_parameters(rotations, translations, tr_s=2.0)
        assert reading.displacement_mm.size == 60 and reading.events.tolist() == [0.8, 1.0] and reading.minutes == pytest.approx(2.0333, rel=1e-3)
        events, size = reading.record_events("a series"), reading.record_size("a series")
        assert events.leaf == "run.events_per_min" and events.value == pytest.approx(2.0 / reading.minutes, rel=1e-3)
        assert size.leaf == "run.motion_mm" and size.value == 0.9 and size.prior == (0.8, 1.0)
        still = MotionReading.from_parameters(np.zeros((10, 3)), np.zeros((10, 3)), 2.0)
        assert still.record_size("a series").value == 0.0 and still.record_events("a series").value == 0.0

    @pytest.mark.slow
    @pytest.mark.skipif(not (SUBJECT / "func").exists(), reason="ds001168's subject 01 is not on this machine")
    def test_a_series_registers_volume_by_volume(self):
        reading = MotionReading.from_series(SUBJECT / "func" / "sub-01_ses-1_task-rest_acq-prefrontal_bold.nii.gz")
        assert reading.tr_s == 4.0 and reading.displacement_mm.size == 149 and 0 < reading.events.size < 20
        assert 0.05 < float(np.median(reading.displacement_mm)) < 0.5


#**************************************************************************************************#
#                                        Class TestRunReading                                      #
#**************************************************************************************************#
class TestRunReading:

    def test_the_tracks_give_back_what_was_put_on_the_transients(self):
        # one FID of two lines, 64 transients 2 s apart, with a drift of 0.6 Hz a minute, breathing of 0.2 Hz amplitude at
        # 0.3 Hz, a phase jitter of 2 degrees, an amplitude jitter of 1 percent, and two motion events (a 3 Hz step)
        rng = np.random.default_rng(1)
        n, dwell, tr = 1024, 1.0 / 2000.0, 2.0
        t = np.arange(n) * dwell
        fid = np.exp(2j * np.pi * 150.0 * t - t / 0.08) + 0.6 * np.exp(2j * np.pi * 280.0 * t - t / 0.06)
        times = np.arange(64) * tr
        frequency = 0.6 * times / 60.0 + 0.2 * np.sin(2.0 * np.pi * 0.3 * times) + np.where(times >= 60.0, 3.0, 0.0) + np.where(times >= 100.0, -3.0, 0.0)
        phase = np.radians(2.0 * rng.standard_normal(64))
        amplitude = 1.0 + 0.01 * rng.standard_normal(64)
        data = np.stack([amplitude[k] * fid * np.exp(2j * np.pi * frequency[k] * t + 1j * phase[k]) for k in range(64)], axis=1)[:, None, :]
        data = data + 0.002 * (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape))
        reading = RunReading.from_arrays(data, dwell, tr, {"Manufacturer": "Siemens"})
        assert abs(reading.drift_hz_per_min - 0.6) < 0.15 and abs(reading.resp_amp_hz - 0.2) < 0.06
        assert 1.0 < reading.phase_jitter_deg < 3.5 and 0.005 < reading.amp_jitter_frac < 0.02
        assert set(reading.events.tolist()) >= {30, 50} and reading.events.size <= 4
        records = {r.leaf: r for r in reading.records("a synthetic run")}
        assert set(records) == {"run.drift_hz_per_min", "run.ar_sigma_hz", "run.resp_amp_hz", "run.phase_jitter_deg", "run.amp_jitter_frac", "run.events_per_min"}
        assert records["run.drift_hz_per_min"].carried_by == "scanner" and records["run.events_per_min"].carried_by == "subject"
        assert records["run.events_per_min"].value == pytest.approx(reading.events.size / reading.minutes, rel=1e-4)

    def test_a_run_whose_lines_are_gone_within_the_window_registers_all_the_same(self):
        # 7 T: lines 13 Hz wide, one transient's start ten times its noise, 128 transients 2.5 s apart with a drift of 1 Hz a
        # minute and breathing of 0.5 Hz at 0.25 Hz; a phase line unwrapped over the 100 ms window wrapped with the noise
        # (every ds005917 run failed the registration rule that way), the cross spectrum's peak does not
        rng = np.random.default_rng(1)
        n, dwell, tr = 2048, 1.0 / 4000.0, 2.5
        t = np.arange(n) * dwell
        fid = np.exp(2j * np.pi * 150.0 * t - t / 0.025) + 0.6 * np.exp(2j * np.pi * 400.0 * t - t / 0.025)
        times = np.arange(128) * tr
        frequency = times / 60.0 + 0.5 * np.sin(2.0 * np.pi * 0.25 * times)
        data = np.stack([fid * np.exp(2j * np.pi * f * t) for f in frequency], axis=1)[:, None, :]
        data = data + 0.1 * (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape))
        reading = RunReading.from_arrays(data, dwell, tr)
        assert abs(reading.drift_hz_per_min - 1.0) < 0.15 and abs(reading.resp_amp_hz - 0.5) < 0.1 and reading.ar_sigma_hz < 0.3
        assert not any(r.failed for r in reading.records("a synthetic 7 T run"))
