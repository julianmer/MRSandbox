####################################################################################################
#                                          test_paradigm.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/11/26                                                                                #
#                                                                                                  #
# Purpose: A functional run: the paradigm's task and response, the courses the levels ride over   #
#          the transients, the header's truth, and the task reader reading the courses back.      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from mrsandbox import Brain, Multiplex, NoDownfield, NoLipids, NoMM, Params, Press, Scanner
from mrsandbox.scan.calibrate import TaskReading
from mrsandbox.scan.paradigm import Paradigm

STILL = dict(drift_hz_per_min=0.0, ar_sigma_hz=0.0, resp_amp_hz=0.0, phase_jitter_deg=0.0,
             amp_jitter_frac=0.0, broaden_hz=0.0, events_per_min=0.0)
QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                eddy_transient_rad=0.0),
               oov=dict(fraction=0.0), run=dict(motion_mm=0.0))


#**************************************************************************************************#
#                                        Class TestParadigm                                        #
#**************************************************************************************************#
class TestParadigm:

    def test_blocks_task_and_response(self):
        paradigm = Paradigm.blocks(rest_s=30.0, task_s=20.0, cycles=2)
        assert paradigm.onsets_s == (30.0, 80.0) and paradigm.durations_s == (20.0, 20.0)
        t = np.arange(0.0, 130.0, 1.0)
        on = paradigm.task(t)
        assert on[29] == 0.0 and on[30] == 1.0 and on[49] == 1.0 and on[50] == 0.0 and on[80] == 1.0
        course = paradigm.response(t, rise_s=2.0, decay_s=4.0)
        # from zero, up toward one within a few rise constants of the onset, down after the end
        assert course[30] == 0.0 and course[40] > 0.99 and course[50] > 0.99
        assert 0.1 < course[58] < 0.2 and course[79] < 0.01
        assert np.all(np.diff(course[30:50]) >= 0.0) and np.all(np.diff(course[50:80]) <= 0.0)


#**************************************************************************************************#
#                                       Class TestTaskRun                                          #
#**************************************************************************************************#
@pytest.mark.render
class TestTaskRun:

    @pytest.fixture(scope="class")
    def run(self):
        # a still scanner and subject, the metabolites alone (no macromolecules, lipids or
        # downfield peaks under the bands), the levels alone moving: lactate a fifth up and
        # glutamate a twentieth at the plateau, reached within a transient; eight transients of
        # rest, then sixteen of task
        courses = ("Lac", "Glu", "GABA", "Asp", "Glc", "GSH", "Gln", "Gly", "NAA", "Ins")
        params = QUIET | Params(fmrs=dict(effect=dict({name: 0.0 for name in courses},
                                                      Lac=0.2, Glu=0.05),
                                          rise_s={name: 0.1 for name in courses},
                                          decay_s={name: 0.1 for name in courses},
                                          bold_hz_per_t=0.0, bold_s=0.1, bold_amplitude=0.0))
        room = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                         {"scan": Press(te1_s=0.011, te2_s=0.019, tr_s=1.0)}, params=params,
                         mm=NoMM(), lipids=NoLipids(), downfield=NoDownfield())
        paradigm = Paradigm.blocks(rest_s=8.0, task_s=16.0)
        data = room.transients((10.0, -20.0, 5.0), 20.0, 24, tr_s=1.0, seed=3, npoints=1024,
                               dwell_s=1.0 / 2000.0, synthesis=STILL, paradigm=paradigm)
        plain = room.transients((10.0, -20.0, 5.0), 20.0, 24, tr_s=1.0, seed=3, npoints=1024,
                                dwell_s=1.0 / 2000.0, synthesis=STILL)
        return data, plain, paradigm

    def test_the_header_holds_the_task_and_the_courses(self, run):
        data, plain, paradigm = run
        task = json.loads(data.header.extensions[0].get_content())["SimulationSchedule"]["task"]
        assert task["paradigm"] == paradigm.to_dict()
        response = np.asarray(task["responses"]["Lac"])
        assert np.all(response[:8] == 0.0) and np.all(response[9:] > 0.99)
        assert np.allclose(task["courses"]["Lac"], 0.2 * response)
        assert np.allclose(task["courses"]["Glu"], 0.05 * np.asarray(task["responses"]["Glu"]))
        assert np.asarray(task["bold"])[-1] > 0.99
        assert not any(task["bold_narrowing_hz"]) and not any(task["bold_amplitude"])
        assert json.loads(plain.header.extensions[0].get_content())["SimulationSchedule"]["task"] is None

    def test_the_task_transients_carry_the_courses_and_rest_is_the_plain_run(self, run):
        data, plain, _ = run
        x, y = (np.asarray(image[:])[0, 0, 0] for image in (data, plain))
        # the rest transients are the plain run's; every task transient at the plateau differs
        # from it by the one course, whose spectrum is lactate's doublet and glutamate's lines
        # and nothing else (the one truth, the same lines otherwise)
        assert np.array_equal(x[:, :8], y[:, :8])
        course = x[:, 12:] - y[:, 12:]
        assert np.abs(course).max() > 0.0 and np.allclose(course, course[:, :1], atol=1e-6)
        f0 = 42.5774785 * 3.0
        ppm = 4.65 + np.fft.fftshift(np.fft.fftfreq(1024, 1.0 / 2000.0)) / f0
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(course[:, 0])))
        peaks = ppm[np.argsort(spectrum)[-4:]]
        assert np.all((np.abs(peaks - 1.31) < 0.08) | (np.abs(peaks - 2.35) < 0.12))
        assert spectrum[(ppm > 2.95) & (ppm < 3.10)].max() < 0.15 * spectrum.max()

    def test_the_reader_reads_the_courses_back(self, run):
        data, _, paradigm = run
        x = np.asarray(data[:])[0, 0, 0]
        reading = TaskReading.from_arrays(x[:, None, :], 1.0 / 2000.0, 42.5774785 * 3.0, 1.0,
                                          paradigm, rise_s=0.5)
        # the response lags the task by a transient, so the onset transient counts as rest; the
        # glutamate band holds glutamine too, so it rises by part of glutamate's twentieth
        assert reading.rest_transients == 9 and reading.task_transients >= 14
        assert reading.over_tcr["Lac"] == pytest.approx(0.2, abs=0.06)
        assert 0.02 < reading.over_tcr["Glu"] < 0.06
        assert abs(reading.over_tcr["NAA"]) < 0.01 and abs(reading.effects["tCr"]) < 0.01
