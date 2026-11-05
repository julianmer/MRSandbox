####################################################################################################
#                                           test_cardiac.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/11/26                                                                                #
#                                                                                                  #
# Purpose: The heart over a run: the pulsatile frequency, phase and amplitude tracks at the        #
#          subject's rate, the line they leave in the tracks' spectra at the rate's alias, and the #
#          reader reading the amplitude back.                                                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from mrsandbox import Brain, Multiplex, Params, Press, Scanner
from mrsandbox.scan.run import RunReading

STILL = dict(drift_hz_per_min=0.0, ar_sigma_hz=0.0, resp_amp_hz=0.0, phase_jitter_deg=0.0,
             amp_jitter_frac=0.0, broaden_hz=0.0, events_per_min=0.0)
QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                eddy_transient_rad=0.0),
               oov=dict(fraction=0.0), run=dict(motion_mm=0.0))


#**************************************************************************************************#
#                                       Class TestCardiacReading                                   #
#**************************************************************************************************#
class TestCardiacReading:

    def test_the_reader_finds_the_heart_at_its_alias(self):
        # a track sampled every second carrying a heart of 1.2 Hz, which aliases to 0.2 Hz, at
        # 0.3 Hz beside a slow wander: the reader gives the amplitude back at the rate, and
        # nothing at a rate whose alias lies elsewhere
        rng = np.random.default_rng(4)
        t = np.arange(240) * 1.0
        track = 0.3 * np.cos(2.0 * np.pi * 1.2 * t + 0.7) + 0.02 * rng.standard_normal(t.size)
        reading = RunReading(frequency_hz=track, phase_deg=np.zeros(t.size),
                             amplitude=np.ones(t.size), tr_s=1.0, header={})
        assert reading.cardiac_amp_hz(1.2) == pytest.approx(0.3, abs=0.03)
        assert reading.cardiac_amp_hz(1.35) < 0.05
        record = reading.record_cardiac(1.2, "a synthetic run")
        assert record.leaf == "run.cardiac_amp_hz" and record.measured["alias_hz"] == 0.2


#**************************************************************************************************#
#                                         Class TestCardiacRun                                     #
#**************************************************************************************************#
@pytest.mark.render
class TestCardiacRun:

    def test_a_run_at_a_stated_rate_shows_the_line_at_its_alias(self):
        # a still scanner and subject but the heart: 1.2 Hz sampled every second aliases to
        # 0.2 Hz, where the frequency track's spectrum peaks and the amplitude track's too, the
        # latter scaled by the voxel's CSF
        params = QUIET | Params(run=dict(cardiac_hz=1.2, cardiac_amp_hz=0.4, cardiac_csf_amp=0.2))
        room = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                         {"scan": Press(te1_s=0.011, te2_s=0.019, tr_s=1.0)}, params=params)
        data = room.transients((0.0, -45.0, 35.0), 30.0, 64, tr_s=1.0, seed=3, npoints=512,
                               dwell_s=1.0 / 2000.0, synthesis=STILL)
        schedule = json.loads(data.header.extensions[0].get_content())["SimulationSchedule"]
        f = np.fft.rfftfreq(64, 1.0)
        csf = room.scans["scan"].voxel((0.0, -45.0, 35.0), 30.0).fractions.csf
        assert csf > 0.02
        for name, swing in (("frequency_hz", 0.4), ("amplitude", 0.2 * csf)):
            track = np.asarray(schedule[name])
            spectrum = np.abs(np.fft.rfft(track - track.mean())) * 2.0 / track.size
            assert f[spectrum[1:].argmax() + 1] == pytest.approx(0.2, abs=0.02)
            assert spectrum.max() == pytest.approx(swing, rel=0.1)
        assert schedule["cardiac"]["rate_hz"] == 1.2
        # the phase follows the frequency's excursion as the breathing's does
        assert np.corrcoef(schedule["phase_deg"], schedule["frequency_hz"])[0, 1] > 0.99
