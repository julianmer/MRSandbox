####################################################################################################
#                                         test_editing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Editing gate: J-difference editing emerges from the density matrix (no editing          #
#          specific physics exists to test): the GABA multiplet at 3.01 ppm differs strongly       #
#          between ON and OFF, uncoupled singlets cancel exactly in the difference, and the        #
#          multiplexed acquisition stacks conditions on DIM_EDIT against one shared truth.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, CoherenceFilter, Engine, Hercules, Hermes, MegaPress, Multiplex,
                       Params, Scanner, SelectivePulse, Sequence, SpinSystem)

QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_deg=0.0))


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


@pytest.fixture(scope="module")
def acquisition():
    multiplex = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                          MegaPress.pair(), params=QUIET)
    return multiplex.acquire(position_mm=(10.0, -20.0, 5.0), size_mm=20.0,
                             seed=7, npoints=2048)


@pytest.fixture(scope="module")
def hermes_acquisition():
    multiplex = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                          Hermes.scheme(), combinations=Hermes.combinations, params=QUIET)
    return multiplex, multiplex.acquire(position_mm=(10.0, -20.0, 5.0), size_mm=20.0,
                                        seed=7, npoints=512)


def render(engine, sequence, name):
    freqs, amps = engine.lines(sequence, *SpinSystem.from_database(name))
    ppm, spec = engine.render_spectrum(freqs, amps, 127.73, npoints=4096, sw_hz=4000.0, lw_hz=3.0)
    return np.asarray(ops.convert_to_numpy(ppm)), np.real(ops.convert_to_numpy(spec))


def edited_pair(engine, name):
    spectra = {}
    for condition, sequence in MegaPress.pair().items():
        ppm, spectra[condition] = render(engine, sequence, name)
    return ppm, spectra["on"] - spectra["off"], spectra["off"]


def hadamard_combos(engine, scheme_cls, name):
    spectra = {}
    for condition, sequence in scheme_cls.scheme().items():
        ppm, spectra[condition] = render(engine, sequence, name)
    combos = {cname: sum(weights[c] * spectra[c] for c in spectra)
              for cname, weights in scheme_cls.combinations.items()}
    return ppm, combos, spectra


#**************************************************************************************************#
#                                         Class TestEditing                                        #
#**************************************************************************************************#
class TestEditing:

    #********************#
    #   gaba is edited   #
    #********************#
    def test_gaba_shows_the_edited_difference(self, engine):
        ppm, diff, off = edited_pair(engine, "GABA")
        window = np.abs(ppm - 3.01) < 0.15
        assert np.abs(diff[window]).max() > 0.5 * np.abs(off[window]).max()

    #************************#
    #   singlets subtract    #
    #************************#
    def test_uncoupled_singlets_cancel_exactly(self, engine):
        ppm, diff, off = edited_pair(engine, "Cr")
        window = np.abs(ppm - 3.03) < 0.1
        assert np.abs(diff[window]).max() < 1e-5 * np.abs(off[window]).max()

    #*******************#
    #   the sequence    #
    #*******************#
    def test_timing_matches_the_typed_source(self):
        on = MegaPress()
        assert on.taus_s == (0.005, 0.017, 0.017, 0.017, 0.012)
        assert abs(sum(on.taus_s) - 0.068) < 1e-12  # the canonical TE 68 ms
        assert on.edit_ppm == 1.88
        assert MegaPress.pair()["off"].edit_ppm == 7.5


#**************************************************************************************************#
#                                       Class TestHadamard                                         #
#**************************************************************************************************#
#                                                                                                  #
# Hadamard-encoded editing: dual-band pulses rotate every band, the typed schemes match their      #
# sources, and the combinations separate the targets with exact crosstalk cancellation.            #
#                                                                                                  #
#**************************************************************************************************#
class TestHadamard:

    #***********************#
    #   dual band rotates   #
    #***********************#
    def test_dual_band_pulse_rotates_every_band(self, engine):
        probe = SpinSystem(name="probe", nucleus="1H", shifts_ppm=(1.9, 4.56, 3.0),
                           j_hz=((0.0, 0.0, 0.0),) * 3, scale=1.0, gamma_mhz_t=42.5774785)
        excite = Sequence(SelectivePulse(90.0, (1.9, 4.56), 60.0), CoherenceFilter(-1))
        freqs, amps = engine.lines(excite, probe)
        ppm = np.asarray(ops.convert_to_numpy(freqs)) / 127.73
        amps = np.abs(ops.convert_to_numpy(amps))
        inside = max(amps[np.abs(ppm - 1.9) < 0.1].sum(), amps[np.abs(ppm - 4.56) < 0.1].sum())
        assert amps[np.abs(ppm - 1.9) < 0.1].sum() > 0.5 * inside
        assert amps[np.abs(ppm - 4.56) < 0.1].sum() > 0.5 * inside
        assert amps[np.abs(ppm - 3.0) < 0.1].sum() < 1e-5 * inside

    #**********************#
    #   the typed tables   #
    #**********************#
    def test_schemes_match_the_typed_sources(self):
        hermes = Hermes.scheme()
        assert list(hermes) == ["a", "b", "c", "d"]
        assert hermes["a"].edit_ppm == (1.9, 4.56)
        assert hermes["d"].edit_ppm == (7.22,)
        hercules = Hercules.scheme()
        assert hercules["a"].edit_ppm == (4.58, 1.9)
        assert hercules["d"].edit_ppm == (4.18,)
        for scheme in (hermes, hercules):
            for sub in scheme.values():
                assert abs(sum(sub.taus_s) - 0.080) < 1e-12  # the published TE 80 ms
        for cls in (Hermes, Hercules):
            gaba, gsh = cls.combinations["gaba"], cls.combinations["gsh"]
            assert sum(gaba.values()) == sum(gsh.values()) == 0.0
            assert sum(gaba[c] * gsh[c] for c in cls.bands) == 0.0  # orthogonal rows

    #**************************#
    #   hermes edits cleanly   #
    #**************************#
    def test_hermes_separates_gaba_and_gsh(self, engine):
        ppm, combos, spectra = hadamard_combos(engine, Hermes, "GABA")
        window = np.abs(ppm - 3.01) < 0.15
        assert np.abs(combos["gaba"][window]).max() > 0.5 * np.abs(spectra["d"][window]).max()
        assert np.abs(combos["gsh"][window]).max() < 1e-5 * np.abs(combos["gaba"][window]).max()
        ppm, combos, spectra = hadamard_combos(engine, Hermes, "GSH")
        window = np.abs(ppm - 2.95) < 0.15
        assert np.abs(combos["gsh"][window]).max() > 0.5 * np.abs(spectra["d"][window]).max()
        assert np.abs(combos["gaba"][window]).max() < 1e-5 * np.abs(combos["gsh"][window]).max()

    #****************************#
    #   hercules edits cleanly   #
    #****************************#
    def test_hercules_separates_gaba_and_gsh(self, engine):
        ppm, combos, spectra = hadamard_combos(engine, Hercules, "GABA")
        window = np.abs(ppm - 3.01) < 0.15
        assert np.abs(combos["gaba"][window]).max() > 0.5 * np.abs(spectra["d"][window]).max()
        assert np.abs(combos["gsh"][window]).max() < 1e-5 * np.abs(combos["gaba"][window]).max()
        ppm, combos, spectra = hadamard_combos(engine, Hercules, "GSH")
        window = np.abs(ppm - 2.95) < 0.15
        assert np.abs(combos["gsh"][window]).max() > 0.5 * np.abs(spectra["d"][window]).max()
        assert np.abs(combos["gaba"][window]).max() < 1e-5 * np.abs(combos["gsh"][window]).max()

    #************************#
    #   singlets subtract    #
    #************************#
    def test_uncoupled_singlets_cancel_in_every_difference(self, engine):
        for cls in (Hermes, Hercules):
            ppm, combos, _ = hadamard_combos(engine, cls, "Cr")
            window = np.abs(ppm - 3.03) < 0.1
            reference = np.abs(combos["sum"][window]).max()
            assert np.abs(combos["gaba"][window]).max() < 1e-5 * reference
            assert np.abs(combos["gsh"][window]).max() < 1e-5 * reference


#**************************************************************************************************#
#                                       Class TestMultiplex                                        #
#**************************************************************************************************#
class TestMultiplex:
    pytestmark = pytest.mark.render  # full-scan renders, skipped on the tensorflow CI leg

    def test_conditions_stack_on_dim_edit(self, acquisition):
        assert acquisition.shape == (1, 1, 1, 2048, 2)
        assert acquisition.dim_tags[0] == "DIM_EDIT"
        header = acquisition.hdr_ext.to_dict()
        assert header["SimulationConditions"]["names"] == ["on", "off"]
        assert "GABA" in header["SimulationGroundTruth"]["params"]["conc"]

    #***********************#
    #   shared truth diff   #
    #***********************#
    def test_difference_spectrum_edits_the_voxel(self, acquisition):
        volume = np.asarray(acquisition[:])[0, 0, 0]
        diff_fid = volume[:, 0] - volume[:, 1]
        spec = np.abs(np.fft.fft(diff_fid))
        freqs = np.fft.fftfreq(2048, d=1.0 / 4000.0) % 4000.0
        ppm = freqs / 127.73
        at_gaba = spec[np.abs(ppm - 3.01) < 0.15].max()
        at_quiet = spec[(ppm > 8.0) & (ppm < 9.5)].max()
        # co-edited neighbours (NAA at 2.008 sits inside the ON band, as in vivo) put
        # real difference signal below 5 ppm, so the empty reference lives far upfield
        assert at_gaba > 5.0 * at_quiet

    #**********************#
    #   hadamard acquire   #
    #**********************#
    def test_hadamard_conditions_and_weights_are_recorded(self, hermes_acquisition):
        _, data = hermes_acquisition
        assert data.shape == (1, 1, 1, 512, 4)
        assert data.dim_tags[0] == "DIM_EDIT"
        record = data.hdr_ext.to_dict()["SimulationConditions"]
        assert record["names"] == ["a", "b", "c", "d"]
        assert record["combinations"] == Hermes.combinations

    #***********************#
    #   hadamard combine    #
    #***********************#
    def test_combine_applies_the_hadamard_weights(self, hermes_acquisition):
        multiplex, data = hermes_acquisition
        volume = np.asarray(data[:])[0, 0, 0]
        manual = volume[:, 0] + volume[:, 1] - volume[:, 2] - volume[:, 3]
        np.testing.assert_allclose(multiplex.combine(data, "gaba"), manual,
                                   atol=1e-5 * np.abs(volume).max())

    #*****************#
    #   replay path   #
    #*****************#
    def test_compiled_multiplex_reproduces_the_plain_acquisition(self, acquisition):
        # the opt-in replay path (a loop acquiring many times from one Multiplex) must
        # give the plain path's data for the same seed
        multiplex = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                              MegaPress.pair(), params=QUIET, compiled=True)
        again = multiplex.acquire(position_mm=(10.0, -20.0, 5.0), size_mm=20.0,
                                  seed=7, npoints=2048)
        reference = np.asarray(acquisition[:])
        np.testing.assert_allclose(np.asarray(again[:]), reference,
                                   atol=1e-5 * np.abs(reference).max())


#**************************************************************************************************#
#                                      Class TestTransients                                        #
#**************************************************************************************************#
#                                                                                                  #
# Repeated transients: the drift clock, jitter, and motion act per transient and are recorded.     #
#                                                                                                  #
#**************************************************************************************************#
class TestTransients:
    pytestmark = pytest.mark.render  # full-scan renders, skipped on the tensorflow CI leg

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def multiplex(**overrides):
        stable = dict(drift_hz_per_min=0.0, jitter_hz=0.0, phase_jitter_deg=0.0,
                      amplitude_sd=0.0, motion_rate_per_min=0.0,
                      motion_shift_hz=4.0, motion_phase_deg=20.0)
        stable.update(overrides)
        params = QUIET | Params(transients=stable)
        return Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                         MegaPress.pair(), params=params)

    #*****************#
    #   single shot   #
    #*****************#
    def test_single_transient_schedule_works(self):
        # n = 1 has no later transient to place a motion event on, so even a high
        # drawn rate must yield zero events and one valid FID
        data = self.multiplex(motion_rate_per_min=30.0).transients(
            (10, -20, 5), 20.0, schedule=1, seed=3, npoints=128)
        # the container squeezes the trailing singleton DIM_DYN axis
        assert np.asarray(data[:]).shape == (1, 1, 1, 128)
        assert np.abs(np.asarray(data[:])).max() > 0.0
        assert data.hdr_ext["SimulationSchedule"]["motion_events"] == []

    #******************#
    #   still scanner  #
    #******************#
    def test_stable_scanner_repeats_the_transient_exactly(self):
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=4, seed=3, npoints=256)
        assert data.shape == (1, 1, 1, 256, 4)
        assert data.dim_tags[0] == "DIM_DYN"
        volume = np.asarray(data[:])[0, 0, 0]
        for k in range(1, 4):
            np.testing.assert_array_equal(volume[:, k], volume[:, 0])

    #*************#
    #   drift     #
    #*************#
    def test_drift_accumulates_on_the_acquisition_clock(self):
        data = self.multiplex(drift_hz_per_min=6.0).transients(
            (10, -20, 5), 20.0, schedule=4, tr_s=2.0, seed=3, npoints=256)
        volume = np.asarray(data[:])[0, 0, 0]
        t = np.arange(256) / 4000.0
        for k in (1, 3):
            expected = volume[:, 0] * np.exp(2j * np.pi * (6.0 * k * 2.0 / 60.0) * t)
            assert np.abs(volume[:, k] - expected).max() < 1e-4 * np.abs(volume[:, 0]).max()
        drift = data.hdr_ext.to_dict()["SimulationSchedule"]["drift_hz"]
        assert np.allclose(drift, [0.0, 0.2, 0.4, 0.6])

    #*************#
    #   motion    #
    #*************#
    def test_motion_events_step_the_frequency_track(self):
        data = self.multiplex(motion_rate_per_min=30.0).transients(
            (10, -20, 5), 20.0, schedule=16, tr_s=2.0, seed=5, npoints=64)
        schedule = data.hdr_ext.to_dict()["SimulationSchedule"]
        events = schedule["motion_events"]
        drift = np.asarray(schedule["drift_hz"])
        assert events  # the forced rate produced at least one event
        steps = {}
        for event in events:  # events may share a transient; the step is their sum
            steps[event["transient"]] = steps.get(event["transient"], 0.0) + event["shift_hz"]
        for at, shift in steps.items():
            assert abs((drift[at] - drift[at - 1]) - shift) < 1e-9

    #**********************#
    #   mixed schedules    #
    #**********************#
    def test_interleaved_conditions_follow_the_schedule(self):
        schedule = ["on", "off", "on", "off"]
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=schedule,
                                           seed=3, npoints=256)
        volume = np.asarray(data[:])[0, 0, 0]
        np.testing.assert_array_equal(volume[:, 0], volume[:, 2])
        np.testing.assert_array_equal(volume[:, 1], volume[:, 3])
        assert np.abs(volume[:, 0] - volume[:, 1]).max() > 0.0  # editing differs
        assert data.hdr_ext.to_dict()["SimulationSchedule"]["conditions"] == schedule
