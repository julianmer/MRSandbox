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
import importlib.util

import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, CoherenceFilter, Engine, Hercules, Hermes, MegaPress, Multiplex,
                       Params, Scanner, SelectivePulse, Sequence, SpinSystem)

QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))


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
        ppm = 4.65 + np.fft.fftfreq(2048, d=1.0 / 4000.0) / 127.73               # the file's frame: the centre at 4.65 ppm
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

    #******************#
    #   components     #
    #******************#
    def test_components_sum_to_the_conditions_and_combine_alike(self, hermes_acquisition):
        # the second file of a training pair: DIM_EDIT then DIM_USER_0, the components
        # of every condition summing to it, and combine reading either file
        multiplex, data = hermes_acquisition
        parts = multiplex.acquire_components(position_mm=(10.0, -20.0, 5.0), size_mm=20.0,
                                             seed=7, npoints=512)
        names = parts.hdr_ext.to_dict()["SimulationComponents"]["names"]
        assert parts.dim_tags[:2] == ["DIM_EDIT", "DIM_USER_0"]
        assert parts.shape == (1, 1, 1, 512, 4, len(names))
        assert names[0] == "metabolites" and names[-1] == "noise" and "water" in names
        summed = np.asarray(parts[:]).sum(axis=-1)
        np.testing.assert_allclose(summed, np.asarray(data[:]), atol=1e-5 * np.abs(summed).max())
        combined = multiplex.combine(parts, "gaba")
        assert combined.shape == (512, len(names))
        np.testing.assert_allclose(combined.sum(axis=-1), multiplex.combine(data, "gaba"),
                                   atol=1e-5 * np.abs(combined).max())
        assert (parts.hdr_ext.to_dict()["SimulationConditions"]
                == data.hdr_ext.to_dict()["SimulationConditions"])
        # with the drawn noise the noise component carries it, the same draw as acquire
        noisy = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"), MegaPress.pair())
        whole = noisy.acquire(position_mm=(10.0, -20.0, 5.0), size_mm=20.0, seed=7, npoints=256)
        parts = noisy.acquire_components(position_mm=(10.0, -20.0, 5.0), size_mm=20.0, seed=7,
                                         npoints=256)
        assert np.abs(np.asarray(parts[:])[..., -1]).max() > 0.0
        summed = np.asarray(parts[:]).sum(axis=-1)
        np.testing.assert_allclose(summed, np.asarray(whole[:]), atol=1e-5 * np.abs(summed).max())

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
# Repeated transients: Augmentrum's instability tracks act per transient and are recorded.         #
#                                                                                                  #
#**************************************************************************************************#
class TestTransients:
    # full-scan renders, skipped on the tensorflow CI leg; the tracks come from
    # Augmentrum (the mrsandbox[augment] extra)
    pytestmark = [pytest.mark.render,
                  pytest.mark.skipif(importlib.util.find_spec("augmentrum") is None,
                                     reason="Augmentrum not installed (mrsandbox[augment])")]

    # a still scanner and subject: every TransientSynthesizer range pinned to zero
    STILL = dict(drift_hz_per_min=0.0, ar_sigma_hz=0.0, resp_amp_hz=0.0,
                 phase_coupling_deg_per_hz=0.0, phase_jitter_deg=0.0, amp_jitter_frac=0.0,
                 broaden_hz=0.0, events_per_min=0.0)

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def multiplex():
        return Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                         MegaPress.pair(), params=QUIET)

    def still(self, **overrides):
        return {**self.STILL, **overrides}

    #*****************#
    #   single shot   #
    #*****************#
    def test_single_transient_schedule_works(self):
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=1, seed=3, npoints=128,
                                           synthesis=self.still(events_per_min=30.0))
        # the container squeezes the trailing singleton DIM_DYN axis
        assert np.asarray(data[:]).shape == (1, 1, 1, 128)
        assert np.abs(np.asarray(data[:])).max() > 0.0
        assert len(data.hdr_ext["SimulationSchedule"]["frequency_hz"]) == 1

    #******************#
    #   still scanner  #
    #******************#
    def test_still_scanner_repeats_the_transient_exactly(self):
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=4, seed=3, npoints=256,
                                           synthesis=self.still())
        assert data.shape == (1, 1, 1, 256, 4)
        assert data.dim_tags[0] == "DIM_DYN"
        volume = np.asarray(data[:])[0, 0, 0]
        for k in range(1, 4):
            np.testing.assert_array_equal(volume[:, k], volume[:, 0])

    #*************#
    #   motion    #
    #*************#
    def test_a_displaced_head_changes_what_the_voxel_holds(self):
        # the head moves 8 mm after the second transient and comes back for the last:
        # the moved transients are the acquisition of the voxel placed 8 mm the other
        # way on the same subject, the others repeat the planned one exactly
        multiplex = self.multiplex()
        moved = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        data = multiplex.transients((10, -20, 5), 20.0, schedule=4, seed=3, npoints=256,
                                    synthesis=self.still(), displacement_mm=moved)
        volume = np.asarray(data[:])[0, 0, 0]
        np.testing.assert_array_equal(volume[:, 1], volume[:, 0])
        np.testing.assert_array_equal(volume[:, 3], volume[:, 0])
        elsewhere = multiplex.transients((2, -20, 5), 20.0, schedule=1, seed=3, npoints=256,
                                         synthesis=self.still())
        np.testing.assert_array_equal(volume[:, 2], np.asarray(elsewhere[:])[0, 0, 0])
        assert np.abs(volume[:, 2] - volume[:, 0]).max() > 1e-3 * np.abs(volume[:, 0]).max()
        assert data.hdr_ext["SimulationSchedule"]["displacement_mm"][2] == [8.0, 0.0, 0.0]
        still = multiplex.transients((10, -20, 5), 20.0, schedule=2, seed=3, npoints=256,
                                     synthesis=self.still())
        assert still.hdr_ext["SimulationSchedule"]["displacement_mm"] is None

    def test_a_turned_head_is_read_where_the_voxel_now_lies_in_it(self):
        # a quarter turn about z: the voxel at (10, -20, 5) reads the head at
        # R^T p = (-20, -10, 5); then a displacement on top
        multiplex = self.multiplex()
        turned = multiplex.transients((10, -20, 5), 20.0, schedule=2, seed=3, npoints=128,
                                      synthesis=self.still(), rotation_deg=[(0, 0, 0), (0, 0, 90)])
        volume = np.asarray(turned[:])[0, 0, 0]
        elsewhere = multiplex.transients((-20, -10, 5), 20.0, schedule=1, seed=3, npoints=128,
                                         synthesis=self.still())
        np.testing.assert_allclose(volume[:, 1], np.asarray(elsewhere[:])[0, 0, 0],
                                   atol=1e-5 * np.abs(volume).max())
        assert np.abs(volume[:, 1] - volume[:, 0]).max() > 1e-3 * np.abs(volume).max()
        record = turned.hdr_ext["SimulationSchedule"]
        assert record["rotation_deg"][1] == [0.0, 0.0, 90.0] and record["pivot_mm"] == [0.0, 0.0, 0.0]

    def test_motion_events_move_the_head_with_their_kicks(self):
        from augmentrum.augmentation import TransientSynthesizer

        if not hasattr(TransientSynthesizer(n_transients=2, tr_s=2.0), "last_events_"):
            pytest.skip("this Augmentrum does not record its motion events yet")
        # one event process: where Augmentrum's tracks kick, the head steps by the
        # asked size, for good after a persistent event; the header carries the path,
        # and a run without motion_mm is the run it always was
        multiplex = self.multiplex()
        kw = dict(schedule=24, seed=3, npoints=128, synthesis=self.still(events_per_min=6.0))
        plain = multiplex.transients((10, -20, 5), 20.0, **kw)
        moving = multiplex.transients((10, -20, 5), 20.0, motion_mm=(4.0, 4.0), **kw)
        record = moving.hdr_ext["SimulationSchedule"]
        assert record["frequency_hz"] == plain.hdr_ext["SimulationSchedule"]["frequency_hz"]
        path = np.asarray(record["displacement_mm"])
        assert path.shape == (24, 3) and record["motion_mm"] == [4.0, 4.0]
        kicked = np.asarray(record["frequency_hz"]) != 0.0
        assert kicked.any() and np.all(np.linalg.norm(path[~kicked], axis=1) == 0.0)
        assert np.linalg.norm(path[kicked], axis=1).max() >= 4.0 - 1e-9
        a, b = np.asarray(plain[:])[0, 0, 0], np.asarray(moving[:])[0, 0, 0]
        np.testing.assert_array_equal(b[:, ~kicked], a[:, ~kicked])
        assert np.abs(b[:, kicked] - a[:, kicked]).max() > 0.0

    #*************#
    #   drift     #
    #*************#
    def test_drift_accumulates_on_the_acquisition_clock(self):
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=4, tr_s=2.0, seed=3,
                                           npoints=256, synthesis=self.still(drift_hz_per_min=6.0))
        volume = np.asarray(data[:])[0, 0, 0]
        t = np.arange(256) / 4000.0
        for k in (1, 3):
            expected = volume[:, 0] * np.exp(2j * np.pi * (6.0 * k * 2.0 / 60.0) * t)
            assert np.abs(volume[:, k] - expected).max() < 1e-4 * np.abs(volume[:, 0]).max()
        schedule = data.hdr_ext.to_dict()["SimulationSchedule"]
        assert np.allclose(schedule["frequency_hz"], [0.0, 0.2, 0.4, 0.6])
        assert schedule["synthesis"]["module"].endswith("TransientSynthesizer")

    #********************#
    #   live envelope    #
    #********************#
    def test_default_envelope_breathes_and_reproduces(self):
        # Augmentrum's 3 T envelope: the frequency track is not flat, and the same
        # seed reproduces every track and every sample
        first = self.multiplex().transients((10, -20, 5), 20.0, schedule=16, tr_s=2.0, seed=5,
                                            npoints=64)
        again = self.multiplex().transients((10, -20, 5), 20.0, schedule=16, tr_s=2.0, seed=5,
                                            npoints=64)
        schedule = first.hdr_ext.to_dict()["SimulationSchedule"]
        assert np.ptp(schedule["frequency_hz"]) > 0.0
        assert len(schedule["broadening_hz"]) == 16
        np.testing.assert_array_equal(np.asarray(first[:]), np.asarray(again[:]))

    #**********************#
    #   mixed schedules    #
    #**********************#
    def test_interleaved_conditions_follow_the_schedule(self):
        schedule = ["on", "off", "on", "off"]
        data = self.multiplex().transients((10, -20, 5), 20.0, schedule=schedule, seed=3,
                                           npoints=256, synthesis=self.still())
        volume = np.asarray(data[:])[0, 0, 0]
        np.testing.assert_array_equal(volume[:, 0], volume[:, 2])
        np.testing.assert_array_equal(volume[:, 1], volume[:, 3])


#**************************************************************************************************#
#                                       Class TestLoopSurface                                      #
#**************************************************************************************************#
#                                                                                                  #
# One subject acquired condition by condition over steps, conditions added as a loop goes, all on  #
# one engine.                                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestLoopSurface:
    pytestmark = pytest.mark.render

    def test_conditions_steps_and_added_designs_share_one_subject_and_engine(self):
        loud = QUIET | Params(noise=dict(sigma=1.0))
        multiplex = Multiplex(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                              MegaPress.pair(), params=loud, compiled=True)
        assert len({id(scan.engine) for scan in multiplex.scans.values()}) == 1
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0, seed=7, npoints=512)
        on = multiplex.acquire(**kw, conditions=("on",), step=0)
        assert on.shape == (1, 1, 1, 512, 1)
        assert on.hdr_ext["SimulationConditions"]["names"] == ["on"]
        later = multiplex.acquire(**kw, conditions=("on",), step=1)
        assert on.hdr_ext["SimulationGroundTruth"]["params"] == later.hdr_ext["SimulationGroundTruth"]["params"]
        assert np.abs(np.asarray(on[:]) - np.asarray(later[:])).max() > 0.0
        both = multiplex.acquire(**kw, step=0)
        np.testing.assert_array_equal(np.asarray(both[:]).reshape(512, 2)[:, 0],
                                      np.asarray(on[:]).reshape(512))
        # a third condition of the same design family joins without a compile
        graphs = len(multiplex.scans["on"].engine._compiled)
        multiplex.add("off_far", MegaPress(edit_ppm=8.0))
        far = multiplex.acquire(**kw, conditions=("off_far",))
        assert far.hdr_ext["SimulationConditions"]["names"] == ["off_far"]
        assert len(multiplex.scans["on"].engine._compiled) == graphs


#**************************************************************************************************#
#                                   Class TestLocalizedMegaPress                                   #
#**************************************************************************************************#
#                                                                                                  #
# Editing with the localization played: the edited multiplet keeps its shape and loses the part    #
# of the voxel where the coupled partners are not refocused together.                              #
#**************************************************************************************************#
class TestLocalizedMegaPress:

    def test_edited_gaba_keeps_its_shape_and_pays_the_displacement(self):
        from mrsandbox import LocalizedMegaPress, SpinSystem

        f0_mhz = 3.0 * 42.5774785
        engine = Engine(3.0, isochromats=32)

        def spectrum(sequence, systems, n=2048, sw_hz=4000.0):
            total = 0.0
            for system in systems:
                freqs, amps = engine.lines(sequence, system)
                _, fid = engine.render_fid(freqs, amps, npoints=n, dwell_s=1.0 / sw_hz, lw_hz=4.0)
                # referenced to 0 ppm, a line at f hertz is exp(+i 2 pi f t)
                total = total + np.fft.fftshift(np.fft.fft(np.asarray(ops.convert_to_numpy(fid))))
            return np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sw_hz)) / f0_mhz, total

        gaba, naa = SpinSystem.from_database("GABA"), SpinSystem.from_database("NAA")
        edited = {}
        for label, pair in (("ideal", MegaPress.pair()), ("localized", LocalizedMegaPress.pair(f0_mhz))):
            ppm, on = spectrum(pair["on"], gaba)
            edited[label] = np.real(on - spectrum(pair["off"], gaba)[1])
        window = (ppm > 2.8) & (ppm < 3.2)
        assert np.corrcoef(edited["localized"][window], edited["ideal"][window])[0, 1] > 0.97
        # 0.96 kHz refocusing slabs put the 1.9 ppm partner 15 percent of the voxel off
        # on two axes: a sixth of the edited signal is lost (the spatial editing loss,
        # Edden and Barker 2007)
        assert 0.7 < edited["localized"][window].sum() / edited["ideal"][window].sum() < 0.95
        # the Gaussian editing pulse reaches the NAA methyl 16 Hz off its centre, and
        # what an editing pulse turns over is dephased: it all but leaves the ON spectrum
        pair = LocalizedMegaPress.pair(f0_mhz)
        ppm, on = spectrum(pair["on"], naa)
        methyl = (ppm > 1.9) & (ppm < 2.1)
        assert np.real(on)[methyl].max() < 0.05 * np.real(spectrum(pair["off"], naa)[1])[methyl].max()
        with pytest.raises(ValueError):
            LocalizedMegaPress(1.88, f0_mhz, taus_s=(0.003, 0.017, 0.017, 0.017, 0.012))

    def test_a_localized_hermes_still_separates_gaba_and_gsh(self):
        # dual-band Gaussians turn both bands over and nothing between them; with real
        # pulses and a real voxel the Hadamard combinations keep most of the edited
        # signal and leak a percent or so into each other, where ideal bands leak none
        from mrsandbox import LocalizedMegaPress

        f0_mhz = 3.0 * 42.5774785
        engine = Engine(3.0, isochromats=32)
        schemes = {"ideal": Hermes.scheme(), "localized": LocalizedMegaPress.scheme(Hermes, f0_mhz)}
        assert list(schemes["localized"]) == ["a", "b", "c", "d"]
        for name, target, other, centre in (("GABA", "gaba", "gsh", 3.01), ("GSH", "gsh", "gaba", 2.95)):
            edited = {}
            for label, scheme in schemes.items():
                spectra = {}
                for condition, sequence in scheme.items():
                    ppm, spectra[condition] = render(engine, sequence, name)
                window = np.abs(ppm - centre) < 0.15
                edited[label] = {k: sum(w[c] * spectra[c] for c in spectra)[window]
                                 for k, w in Hermes.combinations.items()}
            kept = edited["localized"][target].sum() / edited["ideal"][target].sum()
            leak = np.abs(edited["localized"][other]).max() / np.abs(edited["localized"][target]).max()
            assert 0.5 < kept < 1.0 and leak < 0.05, (name, kept, leak)
        with pytest.raises(ValueError):
            LocalizedMegaPress((1.9, 4.56, 7.2), f0_mhz)

    def test_mega_semi_laser_edits_with_its_pulses_at_opposite_signs(self):
        # 90 - AFP - AFP - edit - AFP - edit - AFP: one refocusing pulse between the
        # editing pulses, so their Bloch-Siegert phases cancel (water upright) and GABA
        # is edited; the schedule closes at TE 80 ms and not at 40
        from mrsandbox import LocalizedMegaSemiLaser, Pulse

        f0_mhz = 3.0 * 42.5774785
        engine = Engine(3.0, isochromats=32)
        design = LocalizedMegaSemiLaser(1.88, f0_mhz)
        assert abs(sum(design.taus_s) - 0.080) < 1e-12
        assert abs((design.taus_s[3] + design.taus_s[4]) - 0.040) < 1e-12  # the edits TE/2 apart
        assert design.feasible(Scanner.preset("siemens_prisma_3t"))
        water = SpinSystem(name="water", nucleus="1H", shifts_ppm=(4.65,), j_hz=((0.0,),), scale=2.0,
                           gamma_mhz_t=42.5774785)
        off = np.asarray(ops.convert_to_numpy(engine.lines(
            LocalizedMegaSemiLaser(7.5, f0_mhz, reference_ppm=4.65), water)[1]))
        plain = np.asarray(ops.convert_to_numpy(engine.lines(
            Sequence(Pulse(90.0), CoherenceFilter(-1)), water)[1]))
        k = np.argmax(np.abs(plain))
        assert 0.75 < abs(off[k] / plain[k]) < 0.95 and abs(np.degrees(np.angle(off[k] / plain[k]))) < 1.0
        taus = (0.005, 0.015, 0.025, 0.015, 0.020)
        edited = {}
        for label, pair in (("ideal", MegaPress.pair(taus_s=taus)),
                            ("laser", LocalizedMegaSemiLaser.pair(f0_mhz))):
            spectra = {}
            for condition, sequence in pair.items():
                ppm, spectra[condition] = render(engine, sequence, "GABA")
            window = np.abs(ppm - 3.01) < 0.2
            edited[label] = (spectra["on"] - spectra["off"])[window]
        assert 0.6 < edited["laser"].sum() / edited["ideal"].sum() < 1.0
        assert np.corrcoef(edited["laser"], edited["ideal"])[0, 1] > 0.9
        with pytest.raises(ValueError):
            LocalizedMegaSemiLaser(1.88, f0_mhz, te_s=0.040)
