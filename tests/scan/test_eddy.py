####################################################################################################
#                                           test_eddy.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The eddy currents a design's gradients induce: the timeline places the trapezoids, the  #
#          exponential response reproduces the step law, the readout kernel carries the analytic   #
#          phase, the box dephasing, and the asymmetric line they make, and the measured 3 T       #
#          response puts the consensus-sized sidebands on unsuppressed water.                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from keras import ops

from mrsandbox import Brain, Engine, Params, Scanner, Unsuppressed, VirtualScan
from mrsandbox.phantom.water import WaterSource
from mrsandbox.resources import GIRF
from mrsandbox.scan.eddy import (TIME_STEP_S, exponential_response, induced_fields, induced_phase,
                                 induced_sequence, readout_kernel)
from mrsandbox.sequences.events import (Delay, Gradient, InducedKick, Pulse, SelectivePulse,
                                        Sequence, ShapedPulse)
from mrsandbox.sequences.timeline import gradient_waveforms

GAMMA = 42.5774785
QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                eddy_transient_rad=0.0),
               gradients=dict(b0_short_hz_per_mt_m=0.0, b0_long_hz_per_mt_m=0.0,
                              linear_short=0.0, linear_long=0.0))


def crushed_echo(moment, te_s=0.030):
    crusher = Gradient(moment, 20.0, axis="z")
    return Sequence(Pulse(90.0), Delay(te_s / 2.0), crusher, Pulse(180.0, phase_deg=90.0),
                    crusher, Delay(te_s / 2.0))


#**************************************************************************************************#
#                                        Class TestTimeline                                        #
#**************************************************************************************************#
class TestTimeline:

    def test_kicks_play_as_the_shortest_trapezoids_of_their_moment(self):
        scanner = Scanner.preset("generic_3t")
        # a 30 mT/m, 2 ms crusher exceeds the triangle limit: a flat top at 40 mT/m
        t, waves = gradient_waveforms(crushed_echo(GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3), scanner,
                                      TIME_STEP_S, 0.040)
        z = waves["z"]
        assert set(waves) == {"z"} and abs(z.max() - scanner.max_grad_mt_m) < 1e-6
        area = np.trapezoid(z, t) * 1e-3  # T s / m, both crushers
        assert abs(area - 2.0 * 30e-3 * 2e-3) < 1e-3 * 2.0 * 30e-3 * 2e-3
        # the crusher pair straddles the 180 at TE/2: the first ends there, the second
        # starts there
        assert abs(t[z > 0.5][0] - (0.015 - 2.0 * 0.04 / 200.0 - 0.0013)) < 2.0 * TIME_STEP_S
        assert abs(t[z > 0.5][-1] - (0.015 + 2.0 * 0.04 / 200.0 + 0.0013)) < 2.0 * TIME_STEP_S
        # a signed moment plays negative
        _, negative = gradient_waveforms(Sequence(Gradient(-0.5, 20.0), Delay(0.01)), scanner,
                                         TIME_STEP_S, 0.010)
        assert negative["x"].min() < 0.0 and negative["x"].max() <= 0.0

    def test_slice_gradients_are_flat_under_the_pulse_with_ramps_around(self):
        scanner = Scanner.preset("generic_3t")
        pulse = ShapedPulse.sinc(90.0, 0.002, gradient_hz_per_mm=GAMMA * 10.0, gradient_axis="y")
        t, waves = gradient_waveforms(Sequence(Delay(0.005), pulse, Delay(0.005)), scanner,
                                      TIME_STEP_S, 0.012)
        y = waves["y"]
        flat = (t >= 0.005) & (t <= 0.007)
        np.testing.assert_allclose(y[flat], 10.0, atol=1e-6)
        rise = 10e-3 / scanner.max_slew_t_m_s
        assert y[np.argmin(np.abs(t - (0.005 - rise / 2.0)))] < 6.0  # mid ramp
        assert y[t < 0.005 - rise - TIME_STEP_S].max() == 0.0

    def test_a_modulated_slice_gradient_rides_on_the_flat_top(self):
        scanner = Scanner.preset("generic_3t")
        pulse = ShapedPulse.goia_wurst(0.0045, 10000.0, GAMMA * 15.0,
                                       gradient_hz_per_mm=GAMMA * 10.0, gradient_axis="y")
        t, waves = gradient_waveforms(Sequence(Delay(0.005), pulse, Delay(0.005)), scanner,
                                      TIME_STEP_S, 0.015)
        y = waves["y"]
        assert abs(y[np.argmin(np.abs(t - 0.00725))] - 1.5) < 0.05  # 1 - f at the centre
        assert abs(y[np.argmin(np.abs(t - 0.00502))] - 10.0) < 0.2  # the full gradient at the edge
        assert y.max() <= 10.0 + 1e-6


#**************************************************************************************************#
#                                        Class TestResponse                                        #
#**************************************************************************************************#
class TestResponse:

    def test_a_step_decays_exponentially_in_both_terms(self):
        dt = 1e-5
        waveform = np.where(np.arange(2000) >= 500, 20.0, 0.0)  # a 20 mT/m step at 5 ms
        b0, error = induced_fields(waveform, dt, [0.003], [0.4], [0.01])
        t = (np.arange(2000) - 500) * dt
        after = t >= 0.0
        np.testing.assert_allclose(b0[after], 0.4 * 20.0 * np.exp(-t[after] / 0.003), rtol=1e-6)
        np.testing.assert_allclose(error[after], -0.01 * 20.0 * np.exp(-t[after] / 0.003),
                                   rtol=1e-6)
        assert np.all(b0[~after] == 0.0)

    def test_a_train_carries_its_long_terms_into_every_repetition(self):
        # the steady state of a train against its brute force: five seconds of a
        # 0.5 s repetition played out, the last repetition is the one-repetition
        # response plus what the earlier ones left, D / (1 - exp(-TR / tau)) decaying
        dt, tr, tau = 1e-4, 0.5, 1.2
        one = np.zeros(int(round(tr / dt)))
        one[100:300] = 20.0  # a 20 mT/m, 20 ms lobe per repetition
        train = np.tile(one, 40)
        b0_train, error_train = induced_fields(train, dt, [tau], [0.4], [0.01])
        window = one[:1500]
        b0, error = induced_fields(window, dt, [tau], [0.4], [0.01], tr_s=tr)
        last = slice(39 * one.size, 39 * one.size + window.size)
        np.testing.assert_allclose(b0, b0_train[last], atol=2e-6 * np.abs(b0_train).max() + 1e-9)
        np.testing.assert_allclose(error, error_train[last], atol=1e-6)
        alone, _ = induced_fields(window, dt, [tau], [0.4], [0.01])
        # and it matters: the carried field is twice what the lone lobe leaves behind it
        assert np.abs(b0 - alone)[-1] > 1.5 * np.abs(alone[-1])


#**************************************************************************************************#
#                                         Class TestKernel                                         #
#**************************************************************************************************#
class TestKernel:

    def test_zero_terms_leave_the_acquisition_alone(self):
        scanner = Scanner.preset("generic_3t")
        kernel = readout_kernel(crushed_echo(3.7), scanner, (0.0, 0.0, 0.0), 20.0, 256, 2.5e-4,
                                exponential_response((0.01, 1.0), (0.0, 0.0), (0.0, 0.0)))
        np.testing.assert_array_equal(kernel, np.ones(256, dtype=np.complex64))

    def test_receiver_delay_reads_the_same_kernel_later(self):
        # the phase counts from the readout's start, so a delay of k dwells is the
        # prompt kernel advanced by k samples, off centre with both terms
        scanner = Scanner.preset("generic_3t")
        design = crushed_echo(GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3)
        response = exponential_response((0.05, 0.5), (0.5, 0.2), (0.002, 0.001))
        prompt = readout_kernel(design, scanner, (0.0, 0.0, 15.0), 20.0, 264, 2.5e-4, response)
        late = readout_kernel(design, scanner, (0.0, 0.0, 15.0), 20.0, 256, 2.5e-4, response,
                              start_s=8 * 2.5e-4)
        assert np.abs(prompt - 1.0).max() > 1e-2
        np.testing.assert_allclose(late, prompt[8:], atol=1e-5)

    def test_b0_term_carries_the_analytic_step_phase(self):
        # one long-constant B0 term after a crusher pair whose ramps are short against
        # it: the phase over the readout integrates b0 G exp(-t / tau) from the
        # crusher's falling edge, i.e. 2 pi b0 G tau (exp(-t0 / tau) - exp(-(t0 + t) / tau))
        # per edge, the two edges of each trapezoid (rise and fall) cancelling but for
        # their separation
        scanner = Scanner.preset("generic_3t")
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        tau, b0 = 0.200, 0.5
        kernel = readout_kernel(crushed_echo(moment), scanner, (0.0, 0.0, 0.0), 20.0, 512, 2.5e-4,
                                exponential_response((tau, 1.0), (b0, 0.0), (0.0, 0.0)))
        t, waves = gradient_waveforms(crushed_echo(moment), scanner, TIME_STEP_S, 0.030)
        steps = np.diff(waves["z"], prepend=0.0)
        readout = 0.030 + np.arange(512) * 2.5e-4
        expected = np.zeros(512)
        for k in np.nonzero(steps)[0]:
            since = readout - t[k]
            expected += b0 * steps[k] * tau * (np.exp(-(0.030 - t[k]) / tau) - np.exp(-since / tau))
        np.testing.assert_allclose(np.unwrap(np.angle(kernel)), 2.0 * np.pi * expected, atol=2e-3)
        np.testing.assert_allclose(np.abs(kernel), 1.0, atol=1e-6)

    def test_linear_term_dephases_the_box_as_a_sinc_and_phases_the_position(self):
        scanner = Scanner.preset("generic_3t")
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        tau, fraction = 0.050, 0.02
        response = exponential_response((tau, 1.0), (0.0, 0.0), (fraction, 0.0))
        at_centre = readout_kernel(crushed_echo(moment), scanner, (0.0, 0.0, 0.0), 20.0, 512,
                                   2.5e-4, response)
        shifted = readout_kernel(crushed_echo(moment), scanner, (0.0, 0.0, 15.0), 20.0, 512,
                                 2.5e-4, response)
        t, waves = gradient_waveforms(crushed_echo(moment), scanner, TIME_STEP_S,
                                      0.030 + 512 * 2.5e-4)
        _, error = induced_fields(waves["z"], TIME_STEP_S, [tau], [0.0], [fraction])
        residual = np.interp(0.030 + np.arange(512) * 2.5e-4, t,
                             np.cumsum(error) * TIME_STEP_S) * GAMMA
        residual = residual - residual[0]  # cycles per mm over the readout
        np.testing.assert_allclose(at_centre.real, np.sinc(residual * 20.0), atol=1e-4)
        assert np.abs(at_centre).min() < 0.9  # a visible dephasing at this residual
        # off centre the same box picks up the phase of its position
        np.testing.assert_allclose(shifted, at_centre * np.exp(2j * np.pi * residual * 15.0),
                                   atol=1e-4)

    def test_the_line_turns_asymmetric_in_the_acquisition(self):
        brain = Brain.load("icbm152")
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        strong = Params(gradients=dict(tau_short_s=0.050, tau_long_s=1.0,
                                       b0_short_hz_per_mt_m=0.5, b0_long_hz_per_mt_m=0.0,
                                       linear_short=0.0, linear_long=0.0))
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        design = crushed_echo(moment)
        clean = VirtualScan(brain, Scanner.preset("generic_3t"), design, params=QUIET)
        warped = VirtualScan(brain, Scanner.preset("generic_3t"), design, params=QUIET | strong)
        ppm, s0 = clean.voxel(**kw).spectrum(seed=7, npoints=4096)
        _, s1 = warped.voxel(**kw).spectrum(seed=7, npoints=4096)
        window = np.abs(ppm - 2.01) < 0.15

        def asymmetry(spectrum):
            # the half-maximum widths on either side of the apex, in points (1 Hz)
            real = np.real(spectrum[window])
            apex = np.argmax(real)
            left, right = real[:apex], real[apex:]
            return abs((left > real[apex] / 2.0).sum() - (right > real[apex] / 2.0).sum())

        assert asymmetry(s0) == 0
        assert asymmetry(s1) >= 2
        # the same design carries the eddy kernel through the component split
        voxel = warped.voxel(**kw)
        whole = voxel.acquire(seed=7, npoints=512)
        parts = voxel.acquire_components(seed=7, npoints=512)
        np.testing.assert_allclose(np.asarray(parts[:]).sum(axis=-1), np.asarray(whole[:]),
                                   atol=1e-5 * np.abs(np.asarray(whole[:])).max())


#**************************************************************************************************#
#                                      Class TestInducedDesign                                     #
#**************************************************************************************************#
#                                                                                                  #
# The kicks a response puts on a design before the readout: the analytic echo phase and imbalance. #
#                                                                                                  #
#**************************************************************************************************#
class TestInducedDesign:

    @staticmethod
    def water_line(engine, sequence):
        # the detected water element (the largest amplitude) of a design
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        _, amps = engine.lines(sequence, water, isochromats=64)
        amps = np.asarray(ops.convert_to_numpy(amps))
        return amps[np.argmax(np.abs(amps))]

    def test_kicks_follow_every_timed_event_with_the_fields_integrals(self):
        scanner = Scanner.preset("generic_3t")
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        response = exponential_response((0.050,), (0.5,), (0.01,))
        design = induced_sequence(crushed_echo(moment), scanner, response, (0.0, 0.0, 10.0))
        kicks = [event for event in design.events if isinstance(event, InducedKick)]
        assert len(kicks) == 2 and len(design.events) == 8
        assert (isinstance(design.events[2], InducedKick)
                and isinstance(design.events[7], InducedKick))
        t, cycles, moments = induced_phase(crushed_echo(moment), scanner, response, 0.030)
        edges = np.array([0.0, 0.015, 0.030])
        expected_moment = np.diff(np.interp(edges, t, moments["z"]))
        expected_cycles = np.diff(np.interp(edges, t, cycles)) + expected_moment * 10.0
        np.testing.assert_allclose([k.moment_z for k in kicks], expected_moment, rtol=1e-6)
        np.testing.assert_allclose([k.cycles for k in kicks], expected_cycles, rtol=1e-6)
        assert all(k.moment_x == 0.0 and k.moment_y == 0.0 for k in kicks)
        # a silent response leaves the design itself
        assert induced_sequence(crushed_echo(moment), scanner,
                                exponential_response((0.05,), (0.0,), (0.0,)), (0.0, 0.0, 0.0)) \
            is not design and len(induced_sequence(crushed_echo(moment), scanner,
                                                   exponential_response((0.05,), (0.0,), (0.0,)),
                                                   (0.0, 0.0, 0.0)).events) == 6

    def test_a_shaped_pulse_carries_the_field_as_its_track(self):
        # the field over a shaped pulse rides on the pulse (the spins run off
        # resonance while it plays), its end value the integral a kick would carry,
        # and no kick follows the pulse
        scanner = Scanner.preset("generic_3t")
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        response = exponential_response((0.050,), (0.5,), (0.002,))
        crusher = Gradient(moment, 20.0, axis="z")
        plain = Sequence(Pulse(90.0), Delay(0.010), crusher, ShapedPulse.gaussian(180.0, 0.004),
                         crusher, Delay(0.010))
        design = induced_sequence(plain, scanner, response, (0.0, 0.0, 10.0))
        assert [type(e).__name__ for e in design.events] == [
            "Pulse", "Delay", "InducedKick", "Gradient", "ShapedPulse", "Gradient", "Delay",
            "InducedKick"]
        assert plain.events[3].induced is None  # the design itself is untouched
        track = design.events[4].induced
        assert track.shape == (4, 65)
        t, cycles, moments = induced_phase(plain, scanner, response, 0.024)
        over = lambda series: np.diff(np.interp([0.010, 0.014], t, series))[0]
        np.testing.assert_allclose(track[3, -1], over(moments["z"]), rtol=1e-5)
        np.testing.assert_allclose(track[0, -1], over(cycles) + 10.0 * over(moments["z"]),
                                   rtol=1e-5)
        assert np.all(np.diff(track[0]) != 0.0) and np.all(track[1:3] == 0.0)

    def test_the_b0_term_phases_the_echo_by_the_difference_of_its_halves(self):
        # the detected coherence carries q Phi per interval with q flipping at the
        # 180, so the echo's phase is 2 pi (Phi2 - Phi1): a field after the 180
        # advances the echo as faster precession does, the readout kernel's sign
        scanner, engine = Scanner.preset("generic_3t"), Engine(3.0)
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        response = exponential_response((0.050,), (0.5,), (0.0,))
        plain = crushed_echo(moment)
        kicked = induced_sequence(plain, scanner, response, (0.0, 0.0, 0.0))
        phase = np.angle(self.water_line(engine, kicked) / self.water_line(engine, plain))
        t, cycles, _ = induced_phase(plain, scanner, response, 0.030)
        phi1, phi2 = np.diff(np.interp([0.0, 0.015, 0.030], t, cycles))
        assert abs(phi2 - phi1) > 1e-3  # a visible phase
        np.testing.assert_allclose(phase, 2.0 * np.pi * (phi2 - phi1), atol=1e-4)
        np.testing.assert_allclose(np.abs(self.water_line(engine, kicked)),
                                   np.abs(self.water_line(engine, plain)), rtol=1e-5)

    def test_the_linear_term_unbalances_the_crushers_and_phases_the_position(self):
        # the residual moments of the two halves no longer cancel: the box dephases
        # as sinc of the net moment times its extent, and a voxel off centre picks up
        # the net moment's phase at its position
        scanner, engine = Scanner.preset("generic_3t"), Engine(3.0)
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        response = exponential_response((0.050,), (0.0,), (0.003,))
        plain = crushed_echo(moment)
        at_centre = induced_sequence(plain, scanner, response, (0.0, 0.0, 0.0))
        shifted = induced_sequence(plain, scanner, response, (0.0, 0.0, 30.0))
        t, _, moments = induced_phase(plain, scanner, response, 0.030)
        m1, m2 = np.diff(np.interp([0.0, 0.015, 0.030], t, moments["z"]))
        ratio = np.abs(self.water_line(engine, at_centre) / self.water_line(engine, plain))
        assert 0.9 < ratio < 0.999
        np.testing.assert_allclose(ratio, np.sinc((m2 - m1) * 20.0), atol=2e-3)
        phase = np.angle(self.water_line(engine, shifted) / self.water_line(engine, at_centre))
        np.testing.assert_allclose(phase, np.angle(np.exp(2j * np.pi * (m2 - m1) * 30.0)),
                                   atol=1e-4)


#**************************************************************************************************#
#                                    Class TestMeasuredResponse                                    #
#**************************************************************************************************#
#                                                                                                  #
# The deposit's 3 T Prisma response on a localized echo, gated when the archive is cached.         #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not GIRF.cached(), reason="NWS_Spectroscopy archive not cached (25 MB)")
class TestMeasuredResponse:

    POSITION = (30.0, 14.0, 35.0)  # about the paper's semi-LASER voxel, mm from the isocentre

    @staticmethod
    def localized_echo(edit_ppm=None, te_s=0.040):
        """A slice-selective spin echo (excitation on z, refocusing on x and y) with a
        crusher pair around each 180, so every axis plays; with `edit_ppm` two
        MEGA-style editing pulses sit in the echo halves."""
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        excite = ShapedPulse.sinc(90.0, 0.002, gradient_hz_per_mm=GAMMA * 10.0, gradient_axis="z")
        refocus = {axis: ShapedPulse.sinc(180.0, 0.003, gradient_hz_per_mm=GAMMA * 8.0,
                                          gradient_axis=axis) for axis in "xy"}
        edit = [] if edit_ppm is None else [SelectivePulse(180.0, offset_ppm=edit_ppm,
                                                           bandwidth_hz=60.0)]
        tau = te_s / 4.0
        return Sequence(excite, Gradient(-GAMMA * 10.0 * 0.001, 20.0, axis="z"), Delay(tau - 0.004),
                        Gradient(moment, 20.0, axis="x"), refocus["x"],
                        Gradient(moment, 20.0, axis="x"),
                        Delay(tau - 0.006), *edit, Delay(tau - 0.006),
                        Gradient(moment, 20.0, axis="y"), refocus["y"],
                        Gradient(moment, 20.0, axis="y"),
                        Delay(tau - 0.004), *edit)

    @pytest.mark.slow

    def test_sidebands_on_unsuppressed_water_sit_in_the_consensus_range(self):
        brain, scanner = Brain.load("icbm152"), Scanner.preset("generic_3t")
        engine, design = Engine(scanner), self.localized_echo()
        kw = dict(position_mm=self.POSITION, size_mm=20.0)
        clean = VirtualScan(brain, scanner, design, params=QUIET, water=Unsuppressed(),
                            engine=engine)
        measured = VirtualScan(brain, scanner, design, params=QUIET, water=Unsuppressed(),
                               girf=GIRF.load(), engine=engine)
        ppm, s0 = clean.voxel(**kw).spectrum(seed=7, npoints=4096)
        _, s1 = measured.voxel(**kw).spectrum(seed=7, npoints=4096)
        water = np.abs(s0).max()
        far = np.abs(ppm - 4.65) * scanner.b0_t * GAMMA > 500.0  # beyond every metabolite
        assert np.abs(s0)[far].max() < 0.01 * water
        # the sidebands the response adds: 0 to 2 percent of the parent line (Tkac et
        # al. 2021), here of the order of the metabolites
        sideband = np.abs(s1 - s0)[far].max() / water
        assert 1e-4 < sideband < 0.02, sideband

    @pytest.mark.slow

    def test_the_mega_difference_cancels_the_sidebands(self):
        brain, scanner = Brain.load("icbm152"), Scanner.preset("generic_3t")
        engine, girf = Engine(scanner), GIRF.load()
        on, off = self.localized_echo(edit_ppm=1.88), self.localized_echo(edit_ppm=7.5)
        # the same gradients play in both conditions, so the kernels are one
        kernels = [readout_kernel(design, scanner, self.POSITION, 20.0, 4096, 2.5e-4, girf)
                   for design in (on, off)]
        np.testing.assert_array_equal(kernels[0], kernels[1])
        assert np.abs(np.angle(kernels[0])).max() > 1e-3  # and not trivially so
        # so the sidebands (the response's addition to each condition) subtract away in
        # the difference, up to the modulation of what the editing pulses change
        sidebands = {}
        for name, design in (("on", on), ("off", off)):
            spectra = []
            for response in (None, girf):
                scan = VirtualScan(brain, scanner, design, params=QUIET, water=Unsuppressed(),
                                   girf=response, engine=engine)
                ppm, spectrum = scan.voxel(position_mm=self.POSITION, size_mm=20.0).spectrum(
                    seed=7, npoints=4096)
                spectra.append(spectrum)
            sidebands[name] = spectra[1] - spectra[0]
        far = np.abs(ppm - 4.65) * scanner.b0_t * GAMMA > 500.0
        assert np.abs(sidebands["on"])[far].max() > 0.0
        assert (np.abs(sidebands["on"] - sidebands["off"])[far].max()
                < 1e-2 * np.abs(sidebands["on"])[far].max())

    def test_the_fields_before_the_readout_phase_the_voxel_and_unbalance_its_crushers(self):
        # the design the voxel plays carries the kicks (scan.eddy.induced_sequence):
        # the crusher pairs' residuals phase an off-centre voxel by tens of degrees
        # and dephase the box by under a percent, while the isocentre sees the B0
        # term's echo residue alone
        brain, scanner = Brain.load("icbm152"), Scanner.preset("generic_3t")
        engine, girf = Engine(scanner), GIRF.load()
        moment = GAMMA * 1e6 * 30e-3 * 2e-3 * 1e-3
        design = crushed_echo(moment)
        for position, low, high in (((0.0, 0.0, 0.0), 0.0, 10.0), (self.POSITION, 10.0, 80.0)):
            kw = dict(position_mm=position, size_mm=20.0)
            clean = VirtualScan(brain, scanner, design, params=QUIET, water=Unsuppressed(),
                                engine=engine)
            measured = VirtualScan(brain, scanner, design, params=QUIET, water=Unsuppressed(),
                                   girf=girf, engine=engine)
            _, s0 = clean.voxel(**kw).spectrum(seed=7, npoints=4096)
            voxel = measured.voxel(**kw)
            _, s1 = voxel.spectrum(seed=7, npoints=4096)
            played = voxel.design(voxel.sample(7))
            assert len(played.events) == len(design.events) + 2  # a kick after each delay
            assert played is voxel.design(voxel.sample(7))  # cached per response
            k = np.argmax(np.abs(s0))
            assert low <= abs(np.degrees(np.angle(s1[k] / s0[k]))) <= high, position
            assert 0.97 < np.abs(s1[k]) / np.abs(s0[k]) < 0.999, position

    def test_the_first_order_terms_matter_off_centre(self):
        # the paper's correction removes the predicted phase from the FID; a B0-only
        # response leaves the position-dependent sidebands off centre and nothing at
        # the isocentre, where only the box dephasing (a magnitude) remains
        scanner, girf = Scanner.preset("generic_3t"), GIRF.load()
        b0_only = GIRF({axis: {"b0": girf.terms[axis]["b0"]} for axis in "xyz"}, name="b0 only")
        design = self.localized_echo()
        readout = np.arange(4096) * 2.5e-4
        envelope = np.exp(-readout / 0.080)  # unsuppressed water, T2* 80 ms (stated)
        reference = np.fft.fft(envelope)
        water = np.abs(reference).max()
        far = np.abs(np.fft.fftfreq(4096, 2.5e-4)) > 500.0
        residual = {}
        for name, position in (("centre", (0.0, 0.0, 0.0)), ("off", self.POSITION)):
            full = readout_kernel(design, scanner, position, 20.0, 4096, 2.5e-4, girf)
            truncated = readout_kernel(design, scanner, position, 20.0, 4096, 2.5e-4, b0_only)
            exact = np.fft.fft(envelope * full * np.exp(-1j * np.angle(full)))
            assert np.abs(exact - reference)[far].max() / water < 2e-5
            corrected = np.fft.fft(envelope * full * np.exp(-1j * np.angle(truncated)))
            residual[name] = np.abs(corrected - reference)[far].max() / water
        assert residual["off"] > 10.0 * residual["centre"]
        assert residual["off"] > 1e-4

