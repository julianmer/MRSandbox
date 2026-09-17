####################################################################################################
#                                           test_eddy.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The eddy currents a design's gradients induce: the timeline places the trapezoids, the  #
#          exponential response reproduces the step law, and the readout kernel carries the       #
#          analytic phase, the box dephasing, and the asymmetric line they make.                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox import Brain, Params, Scanner, VirtualScan
from mrsandbox.scan.eddy import TIME_STEP_S, induced_fields, readout_kernel
from mrsandbox.sequences.events import Delay, Gradient, Pulse, Sequence, ShapedPulse
from mrsandbox.sequences.timeline import gradient_waveforms

GAMMA = 42.5774785
QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
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
        np.testing.assert_allclose(error[after], -0.01 * 20.0 * np.exp(-t[after] / 0.003), rtol=1e-6)
        assert np.all(b0[~after] == 0.0)


#**************************************************************************************************#
#                                         Class TestKernel                                         #
#**************************************************************************************************#
class TestKernel:

    def test_zero_terms_leave_the_acquisition_alone(self):
        scanner = Scanner.preset("generic_3t")
        kernel = readout_kernel(crushed_echo(3.7), scanner, (0.0, 0.0, 0.0), 20.0, 256, 2.5e-4,
                                (0.01, 1.0), (0.0, 0.0), (0.0, 0.0))
        np.testing.assert_array_equal(kernel, np.ones(256, dtype=np.complex64))

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
                                (tau, 1.0), (b0, 0.0), (0.0, 0.0))
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
        at_centre = readout_kernel(crushed_echo(moment), scanner, (0.0, 0.0, 0.0), 20.0, 512,
                                   2.5e-4, (tau, 1.0), (0.0, 0.0), (fraction, 0.0))
        shifted = readout_kernel(crushed_echo(moment), scanner, (0.0, 0.0, 15.0), 20.0, 512,
                                 2.5e-4, (tau, 1.0), (0.0, 0.0), (fraction, 0.0))
        t, waves = gradient_waveforms(crushed_echo(moment), scanner, TIME_STEP_S, 0.030 + 512 * 2.5e-4)
        _, error = induced_fields(waves["z"], TIME_STEP_S, [tau], [0.0], [fraction])
        residual = np.interp(0.030 + np.arange(512) * 2.5e-4, t, np.cumsum(error) * TIME_STEP_S) * GAMMA
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
