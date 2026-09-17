####################################################################################################
#                                          test_engine.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Engine capstone: pulse-acquire through the full stack reproduces textbook spectra -     #
#          lactate's 6.933 Hz doublet at 1.3142 ppm and its quartet at 4.0974 ppm in 3:1 ratio,    #
#          creatine's two singlets at 3:2, and a spectrum whose peak positions are independent of  #
#          the rendering grid.                                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import (Chess, Delay, Engine, Press, Pulse, PulseAcquire, SelectivePulse, Sequence,
                       ShapedPulse, SpinSystem)
from mrsandbox.sequences.events import CoherenceFilter

F0_MHZ = 42.5774785 * 3.0


#******************#
#   line helpers   #
#******************#
@pytest.fixture(scope="module")
def sim():
    return Engine(3.0)


def visible_lines(sim, *systems):
    """(|freq|, |amp|) of the non-silent pulse-acquire lines."""
    freqs, amps = (ops.convert_to_numpy(x) for x in sim.lines(PulseAcquire(), *systems))
    keep = np.abs(amps) > 1e-6 * np.abs(amps).max()
    return np.abs(freqs[keep]), np.abs(amps[keep])


def clusters(freqs, amps, center_ppm, half_width_ppm, tol_hz=0.5):
    """Merge lines within `tol_hz` into (center, weight) clusters inside a ppm window."""
    sel = (freqs > (center_ppm - half_width_ppm) * F0_MHZ) & (freqs < (center_ppm + half_width_ppm) * F0_MHZ)
    order = np.argsort(freqs[sel])
    centers, weights = [], []
    for f, a in zip(freqs[sel][order], amps[sel][order]):
        if centers and f - centers[-1] < tol_hz:
            centers[-1] = (centers[-1] * weights[-1] + f * a) / (weights[-1] + a)
            weights[-1] += a
        else:
            centers.append(f)
            weights.append(a)
    return np.array(centers), np.array(weights)


#**************************************************************************************************#
#                                      Class TestPulseAcquire                                      #
#**************************************************************************************************#
class TestPulseAcquire:

    #************************#
    #   lactate multiplets   #
    #************************#
    def test_lactate_doublet_and_quartet_at_3t(self, sim):
        freqs, amps = visible_lines(sim, *SpinSystem.from_database("Lac"))

        ch3, ch3_amp = clusters(freqs, amps, center_ppm=1.3142, half_width_ppm=0.05)
        assert len(ch3) == 2, f"expected a CH3 doublet, got {len(ch3)} clusters"
        assert abs((ch3[1] - ch3[0]) - 6.933) < 0.02
        assert abs(ch3.mean() / F0_MHZ - 1.3142) < 0.002

        ch, ch_amp = clusters(freqs, amps, center_ppm=4.0974, half_width_ppm=0.15)
        assert len(ch) == 4, f"expected a CH quartet, got {len(ch)} clusters"
        np.testing.assert_allclose(np.diff(ch), 6.933, atol=0.05)
        assert abs(ch3_amp.sum() / ch_amp.sum() - 3.0) < 0.15

    #*********************#
    #   creatine groups   #
    #*********************#
    def test_creatine_groups_sum_three_to_two(self, sim):
        freqs, amps = visible_lines(sim, *SpinSystem.from_database("Cr"))
        methyl, _ = clusters(freqs, amps, center_ppm=3.027, half_width_ppm=0.05)
        methylene, _ = clusters(freqs, amps, center_ppm=3.913, half_width_ppm=0.05)
        assert len(methyl) == 1 and len(methylene) == 1
        ratio = clusters(freqs, amps, 3.027, 0.05)[1].sum() / clusters(freqs, amps, 3.913, 0.05)[1].sum()
        assert abs(ratio - 1.5) < 1e-3

    #******************#
    #   grid freedom   #
    #******************#
    def test_spectrum_peak_position_is_grid_independent(self, sim):
        systems = SpinSystem.from_database("NAA")
        _, amps = sim.lines(PulseAcquire(), *systems)
        total = complex(np.asarray(ops.convert_to_numpy(amps)).sum())
        assert total.real > 0.999 * abs(total)  # receiver convention: in-phase signal is real
        peaks = []
        for npoints in (2048, 8192):
            ppm, spec = sim.spectrum(PulseAcquire(), *systems, npoints=npoints, sw_hz=2000.0, lw_hz=2.0)
            ppm = np.asarray(ops.convert_to_numpy(ppm))
            spec_np = ops.convert_to_numpy(spec)

            peaks.append(ppm[np.argmax(np.abs(spec_np))])
        assert abs(peaks[0] - 2.008) < 5e-3 and abs(peaks[1] - 2.008) < 5e-3


#**************************************************************************************************#
#                                       Class TestB1Scale                                          #
#**************************************************************************************************#
#                                                                                                  #
# The transmit scale follows the analytic flip laws through every pulse type.                      #
#                                                                                                  #
#**************************************************************************************************#
class TestB1Scale:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def water_amplitude(sim, sequence, b1_scale):
        from mrsandbox.phantom.water import WaterSource

        water = WaterSource.system(gamma_mhz_t=42.5774785)
        _, amps = sim.lines(sequence, water, b1_scale=b1_scale)
        return float(np.abs(ops.convert_to_numpy(amps)).sum())

    #********************#
    #   excitation law   #
    #********************#
    def test_excitation_follows_sin_of_the_scaled_flip(self, sim):
        full = self.water_amplitude(sim, PulseAcquire(), 1.0)
        for scale in (0.7, 0.85, 1.15):
            ratio = self.water_amplitude(sim, PulseAcquire(), scale) / full
            assert abs(ratio - np.sin(np.radians(scale * 90.0))) < 1e-4

    #**********************#
    #   suppression law    #
    #**********************#
    def test_suppression_residual_follows_cos_to_the_n(self, sim):
        full = self.water_amplitude(sim, PulseAcquire(), 1.0)
        for scale, n in ((0.8, 3), (0.9, 2)):
            residual = self.water_amplitude(sim, Chess(n) + PulseAcquire(), scale) / full
            expected = abs(np.cos(np.radians(scale * 90.0))) ** n * np.sin(np.radians(scale * 90.0))
            assert abs(residual - expected) < 1e-3

    #********************#
    #   shaped pulses    #
    #********************#
    def test_shaped_pulses_scale_with_the_transmit_field(self, sim):
        acquire = Sequence(ShapedPulse.gaussian(90.0, 0.010, offset_ppm=4.65), CoherenceFilter(-1))
        full = self.water_amplitude(sim, acquire, 1.0)
        ratio = self.water_amplitude(sim, acquire, 0.8) / full
        assert abs(ratio - np.sin(np.radians(0.8 * 90.0))) < 1e-2


#**************************************************************************************************#
#                                         Class TestOffset                                         #
#**************************************************************************************************#
#                                                                                                  #
# The frame offset: refocused by spin echoes, seen by selective and shaped pulses.                  #
#                                                                                                  #
#**************************************************************************************************#
class TestOffset:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def water():
        from mrsandbox.phantom.water import WaterSource

        return WaterSource.system(gamma_mhz_t=42.5774785)

    @staticmethod
    def magnitudes(lines):
        return np.abs(np.asarray(ops.convert_to_numpy(lines[1])))

    #******************#
    #   spin echoes    #
    #******************#
    def test_spin_echoes_refocus_the_offset(self, sim):
        # hard pulses are offset blind and the echo refocuses the offset phase, so
        # a PRESS spectrum is exactly the on-resonance one
        lac = SpinSystem.from_database("Lac")
        press = Press(te1_s=0.011, te2_s=0.019)
        on = sim.lines(press, *lac, offset_hz=40.0)
        off = sim.lines(press, *lac)
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(on[1])),
                                   np.asarray(ops.convert_to_numpy(off[1])), atol=1e-5)

    #***********************#
    #   selective pulses    #
    #***********************#
    def test_selective_pulse_misses_an_offset_spin(self, sim):
        water = self.water()
        excite = Sequence(SelectivePulse(90.0, offset_ppm=4.65, bandwidth_hz=40.0))
        full = self.magnitudes(sim.lines(excite, water)).sum()
        assert full > 0.1
        assert self.magnitudes(sim.lines(excite, water, offset_hz=15.0)).sum() == full  # inside
        assert self.magnitudes(sim.lines(excite, water, offset_hz=30.0)).sum() < 1e-9  # outside

    #********************#
    #   shaped pulses    #
    #********************#
    def test_shaped_pulse_offset_equals_a_shifted_carrier(self, sim):
        # spins offset by delta under a carrier at c see what on-resonance spins see
        # under a carrier at c + delta; the carrier ramp is sampled at the step
        # midpoints while the frame term is exact, so the identity holds to the step
        # discretization (6e-5 at 64 steps, 1.4e-5 at 512, the float32 floor)
        water = self.water()
        delta_hz = 200.0
        on_water = Sequence(ShapedPulse.gaussian(90.0, 0.002, offset_ppm=4.65))
        shifted = Sequence(ShapedPulse.gaussian(90.0, 0.002, offset_ppm=4.65 + delta_hz / F0_MHZ))
        offset = self.magnitudes(sim.lines(on_water, water, offset_hz=delta_hz))
        carrier = self.magnitudes(sim.lines(shifted, water))
        assert offset.sum() > 0.1
        np.testing.assert_allclose(offset, carrier, atol=1e-4)
        assert np.abs(offset - self.magnitudes(sim.lines(on_water, water))).max() > 1e-2


#**************************************************************************************************#
#                                     Class TestRenderPrecision                                    #
#**************************************************************************************************#
class TestRenderPrecision:

    #********************#
    #   long fid phase   #
    #********************#
    def test_long_fid_tail_phase_stays_folded(self, sim):
        # the rendered oscillation must track exp(i 2 pi f t) against a float64 oracle
        # on long FIDs: the factorized render folds its block phases on the host in
        # float64, so the tail phase error sits near 5e-5 rad, where any float32
        # product of f and t (the dense fold at 2.6e-3, a power table built in the
        # graph at 3e-3, the unfolded form at 4.3e-3) cannot; the gate holds the
        # factorized floor with margin and catches a regression to any of those
        npoints, dwell = 8192, 1.0 / 4000.0
        t64 = np.arange(npoints, dtype=np.float64) * dwell
        for f0 in (-1907.3, 433.1, 1990.7):
            f = np.asarray([f0], dtype=np.float32)
            a = np.asarray([1.0], dtype=np.complex64)
            _, fid = sim.render_fid(ops.convert_to_tensor(f), ops.convert_to_tensor(a),
                                    npoints=npoints, dwell_s=dwell, lw_hz=0.0)
            fid = np.asarray(ops.convert_to_numpy(fid), dtype=np.complex128)
            oracle = np.exp(1j * 2.0 * np.pi * np.float64(f[0]) * t64)
            assert np.abs(np.angle(fid / oracle)).max() < 5e-4

    #**********************#
    #   factorized render  #
    #**********************#
    def test_factorized_render_matches_the_dense_oracle(self, sim):
        # many Lorentzian lines with per-line widths against a float64 dense sum, at a
        # point count that is not a multiple of the block (the table is cropped), and
        # the Voigt path against its own oracle
        rng = np.random.default_rng(0)
        L, npoints, dwell = 700, 1000, 1.0 / 4000.0
        f = rng.uniform(-1900.0, 1900.0, L).astype(np.float32)
        a = (rng.normal(size=L) + 1j * rng.normal(size=L)).astype(np.complex64)
        lw = rng.uniform(1.0, 30.0, L).astype(np.float32)
        t64 = np.arange(npoints, dtype=np.float64) * dwell
        oracle = np.sum(a[:, None].astype(np.complex128)
                        * np.exp((1j * 2.0 * np.pi * f[:, None].astype(np.float64)
                                  - np.pi * lw[:, None].astype(np.float64)) * t64[None, :]),
                        axis=0)
        _, fid = sim.render_fid(ops.convert_to_tensor(f), ops.convert_to_tensor(a),
                                npoints=npoints, dwell_s=dwell, lw_hz=ops.convert_to_tensor(lw))
        fid = np.asarray(ops.convert_to_numpy(fid), dtype=np.complex128)
        # the float32 accumulation floor over hundreds of lines: 2e-5 measured for the
        # factorized graph, 1e-5 for the dense one
        assert np.abs(fid - oracle).max() < 5e-5 * np.abs(oracle).max()
        gauss = rng.uniform(5.0, 40.0, L).astype(np.float32)
        g = (np.pi * gauss[:, None].astype(np.float64)) ** 2 / (4.0 * np.log(2.0))
        voigt_oracle = np.sum(a[:, None].astype(np.complex128)
                              * np.exp((1j * 2.0 * np.pi * f[:, None].astype(np.float64)
                                        - np.pi * lw[:, None].astype(np.float64)) * t64[None, :]
                                       - g * t64[None, :] ** 2), axis=0)
        _, voigt = sim.render_fid(ops.convert_to_tensor(f), ops.convert_to_tensor(a),
                                  npoints=npoints, dwell_s=dwell, lw_hz=ops.convert_to_tensor(lw),
                                  gauss_hz=ops.convert_to_tensor(gauss))
        voigt = np.asarray(ops.convert_to_numpy(voigt), dtype=np.complex128)
        assert np.abs(voigt - voigt_oracle).max() < 1e-4 * np.abs(voigt_oracle).max()


#**************************************************************************************************#
#                                        Class TestGraph                                           #
#**************************************************************************************************#
#                                                                                                  #
# One compiled graph per sequence structure: timings are inputs, so a TE series replays, a mutated #
# event replays its new physics, and a timing gradient flows.                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestGraph:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def eager(engine, sequence, system):
        _, amps = engine.lines(sequence, system.with_relaxation(t1_s=1.2, t2_s=0.2))
        return np.asarray(ops.convert_to_numpy(amps))

    @staticmethod
    def replay(engine, sequence, system):
        args = [ops.convert_to_tensor(np.float32(v)) for v in (1.2, 0.2, 1.0, 0.0)]
        _, amps = engine.compiled(sequence, system)(*args)
        return np.asarray(ops.convert_to_numpy(amps))

    #**********************#
    #   one graph per TE   #
    #**********************#
    def test_a_te_series_shares_one_graph(self):
        from mrsandbox import SemiLaser

        engine = Engine(3.0)
        lac = SpinSystem.from_database("Lac")[0]
        # float32 with the compiled and the eager fusion orders: 1e-8 on jax, a few
        # 1e-6 on tensorflow
        for te2 in (0.019, 0.024, 0.060):
            sequence = Press(te1_s=0.011, te2_s=te2)
            np.testing.assert_allclose(self.replay(engine, sequence, lac),
                                       self.eager(engine, sequence, lac), atol=1e-5)
        assert len(engine._compiled) == 1
        # waveform arrays are timings too: an adiabatic design replays at another TE
        # (four 256-step propagator products with an eigendecomposition per step in
        # float32: compiled and eager agree to 1e-4 on jax and 2e-3 relative on
        # tensorflow, whose XLA and eager eigh kernels differ; the engine's oracle
        # gate is 1e-3 relative)
        for te in (0.030, 0.040):
            sequence = SemiLaser(te_s=te)
            eager = self.eager(engine, sequence, lac)
            np.testing.assert_allclose(self.replay(engine, sequence, lac), eager,
                                       atol=3e-3 * np.abs(eager).max())
        assert len(engine._compiled) == 2

    #****************#
    #   mutation     #
    #****************#
    def test_a_mutated_event_replays_its_new_physics(self):
        engine = Engine(3.0)
        lac = SpinSystem.from_database("Lac")[0]
        delay = Delay(0.005)
        echo = Sequence(Pulse(90.0), CoherenceFilter(+1), delay, Pulse(180.0, phase_deg=90.0),
                        CoherenceFilter(-1), delay)
        before = self.replay(engine, echo, lac)
        delay.t_s = 0.072
        after = self.replay(engine, echo, lac)
        assert np.abs(after - before).max() > 0.1
        np.testing.assert_allclose(after, self.eager(engine, echo, lac), atol=1e-5)

    #*******************#
    #   timing gradient #
    #*******************#
    def test_the_graph_is_differentiable_in_a_timing(self, needs_autograd):
        from nifti_mrs_plus.backend import value_and_grad

        engine = Engine(3.0)
        lac = SpinSystem.from_database("Lac")[0]
        fn, _ = engine.graph(Press(te1_s=0.011, te2_s=0.019), ((lac,),))
        args = [ops.convert_to_tensor(np.asarray(v, dtype=np.float32)) for v in ([1.2], [0.2], 1.0, 0.0)]

        def observable(te1):
            timings = Engine.tensors(Press(te1_s=te1, te2_s=0.019).timings())
            _, amps = fn(*args, timings)
            return ops.sum(ops.abs(amps))

        te1 = ops.convert_to_tensor(np.float32(0.011))
        value, grad = value_and_grad(observable)(te1)
        h = 1e-4
        finite = (float(observable(ops.convert_to_tensor(np.float32(0.011 + h))))
                  - float(observable(ops.convert_to_tensor(np.float32(0.011 - h))))) / (2.0 * h)
        assert abs(float(grad) - finite) < 0.02 * abs(finite)
