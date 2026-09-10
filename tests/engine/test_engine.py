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

from mrsandbox import (Chess, Engine, Press, PulseAcquire, SelectivePulse, Sequence, ShapedPulse,
                       SpinSystem)
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
        # on long FIDs: folding the cycles before the 2 pi scaling keeps the worst
        # tail phase error at the single-product float32 floor (the unfolded form
        # reaches 4e-3 rad at 8192 points and this gate catches a regression to it)
        npoints, dwell = 8192, 1.0 / 4000.0
        t64 = np.arange(npoints, dtype=np.float64) * dwell
        for f0 in (-1907.3, 433.1, 1990.7):
            f = np.asarray([f0], dtype=np.float32)
            a = np.asarray([1.0], dtype=np.complex64)
            _, fid = sim.render_fid(ops.convert_to_tensor(f), ops.convert_to_tensor(a),
                                    npoints=npoints, dwell_s=dwell, lw_hz=0.0)
            fid = np.asarray(ops.convert_to_numpy(fid), dtype=np.complex128)
            oracle = np.exp(1j * 2.0 * np.pi * np.float64(f[0]) * t64)
            assert np.abs(np.angle(fid / oracle)).max() < 3.4e-3
