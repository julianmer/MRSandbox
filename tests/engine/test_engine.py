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
    sel = ((freqs > (center_ppm - half_width_ppm) * F0_MHZ)
           & (freqs < (center_ppm + half_width_ppm) * F0_MHZ))
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
        ratio = (clusters(freqs, amps, 3.027, 0.05)[1].sum()
                 / clusters(freqs, amps, 3.913, 0.05)[1].sum())
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
            ppm, spec = sim.spectrum(PulseAcquire(), *systems, npoints=npoints, sw_hz=2000.0,
                                     lw_hz=2.0)
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
# The frame offset: refocused by spin echoes, seen by selective and shaped pulses.                 #
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
        # discretization
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
        # float64, so its tail phase error stays far below the milliradians any float32
        # product of f and t reaches; the threshold sits between the two
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
        # the float32 accumulation floor over hundreds of lines
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
        # float32 with the compiled and the eager fusion orders
        for te2 in (0.019, 0.024, 0.060):
            sequence = Press(te1_s=0.011, te2_s=te2)
            np.testing.assert_allclose(self.replay(engine, sequence, lac),
                                       self.eager(engine, sequence, lac), atol=1e-5)
        assert len(engine._compiled) == 1
        # waveform arrays are timings too: an adiabatic design replays at another TE
        # (four 256-step propagator products with an eigendecomposition per step in
        # float32, where the XLA and eager eigh kernels of tensorflow differ)
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
        args = [ops.convert_to_tensor(np.asarray(v, dtype=np.float32))
                for v in ([1.2], [0.2], 1.0, 0.0)]

        def observable(te1):
            timings = Engine.tensors(Press(te1_s=te1, te2_s=0.019).timings())
            _, amps = fn(*args, timings, ops.zeros((3,)))
            return ops.sum(ops.abs(amps))

        te1 = ops.convert_to_tensor(np.float32(0.011))
        value, grad = value_and_grad(observable)(te1)
        h = 1e-4
        finite = (float(observable(ops.convert_to_tensor(np.float32(0.011 + h))))
                  - float(observable(ops.convert_to_tensor(np.float32(0.011 - h))))) / (2.0 * h)
        assert abs(float(grad) - finite) < 0.02 * abs(finite)

    #*******************#
    #   batched draws   #
    #*******************#
    def test_batched_draws_equal_the_looped_replays(self):
        from mrsandbox.sequences.events import Gradient

        engine = Engine(3.0)
        lac = SpinSystem.from_database("Lac")[0]
        t1 = np.asarray([0.9, 1.2, 1.5, 2.0], dtype=np.float32)
        t2 = np.asarray([0.10, 0.20, 0.15, 0.30], dtype=np.float32)
        crusher = Gradient(3.7, 20.0)
        for design in (Press(te1_s=0.011, te2_s=0.019),
                       Sequence(Pulse(90.0), Delay(0.015), crusher, Pulse(180.0, phase_deg=90.0),
                                crusher, Delay(0.015))):
            one = engine.compiled(design, lac)
            many = engine.compiled(design, lac, batched=True)
            b1, offset = (ops.convert_to_tensor(np.float32(1.0)),
                          ops.convert_to_tensor(np.float32(0.0)))
            freqs, amps = many(ops.convert_to_tensor(t1), ops.convert_to_tensor(t2), b1, offset)
            amps = np.asarray(ops.convert_to_numpy(amps))
            assert amps.shape == (4, lac.dim ** 2)
            for k in range(4):
                f, a = one(ops.convert_to_tensor(t1[k]), ops.convert_to_tensor(t2[k]), b1, offset)
                np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(f)),
                                           np.asarray(ops.convert_to_numpy(freqs)), atol=1e-6)
                np.testing.assert_allclose(amps[k], np.asarray(ops.convert_to_numpy(a)), atol=1e-6)
            # the rows render to B FIDs in one call
            _, fids = engine.render_fids(freqs, ops.convert_to_tensor(amps), npoints=256,
                                         dwell_s=1.0 / 4000.0, lw_hz=5.0)
            assert tuple(fids.shape) == (4, 256)
        # one graph per design, the batched replay included
        assert len(engine._compiled) == 2

    #******************#
    #   framed graph   #
    #******************#
    def test_a_framed_graph_equals_the_full_one(self):
        # a played design's graph split at the transmit frame: the pulses once per (b1, offset),
        # the relaxation pass per draw, mapped over draws with the frame shared; the kept lines
        # are the full pruned graph's
        from mrsandbox.sequences.library import LocalizedPress

        engine = Engine(3.0, isochromats=8)
        groups = (tuple(SpinSystem.from_database("Lac")),)
        design = LocalizedPress(te1_s=0.011, te2_s=0.019, f0_mhz=F0_MHZ, tr_s=1.5)
        full, kept = engine.compiled_batch(design, groups, pruned=True)
        frame, lines, draws, kept_framed = engine.framed(design, groups)
        np.testing.assert_array_equal(kept, kept_framed)
        b1, offset = ops.convert_to_tensor(np.float32(0.9)), ops.convert_to_tensor(np.float32(5.0))
        held = frame(b1, offset)
        t1 = np.asarray([[0.9], [1.2], [1.5]], dtype=np.float32)
        t2 = np.asarray([[0.10], [0.25], [0.15]], dtype=np.float32)
        f_many, a_many = (np.asarray(ops.convert_to_numpy(v)) for v in draws(
            ops.convert_to_tensor(t1), ops.convert_to_tensor(t2), b1, offset, held))
        assert a_many.shape == (3, kept[-1])
        for k in range(3):
            args = (ops.convert_to_tensor(t1[k]), ops.convert_to_tensor(t2[k]), b1, offset)
            f_full, a_full = (np.asarray(ops.convert_to_numpy(v)) for v in full(*args))
            f_one, a_one = (np.asarray(ops.convert_to_numpy(v)) for v in lines(*args, held))
            np.testing.assert_array_equal(f_one, f_full)
            np.testing.assert_array_equal(f_many[k], f_full)
            scale = np.abs(a_full).max()
            assert np.abs(a_one - a_full).max() < 1e-5 * scale
            assert np.abs(a_many[k] - a_full).max() < 1e-5 * scale

    def test_a_pruned_graph_keeps_every_line_that_carries_signal(self):
        # the coherence filters silence most of the d*d elements exactly; the pruned graph gathers
        # the rest inside the jit, so a replay transfers a fraction of the elements and renders the
        # same spectrum, for any draw
        from mrsandbox import SpinSystem

        engine = Engine(3.0)
        groups = (tuple(SpinSystem.from_database("Lac")), tuple(SpinSystem.from_database("NAA")))
        design = Press(te1_s=0.011, te2_s=0.024, tr_s=2.0)
        full, bounds = engine.compiled_batch(design, groups)
        pruned, kept = engine.compiled_batch(design, groups, pruned=True)
        assert kept[-1] < bounds[-1] / 4 and len(kept) == len(bounds)
        rng = np.random.default_rng(5)
        for _ in range(6):
            args = (ops.convert_to_tensor(rng.uniform(0.8, 2.0, 2).astype(np.float32)),
                    ops.convert_to_tensor(rng.uniform(0.1, 0.4, 2).astype(np.float32)),
                    ops.convert_to_tensor(np.float32(rng.uniform(0.7, 1.3))),
                    ops.convert_to_tensor(np.float32(rng.uniform(-20.0, 20.0))))
            f_full, a_full = (np.asarray(ops.convert_to_numpy(v)) for v in full(*args))
            f_kept, a_kept = (np.asarray(ops.convert_to_numpy(v)) for v in pruned(*args))
            # per group: what the pruning drops carries nothing, and the spectrum is the same
            for g in range(2):
                fa, aa = f_full[bounds[g]:bounds[g + 1]], a_full[bounds[g]:bounds[g + 1]]
                fk, ak = f_kept[kept[g]:kept[g + 1]], a_kept[kept[g]:kept[g + 1]]
                assert (np.abs(aa).astype(np.float64).sum() - np.abs(ak).astype(np.float64).sum()
                        <= 1e-7 * np.abs(aa).sum())
                _, s_full = engine.render_fid(ops.convert_to_tensor(fa), ops.convert_to_tensor(aa),
                                              npoints=1024, lw_hz=4.0)
                _, s_kept = engine.render_fid(ops.convert_to_tensor(fk), ops.convert_to_tensor(ak),
                                              npoints=1024, lw_hz=4.0)
                s_full, s_kept = (np.asarray(ops.convert_to_numpy(v)) for v in (s_full, s_kept))
                assert np.abs(s_full - s_kept).max() <= 1e-5 * np.abs(s_full).max()

    def test_the_signal_index_persists_across_engines(self, backend, tmp_path):
        # the index is a pure function of the graph, so a second engine reads it back from the
        # cache directory instead of replaying the full graph: a tampered file coming back
        # verbatim proves the read, and proves the key reproduces across engines
        if backend != "jax":
            pytest.skip("the signal index persists by jax's lowered module")
        import jax

        from mrsandbox import SpinSystem

        before = jax.config.jax_compilation_cache_dir
        jax.config.update("jax_compilation_cache_dir", str(tmp_path))

        def no_exports():
            # the graph export would otherwise serve the next engine before the index store
            for file in (tmp_path / "exports").glob("*.npz"):
                file.unlink()

        try:
            groups = (tuple(SpinSystem.from_database("Lac")),)
            design = Press(te1_s=0.012, te2_s=0.02, tr_s=1.5)
            _, kept = Engine(3.0).graph(design, groups, pruned=True)
            (path,) = (tmp_path / "signal_index").glob("*.npz")
            stored = dict(np.load(path))
            np.savez(path, index=stored["index"][:-1], kept=stored["kept"] - 1)
            no_exports()
            _, tampered = Engine(3.0).graph(design, groups, pruned=True)
            assert tampered[-1] == kept[-1] - 1
            path.unlink()
            no_exports()
            _, recomputed = Engine(3.0).graph(design, groups, pruned=True)
            assert np.array_equal(recomputed, kept) and path.exists()
        finally:
            jax.config.update("jax_compilation_cache_dir", before)

    def test_a_graph_reloads_across_engines_untraced(self, backend, tmp_path, monkeypatch):
        # a second engine takes the serialized export: no context is built, no python traced,
        # the lines are the first engine's to the bit, the draws export serves any batch, and
        # a jax transformation falls back to the live graph
        if backend != "jax":
            pytest.skip("graphs persist as jax exports")
        import jax

        from mrsandbox import SpinSystem
        from mrsandbox.backend import Backend

        before = jax.config.jax_compilation_cache_dir
        jax.config.update("jax_compilation_cache_dir", str(tmp_path))
        try:
            groups = (tuple(SpinSystem.from_database("Lac")),)
            design = Press(te1_s=0.012, te2_s=0.02, tr_s=1.5)
            args = (ops.convert_to_tensor(np.asarray([1.4], np.float32)),
                    ops.convert_to_tensor(np.asarray([0.25], np.float32)),
                    ops.convert_to_tensor(np.float32(1.1)),
                    ops.convert_to_tensor(np.float32(3.0)))
            one = Engine(3.0)
            fn, kept = one.compiled_batch(design, groups, pruned=True)
            lines = [np.asarray(ops.convert_to_numpy(v)) for v in fn(*args)]
            rows, _ = one.replay_draws(design, groups)
            batch3 = np.asarray(ops.convert_to_numpy(
                rows(*(ops.stack([a] * 3) for a in args))[1]))
            two = Engine(3.0)
            monkeypatch.setattr(two, "_ensemble_context",
                                lambda *a, **k: pytest.fail("the export was not taken"))
            fn2, kept2 = two.compiled_batch(design, groups, pruned=True)
            again = [np.asarray(ops.convert_to_numpy(v)) for v in fn2(*args)]
            assert np.array_equal(kept, kept2)
            assert all(np.array_equal(a, b) for a, b in zip(lines, again))
            rows2, _ = two.replay_draws(design, groups)
            for draws in (3, 5):
                batch = np.asarray(ops.convert_to_numpy(
                    rows2(*(ops.stack([a] * draws) for a in args))[1]))
                assert np.array_equal(batch[:3], batch3[:3])
            monkeypatch.undo()
            mapped = Backend.vmap(fn2, in_axes=(0, 0, None, None))
            _, amps = mapped(ops.stack([args[0]] * 2), ops.stack([args[1]] * 2),
                             args[2], args[3])
            amps = np.asarray(ops.convert_to_numpy(amps))
            assert np.abs(amps[0] - lines[1]).max() <= 1e-6 * np.abs(lines[1]).max()
        finally:
            jax.config.update("jax_compilation_cache_dir", before)

    def test_rows_render_with_their_own_widths(self):
        # a batch of draws renders in one call, every row with its own Lorentzian widths, and equals
        # the rows rendered one by one
        engine = Engine(3.0)
        rng = np.random.default_rng(3)
        freqs = ops.convert_to_tensor(rng.uniform(-400.0, 400.0, 300).astype(np.float32))
        amps = (rng.normal(size=(5, 300)) + 1j * rng.normal(size=(5, 300))).astype(np.complex64)
        widths = rng.uniform(2.0, 12.0, (5, 300)).astype(np.float32)
        _, together = engine.render_fids(freqs, ops.convert_to_tensor(amps), npoints=1024,
                                         lw_hz=widths)
        together = np.asarray(ops.convert_to_numpy(together))
        for k in range(5):
            _, alone = engine.render_fid(freqs, ops.convert_to_tensor(amps[k]), npoints=1024,
                                         lw_hz=ops.convert_to_tensor(widths[k]))
            alone = np.asarray(ops.convert_to_numpy(alone))
            assert np.abs(together[k] - alone).max() < 1e-5 * np.abs(alone).max()

