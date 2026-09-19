####################################################################################################
#                                        test_gradients.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/09/26                                                                                #
#                                                                                                  #
# Purpose: Gradient gate: a single crusher dephases by exactly sinc(q k L), a balanced pair        #
#          around a perfect refocusing pulse reproduces the idealized coherence mask, an           #
#          imperfect pulse leaves the physical spurious FID a mask cannot show, and strong         #
#          crushers suppress that artifact the way scanners rely on.                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import (CoherenceFilter, Delay, Engine, Gradient, InducedKick, Pulse, Sequence,
                       SpinSystem)
from mrsandbox.phantom.water import WaterSource

TE = 0.03


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


def amps(engine, sequence, system, b1=1.0):
    _, a = engine.lines(sequence, system, b1_scale=b1, isochromats=64)
    return np.asarray(ops.convert_to_numpy(a))


def masked_echo():
    return Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(TE / 2),
                    Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(TE / 2))


def crushed_echo(moment):
    crusher = Gradient(moment, 20.0)
    return Sequence(Pulse(90.0), Delay(TE / 2), crusher,
                    Pulse(180.0, phase_deg=90.0), crusher, Delay(TE / 2))


#**************************************************************************************************#
#                                        Class TestGradient                                        #
#**************************************************************************************************#
class TestGradient:

    #****************#
    #   sinc law     #
    #****************#
    def test_single_crusher_follows_the_sinc_law(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        full = np.abs(amps(engine, Sequence(Pulse(90.0)), water)).sum()
        for kl in (0.3, 0.7, 1.5):
            crushed = np.abs(amps(engine, Sequence(Pulse(90.0), Gradient(kl, 1.0)), water)).sum()
            assert abs(crushed / full - abs(np.sinc(kl))) < 1e-3

    #**********************#
    #   ensemble default   #
    #**********************#
    def test_engine_default_isochromats_reach_the_ensemble(self):
        # the engine-level default must reach lines(): the single isochromat sits at
        # the voxel centre and sees no gradient, so it keeps the signal the true
        # ensemble dephases away (scans hand their isochromats argument to the engine)
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        sequence = Sequence(Pulse(90.0), Gradient(1.5, 1.0))
        full = np.abs(amps(Engine(3.0), Sequence(Pulse(90.0)), water)).sum()
        for isochromats, expected in ((1, 1.0), (None, abs(np.sinc(1.5)))):
            engine = Engine(3.0) if isochromats is None else Engine(3.0, isochromats=isochromats)
            _, a = engine.lines(sequence, water)
            ratio = np.abs(np.asarray(ops.convert_to_numpy(a))).sum() / full
            assert abs(ratio - expected) < 1e-2

    #**********************#
    #   pair equals mask   #
    #**********************#
    def test_balanced_pair_reproduces_the_coherence_mask(self, engine):
        for system in SpinSystem.from_database("Lac"):
            a = amps(engine, masked_echo(), system)
            b = amps(engine, crushed_echo(3.7), system)
            corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert corr > 0.99999
            assert abs(np.abs(b).sum() / np.abs(a).sum() - 1.0) < 1e-3

    #********************#
    #   spurious fid     #
    #********************#
    def test_imperfect_pulse_leaves_the_physical_spurious_fid(self, engine):
        # with a miscalibrated transmit both pulses err: longitudinal magnetization
        # survives the excitation and the imperfect 180 tips it into a spurious FID
        # that only the second crusher touches; a weak crusher shows it, the mask never
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        ideal = np.abs(amps(engine, masked_echo(), water, b1=0.6)).sum()
        weak = np.abs(amps(engine, crushed_echo(0.02), water, b1=0.6)).sum()
        strong = np.abs(amps(engine, crushed_echo(3.7), water, b1=0.6)).sum()
        assert weak > 1.1 * ideal  # the artifact exists and the mask cannot show it
        assert abs(strong / ideal - 1.0) < 0.01  # strong crushers suppress it, as on scanners

    #***********************#
    #   axis independence   #
    #***********************#
    def test_axes_dephase_independently(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        full = np.abs(amps(engine, Sequence(Pulse(90.0)), water)).sum()
        same = np.abs(amps(engine, Sequence(Pulse(90.0), Gradient(1.5, 1.0),
                                            Gradient(-1.5, 1.0)), water)).sum()
        cross = np.abs(amps(engine, Sequence(Pulse(90.0), Gradient(1.5, 1.0, axis="x"),
                                             Gradient(-1.5, 1.0, axis="y")), water)).sum()
        assert abs(same / full - 1.0) < 1e-3  # one axis: the pair still cancels exactly
        assert cross < 0.3 * full  # opposite axes never rephase each other

    def test_balanced_pairs_on_two_axes_refocus_exactly(self, engine):
        for system in SpinSystem.from_database("Lac"):
            a = amps(engine, masked_echo(), system)
            crushed = Sequence(Pulse(90.0), Delay(TE / 2),
                               Gradient(3.7, 20.0, axis="x"), Gradient(2.9, 20.0, axis="z"),
                               Pulse(180.0, phase_deg=90.0),
                               Gradient(3.7, 20.0, axis="x"), Gradient(2.9, 20.0, axis="z"),
                               Delay(TE / 2))
            b = amps(engine, crushed, system)
            corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert corr > 0.9999
            assert abs(np.abs(b).sum() / np.abs(a).sum() - 1.0) < 1e-3

    #********************#
    #   compiled path    #
    #********************#
    def test_compiled_path_carries_the_crushers(self, engine):
        # the isochromat ensemble compiles as a batch axis, so a crushed sequence
        # replays jitted with exactly the eager physics
        system = SpinSystem.from_database("Lac")[0]
        fn = engine.compiled(crushed_echo(3.7), system)
        args = [ops.convert_to_tensor(np.float32(v)) for v in (1.4, 0.28, 1.0, 0.0)]
        _, compiled = fn(*args)
        _, eager = engine.lines(crushed_echo(3.7), system.with_relaxation(t1_s=1.4, t2_s=0.28))
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(compiled)),
                                   np.asarray(ops.convert_to_numpy(eager)), atol=1e-5)


#**************************************************************************************************#
#                                     Class TestSliceSelection                                     #
#**************************************************************************************************#
#                                                                                                  #
# A shaped pulse under a gradient excites a slab; the refocus lobe rephases it; off-resonance      #
# spins are displaced (chemical shift displacement).                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestSliceSelection:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def fraction(engine, gradient_hz_mm, system, refocus=-0.5, duration=0.004):
        from mrsandbox import ShapedPulse

        pulse = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65,
                                     gradient_hz_per_mm=gradient_hz_mm, extent_mm=20.0)
        lobe = Gradient(refocus * gradient_hz_mm * duration, 20.0)
        sliced = np.abs(amps(engine, Sequence(pulse, lobe, CoherenceFilter(-1)), system)).sum()
        plain = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65)
        full = np.abs(amps(engine, Sequence(plain, CoherenceFilter(-1)), system)).sum()
        return sliced / full

    #*****************#
    #   slab + lobe   #
    #*****************#
    def test_slab_fraction_and_refocus_lobe(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        got = self.fraction(engine, 100.0, water)
        # gaussian R = 2.70: bandwidth 675 Hz over 100 Hz/mm = 6.75 mm of the 20 mm voxel
        assert abs(got - 2.70 / 0.004 / 100.0 / 20.0) < 0.05
        assert self.fraction(engine, 100.0, water, refocus=0.0) < 0.2 * got  # no lobe, dephased

    #***********************************#
    #   chemical shift displacement     #
    #***********************************#
    def test_chemical_shift_displaces_the_slab(self, engine):
        """The slab centre of an off-carrier spin sits at (carrier - nu) / g: the NAA
        acetyl singlet, 337 Hz from the water carrier, must land 11 mm away."""
        import dataclasses

        from mrsandbox import ShapedPulse

        g, duration = 30.0, 0.004
        pulse = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65,
                                     gradient_hz_per_mm=g, extent_mm=20.0)
        sequence = Sequence(pulse, Gradient(-0.5 * g * duration, 20.0), CoherenceFilter(-1))
        positions = np.linspace(-30.0, 30.0, 61)

        def slab_centre(system):
            ctx = engine.context(system)
            profile = []
            for x in positions:
                _, a = engine._system_lines(
                    sequence, dataclasses.replace(ctx, gradient_x=float(x)))
                profile.append(float(np.abs(ops.convert_to_numpy(a)).sum()))
            profile = np.asarray(profile)
            return float((positions * profile).sum() / profile.sum())

        water = WaterSource.system(gamma_mhz_t=42.5774785)
        (acetyl,) = [s for s in SpinSystem.from_database("NAA") if "acetyl" in s.name.lower()]
        displacement = slab_centre(acetyl) - slab_centre(water)
        expected = (4.65 - 2.008) * 127.73 / g
        assert abs(displacement - expected) < 1.0  # millimetres


#**************************************************************************************************#
#                                       Class TestPulseFrame                                       #
#**************************************************************************************************#
#                                                                                                  #
# A shaped pulse under a gradient: the batched isochromat ensemble equals a loop over positions.   #
#                                                                                                  #
#**************************************************************************************************#
class TestPulseFrame:

    def test_batched_ensemble_equals_a_loop_over_positions(self, engine):
        # the offset and slice term enter as one frame rotation per isochromat
        # (rank one in the row and column phases); the vector path must equal the
        # scalar path at every position, for a swept pulse with a carrier and an
        # offset frame
        import dataclasses

        from mrsandbox import ShapedPulse

        lac = SpinSystem.from_database("Lac")[0]
        pulse = ShapedPulse.hyperbolic_secant(0.005, 2000.0, 1500.0, offset_ppm=4.65,
                                              gradient_hz_per_mm=100.0, extent_mm=20.0, steps=64)
        ctx = engine._ensemble_context(Sequence(pulse), engine.context(lac, offset_hz=25.0), 8)
        batched = np.asarray(ops.convert_to_numpy(pulse.apply(ctx.rho_eq, ctx)))
        positions = np.asarray(ops.convert_to_numpy(ctx.gradient_x))
        for k, x in enumerate(positions):
            single = dataclasses.replace(ctx, gradient_x=float(x))
            one = np.asarray(ops.convert_to_numpy(pulse.apply(ctx.rho_eq, single)))
            np.testing.assert_allclose(batched[k], one, atol=1e-6)


#**************************************************************************************************#
#                                      Class TestInducedKick                                       #
#**************************************************************************************************#
class TestInducedKick:

    def test_moments_kick_like_gradients_and_cycles_like_faster_precession(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        # a kick's moment over the ensemble is a gradient kick (the ensemble comes from
        # the design's own Gradient, so the kick rides on its axis)
        gradient = Sequence(Pulse(90.0), Gradient(0.02, 20.0, axis="z"), Gradient(0.05, 20.0, axis="z"))
        kicked = Sequence(Pulse(90.0), Gradient(0.02, 20.0, axis="z"), InducedKick(0.0, moment_z=0.05))
        np.testing.assert_allclose(amps(engine, kicked, water), amps(engine, gradient, water),
                                   atol=1e-6)
        # its uniform cycles advance the spins as a field does: a scanner frequency
        # error of minus that field over the delay, offset times duration in cycles,
        # and the sign of free precession itself
        offset = Sequence(Pulse(90.0), Delay(0.010))
        _, reference = engine.lines(offset, water, offset_hz=-30.0)
        _, plain = engine.lines(offset, water)
        kicked = Sequence(Pulse(90.0), Delay(0.010), InducedKick(0.3))
        _, kick = engine.lines(kicked, water)
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(kick)),
                                   np.asarray(ops.convert_to_numpy(reference)), atol=1e-6)
        assert np.abs(np.asarray(ops.convert_to_numpy(kick)) - np.asarray(ops.convert_to_numpy(plain))).max() > 1e-2
        brief = Sequence(Pulse(90.0), CoherenceFilter(-1), InducedKick(0.05))
        line = np.asarray(ops.convert_to_numpy(engine.lines(brief, water)[1]))
        assert abs(np.degrees(np.angle(line[np.argmax(np.abs(line))])) - 18.0) < 1e-3
        # every value is a timing, the axes are structure
        assert InducedKick(0.1, moment_x=0.2).structure() == InducedKick(0.3, moment_x=0.4).structure()
        assert InducedKick(0.1, moment_x=0.2).timings() == (0.1, 0.2, 0.0, 0.0)

    def test_a_field_during_a_shaped_pulse_runs_the_spins_off_resonance(self, engine):
        # the `induced` track of a shaped pulse: a constant field is the scanner's
        # frequency error of minus that field, and a constant first-order term across
        # the ensemble is the pulse's own slice gradient
        from mrsandbox import SpinSystem
        from mrsandbox.sequences.events import ShapedPulse

        lac = SpinSystem.from_database("Lac")[0]
        steps, duration = 64, 0.004
        t = np.append((np.arange(steps) + 0.5) * duration / steps, duration)

        def lines(sequence, **kw):
            return np.asarray(ops.convert_to_numpy(engine.lines(sequence, lac, **kw)[1]))

        track = np.zeros((4, steps + 1), dtype=np.float32)
        track[0] = 40.0 * t
        plain = Sequence(ShapedPulse.gaussian(90.0, duration, offset_ppm=1.3))
        field = Sequence(ShapedPulse.gaussian(90.0, duration, offset_ppm=1.3, induced=track))
        np.testing.assert_allclose(lines(field), lines(plain, offset_hz=-40.0), atol=1e-6)
        assert np.abs(lines(field) - lines(plain)).max() > 0.05
        track = np.zeros((4, steps + 1), dtype=np.float32)
        track[3] = 25.0 * t
        rephase = Gradient(-0.5 * 25.0 * duration, 20.0, axis="z")
        slab = Sequence(ShapedPulse.gaussian(90.0, duration, offset_ppm=1.3, gradient_hz_per_mm=25.0,
                                             gradient_axis="z"), rephase)
        field = Sequence(ShapedPulse.gaussian(90.0, duration, offset_ppm=1.3, induced=track), rephase)
        np.testing.assert_allclose(lines(field), lines(slab), atol=1e-6)
        # the track is a timing: one compiled design replays at every voxel
        assert field.events[0].structure() == Sequence(ShapedPulse.gaussian(
            90.0, duration, offset_ppm=1.3, induced=2.0 * track)).events[0].structure()



#**************************************************************************************************#
#                                       Class TestProjection                                       #
#**************************************************************************************************#
#                                                                                                  #
# Averaging the density matrix axis by axis is the full voxel sum (Zhang, An, Shen 2017).          #
#                                                                                                  #
#**************************************************************************************************#
class TestProjection:

    def test_two_axes_projected_in_turn_are_the_full_two_dimensional_sum(self):
        # a slice-selective 90 on x and a refocusing 180 on y act on lactate: the
        # design that averages over x after the 90 and over y after the 180 (8 + 8
        # positions) equals the mean over all 64 (x, y) pairs, each played with the
        # field of its position typed into the pulses (their `induced` tracks)
        from mrsandbox import Projection, ShapedPulse, SpinSystem

        lac = SpinSystem.from_database("Lac")[0]
        n, extent, g = 8, 40.0, 110.0
        engine = Engine(3.0, isochromats=n)
        kw = dict(offset_ppm=1.3, steps=64)
        short = 0.002

        def lines(sequence):
            return np.asarray(ops.convert_to_numpy(engine.lines(sequence, lac)[1]))

        projected = Sequence(
            ShapedPulse.sinc(90.0, short, gradient_hz_per_mm=g, gradient_axis="x", extent_mm=extent, **kw),
            Gradient(-0.5 * g * short, extent, axis="x"), CoherenceFilter(+1), Projection(),
            Delay(0.004),
            ShapedPulse.sinc(180.0, short, gradient_hz_per_mm=g, gradient_axis="y", extent_mm=extent, **kw),
            CoherenceFilter(-1), Projection(), Delay(0.004))
        t = np.append((np.arange(64) + 0.5) * short / 64, short)

        def track(position_mm):
            field = np.zeros((4, 65), dtype=np.float32)
            field[0] = g * position_mm * t
            return field

        grid = ((np.arange(n) + 0.5) / n - 0.5) * extent
        total = 0.0
        for x in grid:
            for y in grid:
                total = total + lines(Sequence(
                    ShapedPulse.sinc(90.0, short, induced=track(x), **kw),
                    InducedKick(-0.5 * g * x * short), CoherenceFilter(+1), Delay(0.004),
                    ShapedPulse.sinc(180.0, short, induced=track(y), **kw),
                    CoherenceFilter(-1), Delay(0.004)))
        full = total / n ** 2
        assert np.abs(full).max() > 0.01
        np.testing.assert_allclose(lines(projected), full, atol=2e-5)
        # the gain is a timing and scales the average
        assert Projection(1.0).structure() == Projection(2.5).structure()


#**************************************************************************************************#
#                                   Class TestLocalizedSemiLaser                                   #
#**************************************************************************************************#
class TestLocalizedSemiLaser:

    def test_the_voxel_keeps_its_profiles_and_its_phase(self):
        from mrsandbox import LocalizedSemiLaser

        f0_mhz = 3.0 * 42.5774785
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        engine = Engine(3.0, isochromats=64)

        def line(sequence):
            a = np.asarray(ops.convert_to_numpy(engine.lines(sequence, water)[1]))
            return a[np.argmax(np.abs(a))]

        plain = line(Sequence(Pulse(90.0), CoherenceFilter(-1)))
        voxel = line(LocalizedSemiLaser(0.030, f0_mhz, reference_ppm=4.65))
        # the signal per nominal voxel: the slab integrals of the excitation and two
        # GOIA pairs (0.98 x 0.93 x 0.93), in phase with a plain excitation
        assert 0.75 < abs(voxel / plain) < 0.88
        assert abs(np.degrees(np.angle(voxel / plain))) < 0.5
        # a transmit field 20 percent low costs the excitation its sine and the
        # adiabatic pairs a little
        engine_low = np.asarray(ops.convert_to_numpy(engine.lines(
            LocalizedSemiLaser(0.030, f0_mhz, reference_ppm=4.65), water, b1_scale=0.8)[1]))
        assert 0.75 < np.abs(engine_low).max() / abs(voxel) < 0.97
        with pytest.raises(ValueError):
            LocalizedSemiLaser(0.020, f0_mhz)

    def test_a_pulse_object_played_twice_is_computed_once(self):
        # the pair on an axis is one ShapedPulse object: a run asks for its propagator
        # once, a compiled graph keeps the sharing, and the result is the one of four
        # separate pulses
        import copy

        from mrsandbox import LocalizedSemiLaser, ShapedPulse, SpinSystem

        design = LocalizedSemiLaser(0.030, 3.0 * 42.5774785)
        shaped = [e for e in design.events if isinstance(e, ShapedPulse)]
        assert len(shaped) == 5 and shaped[1] is shaped[2] and shaped[3] is shaped[4]
        bound = design.bound(design.timings())
        again = [e for e in bound.events if isinstance(e, ShapedPulse)]
        assert again[1] is again[2] and again[3] is again[4] and again[1] is not again[3]
        calls, original = [], ShapedPulse.propagator

        def counted(self, ctx):
            calls.append(id(self))
            return original(self, ctx)

        lac, engine = SpinSystem.from_database("Lac")[0], Engine(3.0, isochromats=16)
        ShapedPulse.propagator = counted
        try:
            shared = np.asarray(ops.convert_to_numpy(engine.lines(design, lac)[1]))
        finally:
            ShapedPulse.propagator = original
        assert len(calls) == 3
        separate = Sequence(*[copy.copy(e) for e in design.events])
        np.testing.assert_allclose(shared, np.asarray(ops.convert_to_numpy(
            engine.lines(separate, lac)[1])), atol=1e-6)


#**************************************************************************************************#
#                                     Class TestLocalizedPress                                     #
#**************************************************************************************************#
#                                                                                                  #
# The narrow refocusing slabs of PRESS displace coupled partners: lactate pays at TE 144.          #
#                                                                                                  #
#**************************************************************************************************#
class TestLocalizedPress:

    def test_lactate_loses_its_inverted_doublet_to_the_displacement(self):
        from mrsandbox import LocalizedPress, Press, SpinSystem

        lac = SpinSystem.from_database("Lac")[0]
        water = WaterSource.system(gamma_mhz_t=42.5774785)

        def doublet(b0_t, te1_s, te2_s):
            engine, f0_mhz = Engine(b0_t, isochromats=32), b0_t * 42.5774785
            kept = []
            for design in (LocalizedPress(te1_s, te2_s, f0_mhz), Press(te1_s=te1_s, te2_s=te2_s)):
                freqs, a = (np.asarray(ops.convert_to_numpy(x)) for x in engine.lines(design, lac))
                kept.append(np.real(a[freqs < 2.5 * f0_mhz].sum()))
            return kept[0] / kept[1]

        # at TE 144 the methine and the methyl are refocused together in part of the
        # voxel only, more so as the shift difference grows with the field; at TE 30
        # and at TE 288 the doublet is upright either way and keeps its signal
        assert 0.45 < doublet(1.5, 0.030, 0.114) < 0.8
        assert doublet(3.0, 0.030, 0.114) < 0.4
        assert doublet(3.0, 0.015, 0.015) > 0.9
        assert doublet(3.0, 0.030, 0.258) > 0.7
        # the four compartments in closed form: with f the part of the voxel where a
        # refocusing slab misses the methine (its 2.79 ppm from the methyl over the
        # pulse's bandwidth), both hit on (1 - f)^2 of it (the doublet inverted), one
        # misses on 2 f (1 - f) (J evolves over TE1 or TE2 alone: cos(pi J 30 ms) and
        # cos(pi J 114 ms) cancel), both miss on f^2 (upright): -(1 - 2 f) in all
        bandwidth_hz = LocalizedPress.REFOCUSING_TBW / 0.0065
        for b0_t in (1.5, 3.0):
            f = (4.0974 - 1.3142) * b0_t * 42.5774785 / bandwidth_hz
            assert abs(doublet(b0_t, 0.030, 0.114) - (1.0 - 2.0 * f)) < 0.03
        # an uncoupled line keeps the three slab integrals (the Shinnar-Le Roux slabs
        # hold their nominal width), in phase with the ideal echo
        engine = Engine(3.0, isochromats=32)
        a = np.asarray(ops.convert_to_numpy(engine.lines(
            LocalizedPress(0.015, 0.015, 3.0 * 42.5774785, reference_ppm=4.65), water)[1]))
        b = np.asarray(ops.convert_to_numpy(engine.lines(Press(te1_s=0.015, te2_s=0.015), water)[1]))
        k = np.argmax(np.abs(b))
        assert 0.95 < abs(a[k] / b[k]) < 1.05 and abs(np.degrees(np.angle(a[k] / b[k]))) < 0.5
        with pytest.raises(ValueError):
            LocalizedPress(0.004, 0.010, 3.0 * 42.5774785)


#**************************************************************************************************#
#                                     Class TestLocalizedSteam                                     #
#**************************************************************************************************#
class TestLocalizedSteam:

    def test_three_slabs_keep_the_stimulated_echo_upright(self):
        from mrsandbox import LocalizedSteam, Scanner, SpinSystem, Steam

        f0_mhz = 3.0 * 42.5774785
        engine = Engine(3.0, isochromats=32)

        def lines(sequence, system):
            return np.asarray(ops.convert_to_numpy(engine.lines(sequence, system)[1]))

        water = WaterSource.system(gamma_mhz_t=42.5774785)
        ideal = lines(Steam(te_s=0.020, tm_s=0.010), water)
        k = np.argmax(np.abs(ideal))
        voxel = lines(LocalizedSteam(0.020, 0.010, f0_mhz, reference_ppm=4.65), water)
        # three Shinnar-Le Roux slab integrals of 0.98, in phase with the ideal echo
        assert 0.9 < abs(voxel[k] / ideal[k]) < 1.0
        assert abs(np.degrees(np.angle(voxel[k] / ideal[k]))) < 0.5
        # 2.3 kHz on every axis: the coupled multiplets stay close to the ideal ones
        glu = SpinSystem.from_database("Glu")[0]
        a, b = lines(LocalizedSteam(0.020, 0.010, f0_mhz), glu), lines(Steam(te_s=0.020, tm_s=0.010), glu)
        assert abs(np.vdot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b) > 0.98
        # 14 uT pulses play on every preset
        assert LocalizedSteam(0.020, 0.010, f0_mhz).feasible(Scanner.preset("philips_elition_3t"))
        with pytest.raises(ValueError):
            LocalizedSteam(0.004, 0.010, f0_mhz)


#**************************************************************************************************#
#                                     Class TestDifferentiable                                     #
#**************************************************************************************************#
class TestDifferentiable:

    def test_a_localized_design_is_differentiable_in_its_echo_time(self, needs_autograd):
        # the projection, the shaped slabs, and the shift input leave the graph
        # differentiable in a timing: the gradient of the lactate signal with respect
        # to the first delay against a central difference
        from nifti_mrs_plus.backend import value_and_grad

        from mrsandbox import LocalizedPress, SpinSystem

        f0_mhz = 3.0 * 42.5774785
        engine = Engine(3.0, isochromats=8)
        lac = SpinSystem.from_database("Lac")[0]
        design = LocalizedPress(0.030, 0.040, f0_mhz)
        fn, _ = engine.graph(design, ((lac,),))
        args = [ops.convert_to_tensor(np.asarray(v, dtype=np.float32)) for v in ([1.2], [0.2], 1.0, 0.0)]
        timings = Engine.tensors(design.timings())
        delays = [k for k, event in enumerate(design.events) if isinstance(event, Delay)]

        def observable(first_delay):
            moved = list(timings)
            moved[delays[0]] = (first_delay,)
            _, amps = fn(*args, tuple(moved), ops.zeros((3,)))
            return ops.sum(ops.real(amps) ** 2)

        at = timings[delays[0]][0]
        value, grad = value_and_grad(observable)(at)
        h = 2e-5  # the signal curves within 0.1 ms: the difference converges to the gradient from 4e-4 down
        finite = (float(observable(at + h)) - float(observable(at - h))) / (2.0 * h)
        assert abs(finite) > 0.0 and abs(float(grad) - finite) < 0.02 * abs(finite)


#*************************#
#   the library's press   #
#*************************#
def test_the_librarys_press_plays_its_crushers_on_request():
    from mrsandbox import Engine, Press, Scanner, SpinSystem

    engine = Engine(Scanner.preset("generic_3t"), isochromats=64)
    lactate = SpinSystem(name="pair", nucleus="1H", shifts_ppm=(1.31, 4.10), scale=1.0,
                         j_hz=((0.0, 6.9), (6.9, 0.0)), gamma_mhz_t=42.5774785)

    def amps(sequence, b1_scale):
        return np.asarray(ops.convert_to_numpy(engine.lines(sequence, lactate, b1_scale=b1_scale)[1]))

    filtered, strong, weak = (Press(0.011, 0.019), Press(0.011, 0.019, crushers=(0.15, 0.25)),
                              Press(0.011, 0.019, crushers=(0.004, 0.007)))
    # perfect pulses: whole cycles over the ensemble are the filters, to the float
    np.testing.assert_allclose(amps(strong, 1.0), amps(filtered, 1.0), atol=2e-5)
    # a transmit field a fifth low: strong crushers still select the echo as the filters do,
    # weak ones let the unrefocused coherences through
    wanted = amps(filtered, 0.8)
    assert np.abs(amps(strong, 0.8) - wanted).max() < 0.02 * np.abs(wanted).max()
    assert np.abs(amps(weak, 0.8) - wanted).max() > 0.05 * np.abs(wanted).max()


def test_steam_and_semi_laser_play_their_crushers_too():
    from mrsandbox import Engine, Scanner, SemiLaser, SpinSystem, Steam

    engine = Engine(Scanner.preset("generic_3t"), isochromats=64)
    lactate = SpinSystem(name="pair", nucleus="1H", shifts_ppm=(1.31, 4.10), scale=1.0,
                         j_hz=((0.0, 6.9), (6.9, 0.0)), gamma_mhz_t=42.5774785)

    def amps(sequence, b1_scale=1.0):
        return np.asarray(ops.convert_to_numpy(engine.lines(sequence, lactate, b1_scale=b1_scale)[1]))

    # whole cycles over the ensemble select what the filters select
    for filtered, played in ((Steam(0.02, 0.01), Steam(0.02, 0.01, crushers=(0.15, 0.35))),
                             (SemiLaser(0.03), SemiLaser(0.03, crushers=(0.15, 0.25, 0.35, 0.55)))):
        wanted = amps(filtered)
        assert np.abs(amps(played) - wanted).max() < 0.02 * np.abs(wanted).max()
    # a stimulated echo with a transmit field a fifth low: a weak scheme lets the other echoes in
    wanted = amps(Steam(0.02, 0.01), 0.8)
    assert np.abs(amps(Steam(0.02, 0.01, crushers=(0.15, 0.35)), 0.8) - wanted).max() < 0.03 * np.abs(wanted).max()
    assert np.abs(amps(Steam(0.02, 0.01, crushers=(0.004, 0.006)), 0.8) - wanted).max() > 0.05 * np.abs(wanted).max()
