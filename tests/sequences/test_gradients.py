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
        # the signal per nominal voxel: the slab integrals of a three-lobe sinc and two
        # GOIA pairs (0.93 x 0.93 x 0.93), in phase with a plain excitation
        assert 0.75 < abs(voxel / plain) < 0.88
        assert abs(np.degrees(np.angle(voxel / plain))) < 0.5
        # a transmit field 20 percent low costs the excitation its sine and the
        # adiabatic pairs a little
        engine_low = np.asarray(ops.convert_to_numpy(engine.lines(
            LocalizedSemiLaser(0.030, f0_mhz, reference_ppm=4.65), water, b1_scale=0.8)[1]))
        assert 0.75 < np.abs(engine_low).max() / abs(voxel) < 0.97
        with pytest.raises(ValueError):
            LocalizedSemiLaser(0.020, f0_mhz)
