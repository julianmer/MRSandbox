####################################################################################################
#                                          test_cycling.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: Phase cycling as physics: an EXORCYCLE on the two refocusing pulses of a PRESS          #
#          reproduces the coherence-filter idealization exactly, a per-step phase error leaks what #
#          the cycle cancels, and the cycle replays through the compiled graph with the jitter as  #
#          a timing.                                                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import Cycled, Delay, Engine, Gradient, Press, Pulse, Sequence, SpinSystem


@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


def bare_press(te1_s=0.011, te2_s=0.019):
    """PRESS without its coherence filters: the physical design a cycle selects on."""
    return Sequence(Pulse(90.0), Delay(te1_s / 2.0), Pulse(180.0, phase_deg=90.0),
                    Delay((te1_s + te2_s) / 2.0), Pulse(180.0, phase_deg=90.0), Delay(te2_s / 2.0))


def exorcycle():
    """The sixteen-step EXORCYCLE on both refocusing pulses: each steps through 0, 90,
    180, 270 with the receiver following twice the phase (Bodenhausen, Freeman, and
    Turner 1977), selecting a change of two in the coherence order at each pulse."""
    table, receiver = [], []
    for a in range(4):
        for b in range(4):
            table.append([0.0, 90.0 * a, 90.0 * b])
            receiver.append((180.0 * a + 180.0 * b) % 360.0)
    return np.asarray(table), np.asarray(receiver)


def two_spins():
    """A coupled pair, the smallest system with a doublet."""
    return SpinSystem(name="pair", nucleus="1H", shifts_ppm=(1.31, 4.10), scale=1.0,
                      j_hz=((0.0, 6.9), (6.9, 0.0)), gamma_mhz_t=42.5774785)


def crushed_press(te1_s=0.011, te2_s=0.019):
    """PRESS with crusher pairs around its refocusing pulses: moments of 3 and 5
    cycles over the 20 mm ensemble, so every unbalanced pathway dephases to exactly
    zero on the uniform isochromat grid."""
    first, second = Gradient(0.15, 20.0), Gradient(0.25, 20.0)
    return Sequence(Pulse(90.0), Delay(te1_s / 2.0), first, Pulse(180.0, phase_deg=90.0), first,
                    Delay((te1_s + te2_s) / 2.0), second, Pulse(180.0, phase_deg=90.0), second,
                    Delay(te2_s / 2.0))


def amplitudes(engine, sequence, system, b1_scale=1.0):
    _, amps = engine.lines(sequence, system, b1_scale=b1_scale)
    return np.asarray(ops.convert_to_numpy(amps))


#**************************************************************************************************#
#                                        Class TestCycling                                         #
#**************************************************************************************************#
class TestCycling:

    def test_exorcycle_reproduces_the_coherence_filters(self, engine):
        # a two-spin system: the orders reach two, so a change of two at each 180
        # from the excited single quantum is exactly the filters' -1, +1, -1 path
        table, receiver = exorcycle()
        for b1 in (1.0, 0.8):
            cycled = amplitudes(engine, Cycled(bare_press(), table, receiver), two_spins(), b1)
            filtered = amplitudes(engine, Press(te1_s=0.011, te2_s=0.019), two_spins(), b1)
            np.testing.assert_allclose(cycled, filtered, atol=1e-6 * np.abs(filtered).max())
        # without the cycle the imperfect pulses' unrefocused pathways survive
        bare = amplitudes(engine, bare_press(), two_spins(), 0.8)
        assert np.abs(bare - filtered).max() > 1e-2 * np.abs(filtered).max()

    def test_relaxing_cycle_matches_crushers_where_the_filters_leak(self, engine):
        # with T1 and imperfect flips the filters keep what neither a cycle nor a
        # crusher pair does: the magnetization recovered before the first 180,
        # tipped to +1 there and refocused to -1 at the second, an O(TE / T1) term
        table, receiver = exorcycle()
        pair = two_spins().with_relaxation(t1_s=1.3, t2_s=0.25)
        cycled = amplitudes(engine, Cycled(bare_press(), table, receiver), pair, 0.8)
        crushed = amplitudes(engine, crushed_press(), pair, 0.8)
        filtered = amplitudes(engine, Press(te1_s=0.011, te2_s=0.019), pair, 0.8)
        np.testing.assert_allclose(cycled, crushed, atol=1e-6 * np.abs(crushed).max())
        gap = np.abs(cycled - filtered).max() / np.abs(filtered).max()
        assert 1e-3 < gap < 5e-2, gap
        # with four spins the cycle keeps a path crushers dephase (-1, -3, -1 through
        # a triple quantum), which at these flips is below 1e-4 of the signal
        lac = SpinSystem.from_database("Lac")[0].with_relaxation(t1_s=1.3, t2_s=0.25)
        cycled = amplitudes(engine, Cycled(bare_press(), table, receiver), lac, 0.8)
        crushed = amplitudes(engine, crushed_press(), lac, 0.8)
        np.testing.assert_allclose(cycled, crushed, atol=1e-4 * np.abs(crushed).max())

    def test_a_per_step_phase_error_leaks_the_cancelled_pathways(self, engine):
        lac = SpinSystem.from_database("Lac")[0].with_relaxation(t1_s=1.3, t2_s=0.25)
        table, receiver = exorcycle()
        rng = np.random.default_rng(3)
        ideal = amplitudes(engine, Cycled(bare_press(), table, receiver), lac, 0.8)
        leaks = []
        for scale in (2.0, 8.0):
            jitter = rng.normal(0.0, scale, table.shape).astype(np.float32)
            jittered = amplitudes(engine, Cycled(bare_press(), table, receiver, jitter), lac, 0.8)
            leaks.append(np.abs(jittered - ideal).max() / np.abs(ideal).max())
        assert 0.0 < leaks[0] < leaks[1]

    def test_the_cycle_replays_compiled_with_the_jitter_as_a_timing(self, engine):
        lac = SpinSystem.from_database("Lac")[0]
        table, receiver = exorcycle()
        jitter = np.random.default_rng(5).normal(0.0, 4.0, table.shape).astype(np.float32)
        design = Cycled(bare_press(), table, receiver, jitter)
        assert design.timings()[-1][0] is jitter
        assert design.structure() == Cycled(bare_press(), table, receiver, 2.0 * jitter).structure()
        assert design.structure() != Cycled(bare_press(), table, receiver).structure()
        args = [ops.convert_to_tensor(np.float32(v)) for v in (1.3, 0.25, 0.8, 0.0)]
        _, compiled = engine.compiled(design, lac)(*args)
        eager = amplitudes(engine, design, lac.with_relaxation(t1_s=1.3, t2_s=0.25), 0.8)
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(compiled)), eager,
                                   atol=1e-5 * np.abs(eager).max())
        # the design's other faces survive the wrapping
        assert design.duration_s == pytest.approx(0.030)
        assert (Cycled(Press(te1_s=0.011, te2_s=0.019), table, receiver).feasible.__self__
                is not None)
        with pytest.raises(ValueError):
            Cycled(bare_press(), table[:, :2], receiver)
