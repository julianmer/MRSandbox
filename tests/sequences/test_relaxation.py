####################################################################################################
#                                        test_relaxation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Relaxation gate: in-sequence relaxation follows the analytic laws. Echo attenuation is  #
#          exactly exp(-TE/T2), a shaped pulse relaxes over its duration, suppression-train         #
#          recovery is exactly 1 - exp(-delay/T1), systems without relaxation stay purely          #
#          coherent, and the scan path carries the TE dependence.                                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, CoherenceFilter, Delay, Engine, Params, Press, Pulse, PulseAcquire,
                       Scanner, Sequence, ShapedPulse, SpinSystem, SuppressionTrain, VirtualScan)
from mrsandbox.phantom.water import WaterSource


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


def total_amplitude(engine, sequence, system):
    _, amps = engine.lines(sequence, system)
    return float(np.abs(ops.convert_to_numpy(amps)).sum())


#**************************************************************************************************#
#                                  Class TestInSequenceRelaxation                                  #
#**************************************************************************************************#
class TestInSequenceRelaxation:

    #**********************#
    #   echo attenuation   #
    #**********************#
    def test_echo_attenuation_is_exp_minus_te_over_t2(self, engine):
        (scyllo,) = SpinSystem.from_database("sIns")
        relaxed = scyllo.with_relaxation(t1_s=1.3, t2_s=0.1)
        reference = total_amplitude(engine, PulseAcquire(), relaxed)
        for te in (0.03, 0.06, 0.144):
            ratio = total_amplitude(engine, Press(te / 2, te / 2), relaxed) / reference
            assert abs(ratio - math.exp(-te / 0.1)) < 1e-3

    #*******************#
    #   shaped pulses   #
    #*******************#
    def test_a_shaped_pulse_relaxes_over_its_duration(self, engine):
        # an echo whose 180 is a 15 ms Gaussian decays through the pulse too, not over its
        # delays alone: by exp(-TE/T2) between the pulse centres were the magnetization
        # transverse throughout, a little less since the turn passes through the
        # longitudinal axis, where it recovers with T1 instead (the Bloch picture, which
        # the per-step reference reproduces to a part in a thousand)
        (scyllo,) = SpinSystem.from_database("sIns")
        relaxed = scyllo.with_relaxation(t1_s=1.3, t2_s=0.1)
        reference = total_amplitude(engine, PulseAcquire(), relaxed)
        length, te = 0.015, 0.068
        gaussian = ShapedPulse.gaussian(180.0, length, offset_ppm=3.34, truncation=0.0571)
        echo = Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(te / 2.0 - length / 2.0), gaussian,
                        CoherenceFilter(-1), Delay(te / 2.0 - length / 2.0))
        ratio = total_amplitude(engine, echo, relaxed) / reference
        assert math.exp(-te / 0.1) < ratio < math.exp(-(te - length / 2.0) / 0.1)
        assert abs(ratio - math.exp(-te / 0.1)) < 0.03 * math.exp(-te / 0.1)

    #*****************#
    #   t1 recovery   #
    #*****************#
    def test_suppression_recovery_is_one_minus_exp(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=1.0, t2_s=0.05)
        full = total_amplitude(engine, PulseAcquire(), water)
        for delay in (0.1, 0.3, 1.0):
            train = SuppressionTrain([90.0], delays_s=[delay]) + PulseAcquire()
            recovered = total_amplitude(engine, train, water) / full
            assert abs(recovered - (1.0 - math.exp(-delay / 1.0))) < 1e-3

    #**********************#
    #   coherent default   #
    #**********************#
    def test_without_relaxation_delays_stay_coherent(self, engine):
        (scyllo,) = SpinSystem.from_database("sIns")
        reference = total_amplitude(engine, PulseAcquire(), scyllo)
        echoed = total_amplitude(engine, Press(0.072, 0.072), scyllo)
        assert abs(echoed - reference) < 1e-4 * reference

    #********************#
    #   t1-only delays   #
    #********************#
    def test_t1_only_recovers_and_keeps_the_coherences(self, engine):
        # a missing partner means an infinite time, never no relaxation at all: a
        # T1-only system still recovers longitudinally while its echo does not decay
        water = WaterSource.system(gamma_mhz_t=42.5774785).with_relaxation(t1_s=1.0)
        full = total_amplitude(engine, PulseAcquire(), water)
        train = SuppressionTrain([90.0], delays_s=[0.3]) + PulseAcquire()
        recovered = total_amplitude(engine, train, water) / full
        assert abs(recovered - (1.0 - math.exp(-0.3 / 1.0))) < 1e-3
        echoed = total_amplitude(engine, Press(0.072, 0.072), water) / full
        assert abs(echoed - 1.0) < 1e-3

    #*********************#
    #   scan carries it   #
    #*********************#
    @pytest.mark.render  # a full-scan render, skipped on the tensorflow CI leg
    def test_scan_amplitudes_carry_the_te_dependence(self):
        brain = Brain.load("icbm152")
        # quiet receiver: the real-part peak reads an absorption line only without the
        # drawn shift, phase, eddy, and spurious content (each gated by its own test)
        from mrsandbox import NoDownfield, NoMM, NoWater

        # NAA alone: every neighbour (NAAG at 2.04, the glutamate multiplets, the
        # macromolecule at 2.04, the water tail) decays on its own T2 and would drift
        # the 2.008 ppm bin, so the gate on the NAA law reads NAA by itself
        alone = {name: 0.0 for name in VirtualScan.default_params().conc if name != "NAA"}
        quiet = Params(acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
                       spurious=dict(amplitude=0.0), conc=alone)
        heights, truth = [], None
        for te in (0.03, 0.06):
            scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te / 2, te / 2),
                               params=quiet, water=NoWater(), mm=NoMM(), downfield=NoDownfield())
            voxel = scan.voxel(position_mm=(10, -20, 5), size_mm=20)
            truth = voxel.sample(seed=7)  # same seed, same draw for both scans
            ppm, spec = voxel.spectrum(seed=7, npoints=8192)
            ppm, spec = np.asarray(ops.convert_to_numpy(ppm)), ops.convert_to_numpy(spec)
            heights.append(float(spec.real[np.argmin(np.abs(ppm - 2.008))]))
        expected = math.exp(-0.03 / truth.t2.NAA)  # the drawn Gudmundson-pooled NAA T2
        assert abs(heights[1] / heights[0] - expected) < 1e-2


#**************************************************************************************************#
#                                      Class TestRepetition                                        #
#**************************************************************************************************#
#                                                                                                  #
# A sequence with a TR reaches the spoiled steady state: the T1 saturation of the excitation and   #
# the inversion of the recovered magnetization by refocusing pulses follow the analytic laws.      #
#                                                                                                  #
#**************************************************************************************************#
class TestRepetition:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def system(t1_s=1.3):
        return SpinSystem.from_database("sIns")[0].with_relaxation(t1_s=t1_s, t2_s=100.0)

    #**********************#
    #   pulse acquire      #
    #**********************#
    def test_pulse_acquire_saturates_by_one_minus_exp(self, engine):
        from mrsandbox import Sequence

        relaxed = total_amplitude(engine, PulseAcquire(), self.system())
        for tr in (0.5, 1.0, 2.0, 5.0):
            ratio = total_amplitude(engine, PulseAcquire(tr_s=tr), self.system()) / relaxed
            assert abs(ratio - (1.0 - math.exp(-tr / 1.3))) < 1e-5
        with pytest.raises(ValueError):
            Sequence(Delay(0.5), tr_s=0.2)

    #*****************#
    #   spin echo     #
    #*****************#
    def test_spin_echo_inverts_the_recovered_magnetization(self, engine):
        # 90 - tau - 180 - tau: what recovers during the first tau is inverted by the
        # 180, so the steady state is 1 - 2 exp(-(TR - tau) / T1) + exp(-TR / T1) with
        # physical crushers; a coherence filter on the detected order drops the
        # recovered longitudinal part, the stated O(tau / T1) idealization, and the
        # steady state becomes 1 - exp(-(TR - tau) / T1)
        from mrsandbox import CoherenceFilter, Gradient, Pulse, Sequence

        tau, t1 = 0.030, 1.3

        def crushed(tr):
            return Sequence(Pulse(90.0), Gradient(30.0), Delay(tau), Pulse(180.0, phase_deg=90.0),
                            Gradient(30.0), Delay(tau), tr_s=tr)

        def filtered(tr):
            return Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(tau),
                            Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(tau),
                            tr_s=tr)

        for tr in (0.5, 2.0):
            exact = 1.0 - 2.0 * math.exp(-(tr - tau) / t1) + math.exp(-tr / t1)
            ratio = (total_amplitude(engine, crushed(tr), self.system(t1))
                     / total_amplitude(engine, crushed(None), self.system(t1)))
            assert abs(ratio - exact) < 1e-3
            ratio = (total_amplitude(engine, filtered(tr), self.system(t1))
                     / total_amplitude(engine, filtered(None), self.system(t1)))
            assert abs(ratio - (1.0 - math.exp(-(tr - tau) / t1))) < 1e-5

    #*****************#
    #   convergence   #
    #*****************#
    def test_cycles_converge_for_an_imperfect_excitation(self, engine):
        # with b1 off nominal some longitudinal magnetization survives the 90 and the
        # first cycle is off by its cos(flip) exp(-TR / T1) fraction; more cycles close it
        from mrsandbox import CoherenceFilter, Pulse, Sequence

        def amplitude(cycles):
            sequence = Sequence(Pulse(90.0), CoherenceFilter(-1), tr_s=0.5, cycles=cycles)
            _, amps = engine.lines(sequence, self.system(), b1_scale=0.7)
            return float(np.abs(ops.convert_to_numpy(amps)).sum())

        one, four, eight = amplitude(1), amplitude(4), amplitude(8)
        assert abs(one - eight) / eight < 0.2
        assert abs(four - eight) / eight < 1e-3

    #*****************#
    #   graph inputs  #
    #*****************#
    def test_tr_is_a_timing_of_the_design(self):
        # two repetition times of one design share the graph; None is another design
        assert (Press(0.011, 0.019, tr_s=1.0).structure()
                == Press(0.011, 0.019, tr_s=2.0).structure())
        assert Press(0.011, 0.019, tr_s=1.0).structure() != Press(0.011, 0.019).structure()
        assert Press(0.011, 0.019, tr_s=2.0).timings()[-1] == (2.0,)
        assert Press(0.011, 0.019).timings()[-1] == ()
        assert abs(Press(0.011, 0.019).duration_s - 0.030) < 1e-12
