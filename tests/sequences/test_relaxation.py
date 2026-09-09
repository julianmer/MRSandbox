####################################################################################################
#                                        test_relaxation.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Relaxation gate: in-sequence relaxation follows the analytic laws. Echo attenuation is  #
#          exactly exp(-TE/T2), suppression-train recovery is exactly 1 - exp(-delay/T1), systems  #
#          without relaxation stay purely coherent, and the scan path carries the TE dependence.   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, Engine, Params, Press, PulseAcquire, Scanner, SpinSystem,
                       SuppressionTrain, VirtualScan)
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
        quiet = Params(acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_deg=0.0),
                       spurious=dict(amplitude=0.0))
        heights, truth = [], None
        for te in (0.03, 0.06):
            scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te / 2, te / 2),
                               params=quiet)
            voxel = scan.voxel(position_mm=(10, -20, 5), size_mm=20)
            truth = voxel.sample(seed=7)  # same seed, same draw for both scans
            ppm, spec = voxel.spectrum(seed=7, npoints=8192)
            ppm, spec = np.asarray(ops.convert_to_numpy(ppm)), ops.convert_to_numpy(spec)
            heights.append(float(spec.real[np.argmin(np.abs(ppm - 2.008))]))
        expected = math.exp(-0.03 / truth.t2.NAA)  # the drawn Gudmundson-pooled NAA T2
        assert abs(heights[1] / heights[0] - expected) < 0.05
