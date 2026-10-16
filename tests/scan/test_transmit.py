####################################################################################################
#                                         test_transmit.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The transmit field per isochromat: a vector transmit scale batches the pulses like the  #
#          positions batch the frame, equals a loop over isochromats, and a transmit map varying   #
#          across the voxel reads at each isochromat's position.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, Delay, Engine, Params, Pulse, PulseAcquire, Scanner, Sequence,
                       ShapedPulse, SpinSystem, VirtualScan, Volume)
from mrsandbox.sequences.events import CoherenceFilter, Gradient

GAMMA = 42.5774785


@pytest.fixture(scope="module")
def engine():
    return Engine(3.0, isochromats=16)


def slice_echo():
    """A slice-selective spin echo along z with a crusher pair."""
    pulse = ShapedPulse.sinc(90.0, 0.002, gradient_hz_per_mm=GAMMA * 5.0, gradient_axis="z",
                             offset_ppm=4.65)
    crusher = Gradient(0.15, 20.0, axis="z")
    return Sequence(pulse, Gradient(-pulse.gradient_hz_per_mm * 0.001, 20.0, axis="z"),
                    Delay(0.010), crusher, Pulse(180.0, phase_deg=90.0), crusher, Delay(0.012))


#**************************************************************************************************#
#                                     Class TestVectorTransmit                                     #
#**************************************************************************************************#
class TestVectorTransmit:

    def test_a_vector_scale_equals_a_loop_over_isochromats(self, engine):
        lac = SpinSystem.from_database("Lac")[0].with_relaxation(t1_s=1.3, t2_s=0.25)
        design = slice_echo()
        positions = engine.positions(design)["z"]
        profile = (1.0 + 0.02 * positions).astype(np.float32)  # 2 percent per mm
        ctx = engine.context(lac)
        batched = engine._ensemble_context(
            design, dataclasses.replace(ctx, b1_scale=ops.convert_to_tensor(profile)), 16)
        _, together = engine._system_lines(design, batched)
        looped = []
        for z, b1 in zip(positions, profile):
            single = dataclasses.replace(ctx, gradient_z=float(z), b1_scale=float(b1))
            looped.append(np.asarray(ops.convert_to_numpy(engine._system_lines(design, single)[1])))
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(together)),
                                   np.mean(looped, axis=0), atol=1e-6 * np.abs(looped).max())
        # the compiled graph takes the vector too
        fn = engine.compiled(design, SpinSystem.from_database("Lac")[0])
        args = [ops.convert_to_tensor(v)
                for v in (np.float32(1.3), np.float32(0.25), profile, np.float32(0.0))]
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(fn(*args)[1])),
                                   np.asarray(ops.convert_to_numpy(together)),
                                   atol=1e-5 * np.abs(looped).max())

    def test_hard_pulses_follow_the_mean_of_the_flip_law(self, engine):
        # a single spin excited by a hard 90 at a spread of transmit scales: the
        # detected amplitude is the mean of sin(b1 pi / 2) over the isochromats
        probe = SpinSystem(name="probe", nucleus="1H", shifts_ppm=(2.0,), j_hz=((0.0,),), scale=1.0,
                           gamma_mhz_t=GAMMA)
        profile = np.linspace(0.6, 1.2, 16).astype(np.float32)
        # an ensemble, no dephasing
        design = Sequence(Pulse(90.0), CoherenceFilter(-1), Gradient(0.0, 20.0))
        _, amps = engine.lines(design, probe, b1_scale=ops.convert_to_tensor(profile))
        total = abs(complex(np.asarray(ops.convert_to_numpy(ops.sum(amps)))))
        _, nominal = engine.lines(PulseAcquire(), probe)
        reference = abs(complex(np.asarray(ops.convert_to_numpy(ops.sum(nominal)))))
        assert abs(total / reference - np.mean(np.sin(profile * np.pi / 2.0))) < 1e-5


#**************************************************************************************************#
#                                      Class TestTransmitMap                                       #
#**************************************************************************************************#
class TestTransmitMap:

    @staticmethod
    def scan(transmit, sequence):
        shape = (24, 24, 24)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = -23.0
        ones = np.ones(shape, dtype=np.float32)
        brain = Brain.from_maps(gm=ones * 0.5, wm=ones * 0.5, csf=ones * 0.0, affine=affine)
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0), b1_scale=1.0,
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                        eddy_transient_rad=0.0),
                       gradients=dict(b0_short_hz_per_mt_m=0.0, b0_long_hz_per_mt_m=0.0,
                                      linear_short=0.0, linear_long=0.0))
        return VirtualScan(brain, Scanner.preset("generic_7t"), sequence, params=quiet,
                           transmit=transmit, isochromats=8)

    def test_profile_reads_the_map_at_the_isochromats(self):
        shape = (24, 24, 24)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = -23.0
        z = affine[2, 2] * np.arange(shape[2]) + affine[2, 3]
        ramp = np.broadcast_to((1.0 + 0.02 * z)[None, None, :], shape).astype(np.float32)
        scan = self.scan(Volume(ramp, affine, name="ramp"), slice_echo())
        voxel = scan.voxel(position_mm=(0.0, 0.0, 4.0), size_mm=20.0)
        truth = voxel.sample(seed=3)
        profile = voxel.b1_profile(truth)
        positions = scan.engine.positions(scan.sequence)["z"]
        np.testing.assert_allclose(profile, 1.0 + 0.02 * (4.0 + positions),
                                   atol=0.02)  # nearest cells
        assert abs(profile.mean() - voxel.b1_scale(truth)) < 0.02
        # a design without gradients has no ensemble, so the box mean is all there is
        assert self.scan(Volume(ramp, affine), PulseAcquire()).voxel(
            position_mm=(0.0, 0.0, 4.0), size_mm=20.0).b1_profile(truth) is None
        assert voxel.frame(truth)[3] == (voxel.b1_scale(truth), profile.tobytes())

    @pytest.mark.slow

    def test_a_uniform_map_equals_the_scalar_path_and_a_ramp_moves_the_signal(self):
        shape = (24, 24, 24)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = -23.0
        z = affine[2, 2] * np.arange(shape[2]) + affine[2, 3]
        flat = np.full(shape, 0.9, dtype=np.float32)
        ramp = np.broadcast_to((0.9 + 0.03 * z)[None, None, :], shape).astype(np.float32)
        kw = dict(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
        with_map = self.scan(Volume(flat, affine), slice_echo())
        without = self.scan(None, slice_echo())
        _, mapped = with_map.voxel(**kw).fid(seed=3, npoints=512)
        scalar = VirtualScan(without.brain, without.scanner, without.sequence, isochromats=8,
                             params=without.params | Params(b1_scale=0.9))
        _, plain = scalar.voxel(**kw).fid(seed=3, npoints=512)
        np.testing.assert_allclose(mapped, plain, atol=1e-5 * np.abs(plain).max())
        # a field varying across the voxel differs from its box mean
        _, ramped = self.scan(Volume(ramp, affine), slice_echo()).voxel(**kw).fid(
            seed=3, npoints=512)
        assert np.abs(ramped - plain).max() > 1e-2 * np.abs(plain).max()
