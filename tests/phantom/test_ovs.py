####################################################################################################
#                                           test_ovs.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The fat layer rendered where it sits: the ensemble shift, the layer's centroid, the     #
#          excitation profile acting on the lipids, and outer-volume saturation bands removing     #
#          them while the voxel's own isochromats stay untouched.                                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, Delay, Engine, Gradient, OuterVolumeSuppression, Params, Pulse,
                       Scanner, Sequence, ShapedPulse, SpinSystem, VirtualScan)
from mrsandbox.sequences.events import CoherenceFilter

GAMMA = 42.5774785
F0 = GAMMA * 3.0
QUIET = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0), b1_scale=1.0,
               acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
               gradients=dict(b0_short_hz_per_mt_m=0.0, b0_long_hz_per_mt_m=0.0,
                              linear_short=0.0, linear_long=0.0))


def half_brain():
    """Grey matter for x below zero, nothing beyond: the brain boundary is the plane x = 0."""
    shape = (24, 24, 24)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -23.0
    x = affine[0, 0] * np.arange(shape[0]) + affine[0, 3]
    gm = np.broadcast_to((x < 0.0)[:, None, None], shape).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    return Brain.from_maps(gm=gm, wm=zeros, csf=zeros, affine=affine)


def slab_echo(thickness_mm=20.0):
    """A slice-selective 90 along x exciting `thickness_mm` about the voxel for spins
    at the fat's 1.3 ppm (any other shift displaces by its chemical shift over the
    gradient), a crushed 180."""
    duration = 0.003
    gradient = 5.95 / duration / thickness_mm  # the three-lobe sinc's bandwidth over the slab
    pulse = ShapedPulse.sinc(90.0, duration, offset_ppm=1.3, gradient_hz_per_mm=gradient,
                             gradient_axis="x", extent_mm=20.0)
    crusher = Gradient(0.15, 20.0, axis="x")
    return Sequence(pulse, Gradient(-gradient * duration / 2.0, 20.0, axis="x"), Delay(0.010),
                    crusher, Pulse(180.0, phase_deg=90.0), crusher, Delay(0.0115))


#**************************************************************************************************#
#                                        Class TestShift                                           #
#**************************************************************************************************#
class TestShift:

    def test_a_shifted_ensemble_is_the_positions_moved(self):
        engine = Engine(3.0, isochromats=16)
        design = slab_echo()
        lac = SpinSystem.from_database("Lac")[0]
        args = [ops.convert_to_tensor(np.float32(v)) for v in (1.3, 0.25, 1.0, 0.0)]
        _, shifted = engine.compiled(design, lac, shift_mm={"x": 7.5})(*args)
        ctx = engine.context(lac.with_relaxation(t1_s=1.3, t2_s=0.25))
        moved = dataclasses.replace(ctx, gradient_x=ops.convert_to_tensor(engine.positions(design)["x"] + 7.5))
        _, manual = engine._system_lines(design, moved)
        np.testing.assert_allclose(np.asarray(ops.convert_to_numpy(shifted)),
                                   np.asarray(ops.convert_to_numpy(manual)), atol=1e-5)
        assert len(engine._compiled) == 1
        engine.compiled(design, lac)(*args)
        assert len(engine._compiled) == 1  # the shift is an input: one graph for every place

    def test_the_fat_centroid_sits_beyond_the_boundary(self):
        brain = half_brain()
        # a box from -15 to 5 mm: the quarter beyond x = 0 has its centroid at 2.5 mm,
        # 7.5 mm from the voxel centre
        assert brain.lipid_weight((-5.0, 0.0, 0.0), 20.0) == pytest.approx(0.25, abs=0.05)
        np.testing.assert_allclose(brain.lipid_offset((-5.0, 0.0, 0.0), 20.0), [7.5, 0.0, 0.0], atol=0.6)
        np.testing.assert_array_equal(brain.lipid_offset((-15.0, 0.0, 0.0), 10.0), [0.0, 0.0, 0.0])


#**************************************************************************************************#
#                                        Class TestLipids                                          #
#**************************************************************************************************#
class TestLipids:

    @staticmethod
    def lipid_amplitude(sequence, position_mm=(-5.0, 0.0, 0.0)):
        from mrsandbox import NoLipids

        kw = dict(brain=half_brain(), scanner=Scanner.preset("generic_3t"), params=QUIET, isochromats=16)
        with_fat = VirtualScan(sequence=sequence, **kw)
        without = VirtualScan(sequence=sequence, lipids=NoLipids(), **kw)
        seed, n = 3, 1024
        _, fat = with_fat.voxel(position_mm=position_mm, size_mm=20.0).fid(seed=seed, npoints=n)
        _, none = without.voxel(position_mm=position_mm, size_mm=20.0).fid(seed=seed, npoints=n)
        return np.abs(fat - none)[0], np.abs(none)[0]

    def test_lipids_see_the_excitation_profile_where_they_sit(self):
        engine = Engine(3.0, isochromats=16)
        design = slab_echo()
        # the profile of the slab excitation at the fat's positions against the voxel's
        probe = SpinSystem(name="fat", nucleus="1H", shifts_ppm=(1.3,), j_hz=((0.0,),), scale=1.0,
                           gamma_mhz_t=GAMMA)
        ctx = engine.context(probe)

        def excited(shift):
            total = 0.0
            for x in engine.positions(design)["x"] + shift:
                _, a = engine._system_lines(design, dataclasses.replace(ctx, gradient_x=float(x)))
                total += abs(complex(np.asarray(ops.convert_to_numpy(ops.sum(a)))))
            return total / 16.0

        expected = excited(7.5) / excited(0.0)
        assert 0.45 < expected < 0.8
        # the acquisition's lipids follow it: the shifted render against a centred one
        shifted, _ = self.lipid_amplitude(design)
        unlocalized = Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(0.010), Gradient(0.15, 20.0, axis="x"),
                               Pulse(180.0, phase_deg=90.0), Gradient(0.15, 20.0, axis="x"), Delay(0.0115))
        centred, _ = self.lipid_amplitude(unlocalized)
        assert abs(shifted / centred - expected) < 0.1 * expected

    def test_outer_volume_bands_saturate_the_fat_and_spare_the_voxel(self):
        design = slab_echo()
        fat, voxel = self.lipid_amplitude(design)
        # a band from 12 to 22 mm (for the fat's shift) lies beyond the excited slab:
        # what it saturates was never excited, so nothing changes but the slab's
        # stopband ripple (a few percent of the fat there)
        beyond = OuterVolumeSuppression((("x", 17.0, 10.0),), f0_mhz=F0, reference_ppm=1.3)
        fat_beyond, voxel_beyond = self.lipid_amplitude(beyond + design)
        assert abs(voxel_beyond / voxel - 1.0) < 0.01 and abs(fat_beyond / fat - 1.0) < 0.03
        # the fat this voxel sees sits inside its box (0 to 5 mm beyond the brain): a
        # band over it removes the excited fat, the sinc's edges saturating only in
        # part, and that quarter of the voxel with it; the fat ensemble's third that
        # reaches into the brain (its extent about the centroid, stated) stays
        over = OuterVolumeSuppression((("x", 2.5, 5.0),), f0_mhz=F0, reference_ppm=1.3)
        fat_over, voxel_over = self.lipid_amplitude(over + design)
        assert 0.3 < fat_over / fat < 0.7, fat_over / fat
        assert 0.65 < voxel_over / voxel < 0.85, voxel_over / voxel
        wide = OuterVolumeSuppression((("x", 5.0, 10.0),), f0_mhz=F0, reference_ppm=1.3)
        fat_wide, _ = self.lipid_amplitude(wide + design)
        assert fat_wide < fat_over, (fat_wide, fat_over)
        # every band is a physical event: playable on the scanner
        assert (over + design).feasible(Scanner.preset("generic_3t"))


#**************************************************************************************************#
#                                    Class TestProfiledLipids                                      #
#**************************************************************************************************#
#                                                                                                  #
# A projected design sees the fat layer cell by cell through the product of its axis profiles.     #
#                                                                                                  #
#**************************************************************************************************#
class TestProfiledLipids:

    @staticmethod
    def head(fat):
        """A 1 mm grid of 80 mm, grey matter everywhere, `fat` a function of the z centres."""
        shape = (80, 80, 80)
        affine = np.eye(4)
        affine[:3, 3] = -39.5
        z = np.arange(shape[2]) - 39.5
        ones = np.ones(shape, dtype=np.float32)
        layer = np.broadcast_to(fat(z).astype(np.float32)[None, None, :], shape)
        return Brain.from_maps(gm=ones, wm=0.0 * ones, csf=0.0 * ones, affine=affine, fat=layer)

    @staticmethod
    def lipids(brain, design):
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), design, params=QUIET, isochromats=32)
        voxel = scan.voxel(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
        truth = voxel.sample(seed=3)
        lines = scan.lipids.lines(scan.engine, voxel.design(truth), voxel, truth, F0)
        amps = np.zeros(1) if lines is None else np.asarray(ops.convert_to_numpy(lines[1]))
        return amps, truth, scan

    def test_fat_everywhere_is_the_projected_design_itself(self):
        # the identity behind the product of profiles: with fat filling the ensemble's
        # span, the cell sum is the average the design's own Projections take
        from mrsandbox import LocalizedPress
        from mrsandbox.phantom.lipids import Skull

        design = LocalizedPress(0.015, 0.015, F0)
        amps, truth, scan = self.lipids(self.head(lambda z: np.ones_like(z)), design)
        expected = []
        for system in Skull.systems(GAMMA):
            system = system.with_relaxation(t1_s=truth.lipids.t1_s, t2_s=truth.lipids.t2_s)
            _, a = Engine(3.0, isochromats=128).lines(design, system)
            expected.append(np.asarray(ops.convert_to_numpy(a)) * truth.lipids.amplitude_mm)
        expected = np.concatenate(expected)
        np.testing.assert_allclose(amps, expected, atol=0.02 * np.abs(expected).max())

    def test_fat_beyond_the_box_reaches_press_and_hardly_semi_laser(self):
        from mrsandbox import LocalizedPress, LocalizedSemiLaser, Press

        # a fat layer from 12 to 20 mm along z, in both directions: the nominal 20 mm
        # box (to 10 mm) holds none, the ideal design sees none
        brain = self.head(lambda z: (np.abs(z) > 12.0) & (np.abs(z) < 20.0))
        assert brain.lipid_weight((0.0, 0.0, 0.0), 20.0) == 0.0
        ideal = np.abs(self.lipids(brain, Press(te1_s=0.015, te2_s=0.015))[0]).sum()
        press = np.abs(self.lipids(brain, LocalizedPress(0.015, 0.015, F0))[0]).sum()
        laser = np.abs(self.lipids(brain, LocalizedSemiLaser(0.030, F0))[0]).sum()
        assert ideal == 0.0
        # the 0.86 kHz refocusing slab of the fat's 1.3 ppm sits 5 mm off the 3.0 ppm
        # reference and its feet reach further; the 10 kHz GOIA slab moves 0.4 mm
        assert press > 100.0
        assert laser < 0.05 * press
