####################################################################################################
#                                           test_field.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Field map gate: a constant offset gives a pure phase ramp, the voxel shim nulls what a  #
#          shim can null (constant and linear fields) while quadratic structure survives, a box    #
#          outside the map dephases nothing, and a scan with a field renders broader lines than    #
#          the uniform field scan.                                                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, FieldMap, Press, Scanner, VirtualScan
from mrsandbox.phantom import Volume
from mrsandbox.resources import Resources

T = np.arange(1024) / 4000.0
BOX = dict(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)


#*************#
#   helpers   #
#*************#
def centred_map(values):
    affine = np.eye(4)
    affine[:3, 3] = -(np.asarray(values.shape) - 1) / 2.0
    return FieldMap.from_map(values.astype(np.float32), affine)


def quadratic_field(n=50, amplitude_hz=10.0):
    x = np.linspace(-1.0, 1.0, n)
    return amplitude_hz * (x[:, None, None] ** 2 + 0.0 * x[None, :, None] + 0.0 * x[None, None, :])


#**************************************************************************************************#
#                                         Class TestKernel                                         #
#**************************************************************************************************#
class TestKernel:

    #********************#
    #   constant field   #
    #********************#
    def test_constant_offset_is_a_pure_phase_ramp(self):
        field = centred_map(np.full((50, 50, 50), 5.0))
        kernel = field.kernel(**BOX, t_s=T, shim="none")
        assert np.abs(kernel - np.exp(2j * np.pi * 5.0 * T)).max() < 1e-5
        # and the shim nulls it entirely
        assert np.abs(field.kernel(**BOX, t_s=T) - 1.0).max() < 1e-5

    #******************#
    #   linear field   #
    #******************#
    def test_shim_nulls_a_linear_gradient(self):
        gradient = np.tile(np.linspace(-10.0, 10.0, 50), (50, 50, 1))
        field = centred_map(gradient)
        raw = field.kernel(**BOX, t_s=T, shim="none")
        shimmed = field.kernel(**BOX, t_s=T, shim="linear")
        assert np.abs(raw[-1]) < 0.2  # a 20 Hz spread dephases hard
        assert np.abs(shimmed - 1.0).max() < 1e-4  # the first-order shim removes it all

    #***************#
    #   linewidth   #
    #***************#
    def test_linewidth_reads_the_width_the_field_gives_a_line(self):
        # the box spans 20 of the 50 cells of a 20 Hz gradient: an 8 Hz spread, a line about as
        # wide unshimmed; the first-order shim leaves the 1 Hz Lorentzian alone
        gradient = np.tile(np.linspace(-10.0, 10.0, 50), (50, 50, 1))
        field = centred_map(gradient)
        assert 6.0 < field.linewidth(**BOX, shim="none") < 10.0
        assert field.linewidth(**BOX, shim="linear") == pytest.approx(1.0, abs=0.15)

    #**************#
    #   shim box   #
    #**************#
    def test_a_shim_box_larger_than_the_voxel_shims_it_worse(self):
        field = centred_map(quadratic_field())
        own = field.offsets(**BOX, shim="linear").std()
        # the parabola's linear fit over a wider box
        field.shim_box = (BOX["position_mm"], 40.0)
        # symmetric about the voxel: the same slope
        assert field.offsets(**BOX, shim="linear").std() == pytest.approx(own, rel=1e-6)
        # a box off to the side fits the parabola's flank
        field.shim_box = ((12.0, 0.0, 0.0), 20.0)
        assert field.offsets(**BOX, shim="linear").std() > 2.0 * own
        # the voxel's own box is the default's fit
        field.shim_box = (BOX["position_mm"], BOX["size_mm"])
        assert field.offsets(**BOX, shim="linear").std() == pytest.approx(own, rel=1e-9)

    #*********************#
    #   quadratic field   #
    #*********************#
    def test_quadratic_structure_survives_the_shim(self):
        field = centred_map(quadratic_field())
        shimmed = field.kernel(**BOX, t_s=T, shim="linear")
        assert np.abs(shimmed[-1]) < 0.9  # residual dephasing remains
        # the box spans 0.4 of the map, so the 10 Hz parabola is 1.6 Hz across it and
        # its residual after the linear fit about 0.3 of that
        assert field.offsets(**BOX, shim="linear").std() > 0.4

    #***********************#
    #   second order shim   #
    #***********************#
    def test_quadratic_shim_nulls_the_second_order_harmonics(self):
        x = np.linspace(-1.0, 1.0, 50)
        harmonic = 10.0 * (x[:, None, None] ** 2 - x[None, :, None] ** 2 + 0.0 * x[None, None, :])
        field = centred_map(harmonic)
        assert field.offsets(**BOX, shim="linear").std() > 0.3  # a first-order shim leaves it
        assert np.abs(field.offsets(**BOX, shim="quadratic")).max() < 1e-6
        # a non-harmonic r^2 field is beyond any shim set, on a scanner as here
        r2 = 10.0 * (x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
        assert centred_map(r2).offsets(**BOX, shim="quadratic").std() > 0.3
        # the order is the map's own default: the kernel follows it
        quad = FieldMap.from_map(harmonic.astype(np.float32), field.affine, shim="quadratic")
        assert np.abs(quad.kernel(**BOX, t_s=T) - 1.0).max() < 1e-4
        with pytest.raises(ValueError, match="known:"):
            FieldMap.from_map(harmonic.astype(np.float32), field.affine, shim="cubic")

    #*************#
    #   outside   #
    #*************#
    def test_outside_the_map_nothing_dephases(self):
        field = centred_map(quadratic_field())
        kernel = field.kernel(position_mm=(500.0, 0.0, 0.0), size_mm=20.0, t_s=T)
        assert np.abs(kernel - 1.0).max() == 0.0


#**************************************************************************************************#
#                                          Class TestScan                                          #
#**************************************************************************************************#
class TestScan:
    pytestmark = pytest.mark.render  # full-scan renders, skipped on the tensorflow CI leg

    #*******************#
    #   broader lines   #
    #*******************#
    def test_field_broadens_the_rendered_lines(self):
        # oscillation with a period below the box size: structure no shim can null
        brain = Brain.load("icbm152")
        wiggle = 4.0 * np.sin(2.0 * np.pi * np.arange(120) / 9.0)
        field = centred_map(np.tile(wiggle, (120, 120, 1)))
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))

        def naa_height(field_map):
            scan = VirtualScan(brain, *pieces, field=field_map)
            ppm, spec = scan.voxel(position_mm=(10, -20, 5), size_mm=20).spectrum(
                seed=7, npoints=8192)
            ppm, mag = np.asarray(ppm), np.abs(spec)
            return mag[np.abs(ppm - 2.008) < 0.1].max()

        assert naa_height(field) < 0.8 * naa_height(None)


#**************************************************************************************************#
#                                      Class TestBigBrainField                                     #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("bigbrain_mr/background_field_400um"),
                    reason="background field not cached; "
                           "FieldMap.load('bigbrain_mr') fetches it once")
class TestBigBrainField:

    #******************#
    #   preset loads   #
    #******************#
    def test_preset_scales_with_field_and_shims_to_subhertz(self):
        at_3t = FieldMap.load("bigbrain_mr", b0_t=3.0)
        at_7t = FieldMap.load("bigbrain_mr", b0_t=7.0)
        raw_3t = at_3t.offsets((-30.0, 10.0, 10.0), 25.0, shim="none")
        raw_7t = at_7t.offsets((-30.0, 10.0, 10.0), 25.0, shim="none")
        assert abs(raw_7t.mean() / raw_3t.mean() - 7.0 / 3.0) < 1e-3
        shimmed = at_3t.offsets((-30.0, 10.0, 10.0), 25.0)
        assert shimmed.std() < 0.5 * raw_3t.std()  # the shim removes most structure


#**************************************************************************************************#
#                                     Class TestSusceptibility                                     #
#**************************************************************************************************#
#                                                                                                  #
# The k-space dipole model: a sphere's field is zero inside and its dipole outside.                #
#                                                                                                  #
#**************************************************************************************************#
class TestSusceptibility:

    def test_a_sphere_makes_no_field_inside_and_a_dipole_outside(self):
        # with the sphere of Lorentz a susceptible sphere leaves its inside alone and
        # shows chi V (3 cos^2 - 1) / (4 pi r^3) outside: twice as strong and opposite
        # on the field's axis as on the equator
        n, radius = 96, 12.0
        x = np.arange(n) - (n - 1) / 2.0
        r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
        chi = (r2 <= radius ** 2).astype(np.float32)
        field = FieldMap.from_susceptibility(chi, np.eye(4), b0_t=1.0 / 42.5774785, shim="none")
        c, far = n // 2, n // 2 + 24
        ppm = field.offsets_hz  # one tesla-equivalent of 1 Hz per ppm
        dipole = chi.sum() / (4.0 * np.pi * 24.5 ** 3)  # the cell centres sit half a cell off
        assert abs(ppm[c, c, c]) < 1e-4
        assert abs(ppm[c, c, far] - 2.0 * dipole) < 0.03 * 2.0 * dipole
        assert abs(ppm[far, c, c] + dipole) < 0.03 * dipole
        # the main field along another index axis turns the pattern with it
        turned = FieldMap.from_susceptibility(chi, np.eye(4), b0_t=1.0 / 42.5774785, axis=0,
                                              shim="none").offsets_hz
        np.testing.assert_allclose(turned[far, c, c], ppm[c, c, far], rtol=1e-4)


#**************************************************************************************************#
#                                        Class TestQsmField                                        #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("qsm_rc2/chi"),
                    reason="the QSM challenge 2.0 head is not redistributed; place it by hand")
class TestQsmField:

    def test_the_sinuses_spoil_the_frontal_base_and_spare_the_rest(self):
        field = FieldMap.load("qsm_rc2", b0_t=3.0)
        spread = lambda position: float(np.std(field.offsets(position, 20.0, shim="linear")))
        assert spread((0.0, 0.0, 30.0)) < 4.0         # mid brain: a few hertz at 3 T
        assert spread((0.0, -45.0, -5.0)) < 4.0       # the occipital side
        assert 6.0 < spread((0.0, 45.0, -5.0)) < 25.0  # above the sinuses
        # susceptibility fields scale with the main field
        seven = FieldMap.load("qsm_rc2", b0_t=7.0)
        np.testing.assert_allclose(np.std(seven.offsets((0.0, 45.0, -5.0), 20.0, shim="linear")),
                                   spread((0.0, 45.0, -5.0)) * 7.0 / 3.0, rtol=1e-4)

    @pytest.mark.skipif(not Resources.cached("qsm_rc2/total_field"),
                        reason="the head's measured total field is not redistributed; "
                               "place it by hand")
    def test_the_measured_field_is_the_same_head_with_its_texture(self):
        # the measured total field lies on the model's grid (its zero cells outside the head's
        # labels) and carries texture the susceptibility model lacks: wider tails in a deep parietal
        # cube, a few hertz in the core all the same
        measured, model = (FieldMap.load("qsm_rc2_measured", b0_t=3.0),
                           FieldMap.load("qsm_rc2", b0_t=3.0))
        assert measured.volume.shape == Volume.load("qsm_rc2/labels").shape
        outside = np.asarray(measured.volume.data == 0.0) & (Volume.load("qsm_rc2/labels").data > 0)
        assert (np.asarray(measured.volume.data != 0.0)
                & (Volume.load("qsm_rc2/labels").data == 0)).sum() == 0
        assert outside.mean() < 0.2
        tails = lambda field: np.subtract(*np.percentile(
            field.offsets((0.0, -35.0, 20.0), 30.0, shim="linear"), [97.5, 2.5]))
        core = lambda field: np.subtract(*np.percentile(
            field.offsets((0.0, -35.0, 20.0), 30.0, shim="linear"), [84.0, 16.0]))
        assert tails(measured) > 1.2 * tails(model) and core(measured) < 8.0


#**************************************************************************************************#
#                                       Class TestFieldGroups                                      #
#**************************************************************************************************#
#                                                                                                  #
# The field inside the pulses: groups of the voxel's field distribution play the sequence at       #
# their own offset.                                                                                #
#                                                                                                  #
#**************************************************************************************************#
class TestFieldGroups:

    @staticmethod
    def ramp(half_hz):
        """A field rising linearly by +-half_hz across a 20 mm voxel at the origin."""
        shape = (40, 40, 40)
        affine = np.eye(4)
        affine[:3, 3] = -19.5
        x = (np.arange(40) - 19.5)[:, None, None] * np.ones(shape)
        return shape, affine, FieldMap.from_map((x * half_hz / 10.0).astype(np.float32), affine,
                                                shim="mean")

    def test_cells_without_signal_do_not_dephase(self):
        # a box reaching into bone or air: with `weights` (what holds signal per cell) the cells
        # with none neither broaden the distribution nor pull the shim; here the half of the ramp
        # above zero is silenced
        shape, affine, field = self.ramp(40.0)
        x = (np.arange(40) - 19.5)[:, None, None] * np.ones(shape)
        t = np.arange(512) / 4000.0
        whole = field.kernel((0.0, 0.0, 0.0), 20.0, t)
        field.weights = Volume((x < 0.0).astype(np.float32), affine)
        assert field.offsets((0.0, 0.0, 0.0), 20.0).max() <= 0.0 + 1e-6
        half = field.kernel((0.0, 0.0, 0.0), 20.0, t)
        assert np.abs(half[100]) > np.abs(whole[100]) and abs(np.abs(half[0]) - 1.0) < 1e-6

    def test_groups_split_the_distribution_and_sum_to_its_kernel(self):
        _, _, field = self.ramp(40.0)
        t = np.arange(512) / 4000.0
        groups = field.groups((0.0, 0.0, 0.0), 20.0, t, 8)
        means, weights = [g[0] for g in groups], [g[1] for g in groups]
        assert means == sorted(means) and abs(sum(weights) - 1.0) < 1e-9
        np.testing.assert_allclose(weights, 1.0 / 8.0, atol=0.01)
        np.testing.assert_allclose(sum(w * k for _, w, k in groups),
                                   field.kernel((0.0, 0.0, 0.0), 20.0, t), atol=1e-6)

    def test_groups_with_a_spread_carry_it_and_sum_to_the_kernel_times_its_envelope(self):
        # the microscopic field below the map's resolution (lineshape.gauss_hz) smears every cell's
        # offset: the split reaches beyond the map's own offsets and the weighted kernels sum to the
        # map's kernel times the Gaussian's envelope, so a sequence played per group sees the
        # spread where it plays
        _, _, field = self.ramp(40.0)
        t = np.arange(512) / 4000.0
        plain = field.groups((0.0, 0.0, 0.0), 20.0, t, 8)
        spread = field.groups((0.0, 0.0, 0.0), 20.0, t, 8, spread_hz=5.0)
        assert spread[0][0] < plain[0][0] and spread[-1][0] > plain[-1][0]
        envelope = np.exp(-(np.pi * 5.0) ** 2 / (4.0 * np.log(2.0)) * t * t)
        np.testing.assert_allclose(sum(w * k for _, w, k in spread),
                                   field.kernel((0.0, 0.0, 0.0), 20.0, t) * envelope, atol=1e-3)

    def test_a_shim_kept_while_the_head_moves_is_spoiled_by_the_move(self):
        # a linear field is shimmed away where the shim was fitted; the head moves 3 mm
        # along the ramp and the shim stays in the scanner: what is left is the ramp
        # times the move, one frequency, and a second-order field leaves its gradient
        _, _, field = self.ramp(40.0)  # 4 Hz per mm along x
        linear = FieldMap.from_map(field.offsets_hz, field.affine, shim="linear")
        # the head moved +3 mm: the voxel reads x - 3
        here, there = (0.0, 0.0, 0.0), (-3.0, 0.0, 0.0)
        assert np.abs(linear.offsets(here, 10.0)).max() < 1e-6
        assert np.abs(linear.offsets(there, 10.0)).max() < 1e-6  # reshimmed there
        kept = linear.offsets(there, 10.0, shim_at_mm=here)
        np.testing.assert_allclose(kept, -4.0 * 3.0, atol=1e-6)
        x = (np.arange(40) - 19.5)[:, None, None] * np.ones((40, 40, 40))
        bowl = FieldMap.from_map((0.5 * x * x).astype(np.float32), field.affine, shim="linear")
        fresh, spoiled = bowl.offsets(there, 10.0), bowl.offsets(there, 10.0, shim_at_mm=here)
        assert np.std(spoiled) > 2.0 * np.std(fresh)  # a gradient of 3 Hz per mm is back

    @pytest.mark.render
    def test_the_water_alone_per_group_is_the_whole_sequence_per_group(self):
        # under a CHESS train the water of the part of the voxel the field moves out of the 60 Hz
        # band survives: VirtualScan(water_groups=8) plays the water alone per field group and gives
        # the water component the field_groups=8 render gives, at the cost of the water alone; at
        # the mean offset (one group) the train takes it all
        from mrsandbox import Chess, NoDownfield, NoLipids, NoMM, Params, Unsuppressed

        shape, affine, field = self.ramp(60.0)
        ones = np.ones(shape, dtype=np.float32)
        brain = Brain.from_maps(gm=ones, wm=0.0 * ones, csf=0.0 * ones, affine=affine)
        # (no microscopic spread: the water alone plays it through the train, the whole sequence
        # keeps it as an envelope on the result)
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0), b1_scale=1.0,
                       lineshape=dict(gauss_hz=0.0),
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
                       oov=dict(fraction=0.0))

        def water(**groups):
            scan = VirtualScan(brain, Scanner.preset("generic_3t"),
                               Chess(3) + Press(te1_s=0.011, te2_s=0.019),
                               params=quiet, field=field, water=Unsuppressed(), lipids=NoLipids(),
                               mm=NoMM(), downfield=NoDownfield(), **groups)
            voxel = scan.voxel(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
            return voxel._component_fids(voxel.sample(seed=3), 512, 1.0 / 4000.0)["water"]

        whole, alone, mean = water(field_groups=8), water(water_groups=8), water()
        np.testing.assert_allclose(alone, whole, atol=1e-4 * np.abs(whole).max())
        assert np.abs(mean).max() < 1e-3 * np.abs(alone).max()

    @pytest.mark.render
    def test_the_water_plays_its_microscopic_spread_through_the_train(self):
        # the microscopic field (lineshape.gauss_hz) is a spread of frequencies and a train
        # suppresses each at its own: without a train the spread is the Gaussian envelope on the
        # unspread water exactly (nothing the water meets depends on its frequency); Siemens' train
        # played at its null (the flips scaled by 1.2, where the relaxation through its 24 ms
        # pulses leaves half a percent) leaves parts of the spread water at phases apart, and a
        # tenth of what an envelope on the unspread residual would say cancels. The ideal train
        # cannot show it: its box takes every spin alike.
        from mrsandbox import NoDownfield, NoLipids, NoMM, Params, Unsuppressed
        from mrsandbox.sequences import SiemensChess

        shape, affine, field = self.ramp(2.0)
        ones = np.ones(shape, dtype=np.float32)
        brain = Brain.from_maps(gm=ones, wm=0.0 * ones, csf=0.0 * ones, affine=affine)
        t = np.arange(512) / 4000.0
        # the voxel plays the drawn width less the ramp's own, in quadrature (Voxel.gauss_hz)
        played = np.sqrt(36.0 - (field.linewidth((0.0, 0.0, 0.0), 20.0) - 1.0) ** 2)
        envelope = np.exp(-(np.pi * played) ** 2 / (4.0 * np.log(2.0)) * t * t)

        def water(sequence, gauss_hz):
            quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0), b1_scale=1.0,
                           lineshape=dict(gauss_hz=gauss_hz),
                           acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                            delay_s=0.0, eddy_transient_rad=0.0),
                           oov=dict(fraction=0.0))
            scan = VirtualScan(brain, Scanner.preset("generic_3t"), sequence, params=quiet,
                               field=field, water=Unsuppressed(), lipids=NoLipids(), mm=NoMM(),
                               downfield=NoDownfield(), water_groups=8)
            voxel = scan.voxel(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
            return voxel._component_fids(voxel.sample(seed=3), 512, 1.0 / 4000.0)["water"]

        press = Press(te1_s=0.011, te2_s=0.019)
        sharp, spread = water(press, 0.0), water(press, 6.0)
        full = np.abs(sharp).max()
        np.testing.assert_allclose(spread, sharp * envelope, atol=2e-3 * full)
        train = SiemensChess(real=True).scaled(1.2)
        sharp, spread = water(train + press, 0.0), water(train + press, 6.0)
        assert np.abs(sharp).max() < 6e-3 * full  # at the null
        assert np.abs(spread).max() < 0.92 * np.abs(sharp * envelope).max()

    @pytest.mark.render
    def test_an_editing_pulse_misses_the_part_of_the_voxel_the_field_moves_away(self):
        # GABA alone under MEGA-PRESS: with the field on the FID only, a ramp broadens
        # the edited line and keeps its area; played per group, the part of the voxel
        # beyond the 60 Hz editing band is not edited (and where the band reaches the
        # 2.28 ppm protons, 52 Hz from its centre, the pattern changes), a small ramp
        # changes nothing
        from mrsandbox import (MegaPress, Multiplex, NoDownfield, NoLipids, NoMM, NoWater, Params)

        conc = {name: 0.0 for name in VirtualScan.default_params().conc}
        conc["GABA"] = 2.0
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0), conc=conc, b1_scale=1.0,
                       lineshape=dict(gauss_hz=0.0),      # the same envelope under every ramp
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
                       oov=dict(fraction=0.0))
        ppm = np.fft.fftshift(np.fft.fftfreq(1024, 1.0 / 4000.0)) / (3.0 * 42.5774785)
        window = (ppm > 2.6) & (ppm < 3.4)

        def edited(half_hz, groups):
            shape, affine, field = self.ramp(half_hz)
            ones = np.ones(shape, dtype=np.float32)
            brain = Brain.from_maps(gm=ones, wm=0.0 * ones, csf=0.0 * ones, affine=affine)
            multiplex = Multiplex(brain, Scanner.preset("generic_3t"), MegaPress.pair(),
                                  params=quiet, field=field, field_groups=groups, water=NoWater(),
                                  lipids=NoLipids(), mm=NoMM(), downfield=NoDownfield())
            data = np.asarray(multiplex.acquire((0.0, 0.0, 0.0), 20.0, seed=3,
                                                npoints=1024)[:])[0, 0, 0]
            # the FID is referenced to 0 ppm, a line at f hertz is exp(+i 2 pi f t); one envelope
            # on every case, so a truncated FID rings alike under every ramp
            t = np.arange(1024) / 4000.0
            fid = (data[:, 0] - data[:, 1]) * np.exp(-(np.pi * 6.0) ** 2 / (4.0 * np.log(2.0)) * t * t)
            return np.real(np.fft.fftshift(np.fft.fft(fid)))[window].sum()

        uniform = edited(0.0, 1)
        assert abs(edited(40.0, 1) / uniform - 1.0) < 0.01
        assert abs(edited(10.0, 8) / uniform - 1.0) < 0.01
        assert 0.5 < edited(40.0, 16) / uniform < 0.8
