####################################################################################################
#                                           test_hmri.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/09/26                                                                                #
#                                                                                                  #
# Purpose: The hMRI map derivations recover known parameters from synthetic FLASH, SE/STE, and    #
#          dual-echo signals, and the demo subject's derived stack lands in the 3 T ranges.        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.resources import HMRI, Resources


#**************************************************************************************************#
#                                        Class TestFormulas                                        #
#**************************************************************************************************#
class TestFormulas:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def flash(amplitude, r1, flip_rad, tr_s, saturation=0.0):
        """The spoiled steady state with an MT saturation per TR: the signal of
        Helms 2008 before its rational approximation."""
        e1 = np.exp(-r1 * tr_s)
        return amplitude * np.sin(flip_rad) * (1.0 - saturation) * (1.0 - e1) / (
            1.0 - (1.0 - saturation) * e1 * np.cos(flip_rad))

    #****************#
    #   r1 and pd    #
    #****************#
    def test_r1_and_pd_recover_a_flash_pair(self):
        # a grid of R1 and B1 values: the rational approximation in its tangent form
        # lands within 1e-3 at the demo's angles (6 and 21 degrees, TR 25 ms), the
        # small-angle form within 3 percent, with the flips scaled by the transmit map
        r1 = np.linspace(0.5, 1.5, 5)[:, None] * np.ones((5, 3))
        b1 = np.array([0.8, 1.0, 1.2])[None, :] * np.ones((5, 3))
        tr, a, b = 0.025, np.radians(6.0), np.radians(21.0)
        pdw, t1w = self.flash(1000.0, r1, b1 * a, tr), self.flash(1000.0, r1, b1 * b, tr)
        np.testing.assert_allclose(HMRI.r1(pdw, t1w, a, b, tr, tr, b1), r1, rtol=1e-3)
        np.testing.assert_allclose(HMRI.proton_density(pdw, t1w, a, b, tr, tr, b1), 1000.0,
                                   rtol=1e-3)
        np.testing.assert_allclose(HMRI.r1(pdw, t1w, a, b, tr, tr, b1, small_angle=True), r1,
                                   rtol=0.03)
        assert HMRI.r1(np.zeros(3), t1w[0], a, b, tr, tr)[0] == 0.0  # silent voxels stay zero

    #*******************#
    #   mt saturation   #
    #*******************#
    def test_mt_saturation_recovers_the_per_tr_saturation(self):
        r1, tr, a, b = 1.0, 0.025, np.radians(6.0), np.radians(21.0)
        pdw, t1w = self.flash(1000.0, r1, a, tr), self.flash(1000.0, r1, b, tr)
        for delta in (0.01, 0.03):
            mtw = self.flash(1000.0, r1, a, tr, saturation=delta)
            got = HMRI.mt_saturation(
                np.full(2, mtw), a, tr,
                HMRI.proton_density(np.full(2, pdw), np.full(2, t1w), a, b, tr, tr,
                                    small_angle=True),
                HMRI.r1(np.full(2, pdw), np.full(2, t1w), a, b, tr, tr, small_angle=True))
            assert abs(got[0] - 100.0 * delta) < 0.05 * 100.0 * delta
        # the heuristic correction is the identity at nominal B1 and is the toolbox's
        # (1 - C) / (1 - C b1) factor elsewhere
        assert HMRI.mt_corrected(np.float32(2.0), 1.0) == pytest.approx(2.0)
        assert HMRI.mt_corrected(np.float32(2.0), 0.8) == pytest.approx(2.0 * 0.6 / 0.68, rel=1e-5)

    #*************#
    #   r2 star   #
    #*************#
    def test_r2star_fits_a_monoexponential_decay(self):
        te = tuple(0.0023 * k for k in range(1, 9))
        r2 = np.array([[20.0, 40.0], [60.0, 0.0]])
        echoes = np.stack([500.0 * np.exp(-t * r2) for t in te])
        np.testing.assert_allclose(HMRI.r2star(echoes, te), r2, atol=1e-4)
        echoes[3, 0, 0] = 0.0
        assert HMRI.r2star(echoes, te)[0, 0] == 0.0

    #****************#
    #   transmit     #
    #****************#
    def test_transmit_recovers_the_flip_ratio_from_se_ste_pairs(self):
        # the STE over SE ratio is cos(beta b1) exp(-TM / T1); the strongest pairs with
        # the tightest candidate assignment give b1 back, on either side of the
        # arccosine ambiguity
        beta = np.array([float(115 - 5 * k) for k in range(11)])
        tm, t1 = 0.0312, 1.192
        b1 = np.array([0.7, 0.9, 1.0, 1.15, 1.3])
        se = np.ones((11, 5)) * np.sin(np.radians(beta[:, None] * b1[None, :] / 2.0)) ** 3 * 100.0
        ste = se * np.cos(np.radians(beta[:, None] * b1[None, :])) * np.exp(-tm / t1)
        np.testing.assert_allclose(HMRI.transmit(se, ste, beta, tm, t1), b1, atol=1e-3)
        assert HMRI.transmit(np.zeros((11, 1)), np.zeros((11, 1)), beta, tm, t1)[0] == 0.0

    #*************#
    #   field     #
    #*************#
    def test_field_is_the_phase_over_the_echo_spacing(self):
        assert HMRI.field_hz(np.float32(np.pi), 0.00246) == pytest.approx(1.0 / (2.0 * 0.00246),
                                                                           rel=1e-5)

    #*******************#
    #   grid helpers    #
    #*******************#
    def test_resampling_and_smoothing_keep_a_uniform_map(self):
        affine = np.diag([4.0, 4.0, 4.0, 1.0])
        volume = np.ones((6, 6, 6), dtype=np.float32)
        target = np.diag([2.0, 2.0, 2.0, 1.0])
        inside = HMRI.resampled(volume, affine, target, (8, 8, 8))
        np.testing.assert_allclose(inside[:6, :6, :6], 1.0, atol=1e-6)
        volume[0] = 0.0  # a support edge does not fade under the normalized smoothing
        np.testing.assert_allclose(HMRI.smoothed(volume, affine, 8.0)[1:], 1.0, atol=1e-6)
        assert HMRI.smoothed(volume, affine, 8.0)[0].max() == 0.0


#**************************************************************************************************#
#                                         Class TestStack                                          #
#**************************************************************************************************#
#                                                                                                  #
# The demo subject's derived stack, gated when the raw files are in the local cache.               #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not HMRI.cached(), reason="hMRI raw stack not cached (about 410 MB)")
class TestStack:

    def test_maps_land_in_the_3t_ranges(self):
        # the head-masked medians: R1 between the GM and WM values, MT saturation of a
        # few percent units, R2* of tens per second, the transmit field within 30
        # percent of nominal, the field offset within the unambiguous range and
        # centred (the vendor's shim), each on its own grid
        r1 = HMRI.volume("R1map")
        inside = lambda volume: volume.data[volume.data != 0.0]
        assert 0.5 < np.median(inside(r1)) < 1.2
        assert r1.data.shape == (280, 320, 224)
        t1 = HMRI.volume("T1map")
        np.testing.assert_allclose(t1.data[r1.data > 0.0], 1.0 / r1.data[r1.data > 0.0], rtol=1e-5)
        assert 0.3 < np.median(inside(HMRI.volume("MTsat"))) < 5.0
        assert 10.0 < np.median(inside(HMRI.volume("R2starmap"))) < 50.0
        transmit = HMRI.volume("TB1map")
        assert transmit.data.shape == (48, 64, 48)
        assert 0.7 < np.median(inside(transmit)) < 1.3
        field = HMRI.volume("fieldmap")
        assert abs(np.median(inside(field))) < 30.0
        assert np.abs(field.data).max() <= 1.0 / (2.0 * 0.00246) + 1e-3
        # the transmit map reads at a mid-brain box as a relative flip near nominal
        centre = transmit.mean((0.0, 0.0, 0.0), 20.0)
        assert centre is not None and 0.7 < centre < 1.3

    @pytest.mark.skipif(not Resources.cached("bigbrain_mr/t1map_800um")
                        or not Resources.cached("bigbrain_mr/labels_800um"),
                        reason="bigbrain_mr maps not cached")
    def test_maps_register_into_bigbrain_space(self):
        import scipy.ndimage

        from mrsandbox import Brain, Volume

        # the registered 3 T T1 map reads 3 T white and grey matter values where the
        # virtual brain's labels say white and grey matter
        t1 = HMRI.volume("T1map", space="bigbrain_mr")
        labels = Volume.load("bigbrain_mr/labels_800um")
        grid = np.array(np.nonzero(labels.data > 0))[:, ::7]
        world = labels.affine[:3, :3] @ grid + labels.affine[:3, 3:4]
        index = np.linalg.solve(t1.affine[:3, :3], world - t1.affine[:3, 3:4])
        sampled = scipy.ndimage.map_coordinates(np.clip(t1.data, 0.0, 5.0), index, order=1,
                                                cval=0.0)
        tissue = labels.data[tuple(grid)]
        white = np.median(sampled[np.isin(tissue, Brain.BIGBRAIN_TISSUES["wm"])])
        grey = np.median(sampled[np.isin(tissue, Brain.BIGBRAIN_TISSUES["gm"])])
        # an affine fit of another subject (shape, then tissue classes): the tissues
        # order as at 3 T, the medians blurred by the boundaries the affine cannot
        # follow (stated)
        assert 0.9 < white < 1.3 and 1.2 < grey < 1.8 and grey - white > 0.2, (white, grey)
        transmit = HMRI.volume("TB1map", space="bigbrain_mr")
        np.testing.assert_allclose(transmit.affine,
                                   HMRI.registration() @ HMRI.volume("TB1map").affine)
        assert 0.7 < transmit.mean((0.0, 0.0, 0.0), 20.0) < 1.3

    def test_maps_register_into_the_average_brain(self):
        from mrsandbox import Brain

        # the same fit against the ICBM152 brain's tissue probabilities: the 3 T T1 map reads
        # white below grey where the average brain says so, and the transmit map covers the
        # brain near nominal
        body = Brain.load("icbm152")
        t1 = HMRI.volume("T1map", space="icbm152")
        transmit = HMRI.volume("TB1map", space="icbm152")

        def sampled(volume, mask):
            grid = np.argwhere(mask)[::5].T
            world = body.affine[:3, :3] @ grid + body.affine[:3, 3:4]
            index = np.rint(np.linalg.solve(volume.affine[:3, :3],
                                            world - volume.affine[:3, 3:4])).astype(int)
            inside = np.all((index >= 0) & (index < np.array(volume.data.shape)[:, None]), axis=0)
            return volume.data[tuple(index[:, inside])]

        white = np.median(np.clip(sampled(t1, body.wm > 0.8), 0.0, 5.0))
        grey = np.median(np.clip(sampled(t1, body.gm > 0.8), 0.0, 5.0))
        assert 0.9 < white < 1.3 and 1.2 < grey < 1.8 and grey - white > 0.2, (white, grey)
        field = sampled(transmit, (body.gm + body.wm) > 0.5)
        assert np.mean(np.isfinite(field) & (field > 0.0)) > 0.9
        assert 0.9 < np.median(field[field > 0.0]) < 1.1
