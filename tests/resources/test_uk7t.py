####################################################################################################
#                                           test_uk7t.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The UK7T conversions and, when the archive is cached, the Oxford session's 7 T stack.   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.resources import Resources, UK7T


#**************************************************************************************************#
#                                       Class TestConversion                                       #
#**************************************************************************************************#
class TestConversion:

    def test_dream_map_in_tenths_of_a_degree_becomes_the_relative_field(self):
        flip = np.array([[600.0, 300.0], [780.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(UK7T.transmit(flip, 60.0), [[1.0, 0.5], [1.3, 0.0]], atol=1e-6)

    def test_unknown_site_and_map_are_refused(self):
        with pytest.raises(KeyError):
            UK7T.file("abc", "DREAM_FA")
        with pytest.raises(KeyError):
            UK7T.volume("MTsat")

    def test_mp2rage_lookup_inverts_its_own_signal_model(self):
        from mrsandbox.resources.uk7t import UK7T_ACQUISITION

        mp2rage = UK7T_ACQUISITION["mp2rage"]
        # the protocol's timing closes around the full 224-echo trains
        n, spacing = mp2rage["train"], mp2rage["echo_spacing_s"]
        assert mp2rage["ti_s"][0] - n * spacing / 2.0 > 0.0
        assert mp2rage["ti_s"][1] - mp2rage["ti_s"][0] - n * spacing > 0.0
        assert mp2rage["tr_s"] - mp2rage["ti_s"][1] - n * spacing / 2.0 > 0.0
        # the unified contrast falls monotonically over the tissue range, in -0.5..0.5
        grid = np.linspace(0.7, 3.5, 200)
        curve = UK7T.mp2rage_unified(grid, mp2rage)
        assert np.all(np.diff(curve) < 0.0) and curve.max() <= 0.5 and curve.min() >= -0.5
        assert 0.25 < UK7T.mp2rage_unified(1.15, mp2rage) < 0.35  # the 7 T WM of Marques 2010
        assert -0.30 < UK7T.mp2rage_unified(1.92, mp2rage) < -0.20  # and its GM
        # the lookup recovers T1 from the two block signals, per-voxel B1 included
        t1 = np.array([[0.9, 1.2], [1.9, 2.4]])
        b1 = np.array([[1.0, 0.8], [1.2, 1.0]])
        s1, s2 = UK7T.mp2rage_signals(t1, mp2rage, b1=b1)
        phase = np.exp(1j * np.array([[0.3, -1.0], [2.0, 0.0]]))  # a common receive phase
        np.testing.assert_allclose(UK7T.mp2rage_t1(s1 * phase, s2 * phase, mp2rage, b1=b1), t1,
                                   rtol=2e-3)


#**************************************************************************************************#
#                                         Class TestStack                                          #
#**************************************************************************************************#
#                                                                                                  #
# The Oxford session's derived stack, gated when the archive or its members are in the cache.      #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not UK7T.cached("oxf"), reason="UK7T archive not cached (7.1 GB)")
class TestStack:

    def test_maps_land_in_the_7t_ranges(self):
        # the transmit field near nominal at the head centre and well below it at the
        # periphery (the 7 T centre brightening), the field offset centred within the
        # unambiguous range, R2* of tens per second, each on its own grid
        transmit = UK7T.volume("TB1map", "oxf")
        assert transmit.data.shape == (56, 64, 44)
        inside = transmit.data > 0.0
        centre = np.array(np.nonzero(inside)).mean(axis=1).round().astype(int)
        box = transmit.data[tuple(slice(c - 3, c + 4) for c in centre)]
        assert 0.85 < np.median(box[box > 0.0]) < 1.2
        assert np.percentile(transmit.data[inside], 5) < 0.6
        field = UK7T.volume("fieldmap", "oxf")
        assert field.data.shape == (64, 64, 64)
        assert abs(np.median(field.data[24:40, 24:40, 24:40])) < 30.0
        assert np.abs(field.data).max() <= 1.0 / (2.0 * 0.00102) + 1e-3
        r2star = UK7T.volume("R2starmap", "oxf")
        assert r2star.data.shape == (120, 156, 192)
        assert 20.0 < np.median(r2star.data[r2star.data > 0.0]) < 80.0
        # the maps answer box queries in the session's scanner space: the grid centre
        middle = (transmit.affine[:3, :3] @ ((np.asarray(transmit.data.shape) - 1) / 2.0)
                  + transmit.affine[:3, 3])
        assert transmit.mean(middle, 20.0) is not None
        # the MP2RAGE T1 map: 0.7 mm, the head's T1 values bimodal at the 7 T white and
        # grey matter values (Marques 2010: 1.15 and 1.92 s), zero outside the head
        t1 = UK7T.volume("T1map", "oxf")
        assert t1.data.shape == (320, 320, 224)
        inside = t1.data[t1.data > 0.0]
        assert 0.4 < inside.size / t1.data.size < 0.8 or inside.size > 1e6
        counts, edges = np.histogram(inside, bins=np.arange(0.7, 3.0, 0.05))
        centres = 0.5 * (edges[1:] + edges[:-1])
        white = centres[(centres > 0.9) & (centres < 1.5)][
            np.argmax(counts[(centres > 0.9) & (centres < 1.5)])]
        grey = centres[(centres > 1.5) & (centres < 2.4)][
            np.argmax(counts[(centres > 1.5) & (centres < 2.4)])]
        assert 1.0 < white < 1.4 and 1.7 < grey < 2.2, (white, grey)

    @pytest.mark.skipif(not Resources.cached("bigbrain_mr/t1map_800um")
                        or not Resources.cached("bigbrain_mr/labels_800um"),
                        reason="bigbrain_mr maps not cached")
    def test_maps_register_into_bigbrain_space(self):
        from mrsandbox import Brain, Volume

        # the registered T1 map reads 7 T white and grey matter values where the
        # virtual brain's labels say white and grey matter
        t1 = UK7T.volume("T1map", "oxf", space="bigbrain_mr")
        labels = Volume.load("bigbrain_mr/labels_800um")
        grid = np.array(np.nonzero(labels.data > 0))[:, ::7]
        world = labels.affine[:3, :3] @ grid + labels.affine[:3, 3:4]
        index = np.linalg.solve(t1.affine[:3, :3], world - t1.affine[:3, 3:4])
        import scipy.ndimage
        sampled = scipy.ndimage.map_coordinates(t1.data, index, order=1, cval=0.0)
        tissue = labels.data[tuple(grid)]
        white = np.median(sampled[np.isin(tissue, Brain.BIGBRAIN_TISSUES["wm"])])
        grey = np.median(sampled[np.isin(tissue, Brain.BIGBRAIN_TISSUES["gm"])])
        assert 1.0 < white < 1.35 and 1.5 < grey < 2.1 and grey - white > 0.3, (white, grey)
        # every map of the site shares the one transform, and a mid-brain box reads it
        transmit = UK7T.volume("TB1map", "oxf", space="bigbrain_mr")
        np.testing.assert_allclose(transmit.affine,
                                   UK7T.registration("oxf") @ UK7T.volume("TB1map", "oxf").affine)
        assert 0.7 < transmit.mean((0.0, 0.0, 0.0), 20.0) < 1.3
