####################################################################################################
#                                         test_downfield.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: Downfield gate: the nine peaks sit at their published positions, water presaturation    #
#          attenuates each peak by exactly its two-site retention R1 / (R1 + kex) with the         #
#          MacMillan 3 T values, tissue scales the amplitudes, and the peaks reach the rendered    #
#          spectrum.                                                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import (Brain, NoDownfield, ParameterizedDownfield, Params, Press, Scanner,
                       Tissues, VirtualScan)
from mrsandbox.database import downfield

F0_MHZ = 42.5774785 * 3.0
# the echo time the published levels belong to (Ozdemir 2023): no echo loss there
AT_LEVEL_TE = Press(te1_s=0.011, te2_s=0.011)


#**************************************************************************************************#
#                                          Class FakeVoxel                                         #
#**************************************************************************************************#
#                                                                                                  #
# A stub carrying only the fractions the source protocol reads.                                    #
#                                                                                                  #
#**************************************************************************************************#
class FakeVoxel:
    """A stub carrying only the fractions the source protocol reads."""

    def __init__(self, gm=0.6, wm=0.35, csf=0.05):
        self.fractions = Tissues(gm=gm, wm=wm, csf=csf)


#*************#
#   helpers   #
#*************#
def realize(**overrides):
    params = VirtualScan.default_params() | Params(downfield=overrides)
    return params.sample(seed=3)


#**************************************************************************************************#
#                                     Class TestDownfieldLines                                     #
#**************************************************************************************************#
class TestDownfieldLines:

    #****************#
    #   nine peaks   #
    #****************#
    def test_nine_peaks_at_their_published_positions(self):
        truth = realize(presaturated=False)
        freqs, amps, gauss = ParameterizedDownfield().lines(None, AT_LEVEL_TE, FakeVoxel(), truth,
                                                            F0_MHZ)
        expected = sorted(peak["ppm"] * F0_MHZ for peak in downfield.OZDEMIR.values())
        np.testing.assert_allclose(sorted(freqs), expected, rtol=1e-6)
        assert amps.size == 9 and np.all(amps.real > 0.0)
        # every peak carries its digitized 3 T basis width times the drawn scale
        widths = [downfield.POVAZAN_WIDTHS[name]["fwhm_hz"] for name in downfield.OZDEMIR]
        np.testing.assert_allclose(gauss, np.array(widths) * truth.downfield.width_scale, rtol=1e-6)
        assert 14.0 <= min(widths) and max(widths) <= 32.0

    #***************#
    #   echo time   #
    #***************#
    def test_a_later_echo_costs_each_peak_its_apparent_t2(self):
        # the levels are what is left at 22 ms; 30 ms later the 6.83 ppm peak (24.4 ms
        # at 7 T, Fichtner 2017) keeps exp(-30 / 24.4) and the 8.37 ppm amide (16.3 ms
        # at 9.4 T, Borbath 2021) exp(-30 / 16.3); an earlier echo holds more
        truth = realize(presaturated=False)
        source = ParameterizedDownfield()
        _, at, _ = source.lines(None, AT_LEVEL_TE, FakeVoxel(), truth, F0_MHZ)
        _, later, _ = source.lines(None, Press(te1_s=0.022, te2_s=0.030), FakeVoxel(), truth,
                                   F0_MHZ)
        _, earlier, _ = source.lines(None, Press(te1_s=0.006, te2_s=0.006), FakeVoxel(), truth,
                                     F0_MHZ)
        names = list(downfield.OZDEMIR)
        ratio = (later / at).real
        assert abs(ratio[names.index("df683")] - np.exp(-30.0 / 24.4)) < 1e-6
        assert abs(ratio[names.index("df837")] - np.exp(-30.0 / 16.3)) < 1e-6
        assert np.all((earlier / at).real > 1.0)

    #*******************#
    #   presaturation   #
    #*******************#
    def test_presaturation_retains_exactly_r1_over_r1_plus_kex(self):
        # per-leaf sampling keeps every other draw identical between the two trees,
        # so the amplitude ratio isolates the retention factor exactly
        voxel = FakeVoxel()
        source = ParameterizedDownfield()
        _, full, _ = source.lines(None, AT_LEVEL_TE, voxel, realize(presaturated=False), F0_MHZ)
        _, saturated, _ = source.lines(None, AT_LEVEL_TE, voxel, realize(presaturated=True), F0_MHZ)
        expected = [source.retention(name) for name in downfield.OZDEMIR]
        np.testing.assert_allclose((saturated / full).real, expected, rtol=1e-6)
        # spot checks against the published MacMillan medians
        r1 = 1000.0 / 254.0
        assert abs(source.retention("df824") - r1 / (r1 + 7.5)) < 1e-12
        r1 = 1000.0 / 525.0
        assert abs(source.retention("df790") - r1 / (r1 + 0.5)) < 1e-12

    #********************#
    #   tissue scaling   #
    #********************#
    def test_tissue_scales_and_empty_voxels_are_silent(self):
        truth = realize(presaturated=False)
        source = ParameterizedDownfield()
        _, half, _ = source.lines(None, AT_LEVEL_TE, FakeVoxel(gm=0.3, wm=0.2), truth, F0_MHZ)
        _, full, _ = source.lines(None, AT_LEVEL_TE, FakeVoxel(gm=0.6, wm=0.4), truth, F0_MHZ)
        np.testing.assert_allclose(2.0 * half.real, full.real, rtol=1e-6)
        assert source.lines(None, AT_LEVEL_TE, FakeVoxel(gm=0.0, wm=0.0, csf=1.0), truth,
                            F0_MHZ) is None
        assert NoDownfield().lines(None, AT_LEVEL_TE, FakeVoxel(), truth, F0_MHZ) is None


#**************************************************************************************************#
#                                     Class TestDownfieldInScan                                    #
#**************************************************************************************************#
@pytest.mark.render  # a full-scan render, skipped on the tensorflow CI leg
class TestDownfieldInScan:

    #********************#
    #   rendered peaks   #
    #********************#
    def test_downfield_replaces_the_nh_lines_and_presaturates_exactly(self):
        brain = Brain.load("icbm152")
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                        delay_s=0.0, eddy_transient_rad=0.0),
                       oov=dict(fraction=0.0))
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))

        def spectrum(**kwargs):
            params = quiet | kwargs.pop("params", Params())
            scan = VirtualScan(brain, *pieces, params=params, **kwargs)
            ppm, spec = scan.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0).spectrum(
                seed=7, npoints=8192)
            return np.asarray(ppm), np.asarray(ops.convert_to_numpy(spec))

        zeroed = Params(downfield=dict(level={name: 0.0 for name in downfield.OZDEMIR}))
        ppm, none = spectrum(downfield=NoDownfield())
        _, base = spectrum(params=zeroed)
        _, saturated = spectrum()
        _, full = spectrum(params=Params(downfield=dict(presaturated=False)))
        window = (ppm > 7.7) & (ppm < 8.5)
        # the spin systems' narrow full-amplitude NH lines leave the render in favour
        # of the empirical peaks: the region drops well below its NoDownfield level
        assert np.abs(saturated[window]).max() < np.abs(none[window]).max() / 3.0
        # same seed and per-leaf draws, so subtracting the zero-level render isolates
        # the source; presaturation then scales the amides by exactly their retention
        gain_full = (full - base)[window]
        gain_sat = (saturated - base)[window]
        assert (np.abs(gain_full).max()
                > 3.0 * np.abs((full - base)[(ppm > 9.5) & (ppm < 11.0)]).max())
        amides = (ppm[window] > 8.05) & (ppm[window] < 8.30)
        at = np.argmax(np.abs(gain_full) * amides)
        ratio = (gain_sat[at] / gain_full[at]).real
        assert abs(ratio - ParameterizedDownfield.retention("df818")) < 0.03
