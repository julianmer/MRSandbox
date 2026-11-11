####################################################################################################
#                                         test_spurious.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/11/26                                                                                #
#                                                                                                  #
# Purpose: The spurious echo sources: the drawn echo stays the default, the pathway echo is       #
#          nothing under a perfect crush and lands at its pathway's own time otherwise.           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import (Brain, Chess, CoherenceFilter, Delay, Params, ParametricEcho, PathwayEcho,
                       Press, Pulse, Scanner, SelectivePulse, Sequence, VirtualScan)

NPOINTS, DWELL_S = 1024, 1.0 / 4000.0
SCALP, DEEP = (72.0, -20.0, 5.0), (10.0, -20.0, 5.0)
STILL = dict(noise=dict(sigma=0.0), oov=dict(fraction=0.0),
             spurious=dict(amplitude=0.05, width_s2=4000.0),
             acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                              eddy_transient_rad=0.0),
             gradients=dict(b0_short_hz_per_mt_m=0.0, b0_long_hz_per_mt_m=0.0,
                            linear_short=0.0, linear_long=0.0))


@pytest.fixture(scope="module")
def brain():
    return Brain.load("icbm152")


def echo_of(brain, sequence, source, position=SCALP, **params):
    scan = VirtualScan(brain, Scanner.preset("generic_3t"), sequence, params=Params(**{**STILL,
                                                                                       **params}),
                       spurious=source)
    voxel = scan.voxel(position_mm=position, size_mm=20.0)
    truth = voxel.sample(seed=7)
    parts = voxel._component_fids(truth, NPOINTS, DWELL_S)
    dry = np.abs(sum(fid for name, fid in parts.items() if name not in ("water", "spurious"))).max()
    return parts.get("spurious"), dry


#**************************************************************************************************#
#                                        Class TestSpurious                                        #
#**************************************************************************************************#
class TestSpurious:

    #***************#
    #   the drawn   #
    #***************#
    def test_the_drawn_echo_stays_the_default(self, brain):
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        assert isinstance(scan.spurious, ParametricEcho)

    #**************#
    #   silences   #
    #**************#
    def test_the_pathway_echo_is_silent_off_and_away_from_the_scalp(self, brain):
        design = Chess(2) + Press(te1_s=0.011, te2_s=0.019)
        off, _ = echo_of(brain, design, PathwayEcho(), spurious=dict(amplitude=0.0))
        deep, _ = echo_of(brain, design, PathwayEcho(), position=DEEP)
        assert off is None and deep is None

    #*********************#
    #   a perfect crush   #
    #*********************#
    def test_a_perfect_crush_gives_no_echo(self, brain):
        # a region of the voxel's own edges under an ideal localization: no slab edge cuts it,
        # every crusher is whole cycles over it and no pathway's net moment is nothing, so what
        # the crushers leave is what the filters leave, to the replays' float precision, against
        # a region holding several voxels of water
        class Box(PathwayEcho):
            def region(self, voxel):
                _, centre, fat_mm3 = super().region(voxel)
                return np.full(3, 20.0), centre, fat_mm3

        echo, dry = echo_of(brain, Chess(1) + Press(te1_s=0.011, te2_s=0.019), Box())
        assert echo is not None
        assert np.abs(echo).max() < 1e-4 * dry

    #**********************#
    #   the pathway time   #
    #**********************#
    def test_the_echo_comes_at_its_pathways_time(self, brain):
        # a suppression pulse, a gap T, a 90 and one 180 at te: the scalp's water the first pulse
        # excites and the 180 alone refocuses echoes T after the readout's start; the lone
        # crushers short of a whole cycle let that pathway through
        class Leaky(PathwayEcho):
            LONE = 0.3

        gap_s, te_s = 0.050, 0.020
        design = Sequence(SelectivePulse(90.0, offset_ppm=4.65, bandwidth_hz=60.0),
                          CoherenceFilter(0), Delay(gap_s), Pulse(90.0), CoherenceFilter(+1),
                          Delay(te_s / 2.0), CoherenceFilter(+1), Pulse(180.0, phase_deg=90.0),
                          CoherenceFilter(-1), Delay(te_s / 2.0))
        echo, dry = echo_of(brain, design, Leaky())
        t = np.arange(NPOINTS) * DWELL_S
        late = np.where(t > 0.01, np.abs(echo), 0.0)
        # the T2 decay pulls the envelope's peak a little early: by 1 / (2 W T2), about a ms
        assert gap_s - 0.003 <= t[late.argmax()] <= gap_s + 2.0 * DWELL_S
        assert late.max() > 1e-3 * dry
