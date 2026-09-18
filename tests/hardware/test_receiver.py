####################################################################################################
#                                         test_receiver.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The digital receiver: the decimation filter's onset corrupts the first points of a FID  #
#          and leaves the rest of an in-band signal alone.                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from scipy import signal

from mrsandbox import Brain, Params, Press, Receiver, Scanner, VirtualScan


#**************************************************************************************************#
#                                         Class TestFilter                                         #
#**************************************************************************************************#
class TestFilter:

    def test_a_step_gives_the_running_sum_of_the_taps(self):
        # a constant signal switched on at the acquisition's start: output n is the
        # part of the filter that has seen it, half at the first point
        receiver = Receiver(oversampling=4, span_dwells=6)
        fine_n, _ = receiver.grid(32, 2.5e-4)
        out = receiver.decimate(np.ones(fine_n, dtype=np.complex64), 32)
        taps = signal.firwin(2 * 24 + 1, 0.25, window="hamming")
        np.testing.assert_allclose(out.real[:7], np.cumsum(taps)[24::4], atol=1e-6)
        assert abs(out[0].real - 0.5 * (1.0 + taps[24])) < 1e-6
        np.testing.assert_allclose(out[6:], 1.0, atol=1e-6)

    def test_the_group_delay_is_removed(self):
        # an in-band line away from the onset comes out where it went in, to the
        # passband ripple of the window (a few tenths of a percent) and with no phase
        receiver = Receiver()
        fine_n, fine_dwell = receiver.grid(256, 2.5e-4)
        line = lambda n, dwell: np.exp((2j * np.pi * 300.0 - 20.0) * np.arange(n) * dwell)
        out = receiver.decimate(line(fine_n, fine_dwell), 256)
        np.testing.assert_allclose(out[16:], line(256, 2.5e-4)[16:], atol=5e-3)
        assert np.abs(np.angle(out[16:64] / line(256, 2.5e-4)[16:64])).max() < 1e-4


#**************************************************************************************************#
#                                          Class TestScan                                          #
#**************************************************************************************************#
@pytest.mark.render
class TestScan:

    def test_the_first_points_carry_the_filter_onset(self):
        brain = Brain.load("icbm152")
        pieces = (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        quiet = Params(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))
        kw = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        ideal = VirtualScan(brain, *pieces, params=quiet).voxel(**kw)
        real = VirtualScan(brain, *pieces, params=quiet, receiver=Receiver()).voxel(**kw)
        clean, filtered = ideal.fid(seed=7, npoints=512)[1], real.fid(seed=7, npoints=512)[1]
        assert 0.45 < abs(filtered[0] / clean[0]) < 0.65
        assert np.abs(filtered[1:8] - clean[1:8]).max() > 5e-3 * np.abs(clean).max()
        assert np.abs(filtered[16:] - clean[16:]).max() < 2e-3 * np.abs(clean).max()
        # the components pass the same linear filter and the header names the receiver
        whole = real.acquire(seed=7, npoints=256)
        parts = real.acquire_components(seed=7, npoints=256)
        np.testing.assert_allclose(np.asarray(parts[:]).sum(axis=-1), np.asarray(whole[:]),
                                   atol=1e-3 * np.abs(clean).max())
        assert whole.hdr_ext["SimulationPhantom"]["receiver"] == {"oversampling": 8, "span_dwells": 8}
