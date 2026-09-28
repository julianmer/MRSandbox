####################################################################################################
#                                           test_girf.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The measured gradient impulse response: the prediction reproduces the exponential terms #
#          typed as a response, the receiver's band removes what lies beyond it, and the deposit's #
#          3 T Prisma response reads as a unit self term with mechanical resonances.               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.resources import GIRF, Resources
from mrsandbox.resources.girf import GIRF_MEASUREMENT
from mrsandbox.scan.eddy import induced_fields

DT = 1e-5
GRID = np.fft.fftshift(np.fft.fftfreq(40000, GIRF_MEASUREMENT["dwell_s"]))


def trapezoid(n=40000, start=5000, ramp=20, flat=200, amplitude=30.0):
    edges = [start, start + ramp, start + ramp + flat, start + 2 * ramp + flat]
    return np.interp(np.arange(n), edges, [0.0, 1.0, 1.0, 0.0], left=0.0, right=0.0) * amplitude


#**************************************************************************************************#
#                                       Class TestPrediction                                       #
#**************************************************************************************************#
class TestPrediction:

    def test_exponential_terms_typed_as_a_response_reproduce_the_exponential_fields(self):
        # one exponential term as its transfer function: a step G u(t) induces
        # b0 G exp(-t / tau), so H(w) = b0 i w tau / (1 + i w tau) in the B0 term (per
        # mT/m, in mT through the deposit's gammabar) and 1 minus the linear fraction
        # times the same in the self term
        tau, b0_hz, linear = 0.005, 0.4, 0.01
        w = 2.0 * np.pi * GRID
        decay = 1j * w * tau / (1.0 + 1j * w * tau)
        girf = GIRF({"z": {"b0": (GRID, b0_hz / GIRF_MEASUREMENT["gammabar_hz_per_mt"] * decay),
                           "z": (GRID, 1.0 - linear * decay)}}, name="synthetic")
        waveform = trapezoid()
        fields = girf("z", waveform, DT, bandwidth_hz=1e6)
        b0, error = induced_fields(waveform, DT, [tau], [b0_hz], [linear])
        np.testing.assert_allclose(fields["b0"], b0, atol=2e-3 * np.abs(b0).max())
        np.testing.assert_allclose(fields["z"], error, atol=2e-3 * np.abs(error).max())
        assert set(fields) == {"b0", "z"}

    def test_the_receiver_band_removes_a_resonance_beyond_it(self):
        # a B0 term that is one mechanical resonance at 3 kHz and nothing else
        f0, width = 3000.0, 100.0
        peak = 1e-4 * np.exp(-((np.abs(GRID) - f0) / width) ** 2)
        girf = GIRF({"x": {"b0": (GRID, peak)}}, name="synthetic")
        waveform = trapezoid()
        wide = girf("x", waveform, DT, bandwidth_hz=20000.0)["b0"]
        narrow = girf("x", waveform, DT, bandwidth_hz=4000.0)["b0"]
        after = np.arange(waveform.size) > 5300
        assert np.abs(wide[after]).max() > 100.0 * np.abs(narrow[after]).max()
        # the ringing the wide band keeps is the resonance
        spectrum = np.abs(np.fft.rfft(wide[after]))
        assert abs(np.fft.rfftfreq(after.sum(), DT)[np.argmax(spectrum[1:]) + 1] - f0) < 100.0


#**************************************************************************************************#
#                                        Class TestDeposit                                         #
#**************************************************************************************************#
#                                                                                                  #
# The deposit's 3 T Prisma response, gated when the release archive or its members are cached.     #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not GIRF.cached(), reason="NWS_Spectroscopy archive not cached (25 MB)")
class TestDeposit:

    def test_the_registry_serves_the_archive(self):
        assert Resources.cached("nws/release_zip") or GIRF.file("Results39993", "x").exists()
        assert GIRF.file("Results19999", "z").exists()

    def test_the_prisma_response_is_a_unit_self_term_with_mechanical_resonances(self):
        girf = GIRF.load()
        assert girf.name == "nws/prisma_3t"
        for axis in "xyz":
            terms = girf.terms[axis]
            assert set(terms) == {"b0", "x", "y", "z"}
            # the B0 and self terms at the paper's 5 Hz, the cross terms at 10 Hz
            for term, (grid, response) in terms.items():
                step = grid[1] - grid[0]
                assert abs(step - (5.0 if term in ("b0", axis) else 10.0)) < 0.01, (axis, term)
                assert response.shape == grid.shape
            at_zero = {term: response[np.argmin(np.abs(grid))]
                       for term, (grid, response) in terms.items()}
            assert abs(at_zero[axis] - 1.0) < 2e-3  # the nominal gradient plays
            assert all(abs(at_zero[term]) < 2e-3 for term in "xyz" if term != axis)
            assert abs(at_zero["b0"]) < 1e-4
            # within a readout's band the B0 term peaks at a mechanical resonance in
            # the low kilohertz (the paper's sidebands)
            grid, response = terms["b0"]
            band = (np.abs(grid) > 100.0) & (np.abs(grid) < 3000.0)
            resonance = abs(grid[band][np.argmax(np.abs(response[band]))])
            assert 1000.0 < resonance < 2000.0, (axis, resonance)

    def test_the_self_term_steps_to_the_nominal_gradient(self):
        girf = GIRF.load()
        step = np.zeros(60000)
        step[20000:] = 10.0
        t = (np.arange(step.size) - 20000) * DT
        for axis in "xyz":
            fields = girf(axis, step, DT, bandwidth_hz=6000.0)
            played = fields[axis] + step
            assert abs(played[t > 0.05].mean() - 10.0) < 0.02
            assert np.abs(played[t < -0.002]).max() < 0.2  # causal but for the band's edge
            assert np.abs(fields["b0"][t > 0.0]).max() < 20.0  # hertz per 10 mT/m step


#**************************************************************************************************#
#                                     Class TestReverseDeposit                                     #
#**************************************************************************************************#
@pytest.mark.skipif(not Resources.cached("reverse/philips_3t"),
                    reason="reVERSE-GIRF file not cached (46 KB)")
class TestReverseDeposit:

    def test_the_philips_response_holds_the_self_terms_only(self):
        girf = GIRF.philips("3t")
        assert girf.name == "reverse/philips_3t" and GIRF.named(girf.name).name == girf.name
        assert all(list(girf.terms[axis]) == [axis] for axis in "xyz")
        # a gradient follows its demand at low frequency and falls off in the kilohertz
        grid, response = girf.terms["z"]["z"]
        assert (abs(abs(response[np.abs(grid).argmin()]) - 1.0) < 0.03
                and abs(response[np.abs(grid - 5000.0).argmin()]) < 0.8)
        # a crusher's edge rings in its own axis for milliseconds and no B0 term comes with it
        crusher = np.zeros(4000)
        crusher[100:1100] = 20.0
        fields = girf("z", crusher, 1e-5, 20000.0)
        assert (list(fields) == ["z"] and abs(fields["z"][1105]) > 1.0
                and abs(fields["z"][3000]) < 0.01)
