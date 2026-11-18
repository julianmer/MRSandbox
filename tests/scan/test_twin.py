####################################################################################################
#                                           test_twin.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/11/26                                                                                #
#                                                                                                  #
# Purpose: The twin's version gate: the numbers the readers invert the data through, at one fixed  #
#          voxel, are those of TWIN_VERSION; a change of the physics that moves one fails here,    #
#          so the version is raised and the readings inverted under the earlier twin are known     #
#          as stale (scan.calibrations stale) until they are inverted again.                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pytest

from mrsandbox import Brain, Press, Scanner, VirtualScan
from mrsandbox.phantom.water import Unsuppressed
from mrsandbox.scan.calibrate import MacromoleculeReading, NoiseReading, SpectrumReading, _echo_top
from mrsandbox.scan.calibrations import TWIN_VERSION, stale

# the twin's numbers per version: the NAA width without broadening and creatine over NAA, the
# noise level an SNR of 40 (150 apodized) implies, the 0.9 ppm peak over NAA per draw, and the
# water's echo tops at TE 68 and 144 over TE 30's
FINGERPRINTS = {
    1: {"naa_fwhm_hz": 3.6525, "creatine_over_naa": 0.63508, "sigma": (2.4392, 1.7685),
        "peak_over_naa": (0.13421, 0.17617), "tops": (0.68073, 0.36132)},
}


#**************************************************************************************************#
#                                          Class TestTwin                                          #
#**************************************************************************************************#
class TestTwin:

    #*************#
    #   version   #
    #*************#
    def test_the_twins_numbers_are_those_of_its_version(self):
        # a change of the physics that moves these raises TWIN_VERSION with its new numbers
        # here: the readings inverted under the earlier twin are then stale
        expected = FINGERPRINTS[TWIN_VERSION]
        brain, scanner = Brain.load("icbm152"), Scanner.preset("generic_3t")
        place = dict(position_mm=(0.0, -20.0, 20.0), size_mm=20.0)
        voxel = VirtualScan(brain, scanner, Press(0.011, 0.019, tr_s=2.0)).voxel(**place)
        f0 = voxel._f0_mhz()
        width = SpectrumReading(naa_fwhm_hz=8.0, creatine_over_naa=0.6, transients=64, coils=1,
                                npoints=2048, bandwidth_hz=4000.0, f0_mhz=f0,
                                header={}).twin(voxel, seeds=2, gauss_hz=0.0)
        assert width["naa_fwhm_hz"] == pytest.approx(expected["naa_fwhm_hz"], rel=1e-3)
        assert width["creatine_over_naa"] == pytest.approx(expected["creatine_over_naa"], rel=1e-3)
        sigma = NoiseReading(snr=40.0, snr_apodized=150.0, naa_ppm=2.01, transients=64, coils=1,
                             npoints=2048, bandwidth_hz=4000.0, f0_mhz=f0, excitations=1,
                             header={}, line="NAA").sigma(voxel, seeds=2)
        assert sigma == pytest.approx(expected["sigma"], rel=1e-3)
        peak = MacromoleculeReading(peak_over_naa=0.3, transients=64, npoints=2048, coils=1,
                                    dwell_s=1.0 / 4000.0, f0_mhz=f0,
                                    header={}).twin(voxel, seeds=2)["peak_over_naa"]
        assert peak == pytest.approx(expected["peak_over_naa"], rel=1e-3)
        truth, tops = voxel.sample(0), []
        for te in (0.030, 0.068, 0.144):
            reference = VirtualScan(brain, scanner, Press(0.011, te - 0.011, tr_s=2.0),
                                    water=Unsuppressed()).voxel(**place)
            tops.append(_echo_top(reference._water_parts(
                truth, reference.scan.engine, reference.design(truth), reference._f0_mhz(),
                reference._diffusion_factor(truth)), 1.0 / 4000.0))
        assert [t / tops[0] for t in tops[1:]] == pytest.approx(expected["tops"], rel=1e-3)

    #***********#
    #   stale   #
    #***********#
    def test_a_reading_inverted_under_another_twin_is_stale(self):
        import dataclasses

        from mrsandbox.scan.calibrations import CREATINE_THREE_DATASETS_3T

        # inverted through the twin and stamped with another version (or none): stale; this
        # version's: not; a reading straight off the data never is
        old = CREATINE_THREE_DATASETS_3T
        assert stale(old)
        fresh = dataclasses.replace(old, measured={**old.measured, "twin": TWIN_VERSION})
        assert not stale(fresh)
        raw = dataclasses.replace(old, leaf="oov.fraction",
                                  measured={"statistic": "the residual's broad part over the "
                                                         "reference's full water"})
        assert not stale(raw)
