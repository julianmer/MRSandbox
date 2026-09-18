####################################################################################################
#                                        test_calibrate.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: A scanner's own raw data in, its noise level out: the reading recovers the noise a      #
#          virtual acquisition was made with, through coils, transients, and both readings.        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import Brain, Params, Press, Scanner, VirtualScan
from mrsandbox.scan.calibrate import NoiseReading
from mrsandbox.scan.voxel import noise_scale


#**************************************************************************************************#
#                                      Class TestNoiseReading                                      #
#**************************************************************************************************#
@pytest.mark.render
class TestNoiseReading:

    def test_the_reading_recovers_the_noise_the_data_were_made_with(self):
        # one subject of a virtual scan, four channels of known sensitivity, 48 transients of
        # white noise at a known reference sigma: read as raw data, the twin gives it back
        calm = Params(spurious=dict(amplitude=0.0), acquisition=dict(shift_hz=0.0, phase_deg=0.0,
                                                                     eddy_std_rad=0.0))
        scan = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                           Press(te1_s=0.011, te2_s=0.024, tr_s=2.0), params=calm)
        voxel = scan.voxel(position_mm=(0.0, -45.0, 35.0), size_mm=30.0)
        n, dwell_s, sigma = 2048, 1.0 / 2000.0, 5.0
        _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
        # a scanner's carrier sits on water; the package's FIDs are referenced to 0 ppm
        fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n) * dwell_s)
        per_point = sigma * noise_scale(voxel.volume_ml, 3.0, dwell_s)
        rng = np.random.default_rng(1)
        b = np.array([0.8, 0.5, 0.3, 0.1j])
        b = b / np.linalg.norm(b)
        data = b[None, :, None] * np.asarray(fid)[:, None, None] + per_point * (
            rng.standard_normal((n, 4, 48)) + 1j * rng.standard_normal((n, 4, 48)))
        # the package's FIDs run exp(+i 2 pi f t); raw scanner data come either way round
        for stored in (data, np.conj(data)):
            reading = NoiseReading.from_arrays(stored, dwell_s, voxel._f0_mhz(),
                                               header={"Manufacturer": "virtual"})
            assert reading.coils == 4 and reading.transients == 48 and abs(reading.naa_ppm - 2.008) < 0.03
            plain, apodized = reading.sigma(voxel, seeds=1)
            assert abs(plain / sigma - 1.0) < 0.08 and abs(apodized / sigma - 1.0) < 0.08
        record = reading.record(voxel, "a virtual acquisition", presets=("generic_3t",), seeds=1)
        assert record.leaf == "noise.sigma" and abs(record.value / sigma - 1.0) < 0.08
        assert record.scanner["vendor"] == "virtual" and record.coil["channels"] == (4,)
        assert record.prior[0] < record.value < record.prior[1] and record.departures(scan) == {}

    def test_averaged_data_are_refused(self, tmp_path):
        data = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                           Press(te1_s=0.011, te2_s=0.024)).voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=256)
        path = tmp_path / "averaged.nii.gz"
        data.save(path)
        with pytest.raises(ValueError, match="transients"):
            NoiseReading.from_nifti(path)
