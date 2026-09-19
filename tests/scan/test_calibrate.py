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
from mrsandbox.scan.calibrate import CovarianceReading, NoiseReading, SpectrumReading
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
        # a receiver phase cycle left in the stored transients changes nothing
        cycled = NoiseReading.from_arrays(data * np.exp(0.5j * np.pi * (np.arange(48) % 4)), dwell_s, voxel._f0_mhz())
        assert abs(cycled.snr / reading.snr - 1.0) < 0.02
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


#**************************************************************************************************#
#                                   Class TestCovarianceReading                                    #
#**************************************************************************************************#
class TestCovarianceReading:

    def test_the_reading_recovers_the_array_it_was_drawn_from(self):
        # channel noise of a loaded helmet whose channels keep 0.4 of their noise to themselves:
        # the covariance comes back, and with it the share
        from mrsandbox import CoilArray

        array = CoilArray.helmet().loaded(own_noise=0.4)
        noise = array.noise(np.random.default_rng(2), 3.0, 1 << 16)            # (coils, samples)
        reading = CovarianceReading.from_arrays(noise.T, header={"Manufacturer": "virtual"})
        assert reading.coils == 32 and reading.samples == 1 << 16
        r = array.noise_resistance
        np.testing.assert_allclose(np.abs(reading.resistance()), np.abs(r / np.diag(r).mean()), atol=0.03)
        assert abs(reading.own_noise(CoilArray.helmet()) - 0.4) < 0.03
        record = reading.record(CoilArray.helmet(), "a virtual noise scan", load="the model's head sphere")
        assert record.leaf == "coils.loaded.own_noise" and abs(record.value - 0.4) < 0.03
        assert record.measured["strongest_pair"] > record.measured["strongest_partner_mean"] > record.measured["median_pair"]

    def test_an_ismrmrd_file_is_read_off_its_flagged_noise_lines(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        rng = np.random.default_rng(3)
        mix = np.array([[1.0, 0.0], [0.6, 0.8]])                      # a noise correlation of 0.6
        head = np.dtype([("flags", "<u8"), ("active_channels", "<u2")])
        row = np.dtype([("head", head), ("data", h5py.vlen_dtype(np.dtype("<f4")))])
        table = np.zeros(6, dtype=row)
        for k in range(6):
            noise = k < 4
            x = mix @ (rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096)))
            x = x if noise else x + 50.0                               # signal, the same in both channels
            table[k]["head"] = ((1 << 18) if noise else 0, 2)
            table[k]["data"] = np.stack([x.real, x.imag], axis=-1).astype("<f4").ravel()
        with h5py.File(tmp_path / "scan.h5", "w") as file:
            file.create_dataset("dataset/data", data=table)
            file.create_dataset("dataset/xml", data=[b"<systemVendor>virtual</systemVendor>"])
        reading = CovarianceReading.from_ismrmrd(tmp_path / "scan.h5")
        assert reading.samples == 4 * 4096 and abs(reading.statistics()["strongest_pair"] - 0.6) < 0.02
        assert CovarianceReading.from_ismrmrd(tmp_path / "scan.h5", lines=[4, 5]).statistics()["strongest_pair"] > 0.99

    def test_the_tails_of_raw_fids_serve_and_combined_data_do_not(self, tmp_path):
        from mrsandbox import CoilArray

        scan = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.024),
                           coils=CoilArray.ring(8).loaded(own_noise=0.5))
        data = scan.voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=4096)
        path = tmp_path / "channels.nii.gz"
        data.save(path)
        reading = CovarianceReading.from_nifti(path)
        assert reading.coils == 8 and reading.statistics()["strongest_pair"] < 0.9
        plain = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.024))
        single = tmp_path / "single.nii.gz"
        plain.voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=256).save(single)
        with pytest.raises(ValueError, match="channels"):
            CovarianceReading.from_nifti(single)


#**************************************************************************************************#
#                                    Class TestSpectrumReading                                     #
#**************************************************************************************************#
@pytest.mark.render
class TestSpectrumReading:

    def test_linewidth_and_creatine_read_the_same_off_data_and_twin(self):
        # a virtual two-channel acquisition of one subject, the carrier on water: what the
        # reader finds in the raw data is what the twin shows for that subject, and a subject
        # with half the creatine reads half the ratio
        calm = Params(spurious=dict(amplitude=0.0), acquisition=dict(shift_hz=0.0, phase_deg=0.0,
                                                                     eddy_std_rad=0.0))
        pieces = (Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0))
        voxel = VirtualScan(*pieces, params=calm).voxel(position_mm=(0.0, -45.0, 35.0), size_mm=30.0)
        n, dwell_s = 2048, 1.0 / 2000.0

        def raw(subject):
            _, fid = subject._clean_fid(subject.sample(0), n, dwell_s)
            fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * subject._f0_mhz() * np.arange(n) * dwell_s)
            rng = np.random.default_rng(4)
            noise = 0.02 * np.abs(fid).max() * (rng.standard_normal((n, 2, 16)) + 1j * rng.standard_normal((n, 2, 16)))
            return np.array([0.8, 0.6])[None, :, None] * fid[:, None, None] + noise

        reading = SpectrumReading.from_arrays(raw(voxel), dwell_s, voxel._f0_mhz())
        twin = reading.twin(voxel, seeds=1)
        assert reading.coils == 2 and reading.transients == 16
        assert abs(reading.naa_fwhm_hz / twin["naa_fwhm_hz"] - 1.0) < 0.05
        assert abs(reading.creatine_over_naa / twin["creatine_over_naa"] - 1.0) < 0.03
        truth = voxel.sample(0)
        drawn = float(voxel.resolve(truth.conc.Cr)) + float(voxel.resolve(truth.conc.PCr))
        assert abs(reading.total_creatine_mm(voxel, seeds=1) / drawn - 1.0) < 0.03
        lean = VirtualScan(*pieces, params=calm | Params(conc=dict(Cr=0.5 * float(truth.conc.Cr),
                                                                    PCr=0.5 * float(truth.conc.PCr))))
        halved = SpectrumReading.from_arrays(raw(lean.voxel(position_mm=(0.0, -45.0, 35.0), size_mm=30.0)),
                                             dwell_s, voxel._f0_mhz())
        assert 0.45 < halved.creatine_over_naa / reading.creatine_over_naa < 0.65
