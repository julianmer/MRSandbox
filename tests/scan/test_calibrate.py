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
from mrsandbox.scan.calibrate import (CovarianceReading, NoiseReading, RelaxationReading,
                                      SpectrumReading)
from mrsandbox.scan.voxel import noise_scale


#**************************************************************************************************#
#                                      Class TestNoiseReading                                      #
#**************************************************************************************************#
@pytest.mark.render
class TestNoiseReading:

    def test_the_reading_recovers_the_noise_the_data_were_made_with(self):
        # one subject of a virtual scan, four channels of known sensitivity, 48 transients of
        # white noise at a known reference sigma: read as raw data, the twin gives it back
        calm = Params(spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                       eddy_transient_rad=0.0), oov=dict(fraction=0.0))
        scan = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                           Press(te1_s=0.011, te2_s=0.024, tr_s=2.0), params=calm)
        voxel = scan.voxel(position_mm=(0.0, -45.0, 35.0), size_mm=30.0)
        n, dwell_s, sigma = 2048, 1.0 / 2000.0, 5.0
        _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
        # a scanner's carrier sits on water; the package's FIDs are referenced to 0 ppm
        fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n)
                                       * dwell_s)
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
            assert (reading.coils == 4 and reading.transients == 48
                    and abs(reading.naa_ppm - 2.008) < 0.03)
            plain, apodized = reading.sigma(voxel, seeds=1)
            assert abs(plain / sigma - 1.0) < 0.08 and abs(apodized / sigma - 1.0) < 0.08
        # a receiver phase cycle left in the stored transients changes nothing
        cycled = NoiseReading.from_arrays(data * np.exp(0.5j * np.pi * (np.arange(48) % 4)),
                                          dwell_s, voxel._f0_mhz())
        assert abs(cycled.snr / reading.snr - 1.0) < 0.02
        record = reading.record(voxel, "a virtual acquisition", presets=("generic_3t",), seeds=1)
        assert record.leaf == "noise.sigma" and abs(record.value / sigma - 1.0) < 0.08
        assert record.scanner["vendor"] == "virtual" and record.coil["channels"] == (4,)
        assert record.prior[0] < record.value < record.prior[1] and record.departures(scan) == {}

    def test_a_twin_held_at_the_runs_own_readings_reads_the_run_alone(self):
        # two twins whose residual water and delay differ read one run alike once held at the
        # run's own readings, and apart when drawn
        calm = Params(spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                                       eddy_transient_rad=0.0), oov=dict(fraction=0.0))
        twins = [VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                             Press(te1_s=0.011, te2_s=0.024, tr_s=2.0),
                             params=calm | Params(water=dict(adjusted_residual=residual),
                                                  acquisition=dict(delay_s=delay))).voxel(
            position_mm=(0.0, -45.0, 35.0), size_mm=30.0)
            for residual, delay in ((0.01, -0.0002), (0.05, -0.0006))]
        n, dwell_s = 2048, 1.0 / 2000.0
        _, fid = twins[0]._clean_fid(twins[0].sample(0), n, dwell_s)
        fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * twins[0]._f0_mhz() * np.arange(n)
                                       * dwell_s)
        rng = np.random.default_rng(4)
        data = fid[:, None, None] + 0.05 * np.abs(fid).max() * (
            rng.standard_normal((n, 1, 16)) + 1j * rng.standard_normal((n, 1, 16)))
        reading = NoiseReading.from_arrays(data, dwell_s, twins[0]._f0_mhz())
        held = {"water.adjusted_residual": 0.01, "acquisition.delay_s": -0.0002}
        assert (reading.sigma(twins[0], seeds=1, held=held)
                == reading.sigma(twins[1], seeds=1, held=held))
        assert reading.sigma(twins[0], seeds=1) != reading.sigma(twins[1], seeds=1)
        record = reading.record(twins[1], "a run", seeds=1, held=held)
        assert record.measured["twin_held_at"] == held

    def test_a_phantoms_water_line_gives_the_noise_with_the_scanners_own_noise_samples(self):
        from mrsandbox import Phantom, Unsuppressed

        # a phantom scan of known noise: unsuppressed water rings to the end of every FID, so
        # the tails do not serve and the noise comes from samples taken with the RF off
        calm = Params(spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                       eddy_transient_rad=0.0), oov=dict(fraction=0.0))
        scan = VirtualScan(Phantom.spectre(temperature_c=21.0), Scanner.preset("generic_3t"),
                           Press(te1_s=0.011, te2_s=0.024, tr_s=5.0), params=calm,
                           water=Unsuppressed())
        voxel = scan.voxel(position_mm=(0.0, 0.0, 0.0), size_mm=20.0)
        n, dwell_s, sigma = 2048, 1.0 / 2000.0, 6.0
        _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
        # a scanner's carrier sits on the water it sees
        f0 = voxel._f0_mhz()
        fid = np.asarray(fid) * np.exp(-2j * np.pi * (4.65 + 0.0103 * 16.0) * f0 * np.arange(n)
                                       * dwell_s)
        per_point = sigma * noise_scale(voxel.volume_ml, 3.0, dwell_s)
        rng = np.random.default_rng(2)
        draw = lambda *shape: per_point * (rng.standard_normal(shape)
                                           + 1j * rng.standard_normal(shape))
        b = np.array([0.7, 0.6, 0.4j])
        b = b / np.linalg.norm(b)
        data = b[None, :, None] * fid[:, None, None] + draw(n, 3, 16)
        reading = NoiseReading.from_arrays(data, dwell_s, f0, line="water", noise=draw(4 * n, 3))
        assert reading.line == "water" and reading.snr > 1000.0
        # the record's reading is the apodized one; the plain height of a line as narrow as the
        # spectral grid hangs on where the line falls between two points
        plain, apodized = reading.sigma(voxel, seeds=1)
        assert abs(apodized / sigma - 1.0) < 0.05 and abs(plain / sigma - 1.0) < 0.4
        record = reading.record(voxel, "a virtual phantom scan", seeds=1)
        assert record.voxel["load"] == "spectre" and "water" in record.measured["statistic"]
        # without the noise samples the ringing tails pass for noise and the reading is off
        tails = NoiseReading.from_arrays(data, dwell_s, f0, line="water")
        assert tails.sigma(voxel, seeds=1)[1] > 1.5 * sigma

    def test_averaged_data_are_refused(self, tmp_path):
        data = (VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                            Press(te1_s=0.011, te2_s=0.024))
                .voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=256))
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
        np.testing.assert_allclose(np.abs(reading.resistance()), np.abs(r / np.diag(r).mean()),
                                   atol=0.03)
        assert abs(reading.own_noise(CoilArray.helmet()) - 0.4) < 0.03
        record = reading.record(CoilArray.helmet(), "a virtual noise scan",
                                load="the model's head sphere")
        assert record.leaf == "coils.loaded.own_noise" and abs(record.value - 0.4) < 0.03
        assert (record.measured["strongest_pair"] > record.measured["strongest_partner_mean"]
                > record.measured["median_pair"])
        # the correlation is the data's own number; the share is taken from it against the array
        # that plays it
        correlation = reading.record_correlation("a virtual noise scan",
                                                 load="the model's head sphere")
        assert (correlation.leaf == "coils.partner_correlation"
                and correlation.value == pytest.approx(record.measured["strongest_partner_mean"],
                                                       abs=1e-3))
        assert CovarianceReading.share(CoilArray.helmet(), correlation.value) == pytest.approx(
            reading.own_noise(CoilArray.helmet()), abs=0.01)

    def test_multi_coil_kspace_gives_its_noise_beyond_the_head(self, tmp_path):
        # fastMRI's layout: slices, coils, a readout oversampled twofold, phase encodes, half of
        # them sampled; an object in the readout's middle half and noise of a known covariance
        # everywhere: the outer readout reads the noise alone
        import h5py

        rng = np.random.default_rng(4)
        mixing = np.eye(4) + 0.5 * rng.standard_normal((4, 4))
        slices, coils, readout, lines = 3, 4, 128, 64
        noise = np.einsum("ij,sjrl->sirl", mixing,
                          rng.standard_normal((slices, coils, readout, lines))
                          + 1j * rng.standard_normal((slices, coils, readout, lines)))
        image = noise.copy()
        image[:, :, readout // 4: 3 * readout // 4] += 50.0 * rng.standard_normal(
            (coils, readout // 2, lines))  # the head
        kspace = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image, axes=2), axis=2),
                                 axes=2).astype(np.complex64)
        header = ("<ismrmrdHeader><acquisitionSystemInformation>"
                  "<systemVendor>SIEMENS</systemVendor><systemModel>Skyra</systemModel>"
                  "<systemFieldStrength_T>2.8936</systemFieldStrength_T><coilLabel>"
                  "<coilNumber>1</coilNumber><coilName>HeadNeck_20:1:H11"
                  "</coilName></coilLabel></acquisitionSystemInformation></ismrmrdHeader>")
        with h5py.File(tmp_path / "scan.h5", "w") as file:
            file["kspace"] = kspace
            file["mask"] = (np.arange(lines) % 2 == 0).astype(np.float32)
            file["ismrmrd_header"] = header
        reading = CovarianceReading.from_kspace(tmp_path / "scan.h5")
        known = CovarianceReading(covariance=mixing @ mixing.conj().T, samples=0, header={})
        assert reading.coils == 4 and reading.samples == slices * (readout // 4) * (lines // 2)
        assert reading.statistics()["strongest_partner_mean"] == pytest.approx(
            known.statistics()["strongest_partner_mean"], abs=0.03)
        assert reading.header == {"systemVendor": "SIEMENS", "systemModel": "Skyra",
                                  "systemFieldStrength_T": 2.8936, "ReceiveCoilName": "HeadNeck_20"}

    def test_an_ismrmrd_file_is_read_off_its_flagged_noise_lines(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        rng = np.random.default_rng(3)
        mix = np.array([[1.0, 0.0], [0.6, 0.8]])   # a noise correlation of 0.6
        head = np.dtype([("flags", "<u8"), ("active_channels", "<u2")])
        row = np.dtype([("head", head), ("data", h5py.vlen_dtype(np.dtype("<f4")))])
        table = np.zeros(6, dtype=row)
        for k in range(6):
            noise = k < 4
            x = mix @ (rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096)))
            x = x if noise else x + 50.0           # signal, the same in both channels
            table[k]["head"] = ((1 << 18) if noise else 0, 2)
            table[k]["data"] = np.stack([x.real, x.imag], axis=-1).astype("<f4").ravel()
        with h5py.File(tmp_path / "scan.h5", "w") as file:
            file.create_dataset("dataset/data", data=table)
            file.create_dataset("dataset/xml", data=[b"<systemVendor>virtual</systemVendor>"])
        reading = CovarianceReading.from_ismrmrd(tmp_path / "scan.h5")
        assert (reading.samples == 4 * 4096
                and abs(reading.statistics()["strongest_pair"] - 0.6) < 0.02)
        assert CovarianceReading.from_ismrmrd(tmp_path / "scan.h5",
                                              lines=[4, 5]).statistics()["strongest_pair"] > 0.99

    def test_the_tails_of_raw_fids_serve_and_combined_data_do_not(self, tmp_path):
        from mrsandbox import CoilArray

        scan = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                           Press(te1_s=0.011, te2_s=0.024),
                           coils=CoilArray.ring(8).loaded(own_noise=0.5))
        data = scan.voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=4096)
        path = tmp_path / "channels.nii.gz"
        data.save(path)
        reading = CovarianceReading.from_nifti(path)
        assert reading.coils == 8 and reading.statistics()["strongest_pair"] < 0.9
        plain = VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                            Press(te1_s=0.011, te2_s=0.024))
        single = tmp_path / "single.nii.gz"
        plain.voxel((0.0, -45.0, 35.0), 30.0).acquire(seed=1, npoints=256).save(single)
        with pytest.raises(ValueError, match="channels"):
            CovarianceReading.from_nifti(single)
        # a phantom's unsuppressed water rings to the end of the FID: no noise to read there
        from mrsandbox import Phantom, Unsuppressed

        ringing = VirtualScan(Phantom.spectre(), Scanner.preset("generic_3t"),
                              Press(te1_s=0.011, te2_s=0.024),
                              coils=CoilArray.ring(8).loaded(own_noise=0.5), water=Unsuppressed(),
                              params=Params(noise=dict(sigma=0.2)))
        ringing.voxel((0.0, 0.0, 0.0), 20.0).acquire(
            seed=1, npoints=1024, dwell_s=1.0 / 2000.0).save(tmp_path / "ringing.nii.gz")
        with pytest.raises(ValueError, match="still ring"):
            CovarianceReading.from_nifti(tmp_path / "ringing.nii.gz")


#**************************************************************************************************#
#                                    Class TestReceiverReading                                     #
#**************************************************************************************************#
class TestReceiverReading:

    def test_the_noise_colour_gives_back_the_filter_that_made_it(self):
        # noise through a receiver's filter and decimation reads back that filter, to a dwell (the
        # colours of neighbouring spans lie close); white noise reads none, as a flat band does (a
        # vendor's cropped files)
        from mrsandbox.hardware.receiver import Receiver
        from mrsandbox.scan.calibrate import ReceiverReading

        rng = np.random.default_rng(0)
        for span in (3, 5, 9):
            truth = Receiver(8, span)
            n = truth.noise_points(1024)
            noise = truth.colour(rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n)))
            found = ReceiverReading.from_arrays(noise.T).receiver()
            assert found.oversampling == 8 and abs(found.span_dwells - span) <= 1
        white = rng.standard_normal((1024, 64)) + 1j * rng.standard_normal((1024, 64))
        assert ReceiverReading.from_arrays(white).receiver() is None


#**************************************************************************************************#
#                                    Class TestSpectrumReading                                     #
#**************************************************************************************************#
@pytest.mark.render
class TestSpectrumReading:

    def test_linewidth_and_creatine_read_the_same_off_data_and_twin(self):
        # a virtual two-channel acquisition of one subject, the carrier on water: what the
        # reader finds in the raw data is what the twin shows for that subject, and a subject
        # with half the creatine reads half the ratio
        calm = Params(spurious=dict(amplitude=0.0),
                      acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                                       eddy_transient_rad=0.0), oov=dict(fraction=0.0))
        pieces = (Brain.load("icbm152"), Scanner.preset("generic_3t"),
                  Press(te1_s=0.011, te2_s=0.024, tr_s=2.0))
        voxel = VirtualScan(*pieces, params=calm).voxel(position_mm=(0.0, -45.0, 35.0),
                                                        size_mm=30.0)
        n, dwell_s = 2048, 1.0 / 2000.0

        def raw(subject):
            _, fid = subject._clean_fid(subject.sample(0), n, dwell_s)
            fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * subject._f0_mhz() * np.arange(n)
                                           * dwell_s)
            rng = np.random.default_rng(4)
            noise = 0.02 * np.abs(fid).max() * (rng.standard_normal((n, 2, 16))
                                                + 1j * rng.standard_normal((n, 2, 16)))
            return np.array([0.8, 0.6])[None, :, None] * fid[:, None, None] + noise

        reading = SpectrumReading.from_arrays(raw(voxel), dwell_s, voxel._f0_mhz())
        twin = reading.twin(voxel, seeds=1)
        assert reading.coils == 2 and reading.transients == 16
        assert abs(reading.naa_fwhm_hz / twin["naa_fwhm_hz"] - 1.0) < 0.05
        assert abs(reading.creatine_over_naa / twin["creatine_over_naa"] - 1.0) < 0.03
        truth = voxel.sample(0)
        drawn = float(voxel.resolve(truth.conc.Cr)) + float(voxel.resolve(truth.conc.PCr))
        assert abs(reading.total_creatine_mm(voxel, seeds=1) / drawn - 1.0) < 0.03
        lean = VirtualScan(*pieces, params=calm | Params(
            conc=dict(Cr=0.5 * float(truth.conc.Cr), PCr=0.5 * float(truth.conc.PCr))))
        halved = SpectrumReading.from_arrays(raw(lean.voxel(position_mm=(0.0, -45.0, 35.0),
                                                            size_mm=30.0)),
                                             dwell_s, voxel._f0_mhz())
        assert 0.45 < halved.creatine_over_naa / reading.creatine_over_naa < 0.65
        # the transients registered first: a run drifting 6 Hz with a phase jitter reads the
        # line and the noise of the still one, the run's settings playing the drift
        still = raw(voxel)
        k = np.arange(still.shape[2])
        turn = np.exp(2j * np.pi * np.outer(np.arange(n) * dwell_s, 6.0 * k / k[-1])
                      + 1j * np.radians(20.0) * np.sin(k)[None, :])
        drifting = still * turn[:, None, :]
        moved = SpectrumReading.from_arrays(drifting, dwell_s, voxel._f0_mhz())
        assert moved.naa_fwhm_hz == pytest.approx(reading.naa_fwhm_hz, abs=0.2)
        snr = [NoiseReading.from_arrays(d, dwell_s, voxel._f0_mhz()).snr for d in (still, drifting)]
        assert snr[1] == pytest.approx(snr[0], rel=0.05)


#**************************************************************************************************#
#                                   Class TestRelaxationReading                                    #
#**************************************************************************************************#
@pytest.mark.render
class TestRelaxationReading:

    def test_a_phantoms_echo_time_series_gives_back_its_t2(self):
        from mrsandbox import NoWater, Phantom

        t2 = {"NAA": 0.53, "Cr": 0.31, "Cho": 0.20}
        phantom = Phantom.braino(temperature_c=37.0, t2_s=t2)
        n, dwell_s = 4096, 1.0 / 2000.0
        voxels = {te: VirtualScan(phantom, Scanner.preset("generic_3t"),
                                  Press(te1_s=0.012, te2_s=te - 0.012, tr_s=10.0),
                                  water=NoWater()).voxel((0.0, 0.0, 0.0), 20.0)
                  for te in (0.03, 0.06, 0.12, 0.2, 0.3)}
        # the line's width is drawn, and how much a neighbour overlaps a singlet with it: five
        # draws, the median of each reading
        readings = []
        for seed in range(5):
            fids = {}
            for te, voxel in voxels.items():
                _, fid = voxel._clean_fid(voxel.sample(seed), n, dwell_s)
                # a scanner's carrier sits on water; the package's FIDs are referenced to 0 ppm
                fids[te] = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz()
                                                    * np.arange(n) * dwell_s)
            readings.append(RelaxationReading.from_series(fids, dwell_s, voxel._f0_mhz()))
        reading = readings[0]
        assert reading.te_s == (0.03, 0.06, 0.12, 0.2, 0.3) and reading.heights["NAA"][0] == 1.0
        read = {name: float(np.median([r.t2_s[name] for r in readings])) for name in t2}
        # what lies under a singlet swings with the echo time: creatine stands clear, glutamate
        # lies under NAA, myo-inositol under choline, whose reading is a bound
        assert read["Cr"] == pytest.approx(t2["Cr"], rel=0.04)
        assert read["NAA"] == pytest.approx(t2["NAA"], rel=0.08)
        assert 0.8 * t2["Cho"] < read["Cho"] < t2["Cho"]
