####################################################################################################
#                                         test_profile.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gates of scan.profile: a profile's own records outrank the catalogue in the priors and  #
#          the headers, and a profile read off a virtual scanner's raw file replicates it.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np
import pytest

from mrsandbox import Brain, CoilArray, Params, Press, Scanner, ScannerProfile
from mrsandbox.scan.calibrations import NOISE_COWS_PRISMA_3T
from mrsandbox.scan.voxel import noise_scale

PLACE, SIZE = (0.0, -45.0, 35.0), 30.0
CALM = Params(spurious=dict(amplitude=0.0), acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))


#**************************************************************************************************#
#                                     Class TestScannerProfile                                     #
#**************************************************************************************************#
@pytest.mark.render
class TestScannerProfile:

    def test_a_preset_is_served_by_the_catalogue(self):
        profile = ScannerProfile.preset("siemens_prisma_3t")
        assert profile.calibrations()["noise.sigma"] is NOISE_COWS_PRISMA_3T
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (scan.params.noise.sigma.low, scan.params.noise.sigma.high) == NOISE_COWS_PRISMA_3T.prior

    def test_a_coil_with_its_own_record_is_a_preset_of_its_own(self):
        eight, many = (ScannerProfile.preset(name).calibrations()["noise.sigma"]
                       for name in ("ge_mr750_3t", "ge_mr750_32ch_3t"))
        assert eight.coil["channels"] == (8, 8) and many.coil["channels"] == (32,)
        assert eight.value == 7.0 and many.value == 7.9

    def test_a_vendors_receiver_delay_stays_with_its_presets(self):
        # the first-order phase read off Big GABA's Trio and GE runs reaches those presets' trees as a prior on
        # the receiver delay (negative: the first sample before the echo top); a scanner without a record samples on
        # the top, since a vendor's record is no fallback and the delay is not pooled (a generic scanner is clean)
        trio = ScannerProfile(Scanner.preset("siemens_trio_3t")).scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (trio.params.acquisition.delay_s.low, trio.params.acquisition.delay_s.high) == (-0.0004845, -0.0002331)          # beyond the sequence's own creatine phase (item 230)
        prisma = ScannerProfile.preset("siemens_prisma_3t").scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (prisma.params.acquisition.delay_s.low, prisma.params.acquisition.delay_s.high) == (-0.0005563, -0.0001537)
        assert (prisma.params.water.adjusted_residual.low, prisma.params.water.adjusted_residual.high) == (0.00029, 0.00153)
        pooled = ScannerProfile.preset("generic_3t").scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert pooled.params.acquisition.delay_s.value == 0.0
        assert "acquisition.delay_s" not in ScannerProfile.preset("generic_3t").calibrations()
        assert (pooled.params.water.adjusted_residual.low, pooled.params.water.adjusted_residual.high) == (0.00075, 0.029)
        assert (pooled.params.noise.sigma.low, pooled.params.noise.sigma.high) == (4.9, 10.3)
        assert ScannerProfile.preset("philips_mr7700_3t").calibrations()["noise.sigma"].scanner["vendor"] == "pooled"    # the pool serves it
        from mrsandbox.scan.calibrations import calibrations
        plain = ScannerProfile(Scanner.preset("generic_1p5t")).scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert calibrations("generic_1p5t")["noise.sigma"].departures(plain)["field_t"] == (3.0, 1.5)      # the pool, its field named

    def test_own_records_outrank_the_catalogue_in_priors_and_headers(self):
        own = dataclasses.replace(NOISE_COWS_PRISMA_3T, value=8.0, prior=(7.0, 9.0), presets=(),
                                  source={"dataset": "my scanner", "citation": None, "url": None})
        profile = ScannerProfile(Scanner.preset("siemens_prisma_3t"), records=(own,))
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (scan.params.noise.sigma.low, scan.params.noise.sigma.high) == (7.0, 9.0)
        assert set(profile.calibrations()) == {"noise.sigma", "conc.Cr", "water.adjusted_residual", "acquisition.delay_s"}
        data = scan.voxel(PLACE, SIZE).acquire(seed=0, npoints=256)
        named = data.hdr_ext["SimulationPhantom"]["calibrations"]["noise.sigma"]
        assert named["source"] == "my scanner" and named["value"] == 8.0

    def test_a_profile_read_off_a_raw_file_replicates_the_scanner(self, tmp_path):
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        # a virtual scanner of known noise and array: 32 loaded channels, 48 raw transients
        sigma, share, n, dwell_s = 7.0, 0.4, 2048, 1.0 / 2000.0
        truth = ScannerProfile(Scanner.preset("generic_3t"),
                               coils=CoilArray.helmet().loaded(own_noise=share).calibrated(PLACE, SIZE))
        voxel = truth.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0),
                           params=CALM | Params(lineshape=dict(gauss_hz=7.0))).voxel(PLACE, SIZE)
        _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
        fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n) * dwell_s)
        rng = np.random.default_rng(1)
        noise = np.stack([truth.coils.noise(rng, sigma * noise_scale(voxel.volume_ml, 3.0, dwell_s), n)
                          for _ in range(48)], axis=-1)
        data = truth.coils.sensitivities(PLACE, SIZE)[None, :, None] * fid[:, None, None] + np.moveaxis(noise, 0, 1)
        hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
        hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_dim_info(1, "DIM_DYN")
        raw = tmp_path / "raw.nii.gz"
        gen_nifti_mrs_hdr_ext(data[None, None, None], dwell_s, hdr).save(raw)

        # its twin knows the sequence and the place, not the noise or the array's share
        twin = ScannerProfile.preset("generic_3t").scan(
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0), params=CALM).voxel(PLACE, SIZE, compiled=True)
        profile = ScannerProfile.read(raw, twin, "a virtual scanner", array=CoilArray.helmet())
        records = profile.calibrations()
        assert abs(records["noise.sigma"].value / sigma - 1.0) < 0.1
        assert abs(records["coils.loaded.own_noise"].value - share) < 0.05
        assert records["noise.sigma"].source["dataset"] == "a virtual scanner"
        # the shim too, and a scan of the profile draws its lines that wide
        assert abs(records["lineshape.gauss_hz"].value / 7.0 - 1.0) < 0.15
        shim = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024)).params.lineshape.gauss_hz
        assert (shim.low, shim.high) == records["lineshape.gauss_hz"].prior
        assert len(profile.coils) == 32 and abs(profile.coils.noise_gain(PLACE, SIZE) - 1.0) < 1e-6
        assert records["conc.Cr"].carried_by == "subject"

        # saved and loaded, it is the same scanner: records, array, and the scan's header
        profile.save(tmp_path / "profile.json")
        loaded = ScannerProfile.load(tmp_path / "profile.json")
        assert loaded.scanner == profile.scanner and loaded.calibrations().keys() == records.keys()
        assert loaded.calibrations()["noise.sigma"].value == records["noise.sigma"].value
        assert np.allclose(loaded.coils.noise_resistance, profile.coils.noise_resistance)
        assert np.allclose(loaded.coils.sensitivities(PLACE), profile.coils.sensitivities(PLACE))
        header = loaded.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024)).voxel(PLACE, SIZE).acquire(
            seed=0, npoints=256).hdr_ext["SimulationPhantom"]["calibrations"]
        assert header["noise.sigma"]["source"] == "a virtual scanner"

    def test_a_vendors_receiver_colours_the_noise_as_its_raw_data_do(self):
        # GE's noise rolls off to half power at the band's edges (Big GABA, the five dwell filter), Siemens' stays
        # flat since its files are cropped from twice the band: the GE presets carry a receiver, the Siemens none
        from mrsandbox import NoDownfield, NoLipids, NoMM, NoWater, Params

        assert ScannerProfile.preset("siemens_prisma_3t").receiver is None
        profile = ScannerProfile.preset("ge_mr750_3t")
        assert profile.receiver.span_dwells == 5
        quiet = Params(conc={name: 0.0 for name in ScannerProfile.preset("ge_mr750_3t").scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024)).params.conc})
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024), params=quiet, water=NoWater(), lipids=NoLipids(),
                            mm=NoMM(), downfield=NoDownfield())
        voxel = scan.voxel(position_mm=(0.0, -45.0, 35.0), size_mm=20.0)
        power = np.zeros(2048)
        for seed in range(8):
            _, fid = voxel.fid(seed=seed, npoints=2048)
            power += np.abs(np.fft.fftshift(np.fft.fft(fid))) ** 2
        edge = 0.5 * (power[:102].mean() + power[-102:].mean()) / power[1024 - 102: 1024 + 102].mean()
        assert 0.3 < edge < 0.6, edge

    def test_the_residual_water_and_the_receiver_delay_are_read_off_a_scan_and_its_reference(self, tmp_path):
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext
        from mrsandbox import Unsuppressed

        # a virtual scanner whose suppression leaves one percent of the water, upright, and whose receiver samples 400 us
        # before the echo top: a suppressed scan and its water reference, one coil, eight transients, fully relaxed,
        # Lorentzian lines (a Gaussian decay reads a quarter higher through the exponentials, on real data as here)
        n, dwell_s = 2048, 1.0 / 2000.0
        params = CALM | Params(water=dict(residual_fraction=0.01), acquisition=dict(delay_s=-0.0004), lineshape=dict(gauss_hz=0.0))
        press = Press(te1_s=0.011, te2_s=0.024)
        files = {}
        for name, water in (("scan", None), ("reference", Unsuppressed())):
            scan = ScannerProfile.preset("generic_3t").scan(Brain.load("icbm152"), press, params=params, **({} if water is None else {"water": water}))
            voxel = scan.voxel(PLACE, SIZE)
            _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
            fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n) * dwell_s)
            noise = 0.02 * np.abs(fid).max() * (np.random.default_rng(2).normal(size=(n, 1, 8)) + 1j * np.random.default_rng(3).normal(size=(n, 1, 8)))
            data = fid[:, None, None] + noise
            hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
            hdr.set_dim_info(0, "DIM_COIL")
            hdr.set_dim_info(1, "DIM_DYN")
            files[name] = tmp_path / f"{name}.nii.gz"
            gen_nifti_mrs_hdr_ext(data[None, None, None], dwell_s, hdr).save(files[name])
        twin = ScannerProfile.preset("generic_3t").scan(Brain.load("icbm152"), press, params=CALM).voxel(PLACE, SIZE, compiled=True)
        profile = ScannerProfile.read(files["scan"], twin, "a virtual scanner", reference=files["reference"])
        records = profile.calibrations()
        assert records["water.adjusted_residual"].value == pytest.approx(0.01, rel=0.15)          # upright, one percent
        # beyond PRESS's own phase on creatine and the residual's wing; one acquisition reads its delay to about a third
        # (three degrees of noise on creatine's phase and the twin's draws are 60 to 100 us each)
        assert records["acquisition.delay_s"].value == pytest.approx(-0.0004, rel=0.4)
        assert not records["acquisition.delay_s"].fallback
        intrinsic = records["acquisition.delay_s"].measured["intrinsic_phase_deg"]          # the twin's own creatine phase, taken out
        assert intrinsic != 0.0 and -30.0 < intrinsic < 30.0
        # the reading's twin carries both in its tree, as the console's does after a calibration
        twin = profile.scan(Brain.load("icbm152"), press, params=CALM).voxel(PLACE, SIZE)
        assert twin.scan.params["acquisition"]["delay_s"].low < -0.0003

