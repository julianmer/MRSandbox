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

from mrsandbox import Brain, CoilArray, Params, Press, Scanner, ScannerProfile, VirtualScan
from mrsandbox.scan import cohort as cohorts
from mrsandbox.scan.calibrate import WaterReading, _raw
from mrsandbox.scan.calibrations import calibrations
from mrsandbox.scan.voxel import noise_scale

PLACE, SIZE = (0.0, -45.0, 35.0), 30.0
# the leaked water off: these gates set a residual and read it back, and read the noise beside the
# water's wings
CALM = Params(spurious=dict(amplitude=0.0),
              acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0, delay_s=0.0,
                               eddy_transient_rad=0.0), oov=dict(fraction=0.0))


#**************************************************************************************************#
#                                     Class TestScannerProfile                                     #
#**************************************************************************************************#
@pytest.mark.render
class TestScannerProfile:

    def test_a_preset_is_served_by_its_pooled_rows(self):
        profile = ScannerProfile.preset("siemens_prisma_3t")
        served = profile.calibrations()["noise.sigma"]
        assert served == cohorts.pooled("noise.sigma", Scanner.preset("siemens_prisma_3t"))
        assert (served.measured["matched_on"]["model"] == "Prisma"
                and served.source["dataset"].startswith("pooled over"))
        assert "cows_prisma" in served.source["dataset"] and served.scanner["vendor"] == "siemens"
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (round(scan.params.noise.sigma.low, 7),
                round(scan.params.noise.sigma.high, 7)) == served.prior

    def test_the_receive_arrays_count_is_one_of_the_conditions(self):
        # the MR750 was read with an 8-channel array (Big GABA) and a 32-channel one (ds006444):
        # the array on the floor picks the rows, one ideal channel pools both
        eight, many = (calibrations("ge_mr750_3t", channels=n)["noise.sigma"] for n in (8, 32))
        assert eight.coil["channels"] == (8,) and many.coil["channels"] == (32,)
        assert ("big_gaba_ge" in eight.source["dataset"]
                and "ds006444_ge_mr750" in many.source["dataset"])
        both = calibrations("ge_mr750_3t")["noise.sigma"]
        assert both.measured["subjects"] == eight.measured["subjects"] + many.measured["subjects"]
        scan = VirtualScan(Brain.load("icbm152"), Scanner.preset("ge_mr750_3t"),
                           Press(te1_s=0.011, te2_s=0.024), coils=CoilArray.ring(8))
        assert (round(scan.params.noise.sigma.low, 7),
                round(scan.params.noise.sigma.high, 7)) == eight.prior

    def test_the_pool_is_as_specific_as_the_scanner_on_the_floor(self):
        # the receiver delay: the Trio's own rows for the Trio, every 3 T system for the generic
        # 3 T, the vendor's for a Philips system the package has no rows of, and everything for a
        # field it has none at, the departure named
        trio = calibrations("siemens_trio_3t")["acquisition.delay_s"]
        assert trio.measured["matched_on"]["model"] == "TrioTim" and trio.value < 0.0
        scan = ScannerProfile(Scanner.preset("siemens_trio_3t")).scan(
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (round(scan.params.acquisition.delay_s.low, 7),
                round(scan.params.acquisition.delay_s.high, 7)) == trio.prior
        generic = calibrations("generic_3t")["acquisition.delay_s"]
        assert (generic.measured["matched_on"] == {"field_t": 3.0}
                and generic.measured["subjects"] > trio.measured["subjects"])
        pooled = ScannerProfile.preset("generic_3t").scan(Brain.load("icbm152"),
                                                          Press(te1_s=0.011, te2_s=0.024))
        assert (round(pooled.params.acquisition.delay_s.low, 7),
                round(pooled.params.acquisition.delay_s.high, 7)) == generic.prior
        assert calibrations("philips_mr7700_3t")["noise.sigma"].measured["matched_on"] == {
            "field_t": 3.0, "vendor": "philips"}
        low = calibrations("generic_1p5t")["noise.sigma"]
        plain = ScannerProfile(Scanner.preset("generic_1p5t")).scan(
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (low.measured["matched_on"] == {}
                and low.departures(plain)["field_t"] == (low.scanner["field_t"], 1.5))

    def test_one_cohorts_rows_alone_calibrate_a_scanner(self):
        # the scanner is hardware; what calibrates it is chosen apart: the pool, or one dataset's
        # rows
        trio = ScannerProfile.preset("siemens_trio_3t", cohort="big_gaba_trio")
        held = {r.leaf for c in cohorts.package() if c.name == "big_gaba_trio"
                for m in c.members for r in m.profile.records if not r.failed}
        # the Trio's own leaves; the fMRI cohort's are not among them
        assert {record.leaf for record in trio.records} == held < set(cohorts.leaves())
        assert all(record.source["dataset"].endswith(": big_gaba_trio") for record in trio.records)
        assert trio.calibrations()["noise.sigma"] == cohorts.pooled(
            "noise.sigma", Scanner.preset("siemens_trio_3t"), within=("big_gaba_trio",))
        assert "big_gaba_trio" in cohorts.holding(Scanner.preset("siemens_trio_3t"))
        assert cohorts.holding(Scanner("unnamed", 11.7)) == []  # nothing at the field
        assert set(cohorts.holding(Scanner.preset("generic_3t"))) >= {
            "big_gaba_trio", "big_gaba_ge", "big_gaba_philips"}

    def test_own_records_outrank_the_catalogue_in_priors_and_headers(self):
        own = dataclasses.replace(calibrations("siemens_prisma_3t")["noise.sigma"], value=8.0,
                                  prior=(7.0, 9.0),
                                  source={"dataset": "my scanner", "citation": None, "url": None})
        profile = ScannerProfile(Scanner.preset("siemens_prisma_3t"), records=(own,))
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (scan.params.noise.sigma.low, scan.params.noise.sigma.high) == (7.0, 9.0)
        transfer = {f"mt.rate.{name}" for name in ("Cr", "PCr", "NAA", "NAAG", "PCho", "GPC")}
        assert set(profile.calibrations()) == {"conc.Cr", *transfer, *cohorts.leaves()}
        data = scan.voxel(PLACE, SIZE).acquire(seed=0, npoints=256)
        named = data.hdr_ext["SimulationPhantom"]["calibrations"]["noise.sigma"]
        assert named["source"] == "my scanner" and named["value"] == 8.0

    def test_a_profile_read_off_a_raw_file_replicates_the_scanner(self, tmp_path):
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        # a virtual scanner of known noise and array: 32 loaded channels, 48 raw transients
        sigma, share, n, dwell_s = 7.0, 0.4, 2048, 1.0 / 2000.0
        truth = ScannerProfile(
            Scanner.preset("generic_3t"),
            coils=CoilArray.helmet().loaded(own_noise=share).calibrated(PLACE, SIZE))
        voxel = truth.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0),
                           params=CALM | Params(lineshape=dict(gauss_hz=7.0))).voxel(PLACE, SIZE)
        _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
        fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n)
                                       * dwell_s)
        rng = np.random.default_rng(1)
        noise = np.stack([
            truth.coils.noise(rng, sigma * noise_scale(voxel.volume_ml, 3.0, dwell_s), n)
            for _ in range(48)], axis=-1)
        data = (truth.coils.sensitivities(PLACE, SIZE)[None, :, None] * fid[:, None, None]
                + np.moveaxis(noise, 0, 1))
        hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
        hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_dim_info(1, "DIM_DYN")
        raw = tmp_path / "raw.nii.gz"
        gen_nifti_mrs_hdr_ext(data[None, None, None], dwell_s, hdr).save(raw)

        # its twin knows the sequence and the place, not the noise or the array's share
        twin = ScannerProfile.preset("generic_3t").scan(
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0),
            params=CALM).voxel(PLACE, SIZE, compiled=True)
        profile = ScannerProfile.read(raw, twin, "a virtual scanner", array=CoilArray.helmet())
        records = profile.calibrations()
        assert abs(records["noise.sigma"].value / sigma - 1.0) < 0.1
        assert abs(records["coils.loaded.own_noise"].value - share) < 0.05
        assert records["noise.sigma"].source["dataset"] == "a virtual scanner"
        # the shim too, and a scan of the profile draws its lines that wide
        assert abs(records["lineshape.gauss_hz"].value / 7.0 - 1.0) < 0.15
        shim = profile.scan(Brain.load("icbm152"),
                            Press(te1_s=0.011, te2_s=0.024)).params.lineshape.gauss_hz
        assert (shim.low, shim.high) == records["lineshape.gauss_hz"].prior
        # the run's tracks: the scanner's drift and jitters, not the person's breathing or moves
        assert {"run.drift_hz_per_min", "run.phase_jitter_deg"} <= set(records)
        assert not {"run.resp_amp_hz", "run.events_per_min"} & {r.leaf for r in profile.records}
        # read in vitro (a phantom on the table), the receiver's own noise samples given: the
        # hardware alone; no noise level (a phantom's is but a head's lower bound) and no width
        samples = truth.coils.noise(rng, 1.0, 4096).T
        vitro = ScannerProfile.read(raw, twin, "a phantom scan", load="a phantom", in_vitro=True,
                                    noise=samples)
        assert {r.leaf for r in vitro.records} == {
            "coils.partner_correlation", "run.drift_hz_per_min", "run.ar_sigma_hz",
            "run.phase_jitter_deg", "run.amp_jitter_frac"}
        assert len(profile.coils) == 32 and abs(profile.coils.noise_gain(PLACE, SIZE) - 1.0) < 1e-6
        assert records["conc.Cr"].carried_by == "subject"

        # saved and loaded, it is the same scanner: records, array, and the scan's header
        profile.save(tmp_path / "profile.json")
        loaded = ScannerProfile.load(tmp_path / "profile.json")
        assert loaded.scanner == profile.scanner and loaded.calibrations().keys() == records.keys()
        assert loaded.calibrations()["noise.sigma"].value == records["noise.sigma"].value
        assert np.allclose(loaded.coils.noise_resistance, profile.coils.noise_resistance)
        assert np.allclose(loaded.coils.sensitivities(PLACE), profile.coils.sensitivities(PLACE))
        header = loaded.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024)).voxel(
            PLACE, SIZE).acquire(seed=0, npoints=256).hdr_ext["SimulationPhantom"]["calibrations"]
        assert header["noise.sigma"]["source"] == "a virtual scanner"

    def test_a_vendors_receiver_colours_the_noise_as_its_raw_data_do(self):
        # GE's noise rolls off to half power at the band's edges (Big GABA, the four dwell
        # filter), Siemens' stays flat since its files are cropped from twice the band: the GE
        # presets carry a receiver, the Siemens none
        from mrsandbox import NoDownfield, NoLipids, NoMM, NoWater, Params

        assert ScannerProfile.preset("siemens_prisma_3t").receiver is None
        profile = ScannerProfile.preset("ge_mr750_3t")
        assert profile.receiver.span_dwells == 4
        quiet = Params(conc={name: 0.0 for name in ScannerProfile.preset("ge_mr750_3t").scan(
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024)).params.conc})
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024), params=quiet,
                            water=NoWater(), lipids=NoLipids(),
                            mm=NoMM(), downfield=NoDownfield())
        voxel = scan.voxel(position_mm=(0.0, -45.0, 35.0), size_mm=20.0)
        power = np.zeros(2048)
        for seed in range(8):
            _, fid = voxel.fid(seed=seed, npoints=2048)
            power += np.abs(np.fft.fftshift(np.fft.fft(fid))) ** 2
        edge = (0.5 * (power[:102].mean() + power[-102:].mean())
                / power[1024 - 102: 1024 + 102].mean())
        assert 0.3 < edge < 0.6, edge

    def test_the_residual_water_and_the_receiver_delay_are_read_off_a_scan_and_its_reference(
            self, tmp_path):
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext
        from mrsandbox import Unsuppressed

        # a virtual scanner whose suppression leaves one percent of the water, upright, and whose
        # receiver samples 400 us before the echo top: a suppressed scan and its water reference,
        # one coil, eight transients, fully relaxed, Lorentzian lines (a Gaussian decay reads a
        # quarter higher through the exponentials, on real data as here)
        n, dwell_s = 2048, 1.0 / 2000.0
        params = CALM | Params(water=dict(residual_fraction=0.01),
                               acquisition=dict(delay_s=-0.0004), lineshape=dict(gauss_hz=0.0))
        press = Press(te1_s=0.011, te2_s=0.024)
        files, arrays = {}, {}
        for name, water in (("scan", None), ("reference", Unsuppressed())):
            scan = ScannerProfile.preset("generic_3t").scan(
                Brain.load("icbm152"), press, params=params,
                **({} if water is None else {"water": water}))
            voxel = scan.voxel(PLACE, SIZE)
            _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
            fid = np.asarray(fid) * np.exp(-2j * np.pi * 4.65 * voxel._f0_mhz() * np.arange(n)
                                           * dwell_s)
            noise = 0.02 * np.abs(fid).max() * (
                np.random.default_rng(2).normal(size=(n, 1, 8))
                + 1j * np.random.default_rng(3).normal(size=(n, 1, 8)))
            arrays[name] = fid[:, None, None] + noise
        # the same run with two of its water's transients interleaved (ds005917 stores one every
        # 16th)
        arrays["mixed"] = np.concatenate([arrays["scan"][..., :4], arrays["reference"][..., :1],
                                          arrays["scan"][..., 4:], arrays["reference"][..., 1:2]],
                                         axis=2)
        for name, data in arrays.items():
            hdr = Hdr_Ext(voxel._f0_mhz(), "1H")
            hdr.set_dim_info(0, "DIM_COIL")
            hdr.set_dim_info(1, "DIM_DYN")
            files[name] = tmp_path / f"{name}.nii.gz"
            gen_nifti_mrs_hdr_ext(data[None, None, None], dwell_s, hdr).save(files[name])
        twin = ScannerProfile.preset("generic_3t").scan(
            Brain.load("icbm152"), press, params=CALM).voxel(PLACE, SIZE, compiled=True)
        profile = ScannerProfile.read(files["scan"], twin, "a virtual scanner",
                                      reference=files["reference"])
        records = profile.calibrations()
        # upright, one percent
        assert records["water.adjusted_residual"].value == pytest.approx(0.01, rel=0.15)
        # the leaked water and the eddy transient are read too: none in this synthetic scan (CALM
        # has the leak off)
        assert (records["oov.fraction"].value < 0.002
                and abs(records["acquisition.eddy_transient_rad"].value) < 0.05)
        assert not records["acquisition.eddy_transient_rad"].fallback
        # beyond PRESS's own phase on creatine and the residual's wing; one acquisition reads its
        # delay to about a third (three degrees of noise on creatine's phase and the twin's draws
        # are 60 to 100 us each)
        assert records["acquisition.delay_s"].value == pytest.approx(-0.0004, rel=0.4)
        assert not records["acquisition.delay_s"].fallback
        # the twin's own creatine phase, taken out
        intrinsic = records["acquisition.delay_s"].measured["intrinsic_phase_deg"]
        assert intrinsic != 0.0 and -30.0 < intrinsic < 30.0
        assert "quarter turn" not in records["acquisition.delay_s"].failed
        # creatine nearly inverted against its twin reads no delay: the rule fails it
        inverted = dataclasses.replace(WaterReading.from_nifti(files["scan"], files["reference"]),
                                       creatine_phase_deg=intrinsic + 160.0)
        assert "quarter turn" in inverted.record_delay(twin, "inverted").failed
        # the interleaved run is parted on loading: the run alone, and its water as its own
        # reference, read as the two files
        assert (_raw(files["mixed"])[0].shape[2] == 8
                and _raw(files["mixed"], water=True)[0].shape[2] == 2)
        apart, together = (WaterReading.from_nifti(files["scan"], files["reference"]),
                           WaterReading.from_nifti(files["mixed"]))
        assert together.residual_over_full == pytest.approx(apart.residual_over_full, rel=0.05)
        assert together.creatine_phase_deg == pytest.approx(apart.creatine_phase_deg, abs=0.5)
        with pytest.raises(ValueError, match="interleaves no water"):
            WaterReading.from_nifti(files["scan"])
        # the reading's twin carries both in its tree, as the console's does after a calibration
        # (nothing set on them here)
        twin = profile.scan(Brain.load("icbm152"), press,
                            params=Params(spurious=dict(amplitude=0.0))).voxel(PLACE, SIZE)
        assert twin.scan.params["acquisition"]["delay_s"].low < -0.0003
