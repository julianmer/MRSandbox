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

    def test_own_records_outrank_the_catalogue_in_priors_and_headers(self):
        own = dataclasses.replace(NOISE_COWS_PRISMA_3T, value=8.0, prior=(7.0, 9.0), presets=(),
                                  source={"dataset": "my scanner", "citation": None, "url": None})
        profile = ScannerProfile(Scanner.preset("siemens_prisma_3t"), records=(own,))
        scan = profile.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024))
        assert (scan.params.noise.sigma.low, scan.params.noise.sigma.high) == (7.0, 9.0)
        assert set(profile.calibrations()) == {"noise.sigma", "conc.Cr"}
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
        voxel = truth.scan(Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0), params=CALM).voxel(PLACE, SIZE)
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
            Brain.load("icbm152"), Press(te1_s=0.011, te2_s=0.024, tr_s=2.0), params=CALM).voxel(PLACE, SIZE)
        profile = ScannerProfile.read(raw, twin, "a virtual scanner", array=CoilArray.helmet())
        records = profile.calibrations()
        assert abs(records["noise.sigma"].value / sigma - 1.0) < 0.1
        assert abs(records["coils.loaded.own_noise"].value - share) < 0.05
        assert records["noise.sigma"].source["dataset"] == "a virtual scanner"
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
