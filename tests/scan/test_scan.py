####################################################################################################
#                                           test_scan.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Scan gate: the north-star script works end to end. Defaults derive from the database    #
#          (including the Cho interval built from GPC plus PCho), acquisitions are seeded and      #
#          deterministic, the NIfTI-MRS header carries the full provenance, and knocking a         #
#          metabolite out of the params removes its peak.                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import Brain, Fixed, Normal, Params, Press, Scanner, Uniform, VirtualScan

# full-scan integration renders: the tensorflow CI leg skips these (-m 'not render')
pytestmark = pytest.mark.render


#*****************#
#   shared scan   #
#*****************#
def typed_truth(voxel, seed):
    """The recorded truth of an acquisition as the header carries it (a JSON round trip)."""
    import json

    return json.loads(json.dumps(voxel.acquire(seed=seed, npoints=256)
                                 .hdr_ext["SimulationGroundTruth"]["params"]))


@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))


#**************************************************************************************************#
#                                      Class TestVirtualScan                                       #
#**************************************************************************************************#
class TestVirtualScan:

    #**************#
    #   defaults   #
    #**************#
    def test_default_params_derive_from_the_database(self, scan):
        assert len(scan.params.conc) == 21
        assert scan.params.conc.NAA == Uniform(7.5, 12.0)
        assert scan.params.conc.Glu == Uniform(6.0, 12.5)
        assert "Cho" not in scan.params.conc  # covered by GPC and PCho
        assert isinstance(scan.params.t2.NAA, Normal)  # pooled from Gudmundson
        assert 0.2 < scan.params.t2.NAA.mean < 0.35
        assert scan.params.t2.Ser == Fixed(0.160)  # no rows: the stated fallback
        assert isinstance(scan.params.t1.NAA, Normal)  # pooled from the typed literature
        assert 1.3 < scan.params.t1.NAA.mean < 1.5
        assert scan.params.t1.Ser == Fixed(1.300)  # no rows: the stated fallback
        assert scan.params.b1_scale == Normal(1.0, 0.03)  # transmit scatter, stated

    #*****************#
    #   determinism   #
    #*****************#
    def test_acquisitions_are_seeded(self, scan):
        voxel = scan.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        _, fid_a = voxel.fid(seed=7)
        _, fid_b = voxel.fid(seed=7)
        _, fid_c = voxel.fid(seed=8)
        np.testing.assert_array_equal(ops.convert_to_numpy(fid_a), ops.convert_to_numpy(fid_b))
        assert np.abs(ops.convert_to_numpy(fid_a) - ops.convert_to_numpy(fid_c)).max() > 0.0

    #*******************#
    #   isochromats     #
    #*******************#
    def test_isochromats_reach_the_engine(self, scan):
        assert scan.engine.isochromats == 32  # the stated default
        eight = VirtualScan(scan.brain, scan.scanner, scan.sequence, isochromats=8)
        assert eight.engine.isochromats == 8

    #***********************#
    #   serializable seed   #
    #***********************#
    def test_acquire_records_a_regenerating_seed(self, scan):
        # seed None must still yield a file that regenerates itself: a fresh seed is
        # drawn and recorded; a Generator cannot be recorded and is refused
        voxel = scan.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        data = voxel.acquire(seed=None, npoints=256)
        recorded = data.hdr_ext["SimulationGroundTruth"]["seed"]
        assert isinstance(recorded, int)
        again = voxel.acquire(seed=recorded, npoints=256)
        np.testing.assert_array_equal(np.asarray(data[:]), np.asarray(again[:]))
        with pytest.raises(TypeError):
            voxel.acquire(seed=np.random.default_rng(3), npoints=256)
        # one integer is one subject everywhere: the sampled truth, the quick-look
        # FID, and the acquisition of a seed agree, and the noise is the seed's own
        assert voxel.sample(seed=11).to_dict() == typed_truth(voxel, 11)
        _, fid = voxel.fid(seed=11, npoints=256)
        np.testing.assert_array_equal(fid, np.asarray(voxel.acquire(seed=11, npoints=256)[:])[0, 0, 0])
        # a numpy integer (a loop drawing seeds from its own generator) records as a
        # plain int and means the same acquisition
        typed = voxel.acquire(seed=np.int64(11), npoints=256)
        assert type(typed.hdr_ext["SimulationGroundTruth"]["seed"]) is int
        assert typed.hdr_ext["SimulationGroundTruth"]["params"] == typed_truth(voxel, 11)
        np.testing.assert_array_equal(np.asarray(typed[:]),
                                      np.asarray(voxel.acquire(seed=11, npoints=256)[:]))

    #****************#
    #   provenance   #
    #****************#
    def test_acquire_returns_nifti_mrs_with_provenance(self, scan):
        voxel = scan.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
        data = voxel.acquire(seed=7, npoints=1024, dwell_s=1.0 / 4000.0)
        assert data.shape == (1, 1, 1, 1024)
        assert abs(data.dwelltime - 1.0 / 4000.0) < 1e-12
        truth = data.hdr_ext["SimulationGroundTruth"]
        assert truth["seed"] == 7
        assert len(truth["params"]["conc"]) == 21
        assert {"NAA", "Glu", "Ins", "GABA", "Lac"} <= set(truth["params"]["conc"])
        assert 0.0 <= truth["tissue_fractions"]["gm"] <= 1.0
        assert data.hdr_ext["SimulationSequence"]["name"] == "Press"
        assert data.hdr_ext["SimulationSequence"]["te1_s"] == 0.011
        assert data.hdr_ext["SimulationPhantom"]["brain"] == "icbm152"

    #********************#
    #   gm wm contrast   #
    #********************#
    def test_concentrations_split_by_the_challenge_contrast(self, scan):
        from mrsandbox import Brain

        shape = (8, 8, 8)
        affine = np.eye(4)
        affine[:3, 3] = -3.5
        zeros = np.zeros(shape, dtype=np.float32)

        def amplitudes(gm, wm):
            brain = Brain.from_maps(gm=gm, wm=wm, csf=zeros, affine=affine)
            source = VirtualScan(brain, scan.scanner, scan.sequence)
            voxel = source.voxel(position_mm=(0.0, 0.0, 0.0), size_mm=4.0)
            truth = voxel.sample(seed=7)
            parts = voxel.components(truth)
            return truth, {name: float(np.abs(ops.convert_to_numpy(parts[i][1])).sum())
                           for i, name in enumerate(truth.conc)}

        ones = np.ones(shape, dtype=np.float32)
        truth, in_gm = amplitudes(ones, zeros)
        _, in_wm = amplitudes(zeros, ones)
        _, mixed = amplitudes(ones * 0.5, ones * 0.5)
        # pure tissues realize exactly the drawn ratio, published as 10.98/7.49 mM
        assert abs(truth.contrast.NAA - 10.98 / 7.49) < 1e-12
        assert abs(in_gm["NAA"] / in_wm["NAA"] - truth.contrast.NAA) < 1e-4
        assert abs(in_gm["NAAG"] / in_wm["NAAG"] - 1.40 / 2.60) < 1e-4  # WM-dominant
        assert abs(in_gm["Ser"] / in_wm["Ser"] - 1.0) < 1e-4  # no table row: flat
        # an equal mix keeps exactly the drawn value, the mean of the pure tissues
        for name in ("NAA", "Ser", "GABA"):
            assert abs(mixed[name] - (in_gm[name] + in_wm[name]) / 2.0) < 1e-4 * mixed[name]

    #*******************************#
    #   physics through the stack   #
    #*******************************#
    def test_knocking_out_naa_removes_its_peak(self, scan):
        # quiet the receiver-side imperfections: this gates composition, and a drawn
        # frequency shift, phase, or spurious echo would move or pollute the read bin
        quiet = Params(acquisition=dict(shift_hz=0.0, phase_deg=0.0),
                       spurious=dict(amplitude=0.0))
        clean = VirtualScan(scan.brain, scan.scanner, scan.sequence, params=quiet)
        knockout = VirtualScan(
            scan.brain, scan.scanner, scan.sequence, params=quiet | Params(conc=dict(NAA=0.0))
        )
        heights = []
        for source in (clean, knockout):
            voxel = source.voxel(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)
            ppm, spec = voxel.spectrum(seed=7, npoints=8192)
            ppm, spec = np.asarray(ppm), ops.convert_to_numpy(spec)
            heights.append(spec.real[np.argmin(np.abs(ppm - 2.008))])
        # NAA must dominate its own peak: its contribution (the difference) exceeds the
        # genuine neighbors that remain there (NAAG acetyl, Glu, Gln, GABA), as in vivo.
        assert heights[0] - heights[1] > max(heights[1], 0.0)


#**************************************************************************************************#
#                                     Class TestAcquisition                                        #
#**************************************************************************************************#
#                                                                                                  #
# The drawn acquisition imperfections modulate the FID by exactly the analytic factor.             #
#                                                                                                  #
#**************************************************************************************************#
class TestAcquisitionImperfections:

    #**********************#
    #   exact modulation   #
    #**********************#
    def test_shift_and_phase_modulate_the_fid_exactly(self, scan):
        # the frequency error also runs the sequence off resonance; with exact 180s
        # the spin echo refocuses it and the readout is a pure modulation (an
        # imperfect 180 tips T1-recovered magnetization the echo never refocuses,
        # a physical offset-dependent residue the coherence filters let through)
        brain, pieces = scan.brain, (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        clean = Params(noise=dict(sigma=0.0), b1_scale=1.0,
                       acquisition=dict(shift_hz=0.0, phase_deg=0.0))
        bent = Params(noise=dict(sigma=0.0), b1_scale=1.0,
                      acquisition=dict(shift_hz=5.0, phase_deg=90.0))
        base = VirtualScan(brain, *pieces, params=clean).voxel(position_mm=(10, -20, 5), size_mm=20)
        warp = VirtualScan(brain, *pieces, params=bent).voxel(position_mm=(10, -20, 5), size_mm=20)
        _, fid0 = base.fid(seed=7, npoints=512)
        _, fid1 = warp.fid(seed=7, npoints=512)
        t = np.arange(512) / 4000.0
        expected = fid0 * np.exp(1j * (2.0 * np.pi * 5.0 * t + np.pi / 2.0))
        assert np.abs(fid1 - expected).max() < 1e-4 * np.abs(fid0).max()

    #**********************#
    #   eddy phase         #
    #**********************#
    def test_eddy_phase_is_augmentrums_seeded_trajectory(self, scan):
        # the eddy phase is Augmentrum's synthetic trajectory: a pure phase that
        # regenerates from the drawn seed, and exactly what its module applies to
        # the clean FID
        from augmentrum.augmentation import EddyCurrent

        brain, pieces = scan.brain, (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        quiet = dict(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0))
        still = Params(**quiet, acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))

        def rung(seed, std=0.5):
            return Params(**quiet, acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=std,
                                                    eddy_lp_cut_hz=25.0, eddy_seed=float(seed)))

        def fid(params):
            voxel = VirtualScan(brain, *pieces, params=params).voxel(position_mm=(10, -20, 5),
                                                                     size_mm=20)
            return voxel.fid(seed=7, npoints=512)[1]

        clean, warped, again, other = fid(still), fid(rung(11)), fid(rung(11)), fid(rung(12))
        np.testing.assert_allclose(np.abs(warped), np.abs(clean), rtol=1e-5)  # a pure phase
        np.testing.assert_array_equal(warped, again)
        assert np.abs(warped - clean).max() > 1e-2 * np.abs(clean).max()
        assert np.abs(other - warped).max() > 1e-2 * np.abs(clean).max()
        expected = EddyCurrent(std_rad=0.5, lp_cut_hz=25.0, seed=11).process_tensor(
            clean[None, :], sw_hz=4000.0)[0][0]
        np.testing.assert_allclose(warped, expected, atol=1e-5 * np.abs(clean).max())

    #********************#
    #   spurious echo    #
    #********************#
    def test_spurious_echo_rides_at_its_drawn_position(self, scan):
        brain, pieces = scan.brain, (Scanner.preset("generic_3t"), Press(te1_s=0.011, te2_s=0.019))
        still = dict(noise=dict(sigma=0.0),
                     acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))
        ghost = dict(tau_s=0.1, width_s2=2000.0, ppm=2.5, phase_deg=0.0)
        off = Params(**still, spurious=dict(amplitude=0.0, **ghost))
        on = Params(**still, spurious=dict(amplitude=0.1, **ghost))
        base = VirtualScan(brain, *pieces, params=off).voxel(position_mm=(10, -20, 5), size_mm=20)
        warp = VirtualScan(brain, *pieces, params=on).voxel(position_mm=(10, -20, 5), size_mm=20)
        _, fid0 = base.fid(seed=7, npoints=1024)
        _, fid1 = warp.fid(seed=7, npoints=1024)
        t = np.arange(1024) / 4000.0
        f0_mhz = 42.5774785 * 3.0
        echo = 0.1 * np.abs(fid0).max() * np.exp(-2000.0 * (t - 0.1) ** 2) * np.exp(
            1j * 2.0 * np.pi * 2.5 * f0_mhz * t)
        assert np.abs((fid1 - fid0) - echo).max() < 1e-4 * np.abs(fid0).max()


#**************************************************************************************************#
#                                          Class TestMaps                                          #
#**************************************************************************************************#
#                                                                                                  #
# Map-valued leaves: the voxel resolves them over its box, and the header records the result.       #
#                                                                                                  #
#**************************************************************************************************#
class TestMaps:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def uniform_scan(scan, params, **kwargs):
        from mrsandbox import Brain

        shape = (8, 8, 8)
        affine = np.eye(4)
        affine[:3, 3] = -3.5
        ones, zeros = np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
        brain = Brain.from_maps(gm=ones, wm=zeros, csf=zeros, affine=affine)
        return VirtualScan(brain, scan.scanner, scan.sequence, params=params, **kwargs)

    @staticmethod
    def naa_amplitude(voxel, truth):
        parts = voxel.components(truth)
        index = list(truth.conc).index("NAA")
        return float(np.abs(ops.convert_to_numpy(parts[index][1])).sum())

    #*************#
    #   resolve   #
    #*************#
    def test_resolve_weighs_tissues_reads_maps_and_passes_scalars(self, scan):
        from mrsandbox import Tissues, Volume

        voxel = self.uniform_scan(scan, None).voxel(position_mm=(0.0, 0.0, 0.0), size_mm=4.0)
        assert voxel.resolve(2.5) == 2.5
        assert voxel.resolve(Tissues(gm=1.0, wm=3.0, csf=5.0)) == 1.0  # a pure GM voxel
        ramp = np.broadcast_to(np.arange(8, dtype=np.float64)[:, None, None], (8, 8, 8))
        affine = np.eye(4)
        affine[:3, 3] = -3.5
        volume = Volume(ramp.copy(), affine)
        assert abs(voxel.resolve(volume) - 3.5) < 1e-12  # the box mean of the ramp
        assert voxel.resolve(Volume(ramp.copy(), np.eye(4) * 100.0)) == 0.0  # outside

    #*******************#
    #   mapped offset   #
    #*******************#
    def test_mapped_offset_runs_the_sequence_and_shifts_the_readout(self, scan):
        # a B0 offset map resolves at the voxel for both the sequence frame and the
        # receiver modulation, so a flat 30 Hz map is the fixed 30 Hz world exactly
        from mrsandbox import Mapped, Volume

        affine = np.eye(4)
        affine[:3, 3] = -3.5
        flat = Volume(np.full((8, 8, 8), 30.0), affine, name="offset_flat")

        def fid(shift):
            params = Params(acquisition=dict(shift_hz=shift))
            voxel = self.uniform_scan(scan, params).voxel(position_mm=(0.0, 0.0, 0.0), size_mm=2.0)
            return voxel._clean_fid(voxel.sample(seed=7), 256, 1.0 / 4000.0)[1]

        np.testing.assert_allclose(fid(Mapped(flat)), fid(30.0), rtol=1e-6)

    #*************************#
    #   mapped concentration  #
    #*************************#
    def test_mapped_concentration_reads_the_box_mean(self, scan):
        from mrsandbox import Mapped, Volume

        affine = np.eye(4)
        affine[:3, 3] = -3.5
        flat = Volume(np.full((8, 8, 8), 10.0), affine, name="naa_flat")
        ramp = Volume(np.broadcast_to(np.arange(8, dtype=np.float64)[:, None, None],
                                      (8, 8, 8)).copy() * 2.0, affine, name="naa_ramp")

        def amplitude(params, position):
            voxel = self.uniform_scan(scan, params).voxel(position_mm=position, size_mm=2.0)
            return self.naa_amplitude(voxel, voxel.sample(seed=7))

        fixed = amplitude(Params(conc=dict(NAA=10.0)), (0.0, 0.0, 0.0))
        mapped = amplitude(Params(conc=dict(NAA=Mapped(flat))), (0.0, 0.0, 0.0))
        assert abs(mapped - fixed) < 1e-6 * fixed  # a flat 10 mM map is a fixed 10 mM
        assert abs(amplitude(Params(conc=dict(NAA=5.0)), (0.0, 0.0, 0.0)) - fixed / 2.0) < 1e-6 * fixed
        # the ramp reads 2 x (box mean of the index): 2 x 3.5 at the centre, 2 x 5.5 at +2
        centre = amplitude(Params(conc=dict(NAA=Mapped(ramp))), (0.0, 0.0, 0.0))
        shifted = amplitude(Params(conc=dict(NAA=Mapped(ramp))), (2.0, 0.0, 0.0))
        assert abs(centre / fixed - 0.7) < 1e-6 and abs(shifted / fixed - 1.1) < 1e-6

    #*************#
    #   header    #
    #*************#
    def test_header_records_the_resolved_local_values(self, scan):
        from mrsandbox import Mapped, Volume

        affine = np.eye(4)
        affine[:3, 3] = -3.5
        t1 = Volume(np.full((8, 8, 8), 1.3), affine, name="t1_synthetic")
        params = Params(water=dict(t1_s=Mapped(t1)), conc=dict(NAA=Mapped(t1)))
        voxel = self.uniform_scan(scan, params).voxel(position_mm=(0.0, 0.0, 0.0), size_mm=4.0)
        data = voxel.acquire(seed=7, npoints=64)
        truth = data.hdr_ext["SimulationGroundTruth"]
        assert truth["params"]["water"]["t1_s"] == "t1_synthetic"
        assert truth["local"] == {"conc.NAA": 1.3, "water.t1_s": 1.3}

    #****************#
    #   transmit     #
    #****************#
    def test_transmit_map_scales_the_pulses_like_b1_scale(self, scan):
        from mrsandbox import Volume

        affine = np.eye(4)
        affine[:3, 3] = -3.5
        # the relative B1+ ramps along x: 0.9 at the centre box (index mean 3.5),
        # 1.1 two millimetres over (index mean 5.5)
        ramp = Volume(0.55 + 0.1 * np.broadcast_to(np.arange(8, dtype=np.float64)[:, None, None],
                                                    (8, 8, 8)).copy(), affine, name="b1_ramp")
        nominal = Params(b1_scale=1.0)

        def fid(params, position, **kwargs):
            voxel = self.uniform_scan(scan, params, **kwargs).voxel(position_mm=position,
                                                                     size_mm=2.0)
            return voxel.fid(seed=7, npoints=256)[1]

        mapped_centre = fid(nominal, (0.0, 0.0, 0.0), transmit=ramp)
        mapped_over = fid(nominal, (2.0, 0.0, 0.0), transmit=ramp)
        np.testing.assert_allclose(mapped_centre, fid(Params(b1_scale=0.9), (0.0, 0.0, 0.0)),
                                   atol=1e-6 * np.abs(mapped_centre).max())
        np.testing.assert_allclose(mapped_over, fid(Params(b1_scale=1.1), (2.0, 0.0, 0.0)),
                                   atol=1e-6 * np.abs(mapped_over).max())
        # the header records the map and the scale the pulses ran at
        voxel = self.uniform_scan(scan, nominal, transmit=ramp).voxel(position_mm=(0.0, 0.0, 0.0),
                                                                      size_mm=2.0)
        data = voxel.acquire(seed=7, npoints=64)
        assert abs(data.hdr_ext["SimulationGroundTruth"]["local"]["b1_scale"] - 0.9) < 1e-12
        assert data.hdr_ext["SimulationPhantom"]["transmit"] == "b1_ramp"


#**************************************************************************************************#
#                                       Class TestLineshape                                        #
#**************************************************************************************************#
#                                                                                                  #
# The microscopic field term: an exact Gaussian envelope on the spin-system components, and none   #
# on the parameterized background whose widths already carry it.                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestLineshape:
    STILL = dict(noise=dict(sigma=0.0), spurious=dict(amplitude=0.0),
                 acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0))

    #*********************#
    #   exact envelope    #
    #*********************#
    def test_gaussian_width_is_an_exact_envelope_on_the_spin_systems(self, scan):
        from mrsandbox import NoDownfield, NoMM

        def fid(gauss_hz):
            params = Params(**self.STILL, lineshape=dict(gauss_hz=gauss_hz))
            voxel = VirtualScan(scan.brain, scan.scanner, scan.sequence, params=params,
                                mm=NoMM(), downfield=NoDownfield()).voxel(
                position_mm=(10, -20, 5), size_mm=20)
            return voxel.fid(seed=7, npoints=512)[1]

        sharp, broad = fid(0.0), fid(4.0)
        t = np.arange(512) / 4000.0
        envelope = np.exp(-(np.pi * 4.0) ** 2 / (4.0 * np.log(2.0)) * t * t)
        np.testing.assert_allclose(broad, sharp * envelope, atol=1e-5 * np.abs(sharp).max())
        assert np.abs(broad[-1]) < 0.5 * np.abs(sharp[-1])  # the tail is damped

    #***********************#
    #   background exempt   #
    #***********************#
    def test_parameterized_background_keeps_its_own_widths(self, scan):
        from mrsandbox import NoLipids, NoWater, ParameterizedMM

        def fid(gauss_hz):
            params = Params(**self.STILL, lineshape=dict(gauss_hz=gauss_hz),
                            conc={name: 0.0 for name in scan.params.conc})
            voxel = VirtualScan(scan.brain, scan.scanner, scan.sequence, params=params,
                                water=NoWater(), lipids=NoLipids(),
                                mm=ParameterizedMM()).voxel(position_mm=(10, -20, 5), size_mm=20)
            return voxel.fid(seed=7, npoints=512)[1]

        only_background = fid(0.0)
        assert np.abs(only_background).max() > 0.0
        np.testing.assert_array_equal(fid(6.0), only_background)
