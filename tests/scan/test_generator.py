####################################################################################################
#                                        test_generator.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Training-mode gate: the stream is seeded and reproducible, shapes and targets line up   #
#          with the params tree, the noise leaf controls the noise floor, and consecutive batches  #
#          are fresh draws.                                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import (Brain, CoherenceFilter, Delay, Gradient, Params, Press, Projection,
                       Scanner, Sequence, ShapedPulse, VirtualScan)

# full-scan integration renders: the tensorflow CI leg skips these (-m 'not render')
pytestmark = pytest.mark.render

VOXEL = dict(position_mm=(10.0, -20.0, 5.0), size_mm=20.0)


#*****************#
#   shared scan   #
#*****************#
@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Brain.load("icbm152"), Scanner.preset("generic_3t"),
                       Press(te1_s=0.011, te2_s=0.019))


#**************************************************************************************************#
#                                       Class TestGenerator                                        #
#**************************************************************************************************#
class TestGenerator:

    #************************#
    #   shapes and targets   #
    #************************#
    def test_shapes_dtypes_and_target_order(self, scan):
        gen = scan.generator(**VOXEL, batch_size=4, seed=1, npoints=512)
        x, y = next(gen)
        assert x.shape == (4, 2, 512) and x.dtype == np.float32
        assert y.shape == (4, len(scan.params.conc)) and y.dtype == np.float32
        assert gen.metabolites == tuple(sorted(scan.params.conc))
        column = gen.metabolites.index("NAA")
        assert all(7.5 <= value < 12.0 for value in y[:, column])
        assert len(gen.truths) == 4

    #*******************#
    #   random access   #
    #*******************#
    def test_batches_are_random_access_and_a_pydataset(self, scan):
        # a keras PyDataset feeds model.fit directly; batch index keys the draws, so
        # any batch reproduces on its own regardless of iteration order
        import keras

        gen = scan.generator(**VOXEL, batch_size=2, seed=7, npoints=256)
        assert isinstance(gen, keras.utils.PyDataset)
        assert gen.num_batches is None  # infinite: set steps_per_epoch when fitting
        first, second = next(gen), next(gen)
        np.testing.assert_array_equal(gen[1][0], second[0])
        np.testing.assert_array_equal(gen[0][0], first[0])
        assert not np.array_equal(first[0], second[0])

    #*************#
    #   seeding   #
    #*************#
    def test_stream_is_seeded_and_fresh(self, scan):
        a1, b1 = next(scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256))
        a2, b2 = next(scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256))
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
        gen = scan.generator(**VOXEL, batch_size=3, seed=7, npoints=256)
        first, second = next(gen), next(gen)
        assert np.abs(first[0] - second[0]).max() > 0.0
        other, _ = next(scan.generator(**VOXEL, batch_size=3, seed=8, npoints=256))
        assert np.abs(a1 - other).max() > 0.0

    #*******************#
    #   noise control   #
    #*******************#
    def test_noise_leaf_controls_the_floor(self, scan):
        quiet_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                 params=Params(noise=dict(sigma=0.0)))
        loud_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                params=Params(noise=dict(sigma=2.0)))
        quiet, _ = next(quiet_scan.generator(**VOXEL, batch_size=1, seed=5, npoints=1024))
        loud, _ = next(loud_scan.generator(**VOXEL, batch_size=1, seed=5, npoints=1024))
        # same seed and identical draw order: the difference is exactly the added noise,
        # whose real and imaginary channels each carry the sampled sigma at the
        # default 4 kHz (the sigma belongs to 5 kHz and white noise scales by the root)
        noise = loud - quiet
        assert np.abs(noise).max() > 0.0
        assert abs(np.std(noise) - 2.0 * np.sqrt(0.8)) < 0.2

    def test_noise_draws_never_shift_the_next_truth(self, scan):
        # every sample owns its generator, so a quiet and a noisy stream of one seed
        # draw the same truths for the whole batch, not only its first sample
        quiet_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                 params=Params(noise=dict(sigma=0.0)))
        loud_scan = VirtualScan(scan.brain, scan.scanner, scan.sequence,
                                params=Params(noise=dict(sigma=2.0)))
        quiet_gen = quiet_scan.generator(**VOXEL, batch_size=3, seed=5, npoints=512)
        loud_gen = loud_scan.generator(**VOXEL, batch_size=3, seed=5, npoints=512)
        (quiet, y_quiet), (loud, y_loud) = next(quiet_gen), next(loud_gen)
        np.testing.assert_array_equal(y_quiet, y_loud)
        for a, b in zip(quiet_gen.truths, loud_gen.truths):
            assert a.conc.to_dict() == b.conc.to_dict()
        for i in range(3):
            assert abs(np.std(loud[i] - quiet[i]) - 2.0 * np.sqrt(0.8)) < 0.2  # 4 kHz

    def test_a_batch_equals_its_draws_rendered_one_by_one(self, scan):
        # the batch's metabolite lines come from one replay mapped over the draws; every draw's
        # FID is the one its own replay renders (the same truths, the same noise: the rng draws
        # the truth, then the noise)
        gen = scan.generator(batch_size=6, seed=11, npoints=512, **VOXEL)
        x, _ = gen[3]
        scan._frame_lines.clear()  # the lone replays make their own lines
        alone = scan.voxel(compiled=True, **VOXEL)
        for i in range(6):
            rng = np.random.default_rng(np.random.SeedSequence([gen.root, 3, i]))
            truth = scan.params.sample(rng)
            _, fid = alone._render(truth, rng, 512, gen.dwell_s)
            assert np.abs(x[i, 0] + 1j * x[i, 1] - fid).max() < 1e-5 * np.abs(fid).max()
        assert len(scan._lines) <= 256

    #*********************#
    #   one frame shared  #
    #*********************#
    def test_draws_at_one_frame_simulate_the_metabolites_once(self, scan):
        # a closed loop over concentrations and noise: the relaxation, the transmit scale and
        # the frequency offset pinned, every draw sits at one frame, whose lines are made once
        # and shared; what differs between the draws is theirs alone
        names = tuple(scan.params.conc)
        fixed = Params(t1={name: 1.4 for name in names}, t2={name: 0.2 for name in names},
                       b1_scale=1.0, acquisition=dict(shift_hz=0.0))
        source = VirtualScan(scan.brain, scan.scanner, scan.sequence, params=fixed)
        gen = source.generator(**VOXEL, batch_size=4, seed=3, npoints=256)
        x, y = next(gen)
        assert len(source._frame_lines) == 1
        first, second = gen.truths[:2]
        assert gen.voxel._metabolite_lines(first) is gen.voxel._metabolite_lines(second)
        assert first.conc.NAA != second.conc.NAA and not np.array_equal(x[0], x[1])

    #*********************#
    #   a played design   #
    #*********************#
    def test_a_played_batch_shares_each_transmit_frame(self, scan):
        # under shaped pulses on an ensemble the batch's lines come from the graphs split at the
        # transmit frame (Engine.framed): the pulses of a frame once on the scan, the draws'
        # relaxation mapped over them; every draw's FID is the one its own full replay renders
        design = Sequence(
            ShapedPulse.sinc(90.0, 0.002, offset_ppm=3.0, steps=32, gradient_hz_per_mm=150.0,
                             gradient_axis="z", extent_mm=40.0),
            Gradient(-0.5 * 150.0 * 0.002, 40.0, axis="z"), CoherenceFilter(-1), Projection(2.0),
            Delay(0.004))
        fixed = Params(b1_scale=1.0, acquisition=dict(shift_hz=0.0))
        source = VirtualScan(scan.brain, scan.scanner, design, params=fixed, isochromats=16)
        gen = source.generator(**VOXEL, batch_size=4, seed=11, npoints=256)
        x, _ = gen[2]
        (pulses,) = source._propagators.values()
        assert len(pulses) == len(source.params.conc) and all(p is not None for p in pulses)
        assert len(source._frame_lines) == 4  # the relaxation still draws per sample
        source._frame_lines.clear()  # the lone replays run the full graphs
        alone = source.voxel(compiled=True, **VOXEL)
        for i in range(4):
            rng = np.random.default_rng(np.random.SeedSequence([gen.root, 2, i]))
            truth = source.params.sample(rng)
            _, fid = alone._render(truth, rng, 256, gen.dwell_s)
            assert np.abs(x[i, 0] + 1j * x[i, 1] - fid).max() < 1e-5 * np.abs(fid).max()

