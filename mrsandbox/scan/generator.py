####################################################################################################
#                                           generator.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The training mode: an infinite, seeded stream of freshly simulated batches from one     #
#          voxel. Every batch draws its realizations and noise from a generator keyed by the       #
#          stream seed and the batch index, so a seed reproduces the whole stream and any batch    #
#          alone; inputs come as real and imaginary channels (the network boundary), targets as    #
#          the concentration vector, and the full realized truth rides along.                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import keras
import numpy as np


#**************************************************************************************************#
#                                         Class Generator                                          #
#**************************************************************************************************#
#                                                                                                  #
# An infinite, seeded (inputs, targets) stream simulated on the fly.                               #
#                                                                                                  #
#**************************************************************************************************#
class Generator(keras.utils.PyDataset):
    """An infinite, seeded (inputs, targets) stream simulated on the fly.

    A keras.utils.PyDataset, so model.fit consumes it directly (pass steps_per_epoch;
    workers and queueing come from the PyDataset kwargs), and a plain iterator for
    direct use. Batch `index` draws from its own generator derived from the stream
    seed and the index, so any batch is reproducible on its own and the iteration
    order never matters. Each batch is (x, y) with x of shape (batch, 2, npoints)
    float32 holding the real and imaginary FID channels, and y of shape
    (batch, n_metabolites) float32 holding the drawn concentrations in the order of
    the `metabolites` attribute. `batches(n)` yields n of them; `truths` of the
    latest batch are kept for inspection.

    Args:
        voxel: A placed scan.Voxel.
        batch_size: Samples per batch.
        seed: Seed of the stream; the same seed reproduces the same stream.
        npoints: FID points per sample.
        dwell_s: Dwell time in seconds.
        **kwargs: PyDataset options (workers, use_multiprocessing, max_queue_size).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, voxel, batch_size=16, seed=None, npoints=2048, dwell_s=1.0 / 4000.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.voxel = voxel
        self.batch_size = batch_size
        self.npoints = npoints
        self.dwell_s = dwell_s
        self.metabolites = tuple(sorted(voxel.scan.params.conc))
        self.root = int(np.random.SeedSequence(seed).entropy) if seed is None else int(seed)
        self.truths = ()
        self._next = 0

    #**************#
    #   batches    #
    #**************#
    @property
    def num_batches(self):
        """None: the stream is infinite; set steps_per_epoch when fitting."""
        return None

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.root, int(index)]))
        x = np.empty((self.batch_size, 2, self.npoints), dtype=np.float32)
        y = np.empty((self.batch_size, len(self.metabolites)), dtype=np.float32)
        truths = []
        for i in range(self.batch_size):
            truth = self.voxel.scan.params.sample(rng)
            _, fid = self.voxel._render(truth, rng, self.npoints, self.dwell_s)
            x[i, 0], x[i, 1] = fid.real, fid.imag
            y[i] = [truth.conc[name] for name in self.metabolites]
            truths.append(truth)
        self.truths = tuple(truths)
        return x, y

    #***************#
    #   iteration   #
    #***************#
    def __iter__(self):
        return self

    def __next__(self):
        batch = self[self._next]
        self._next += 1
        return batch

    #*******************#
    #   finite stream   #
    #*******************#
    def batches(self, n):
        """Yield exactly n batches."""
        for _ in range(n):
            yield next(self)
