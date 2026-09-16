####################################################################################################
#                                             trees.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The Params tree: a dict with attribute access whose leaves are Value providers. Trees   #
#          nest per area (brain, hardware, sequence), compose with the | operator instead of copy  #
#          pasting presets, and realize with one seeded sample whose result is the ground truth    #
#          record.                                                                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import functools
import hashlib

import numpy as np

from mrsandbox.params.values import Value


#********************#
#   leaf seed keys   #
#********************#
@functools.lru_cache(maxsize=4096)
def _path_entropy(path):
    """A stable integer derived from a leaf's key path, mixed with the draw seed to
    give every leaf its own independent stream."""
    return int.from_bytes(hashlib.sha256("/".join(path).encode()).digest()[:8], "little")


#**************************************************************************************************#
#                                           Class Params                                           #
#**************************************************************************************************#
#                                                                                                  #
# A nested, composable, samplable parameter tree.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class Params(dict):
    """A nested, composable, samplable parameter tree.

    Leaves are coerced on the way in: a Value passes through, a numeric 2-tuple becomes
    Uniform, a dict becomes a nested Params, and anything else becomes Fixed. Keys are
    reachable both as items and as attributes.

    Examples:
        >>> p = Params(brain=dict(NAA=dict(conc=(7.5, 12.0))), te1_s=0.011)
        >>> p.brain.NAA.conc
        Uniform(low=7.5, high=12.0)
        >>> truth = p.sample(seed=7)
        >>> truth.te1_s
        0.011
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, mapping=(), **kwargs):
        super().__init__()
        for key, value in dict(mapping, **kwargs).items():
            self[key] = value

    #*****************#
    #   item access   #
    #*****************#
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Params):
            value = Params(value)
        elif not isinstance(value, (Params, Value)):
            value = Value.of(value)
        super().__setitem__(key, value)

    #**********************#
    #   attribute access   #
    #**********************#
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"no parameter {key!r}; known: {', '.join(sorted(self))}") from None

    def __setattr__(self, key, value):
        self[key] = value

    #*****************#
    #   composition   #
    #*****************#
    def copy(self):
        """A copy of the tree structure; the leaves are shared (they are frozen)."""
        return Params({key: value.copy() if isinstance(value, Params) else value
                       for key, value in self.items()})

    def __or__(self, other):
        """Deep, right biased merge: leaves in `other` override, subtrees merge recursively.
        The result shares no subtree with either operand, so editing it never reaches
        back into the defaults it was composed from."""
        merged = self.copy()
        for key, value in Params(other).items():
            if key in merged and isinstance(merged[key], Params) and isinstance(value, Params):
                merged[key] = merged[key] | value
            else:
                merged[key] = value.copy() if isinstance(value, Params) else value
        return merged

    #**************#
    #   sampling   #
    #**************#
    def sample(self, seed=None):
        """Realize every leaf independently from one seed and return the concrete tree.

        Every leaf draws from its own generator, derived from the seed and the leaf's
        key path, so a given seed always yields the same realization for a given leaf
        no matter what else the tree holds: adding, removing, or renaming parameters
        never shifts the other draws. The returned tree contains no Value providers;
        it is the ground truth record of this draw.

        Args:
            seed: Seed, or an existing Generator (one root integer is drawn from it,
                so a stream of samples stays reproducible).

        Returns:
            Params of concrete values.
        """
        if isinstance(seed, np.random.Generator):
            root = int(seed.integers(2 ** 63))
        elif seed is None:
            root = int(np.random.SeedSequence().entropy)
        else:
            root = int(seed)
        return self._realize(root, ())

    def _realize(self, root, path):
        """Realize the subtree at `path`, giving every leaf its path-keyed generator."""
        realization = Params()
        for key in sorted(self):
            value = self[key]
            if isinstance(value, Params):
                drawn = value._realize(root, path + (key,))
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([root, _path_entropy(path + (key,))]))
                drawn = value.sample(rng)
            dict.__setitem__(realization, key, drawn)
        return realization

    #*******************#
    #   serialization   #
    #*******************#
    def to_dict(self):
        """Plain nested dict view, with dataclass leaves flattened to dicts and map
        leaves reduced to their names."""
        from mrsandbox.phantom.volume import Volume

        def plain(value):
            if isinstance(value, Params):
                return value.to_dict()
            if isinstance(value, Volume):
                return value.name
            if hasattr(value, "_asdict"):
                return dict(value._asdict())
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return dataclasses.asdict(value)
            return value

        return {key: plain(value) for key, value in self.items()}
