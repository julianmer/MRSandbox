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

import numpy as np

from mrsandbox.params.values import Value


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
    def __or__(self, other):
        """Deep, right biased merge: leaves in `other` override, subtrees merge recursively."""
        merged = Params(self)
        for key, value in Params(other).items():
            if key in merged and isinstance(merged[key], Params) and isinstance(value, Params):
                merged[key] = merged[key] | value
            else:
                merged[key] = value
        return merged

    #**************#
    #   sampling   #
    #**************#
    def sample(self, seed=None):
        """Realize every leaf with one seeded generator and return the concrete tree.

        Leaves are visited in sorted key order, so a given seed always yields the same
        realization. The returned tree contains no Value providers; it is the ground
        truth record of this draw.

        Args:
            seed: Seed for numpy's default generator, or an existing Generator.

        Returns:
            Params of concrete values.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        realization = Params()
        for key in sorted(self):
            value = self[key]
            dict.__setitem__(realization, key, value.sample(rng) if isinstance(value, (Params, Value)) else value)
        return realization

    #*******************#
    #   serialization   #
    #*******************#
    def to_dict(self):
        """Plain nested dict view, with dataclass leaves flattened to dicts."""

        def plain(value):
            if isinstance(value, Params):
                return value.to_dict()
            if hasattr(value, "_asdict"):
                return dict(value._asdict())
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return dataclasses.asdict(value)
            return value

        return {key: plain(value) for key, value in self.items()}
