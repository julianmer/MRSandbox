####################################################################################################
#                                            values.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Value providers: every configurable quantity knows how to sample itself. Priors and     #
#          learned posteriors share this one interface, so updating beliefs in the closed loop     #
#          means swapping a leaf, nothing more. Realizations are host side configuration values;   #
#          reparameterized in graph sampling belongs to the signal model.                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc
import dataclasses
import numbers
import typing


#**************************************************************************************************#
#                                           Class Value                                            #
#**************************************************************************************************#
#                                                                                                  #
# One configurable quantity that can realize itself from a random generator.                       #
#                                                                                                  #
#**************************************************************************************************#
class Value(abc.ABC):
    """One configurable quantity that can realize itself from a random generator."""

    #************#
    #   sample   #
    #************#
    @abc.abstractmethod
    def sample(self, rng):
        """Return a concrete realization of this quantity.

        Args:
            rng: A numpy random Generator.
        """

    #**************#
    #   coercion   #
    #**************#
    @staticmethod
    def of(x):
        """Coerce shorthand into a Value: a Value passes through, a numeric 2-tuple
        becomes Uniform(low, high), and anything else becomes Fixed(x)."""
        if isinstance(x, Value):
            return x
        if isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, numbers.Real) for v in x):
            return Uniform(*x)
        return Fixed(x)


#**************************************************************************************************#
#                                           Class Fixed                                            #
#**************************************************************************************************#
#                                                                                                  #
# A quantity that always realizes to the same value.                                               #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Fixed(Value):
    """A quantity that always realizes to the same value."""

    value: object

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        return self.value


#**************************************************************************************************#
#                                          Class Uniform                                           #
#**************************************************************************************************#
#                                                                                                  #
# A quantity drawn uniformly from [low, high), the low/high range convention.                      #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Uniform(Value):
    """A quantity drawn uniformly from [low, high), the low/high range convention."""

    low: float
    high: float

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))


#**************************************************************************************************#
#                                           Class Normal                                           #
#**************************************************************************************************#
#                                                                                                  #
# A quantity drawn from a normal distribution.                                                     #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Normal(Value):
    """A quantity drawn from a normal distribution."""

    mean: float
    sd: float

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        return float(rng.normal(self.mean, self.sd))


#**************************************************************************************************#
#                                          Class Tissues                                           #
#**************************************************************************************************#
#                                                                                                  #
# A realized grey matter, white matter, CSF triple.                                                #
#                                                                                                  #
#**************************************************************************************************#
class Tissues(typing.NamedTuple):
    """A realized grey matter, white matter, CSF triple."""

    gm: float
    wm: float
    csf: float = 0.0


#**************************************************************************************************#
#                                         Class PerTissue                                          #
#**************************************************************************************************#
#                                                                                                  #
# Grey matter, white matter, and CSF components, each with the usual shorthand.                    #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class PerTissue(Value):
    """Grey matter, white matter, and CSF components, each with the usual shorthand
    (number, 2-tuple range, or Value). Sampling returns a concrete Tissues record,
    ready to be weighted by the tissue fractions of a voxel."""

    gm: object
    wm: object
    csf: object = 0.0

    #**************#
    #   coercion   #
    #**************#
    def __post_init__(self):
        for field in ("gm", "wm", "csf"):
            object.__setattr__(self, field, Value.of(getattr(self, field)))

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        return Tissues(gm=self.gm.sample(rng), wm=self.wm.sample(rng), csf=self.csf.sample(rng))
