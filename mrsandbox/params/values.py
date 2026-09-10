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
#                                          Class Clipped                                           #
#**************************************************************************************************#
#                                                                                                  #
# Any scalar provider held to physically plausible bounds: the draw is clipped, not resampled.     #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Clipped(Value):
    """Any scalar provider held to physically plausible bounds: the draw is clipped,
    not resampled, so it consumes exactly the inner provider's randomness and the
    seeded draw stream stays stable. Wrap any provider (or its shorthand) anywhere in
    a Params tree, e.g. Clipped(Normal(0.039, 0.013), low=0.005) for a relaxation time
    that must stay positive; out-of-bound mass lands on the bound, the stated clipping
    semantics.

    Args:
        value: The provider to draw from, or its coercible shorthand.
        low, high: Physical bounds; None leaves that side open.
    """

    value: Value
    low: float = None
    high: float = None

    #**************#
    #   coercion   #
    #**************#
    def __post_init__(self):
        object.__setattr__(self, "value", Value.of(self.value))

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        drawn = self.value.sample(rng)
        if self.low is not None:
            drawn = max(drawn, self.low)
        if self.high is not None:
            drawn = min(drawn, self.high)
        return drawn


#**************************************************************************************************#
#                                           Class Mapped                                           #
#**************************************************************************************************#
#                                                                                                  #
# A quantity read from a spatial map, resolved by the placed voxel at render time.                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Mapped(Value):
    """A quantity read from a spatial map (a phantom.Volume) instead of drawn: sampling
    returns the map itself, and the placed voxel resolves it to the overlap-weighted
    mean over its box at render time (scan.Voxel.resolve), so one truth serves every
    position while each voxel reads its own local value. The ground truth records the
    map's name, the acquisition header the resolved value.

    Args:
        volume: A phantom.Volume in the quantity's own units (T1 in seconds, mM, ...).
    """

    volume: object

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        return self.volume


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

    #**************#
    #   weighing   #
    #**************#
    def weighted(self, fractions):
        """This per-tissue value averaged by a voxel's composition (another Tissues of
        gm, wm, csf fractions, normalized); an empty voxel falls back to the gm value."""
        total = fractions.gm + fractions.wm + fractions.csf
        if total <= 0.0:
            return self.gm
        return (fractions.gm * self.gm + fractions.wm * self.wm
                + fractions.csf * self.csf) / total


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
