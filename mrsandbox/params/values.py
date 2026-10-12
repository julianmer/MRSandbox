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
import functools
import numbers
import typing

import numpy as np


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
#                                         Class Empirical                                          #
#**************************************************************************************************#
#                                                                                                  #
# A quantity drawn as one of the values measured on a population, however that is shaped.          #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Empirical(Value):
    """A quantity drawn from the population as it was measured, however that is shaped
    (the package's pooled readings fall into sites and vendors, not about one middle):
    a kernel density over the subjects' values. A draw is one subject's value plus a
    Gaussian as wide as the subjects' typical gap (the median distance of a value to its
    nearest neighbour), held inside the measured range, so the draws fill the shape
    between neighbouring subjects and never smear one site's cluster into another's.
    `low`, `high` and `value` read the 5th and 95th percentile and the median, the
    range convention beside the draw. `weights`, one per value, make a subject that much
    likelier to be drawn and count as much in the percentiles (subjects of a small
    dataset weigh more each, so no dataset outvotes another by its size alone); None, or
    all equal, weighs every subject alike."""

    values: tuple
    weights: tuple = None

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if not self.values:
            raise ValueError("an empirical value needs at least one measured value")
        if self.weights is not None:
            weights = tuple(float(w) for w in self.weights)
            if len(weights) != len(self.values):
                raise ValueError("an empirical value needs one weight per value")
            object.__setattr__(self, "weights",
                               None if len(set(weights)) == 1 else weights)

    def percentile(self, q):
        """The value at percentile `q` of the weighted population: each subject's weight
        centred on its value, interpolated between them (with no weights, numpy's own)."""
        if self.weights is None:
            return float(np.percentile(self.values, q))
        order = np.argsort(self.values)
        values = np.asarray(self.values)[order]
        weights = np.asarray(self.weights)[order]
        centres = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
        return float(np.interp(q / 100.0, centres, values))

    @functools.cached_property
    def bandwidth(self):
        """The kernel's width: the median distance of a value to its nearest neighbour;
        zero for one value."""
        values = np.sort(np.asarray(self.values))
        if values.size < 2:
            return 0.0
        gaps = np.diff(values)
        nearest = np.minimum(np.append(gaps, np.inf), np.insert(gaps, 0, np.inf))
        return float(np.median(nearest))

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        if self.weights is None:
            index = int(rng.integers(len(self.values)))
        else:
            cumulative = np.cumsum(self.weights)
            index = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1],
                                            side="right")), len(self.values) - 1)
        drawn = self.values[index] + self.bandwidth * float(rng.standard_normal())
        return float(min(max(drawn, min(self.values)), max(self.values)))

    @property
    def value(self):
        return self.percentile(50)

    @property
    def low(self):
        return self.percentile(5)

    @property
    def high(self):
        return self.percentile(95)


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
#                                          Class Mixture                                           #
#**************************************************************************************************#
#                                                                                                  #
# A quantity drawn from one of several populations, each a voice, then from that population.       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Mixture(Value):
    """A quantity drawn from one of several populations, each a voice, then from that
    population: the literature's studies and the measured datasets side by side, each as
    likely as another unless `weights` say otherwise (the reference range over individuals
    from several studies with equal study weights, Siegel, Murad and Chu 2021, Res Synth
    Methods 12:148). `value`, `low` and `high` read the median and the 5th and 95th
    percentile of the mixture off a fixed draw of it."""

    components: tuple
    weights: tuple = None

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(Value.of(c) for c in self.components))
        if not self.components:
            raise ValueError("a mixture needs at least one component")
        if self.weights is not None:
            weights = tuple(float(w) for w in self.weights)
            if len(weights) != len(self.components):
                raise ValueError("a mixture needs one weight per component")
            object.__setattr__(self, "weights",
                               None if len(set(weights)) == 1 else weights)

    #************#
    #   sample   #
    #************#
    def sample(self, rng):
        if self.weights is None:
            index = int(rng.integers(len(self.components)))
        else:
            cumulative = np.cumsum(self.weights)
            index = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1],
                                            side="right")), len(self.components) - 1)
        return self.components[index].sample(rng)

    def percentile(self, q):
        """The value at percentile `q` of the mixture, off 4000 draws of a fixed seed."""
        rng = np.random.default_rng(0)
        return float(np.percentile([self.sample(rng) for _ in range(4000)], q))

    @property
    def value(self):
        return self.percentile(50)

    @property
    def low(self):
        return self.percentile(5)

    @property
    def high(self):
        return self.percentile(95)


# the sources a leaf's value can come from, in the order they are shown
SOURCES = ("literature", "pooled", "mixed", "dataset", "own", "custom")


#**************************************************************************************************#
#                                          Class Sourced                                           #
#**************************************************************************************************#
#                                                                                                  #
# A leaf with a value per source, literature, measured and custom, and the one chosen.             #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Sourced(Value):
    """A leaf with a value per source, literature, pooled, own and custom, and the one chosen.

    `literature` is a published value or a manufacturer's number (None where none is
    published: a zero written for want of a number is not literature), `pooled` the
    package's measured rows pooled for the scanner (scan.cohort), `mixed` the literature's
    studies and the measured datasets as equal voices (a Mixture), `dataset` one real
    subject's own reading from a package dataset on the floor, `own` a reading made on
    the operator's own scan, `custom` the user's (None until one is set: merging a plain
    Value onto this leaf sets it and chooses it). `provenance` says where each source's
    value comes from, {source: text}. The leaf samples, compares and reads (low, high,
    mean, sd, value) as the chosen source's value does, so the rest of the package sees
    one Value.
    """
    literature: Value = None
    pooled: Value = None
    mixed: Value = None
    dataset: Value = None
    own: Value = None
    custom: Value = None
    chosen: str = "literature"
    provenance: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.current is None:
            raise ValueError(f"no {self.chosen} value to choose")

    @property
    def current(self):
        """The chosen source's Value."""
        return getattr(self, self.chosen)

    @property
    def available(self):
        """The sources this leaf has a value for, in the order literature, pooled, mixed,
        dataset, own, custom."""
        return tuple(name for name in SOURCES if getattr(self, name) is not None)

    def choose(self, source, strict=False):
        """This leaf with `source` chosen; a source it has no value for leaves the choice as
        it stands, or raises with `strict`."""
        if getattr(self, source, None) is None:
            if strict:
                raise ValueError(f"no {source} value for this leaf; "
                                 f"it has {', '.join(self.available)}")
            return self
        return dataclasses.replace(self, chosen=source)

    def with_custom(self, value):
        """This leaf with `value` as its custom source, chosen."""
        return dataclasses.replace(self, custom=Value.of(value), chosen="custom",
                                   provenance={**self.provenance, "custom": "set by the operator"})

    def sample(self, rng):
        return self.current.sample(rng)

    def __getattr__(self, name):
        if name in (*SOURCES, "chosen", "provenance") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __eq__(self, other):
        if isinstance(other, Sourced):
            return dataclasses.astuple(self) == dataclasses.astuple(other)
        return self.current == other

    def __hash__(self):
        return hash((self.literature, self.pooled, self.mixed, self.dataset, self.own,
                     self.custom, self.chosen))


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
