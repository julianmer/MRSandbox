####################################################################################################
#                                          test_params.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Parameter system gate: coercion sugar, nesting, attribute access, right biased          #
#          composition, seeded determinism, and realizations that contain no providers and         #
#          serialize cleanly.                                                                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import (Clipped, Fixed, Mapped, Normal, Params, PerTissue, Tissues, Uniform, Value,
                       Volume)


#**************************************************************************************************#
#                                         Class TestValues                                         #
#**************************************************************************************************#
class TestValues:

    #**************#
    #   coercion   #
    #**************#
    def test_coercion_sugar(self):
        assert Value.of(0.011) == Fixed(0.011)
        assert Value.of((2.0, 15.0)) == Uniform(2.0, 15.0)
        assert Value.of("icbm152") == Fixed("icbm152")
        assert Value.of((1, 2, 3)) == Fixed((1, 2, 3))
        u = Uniform(0.0, 1.0)
        assert Value.of(u) is u

    #*******************#
    #   distributions   #
    #*******************#
    def test_distributions_sample_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert 7.5 <= Uniform(7.5, 12.0).sample(rng) < 12.0
        draws = [Normal(0.0, 1.0).sample(rng) for _ in range(2000)]
        assert abs(np.mean(draws)) < 0.1 and abs(np.std(draws) - 1.0) < 0.1

    #*************#
    #   clipped   #
    #*************#
    def test_clipped_bounds_the_draw_without_touching_the_stream(self):
        raw = [Normal(0.0, 1.0).sample(np.random.default_rng(s)) for s in range(60)]
        held = [Clipped(Normal(0.0, 1.0), low=-0.5, high=0.5).sample(np.random.default_rng(s))
                for s in range(60)]
        # the same randomness, clipped: out-of-bound mass lands exactly on the bound
        assert held == [min(max(x, -0.5), 0.5) for x in raw]
        assert any(x == -0.5 or x == 0.5 for x in held)  # the bounds actually engage
        # shorthand coerces inside the wrapper, one side may stay open
        assert Clipped((1.0, 5.0), high=3.0).sample(np.random.default_rng(2)) <= 3.0

    #****************#
    #   per tissue   #
    #****************#
    def test_a_weighted_population_draws_by_its_weights(self):
        from mrsandbox import Empirical

        # two distant clusters, three subjects against one: weighted one quarter each, the
        # lone subject half the draws
        population = Empirical((1.0, 1.1, 1.2, 9.0), (1 / 6, 1 / 6, 1 / 6, 1 / 2))
        rng = np.random.default_rng(0)
        draws = np.array([population.sample(rng) for _ in range(4000)])
        assert np.mean(draws > 5.0) == pytest.approx(0.5, abs=0.03)
        assert population.value > 1.2 and population.high == 9.0
        # equal weights are no weights: the draw stream and the percentiles unchanged
        plain = Empirical((1.0, 2.0, 3.0))
        assert Empirical((1.0, 2.0, 3.0), (0.2, 0.2, 0.2)) == plain and plain.value == 2.0

    def test_a_mixture_draws_each_voice_alike(self):
        from mrsandbox import Empirical, Mixture

        # a study about 10 and a dataset of two subjects about 20: half the draws each
        mixture = Mixture((Normal(10.0, 0.1), Empirical((19.9, 20.1))))
        rng = np.random.default_rng(1)
        draws = np.array([mixture.sample(rng) for _ in range(4000)])
        assert np.mean(draws > 15.0) == pytest.approx(0.5, abs=0.03)
        assert mixture.low < 10.5 and mixture.high > 19.5

    def test_per_tissue_realizes_components(self):
        value = PerTissue(gm=(6.0, 12.0), wm=Uniform(4.0, 9.0), csf=0.0)
        concrete = value.sample(np.random.default_rng(1))
        assert isinstance(concrete, Tissues)
        assert 6.0 <= concrete.gm < 12.0 and 4.0 <= concrete.wm < 9.0 and concrete.csf == 0.0

    #************#
    #   mapped   #
    #************#
    def test_mapped_samples_to_its_map_and_records_its_name(self):
        # a map is not drawn: the realization carries the map itself for the voxel to
        # resolve, and the serialized truth carries its name
        volume = Volume(np.full((4, 4, 4), 1.3), np.eye(4), name="t1_synthetic")
        truth = Params(water=dict(t1_s=Mapped(volume), t2_s=0.08)).sample(seed=1)
        assert truth.water.t1_s is volume
        assert truth.to_dict() == {"water": {"t1_s": "t1_synthetic", "t2_s": 0.08}}


#**************************************************************************************************#
#                                         Class TestParams                                         #
#**************************************************************************************************#
class TestParams:

    #******************#
    #   construction   #
    #******************#
    def test_nesting_and_attribute_access(self):
        p = Params(brain=dict(NAA=dict(conc=(7.5, 12.0), t2=0.16)), te1_s=0.011)
        assert isinstance(p.brain, Params)
        assert p.brain.NAA.conc == Uniform(7.5, 12.0)
        assert p["brain"]["NAA"]["t2"] == Fixed(0.16)
        with pytest.raises(AttributeError, match="known:"):
            p.brain.GABA

    #*****************#
    #   composition   #
    #*****************#
    def test_or_merges_deep_and_right_biased(self):
        base = Params(brain=dict(NAA=dict(conc=(7.5, 12.0)), Cr=dict(conc=(4.5, 10.5))), noise=0.0)
        override = Params(brain=dict(NAA=dict(conc=10.0)), noise=(0.0, 5.0))
        merged = base | override
        assert merged.brain.NAA.conc == Fixed(10.0)
        assert merged.brain.Cr.conc == Uniform(4.5, 10.5)
        assert merged.noise == Uniform(0.0, 5.0)
        assert base.brain.NAA.conc == Uniform(7.5, 12.0)  # inputs untouched

    def test_or_shares_no_subtree_with_its_operands(self):
        # a loop composes defaults | learned and edits the result: neither operand
        # may see the edit, whether the subtree came untouched from the left or
        # whole from the right
        base = Params(conc=dict(NAA=(7.5, 12.0)), noise=dict(sigma=1.0))
        override = Params(mm=dict(kappa=7.0))
        merged = base | override
        merged.conc.NAA = 5.0
        merged.mm.kappa = 1.0
        assert base.conc.NAA == Uniform(7.5, 12.0)
        assert override.mm.kappa == Fixed(7.0)
        assert merged.conc is not base.conc and merged.mm is not override.mm

    #*****************#
    #   determinism   #
    #*****************#
    def test_sampling_is_seeded_and_deterministic(self):
        p = Params(brain=dict(NAA=dict(conc=(7.5, 12.0))), lw_hz=(2.0, 15.0), te1_s=0.011)
        a, b, c = p.sample(seed=7), p.sample(seed=7), p.sample(seed=8)
        assert a.to_dict() == b.to_dict()
        assert a.to_dict() != c.to_dict()
        assert a.te1_s == 0.011
        assert 7.5 <= a.brain.NAA.conc < 12.0

    #********************#
    #   leaf stability   #
    #********************#
    def test_draws_are_stable_under_tree_membership(self):
        # every leaf draws from its own path-keyed stream: adding an unrelated
        # parameter never shifts the realization of the others, so recorded ground
        # truths stay reproducible as the tree grows
        small = Params(brain=dict(NAA=dict(conc=(7.5, 12.0))), lw_hz=(2.0, 15.0))
        grown = small | Params(aaa=(0.0, 1.0), zzz=(0.0, 1.0))
        a, b = small.sample(seed=7), grown.sample(seed=7)
        assert a.brain.NAA.conc == b.brain.NAA.conc
        assert a.lw_hz == b.lw_hz

    #*****************#
    #   realization   #
    #*****************#
    def test_realization_holds_no_providers(self):
        p = Params(
            brain=dict(NAA=dict(conc=PerTissue(gm=(6.0, 12.0), wm=(4.0, 9.0)))),
            te1_s=(0.01, 0.1),
        )
        truth = p.sample(seed=3)

        def providers(tree):
            for value in tree.values():
                if isinstance(value, Params):
                    yield from providers(value)
                elif isinstance(value, Value):
                    yield value

        assert list(providers(truth)) == []
        assert isinstance(truth.brain.NAA.conc.gm, float)
        d = truth.to_dict()
        assert d["brain"]["NAA"]["conc"]["gm"] == truth.brain.NAA.conc.gm


#**************************************************************************************************#
#                                        Class TestSourced                                         #
#**************************************************************************************************#
class TestSourced:

    #**************#
    #   one leaf   #
    #**************#
    def test_a_sourced_leaf_reads_and_samples_as_its_chosen_source(self):
        from mrsandbox import Sourced

        leaf = Sourced(literature=Uniform(1.0, 2.0), pooled=Normal(1.5, 0.1), chosen="pooled")
        assert (leaf == Normal(1.5, 0.1) and (leaf.mean, leaf.sd) == (1.5, 0.1)
                and leaf.available == ("literature", "pooled"))
        rng = np.random.default_rng(3)
        assert leaf.sample(rng) == Normal(1.5, 0.1).sample(np.random.default_rng(3))
        assert leaf.choose("literature") == Uniform(1.0, 2.0)
        assert leaf.choose("custom") is leaf  # nothing to choose: the choice stands
        with pytest.raises(ValueError):
            leaf.choose("custom", strict=True)
        assert (leaf.with_custom(3.0) == Fixed(3.0)
                and leaf.with_custom(3.0).literature == Uniform(1.0, 2.0))
        with pytest.raises(ValueError):
            Sourced(literature=Uniform(1.0, 2.0), chosen="pooled")

    #****************#
    #   in a tree    #
    #****************#
    def test_a_tree_selects_its_sources_and_a_merge_sets_the_custom_one(self):
        from mrsandbox import Sourced

        tree = Params(a=dict(b=Sourced(literature=Uniform(1.0, 2.0), pooled=Normal(1.5, 0.1),
                                       chosen="pooled"),
                             c=Sourced(pooled=Fixed(4.0), chosen="pooled"), d=(0.0, 1.0)))
        assert list(tree.sources()) == ["a.b", "a.c"]
        literature = tree.select("literature")
        # no literature for c: it keeps measured
        assert literature.a.b == Uniform(1.0, 2.0) and literature.a.c == Fixed(4.0)
        assert tree.a.b == Normal(1.5, 0.1)  # the original untouched
        custom = tree | Params(a=dict(b=3.0))
        assert (custom.a.b == Fixed(3.0) and custom.a.b.chosen == "custom"
                and custom.a.b.pooled == Normal(1.5, 0.1))
        assert (custom.select("pooled").a.b == Normal(1.5, 0.1)
                and custom.select("pooled", **{"a.b": "custom"}).a.b == Fixed(3.0))
        with pytest.raises(ValueError):
            tree.select(**{"a.c": "literature"})
        assert tree.sample(seed=1).a.b != literature.sample(seed=1).a.b
