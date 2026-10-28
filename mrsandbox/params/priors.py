####################################################################################################
#                                             priors.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Builders that turn database sources into Params subtrees, so priors come from cited     #
#          values and compose like any other Params. In the closed loop a fitted posterior         #
#          replaces any of these leaves without further changes.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import itertools

import numpy as np

from mrsandbox.database import Database, concentrations, regional
from mrsandbox.database.gudmundson import Gudmundson
from mrsandbox.params.trees import Params
from mrsandbox.params.values import Fixed, Normal, PerTissue, Uniform


#**************************************************************************************************#
#                                           Class Priors                                           #
#**************************************************************************************************#
#                                                                                                  #
# Params subtrees built from database sources.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class Priors:
    """Params subtrees built from database sources."""

    #**************#
    #   de graaf   #
    #**************#
    @staticmethod
    def de_graaf(metabolites=None, database=None):
        """Uniform concentration priors (mM) from the de Graaf ranges.

        Args:
            metabolites: Names to include; defaults to every metabolite in the source.
            database: Database instance; defaults to the packaged one.

        Returns:
            Params mapping each metabolite to a Uniform(low, high) leaf.
        """
        db = database if database is not None else Database()
        names = metabolites if metabolites is not None else sorted(concentrations.DE_GRAAF)
        return Params({name: Uniform(*db.concentration(name, source="de_graaf"))
                       for name in names})

    #*******************#
    #   gudmundson t2   #
    #*******************#
    @staticmethod
    def gudmundson_t2(metabolites, fallback_s=0.160, tesla=3):
        """Per-metabolite T2 priors pooled from the Gudmundson database, in seconds.

        Metabolites with pooled rows at `tesla` get Normal(mean, between-study spread);
        without rows at that field they take the 3 T pool (stated), and without any
        rows the Fixed stated assumption. Requires the externals/mrs-database
        submodule; without it, everything falls back.

        Args:
            metabolites: Names to include.
            fallback_s: T2 for metabolites without pooled data.
            tesla: Nominal field filter.

        Returns:
            Params mapping each metabolite to its T2 leaf.
        """
        source = Gudmundson() if Gudmundson.available() else None
        leaves = {}
        for name in metabolites:
            pooled = source.pooled_t2(name, tesla=tesla) if source else None
            if pooled is None and source is not None and tesla != 3:
                pooled = source.pooled_t2(name, tesla=3)
            leaves[name] = Normal(pooled.mean_s, pooled.sd_s) if pooled else Fixed(fallback_s)
        return Params(leaves)

    #*******************************#
    #   gudmundson concentrations   #
    #*******************************#
    @staticmethod
    def gudmundson_concentrations(metabolites, groups=("Healthy", "Control"), tesla=None):
        """Concentration priors pooled from the Gudmundson database for a cohort, in mM.

        Pass a clinical group name (the database's own strings, e.g. "Multiple
        Sclerosis", "Schizophrenia") to build disease priors; only metabolites with
        pooled molar rows appear in the result, so compose over the healthy defaults:
        `Priors.de_graaf(...) | Params(conc=Priors.gudmundson_concentrations(...))`.

        Returns:
            Params mapping each covered metabolite to Normal(mean, spread).
        """
        source = Gudmundson() if Gudmundson.available() else None
        leaves = {}
        for name in metabolites:
            pooled = (source.pooled_concentration(name, groups=groups, tesla=tesla) if source
                      else None)
            if pooled is not None:
                leaves[name] = Normal(pooled.mean_s, pooled.sd_s)
        return Params(leaves)

    #**************#
    #   regional   #
    #**************#
    @staticmethod
    def regional(metabolites, contrast):
        """Each metabolite's level by brain region (phantom.regions) as factors over the drawn
        grey and white matter concentrations, {metabolite: {region: PerTissue}}.

        A study's value counts relative to its own reference (its whole cortex and white
        matter, or the mean of its lobes), so studies with their own units and referencing give
        the pattern alone, and the package's GM to WM contrast keeps setting the tissues' levels.
        A lobe takes a factor per tissue, against the reference in that tissue. A nucleus holds
        one concentration whatever the tissue maps call its cells (stated): its value over the
        mean of the grey and white references times the drawn concentration, that of an equal
        mix, so its parts carry the contrast `r` back out. The newest study gives a region where
        several do (database.regional STUDIES); a value with a spread above its mean is left
        out; a region no study gives stays at 1.

        Args:
            metabolites: The package's names to cover.
            contrast: {name: GM to WM ratio r}, the tree's contrast values.

        Returns:
            Params of the covered metabolites, each {region: PerTissue(gm, wm)}.
        """
        from mrsandbox.phantom.regions import LOBES

        found = {}
        for study in regional.STUDIES:
            values = study["values"]
            for row, (regions, tissue) in study["stands_for"].items():
                for paper, (mean, sd) in values[row].items():
                    if paper not in study["quantified"] or sd > mean:
                        continue
                    reference = np.mean([np.mean([values[r][paper][0] for r in rows])
                                         for t, rows in study["reference"].items()
                                         if tissue in (t, None)])
                    for name, region in itertools.product(regional.NAMES[paper], regions):
                        found.setdefault((name, region), {}).setdefault(tissue, mean / reference)
        leaves = {}
        for (name, region), by in found.items():
            if name not in metabolites:
                continue
            r = contrast[name]
            if region in LOBES:
                gm, wm = by.get("gm", 1.0), by.get("wm", 1.0)
            else:
                gm, wm = by[None] * (1.0 + r) / (2.0 * r), by[None] * (1.0 + r) / 2.0
            leaves.setdefault(name, {})[region] = PerTissue(gm=gm, wm=wm)
        return Params(leaves)

    #*******************#
    #   literature t1   #
    #*******************#
    @staticmethod
    def literature_t1(metabolites, fallback_s=1.300, database=None, tesla=3.0):
        """Per-metabolite T1 priors pooled from the typed literature tables, in seconds.

        Metabolites with pooled rows at `tesla` (the tables Database.T1_SOURCES lists
        per field, gathered under Database.T1_ALIASES) get Normal(mean, between-study
        spread); without rows at that field they take the 3 T pool (stated), and
        without any rows the Fixed stated assumption.

        Args:
            metabolites: Names to include.
            fallback_s: T1 for metabolites without pooled data.
            database: Database instance; defaults to the packaged one.
            tesla: The field whose tables to pool.

        Returns:
            Params mapping each metabolite to its T1 leaf.
        """
        db = database if database is not None else Database()
        leaves = {}
        for name in metabolites:
            pooled = db.pooled_t1(name, tesla=tesla)
            if pooled is None and float(tesla) != 3.0:
                pooled = db.pooled_t1(name, tesla=3.0)
            leaves[name] = Normal(pooled.mean_s, pooled.sd_s) if pooled else Fixed(fallback_s)
        return Params(leaves)

    #***********************#
    #   simple relaxation   #
    #***********************#
    @staticmethod
    def simple_relaxation(database=None):
        """Fixed T1 and T2 leaves from the stated simple assumptions."""
        db = database if database is not None else Database()
        return Params(t2_s=Fixed(db.t2(source="simple")), t1_s=Fixed(db.t1(source="simple")))
