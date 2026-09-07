####################################################################################################
#                                          gudmundson.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Query loader for the Gudmundson MRS database (agudmundson/mrs-database), pinned as a    #
#          git submodule under externals/. Serves per-study rows with their full reference         #
#          metadata merged in (field strength, sequence, cohort, tissue fractions), exactly as     #
#          published; pooling across studies is code, never data.                                  #
#                                                                                                  #
#          Gudmundson, A. T., Koo, A., Virovka, A., Amirault, A. L., Soo, M., Cho, J. H.,          #
#          Oeltzschner, G., Edden, R. A. E., Stark, C. Meta-analysis and open-source database      #
#          for in vivo brain Magnetic Resonance spectroscopy in health and disease.                #
#          Analytical Biochemistry 676 (2023) 115227.                                              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import csv
import functools
import math
import pathlib
import typing


#**************************************************************************************************#
#                                         Class Gudmundson                                         #
#**************************************************************************************************#
#                                                                                                  #
# Row-level queries over the concentration and T2 tables of the Gudmundson database.               #
#                                                                                                  #
#**************************************************************************************************#
class Gudmundson:
    """Row-level queries over the concentration and T2 tables of the Gudmundson database.

    Each returned row is one published measurement: the value columns (mM, institutional
    units, tCr ratio, with their standard deviations) merged with the study's reference
    metadata (DOI, field strength, localization, echo time, cohort, tissue fractions).
    """

    ROOT = pathlib.Path(__file__).resolve().parents[2] / "externals" / "mrs-database"

    # conservative synonym map for pooling: moiety variants of the SAME molecule only,
    # never compound totals (tCr, tCho and friends stay excluded by the naming policy)
    T2_ALIASES = {
        "NAA": ("NAA", "NAA_CH3"),
        "Cr": ("Cr", "Cre", "Cr_CH3", "Cre_CH3", "Cr_CH2", "Cre_CH2"),
        "PCr": ("PCr", "pCr"),
        "Ins": ("Ins", "mI", "mIns", "Myo"),
        "sIns": ("Scy",),
        "PCho": ("PCh",),
        "GPC": ("GPC",),
        "NAAG": ("NAAG", "naag"),
        "Cho": ("Cho", "Cho_CH3"),
    }

    #******************#
    #   availability   #
    #******************#
    @classmethod
    def available(cls):
        """True when the externals/mrs-database submodule is checked out."""
        return (cls.ROOT / "Values_Concentrations.csv").exists()

    #*************#
    #   queries   #
    #*************#
    def concentrations(self, metabolite, group=None, tesla=None):
        """Concentration rows for one metabolite, optionally filtered by cohort and field."""
        return self._select("Concentrations", metabolite, group, tesla)

    def t2(self, metabolite, group=None, tesla=None):
        """T2 rows for one metabolite, optionally filtered by cohort and field."""
        return self._select("T2", metabolite, group, tesla)

    #*******************#
    #   row selection   #
    #*******************#
    def _select(self, table, metabolite, group, tesla):
        if not self.available():
            raise FileNotFoundError(
                "the Gudmundson database is not checked out; run "
                "`git submodule update --init externals/mrs-database`"
            )
        references = self._table(str(self.ROOT / f"References_{table}.csv"), key=True)
        rows = []
        for row in self._table(str(self.ROOT / f"Values_{table}.csv"), key=False):
            if row.get("Metabolite") != metabolite:
                continue
            merged = {**references.get((row.get("Reference"), row.get("ID")), {}), **row}
            if group is not None and merged.get("Group") != group:
                continue
            if tesla is not None and merged.get("Tesla") != str(tesla):
                continue
            rows.append(merged)
        return rows

    #***************#
    #   pooling     #
    #***************#
    def pooled_t2(self, metabolite, tesla=3, groups=("Healthy", "Control"), subject="Human"):
        """DerSimonian and Laird random-effects pool of the T2 rows, in seconds.

        Rows are gathered under the metabolite's same-molecule aliases, filtered to the
        cohorts, field, and subject given. Study variances use the standard error
        (StdDev over sqrt(N_Total)) where the reference carries N, else the StdDev
        itself. The returned sd is the between-study spread (sqrt of tau squared, or the
        plain spread of study means when tau vanishes), the honest width for a prior.

        Returns:
            PooledValue(mean_s, sd_s, n_studies), or None when no rows match.
        """
        names = self.T2_ALIASES.get(metabolite, (metabolite,))
        rows = [row for name in names for row in self.t2(name, tesla=tesla)
                if row.get("Group") in groups and row.get("Subject") == subject
                and row.get("T2") and row.get("StdDev")]
        if not rows:
            return None
        y = [float(row["T2"]) / 1000.0 for row in rows]
        v = []
        for row in rows:
            sd = float(row["StdDev"]) / 1000.0
            n = row.get("N_Total")
            se = sd / math.sqrt(float(n)) if n not in (None, "") else sd
            v.append(max(se, 1e-4) ** 2)
        return PooledValue.from_studies(y, v)

    def pooled_concentration(self, metabolite, groups=("Healthy", "Control"), tesla=None,
                             subject="Human"):
        """DerSimonian and Laird random-effects pool of the concentration rows, in mM.

        Rows are gathered under the metabolite's same-molecule aliases and filtered to
        the cohorts given (pass a clinical group name for disease priors; the Group
        strings are the database's own). Only rows carrying molar units (mM_u with an
        mM_std) enter; the returned sd is the between-study spread.

        Returns:
            PooledValue(mean_s, sd_s, n_studies) with values in mM, or None.
        """
        names = self.T2_ALIASES.get(metabolite, (metabolite,))
        wanted = (groups,) if isinstance(groups, str) else tuple(groups)
        rows = [row for name in names for row in self.concentrations(name, tesla=tesla)
                if row.get("Group") in wanted and row.get("Subject", "Human") in (subject, "", None)
                and row.get("mM_u") and row.get("mM_std")]
        if not rows:
            return None
        y = [float(row["mM_u"]) for row in rows]
        v = []
        for row in rows:
            sd = float(row["mM_std"])
            n = row.get("N_Total")
            se = sd / math.sqrt(float(n)) if n not in (None, "") else sd
            v.append(max(se, 1e-3) ** 2)
        return PooledValue.from_studies(y, v)

    #*****************#
    #   csv parsing   #
    #*****************#
    @staticmethod
    @functools.cache
    def _table(path, key):
        """Parse one CSV; keyed tables map (Reference, ID) to the row for merging."""
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
        if key:
            return {(row.get("Reference"), row.get("ID")): row for row in rows}
        return rows


#**************************************************************************************************#
#                                        Class PooledValue                                         #
#**************************************************************************************************#
#                                                                                                  #
# A pooled literature value with its between-study spread.                                         #
#                                                                                                  #
#**************************************************************************************************#
class PooledValue(typing.NamedTuple):
    """A pooled literature value with its between-study spread."""

    mean_s: float
    sd_s: float
    n_studies: int

    #***********#
    #   pool    #
    #***********#
    @classmethod
    def from_studies(cls, y, v):
        """DerSimonian and Laird random-effects pool of study means `y` and variances `v`.

        The returned sd is the between-study spread (sqrt of tau squared, or the plain
        spread of study means when tau vanishes, floored at 5 percent of the mean), the
        honest width for a prior rather than the standard error of the pooled mean.
        """
        w = [1.0 / vi for vi in v]
        mean_fixed = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
        q = sum(wi * (yi - mean_fixed) ** 2 for wi, yi in zip(w, y))
        denom = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
        tau2 = max(0.0, (q - (len(y) - 1)) / denom) if denom > 0 else 0.0
        w_star = [1.0 / (vi + tau2) for vi in v]
        mean = sum(wi * yi for wi, yi in zip(w_star, y)) / sum(w_star)
        spread = math.sqrt(tau2) if tau2 > 0 else (
            math.sqrt(sum((yi - mean) ** 2 for yi in y) / max(len(y) - 1, 1))
        )
        return cls(mean_s=mean, sd_s=max(spread, 0.05 * mean), n_studies=len(y))
