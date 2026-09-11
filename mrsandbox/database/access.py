####################################################################################################
#                                            access.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The value store: literature values live in sibling modules as plain, source named       #
#          Python dicts (readable, citable, diffable); this class is the one typed access path so  #
#          downstream code never touches raw dicts. Names are canonical and exact: compound        #
#          totals (tCr, tNAA, Glx) are distinct keys from single molecules and never               #
#          interchangeable.                                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import copy
import math

from mrsandbox.database import concentrations, nuclei, relaxation, spinsystems
from mrsandbox.database.gudmundson import PooledValue


#**************************************************************************************************#
#                                          Class Database                                          #
#**************************************************************************************************#
#                                                                                                  #
# Typed access to the literature tables, one method per physical quantity.                         #
#                                                                                                  #
#**************************************************************************************************#
class Database:
    """Typed access to the literature tables, one method per physical quantity."""

    # T1 pooling map from single molecules to the resonances the literature actually
    # measures. Unlike concentrations, in vivo T1 exists only for composite resonances
    # (tCr, choline compounds, NA as NAA plus NAAG, Glx), so applying a composite's T1
    # to its constituents is the community approximation, made explicit here.
    T1_ALIASES = {
        "NAA": ("NAA", "NA"),
        "NAAG": ("NAAG", "NA"),
        "Cr": ("tCr", "tCr_CH3", "tCr_CH2"),
        "PCr": ("tCr", "tCr_CH3", "tCr_CH2"),
        "GPC": ("Cho",),
        "PCho": ("Cho",),
        "Cho": ("Cho",),
        "Ins": ("Ins", "Ins_3.57", "Ins_3.66"),
        "Glu": ("Glu",),
        "Gln": ("Gln", "Glx"),
    }

    # the typed T1 tables per field; a metabolite absent from a field's tables falls
    # back to the 3 T pool in the priors (stated)
    T1_SOURCES = {
        1.5: (relaxation.ETHOFER_1P5, relaxation.RUTGERS),
        3.0: (relaxation.MLYNARIK, relaxation.ETHOFER, relaxation.TRABER),
        7.0: (relaxation.XIN,),
    }

    #**************#
    #   nuclei     #
    #**************#
    def gamma(self, nucleus="1H"):
        """Gyromagnetic ratio gamma-bar in MHz/T."""
        return self._lookup(nuclei.GAMMA_MHZ_PER_T, nucleus, "nucleus")

    #********************#
    #   spin systems     #
    #********************#
    def metabolites(self):
        """Names of all metabolites with a spin system."""
        return tuple(sorted(spinsystems.GOVINDARAJU))

    def spin_system(self, metabolite):
        """Spin system record of one metabolite: its groups (spins, j triplets, scale)."""
        return copy.deepcopy(self._lookup(spinsystems.GOVINDARAJU, metabolite, "metabolite"))

    #**********************#
    #   concentrations     #
    #**********************#
    def concentration(self, metabolite, source="de_graaf"):
        """Concentration entry of one metabolite from one source, exactly as published."""
        table = self._source(concentrations, source)
        return self._lookup(table, metabolite, f"metabolite in concentrations.{source.upper()}")

    #****************#
    #   relaxation   #
    #****************#
    def t2(self, metabolite=None, source="simple"):
        """T2 (seconds) from one source; the simple source is one value for every metabolite."""
        table = self._source(relaxation, source)
        if "t2_s" in table:
            return table["t2_s"]
        return self._lookup(table, metabolite, f"metabolite in relaxation.{source.upper()}")

    def t1(self, metabolite=None, source="simple"):
        """T1 (seconds) from one source; the simple source is one value for every metabolite."""
        table = self._source(relaxation, source)
        if "t1_s" in table:
            return table["t1_s"]
        return self._lookup(table, metabolite, f"metabolite in relaxation.{source.upper()}")

    def pooled_t1(self, metabolite, tesla=3.0):
        """DerSimonian and Laird random-effects pool of the typed literature T1 rows at
        one field.

        Rows come from the tables T1_SOURCES lists for `tesla` (Mlynarik 2001, Ethofer
        2003, and Traber 2004 at 3 T; Ethofer's 1.5 T rows and Rutgers 2003 at 1.5 T;
        Xin 2013 at 7 T), gathered under T1_ALIASES; a printed standard deviation
        becomes a standard error through sqrt(n). The returned sd is the between-study
        spread.

        Returns:
            PooledValue(mean_s, sd_s, n_studies), or None when no rows match.
        """
        names = self.T1_ALIASES.get(metabolite, (metabolite,))
        y, v = [], []
        for source in self.T1_SOURCES.get(float(tesla), ()):
            for rows in source["t1_s"].values():
                for name in names:
                    if name in rows:
                        mean, error, n = rows[name]
                        se = error if source["error"] == "se" else error / math.sqrt(n)
                        y.append(mean)
                        v.append(max(se, 1e-3) ** 2)
        if not y:
            return None
        return PooledValue.from_studies(y, v)

    #**********************#
    #   lookup helpers     #
    #**********************#
    @staticmethod
    def _source(module, name):
        """Resolve a source named dict (case insensitive) from a table module."""
        try:
            return getattr(module, name.upper())
        except AttributeError:
            known = ", ".join(n.lower() for n in vars(module) if n.isupper())
            raise KeyError(f"unknown source {name!r}; known: {known}") from None

    @staticmethod
    def _lookup(table, key, kind):
        try:
            return table[key]
        except KeyError:
            raise KeyError(f"unknown {kind} {key!r}; known: {', '.join(map(str, table))}") from None
