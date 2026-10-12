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

from mrsandbox.database import (concentrations, deuterium, nuclei, relaxation, shifts,
                                spinsystems)
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

    def spin(self, nucleus="1H"):
        """Nuclear spin quantum number I."""
        return self._lookup(nuclei.SPIN, nucleus, "nucleus")

    #********************#
    #   spin systems     #
    #********************#
    SPIN_SYSTEMS = {"govindaraju": spinsystems.GOVINDARAJU, "govind_2015": spinsystems.GOVIND_2015,
                    "degraaf_2019": spinsystems.DEGRAAF_2019, "vespa": spinsystems.VESPA,
                    "dmi": deuterium.DMI}

    def metabolites(self, source="govindaraju"):
        """Names of all metabolites with a spin system in one source table (the 1H
        Govindaraju set by default; "dmi" is the deuterium set)."""
        return tuple(sorted(self._lookup(self.SPIN_SYSTEMS, source, "spin system source")))

    def spin_system(self, metabolite, source="govindaraju"):
        """Spin system record of one metabolite: its groups (spins, j triplets, scale)
        and, for a table of another nucleus, the nucleus and its relaxation."""
        table = self._lookup(self.SPIN_SYSTEMS, source, "spin system source")
        return copy.deepcopy(self._lookup(table, metabolite, f"metabolite in {source}"))

    #**********************#
    #   concentrations     #
    #**********************#
    def shift_change(self, metabolite, shift_ppm, temperature_c, ph, within_ppm=0.05):
        """How far a proton of `metabolite` typed at `shift_ppm` (37 C, pH 7) moves at
        another temperature and pH, in ppm: the fitted function of the WERMTER_2026
        group whose own shift at 37 C and pH 7 lies nearest, within `within_ppm`,
        evaluated at the conditions minus at the reference. Zero where the table has
        no such group (compounds it did not measure, the phosphorus dummies)."""
        group = self._shift_group(metabolite, shift_ppm, within_ppm)
        if group is None or (temperature_c == 37.0 and ph == 7.0):
            return 0.0
        abc = shifts.WERMTER_2026[metabolite][group]
        return self._titration(abc, temperature_c, ph) - self._titration(abc, 37.0, 7.0)

    def coupling_change(self, metabolite, shift_a_ppm, shift_b_ppm, temperature_c, ph):
        """How far the coupling between two protons of `metabolite`, named by their
        typed shifts, moves at another temperature and pH, in Hz: the WERMTER_2026_J
        function of the coupling between the two protons' groups, at the conditions
        minus at 37 C and pH 7. Zero where the table has none (it holds what the
        authors could read off the multiplets: NAA, GABA's 2-3, Asp, Lac)."""
        table = shifts.WERMTER_2026_J.get(metabolite)
        groups = [self._shift_group(metabolite, shift) for shift in (shift_a_ppm, shift_b_ppm)]
        if (table is None or None in groups or (temperature_c == 37.0 and ph == 7.0)
                or not all("C" in group for group in groups)):  # an NH has no carbon number
            return 0.0
        # a group's number: its carbon and the prime of a CH2's second proton
        numbers = [group[:group.index("C")] for group in groups]
        for label, abc in table.items():
            for coupling in label.replace(" ", "").split(","):
                if sorted(coupling.split("-")) == sorted(numbers):
                    return self._titration(abc, temperature_c, ph) - self._titration(abc, 37.0, 7.0)
        return 0.0

    @staticmethod
    def _titration(abc, temperature_c, ph):
        """a + (b - a) / (1 + 10^(c - pH)) with a, b, c quadratic in the Celsius
        temperature, the model function of the WERMTER_2026 tables."""
        a, b, c = (p[0] + p[1] * temperature_c + p[2] * temperature_c ** 2 for p in abc)
        return a + (b - a) / (1.0 + 10.0 ** (c - ph))

    def _shift_group(self, metabolite, shift_ppm, within_ppm=0.05):
        """The WERMTER_2026 group of `metabolite` whose shift at 37 C and pH 7 lies
        nearest `shift_ppm`, within `within_ppm`; None without one."""
        groups = shifts.WERMTER_2026.get(metabolite)
        if groups is None:
            return None
        at = {name: self._titration(abc, 37.0, 7.0) for name, abc in groups.items()}
        nearest = min(at, key=lambda name: abs(at[name] - shift_ppm))
        return nearest if abs(at[nearest] - shift_ppm) <= within_ppm else None

    def concentration(self, metabolite, source="de_graaf"):
        """Concentration entry of one metabolite from one source, exactly as published."""
        table = self._source(concentrations, source)
        return self._lookup(table, metabolite, f"metabolite in concentrations.{source.upper()}")

    #****************#
    #   relaxation   #
    #****************#
    def t2(self, source="simple"):
        """T2 (seconds) of the simple card: one value for every metabolite."""
        return self._source(relaxation, source)["t2_s"]

    def t1(self, source="simple"):
        """T1 (seconds) of the simple card: one value for every metabolite."""
        return self._source(relaxation, source)["t1_s"]

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

    #***********#
    #   cards   #
    #***********#
    @staticmethod
    def cards():
        """Every typed table of the literature modules as a card, {"module", "name", "source",
        "data", "units", "text"}: the Source, Data and Units fields read off the comment block
        above the table (each continued on the lines after it), "text" the table as typed,
        its notes beside the values kept."""
        import ast
        import pathlib
        import re

        cards = []
        for path in sorted(pathlib.Path(__file__).parent.glob("*.py")):
            lines = path.read_text().splitlines()
            for node in ast.parse("\n".join(lines)).body:
                target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                          else node.target if isinstance(node, ast.AnnAssign) else None)
                if not isinstance(target, ast.Name) or not target.id.isupper():
                    continue
                above, k = [], node.lineno - 2
                while k >= 0 and lines[k].startswith("#") and not lines[k].startswith("#*"):
                    above.insert(0, lines[k][1:].strip())
                    k -= 1
                fields, key = {}, None
                for line in above:
                    found = re.match(r"(Source|Data|Units):\s*(.*)", line)
                    if found:
                        key = found.group(1).lower()
                        fields[key] = found.group(2)
                    elif key:
                        fields[key] += " " + line
                if "source" in fields:
                    cards.append({"module": path.stem, "name": target.id,
                                  **{k: fields.get(k, "") for k in ("source", "data", "units")},
                                  "text": "\n".join(lines[node.lineno - 1:node.end_lineno])})
        return cards

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
