####################################################################################################
#                                             cohort.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 26/09/26                                                                                #
#                                                                                                  #
# Purpose: A cohort: subjects measured on one dataset, each a scanner profile of its own raw run   #
#          with the voxel's place and the brain that stands in, so a draw lands every measured     #
#          value on one real subject at once; and the package's cohorts as its measured            #
#          database, pooled for a scanner by the conditions it matches.                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import functools
import json
import pathlib
import re

import numpy as np

from mrsandbox.params import Empirical
from mrsandbox.phantom import Brain
from mrsandbox.scan.calibrations import Calibration
from mrsandbox.scan.profile import ScannerProfile

COHORTS = pathlib.Path(__file__).resolve().parent.parent / "resources" / "cohorts"
# the operator's own cohorts, joined to the pool beside the package's (Cohort.save a cohort
# here and it is pooled and listed; forget() in a running process)
OWN = pathlib.Path.home() / "MRSandbox" / "cohorts"
# how far a row's field may sit from the scanner's and still be its field (the Siemens 3 T run
# at 2.89)
FIELD_T = 0.25
# the leaves of the receive chain, which the receive array's channel count is a condition of;
# the suppression's residual, the delay or the eddy transient do not change with the array
RECEIVE = ("noise.sigma", "coils.loaded.own_noise", "coils.partner_correlation")
# the person a query asks for when it names no one: a healthy adult, the usual volunteer; children
# and patients enter the pool only when asked. A subject's traits are conditions of the rows it
# carries (the water's relaxation, its breathing and motion), not of the scanner's
ADULT = {"age_y": (18.0, None), "health": "healthy"}
# a number asked of a trait that varies continuously means about that value: this far either
# side (stated: a registered age of 45 asks for people of 35 to 55)
NEAR = {"age_y": 10.0}
# the measurement methods that read a leaf (and its tissues' leaves) as the tissue has it, a
# map row's method its sequence family: methods part far beyond the spread between people (white
# matter T1 over 30 percent apart, Stikov et al. 2015, Magn Reson Med 73:514), so a row of
# another method joins the pool only when asked, or when no faithful row holds, named
FAITHFUL = {
    "water.t1_s": ("inversion recovery", "MP2RAGE"),
    "water.t2_s": ("EPG multi-echo spin echo",),
}
# the tissue's relaxation (the water's, the metabolites' T2), which the field sets (T1 lengthens
# and T2 shortens with it) and no scanner's vendor, model or coil: its rows pool at the scanner's
# field alone, whatever system read them, and with none there the leaf stays with the
# literature's own table
RELAXATION = ("water.t1_s", "water.t2_s", "t2")


#**************************************************************************************************#
#                                           Class Member                                           #
#**************************************************************************************************#
#                                                                                                  #
# One subject of a cohort: its profile, its voxel and the brain that stands in for it.             #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Member:
    """One subject of a cohort: its profile, its voxel and the brain that stands in for it.

    Args:
        subject: The subject's name in the dataset.
        profile: The ScannerProfile read off the subject's own raw run.
        position_mm, size_mm: The voxel as the dataset placed it.
        brain: The name of the package brain standing in (none of the read datasets ships
            an image the package can use).
        traits: What the dataset states of the person: "age_y" in years, "sex" ("F", "M"),
            "health" ("healthy" or the condition by name), and any other it states; a
            number the dataset gives only as a range (all adults) is that range, (low,
            high) with None open. A trait not stated is unknown and meets no query of it.
        taken: The traits a rule gave where the dataset does not state them (a volunteer's
            health taken as healthy), shown as taken, not stated.
    """
    subject: str
    profile: ScannerProfile
    position_mm: tuple
    size_mm: tuple
    brain: str = "icbm152"
    traits: dict = dataclasses.field(default_factory=dict)
    taken: tuple = ()

    #***********#
    #   fixed   #
    #***********#
    def fixed(self):
        """The profile with every record's prior closed on its value: a member is one
        subject, not a range about one. A failed reading is dropped: its leaf falls to
        the pooled record."""
        records = tuple(dataclasses.replace(
            record, prior=(record.value, record.value),
            source={**record.source,
                    "dataset": f"subject {self.subject} of {record.source['dataset']}"})
            for record in self.profile.records if not record.failed)
        return dataclasses.replace(self.profile, records=records)

    #**********#
    #   scan   #
    #**********#
    def scan(self, sequence, **kwargs):
        """A VirtualScan of this subject: its scanner's profile with its own values fixed,
        its brain, the leaves it has no reading of pooled over people like it (its traits);
        `kwargs` are VirtualScan's."""
        return self.fixed().scan(Brain.load(self.brain), sequence,
                                 **{"subject": dict(self.traits) or None, **kwargs})


#**************************************************************************************************#
#                                           Class Cohort                                           #
#**************************************************************************************************#
#                                                                                                  #
# Subjects measured on one dataset, drawn one at a time.                                           #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Cohort:
    """Subjects measured on one dataset, drawn one at a time.

    Args:
        name: The cohort's name (the dataset and its site).
        members: The Members, one per subject.
    """
    name: str
    members: tuple

    #***********#
    #   label   #
    #***********#
    def label(self, dataset=None):
        """The cohort in one line, the same shape for every cohort: "<dataset> · <vendor model>
        <field> T · <sequence> · N subjects", the conditions read off its members' records
        (`dataset` in place of the name)."""
        records = [r for m in self.members for r in m.profile.records]
        tidy = {"ge": "GE", "TrioTim": "Trio", "Prisma_fit": "Prisma", "Discovery MR750": "MR750",
                "Investigational_Device_7T": "", "Magnetom 7T": "Magnetom", "CBICMR30X": ""}
        vendors = sorted({tidy.get(r.scanner["vendor"], str(r.scanner["vendor"]).title())
                          for r in records if r.scanner.get("vendor")})
        models = sorted({tidy.get(r.scanner["model"], str(r.scanner["model"]))
                         for r in records if r.scanner.get("model")} - {""})
        # to the nearest half tesla: 1.5 stays 1.5, a Siemens 3 T's 2.89 reads 3
        fields = sorted({round(2.0 * r.scanner["field_t"]) / 2.0 for r in records
                         if r.scanner.get("field_t") is not None})
        families = sorted({r.sequence.get("family", "") for r in records
                           if r.sequence.get("family")}) or (
            ["fMRI"] if any("series" in str(r.sequence.get("name", ""))
                            or "physio" in str(r.sequence.get("name", "")) for r in records)
            else ["?"])
        names = {"Press": "PRESS", "SemiLaser": "semi-LASER", "Steam": "STEAM"}
        system = " ".join(vendors + models) or "unstated system"
        return (f"{dataset or self.name} · {system} "
                f"{' / '.join(f'{f:g}' for f in fields) or '?'} T · "
                f"{', '.join(names.get(f, f) for f in families)} · {len(self.members)} subjects")

    #**********#
    #   draw   #
    #**********#
    def draw(self, seed=None):
        """One member, by the seed."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        return self.members[int(rng.integers(len(self.members)))]

    #*************#
    #   checked   #
    #*************#
    def checked(self):
        """The cohort with every run whose noise reads over twice the cohort's median failed in
        its noise and width readings: a failed acquisition (motion, a lost coil), not the
        scanner's noise or shim (stated). The noise reading is taken against each run's own twin,
        so runs of other settings compare."""
        noise = {k: r.value for k, m in enumerate(self.members) for r in m.profile.records
                 if r.leaf == "noise.sigma" and not r.failed}
        if not noise:
            return self
        median = float(np.median(list(noise.values())))
        rule = (f"the run's noise reads over twice its cohort's median ({median:.3g}): a failed "
                f"acquisition")
        return dataclasses.replace(self, members=tuple(
            dataclasses.replace(m, profile=dataclasses.replace(m.profile, records=tuple(
                dataclasses.replace(r, failed=rule)
                if r.leaf in ("noise.sigma", "lineshape.gauss_hz") and not r.failed else r
                for r in m.profile.records)))
            if noise.get(k, 0.0) > 2.0 * median else m for k, m in enumerate(self.members)))

    #**********#
    #   save   #
    #**********#
    def save(self, path):
        """The cohort as one JSON file: its members' profiles (ScannerProfile.to_dict), voxels
        and brains; numbers with their conditions, no raw data."""
        with open(path, "w") as file:
            json.dump({"name": self.name,
                       "members": [{"subject": m.subject, "position_mm": list(m.position_mm),
                                    "size_mm": list(m.size_mm), "brain": m.brain,
                                    "traits": m.traits, "taken": list(m.taken),
                                    "profile": m.profile.to_dict()} for m in self.members]},
                      file, indent=1)

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """The cohort `save` wrote."""
        with open(path) as file:
            stored = json.load(file)
        return cls(stored["name"], tuple(
            Member(m["subject"], ScannerProfile.from_dict(m["profile"]), tuple(m["position_mm"]),
                   tuple(m["size_mm"]), m["brain"],
                   {k: tuple(v) if isinstance(v, list) else v
                    for k, v in m.get("traits", {}).items()}, tuple(m.get("taken", ())))
            for m in stored["members"]))

    #***********#
    #   named   #
    #***********#
    @classmethod
    def named(cls, name):
        """One of the package's cohorts (mrsandbox/resources/cohorts) or the operator's
        own (~/MRSandbox/cohorts), by name; the package's wins the name."""
        path = COHORTS / f"{name}.json"
        return cls.load(path if path.exists() else OWN / f"{name}.json")

    @staticmethod
    def names():
        """The package's cohorts and the operator's own."""
        return sorted({path.stem for folder in (COHORTS, OWN) if folder.is_dir()
                       for path in folder.glob("*.json")})


#*******************#
#   the database    #
#*******************#
@functools.lru_cache(maxsize=None)
def package():
    """The package's cohorts and the operator's own, loaded once: the measured database."""
    return tuple(Cohort.named(name) for name in Cohort.names())


def rows(leaf, within=()):
    """Every reading of `leaf` in the package's cohorts (those named in `within`, else all)
    that did not fail, as [(cohort name, subject, Calibration)]."""
    return [(cohort.name, member.subject, record)
            for cohort in package() if not within or cohort.name in within
            for member in cohort.members for record in member.profile.records
            if record.leaf == leaf and not record.failed]


def leaves():
    """The leaves the package's cohorts hold readings of."""
    return sorted({record.leaf for cohort in package() for member in cohort.members
                   for record in member.profile.records})


def stated(trait):
    """The values of a person's trait the package's cohorts state, sorted (the conditions the
    database holds, for one): what a query can ask of it and find."""
    return sorted({str(member.traits[trait]) for cohort in package() for member in cohort.members
                   if trait in member.traits})


def table():
    """The measured database as one pandas DataFrame, a row per reading, failed ones included
    (`failed` their reason, empty where they hold): the dataset and subject, the person (age_y a
    stated number, age_low and age_high the range it lies in, open ends NaN; sex, health, the
    traits a rule gave), the leaf, its value and prior, and what it was read under (method, the
    sequence, field, vendor, model, channels, date, source). Query it as any DataFrame, e.g.
    table().query("leaf == 'water.t2_s.wm' and age_low >= 40"). Needs pandas."""
    import pandas

    found = []
    for cohort in package():
        for member in cohort.members:
            age = member.traits.get("age_y")
            low, high = age if _span(age) else (age, age)
            for record in member.profile.records:
                found.append({
                    "dataset": cohort.name, "subject": member.subject,
                    "age_y": age if not _span(age) else None, "age_low": low, "age_high": high,
                    "sex": member.traits.get("sex"), "health": member.traits.get("health"),
                    "taken": ", ".join(member.taken), "leaf": record.leaf,
                    "value": record.value, "low": record.prior[0], "high": record.prior[1],
                    "failed": record.failed, "method": record.sequence.get("family") or None,
                    "sequence": record.sequence.get("name"),
                    "field_t": record.scanner.get("field_t"),
                    "vendor": record.scanner.get("vendor"), "model": record.scanner.get("model"),
                    "channels": ", ".join(str(c) for c in record.coil.get("channels") or ()),
                    "date": record.date, "source": record.source.get("dataset")})
    frame = pandas.DataFrame(found)
    return frame.astype({key: float for key in ("age_y", "age_low", "age_high", "value", "low",
                                                "high", "field_t")})


def _plain(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _under(leaf, keys):
    """The key of `keys` that `leaf` is or lies under ("water.t2_s.wm" under "water.t2_s"), or
    None."""
    return next((key for key in keys if leaf == key or leaf.startswith(f"{key}.")), None)


def _matches(record, field_t, vendor, model, channels, method=None):
    """Whether a row was read under the conditions asked (None asks nothing; `method` the
    methods asked); a row that does not state a condition does not fail it."""
    if (field_t is not None and record.scanner.get("field_t") is not None
            and abs(record.scanner["field_t"] - field_t) > FIELD_T):
        return False
    if vendor is not None and record.scanner.get("vendor") and record.scanner["vendor"] != vendor:
        return False
    if (model is not None and record.scanner.get("model")
            and _plain(model) not in _plain(record.scanner["model"])
            and _plain(record.scanner["model"]) not in _plain(model)):
        return False
    if (channels is not None and record.coil.get("channels")
            and channels not in record.coil["channels"]):
        return False
    if (method is not None and record.sequence.get("family")
            and record.sequence["family"] not in method):
        return False
    return True


def _span(value):
    """Whether a trait's value is a range, (low, high) of numbers with None open."""
    return (isinstance(value, tuple) and len(value) == 2
            and all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
                    for v in value))


def _suits(traits, asked):
    """Whether a person's traits meet the traits asked: a number, or a range the dataset
    states, inside the range asked; a word the word asked or one of the words asked (a
    tuple). A trait not stated meets nothing asked of it."""
    for key, want in asked.items():
        have = traits.get(key)
        if have is None:
            return False
        if _span(want):
            low, high = have if _span(have) else (have, have)
            if want[0] is not None and (low is None or low < want[0]):
                return False
            if want[1] is not None and (high is None or high > want[1]):
                return False
        elif have not in (want if isinstance(want, tuple) else (want,)):
            return False
    return True


#**************#
#   clusters   #
#**************#
def _clusters(groups):
    """How the datasets of a pool weigh, {dataset: its subjects' values}, as a reference range
    over individuals from several studies is built (Siegel, Murad and Chu 2021, Res Synth
    Methods 12:148): the spread of subjects within the datasets (their pooled variance), the
    spread of the datasets' means beyond what their sizes explain (the between-dataset
    variance, by restricted maximum likelihood as Langan et al. 2019, Res Synth Methods
    10:83, recommend over DerSimonian and Laird), and each dataset's share of the draws,
    1 / (between + within / n) normalized: equal shares where the datasets disagree far
    beyond their sizes, shares by size where they agree. Returns (within, between,
    {dataset: share})."""
    names = sorted(groups)
    sizes = np.array([len(groups[name]) for name in names], dtype=float)
    means = np.array([np.mean(groups[name]) for name in names])
    freedom = sum(len(groups[name]) - 1 for name in names)
    within = (sum((len(groups[name]) - 1) * np.var(groups[name], ddof=1)
                  for name in names if len(groups[name]) > 1) / freedom) if freedom else 0.0
    variances = within / sizes
    between = 0.0
    if len(names) > 1:
        # a dataset of one subject, or of subjects who all read alike, still has some variance
        floor = 1e-12 * max(float(np.var(means)), 1e-30)
        variances = np.maximum(variances, floor)
        between = max(float(np.var(means, ddof=1) - variances.mean()), 0.0)
        for _ in range(200):
            weights = 1.0 / (variances + between)
            centre = np.sum(weights * means) / np.sum(weights)
            estimate = max(float(np.sum(weights ** 2 * ((means - centre) ** 2 - variances))
                                 / np.sum(weights ** 2) + 1.0 / np.sum(weights)), 0.0)
            converged = abs(estimate - between) <= 1e-10 * (between + floor)
            between = estimate
            if converged:
                break
    total = variances + between
    weights = 1.0 / total if total.min() > 0.0 else np.ones(len(names))
    return float(within), between, dict(zip(names, weights / weights.sum()))


#************#
#   pooled   #
#************#
def forget():
    """Drop what was pooled and matched so far: for a test that stands in another database, or
    a package whose cohorts were rebuilt in this process."""
    _pooled.cache_clear()
    _holding.cache_clear()


def pooled(leaf, scanner, channels=None, within=(), subject=None, method=None):
    """See _pool: the same, computed once per leaf, conditions and database (the package's
    rows do not change; a database stood in for it, as a test does, is another key).
    `subject` names the person's traits asked, over the healthy adult (ADULT) it leaves
    unnamed; a trait asked as None asks nothing of it, a number of a trait in NEAR asks for
    about that number. `method` names the measurement methods asked (one or several), over
    the leaf's faithful ones (FAITHFUL) it leaves unnamed; empty asks any."""
    asked = {key: (value - NEAR[key], value + NEAR[key])
             if key in NEAR and isinstance(value, (int, float)) else value
             for key, value in {**ADULT, **(subject or {})}.items() if value is not None}
    if method is None:
        method = FAITHFUL.get(_under(leaf, FAITHFUL), ())
    methods = (method,) if isinstance(method, str) else tuple(method)
    return _pooled(leaf, float(scanner.b0_t), scanner.vendor, scanner.model, channels,
                   tuple(within), tuple(sorted(asked.items())), methods or None, id(package()))


@functools.lru_cache(maxsize=4096)
def _pooled(leaf, b0_t, vendor, model, channels, within, subject, methods, database):
    scanner = type("Asked", (), {"b0_t": b0_t, "vendor": vendor, "model": model})
    return _pool(leaf, scanner, channels, within, dict(subject), methods)


def _pool(leaf, scanner, channels=None, within=(), subject=None, methods=None):
    """The Calibration of `leaf` for `scanner` (a hardware.Scanner) pooled over the package's
    cohort rows read under its conditions: the median over subjects is the value, their
    5th to 95th percentile the prior (the range over fewer than five, one row's own prior
    alone). The conditions are as specific as the scanner is: its field, its vendor and
    model where it states them, `methods` the measurement methods asked of the rows (None
    any), and `channels`, the receive array's count, where given and the leaf is the
    receive chain's (RECEIVE); a row the subject carries is also asked of the person's
    traits (`subject`, see _suits), and a tissue's relaxation (RELAXATION) asks no vendor,
    model or coil. When nothing matches, the fewest conditions give way, the model the
    first to, then the method, the person's age, sex and other traits, then channels,
    vendor and the field itself, keeping whatever can still hold once a dearer one has
    gone (the departure names it); the person's health never gives way, nor the field of a
    tissue's relaxation, so a condition no rows hold stays with the literature. `within`
    names the cohorts whose rows alone are pooled (one dataset as the scanner's
    calibration); empty pools them all. None when no row of the leaf is there."""
    subject = subject or {}
    found = [(cohort.name, member, record)
             for cohort in package() if not within or cohort.name in within
             for member in cohort.members for record in member.profile.records
             if record.leaf == leaf and not record.failed]
    tissue = _under(leaf, RELAXATION) is not None
    asked = dict(field_t=float(scanner.b0_t), vendor=None if tissue else scanner.vendor,
                 model=None if tissue else scanner.model,
                 channels=channels if leaf in RECEIVE else None, method=methods)
    order = (["model", "method"] + [key for key in ("age_y", "sex") if key in subject]
             + sorted(set(subject) - {"age_y", "sex", "health"}) + ["channels", "vendor"]
             + ([] if tissue else ["field_t"]))
    # the fewest conditions given way, the later in `order` the dearer: a set is tried before
    # every set that drops a dearer condition, so what can hold once a dearer one has gone
    # holds (an adult at another field before a child at this one)
    for mask in range(2 ** len(order)):
        dropped = [key for bit, key in enumerate(order) if mask >> bit & 1]
        if "vendor" in dropped and "model" not in dropped:
            continue                    # a model is its vendor's: it goes with the vendor
        conditions = {key: None if key in dropped else value for key, value in asked.items()}
        traits = {key: value for key, value in subject.items() if key not in dropped}
        matched = [row for row in found if _matches(row[2], **conditions)
                   and (row[2].carried_by != "subject" or _suits(row[1].traits, traits))]
        if matched:
            break
    if not matched:
        return None
    groups = {}
    for name, _, record in matched:
        groups.setdefault(name, []).append(float(record.value))
    within, between, share = _clusters(groups)
    # the values and weights as the record keeps them, so its prior and its draws agree
    pairs = sorted((round(float(record.value), 7), round(share[name] / len(groups[name]), 9),
                    name) for name, _, record in matched)
    values = np.array([v for v, _, _ in pairs])
    population = Empirical(tuple(values), tuple(w for _, w, _ in pairs))
    value = population.value
    percentiles = {}
    if len(values) == 1:
        prior = matched[0][2].prior
    elif len(values) < 5:
        prior = (float(values.min()), float(values.max()))
    else:
        prior = (population.low, population.high)
        percentiles = {str(q): round(population.percentile(q), 7) for q in (5, 25, 50, 75, 95)}
    records = [record for _, _, record in matched]
    cohorts = sorted({name for name, _, _ in matched})

    def one(values):
        return values[0] if len(values) == 1 else "pooled"
    # what the rows were matched on: the conditions asked and kept that at least one row states
    states = {"field_t": lambda r: True, "vendor": lambda r: bool(r.scanner.get("vendor")),
              "model": lambda r: bool(r.scanner.get("model")),
              "channels": lambda r: bool(r.coil.get("channels")),
              "method": lambda r: bool(r.sequence.get("family"))}
    stated = {key: value for key, value in conditions.items()
              if value is not None and any(states[key](r) for r in records)}
    per_cohort = {name: round(float(np.median([r.value for c, _, r in matched if c == name])), 7)
                  for name in cohorts}
    return Calibration(
        leaf=leaf, value=round(value, 7), prior=tuple(round(float(v), 7) for v in prior),
        measured={"statistic": "each dataset a cluster, drawn by its random-effects share, then "
                               "one of its subjects; the value the weighted median, the prior the "
                               "weighted 5th to 95th percentile (scan.cohort.pooled)",
                  "subjects": len(matched), "matched_on": stated, "percentiles": percentiles,
                  "per_cohort_median": per_cohort,
                  "values": [float(v) for v in values],
                  "weights": [w for _, w, _ in pairs],
                  "datasets": [name for _, _, name in pairs],
                  "dataset_share": {name: round(float(s), 6) for name, s in share.items()},
                  "within_dataset_sd": round(float(np.sqrt(within)), 7),
                  "between_dataset_sd": round(float(np.sqrt(between)), 7),
                  "subjects_per_cohort": {name: sum(1 for c, _, _ in matched if c == name)
                                          for name in cohorts},
                  # the person the rows were pooled for, and what of it no rows held
                  **({"subject": traits, "subject_dropped": [k for k in dropped if k in subject]}
                     if records[0].carried_by == "subject" else {}),
                  # the methods the rows were read by, and whether the ones asked gave way
                  **({"methods": sorted({r.sequence.get("family") or "unstated"
                                         for r in records}),
                      "method_dropped": "method" in dropped} if methods else {})},
        source={"dataset": f"pooled over {len(matched)} subjects of {len(cohorts)} "
                           f"cohort{'s' if len(cohorts) > 1 else ''}: "
                           + ", ".join(cohorts), "citation": None, "url": None},
        scanner={"vendor": one(sorted({r.scanner.get("vendor") or "unstated" for r in records})),
                 "model": one(sorted({str(r.scanner.get("model") or "unstated")
                                      for r in records})),
                 "software": None,
                 "field_t": round(float(np.median([r.scanner["field_t"] for r in records
                                                   if r.scanner.get("field_t") is not None]
                                                  or [float("nan")])), 2)},
        coil={"channels": tuple(sorted({int(c) for r in records
                                        for c in (r.coil.get("channels") or ())}))},
        sequence={"name": ", ".join(sorted({str(r.sequence.get("name")) for r in records})),
                  "family": one(sorted({r.sequence.get("family", "")
                                        for r in records})).replace("pooled", "")},
        acquisition={"the subjects' own": True},
        voxel={"the subjects' own": True},
        processing="the rows as their readers made them (scan.calibrate), pooled by "
                   "scan.cohort.pooled",
        date=max(r.date for r in records), carried_by=records[0].carried_by,
        note="the package's measured database pooled for this scanner; a scan's departures name "
             "what it does not match",
    )


def holding(scanner):
    """The names of the package's cohorts with rows of `scanner`'s vendor and model at its
    field (its own systems); with none, those at its field (a generic scanner)."""
    return _holding(float(scanner.b0_t), scanner.vendor, scanner.model, id(package()))


@functools.lru_cache(maxsize=None)
def _holding(b0_t, vendor, model, database):
    scanner = type("Asked", (), {"b0_t": b0_t, "vendor": vendor, "model": model})
    for model in (scanner.model, None):
        found = sorted({cohort.name for cohort in package() for member in cohort.members
                        for record in member.profile.records
                        if not record.failed
                        and _matches(record, float(scanner.b0_t), scanner.vendor, model, None)})
        if found:
            return found
    return sorted({cohort.name for cohort in package() for member in cohort.members
                   for record in member.profile.records
                   if not record.failed
                   and _matches(record, float(scanner.b0_t), None, None, None)})
