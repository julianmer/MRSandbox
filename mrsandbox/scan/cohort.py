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
#          value on one real subject at once (info/HANDOFF.md item 239); and the package's         #
#          cohorts as its measured database, pooled for a scanner by the conditions it matches     #
#          (item 241).                                                                             #
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

from mrsandbox.phantom import Brain
from mrsandbox.scan.calibrations import Calibration
from mrsandbox.scan.profile import ScannerProfile

COHORTS = pathlib.Path(__file__).resolve().parent.parent / "resources" / "cohorts"
# how far a row's field may sit from the scanner's and still be its field (the Siemens 3 T run at 2.89)
FIELD_T = 0.25
# the leaves of the receive chain, which the receive array's channel count is a condition of;
# the suppression's residual, the delay or the eddy transient do not change with the array
RECEIVE = ("noise.sigma", "coils.loaded.own_noise", "coils.partner_correlation")


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
    """
    subject: str
    profile: ScannerProfile
    position_mm: tuple
    size_mm: tuple
    brain: str = "icbm152"

    #***********#
    #   fixed   #
    #***********#
    def fixed(self):
        """The profile with every record's prior closed on its value: a member is one
        subject, not a range about one. A failed reading is dropped: its leaf falls to
        the pooled record."""
        records = tuple(dataclasses.replace(record, prior=(record.value, record.value),
                                            source={**record.source, "dataset": f"subject {self.subject} of {record.source['dataset']}"})
                        for record in self.profile.records if not record.failed)
        return dataclasses.replace(self.profile, records=records)

    #**********#
    #   scan   #
    #**********#
    def scan(self, sequence, **kwargs):
        """A VirtualScan of this subject: its scanner's profile with its own values fixed,
        its brain; `kwargs` are VirtualScan's."""
        return self.fixed().scan(Brain.load(self.brain), sequence, **kwargs)


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
        """The cohort in one line, the same shape for every cohort: "<dataset> · <vendor model> <field> T ·
        <sequence> · N subjects", the conditions read off its members' records (`dataset` in place of
        the name)."""
        records = [r for m in self.members for r in m.profile.records]
        tidy = {"ge": "GE", "TrioTim": "Trio", "Prisma_fit": "Prisma", "Discovery MR750": "MR750", "Investigational_Device_7T": "", "Magnetom 7T": "Magnetom",
                "CBICMR30X": ""}
        vendors = sorted({tidy.get(r.scanner["vendor"], str(r.scanner["vendor"]).title()) for r in records if r.scanner.get("vendor")})
        models = sorted({tidy.get(r.scanner["model"], str(r.scanner["model"])) for r in records if r.scanner.get("model")} - {""})
        fields = sorted({round(r.scanner["field_t"]) for r in records if r.scanner.get("field_t") is not None})
        families = sorted({r.sequence.get("family", "") for r in records if r.sequence.get("family")}) or (
            ["fMRI"] if any("series" in str(r.sequence.get("name", "")) or "physio" in str(r.sequence.get("name", "")) for r in records) else ["?"])
        names = {"Press": "PRESS", "SemiLaser": "semi-LASER", "Steam": "STEAM"}
        system = " ".join(vendors + models) or "unstated system"
        return (f"{dataset or self.name} · {system} {' / '.join(str(f) for f in fields) or '?'} T · "
                f"{', '.join(names.get(f, f) for f in families)} · {len(self.members)} subjects")

    #**********#
    #   draw   #
    #**********#
    def draw(self, seed=None):
        """One member, by the seed."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        return self.members[int(rng.integers(len(self.members)))]

    #**********#
    #   save   #
    #**********#
    def save(self, path):
        """The cohort as one JSON file: its members' profiles (ScannerProfile.to_dict), voxels
        and brains; numbers with their conditions, no raw data."""
        with open(path, "w") as file:
            json.dump({"name": self.name,
                       "members": [{"subject": m.subject, "position_mm": list(m.position_mm), "size_mm": list(m.size_mm),
                                    "brain": m.brain, "profile": m.profile.to_dict()} for m in self.members]}, file, indent=1)

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """The cohort `save` wrote."""
        with open(path) as file:
            stored = json.load(file)
        return cls(stored["name"], tuple(Member(m["subject"], ScannerProfile.from_dict(m["profile"]), tuple(m["position_mm"]),
                                                tuple(m["size_mm"]), m["brain"]) for m in stored["members"]))

    #***********#
    #   named   #
    #***********#
    @classmethod
    def named(cls, name):
        """One of the package's cohorts (mrsandbox/resources/cohorts), by name."""
        return cls.load(COHORTS / f"{name}.json")

    @staticmethod
    def names():
        """The package's cohorts."""
        return sorted(path.stem for path in COHORTS.glob("*.json"))


#*******************#
#   the database    #
#*******************#
@functools.lru_cache(maxsize=None)
def package():
    """The package's cohorts, loaded once: its measured database."""
    return tuple(Cohort.named(name) for name in Cohort.names())


def rows(leaf, within=()):
    """Every reading of `leaf` in the package's cohorts (those named in `within`, else all)
    that did not fail, as [(cohort name, subject, Calibration)]."""
    return [(cohort.name, member.subject, record) for cohort in package() if not within or cohort.name in within
            for member in cohort.members for record in member.profile.records if record.leaf == leaf and not record.failed]


def leaves():
    """The leaves the package's cohorts hold readings of."""
    return sorted({record.leaf for cohort in package() for member in cohort.members for record in member.profile.records})


def _plain(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _matches(record, field_t, vendor, model, channels):
    """Whether a row was read under the conditions asked (None asks nothing); a row that
    does not state a condition does not fail it."""
    if field_t is not None and record.scanner.get("field_t") is not None and abs(record.scanner["field_t"] - field_t) > FIELD_T:
        return False
    if vendor is not None and record.scanner.get("vendor") and record.scanner["vendor"] != vendor:
        return False
    if model is not None and record.scanner.get("model") and _plain(model) not in _plain(record.scanner["model"]) \
            and _plain(record.scanner["model"]) not in _plain(model):
        return False
    if channels is not None and record.coil.get("channels") and channels not in record.coil["channels"]:
        return False
    return True


#************#
#   pooled   #
#************#
def forget():
    """Drop what was pooled and matched so far: for a test that stands in another database, or a package
    whose cohorts were rebuilt in this process."""
    _pooled.cache_clear()
    _holding.cache_clear()


def pooled(leaf, scanner, channels=None, within=()):
    """See _pooled: the same, computed once per leaf, conditions and database (the package's rows do
    not change; a database stood in for it, as a test does, is another key)."""
    return _pooled(leaf, float(scanner.b0_t), scanner.vendor, scanner.model, channels, tuple(within), id(package()))


@functools.lru_cache(maxsize=4096)
def _pooled(leaf, b0_t, vendor, model, channels, within, database):
    scanner = type("Asked", (), {"b0_t": b0_t, "vendor": vendor, "model": model})
    return _pool(leaf, scanner, channels, within)


def _pool(leaf, scanner, channels=None, within=()):
    """The Calibration of `leaf` for `scanner` (a hardware.Scanner) pooled over the package's
    cohort rows read under its conditions: the median over subjects is the value, their
    5th to 95th percentile the prior (the range over fewer than five, one row's own prior
    alone). The conditions are as specific as the scanner is: its field, its vendor and
    model where it states them, and `channels`, the receive array's count, where given and
    the leaf is the receive chain's (RECEIVE); a condition nothing matches is dropped, model
    first, then channels, then vendor, then the field itself (the departure names it).
    `within` names the cohorts whose rows alone are pooled (one dataset as the scanner's
    calibration); empty pools them all. None when no row of the leaf is there."""
    found = rows(leaf, within)
    asked = dict(field_t=float(scanner.b0_t), vendor=scanner.vendor, model=scanner.model,
                 channels=channels if leaf in RECEIVE else None)
    for dropped in ((), ("model",), ("model", "channels"), ("model", "channels", "vendor"),
                    ("model", "channels", "vendor", "field_t")):
        conditions = {key: None if key in dropped else value for key, value in asked.items()}
        matched = [row for row in found if _matches(row[2], **conditions)]
        if matched:
            break
    if not matched:
        return None
    values = np.array([record.value for _, _, record in matched], dtype=float)
    value = float(np.median(values))
    percentiles = {}
    if len(values) == 1:
        prior = matched[0][2].prior
    elif len(values) < 5:
        prior = (float(values.min()), float(values.max()))
    else:
        prior = (float(np.percentile(values, 5)), float(np.percentile(values, 95)))
        percentiles = {str(q): round(float(np.percentile(values, q)), 7) for q in (5, 25, 50, 75, 95)}
    records = [record for _, _, record in matched]
    cohorts = sorted({name for name, _, _ in matched})

    def one(values):
        return values[0] if len(values) == 1 else "pooled"
    # what the rows were matched on: the conditions asked and kept that at least one row states
    states = {"field_t": lambda r: True, "vendor": lambda r: bool(r.scanner.get("vendor")),
              "model": lambda r: bool(r.scanner.get("model")), "channels": lambda r: bool(r.coil.get("channels"))}
    stated = {key: value for key, value in conditions.items() if value is not None and any(states[key](r) for r in records)}
    per_cohort = {name: round(float(np.median([r.value for c, _, r in matched if c == name])), 7) for name in cohorts}
    return Calibration(
        leaf=leaf, value=round(value, 7), prior=tuple(round(float(v), 7) for v in prior),
        measured={"statistic": "median over the subjects of the package's cohorts read under the scanner's conditions; "
                               "the prior their 5th to 95th percentile (scan.cohort.pooled)",
                  "subjects": len(matched), "matched_on": stated, "percentiles": percentiles, "per_cohort_median": per_cohort,
                  "values": [round(float(v), 7) for v in sorted(values)],
                  "subjects_per_cohort": {name: sum(1 for c, _, _ in matched if c == name) for name in cohorts}},
        source={"dataset": f"pooled over {len(matched)} subjects of {len(cohorts)} cohort{'s' if len(cohorts) > 1 else ''}: "
                           + ", ".join(cohorts), "citation": None, "url": None},
        scanner={"vendor": one(sorted({r.scanner.get("vendor") or "unstated" for r in records})),
                 "model": one(sorted({str(r.scanner.get("model") or "unstated") for r in records})),
                 "software": None, "field_t": round(float(np.median([r.scanner["field_t"] for r in records if r.scanner.get("field_t") is not None] or [float("nan")])), 2)},
        coil={"channels": tuple(sorted({int(c) for r in records for c in (r.coil.get("channels") or ())}))},
        sequence={"name": ", ".join(sorted({str(r.sequence.get("name")) for r in records})),
                  "family": one(sorted({r.sequence.get("family", "") for r in records})).replace("pooled", "")},
        acquisition={"the subjects' own": True},
        voxel={"the subjects' own": True},
        processing="the rows as their readers made them (scan.calibrate), pooled by scan.cohort.pooled",
        date=max(r.date for r in records), carried_by=records[0].carried_by,
        note="the package's measured database pooled for this scanner; a scan's departures name what it does not match",
    )


def holding(scanner):
    """The names of the package's cohorts with rows of `scanner`'s vendor and model at its
    field (its own systems); with none, those at its field (a generic scanner)."""
    return _holding(float(scanner.b0_t), scanner.vendor, scanner.model, id(package()))


@functools.lru_cache(maxsize=None)
def _holding(b0_t, vendor, model, database):
    scanner = type("Asked", (), {"b0_t": b0_t, "vendor": vendor, "model": model})
    for model in (scanner.model, None):
        found = sorted({cohort.name for cohort in package() for member in cohort.members for record in member.profile.records
                        if not record.failed and _matches(record, float(scanner.b0_t), scanner.vendor, model, None)})
        if found:
            return found
    return sorted({cohort.name for cohort in package() for member in cohort.members for record in member.profile.records
                   if not record.failed and _matches(record, float(scanner.b0_t), None, None, None)})
