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
#          value on one real subject at once (info/HANDOFF.md item 239).                           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import json
import pathlib

import numpy as np

from mrsandbox.phantom import Brain
from mrsandbox.scan.profile import ScannerProfile

COHORTS = pathlib.Path(__file__).resolve().parent.parent / "resources" / "cohorts"


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
        subject, not a range about one."""
        records = tuple(dataclasses.replace(record, prior=(record.value, record.value)) for record in self.profile.records)
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
