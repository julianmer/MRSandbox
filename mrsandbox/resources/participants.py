####################################################################################################
#                                          participants.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 29/09/26                                                                                #
#                                                                                                  #
# Purpose: Who each dataset's subjects are, as the dataset states it: age, sex and health per      #
#          subject off its own table, or what its paper says of everyone, set on the members of    #
#          its cohort so the measured database is queried by the person (scan.cohort).             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import csv
import dataclasses
import json
import pathlib
import re

#***************#
#   constants   #
#***************#
# the words a dataset's group levels use for its healthy subjects
HEALTHY = ("healthy", "healthy control", "healthy controls", "control", "hc")
# a research volunteer whose health the dataset does not state is taken as healthy, the usual
# volunteer (taken, not stated: TAKEN names every trait that rests on a rule)
VOLUNTEER = "healthy"
# the ages of an adult, where a dataset says so, or where a volunteer is taken as one
ADULT_Y = (18.0, None)

# Source: per entry, beside each value.
# Data:   what a dataset's paper states of all its subjects where the dataset has no table of
#         them: a number stated only as a range is that range; health is not stated where the
#         paper does not state it.
STATED = {
    # Mikkelsen et al. 2017, NeuroImage 159:32: 272 healthy adults (for a subject the
    # demographics table leaves out)
    **{f"big_gaba_{vendor}": {"age_y": ADULT_Y, "health": "healthy"}
       for vendor in ("ge", "philips", "trio")},
    # Archibald et al. 2020, Sci Rep 10:19218: eighteen healthy participants, 9 F and 9 M,
    # aged 21 to 36
    "fmrs_pain_achieva": {"age_y": (21.0, 36.0), "health": "healthy"},
    # Riemann et al. 2022, MRM 87:1119: nine healthy volunteers, aged 39 +- 13 (mean, sd)
    "neuromet_siemens_7t": {"health": "healthy"},
    # Clarke et al. 2020, NeuroImage 206:116335, section 2.5: the same subject at every site,
    # male, 28 years old, 82 kg; the health not stated, a volunteer's taken
    "uk7t_oxford_7t": {"age_y": 28.0, "sex": "M", "health": VOLUNTEER},
    # Callaghan et al. 2019, Data in Brief 25:104132: one participant, who gave their own
    # written consent (an adult); a volunteer's health taken
    "hmri_demo_prisma_3t": {"age_y": ADULT_Y, "health": VOLUNTEER},
}


# the traits of a cohort's subjects that a rule gave, not the dataset: shown as taken
TAKEN = {
    "ds001168_siemens_7t": ("health",),
    "uk7t_oxford_7t": ("health",),
    "hmri_demo_prisma_3t": ("age_y", "health"),  # an adult by their own consent, a volunteer
}


#**********#
#   bids   #
#**********#
def bids(folder, health=None):
    """{participant_id: traits} off a BIDS dataset's participants.tsv: "age_y" off its age
    column (age, or age_at_first_scan_years), "sex" off sex, "health" off a group column
    through the level descriptions of its participants.json (a healthy control is
    "healthy", a condition its name in lower case), else `health`, what the dataset states
    of everyone. A value the table leaves empty or n/a is not stated."""
    folder = pathlib.Path(folder)
    levels = {}
    if (folder / "participants.json").exists():
        described = json.loads((folder / "participants.json").read_text())
        levels = described.get("group", {}).get("Levels", {})
    people = {}
    with open(folder / "participants.tsv") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            row = {k.strip(): (v or "").strip() for k, v in row.items()}
            traits = {}
            age = row.get("age") or row.get("age_at_first_scan_years")
            if age and age.lower() != "n/a":
                traits["age_y"] = float(age)
            if row.get("sex") and row["sex"].lower() != "n/a":
                traits["sex"] = row["sex"].upper()
            group = row.get("group")
            if group and group.lower() != "n/a":
                named = levels.get(group, group).lower()
                traits["health"] = "healthy" if named in HEALTHY else named
            elif health is not None:
                traits["health"] = health
            people[row["participant_id"]] = traits
    return people


#**************#
#   big gaba   #
#**************#
def big_gaba(path):
    """{"<site>/<subject>": traits} off the Big GABA demographics table (site, subject id,
    sex, age), every subject healthy as Mikkelsen et al. 2017 (NeuroImage 159:32) state of
    all 272: healthy adults."""
    with open(path) as file:
        return {f"{row['site']}/{row['subjID']}": {"age_y": float(row["age"]),
                                                  "sex": row["sex"].upper(),
                                                  "health": "healthy"}
                for row in csv.DictReader(file) if row.get("age")}


#************#
#   attach   #
#************#
def attach(cohort, people=None):
    """The cohort with each member's traits: `people` {key: traits} looked up by the
    member's subject (a BIDS participant, sub-..., found in the name; a Big GABA site and
    subject, G4/S02 however written), else what STATED holds of the cohort's subjects; the
    traits a rule gave marked taken (TAKEN)."""
    def key(subject):
        found = re.search(r"sub-[A-Za-z0-9]+", subject)
        if found:
            return found.group(0)
        found = re.match(r"([GPS]\d+)\D*?[/_]?(S\d+)", subject, re.I)
        return f"{found.group(1).upper()}/{found.group(2).upper()}" if found else subject

    stated = STATED.get(cohort.name, {})
    return dataclasses.replace(cohort, members=tuple(
        dataclasses.replace(member, traits=dict((people or {}).get(key(member.subject), stated)),
                            taken=TAKEN.get(cohort.name, ()))
        for member in cohort.members))

