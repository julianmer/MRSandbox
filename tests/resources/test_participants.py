####################################################################################################
#                                       test_participants.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 29/09/26                                                                                #
#                                                                                                  #
# Purpose: Each dataset's subjects get the age, sex and health it states, a group column's         #
#          condition by name, and what the paper says of everyone where the table is silent.       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json

from mrsandbox import Scanner
from mrsandbox.resources import participants
from mrsandbox.scan import Cohort, Member, ScannerProfile


def cohort(name, *subjects):
    return Cohort(name, tuple(Member(s, ScannerProfile(Scanner.preset("generic_3t")),
                                     (0.0, 0.0, 0.0), (20.0, 20.0, 20.0)) for s in subjects))


#**************************************************************************************************#
#                                      Class TestParticipants                                      #
#**************************************************************************************************#
class TestParticipants:

    def test_a_bids_table_names_the_condition_and_the_healthy(self, tmp_path):
        (tmp_path / "participants.tsv").write_text(
            "participant_id\tage\tsex\tgroup\nsub-A\t29\tM\tMDD\nsub-B\t41\tf\tHC\n"
            "sub-C\tn/a\tn/a\tn/a\n")
        (tmp_path / "participants.json").write_text(json.dumps({"group": {"Levels": {
            "MDD": "major depressive disorder", "HC": "healthy control"}}}))
        people = participants.bids(tmp_path, health="healthy")
        assert people == {"sub-A": {"age_y": 29.0, "sex": "M",
                                    "health": "major depressive disorder"},
                          "sub-B": {"age_y": 41.0, "sex": "F", "health": "healthy"},
                          "sub-C": {"health": "healthy"}}   # what the dataset says of all
        found = participants.attach(cohort("x", "sub-A_ses-01", "sub-B vapor7 PFL"), people)
        assert [m.traits["health"] for m in found.members] == ["major depressive disorder",
                                                               "healthy"]

    def test_big_gaba_subjects_however_written_and_the_paper_for_the_rest(self, tmp_path):
        (tmp_path / "demographics.csv").write_text(
            "subjNum,vendor,site,subjID,sex,age\n1,GE,G4,S02,F,21\n2,Philips,P10,S01,M,28\n")
        people = participants.big_gaba(tmp_path / "demographics.csv")
        found = participants.attach(cohort("big_gaba_ge", "g4s02", "P10_P/S01", "P1_P/S02"),
                                    people)
        assert [m.traits for m in found.members] == [
            {"age_y": 21.0, "sex": "F", "health": "healthy"},
            {"age_y": 28.0, "sex": "M", "health": "healthy"},
            {"age_y": (18.0, None), "health": "healthy"}]  # left out of the table
