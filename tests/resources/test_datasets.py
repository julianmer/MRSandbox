####################################################################################################
#                                         test_datasets.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: Every package cohort has the recipe that rebuilds it, and one call rebuilds them from   #
#          the local copies, stopping before any build when a copy is missing.                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pytest

from mrsandbox import Scanner
from mrsandbox.resources import datasets
from mrsandbox.scan import Cohort, Member, ScannerProfile
from mrsandbox.scan import cohort as cohorts


#**************************************************************************************************#
#                                         Class TestDatasets                                       #
#**************************************************************************************************#
class TestDatasets:

    def test_every_package_cohort_has_its_recipe(self):
        shipped = {path.stem for path in cohorts.COHORTS.glob("*.json")}
        assert set(datasets.DATASETS) == shipped

    def test_a_missing_copy_stops_the_call_before_any_build(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="openneuro.org/datasets/ds006812"):
            datasets.rebuild(tmp_path, names=["neuromet_siemens_7t", "cows_prisma"],
                             into=tmp_path / "built")
        assert not (tmp_path / "built").exists()

    def test_the_call_saves_each_cohort_it_builds(self, tmp_path, monkeypatch):
        (tmp_path / "mine").mkdir()
        made = Cohort("mine", (Member("S1", ScannerProfile(Scanner.preset("generic_3t")),
                                      (0.0, 0.0, 0.0), (20.0, 20.0, 20.0)),))
        monkeypatch.setitem(datasets.DATASETS, "mine",
                            ("mine", "a URL", lambda root, data: made))
        saved = datasets.rebuild(tmp_path, names=["mine"], into=tmp_path / "built")
        assert saved == [tmp_path / "built" / "mine.json"]
        assert Cohort.load(saved[0]).members[0].subject == "S1"
