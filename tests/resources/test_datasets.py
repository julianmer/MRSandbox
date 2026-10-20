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
#                                        Class TestDatasets                                        #
#**************************************************************************************************#
class TestDatasets:

    def test_every_package_cohort_has_its_recipe(self):
        shipped = {path.stem for path in cohorts.COHORTS.glob("*.json")}
        assert set(datasets.DATASETS) == shipped

    def test_a_copy_fetched_by_hand_stops_the_call_before_anything(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nitrc.org/frs"):
            datasets.rebuild(tmp_path, names=["cows_prisma", "big_gaba_trio"],
                             into=tmp_path / "built")
        assert not (tmp_path / "built").exists() and not (tmp_path / "openneuro_ds006812").exists()
        # asked not to fetch, any missing copy stops it
        with pytest.raises(FileNotFoundError, match="ds006812"):
            datasets.rebuild(tmp_path, names=["cows_prisma"], fetch=False)

    def test_a_missing_copy_is_fetched_then_built_and_saved(self, tmp_path, monkeypatch):
        made = Cohort("mine", (Member("S1", ScannerProfile(Scanner.preset("generic_3t")),
                                      (0.0, 0.0, 0.0), (20.0, 20.0, 20.0)),))
        monkeypatch.setitem(datasets.DATASETS, "mine", (
            "mine", "a URL", lambda root: root.mkdir(),
            lambda root, data: made if root.is_dir() else None))
        saved = datasets.rebuild(tmp_path, names=["mine"], into=tmp_path / "built")
        assert saved == [tmp_path / "built" / "mine.json"] and (tmp_path / "mine").is_dir()
        assert Cohort.load(saved[0]).members[0].subject == "S1"
