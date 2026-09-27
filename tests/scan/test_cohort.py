####################################################################################################
#                                          test_cohort.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 26/09/26                                                                                #
#                                                                                                  #
# Purpose: A cohort's members are drawn one at a time, each a subject whose own readings every     #
#          measured leaf lands on, saved and loaded as one file.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

from mrsandbox import Fixed, Press, Scanner
from mrsandbox.scan import Cohort, Member, ScannerProfile
from mrsandbox.scan.calibrations import NOISE_BIG_GABA_SIEMENS_3T, RESIDUAL_WATER_BIG_GABA_SIEMENS_3T

PLACE, SIZE = (0.0, -35.0, 20.0), (30.0, 30.0, 30.0)


def member(subject, sigma, residual):
    own = {"dataset": "a cohort", "citation": None, "url": None}
    records = (dataclasses.replace(NOISE_BIG_GABA_SIEMENS_3T, value=sigma, prior=(0.75 * sigma, 1.25 * sigma), presets=(), source=own),
               dataclasses.replace(RESIDUAL_WATER_BIG_GABA_SIEMENS_3T, value=residual, prior=tuple(sorted((0.75 * residual, 1.25 * residual))),
                                   presets=(), source=own))
    return Member(subject, ScannerProfile(Scanner.preset("siemens_trio_3t"), records), PLACE, SIZE)


#**************************************************************************************************#
#                                         Class TestCohort                                         #
#**************************************************************************************************#
class TestCohort:

    #**********#
    #   draw   #
    #**********#
    def test_a_draw_is_one_subject_with_its_values_fixed(self):
        cohort = Cohort("two subjects", (member("S01", 6.0, -0.006), member("S02", 9.0, -0.012)))
        drawn = {cohort.draw(seed).subject for seed in range(24)}
        assert drawn == {"S01", "S02"}
        one = next(m for m in cohort.members if m.subject == "S02")
        scan = one.scan(Press(te1_s=0.011, te2_s=0.024))
        sigma, residual = scan.params.noise.sigma, scan.params.water.adjusted_residual
        assert sigma == Fixed(9.0) and residual == Fixed(-0.012)                    # the subject's own, not a range about it
        assert sigma.chosen == "measured" and scan.voxel(PLACE, SIZE).sample(seed=3).noise.sigma == 9.0
        assert one.profile.records[0].prior == (6.75, 11.25) and one.fixed().records[0].prior == (9.0, 9.0)   # the member's own untouched

    #*******************#
    #   save and load   #
    #*******************#
    def test_a_cohort_is_saved_and_loaded_as_one_file(self, tmp_path):
        cohort = Cohort("two subjects", (member("S01", 6.0, -0.006), member("S02", 9.0, -0.012)))
        cohort.save(tmp_path / "cohort.json")
        loaded = Cohort.load(tmp_path / "cohort.json")
        assert loaded.name == cohort.name and [m.subject for m in loaded.members] == ["S01", "S02"]
        assert loaded.members[1].profile.calibrations()["noise.sigma"].value == 9.0
        assert loaded.members[1].position_mm == PLACE and loaded.members[1].brain == "icbm152"
        header = loaded.members[0].scan(Press(te1_s=0.011, te2_s=0.024)).voxel(PLACE, SIZE).acquire(
            seed=0, npoints=256).hdr_ext["SimulationPhantom"]["calibrations"]
        assert header["noise.sigma"]["source"] == "a cohort"
