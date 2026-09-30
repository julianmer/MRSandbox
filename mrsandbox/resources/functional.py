####################################################################################################
#                                          functional.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: The open functional MRI datasets the subject's run is read off: the breathing and the   #
#          heart off the physiological traces, the head's motion off the series registered volume  #
#          by volume (scan.run), each a cohort of its subjects' rows.                              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import json
import pathlib

from mrsandbox.hardware import Scanner
from mrsandbox.resources import participants
from mrsandbox.scan.cohort import Cohort, Member
from mrsandbox.scan.profile import ScannerProfile
from mrsandbox.scan.run import MotionReading, PhysioReading


#**************************************************************************************************#
#                                       Class RestingState7T                                       #
#**************************************************************************************************#
#                                                                                                  #
# A 7 T resting-state test-retest dataset with physiological recordings.                           #
#                                                                                                  #
#**************************************************************************************************#
class RestingState7T:
    """A 7 T resting-state test-retest dataset with physiological recordings.

    Source: Gorgolewski, K. J., et al. A high resolution 7-Tesla resting-state fMRI test-retest
            dataset with cognitive and physiological measures. Sci Data 2 (2015) 140054;
            OpenNeuro ds001168 (CC0), doi 10.18112/openneuro.ds001168.v1.0.0.
    Data:   volunteers of earlier 7 T studies, aged 21 to 30, their health not stated (a
            volunteer's taken as healthy), on a Siemens Magnetom 7T (its README; the sidecars say
            nothing of the scanner), session 1's prefrontal resting-state run with its
            respiratory and cardiac traces.
    """

    DATASET = ("OpenNeuro ds001168, 7 T resting state with physiological recordings (session 1, "
               "the prefrontal run)")
    CITED = {"citation": "doi 10.18112/openneuro.ds001168.v1.0.0",
             "url": "https://openneuro.org/datasets/ds001168"}
    SCANNER = {"Manufacturer": "Siemens", "ManufacturersModelName": "Magnetom 7T",
               "MagneticFieldStrength": 7.0}

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The ds001168_siemens_7t cohort from the dataset's local copy at `root`: every
        subject's breathing and heart off its physiological trace and, where the BOLD series
        is there, the head's motion; the scanner a generic 7 T, since nothing of the receive
        chain is read off an image series."""
        root = pathlib.Path(root)
        sidecar = root / "task-rest_acq-prefrontal_bold.json"
        members = []
        for folder in sorted(root.glob("sub-*/ses-1/func")):
            records = ()
            physio = next(folder.glob("*_acq-prefrontal_physio.tsv.gz"), None)
            series = next(folder.glob("*_acq-prefrontal_bold.nii.gz"), None)
            if physio is not None:
                reading = PhysioReading.from_bids(physio, scan_sidecar=sidecar)
                reading = dataclasses.replace(reading, header={**reading.header, **cls.SCANNER})
                if reading.resp_hz:
                    records += (reading.record_breathing(cls.DATASET, **cls.CITED),)
                if reading.cardiac_hz:
                    records += (reading.record_heart(cls.DATASET, **cls.CITED),)
            if series is not None:
                motion = MotionReading.from_series(
                    series, header={**json.loads(sidecar.read_text()), **cls.SCANNER})
                records += (motion.record_events(cls.DATASET, **cls.CITED),
                            motion.record_size(cls.DATASET, **cls.CITED))
            if records:
                members.append(Member(folder.parents[1].name,
                                      ScannerProfile(Scanner.preset("generic_7t"), records),
                                      (0.0, 40.0, 8.0), (20.0, 20.0, 20.0), "icbm152"))
        return participants.attach(Cohort("ds001168_siemens_7t", tuple(members)),
                                   participants.bids(root, health=participants.VOLUNTEER))
