####################################################################################################
#                                          test_dicom.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gate of io.dicom: a public DICOM image (pydicom's own test data) comes back as a NIfTI  #
#          volume, and a folder without DICOM says so.                                             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pathlib
import shutil
import sys

import pytest

pytest.importorskip("pydicom")
if (shutil.which("dcm2niix") is None
        and shutil.which("dcm2niix", path=str(pathlib.Path(sys.executable).parent)) is None):
    pytest.skip("dcm2niix is not installed", allow_module_level=True)


#**************************************************************************************************#
#                                         Class TestDicom                                          #
#**************************************************************************************************#
class TestDicom:

    def test_a_dicom_image_becomes_a_nifti_volume(self, tmp_path):
        import nibabel as nib
        from pydicom.data import get_testdata_file

        from mrsandbox.io import read_dicom

        (tmp_path / "in").mkdir()
        shutil.copy(get_testdata_file("MR_small.dcm"), tmp_path / "in" / "MR_small.dcm")
        found = read_dicom(tmp_path / "in", tmp_path / "out")
        assert len(found) == 1 and nib.load(str(next(iter(found.values())))).shape[:2] == (64, 64)
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError):
            read_dicom(tmp_path / "empty")
