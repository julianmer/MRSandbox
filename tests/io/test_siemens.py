####################################################################################################
#                                          test_siemens.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: The Siemens twix reader on a deposit's file whose measurement table is off its scan.    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pathlib

import pytest

from mrsandbox.io.siemens import read_twix

SHORT = (pathlib.Path(__file__).resolve().parents[2] / "data" / "openneuro_ds006812" / "sub-01" / "mrs"
         / "sourcedata" / "sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat")


#**************************************************************************************************#
#                                          Class TestMended                                        #
#**************************************************************************************************#
class TestMended:

    #*******************#
    #   the 90th scan   #
    #*******************#
    def test_a_table_off_its_scan_is_mended_and_the_scan_read_whole(self):
        # OpenNeuro's copy is four bytes short with its table 516 bytes off the scan's header;
        # mapVBVD alone stops, the mended copy gives every transient and coil, the water too
        if not SHORT.exists():
            pytest.skip("the COWS copy is not placed")
        x, dwell, f0, header = read_twix(SHORT, phs=1)
        y, _, _, _ = read_twix(SHORT, phs=0)
        assert x.shape == (2048, 32, 32) and y.shape[:2] == (2048, 32) and 1 <= y.shape[2] <= 32
        assert abs(dwell - 1.0 / 4000.0) < 1e-12 and 123.2 < f0 < 123.3
        assert header["EchoTime"] == 0.026 and header["RepetitionTime"] == 2.0
