####################################################################################################
#                                          test_arrays.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: The scanners' receive arrays build as published, a Tim coil's modes carry what its      #
#          loops do, and an array with modes is saved and read back whole.                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import CoilArray, Scanner, ScannerProfile
from mrsandbox.hardware.arrays import PRESETS

PLACES = ((0.0, 40.0, 30.0), (0.0, -16.0, 6.0), (50.0, -40.0, 20.0), (-45.0, 10.0, -20.0))


#**************************************************************************************************#
#                                          Class TestArrays                                        #
#**************************************************************************************************#
class TestArrays:

    def test_every_preset_builds_its_channels_and_names_its_sources(self):
        for key, preset in PRESETS.items():
            array = preset.array()
            assert (len(array) == preset.channels
                    and len(array.sensitivities(PLACES[0])) == preset.channels), key
            assert preset.cited and preset.stated and preset.vendor in ("siemens", "ge", "philips",
                                                                        "research")

    def test_the_triple_mode_carries_what_the_loops_do(self):
        # Reykowski and Blasche 2004: primary, secondary and tertiary together are the raw loops'
        # signals: a unitary combiner, so the optimum combination's noise is the loops' own
        # everywhere; the primary alone (CP) loses where the loops differ most, away from the
        # centre it was phased for
        triple = PRESETS["siemens_head_matrix_12_triple"].array()
        assert np.allclose(triple.modes @ triple.modes.conj().T, np.eye(12), atol=1e-12)
        loops = CoilArray(triple.loops).loaded()
        moded = triple.loaded()
        cp = PRESETS["siemens_head_matrix_12_cp"].array().loaded()
        for place in PLACES:
            assert moded.noise_gain(place) == pytest.approx(loops.noise_gain(place), rel=1e-4)
            assert cp.noise_gain(place) >= loops.noise_gain(place) * (1.0 - 1e-6)
        assert cp.noise_gain(PLACES[2]) > 1.1 * loops.noise_gain(PLACES[2])

    def test_an_array_with_modes_is_saved_and_read_back(self, tmp_path):
        array = PRESETS["siemens_head_neck_mrpet"].array().loaded(own_noise=0.3)
        assert np.iscomplexobj(array.noise_resistance)
        ScannerProfile(Scanner.preset("generic_3t"), coils=array).save(tmp_path / "scanner.json")
        back = ScannerProfile.load(tmp_path / "scanner.json").coils
        assert (len(back) == 16 and np.allclose(back.modes, array.modes)
                and np.allclose(back.noise_resistance, array.noise_resistance))
        assert back.noise_gain(PLACES[0]) == pytest.approx(array.noise_gain(PLACES[0]), rel=1e-6)
        # an array without modes is saved plain, its resistance real
        ScannerProfile(Scanner.preset("generic_3t"),
                       coils=CoilArray.helmet().loaded()).save(tmp_path / "plain.json")
        plain = ScannerProfile.load(tmp_path / "plain.json").coils
        assert plain.modes is None and not np.iscomplexobj(plain.noise_resistance)
