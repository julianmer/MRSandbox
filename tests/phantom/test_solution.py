####################################################################################################
#                                         test_solution.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gates of phantom.solution: a sphere of solution takes a brain's place, its scan holds   #
#          the contents and nothing else, and the spectrum shows them as the recipe says.          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox import (NoDownfield, NoLipids, NoMM, NoWater, Phantom, Press, Scanner, Unsuppressed,
                       VirtualScan)
from mrsandbox.database.concentrations import SPECTRE


#*************#
#   fixture   #
#*************#
@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Phantom.spectre(temperature_c=21.0), Scanner.preset("generic_3t"),
                       Press(te1_s=0.015, te2_s=0.025, tr_s=2.0), water=Unsuppressed())


#**************************************************************************************************#
#                                        Class TestPhantom                                         #
#**************************************************************************************************#
@pytest.mark.render
class TestPhantom:

    def test_the_sphere_is_solution_inside_and_nothing_outside(self):
        phantom = Phantom.spectre()
        inside, outside = phantom.fractions((0.0, 0.0, 0.0), 20.0), phantom.fractions((0.0, 0.0, 120.0), 20.0)
        assert inside.gm == pytest.approx(1.0) and inside.wm == inside.csf == 0.0
        assert outside.gm == outside.wm == outside.csf == 0.0
        assert phantom.lipid_weight((60.0, 0.0, 0.0), 20.0) == 0.0
        assert Phantom.spectre(pico=True).fractions((0.0, 0.0, 70.0), 10.0).gm == 0.0

    def test_a_scan_of_it_is_in_vitro(self, scan):
        assert isinstance(scan.mm, NoMM) and isinstance(scan.lipids, NoLipids) and isinstance(scan.downfield, NoDownfield)
        assert scan.temperature_c == 21.0 and scan.ph == 7.4
        truth = scan.voxel((0.0, 0.0, 0.0), 20.0).sample(3)
        assert {name: value for name, value in truth.conc.items() if value} == SPECTRE
        assert truth.t2.Cho == truth.t2.NAA == 0.5 and truth.contrast.Cho == 1.0 and truth.lineshape.gauss_hz == 0.0
        # the caller still has the last word
        warm = VirtualScan(Phantom.spectre(), Scanner.preset("generic_3t"), Press(te1_s=0.015, te2_s=0.025),
                           temperature_c=37.0)
        assert warm.temperature_c == 37.0

    def test_the_spectrum_shows_the_recipe_and_the_temperature(self, scan):
        n, dwell_s = 8192, 1.0 / 2000.0

        def spectrum(of):
            voxel = of.voxel((0.0, 0.0, 0.0), 20.0)
            _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
            ppm = np.fft.fftshift(np.fft.fftfreq(n, dwell_s)) / voxel._f0_mhz()
            return ppm, np.abs(np.fft.fftshift(np.fft.fft(np.asarray(fid) * np.exp(-np.pi * 2.0 * np.arange(n) * dwell_s))))

        # without water: two singlets of three protons under one relaxation go as the concentrations
        dry = VirtualScan(Phantom.spectre(temperature_c=21.0), Scanner.preset("generic_3t"),
                          Press(te1_s=0.015, te2_s=0.025, tr_s=2.0), water=NoWater())
        ppm, metabolites = spectrum(dry)
        height = lambda s, low, high: s[(ppm > low) & (ppm < high)].max()
        assert height(metabolites, 2.97, 3.09) / height(metabolites, 1.95, 2.07) == pytest.approx(
            SPECTRE["Cr"] / SPECTRE["NAA"], rel=0.06)
        # with it: the water line sits where 21 C puts it, 0.0103 ppm per degree above its place
        # at 37 C, and it is all water, 55.5 M against 12.5 mM of three protons
        ppm, wet = spectrum(scan)
        band = (ppm > 4.5) & (ppm < 5.1)
        assert ppm[band][wet[band].argmax()] == pytest.approx(4.65 + 0.0103 * (37.0 - 21.0), abs=0.01)
        assert height(wet, 4.5, 5.1) / height(metabolites, 1.95, 2.07) > 1000.0

    def test_a_heads_record_on_a_phantom_names_the_load(self, scan):
        named = scan.voxel((0.0, 0.0, 0.0), 20.0).acquire(seed=0, npoints=256).hdr_ext["SimulationPhantom"]
        assert named["brain"] == "spectre"
        assert named["calibrations"]["noise.sigma"]["extrapolated"]["load"] == ["a head", "spectre"]
