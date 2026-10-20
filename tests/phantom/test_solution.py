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
from mrsandbox.database.phantoms import BRAINO, NIST_SYSTEM, SPECTRE
from mrsandbox.phantom.solution import Vial


#*************#
#   fixture   #
#*************#
@pytest.fixture(scope="module")
def scan():
    return VirtualScan(Phantom.spectre(temperature_c=21.0), Scanner.preset("generic_3t"),
                       Press(te1_s=0.015, te2_s=0.025, tr_s=2.0), water=Unsuppressed())


#**************************************************************************************************#
#                                         Class TestPhantom                                        #
#**************************************************************************************************#
@pytest.mark.render
class TestPhantom:

    def test_the_sphere_is_solution_inside_and_nothing_outside(self):
        phantom = Phantom.spectre()
        inside, outside = (phantom.fractions((0.0, 0.0, 0.0), 20.0),
                           phantom.fractions((0.0, 0.0, 120.0), 20.0))
        assert inside.gm == pytest.approx(1.0) and inside.wm == inside.csf == 0.0
        assert outside.gm == outside.wm == outside.csf == 0.0
        assert phantom.lipid_weight((60.0, 0.0, 0.0), 20.0) == 0.0
        assert Phantom.spectre(pico=True).fractions((0.0, 0.0, 70.0), 10.0).gm == 0.0

    def test_a_scan_of_it_is_in_vitro(self, scan):
        assert (isinstance(scan.mm, NoMM) and isinstance(scan.lipids, NoLipids)
                and isinstance(scan.downfield, NoDownfield))
        assert scan.temperature_c == 21.0 and scan.ph == 7.4
        truth = scan.voxel((0.0, 0.0, 0.0), 20.0).sample(3)
        assert {name: value for name, value in truth.conc.items() if value} == SPECTRE
        assert (truth.t2.Cho == truth.t2.NAA == 0.5 and truth.contrast.Cho == 1.0
                and truth.lineshape.gauss_hz == 0.0)
        # the caller still has the last word
        warm = VirtualScan(Phantom.spectre(), Scanner.preset("generic_3t"),
                           Press(te1_s=0.015, te2_s=0.025), temperature_c=37.0)
        assert warm.temperature_c == 37.0

    def test_the_spectrum_shows_the_recipe_and_the_temperature(self, scan):
        n, dwell_s = 8192, 1.0 / 2000.0

        def spectrum(of):
            voxel = of.voxel((0.0, 0.0, 0.0), 20.0)
            _, fid = voxel._clean_fid(voxel.sample(0), n, dwell_s)
            ppm = np.fft.fftshift(np.fft.fftfreq(n, dwell_s)) / voxel._f0_mhz()
            return ppm, np.abs(np.fft.fftshift(np.fft.fft(
                np.asarray(fid) * np.exp(-np.pi * 2.0 * np.arange(n) * dwell_s))))

        # without water: two singlets of three protons under one relaxation go as the
        # concentrations
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
        assert ppm[band][wet[band].argmax()] == pytest.approx(4.65 + 0.0103 * (37.0 - 21.0),
                                                              abs=0.01)
        assert height(wet, 4.5, 5.1) / height(metabolites, 1.95, 2.07) > 1000.0

    def test_a_heads_record_on_a_phantom_names_the_load(self, scan):
        named = (scan.voxel((0.0, 0.0, 0.0), 20.0).acquire(seed=0, npoints=256)
                 .hdr_ext["SimulationPhantom"])
        assert named["brain"] == "spectre"
        assert named["calibrations"]["noise.sigma"]["extrapolated"]["load"] == ["a head", "spectre"]


#**************************************************************************************************#
#                                          Class TestVials                                         #
#**************************************************************************************************#
@pytest.mark.render
class TestVials:

    def test_a_vial_holds_its_own_solution_and_every_voxel_reads_its_place(self):
        vial = Vial((40.0, 0.0, 0.0), 30.0, contents={"NAA": 25.0}, t2_s={"NAA": 0.2},
                    water_t1_s=1.0)
        phantom = Phantom({"NAA": 10.0, "Cr": 8.0}, t2_s={"Cr": 0.3}, vials=(vial,), cell_mm=2.0)
        scan = VirtualScan(phantom, Scanner.preset("generic_3t"), Press(te1_s=0.015, te2_s=0.025))
        inside, outside = scan.voxel((40.0, 0.0, 0.0), 10.0), scan.voxel((-40.0, 0.0, 0.0), 10.0)
        truth = inside.sample(0)
        read = lambda voxel, leaf: float(voxel.resolve(leaf))
        assert (read(inside, truth.conc.NAA) == pytest.approx(25.0)
                and read(outside, truth.conc.NAA) == pytest.approx(10.0))
        assert (read(inside, truth.conc.Cr) == pytest.approx(0.0)
                and read(outside, truth.conc.Cr) == pytest.approx(8.0))
        assert (read(inside, truth.t2.NAA) == pytest.approx(0.2)
                and read(outside, truth.t2.NAA) == pytest.approx(0.5))
        assert truth.t2.Cr == 0.3 and truth.t1.NAA == 1.5  # what no vial changes stays a number
        assert (read(inside, truth.water.t1_s) == pytest.approx(1.0)
                and read(outside, truth.water.t1_s) == pytest.approx(0.4))
        # and the spectra follow: the vial's NAA line is two and a half times as strong before
        # relaxation
        assert (inside.acquire(seed=0, npoints=256).hdr_ext["SimulationPhantom"]["brain"]
                == "phantom")

    def test_the_braino_is_the_literatures_recipe(self):
        phantom = Phantom.braino(t1_s={"NAA": 0.8, "Cho": 0.26})
        assert (phantom.contents == BRAINO and phantom.diameter_mm == 169.0
                and "GABA" not in phantom.contents)
        truth = VirtualScan(phantom, Scanner.preset("generic_7t"),
                            Press(te1_s=0.015, te2_s=0.025)).voxel((0.0, 0.0, 0.0), 20.0).sample(0)
        assert truth.t1.NAA == 0.8 and truth.t1.Cho == 0.26 and truth.t1.Cr == 1.5

    def test_the_nist_system_phantom_carries_its_measured_arrays(self):
        phantom = Phantom.nist(b0_t=3.0)
        scan = VirtualScan(phantom, Scanner.preset("generic_3t"), Press(te1_s=0.015, te2_s=0.025),
                           water=Unsuppressed())
        truth = scan.voxel((0.0, 0.0, 0.0), 8.0).sample(0)
        for array, number in (("t1_array", 4), ("t2_array", 8), ("pd_array", 11)):
            place, _, _, (t1_ms, t2_ms) = NIST_SYSTEM[array][number]
            voxel = scan.voxel(place, 6.0)
            assert float(voxel.resolve(truth.water.t1_s)) == pytest.approx(t1_ms / 1000.0, rel=1e-4)
            assert float(voxel.resolve(truth.water.t2_s)) == pytest.approx(t2_ms / 1000.0, rel=1e-4)
        # the fill water between the plates, and the proton density sphere of 20 percent H2O
        assert (float(scan.voxel((0.0, 80.0, 0.0), 6.0).resolve(truth.water.t1_s))
                == pytest.approx(1.0 / 0.316, rel=1e-4))
        assert (scan.voxel(NIST_SYSTEM["pd_array"][3][0], 6.0).fractions.gm
                == pytest.approx(0.2, abs=1e-6))
        assert not any(float(scan.voxel((0.0, 0.0, 0.0), 8.0).resolve(c))
                       for c in truth.conc.values())
        assert Phantom.nist(b0_t=1.5).vials[0].water_t1_s == pytest.approx(2.033)
        # the commercial phantom's certificate: 3 T only, by temperature
        bought = Phantom.nist(commercial=True, temperature_c=23.2)
        assert (bought.name == "nist_model_130"
                and bought.vials[0].water_t1_s == pytest.approx(1.98750))
        assert (bought.vials[27].water_t2_s == pytest.approx(0.00548)
                and bought.vials[31].water == pytest.approx(0.2))
        with pytest.raises(ValueError, match="3 T only"):
            Phantom.nist(b0_t=1.5, commercial=True)
