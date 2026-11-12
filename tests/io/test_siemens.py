####################################################################################################
#                                          test_siemens.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: The Siemens twix reader on a deposit's file whose measurement table is off its scan,    #
#          and the sequence simulation reader: a written simulation read back and played, and the  #
#          public semi-LASER simulation against the library's design.                              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
import pathlib

import numpy as np
import pytest

from mrsandbox import Engine, SpinSystem
from mrsandbox.io import read_dsv
from mrsandbox.io.siemens import read_twix
from mrsandbox.sequences.events import Delay, Gradient, Sequence, ShapedPulse
from mrsandbox.sequences.library import LocalizedSemiLaser

GAMMA = 42.5774785
F0_MHZ = 123.2
LAC = SpinSystem(name="Lac", nucleus="1H", shifts_ppm=(4.0974, 1.3142, 1.3142, 1.3142),
                 j_hz=((0, 6.933, 6.933, 6.933), (6.933, 0, 0, 0), (6.933, 0, 0, 0),
                       (6.933, 0, 0, 0)), scale=1.0, gamma_mhz_t=GAMMA)
SLASER = (pathlib.Path(__file__).resolve().parents[2] / "data" / "bacon_2026" / "repo"
          / "Data_for_correction" / "gradients_sLaser")

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


#*************#
#   helpers   #
#*************#
def write_dsv(path, samples, spacing_us, factor):
    """A DSV file as the simulator writes it: differences in units of 1 / factor, a value
    written twice followed by how many more times it repeats."""
    deltas = np.diff(np.round(np.asarray(samples) * factor).astype(np.int64), prepend=0)
    edges = np.r_[0, np.flatnonzero(np.diff(deltas)) + 1, deltas.size]
    values = []
    for a, b in zip(edges[:-1], edges[1:]):
        values += [deltas[a]] if b - a == 1 else [deltas[a], deltas[a], b - a - 2]
    path.write_text(f"[DEFINITIONS]\nSAMPLES={deltas.size}\nHORIDELTA={spacing_us}\n"
                    f"VERTFACTOR={factor}\nNUCLEUS=1H\nNOMINALFREQUENCY={F0_MHZ * 1e6:.0f}\n"
                    f"[VALUES]\n" + "\n".join(map(str, values)) + "\n;  end\n",
                    encoding="latin-1")


def error(ours, theirs):
    """The largest spectral difference of two line lists over the reference's peak, after one
    complex scale; each list's lines are moved in time by its own shift first."""
    t = np.arange(2048) / 4000.0
    s, r = (np.fft.fft((np.asarray(a)[:, None] * np.exp(
        2j * np.pi * np.asarray(f)[:, None] * (t - shift) - 6.0 * t)).sum(axis=0))
        for f, a, shift in (ours, theirs))
    scale = np.vdot(s, r) / np.vdot(s, s)
    return float(np.abs(r - scale * s).max() / np.abs(r).max())


def simulation(folder, refocusing, phases):
    """A simulation's folder written from a design in the voxel's frame, with the voxel off the
    isocentre: a sinc 90 under z, its rephasing lobe and a crusher, `refocusing` under its own
    gradient, the crusher again, the ADC; the pulses' own phase written where `phases`. Returns
    the design."""
    excitation = ShapedPulse.sinc(90.0, 0.0024, offset_ppm=3.0, phase_deg=20.0, steps=120,
                                  gradient_hz_per_mm=150.0, gradient_axis="z", extent_mm=30.0)
    rephase = -0.5 * 150.0 * 0.0024
    design = Sequence(excitation, Delay(0.002), Gradient(rephase, 30.0, axis="z"),
                      Gradient(0.9, 30.0, axis="x"), Delay(0.002), refocusing, Delay(0.003),
                      Gradient(0.9, 30.0, axis="x"), Delay(0.003))
    centre = np.array([12.0, -9.0, 7.0])                 # mm along the gradient axes
    starts, n = (0.010, 0.0164), 30000                   # 30 ms on the microsecond raster
    grads = np.zeros((3, n // 10))                       # Hz/mm on the 10 us raster

    def lobe(axis, start, duration, moment):
        grads["xyz".index(axis), round(start * 1e5):round((start + duration) * 1e5)] \
            += moment / duration

    lobe("z", 0.0126, 0.001, rephase)
    lobe("x", 0.0138, 0.001, 0.9)
    lobe("x", 0.0222, 0.001, 0.9)
    volts, own, nco, rows = np.zeros(n // 5), np.zeros(n // 5), np.zeros(n), []
    for start, pulse, name, flip in zip(starts, (excitation, refocusing), ("rf_exc", "rf_ref"),
                                        (90.0, 180.0)):
        steps = np.asarray(pulse.envelope).size
        dt = pulse.duration_s / steps
        slope = pulse.gradient_hz_per_mm * (np.ones(steps) if pulse.gradient_shape is None
                                            else np.asarray(pulse.gradient_shape))
        g = grads["xyz".index(pulse.gradient_axis)]
        g[round(start * 1e5):round((start + pulse.duration_s) * 1e5)] = np.repeat(
            slope, round(dt * 1e5))
        sweep = np.zeros(steps) if pulse.frequency_hz is None else np.asarray(pulse.frequency_hz)
        inside = 360.0 * (np.cumsum(sweep) - sweep / 2.0) * dt
        envelope = np.asarray(pulse.envelope, dtype=float)
        stretch = slice(round(start / 5e-6), round((start + pulse.duration_s) / 5e-6))
        volts[stretch] = np.repeat(np.abs(envelope) * pulse.played_peak_hz / 4.0
                                   / np.abs(envelope).max(), round(dt / 5e-6))
        own[stretch] = np.repeat((inside + np.where(envelope < 0, 180.0, 0.0) + 180.0) % 360.0
                                 - 180.0, round(dt / 5e-6))
        # the oscillator in the laboratory: the pulse's phase and carrier in the voxel's
        # frame plus what the gradients give the voxel centre, the slab's position with it;
        # under a modulated gradient the position rides in the pulse's own phase, which a
        # simulation without phases does not hold, and the frame stands at the pulse's centre
        moved = np.concatenate([[0.0], np.cumsum(centre @ grads) * 1e-5])
        t = np.arange(round(pulse.duration_s * 1e6) + 1) * 1e-6
        held = not phases and pulse.gradient_shape is not None
        frame = np.interp(start + (pulse.duration_s / 2.0 if held else t),
                          np.arange(moved.size) * 1e-5, moved)
        carrier = (pulse.offset_ppm - 4.65) * F0_MHZ
        nco[round(start * 1e6):round(start * 1e6) + t.size] = (
            (pulse.phase_deg + 360.0 * (carrier * t + frame) + 180.0) % 360.0 - 180.0)
        rows.append(f"|  {round((start - 0.001) * 1e6)} | [{name}]:  {flip}/"
                    f"{round(pulse.duration_s * 1e6)} | [ph]: 0/ 0.00 |")
    adc = np.zeros(n)
    adc[27200:] = 1.0
    stem = str(folder / "Simulation_")
    write_dsv(pathlib.Path(stem + "RFD.dsv"), volts, 5, 1000)
    if phases:
        write_dsv(pathlib.Path(stem + "RFP.dsv"), own, 5, 100000)
    write_dsv(pathlib.Path(stem + "NC1.dsv"), nco, 1, 100000)
    write_dsv(pathlib.Path(stem + "ADC.dsv"), adc, 1, 1000000)
    for axis, g in zip("XYZ", grads):
        write_dsv(pathlib.Path(stem + f"GR{axis}.dsv"), g / GAMMA, 10, 1000000)
    pathlib.Path(stem + "INF.dsv").write_text("[INFO]\n* EventBlock 1000 29000\n"
                                              + "\n".join(rows) + "\n")
    (folder / "Simulation.pro").write_text(
        'tSequenceFileName\t = \t"C:\\seq\\svs_test"\nsSpecPara.lVectorSize\t = \t1024\n'
        f"sSpecPara.sVoI.sPosition.dSag\t = \t{centre[0]}\n"
        f"sSpecPara.sVoI.sPosition.dCor\t = \t{-centre[1]}\n"
        f"sSpecPara.sVoI.sPosition.dTra\t = \t{-centre[2]}\n"
        "sSpecPara.sVoI.sNormal.dCor\t = \t1.0\nsSpecPara.sVoI.dThickness\t = \t16.0\n"
        "sSpecPara.sVoI.dPhaseFOV\t = \t20.0\nsSpecPara.sVoI.dReadoutFOV\t = \t25.0\n")
    return design


#**************************************************************************************************#
#                                        Class TestSimulation                                      #
#**************************************************************************************************#
class TestSimulation:

    #*****************#
    #   with phases   #
    #*****************#
    def test_a_written_simulation_plays_as_its_design(self, tmp_path):
        # a swept refocusing pulse under a flat gradient, the voxel off the isocentre: read
        # back, the design's lines to a part in a thousand
        refocusing = ShapedPulse.hyperbolic_secant(
            0.0048, 4000.0, 900.0, steps=120, offset_ppm=3.0, phase_deg=70.0,
            gradient_hz_per_mm=200.0, gradient_axis="x", extent_mm=30.0)
        design = simulation(tmp_path, refocusing, phases=True)
        read = read_dsv(tmp_path, steps=120, extent_mm=30.0)
        assert read.name == "svs_test" and read.pulses == ("rf_exc", "rf_ref")
        assert read.npoints == 1024 and abs(read.dwell_s - 0.0028 / 1024) < 1e-12
        assert abs(read.duration_s - design.duration_s) < 1e-9
        assert abs(read.events[5].peak_hz - 900.0) < 0.5
        engine = Engine(F0_MHZ / GAMMA, isochromats=64)
        assert error((*engine.lines(read, LAC), 0.0), (*engine.lines(design, LAC), 0.0)) < 1e-3

    #********************#
    #   without phases   #
    #********************#
    def test_without_phases_the_read_is_refused_or_restored(self, tmp_path):
        # the magnitude alone: refused as it stands; restored, the sinc has its signs back and
        # the pulse under the modulated gradient its sweep, over the slab the voxel gives it
        refocusing = ShapedPulse.goia_wurst(
            0.0048, 8000.0, 800.0, steps=120, offset_ppm=3.0, phase_deg=70.0,
            gradient_hz_per_mm=500.0, gradient_axis="y", extent_mm=30.0)
        design = simulation(tmp_path, refocusing, phases=False)
        with pytest.raises(ValueError, match="RFP"):
            read_dsv(tmp_path, steps=120)
        read = read_dsv(tmp_path, steps=120, extent_mm=30.0, restore=True)
        excitation, restored = read.events[0], read.events[5]
        sinc = np.asarray(design.events[0].envelope)
        assert excitation.flip_deg == 90.0 and abs(excitation.offset_ppm - 3.0) < 1e-3
        assert np.abs(excitation.envelope - sinc / np.abs(sinc).max()).max() < 2e-3
        assert abs(restored.peak_hz - 800.0) < 0.5 and abs(restored.offset_ppm - 3.0) < 1e-3
        assert restored.gradient_axis == "y"
        assert np.abs(restored.gradient_hz_per_mm * restored.gradient_shape
                      - 500.0 * refocusing.gradient_shape).max() < 1e-2
        assert np.abs(restored.frequency_hz - refocusing.frequency_hz).max() < 0.01 * 8000.0
        engine = Engine(F0_MHZ / GAMMA, isochromats=64)
        assert error((*engine.lines(read, LAC), 0.0), (*engine.lines(design, LAC), 0.0)) < 1e-2

    #*******************#
    #   the semi-LASER  #
    #*******************#
    def test_the_public_semi_laser_reads_as_the_library_design(self):
        # Bacon, Jezzard and Clarke's simulation of the CMRR semi-LASER (github jbbacon/
        # NWS_Spectroscopy), written without the pulses' phase: five pulses centred 1.7 ppm
        # below the nominal frequency, the refocusing ones the library's GOIA-W(16, 4) at the
        # 19.27 uT the simulator's log states, and lactate as the library's semi-LASER gives
        # it once each echo's own distance from its readout is taken off
        if not SLASER.exists():
            pytest.skip("the semi-LASER simulation is not placed")
        read = read_dsv(SLASER, extent_mm=30.0, restore=True, start_s=0.8)
        pulses = [event for event in read.events if isinstance(event, ShapedPulse)]
        assert read.name == "svs_slaser_dkd" and read.npoints == 4096
        assert read.pulses == ("rf_ro", "rf_WURST_sl", "rf_WURST_ph", "rf_WURST_ph", "rf_WURST_sl")
        assert [round(pulse.duration_s * 1e6) for pulse in pulses] == [2600] + [4500] * 4
        assert all(abs(pulse.offset_ppm - 2.95) < 0.02 for pulse in pulses)
        assert [pulse.gradient_axis for pulse in pulses] == list("zyxxy")
        goia = ShapedPulse.goia_wurst(0.0045, 10000.0, 1.0, steps=128, gradient_hz_per_mm=1.0)
        for pulse in pulses[1:]:
            assert abs(pulse.peak_hz / GAMMA - 19.27) < 0.2
            assert abs(abs(pulse.gradient_hz_per_mm) - 500.0) < 0.5
            assert np.abs(pulse.envelope - goia.envelope).max() < 0.02
            assert np.abs(pulse.gradient_shape - goia.gradient_shape).max() < 0.01
            assert np.abs(pulse.frequency_hz - goia.frequency_hz).max() < 0.01 * 10000.0
        library = LocalizedSemiLaser(0.030, read.f0_mhz, reference_ppm=2.95)
        played = {}
        for name, design, isochromats in (("read", read.as_whole(), 12), ("library", library, 64)):
            engine = Engine(read.f0_mhz / GAMMA, isochromats=isochromats)
            # the echo's distance from the readout: a singlet's phase over its frequency
            totals = [np.asarray(engine.lines(design, SpinSystem(
                name="singlet", nucleus="1H", shifts_ppm=(ppm,), j_hz=((0,),), scale=1.0,
                gamma_mhz_t=GAMMA))[1]).sum() for ppm in (2.0, 4.0)]
            shift = np.angle(totals[1] / totals[0]) / (2.0 * math.pi * 2.0 * read.f0_mhz)
            played[name] = (*engine.lines(design, LAC), shift)
        assert -3.0e-4 < played["read"][2] < -1.5e-4
        assert error(played["read"], played["library"]) < 0.05
