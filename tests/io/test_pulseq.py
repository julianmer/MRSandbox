####################################################################################################
#                                          test_pulseq.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/09/26                                                                                #
#                                                                                                  #
# Purpose: Pulseq import gate: .seq files written by pypulseq translate to the event primitives    #
#          (RF to shaped pulses in Hz, trapezoids to gradient kicks, ADC to the readout) and       #
#          reproduce the native-event physics through the engine.                                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

pypulseq = pytest.importorskip("pypulseq")

from mrsandbox import CoherenceFilter, Delay, Engine, Gradient, Pulse, Sequence, SpinSystem
from mrsandbox.io import read_seq
from mrsandbox.sequences.events import ShapedPulse


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


@pytest.fixture(scope="module")
def system():
    system = pypulseq.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    return system


def spectrum(engine, sequence, systems):
    freqs, amps = engine.lines(sequence, *systems)
    _, spec = engine.render_spectrum(freqs, amps, 127.73, npoints=4096, sw_hz=4000.0, lw_hz=3.0)
    return np.asarray(ops.convert_to_numpy(spec))


def correlation(a, b):
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


#**************************************************************************************************#
#                                       Class TestPulseqImport                                     #
#**************************************************************************************************#
class TestPulseqImport:

    #*******************#
    #   pulse acquire   #
    #*******************#
    def test_pulse_acquire_matches_the_native_hard_pulse(self, engine, system, tmp_path):
        seq = pypulseq.Sequence(system=system)
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi / 2, duration=1e-4, system=system))
        seq.add_block(pypulseq.make_adc(num_samples=512, dwell=2.5e-4, system=system))
        path = tmp_path / "pulse_acquire.seq"
        seq.write(str(path))

        imported = read_seq(path)
        assert imported.npoints == 512 and abs(imported.dwell_s - 2.5e-4) < 1e-12
        lac = SpinSystem.from_database("Lac")
        native = Sequence(Pulse(90.0), CoherenceFilter(-1))
        corr = correlation(spectrum(engine, imported, lac), spectrum(engine, native, lac))
        assert corr > 0.999  # a 0.1 ms block pulse is the hard pulse to this tolerance

    #****************#
    #   spin echo    #
    #****************#
    def test_crushed_spin_echo_matches_the_filtered_native(self, engine, system, tmp_path):
        # 90 - tau - crusher - 180 - crusher - tau - adc: the crusher pair selects the
        # inverted pathway physically; the native reference uses coherence filters
        d90, d180, crush_s, tau = 1e-4, 2e-4, 2e-3, 0.020
        seq = pypulseq.Sequence(system=system)
        crusher = pypulseq.make_trapezoid(channel="z", area=2000.0, duration=crush_s, system=system)
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi / 2, duration=d90, system=system))
        seq.add_block(pypulseq.make_delay(tau))
        seq.add_block(crusher)
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi, duration=d180,
                                                phase_offset=np.pi / 2, system=system))
        seq.add_block(crusher)
        seq.add_block(pypulseq.make_delay(tau))
        seq.add_block(pypulseq.make_adc(num_samples=512, dwell=2.5e-4, system=system))
        path = tmp_path / "spin_echo.seq"
        seq.write(str(path))

        imported = read_seq(path)
        half = d90 / 2.0 + tau + crush_s + d180 / 2.0  # pulse centre to pulse centre
        lac = SpinSystem.from_database("Lac")
        native = Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(half),
                          Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(half))
        corr = correlation(spectrum(engine, imported, lac), spectrum(engine, native, lac))
        assert corr > 0.99

    #*******************#
    #   translation     #
    #*******************#
    def test_blocks_translate_to_the_expected_events(self, system, tmp_path):
        seq = pypulseq.Sequence(system=system)
        rf, gz, _ = pypulseq.make_sinc_pulse(flip_angle=np.pi / 2, duration=2e-3,
                                             slice_thickness=5e-3, apodization=0.5,
                                             time_bw_product=4, system=system, return_gz=True)
        seq.add_block(rf, gz)
        crusher = pypulseq.make_trapezoid(channel="x", area=1000.0, duration=1e-3, system=system)
        seq.add_block(crusher)
        seq.add_block(pypulseq.make_delay(5e-3))
        seq.add_block(pypulseq.make_adc(num_samples=256, dwell=5e-4, system=system))
        path = tmp_path / "blocks.seq"
        seq.write(str(path))

        imported = read_seq(path, steps=64)
        pulses = [e for e in imported.events if isinstance(e, ShapedPulse)]
        kicks = [e for e in imported.events if isinstance(e, Gradient)]
        assert len(pulses) == 1 and len(kicks) == 1
        assert pulses[0].peak_hz > 0.0  # Pulseq amplitudes play in exact Hz
        assert abs(pulses[0].gradient_hz_per_mm - float(gz.amplitude) / 1000.0) < 1e-9
        # the file round trip quantizes the trapezoid to the raster, hence the slack
        assert abs(kicks[0].moment_per_mm - float(crusher.area) / 1000.0) < 1e-4
        delays = sum(float(e.t_s) for e in imported.events if isinstance(e, Delay))
        # the crusher and delay blocks, plus the RF block's time around the pulse
        # (gradient ramp-up before it, ringdown after it)
        around_rf = float(seq.get_block(1).block_duration) - float(rf.shape_dur)
        assert abs(delays - (1e-3 + 5e-3 + around_rf)) < 1e-6


#**************************************************************************************************#
#                                      Class TestPulseqExport                                      #
#**************************************************************************************************#
#                                                                                                  #
# Export round trips: a physical sequence written to .seq and read back reproduces its physics,    #
# and idealized events refuse to pretend they are hardware.                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestPulseqExport:

    #*******************#
    #   shaped pulse    #
    #*******************#
    def test_shaped_pulse_round_trips(self, engine, tmp_path):
        from mrsandbox.io import write_seq

        native = Sequence(ShapedPulse.gaussian(90.0, 0.004, offset_ppm=2.0), Delay(0.010))
        path = tmp_path / "gaussian.seq"
        write_seq(native, path, npoints=256, dwell_s=2.5e-4, f0_mhz=127.73)
        back = read_seq(path)
        assert back.npoints == 256 and abs(back.dwell_s - 2.5e-4) < 1e-12
        lac = SpinSystem.from_database("Lac")
        corr = correlation(spectrum(engine, back, lac), spectrum(engine, native, lac))
        assert corr > 0.999

    #*****************#
    #   spin echo     #
    #*****************#
    def test_crushed_spin_echo_round_trips(self, engine, tmp_path):
        from mrsandbox.io import write_seq

        crusher = Gradient(2.0, extent_mm=20.0)
        native = Sequence(Pulse(90.0), Delay(0.021), crusher,
                          Pulse(180.0, phase_deg=90.0), crusher, Delay(0.021))
        path = tmp_path / "echo.seq"
        write_seq(native, path, npoints=512, dwell_s=2.5e-4, f0_mhz=127.73)
        back = read_seq(path)
        lac = SpinSystem.from_database("Lac")
        corr = correlation(spectrum(engine, back, lac), spectrum(engine, native, lac))
        assert corr > 0.99

    #*******************#
    #   idealizations   #
    #*******************#
    def test_idealized_events_refuse_to_export(self, tmp_path):
        from mrsandbox.io import write_seq

        masked = Sequence(Pulse(90.0), CoherenceFilter(-1), Delay(0.01))
        with pytest.raises(ValueError, match="idealization"):
            write_seq(masked, tmp_path / "bad.seq", npoints=64, dwell_s=2.5e-4, f0_mhz=127.73)
        short = Sequence(Pulse(90.0), Delay(1e-5))
        with pytest.raises(ValueError, match="too short"):
            write_seq(short, tmp_path / "short.seq", npoints=64, dwell_s=2.5e-4, f0_mhz=127.73)


#**************************************************************************************************#
#                                     Class TestMultiAxisImport                                    #
#**************************************************************************************************#
class TestMultiAxisImport:

    #*******************#
    #   per-axis kicks  #
    #*******************#
    def test_every_gradient_axis_imports_as_its_own_kick(self, system, tmp_path):
        seq = pypulseq.Sequence(system=system)
        seq.add_block(pypulseq.make_block_pulse(flip_angle=np.pi / 2, duration=1e-4,
                                                system=system))
        gx = pypulseq.make_trapezoid(channel="x", area=1000.0, duration=1e-3, system=system)
        gz = pypulseq.make_trapezoid(channel="z", area=-500.0, duration=1.45e-3, system=system)
        seq.add_block(gx, gz)
        seq.add_block(pypulseq.make_adc(num_samples=64, dwell=2.5e-4, system=system))
        path = tmp_path / "axes.seq"
        seq.write(str(path))

        kicks = [e for e in read_seq(path).events if isinstance(e, Gradient)]
        assert {(k.axis, round(k.moment_per_mm, 3)) for k in kicks} == {("x", 1.0), ("z", -0.5)}
