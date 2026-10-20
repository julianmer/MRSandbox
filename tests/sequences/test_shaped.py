####################################################################################################
#                                          test_shaped.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 04/09/26                                                                                #
#                                                                                                  #
# Purpose: Shaped-pulse gate: the piecewise propagation reduces to the hard pulse in the short     #
#          limit, calibrates the on-resonance flip exactly, is frequency selective with the right  #
#          carrier sign, matches an independent complex128 oracle off resonance on a coupled       #
#          system, matches fsl_denmatsim on identical waveform samples, and stays differentiable   #
#          through the envelope.                                                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json
import pathlib

import numpy as np
import pytest
import scipy.linalg
from keras import ops

from mrsandbox import (CoherenceFilter, Engine, Pulse, PulseAcquire, Sequence, ShapedPulse,
                       SpinSystem)
from mrsandbox.database import pulses
from mrsandbox.phantom.water import WaterSource

B0_T = 3.0
GAMMA = 42.5774785
F0_MHZ = GAMMA * B0_T
NPOINTS = 2048
SW_HZ = 4000.0
LW_HZ = 2.0


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(B0_T)


def total_amplitude(engine, sequence, system):
    _, amps = engine.lines(sequence, system)
    return float(np.abs(ops.convert_to_numpy(amps)).sum())


def acquire(pulse):
    return Sequence(pulse, CoherenceFilter(-1))


#**************************************************************************************************#
#                                       Class TestShapedPulse                                      #
#**************************************************************************************************#
class TestShapedPulse:

    #**********************#
    #   hard pulse limit   #
    #**********************#
    def test_short_flat_envelope_is_the_hard_pulse(self, engine):
        for system in SpinSystem.from_database("Lac"):
            ideal = total_amplitude(engine, acquire(Pulse(90.0)), system)
            shaped = total_amplitude(
                engine, acquire(ShapedPulse(90.0, 1e-6, np.ones(8, np.float32))), system
            )
            assert abs(shaped - ideal) < 1e-4 * ideal

    #*****************#
    #   calibration   #
    #*****************#
    def test_on_resonance_flip_is_calibrated(self, engine):
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        full = total_amplitude(engine, PulseAcquire(), water)
        excited = total_amplitude(
            engine, acquire(ShapedPulse.gaussian(90.0, 0.010, offset_ppm=4.65)), water
        )
        inverted = total_amplitude(
            engine, acquire(ShapedPulse.gaussian(180.0, 0.010, offset_ppm=4.65)), water
        )
        assert abs(excited - full) < 1e-3 * full
        assert inverted < 0.05 * full

    #*****************#
    #   selectivity   #
    #*****************#
    def test_carrier_sign_and_selectivity(self, engine):
        # a 20 ms gaussian at the water position excites water in full, barely touches
        # the NAA acetyl singlet 336 Hz away, and a wrong-sign carrier touches nothing
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        (acetyl,) = [s for s in SpinSystem.from_database("NAA") if "acetyl" in s.name.lower()]
        full = total_amplitude(engine, PulseAcquire(), water)
        plain = total_amplitude(engine, PulseAcquire(), acetyl)
        at_water = ShapedPulse.gaussian(90.0, 0.020, offset_ppm=4.65)
        mirrored = ShapedPulse.gaussian(90.0, 0.020, offset_ppm=-4.65)
        assert total_amplitude(engine, acquire(at_water), water) > 0.99 * full
        assert total_amplitude(engine, acquire(at_water), acetyl) < 0.01 * plain
        assert total_amplitude(engine, acquire(mirrored), water) < 0.01 * full

    #***********************#
    #   complex128 oracle   #
    #***********************#
    def test_off_resonance_oracle_parity_on_glutamate(self, engine):
        """Independent simulation: own operators, scipy expm, same waveform samples."""
        fsl = pytest.importorskip("fsl_denmatsim")
        table = json.loads((pathlib.Path(fsl.__file__).parent / "spinSystems.json").read_text())
        record = table["sysGlu"]
        shifts = np.asarray(record["shifts"], dtype=float)
        j = np.asarray(record["j"], dtype=float)
        j = j + j.T

        flip_deg, duration, offset_ppm = 90.0, 0.005, 3.0
        steps = 32
        tau = ((np.arange(steps) + 0.5) / steps) * 2.0 - 1.0
        env = np.exp(np.log(0.01) * tau * tau)

        n = len(shifts)
        sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
        sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

        def embed(op, i):
            out = np.eye(1, dtype=complex)
            for k in range(n):
                out = np.kron(out, op if k == i else np.eye(2, dtype=complex))
            return out

        ix = [embed(sx, i) for i in range(n)]
        iy = [embed(sy, i) for i in range(n)]
        iz = [embed(sz, i) for i in range(n)]
        fx, fy, fz = sum(ix), sum(iy), sum(iz)
        h = sum(s * GAMMA * B0_T * z for s, z in zip(shifts, iz))
        for a in range(n):
            for b in range(a + 1, n):
                if j[a, b]:
                    h = h + j[a, b] * (ix[a] @ ix[b] + iy[a] @ iy[b] + iz[a] @ iz[b])
        m = np.real(np.diag(fz))
        order = np.rint(m[:, None] - m[None, :]).astype(int)

        rho = fz / 2 ** (n - 1)
        dt = duration / steps
        f1 = env * (flip_deg / 360.0) / (env.sum() * dt)
        phase = 2.0 * np.pi * (offset_ppm * GAMMA * B0_T) * (np.arange(steps) + 0.5) * dt
        for k in range(steps):
            h_k = h + f1[k] * (np.cos(phase[k]) * fx + np.sin(phase[k]) * fy)
            u = scipy.linalg.expm(-2j * np.pi * h_k * dt)
            rho = u @ rho @ u.conj().T
        rho = np.where(order == -1, rho, 0.0)

        u_dwell = scipy.linalg.expm(-2j * np.pi * h / SW_HZ)
        fplus = fx + 1j * fy
        fid = np.empty(NPOINTS, dtype=complex)
        for k in range(NPOINTS):
            fid[k] = np.trace(fplus @ rho)
            rho = u_dwell @ rho @ u_dwell.conj().T
        t = np.arange(NPOINTS) / SW_HZ
        reference = 1j * fid * np.exp(-np.pi * LW_HZ * t)  # the same +90 degree receiver

        system = SpinSystem(name="parity", nucleus="1H", shifts_ppm=tuple(shifts),
                            j_hz=tuple(map(tuple, j.tolist())), scale=1.0, gamma_mhz_t=GAMMA)
        sequence = acquire(ShapedPulse(flip_deg, duration, env.astype(np.float32),
                                       offset_ppm=offset_ppm))
        freqs, amps = engine.lines(sequence, system)
        _, ours = engine.render_fid(freqs, amps, npoints=NPOINTS, dwell_s=1.0 / SW_HZ, lw_hz=LW_HZ)
        ours = np.asarray(ops.convert_to_numpy(ours), dtype=np.complex128)
        assert np.abs(ours - reference).max() / np.abs(reference).max() < 1e-3

    #**********************#
    #   denmatsim parity   #
    #**********************#
    def test_denmatsim_parity_on_lactate(self, engine):
        fsl = pytest.importorskip("fsl_denmatsim")
        from fsl_denmatsim.simseq import simseq

        table = json.loads((pathlib.Path(fsl.__file__).parent / "spinSystems.json").read_text())

        flip_deg, duration, steps = 90.0, 0.002, 32
        tau = ((np.arange(steps) + 0.5) / steps) * 2.0 - 1.0
        env = np.exp(np.log(0.01) * tau * tau)
        dt = duration / steps
        f1 = env * (flip_deg / 360.0) / (env.sum() * dt)

        parameters = {
            "sequenceName": "parity", "description": "", "B0": B0_T, "centralShift": 0.0,
            "Rx_Points": NPOINTS, "Rx_SW": SW_HZ, "Rx_LW": LW_HZ, "Rx_Phase": np.pi / 2,
            "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "resolution": [1, 1, 1],
            "CoherenceFilter": [-1], "RFUnits": "Hz", "GradUnits": "mT", "spaceUnits": "mm",
            "RF": [{"time": duration, "amp": f1.tolist(), "phase": [0.0] * steps,
                    "grad": [0.0, 0.0, 0.0], "frequencyOffset": 0.0, "phaseOffset": 0.0}],
            "delays": [0.0], "rephaseAreas": [[0.0, 0.0, 0.0]],
        }
        fid, _, _ = simseq(table["sysLac"], parameters, verbose=False)
        theirs = -0.5 * np.asarray(fid, dtype=np.complex128).squeeze()  # the derived adapter

        record = table["sysLac"]
        shifts = np.asarray(record["shifts"], dtype=float)
        j = np.asarray(record["j"], dtype=float)
        j = j + j.T
        system = SpinSystem(name="parity", nucleus="1H", shifts_ppm=tuple(shifts),
                            j_hz=tuple(map(tuple, j.tolist())), scale=1.0, gamma_mhz_t=GAMMA)
        sequence = acquire(ShapedPulse(flip_deg, duration, env.astype(np.float32)))
        freqs, amps = engine.lines(sequence, system)
        _, ours = engine.render_fid(freqs, amps, npoints=NPOINTS, dwell_s=1.0 / SW_HZ, lw_hz=LW_HZ)
        ours = np.asarray(ops.convert_to_numpy(ours), dtype=np.complex128)

        corr = np.abs(np.vdot(ours, theirs)) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
        err = np.abs(ours - theirs).max() / np.abs(ours).max()
        assert corr > 0.9999, f"complex correlation {corr:.6f}"
        assert err < 0.02, f"relative max error {err:.4f}"

    #***********************#
    #   differentiability   #
    #***********************#
    def test_gradient_flows_through_the_envelope(self, engine, needs_autograd):
        from nifti_mrs_plus.backend import value_and_grad

        water = WaterSource.system(gamma_mhz_t=GAMMA)

        def residual(envelope):
            sequence = acquire(ShapedPulse(180.0, 0.010, envelope, offset_ppm=4.65))
            _, amps = engine.lines(sequence, water)
            return ops.sum(ops.abs(amps))

        envelope = ops.convert_to_tensor(np.full(16, 0.5, dtype=np.float32))
        value, grad = value_and_grad(residual)(envelope)
        grad = ops.convert_to_numpy(grad)
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0

    def test_envelope_gradient_is_finite_on_equivalent_spins(self, engine, needs_autograd):
        # creatine's equivalent protons give degenerate eigenvalues, where the
        # derivative of an eigendecomposition divides by zero; the matmul-only
        # propagator has no such term, so waveform optimization works on any system
        from nifti_mrs_plus.backend import value_and_grad

        (cr, *_) = SpinSystem.from_database("Cr")

        def residual(envelope):
            sequence = acquire(ShapedPulse(90.0, 0.004, envelope, offset_ppm=3.0))
            _, amps = engine.lines(sequence, cr)
            return ops.sum(ops.abs(amps))

        envelope = ops.convert_to_tensor(np.full(16, 0.5, dtype=np.float32))
        _, grad = value_and_grad(residual)(envelope)
        grad = ops.convert_to_numpy(grad)
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0


#**************************************************************************************************#
#                                      Class TestPulseProfiles                                     #
#**************************************************************************************************#
#                                                                                                  #
# Excitation and inversion bandwidths reproduce the published bandwidth-length products.           #
#                                                                                                  #
#**************************************************************************************************#
class TestPulseProfiles:

    #*******************#
    #   r value sweep   #
    #*******************#
    @staticmethod
    def r_value(engine, factory, angle, **kw):
        """FWHM bandwidth times pulse length, on Mxy for 90 and on the Mz dip for 180."""
        duration, f0 = 0.010, GAMMA * B0_T
        # sweep +-0.8 of the published FWHM: the half-max crossings sit at +-0.5 FWHM
        fwhm_hz = pulses.DE_GRAAF[(kw.pop("published"), angle)]["r_bw"] / duration
        offsets_hz = np.linspace(-0.8 * fwhm_hz, 0.8 * fwhm_hz, 121)
        fz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
        response = []
        for offset in offsets_hz:
            probe = SpinSystem(name="probe", nucleus="1H", shifts_ppm=(offset / f0,),
                               j_hz=((0.0,),), scale=1.0, gamma_mhz_t=GAMMA)
            ctx = engine.context(probe)
            rho = Sequence(factory(angle, duration, **kw)).run(
                ops.convert_to_tensor(probe.thermal_rho()), ctx
            )
            rho = np.asarray(ops.convert_to_numpy(rho))
            fplus = np.asarray(ops.convert_to_numpy(ctx.fplus))
            if angle == 90:
                response.append(abs(np.trace(fplus @ rho)))
            else:
                response.append(0.5 - np.trace(fz @ rho).real)
        response = np.asarray(response)
        above = offsets_hz[response >= response.max() / 2.0]
        return (above.max() - above.min()) * duration

    #********************#
    #   de graaf table   #
    #********************#
    def test_bandwidths_match_de_graaf_table_5_1(self, engine):
        cases = (
            (ShapedPulse.gaussian, dict(truncation=0.01), "gaussian_1pct"),
            (ShapedPulse.gaussian, dict(truncation=0.10), "gaussian_10pct"),
            (ShapedPulse.sinc, dict(lobes=3), "sinc"),
            (ShapedPulse.hermite, dict(), "hermite"),
        )
        for factory, kw, name in cases:
            for angle in (90, 180):
                published = pulses.DE_GRAAF[(name, angle)]["r_bw"]
                measured = self.r_value(engine, factory, angle, published=name, **kw)
                assert abs(measured - published) < 0.15, (name, angle, measured)


#**************************************************************************************************#
#                                        Class TestAdiabatic                                       #
#**************************************************************************************************#
#                                                                                                  #
# Hyperbolic secant inversion is B1 and offset insensitive, and semi-LASER inherits it.            #
#                                                                                                  #
#**************************************************************************************************#
class TestAdiabatic:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def inversion(engine, pulse, b1=1.0):
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        ctx = engine.context(water, b1_scale=b1)
        rho = Sequence(pulse).run(ops.convert_to_tensor(water.thermal_rho()), ctx)
        fz = np.asarray(ops.convert_to_numpy(ctx.fz))
        start = np.trace(fz @ np.asarray(water.thermal_rho())).real
        return float(np.trace(fz @ np.asarray(ops.convert_to_numpy(rho))).real / start)

    #***************#
    #   inversion   #
    #***************#
    def test_inversion_is_b1_and_offset_insensitive(self, engine):
        for scale in (1.0, 1.4, 1.8):
            pulse = ShapedPulse.hyperbolic_secant(0.008, 1500.0, 700.0 * scale,
                                                  offset_ppm=4.65)
            assert self.inversion(engine, pulse) < -0.99
        for offset_hz in (-400.0, 400.0):
            pulse = ShapedPulse.hyperbolic_secant(0.008, 1500.0, 700.0,
                                                  offset_ppm=4.65 + offset_hz / F0_MHZ)
            assert self.inversion(engine, pulse) < -0.96
        # a hard 180 at 1.4 times nominal has already lost most of its inversion
        assert self.inversion(engine, Pulse(180.0), b1=1.4) > -0.5

    def test_adiabatic_profile_is_symmetric_in_offset(self, engine):
        # the sweep phase is sampled at the same step midpoints as the envelope; a
        # half-step lead of the frequency over the amplitude skews the profile, and what
        # remains is the discretization error
        for steps, bound in ((64, 1e-2), (256, 1e-3)):
            for offset_hz in (300.0, 500.0):
                mz = [self.inversion(engine, ShapedPulse.hyperbolic_secant(
                          0.008, 1500.0, 700.0, steps=steps,
                          offset_ppm=4.65 + sign * offset_hz / F0_MHZ))
                      for sign in (+1.0, -1.0)]
                assert abs(mz[0] - mz[1]) < bound

    #*********************#
    #   shinnar le roux   #
    #*********************#
    def test_slr_pulses_hold_their_slab(self):
        # a 90 of time-bandwidth 6 and a spin-echo 180 of 6.24 over a 20 mm slab: flat
        # inside, half amplitude at the nominal edge, a clean stopband, and the slab
        # integral of the nominal width; the peak B1 is the design's own
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        engine = Engine(B0_T)
        x = np.linspace(-25.0, 25.0, 201)
        plain = np.abs(np.asarray(ops.convert_to_numpy(
            engine.lines(Sequence(Pulse(90.0), CoherenceFilter(-1)), water)[1]))).max()

        def profile(sequence):
            _, amps = engine.profile(sequence, water, "x", x)
            return np.abs(amps).max(axis=1) / plain

        slope = 6.0 / 0.0026 / 20.0
        ninety = ShapedPulse.slr(90.0, 0.0026, 6.0, "excitation", offset_ppm=4.65, steps=128,
                                 gradient_hz_per_mm=slope, gradient_axis="x", extent_mm=50.0)
        from mrsandbox import Gradient

        excited = profile(Sequence(ninety, Gradient(-0.52 * slope * 0.0026, 50.0, axis="x"),
                                   CoherenceFilter(-1)))
        slope = 6.24 / 0.0065 / 20.0
        one_eighty = ShapedPulse.slr(180.0, 0.0065, 6.24, "refocusing", offset_ppm=4.65,
                                     phase_deg=90.0, steps=128, gradient_hz_per_mm=slope,
                                     gradient_axis="x", extent_mm=50.0)
        refocused = profile(Sequence(Pulse(90.0), CoherenceFilter(+1), one_eighty,
                                     CoherenceFilter(-1)))
        for kept in (excited, refocused):
            assert kept[100 - 28:100 + 29].min() > 0.95            # to 7 mm: the passband
            assert 0.4 < kept[140] < 0.6 and 0.4 < kept[60] < 0.6  # the edge at 10 mm
            assert kept[:40].max() < 0.02 and kept[161:].max() < 0.02  # beyond 15 mm
            assert abs(kept[:-1].mean() * 2.5 - 1.0) < 0.05
        assert (abs(ninety.peak_hz / GAMMA - 13.9) < 0.5
                and abs(one_eighty.peak_hz / GAMMA - 24.5) < 0.5)
        # the minimum-phase 90: the same slab in magnitude, its main lobe late, less to
        # rephase, and a tenth of the slab integral lost to its nonlinear phase
        late = ShapedPulse.slr(90.0, 0.0026, 6.0, "excitation", offset_ppm=4.65, steps=128,
                               minimum_phase=True, gradient_hz_per_mm=6.0 / 0.0026 / 20.0,
                               gradient_axis="x", extent_mm=50.0)
        assert np.argmax(np.abs(np.asarray(late.envelope))) > 0.7 * 128
        slope = 6.0 / 0.0026 / 20.0
        _, amps = engine.profile(Sequence(late, Gradient(-0.23 * slope * 0.0026, 50.0, axis="x"),
                                          CoherenceFilter(-1)), water, "x", x)
        kept = np.abs(amps).max(axis=1) / plain
        assert kept[100 - 28:100 + 29].min() > 0.95 and kept[:40].max() < 0.03
        line = np.argmax(np.abs(amps[100]))
        assert 0.8 < abs(amps[:-1, line].mean()) * 2.5 / plain < 0.95
        with pytest.raises(ValueError):
            ShapedPulse.slr(90.0, 0.0026, 6.0, "inversion")

    #****************#
    #   goia wurst   #
    #****************#
    def test_goia_wurst_inverts_a_10_khz_slab_at_15_ut(self):
        # the across-vendor 3 T refocusing pulse (Deelchand 2021: GOIA-W(16,4), 4.5 ms,
        # 10 kHz, 15 uT) under its modulated gradient: the slab inverts, three slab
        # widths average to a third, and a plain hyperbolic secant of the same
        # length, bandwidth, and peak under a flat gradient cannot follow
        water = WaterSource.system(gamma_mhz_t=GAMMA)
        slab = dict(offset_ppm=4.65, gradient_hz_per_mm=10000.0 / 20.0, gradient_axis="z")

        def mz(pulse):
            engine = Engine(B0_T, isochromats=64)
            sequence = Sequence(pulse, CoherenceFilter(0), Pulse(90.0))
            _, amps = engine.lines(sequence, water)
            return float(np.real(ops.convert_to_numpy(amps)).sum())

        full = mz(ShapedPulse.gaussian(1e-4, 0.001, offset_ppm=4.65))
        goia = lambda extent: ShapedPulse.goia_wurst(0.0045, 10000.0, GAMMA * 15.0,
                                                     extent_mm=extent, **slab)
        assert mz(goia(16.0)) / full < -0.97
        assert abs(mz(goia(60.0)) / full - 1.0 / 3.0) < 0.03
        secant = ShapedPulse.hyperbolic_secant(0.0045, 10000.0, GAMMA * 15.0, extent_mm=16.0,
                                               **slab)
        assert mz(secant) / full > -0.7
        # the gradient dips to 1 - f where the B1 peaks, and both are timings
        pulse = goia(16.0)
        assert abs(float(np.min(pulse.gradient_shape)) - 0.15) < 1e-3
        assert abs(float(np.max(pulse.gradient_shape)) - 1.0) < 1e-3
        assert pulse.structure() == ShapedPulse.goia_wurst(0.004, 8000.0, 500.0, extent_mm=16.0,
                                                           **slab).structure()

    #****************#
    #   semi-laser   #
    #****************#
    def test_the_semi_laser_echo_time_counts_its_pulses(self):
        # the events of a 30 ms semi-LASER last 30 ms, its four adiabatic pulses included,
        # and a weakly coupled doublet has evolved for that long: each line by pi J TE
        # (41.6 degrees at 7.7 Hz), not by the 24 ms of pulses more (74.8)
        from mrsandbox import SemiLaser
        from mrsandbox.sequences.events import Delay

        design = SemiLaser(te_s=0.030)
        played = sum(float(e.t_s) if isinstance(e, Delay) else float(getattr(e, "duration_s", 0.0))
                     for e in design.events)
        assert abs(played - 0.030) < 1e-9
        pair = SpinSystem(name="ax", nucleus="1H", shifts_ppm=(0.91, 2.04),
                          j_hz=((0.0, 7.7), (7.7, 0.0)), scale=1.0, gamma_mhz_t=GAMMA)
        f, a = (np.asarray(ops.convert_to_numpy(x)) for x in Engine(3.0).lines(design, pair))
        strong = (f / (3.0 * GAMMA) < 1.2) & (np.abs(a) > 0.2 * np.abs(a).max())
        assert np.all(np.abs(np.abs(np.degrees(np.angle(a[strong]))) - 41.6) < 8.0)
        with pytest.raises(ValueError):
            SemiLaser(te_s=0.020)

    def test_semi_laser_keeps_the_echo_where_press_loses_it(self, engine):
        from mrsandbox import Press, SemiLaser

        water = WaterSource.system(gamma_mhz_t=GAMMA)

        def kept(sequence, b1):
            full = total_amplitude(engine, sequence, water)
            _, a = engine.lines(sequence, water, b1_scale=b1)
            return float(np.abs(ops.convert_to_numpy(a)).sum()) / full

        press, slaser = Press(te1_s=0.015, te2_s=0.015), SemiLaser(te_s=0.030)
        assert kept(slaser, 0.7) > 0.85  # at the excitation-only bound sin(63 degrees)
        assert kept(press, 0.7) < 0.65   # PRESS refocusing collapses
