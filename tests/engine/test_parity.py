####################################################################################################
#                                          test_parity.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Ring 1, cross-engine parity. Two independent oracles validate the engine: an in-test    #
#          complex128 reference implementation (own Pauli/kron/expm/masks, sharing nothing with    #
#          the engine but the input values), and Clarke's fsl_denmatsim run on identical spin      #
#          systems and sequences. Convention adapter between the engines, derived not fitted:      #
#          their receiver applies exp(-i pi/2) where ours applies exp(+i pi/2) (ratio -1), and     #
#          their thermal state weighs 1.0 per proton where ours weighs 0.5, so ours equals -0.5    #
#          times theirs.                                                                           #
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

from mrsandbox import Engine, Press, PulseAcquire
from mrsandbox.spins.systems import SpinSystem

B0_T = 3.0
GAMMA = 42.5774785
NPOINTS = 2048
SW_HZ = 4000.0
LW_HZ = 2.0


#********************#
#   system helpers   #
#********************#
def denmatsim_values(name):
    """Shifts and symmetric J from denmatsim's own table, shared by both engines."""
    fsl = pytest.importorskip("fsl_denmatsim")
    table = json.loads((pathlib.Path(fsl.__file__).parent / "spinSystems.json").read_text())
    record = table[name]
    j = np.asarray(record["j"], dtype=float)
    return np.asarray(record["shifts"], dtype=float), j + j.T


def our_system(shifts, j):
    return SpinSystem(name="parity", nucleus="1H", shifts_ppm=tuple(shifts),
                      j_hz=tuple(map(tuple, j.tolist())), scale=1.0, gamma_mhz_t=GAMMA)


def our_fid(sequence, shifts, j):
    engine = Engine(B0_T)
    freqs, amps = engine.lines(sequence, our_system(shifts, j))
    _, fid = engine.render_fid(freqs, amps, npoints=NPOINTS, dwell_s=1.0 / SW_HZ, lw_hz=LW_HZ)
    return np.asarray(ops.convert_to_numpy(fid), dtype=np.complex128)


#**************************************************************************************************#
#                                    Class TestReferenceOracle                                     #
#**************************************************************************************************#
#                                                                                                  #
# The engine against a from-scratch complex128 implementation.                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestReferenceOracle:
    """The engine against a from-scratch complex128 implementation."""

    #******************************#
    #   reference implementation   #
    #******************************#
    @staticmethod
    def reference_fid(sequence_events, shifts, j):
        """Independent simulation: own operators, scipy expm, time-stepped readout."""
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
        for kind, value in sequence_events:
            if kind == "pulse":
                flip_deg, phase_deg = value
                axis = np.cos(np.radians(phase_deg)) * fx + np.sin(np.radians(phase_deg)) * fy
                u = scipy.linalg.expm(-2j * np.pi * axis * (flip_deg / 360.0))
                rho = u @ rho @ u.conj().T
            elif kind == "delay":
                u = scipy.linalg.expm(-2j * np.pi * h * value)
                rho = u @ rho @ u.conj().T
            elif kind == "filter":
                rho = np.where(order == value, rho, 0.0)

        u_dwell = scipy.linalg.expm(-2j * np.pi * h / SW_HZ)
        fplus = fx + 1j * fy
        fid = np.empty(NPOINTS, dtype=complex)
        for k in range(NPOINTS):
            fid[k] = np.trace(fplus @ rho)
            rho = u_dwell @ rho @ u_dwell.conj().T
        t = np.arange(NPOINTS) / SW_HZ
        return 1j * fid * np.exp(-np.pi * LW_HZ * t)  # the same +90 degree receiver

    #*******************#
    #   pulse acquire   #
    #*******************#
    def test_pulse_acquire_lactate(self):
        shifts, j = denmatsim_values("sysLac")
        a = our_fid(PulseAcquire(), shifts, j)
        b = self.reference_fid([("pulse", (90.0, 0.0)), ("filter", -1)], shifts, j)
        assert np.abs(a - b).max() / np.abs(a).max() < 1e-3

    #********************************#
    #   press with strong coupling   #
    #********************************#
    def test_press_glutamate(self):
        shifts, j = denmatsim_values("sysGlu")
        te1, te2 = 0.011, 0.019
        a = our_fid(Press(te1, te2), shifts, j)
        b = self.reference_fid(
            [("pulse", (90.0, 0.0)), ("filter", -1), ("delay", te1 / 2),
             ("pulse", (180.0, 90.0)), ("filter", +1), ("delay", (te1 + te2) / 2),
             ("pulse", (180.0, 90.0)), ("filter", -1), ("delay", te2 / 2)],
            shifts, j,
        )
        assert np.abs(a - b).max() / np.abs(a).max() < 1e-3


#**************************************************************************************************#
#                                    Class TestDenmatsimParity                                     #
#**************************************************************************************************#
#                                                                                                  #
# The engine against Clarke's fsl_denmatsim on identical inputs.                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestDenmatsimParity:
    """The engine against Clarke's fsl_denmatsim on identical inputs."""

    #*******************#
    #   denmatsim run   #
    #*******************#
    @staticmethod
    def denmatsim_fid(system_name, pulses, delays, coherence_filter):
        fsl = pytest.importorskip("fsl_denmatsim")
        from fsl_denmatsim.simseq import simseq

        table = json.loads((pathlib.Path(fsl.__file__).parent / "spinSystems.json").read_text())
        time, samples = 1e-6, 4
        blocks = [
            {"time": time, "amp": [flip / 360.0 / time] * samples, "phase": [0.0] * samples,
             "grad": [0.0, 0.0, 0.0], "frequencyOffset": 0.0, "phaseOffset": np.radians(phase)}
            for flip, phase in pulses
        ]
        parameters = {
            "sequenceName": "parity", "description": "", "B0": B0_T, "centralShift": 0.0,
            "Rx_Points": NPOINTS, "Rx_SW": SW_HZ, "Rx_LW": LW_HZ, "Rx_Phase": np.pi / 2,
            "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "resolution": [1, 1, 1],
            "CoherenceFilter": coherence_filter, "RFUnits": "Hz", "GradUnits": "mT",
            "spaceUnits": "mm", "RF": blocks, "delays": delays,
            "rephaseAreas": [[0.0, 0.0, 0.0]] * len(blocks),
        }
        fid, _, _ = simseq(table[system_name], parameters, verbose=False)
        return -0.5 * np.asarray(fid, dtype=np.complex128).squeeze()  # the derived adapter

    #****************#
    #   comparison   #
    #****************#
    def compare(self, system_name, sequence, pulses, delays, coherence_filter):
        shifts, j = denmatsim_values(system_name)
        a = our_fid(sequence, shifts, j)
        b = self.denmatsim_fid(system_name, pulses, delays, coherence_filter)
        corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        err = np.abs(a - b).max() / np.abs(a).max()
        assert corr > 0.9999, f"complex correlation {corr:.6f}"
        assert err < 0.02, f"relative max error {err:.4f}"  # floor: their 1 us finite pulses

    #***********#
    #   cases   #
    #***********#
    def test_pulse_acquire_lactate(self):
        self.compare("sysLac", PulseAcquire(), [(90.0, 0.0)], [0.0], [-1])

    def test_press_lactate(self):
        te1, te2 = 0.011, 0.019
        self.compare("sysLac", Press(te1, te2),
                     [(90.0, 0.0), (180.0, 90.0), (180.0, 90.0)],
                     [te1 / 2, (te1 + te2) / 2, te2 / 2], [-1, 1, -1])

    def test_pulse_acquire_glutamate(self):
        self.compare("sysGlu", PulseAcquire(), [(90.0, 0.0)], [0.0], [-1])


#**************************************************************************************************#
#                                      Class TestMegaParity                                        #
#**************************************************************************************************#
#                                                                                                  #
# Cross-engine parity for J-difference editing: MEGA-PRESS on GABA with shaped editing pulses.     #
#                                                                                                  #
#**************************************************************************************************#
class TestMegaParity:

    #**********************#
    #   edited sequence    #
    #**********************#
    def test_mega_press_on_gaba(self):
        from mrsandbox import CoherenceFilter, Delay, Pulse, Sequence, ShapedPulse

        fsl = pytest.importorskip("fsl_denmatsim")
        from fsl_denmatsim.simseq import simseq

        table = json.loads((pathlib.Path(fsl.__file__).parent / "spinSystems.json").read_text())
        record = table["sysGABA"]
        shifts = np.asarray(record["shifts"], dtype=float)
        j = np.asarray(record["j"], dtype=float)
        j = j + j.T
        system = SpinSystem(name="parity", nucleus="1H", shifts_ppm=tuple(shifts),
                            j_hz=tuple(map(tuple, j.tolist())), scale=1.0, gamma_mhz_t=GAMMA)

        steps, edit_dur, edit_ppm = 64, 0.014, 1.9
        tau = ((np.arange(steps) + 0.5) / steps) * 2.0 - 1.0
        env = np.exp(np.log(0.01) * tau * tau)
        f1 = env * (180.0 / 360.0) / (env.sum() * (edit_dur / steps))
        taus = [0.005, 0.017, 0.017, 0.017, 0.012]

        def edit():
            return ShapedPulse(180.0, edit_dur, env.astype(np.float32), offset_ppm=edit_ppm)

        sequence = Sequence(Pulse(90.0), CoherenceFilter(-1), Delay(taus[0]),
                            Pulse(180.0, phase_deg=90.0), CoherenceFilter(+1), Delay(taus[1]),
                            edit(), Delay(taus[2]),
                            Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(taus[3]),
                            edit(), Delay(taus[4]))
        engine = Engine(B0_T)
        freqs, amps = engine.lines(sequence, system)
        _, ours = engine.render_fid(freqs, amps, npoints=NPOINTS, dwell_s=1.0 / SW_HZ, lw_hz=LW_HZ)
        ours = np.asarray(ops.convert_to_numpy(ours), dtype=np.complex128)

        time, samples = 1e-6, 4
        hard = lambda flip, phase: {"time": time, "amp": [flip / 360.0 / time] * samples,
                                    "phase": [0.0] * samples, "grad": [0.0] * 3,
                                    "frequencyOffset": 0.0, "phaseOffset": np.radians(phase)}
        editor = {"time": edit_dur, "amp": f1.tolist(), "phase": [0.0] * steps,
                  "grad": [0.0] * 3, "frequencyOffset": edit_ppm * GAMMA * B0_T,
                  "phaseOffset": 0.0}
        parameters = {
            "sequenceName": "mega", "description": "", "B0": B0_T, "centralShift": 0.0,
            "Rx_Points": NPOINTS, "Rx_SW": SW_HZ, "Rx_LW": LW_HZ, "Rx_Phase": np.pi / 2,
            "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "resolution": [1, 1, 1],
            "CoherenceFilter": [-1, 1, None, -1, None], "RFUnits": "Hz", "GradUnits": "mT",
            "spaceUnits": "mm",
            "RF": [hard(90, 0), hard(180, 90), dict(editor), hard(180, 90), dict(editor)],
            "delays": taus, "rephaseAreas": [[0.0] * 3] * 5,
        }
        fid, _, _ = simseq(record, parameters, verbose=False)
        theirs = -0.5 * np.asarray(fid, dtype=np.complex128).squeeze()

        corr = np.abs(np.vdot(ours, theirs)) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
        err = np.abs(ours - theirs).max() / np.abs(ours).max()
        assert corr > 0.9999, f"complex correlation {corr:.6f}"
        assert err < 0.02, f"relative max error {err:.4f}"
