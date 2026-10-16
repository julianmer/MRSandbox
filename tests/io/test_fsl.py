####################################################################################################
#                                           test_fsl.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/10/26                                                                                #
#                                                                                                  #
# Purpose: FSL-MRS sequence import gate: a description read into the event primitives and played   #
#          by the engine gives FSL-MRS's own simulation of it, and WIN's published basis.          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json
import math
import pathlib

import numpy as np
import pytest

from mrsandbox import Engine, SpinSystem
from mrsandbox.io import read_fsl_sequence

GAMMA = 42.5774785
LAC = {"shifts": [4.0974, 1.3142, 1.3142, 1.3142], "scaleFactor": 1,
       "j": [[0, 6.933, 6.933, 6.933], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
WIN = (pathlib.Path(__file__).resolve().parents[2] / "data" / "win-mrs-basis-sets" / "basis_sets"
       / "eja_svs_press" / "35ms_low_res")


#*************#
#   helpers   #
#*************#
def systems(spins, name):
    """FSL-MRS spin systems (its upper-triangular couplings; a singlet's may be bare numbers) as
    the package's."""
    out = []
    for s in spins if isinstance(spins, list) else [spins]:
        j = np.atleast_2d(np.asarray(s["j"], dtype=float))
        out.append(SpinSystem(name=name, nucleus="1H",
                              shifts_ppm=tuple(np.atleast_1d(s["shifts"]).astype(float)),
                              j_hz=tuple(map(tuple, j + j.T)),
                              scale=float(s.get("scaleFactor", 1.0)), gamma_mhz_t=GAMMA))
    return tuple(out)


def error(ours, theirs):
    """The largest spectral difference over the reference's peak, after one complex scale (a
    global phase and size the two need not share)."""
    s, r = np.fft.fft(ours), np.fft.fft(theirs)
    scale = np.vdot(s, r) / np.vdot(s, s)
    return float(np.abs(r - scale * s).max() / np.abs(r).max())


def played(description, spins, name="Lac"):
    """The description played by the engine at its own field and resolution, as its FID."""
    sequence = read_fsl_sequence(description)
    engine = Engine(sequence.b0_t, isochromats=sequence.resolution[0])
    return sequence.fid(*(np.asarray(v) for v in engine.lines(sequence, *systems(spins, name))))


def press(refocusing):
    """A PRESS description in FSL-MRS's form at 2.89 T: a signed sinc 90 under x and two
    refocusing pulses under y and z, the carrier 150 Hz off the centre, the 90's rephase lobe."""
    tau = (np.arange(200) + 0.5) / 200 * 2.0 - 1.0
    sinc = np.sinc(3.0 * tau)
    excite = {"time": 0.0026, "frequencyOffset": -150.0, "phaseOffset": 0.0,
              "grad": [4.0, 0.0, 0.0], "amp": (np.abs(sinc) * 1200.0).tolist(),
              "phase": np.where(sinc < 0, math.pi, 0.0).tolist()}
    pulses = [excite] + [dict(refocusing, grad=[0.0, g, 0.0] if k == 0 else [0.0, 0.0, g])
                         for k, g in enumerate((refocusing["g"],) * 2)]
    for pulse in pulses[1:]:
        pulse.pop("g")
    return {"sequenceName": "test", "B0": 2.89, "centralShift": 4.65, "Rx_Points": 2048,
            "Rx_SW": 4000, "Rx_LW": 2.0, "Rx_Phase": 0.0, "x": [-20, 20], "y": [-20, 20],
            "z": [-20, 20], "resolution": [21, 21, 21], "CoherenceFilter": [-1, 1, -1],
            "RFUnits": "Hz", "GradUnits": "mT", "spaceUnits": "mm", "RF": pulses,
            "delays": [0.006, 0.014, 0.008],
            "rephaseAreas": [[-4.0 * 0.0013, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}


#**************************************************************************************************#
#                                          Class TestFsl                                           #
#**************************************************************************************************#
class TestFsl:

    #****************#
    #   simulation   #
    #****************#
    def test_a_description_plays_as_fsl_mrs_plays_it(self):
        # the engine on the read description against FSL-MRS's simulator on the same dict: a signed
        # sinc refocusing pair, then an adiabatic pair whose phase track is no sign flip, played
        # without a gradient so the position grids (FSL-MRS's ends, the engine's midpoints) do not
        # enter its sharp slab edges
        simseq = pytest.importorskip("fsl_denmatsim.simseq").simseq
        tau = (np.arange(200) + 0.5) / 200 * 2.0 - 1.0
        sinc = np.sinc(3.0 * tau)
        signed = {"time": 0.0052, "frequencyOffset": -150.0, "phaseOffset": math.pi / 2, "g": 1.4,
                  "amp": (np.abs(sinc) * 1100.0).tolist(),
                  "phase": np.where(sinc < 0, math.pi, 0.0).tolist()}
        beta, half = 5.3, 0.5 * 2000.0
        sech = 1.0 / np.cosh(beta * tau)
        sweep = half * np.tanh(beta * tau)
        hs = {"time": 0.006, "frequencyOffset": -150.0, "phaseOffset": 0.0, "g": 0.0,
              "amp": (sech * 900.0).tolist(),
              "phase": (2.0 * math.pi * np.cumsum(sweep) * 0.006 / 200).tolist()}
        for refocusing in (signed, hs):
            description = press(refocusing)
            theirs, _, _ = simseq(dict(LAC, name="Lac"), json.loads(json.dumps(description)),
                                  verbose=False)
            assert error(played(description, LAC), np.asarray(theirs)) < 0.01

    #*******************#
    #   a WIN sequence  #
    #*******************#
    def test_win_cmrr_press_gives_win_basis(self):
        # WIN's description of the CMRR PRESS at 2.89 T, TE 35 (Clarke's win-mrs-basis-sets, a
        # local copy in data/) against the basis FSL-MRS made of it: one phase and size for all
        if not WIN.exists():
            pytest.skip("WIN's basis sets are not in data/")
        description = next(WIN.glob("eja_svs_press_*.json"))
        phases = []
        for name in ("Cr", "Lac", "Glu"):
            basis = json.loads((WIN / "eja_svs_press_35.0" / f"{name}.json").read_text())
            theirs = (np.asarray(basis["basis"]["basis_re"])
                      + 1j * np.asarray(basis["basis"]["basis_im"]))
            ours = played(json.loads(description.read_text()), basis["spinSys"], name)
            assert error(ours, theirs) < 0.005
            phases.append(np.angle(np.vdot(np.fft.fft(ours), np.fft.fft(theirs))))
        assert np.ptp(np.unwrap(phases)) < 0.02
