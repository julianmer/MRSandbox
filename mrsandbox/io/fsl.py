####################################################################################################
#                                              fsl.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/10/26                                                                                #
#                                                                                                  #
# Purpose: FSL-MRS sequence descriptions, the JSON files fsl_mrs_sim plays, read into the event    #
#          primitives: every block's RF waveform under its slice gradient, the rephase gradient    #
#          over the delay, the coherence filter; the readout carried along.                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import json
import math
import pathlib

import numpy as np

from mrsandbox.database.nuclei import GAMMA_MHZ_PER_T
from mrsandbox.sequences.events import (CoherenceFilter, Delay, Gradient, Projection, Sequence,
                                        ShapedPulse)

# every file names its own units (FSL-MRS's defaults Hz, T and m where it does not): the RF in
# tesla or Hz, the gradients in T/m or Hz/m, the positions in m, cm or mm; the factors to T and
# mm, a field in Hz then taken over by the nucleus's gamma
TESLA = {"t": 1.0, "mt": 1e-3, "ut": 1e-6}
SPACE_UNITS = {"m": 1000.0, "cm": 10.0, "mm": 1.0}


#**************************************************************************************************#
#                                         Class FslSequence                                        #
#**************************************************************************************************#
#                                                                                                  #
# A Sequence read from an FSL-MRS sequence description, carrying its readout and simulation grid.  #
#                                                                                                  #
#**************************************************************************************************#
class FslSequence(Sequence):
    """A Sequence read from an FSL-MRS sequence description, carrying its readout and
    simulation grid.

    Attributes:
        name: The description's sequenceName.
        b0_t: Its field in tesla.
        nucleus: The nucleus it plays (FSL-MRS's Nucleus, protons where it names none).
        center_ppm: The receiver's centre (centralShift): FSL-MRS's FIDs sit with this ppm
            on the carrier, the package's at 0 ppm.
        npoints, dwell_s: The readout (Rx_Points, 1 / Rx_SW).
        linewidth_hz: The Lorentzian FSL-MRS puts on its FIDs (Rx_LW).
        receiver_phase_deg: The phase it takes off them (Rx_Phase).
        resolution: Its positions per axis (x, y, z).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, events, name, b0_t, nucleus, center_ppm, npoints, dwell_s, linewidth_hz,
                 receiver_phase_deg, resolution):
        super().__init__(*events)
        self.name = name
        self.b0_t = b0_t
        self.nucleus = nucleus
        self.center_ppm = center_ppm
        self.npoints = npoints
        self.dwell_s = dwell_s
        self.linewidth_hz = linewidth_hz
        self.receiver_phase_deg = receiver_phase_deg
        self.resolution = resolution

    #*********#
    #   fid   #
    #*********#
    def fid(self, freqs, amps):
        """The engine's lines (Hz in the package's frame, complex amplitudes) as FSL-MRS's FID of
        this readout: center_ppm on the carrier, its Lorentzian on; lines beyond the band
        dropped."""
        t = np.arange(self.npoints) * self.dwell_s
        f0_mhz = GAMMA_MHZ_PER_T[self.nucleus] * self.b0_t
        offset = np.asarray(freqs, dtype=np.float64) - self.center_ppm * f0_mhz
        inside = np.abs(offset) < 0.5 / self.dwell_s
        lines = np.exp(2j * np.pi * np.outer(t, offset[inside]))
        return lines @ np.asarray(amps, dtype=np.complex128)[inside] * np.exp(
            -np.pi * self.linewidth_hz * t)


#*************#
#   reading   #
#*************#
def read_fsl_sequence(source):
    """Read an FSL-MRS sequence description (a path to its JSON, or the dict) as fsl_mrs_sim
    plays it (fsl_denmatsim.simseq): block by block the RF pulse under its constant slice
    gradient, then the delay with the rephase gradient's area over it (a kick: the gradient
    commutes with the shifts and couplings), then the coherence filter, the readout after the
    last block. The pulse's carrier sits at centralShift plus its frequencyOffset in Hz. Each
    sample plays at its own amplitude and phase: a waveform whose phase only flips by half a
    turn becomes a signed envelope, any other phase track the frequency track whose step
    midpoints carry the samples' phases. Where every gradient axis plays in one run (FSL-MRS's
    1D mode) the ensemble is averaged as the axis changes (a Projection, as FSL-MRS averages
    its positions), else it plays whole. The positions span each axis's bounds, which must be
    centred.

    Args:
        source: The JSON's path, or the description as a dict.

    Returns:
        An FslSequence.
    """
    d = source if isinstance(source, dict) else json.loads(pathlib.Path(source).read_text())
    nucleus = d.get("Nucleus", "1H")
    gamma_hz_t = GAMMA_MHZ_PER_T[nucleus] * 1e6
    units = d.get("RFUnits", "Hz").lower()
    rf_hz = 1.0 if units == "hz" else TESLA[units] * gamma_hz_t
    units = d.get("GradUnits", "T").lower()
    grad_t_m = 1.0 / gamma_hz_t if units == "hz" else TESLA[units]
    space_mm = SPACE_UNITS[d.get("spaceUnits", "m").lower()]
    center_ppm, b0_t = float(d.get("centralShift", 0.0)), float(d["B0"])
    f0_mhz = gamma_hz_t * 1e-6 * b0_t
    extent = {}
    for axis in "xyz":
        low, high = (space_mm * float(v) for v in d[axis])
        if abs(low + high) > 1e-9 * max(abs(low), abs(high), 1.0):
            raise ValueError(f"the {axis} bounds {d[axis]} are off centre")
        extent[axis] = high - low
    hz_per_mm = grad_t_m * gamma_hz_t / 1000.0     # a gradient in the file's units to Hz/mm
    blocks = list(zip(d["RF"], d["delays"], d["rephaseAreas"], d["CoherenceFilter"]))

    # FSL-MRS's 1D mode: no pulse under two axes, and every axis in one run
    order = [axis for rf, *_ in blocks for axis in "xyz" if rf["grad"]["xyz".index(axis)]]
    multiple = any(np.count_nonzero(rf["grad"]) > 1 for rf, *_ in blocks)
    runs = [axis for k, axis in enumerate(order) if k == 0 or axis != order[k - 1]]
    one_d = not multiple and len(runs) == len(set(runs))

    events, last = [], None

    def addressed(axis):
        """A Projection before an event on a new axis, in the 1D mode."""
        nonlocal last
        if one_d and last is not None and axis != last:
            events.append(Projection())
        last = axis

    for rf, delay, rephase, coherence in blocks:
        samples = (rf_hz * float(rf.get("ampScale", 1.0)) * np.asarray(rf["amp"], dtype=float)
                   * np.exp(1j * (np.asarray(rf.get("phase", 0.0), dtype=float)
                                  + float(rf.get("phaseOffset", 0.0)))))
        dt = float(rf["time"]) / samples.size
        amplitude = np.abs(samples)
        peak = float(amplitude.max())
        phase = np.unwrap(np.angle(samples))
        flips = np.round((phase - phase[0]) / math.pi)
        if np.allclose(phase - phase[0], flips * math.pi, atol=1e-6):   # a signed waveform
            envelope, sweep = amplitude * np.where(flips % 2 == 0, 1.0, -1.0), None
        else:                       # each step's midpoint phase is the sample's own
            target = (phase - phase[0]) / (2.0 * math.pi * dt)
            sweep, total = np.zeros(samples.size), 0.0
            for k in range(samples.size):
                sweep[k] = 2.0 * (target[k] - total)
                total += sweep[k]
            envelope = amplitude
        gradient = hz_per_mm * np.asarray(rf["grad"], dtype=float)
        axis = "xyz"[int(np.argmax(np.abs(gradient)))]
        if np.any(gradient):
            addressed(axis)
        events.append(ShapedPulse(
            None, float(rf["time"]), (envelope / peak).astype(np.float32),
            offset_ppm=center_ppm + float(rf.get("frequencyOffset", 0.0)) / f0_mhz,
            phase_deg=math.degrees(float(phase[0])),
            frequency_hz=None if sweep is None else sweep.astype(np.float32), peak_hz=peak,
            gradient_hz_per_mm=(float(np.linalg.norm(gradient)) if multiple
                                else float(gradient["xyz".index(axis)])),
            gradient_axis=axis, extent_mm=extent[axis],
            gradient_normal=gradient / np.linalg.norm(gradient) if multiple and np.any(gradient)
            else None))
        for k, area in enumerate(rephase):          # the area in the file's units times s
            if area:
                addressed("xyz"[k])
                events.append(Gradient(hz_per_mm * float(area), extent["xyz"[k]], axis="xyz"[k]))
        if delay:
            events.append(Delay(float(delay)))
        if coherence is not None:
            events.append(CoherenceFilter(int(coherence)))
    return FslSequence(events, name=d.get("sequenceName"), b0_t=b0_t, nucleus=nucleus,
                       center_ppm=center_ppm,
                       npoints=int(d["Rx_Points"]), dwell_s=1.0 / float(d["Rx_SW"]),
                       linewidth_hz=float(d.get("Rx_LW", 0.0)),
                       receiver_phase_deg=math.degrees(float(d.get("Rx_Phase", 0.0))),
                       resolution=tuple(int(v) for v in np.broadcast_to(d["resolution"], 3)))
