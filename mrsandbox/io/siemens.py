####################################################################################################
#                                            siemens.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: Siemens raw spectroscopy (TWIX .dat, read by mapVBVD) as the package's readers take a   #
#          run: the samples as (points, coils, transients) without the receiver's oversampling,    #
#          the dwell, the carrier and what the header says of the scanner. And Siemens sequence    #
#          simulations (the DSV files of IDEA's simulator) read into the event primitives.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import functools
import math
import pathlib
import re
import struct
import tempfile

import numpy as np

from mrsandbox.database.nuclei import GAMMA_MHZ_PER_T
from mrsandbox.sequences.events import Delay, Gradient, Sequence, ShapedPulse


#*****************#
#   measurement   #
#*****************#
@functools.lru_cache(maxsize=1)
def _measurement(path):
    """The file's last measurement (a file holds its adjustments first) as mapVBVD reads it:
    the samples squeezed with the oversampling kept, their dimension names, the header. A
    file whose measurement table does not land on its scan is read through `_mended`."""
    import mapvbvd

    try:
        twix = mapvbvd.mapVBVD(path, quiet=True)
    except IndexError:
        twix = mapvbvd.mapVBVD(_mended(path), quiet=True)
    twix = twix[-1] if isinstance(twix, list) else twix
    image = twix.image
    image.squeeze = True
    image.flagRemoveOS = False
    return np.asarray(image[""]), list(image.sqzDims), twix.hdr


#************#
#   mended   #
#************#
def _mended(path):
    """A copy of a multi-measurement file whose table lists the last measurement alone, at
    the offset where its header's buffers actually begin, in the temporary folder. A deposit's
    file can have lost bytes before its scan (OpenNeuro ds006812's sub-01 acq-06 is four
    bytes short with its table 516 bytes off the scan's header), and mapVBVD then reads the
    adjustments' MDH chain off the end of the file; the scan itself is whole."""
    raw = bytearray(pathlib.Path(path).read_bytes())
    # a header starts with its length and its buffer count, then the first buffer's name
    starts = [m.start() - 4 for m in re.finditer(rb"[\x01-\x20]\x00\x00\x00Config\x00", raw)]
    count = struct.unpack_from("<I", raw, 4)[0]
    if count < 2 or not starts:
        raise ValueError(f"{path}: mapVBVD does not read it and its table cannot be mended")
    entry = bytes(raw[8 + 152 * (count - 1):8 + 152 * count])
    raw[8:8 + 152] = entry
    struct.pack_into("<I", raw, 4, 1)
    struct.pack_into("<QQ", raw, 16, starts[-1], len(raw) - starts[-1])
    copy = pathlib.Path(tempfile.gettempdir()) / f"mrsandbox-mended-{pathlib.Path(path).name}"
    copy.write_bytes(bytes(raw))
    return str(copy)


#***************#
#   read twix   #
#***************#
def read_twix(path, phs=1):
    """A TWIX file as ((points, coils, transients), dwell_s, f0_mhz, header), the transients
    every stored line that holds a sample, the oversampling taken off by the central half of
    the spectrum; `phs` picks the slot of a file that stores two scans along Phs (the water
    reference in slot 0 beside the run in 1). Needs mapVBVD (pip install mapvbvd)."""
    data, dims, header = _measurement(str(path))
    if "Phs" in dims:
        data = np.take(data, phs, axis=dims.index("Phs"))
        dims = [d for d in dims if d != "Phs"]
    order = ([dims.index("Col"), dims.index("Cha")]
             + [k for k, d in enumerate(dims) if d not in ("Col", "Cha")])
    x = np.transpose(data, order)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x[:, :, np.abs(x).sum(axis=(0, 1)) > 0]
    n = x.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)[n // 4: n // 4 + n // 2]
    x = np.fft.ifft(np.fft.ifftshift(spectrum, axes=0), axis=0)
    yaps, dicom = header["MeasYaps"], header["Dicom"]
    said = {"Manufacturer": dicom.get("Manufacturer"),
            "ManufacturersModelName": dicom.get("ManufacturersModelName"),
            "SoftwareVersions": dicom.get("SoftwareVersions"),
            "SequenceName": (str(header["Meas"].get("tSequenceFileName", "")).split("\\")[-1]
                             if "Meas" in header else None),
            "EchoTime": float(yaps[("alTE", "0")]) * 1e-6,
            "RepetitionTime": float(yaps[("alTR", "0")]) * 1e-6}
    return (x, float(yaps[("sRXSPEC", "alDwellTime", "0")]) * 2e-9,
            float(yaps[("sTXSPEC", "asNucleusInfo", "0", "lFrequency")]) * 1e-6, said)


#**************************************************************************************************#
#                                       Class SiemensSequence                                      #
#**************************************************************************************************#
#                                                                                                  #
# A Sequence read from a Siemens sequence simulation, carrying its readout.                        #
#                                                                                                  #
#**************************************************************************************************#
class SiemensSequence(Sequence):
    """A Sequence read from a Siemens sequence simulation, carrying its readout.

    Attributes:
        name: The protocol's sequence file name.
        f0_mhz: The simulation's nominal frequency.
        npoints, dwell_s: The readout (the protocol's vector size over the ADC's window).
        pulses: The event table's name of every pulse played, in order.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, events, name, f0_mhz, npoints, dwell_s, pulses):
        super().__init__(*events)
        self.name = name
        self.f0_mhz = f0_mhz
        self.npoints = npoints
        self.dwell_s = dwell_s
        self.pulses = pulses


#*********#
#   dsv   #
#*********#
def _dsv(path):
    """A DSV file's samples and their spacing in seconds: the values are differences in units
    of 1 / VERTFACTOR, and a value written twice is followed by how many more times it
    repeats."""
    text = pathlib.Path(path).read_text(encoding="latin-1")
    said = dict(re.findall(r"^(\w+)=(\S+)", text, flags=re.M))
    values = np.array(text.split("[VALUES]")[1].split(";")[0].split(), dtype=np.int64)
    runs, k = [], 0
    while k < values.size:
        if k + 1 < values.size and values[k] == values[k + 1]:
            runs.append(np.full(2 + values[k + 2], values[k]))
            k += 3
        else:
            runs.append(values[k:k + 1])
            k += 1
    return (np.cumsum(np.concatenate(runs)) / float(said["VERTFACTOR"]),
            float(said["HORIDELTA"]) * 1e-6, said)


#************#
#   signed   #
#************#
def _signed(magnitude):
    """A real waveform from its magnitude: the runs of equal samples are the pulse's own
    samples, and each takes the sign the line through the two before it predicts, so the
    sign turns where the magnitude runs through zero and stays where it only dips; the
    largest lobe is positive."""
    edges = np.r_[0, np.flatnonzero(np.diff(magnitude)) + 1, magnitude.size]
    centre = (edges[:-1] + edges[1:] - 1) / 2.0
    values = magnitude[edges[:-1]].astype(float)
    for k in range(2, values.size):
        predicted = values[k - 1] + ((values[k - 1] - values[k - 2]) * (centre[k] - centre[k - 1])
                                     / (centre[k - 1] - centre[k - 2]))
        if predicted < 0.0 or (predicted == 0.0 and values[k - 1] < 0.0):
            values[k] = -values[k]
    signed = np.repeat(values, np.diff(edges))
    return signed * np.sign(signed[np.abs(signed).argmax()])


#**************#
#   read dsv   #
#**************#
def read_dsv(folder, center_ppm=4.65, steps=128, extent_mm=20.0, restore=False, start_s=0.0,
             readout=0):
    """Read a Siemens sequence simulation (the DSV files IDEA's simulator writes beside its
    .pro protocol) into a SiemensSequence of event primitives, as the scanner would play it.

    The folder holds the RF magnitude in volts (RFD), the oscillator's phase (NC1: every
    pulse's carrier and phase), the three gradients (GRX, GRY, GRZ), the ADC, the event table
    (INF: every pulse's start, duration and flip) and the protocol (.pro: the voxel), and,
    where the simulation was told to write it, the pulses' own phase (RFP). Each pulse of the
    table becomes a ShapedPulse resampled at `steps` under the gradient that plays during it
    (tilted where the voxel is, modulated where the gradient is), the gradients between two
    pulses a kick per axis of their exact area in the middle of the delay, and what plays
    before the first pulse is left out. Everything is carried in the voxel's own frame: the
    gradients' phase at the voxel centre (the protocol's position, head first supine) is taken
    off every pulse, whose carrier is then the shift it is centred on, with the simulation's
    nominal frequency at `center_ppm`.

    The volts need the transmitter's reference, which the files do not hold: a pulse whose
    phase only turns by half turns (amplitude modulated) plays at the flip the event table
    states for it, any other at its volts times the hertz per volt of the last such pulse
    before it.

    Without RFP the phase inside a pulse is not in the files and the read is refused, unless
    `restore` allows the two cases the magnitude decides: a pulse under a flat gradient or
    none is taken as real, its signs restored where the magnitude runs through zero, and a
    pulse under a modulated gradient as a GOIA pulse, whose frequency is its gradient times
    the running integral of B1^2 / G (Andronesi et al., J Magn Reson 203 (2010) 283) spanning
    the slab the protocol's voxel gives it. A complex pulse under a flat gradient cannot be
    told from a real one there.

    Args:
        folder: The simulation's folder.
        center_ppm: Position of the nominal frequency on the ppm axis.
        steps: Uniform resampling steps per RF pulse.
        extent_mm: Isochromat span handed to gradient-bearing events.
        restore: True reads a simulation without RFP, as stated above.
        start_s: Pulses that start before this time are left out (preparation that is not
            wanted, outer-volume bands).
        readout: Index of the ADC window whose readout the sequence carries; its pulses are
            those after the window before it.
    """
    folder = pathlib.Path(folder)
    stem = str(next(folder.glob("*_RFD.dsv")))[:-7]
    volts, dt_rf, said = _dsv(stem + "RFD.dsv")
    nco, dt_nco, _ = _dsv(stem + "NC1.dsv")
    adc, dt_adc, _ = _dsv(stem + "ADC.dsv")
    own = pathlib.Path(stem + "RFP.dsv")
    if not own.exists() and not restore:
        raise ValueError(f"{folder}: the simulation wrote no RF phase (RFP); simulate with it "
                         f"on, or read with restore=True (real and GOIA pulses only)")
    own = _dsv(own)[0] if own.exists() else None
    # a gradient in mT/m is gamma in MHz/T times it in Hz/mm; the moments in cycles/mm run
    # from the simulation's start to each sample
    grads = [_dsv(stem + f"GR{axis}.dsv") for axis in "XYZ"]
    dt_g = grads[0][1]
    hz_per_mm = GAMMA_MHZ_PER_T[said["NUCLEUS"]] * np.stack([g for g, _, _ in grads])
    moment = np.concatenate([np.zeros((3, 1)), np.cumsum(hz_per_mm, axis=1) * dt_g], axis=1)
    protocol = dict(re.findall(r"^(\S+)\s*=\s*(.+?)\s*$", next(folder.glob("*.pro")).read_text(
        encoding="latin-1"), flags=re.M))
    voi = {key: float(protocol.get("sSpecPara.sVoI." + key, 0.0))
           for key in ("sPosition.dSag", "sPosition.dCor", "sPosition.dTra", "sNormal.dSag",
                       "sNormal.dCor", "sNormal.dTra", "dThickness", "dPhaseFOV", "dReadoutFOV")}
    # patient to gradient axes, head first supine: x is sagittal, y and z run against
    # coronal and transverse
    patient = np.array([1.0, -1.0, -1.0])
    centre = patient * [voi["sPosition.dSag"], voi["sPosition.dCor"], voi["sPosition.dTra"]]
    # the voxel's edges: its thickness along the normal, its phase edge across the normal in
    # the coronal-transverse plane of a mostly transverse voxel and in the sagittal-coronal
    # plane of any other, its readout edge across both
    sag, cor, tra = voi["sNormal.dSag"], voi["sNormal.dCor"], voi["sNormal.dTra"]
    normal = patient * [sag, cor, tra]
    phase_axis = patient * ([0.0, tra, -cor] if abs(tra) > max(abs(sag), abs(cor))
                            else [cor, -sag, 0.0])
    phase_axis /= np.linalg.norm(phase_axis)
    edges = ((normal, voi["dThickness"]), (phase_axis, voi["dPhaseFOV"]),
             (np.cross(normal, phase_axis), voi["dReadoutFOV"]))

    def unwrapped(degrees):
        return np.unwrap(np.radians(degrees))

    def moved(t):
        """The gradients' moment per axis from the simulation's start to the time t."""
        return np.array([np.interp(t, np.arange(m.size) * dt_g, m) for m in moment])

    turns = np.flatnonzero(np.diff(np.r_[0, adc > 0, 0]))
    opens, closes = turns[::2] * dt_adc, turns[1::2] * dt_adc
    if not 0 <= readout < opens.size:
        raise ValueError(f"readout {readout} does not exist: the simulation holds "
                         f"{opens.size} ADC window(s)")
    first = max(start_s, closes[readout - 1] if readout else 0.0)
    table = []
    for block in pathlib.Path(stem + "INF.dsv").read_text(encoding="latin-1").split("[INFO]"):
        head = re.search(r"\* EventBlock (\d+)", block)
        for row in re.finditer(r"\|\s*(\d+) \|\s*\[(\w+)(?:\[\d+\])?\]:\s*([\d.]+)/\s*(\d+) \|",
                               block):
            start = (int(head[1]) + int(row[1])) * 1e-6
            if first <= start < opens[readout]:
                table.append((start, int(row[4]) * 1e-6, row[2], float(row[3])))
    if not table:
        raise ValueError(f"{folder}: no pulse plays before readout {readout}")
    table.sort()
    events, hz_per_volt = [], None
    for k, (start, duration, name, flip) in enumerate(table):
        end = start + duration
        dt = duration / steps
        t = (np.arange(steps) + 0.5) * dt

        def sampled(samples, spacing, through=np.asarray):
            """The pulse's stretch of a waveform, taken through a function, at the step
            midpoints."""
            stretch = through(samples[int(round(start / spacing)):int(round(end / spacing)) + 1])
            return np.interp(t, np.arange(stretch.size) * spacing, stretch)

        course = np.stack([sampled(g, dt_g) for g in hz_per_mm])
        peak = np.abs(course).sum(axis=0).argmax()
        strength = float(np.linalg.norm(course[:, peak]))
        along = course[:, peak] / strength if strength else None
        shape = None if along is None else along @ course / strength
        modulated = shape is not None and np.ptp(shape) > 1e-3
        envelope = sampled(volts, dt_rf)
        # the phase in the voxel's frame: the oscillator's less the gradients' at the centre.
        # Under a modulated gradient the slab's position rides in the pulse's own phase:
        # where the files do not hold it the oscillator carries the shift alone, and the
        # frame stands where it is at the pulse's centre
        held = own is None and modulated
        psi = sampled(nco, dt_nco, unwrapped) - 2.0 * math.pi * centre @ (
            moved(start + duration / 2.0)[:, None] if held else moved(start + t))
        inside = np.zeros(steps)
        if own is not None:
            inside = sampled(own, dt_rf, unwrapped)
        elif modulated:
            thickness = edges[int(np.argmax([abs(along @ axis) for axis, _ in edges]))][1]
            power = envelope ** 2 / shape
            run = np.cumsum(power) - power / 2.0
            frequency = strength * thickness * shape * (run / power.sum() - 0.5)
            inside = 2.0 * math.pi * (np.cumsum(frequency) - frequency / 2.0) * dt
        else:
            envelope = sampled(volts, dt_rf, _signed)
        carrier = float(np.polyfit(t, psi, 1)[0]) / (2.0 * math.pi)
        rest = psi + inside - 2.0 * math.pi * carrier * t
        rest = rest - rest[0]
        halves = np.round(rest / math.pi)
        phase_deg = math.degrees(psi[0] + inside[0] - 2.0 * math.pi * carrier * t[0])
        if np.allclose(rest, halves * math.pi, atol=1e-2):   # amplitude modulated
            envelope = envelope * np.where(halves % 2 == 0, 1.0, -1.0)
            if envelope.sum() < 0.0:
                envelope, phase_deg = -envelope, phase_deg + 180.0
            hz_per_volt = flip / 360.0 / (envelope.sum() * dt)
            sweep, peak_hz = None, None
        else:
            if hz_per_volt is None:
                raise ValueError(f"{name}: no amplitude-modulated pulse before it gives the "
                                 f"volts their hertz")
            # the first step takes the track's first slope, every later one what its
            # midpoint's phase asks for
            cycles = rest / (2.0 * math.pi * dt)
            sweep = np.full(steps, cycles[1])
            for step in range(1, steps):
                sweep[step] = 2.0 * (cycles[step] + sweep[0] / 2.0 - sweep[:step].sum())
            phase_deg -= 180.0 * sweep[0] * dt
            peak_hz = float(envelope.max()) * hz_per_volt
        tilted = along is not None and np.abs(along).max() < 1.0 - 1e-6
        axis = "x" if along is None else "xyz"[int(np.abs(along).argmax())]
        events.append(ShapedPulse(
            None if peak_hz else flip, duration,
            (envelope / np.abs(envelope).max()).astype(np.float32),
            offset_ppm=center_ppm + carrier / float(said["NOMINALFREQUENCY"]) * 1e6,
            phase_deg=phase_deg, peak_hz=peak_hz,
            frequency_hz=None if sweep is None else sweep.astype(np.float32),
            gradient_hz_per_mm=(0.0 if along is None else strength if tilted
                                else float(course["xyz".index(axis), peak])),
            gradient_axis=axis, extent_mm=extent_mm,
            gradient_shape=shape.astype(np.float32) if modulated else None,
            gradient_normal=along if tilted else None))
        # to the next pulse, or the ADC: the gradients' area as kicks in the delay's middle
        until = table[k + 1][0] if k + 1 < len(table) else opens[readout]
        kicks = [Gradient(float(area), extent_mm, axis=axis)
                 for axis, area in zip("xyz", moved(until) - moved(end))
                 if abs(area) * extent_mm > 1e-3]
        if kicks:
            events.extend([Delay((until - end) / 2.0), *kicks, Delay((until - end) / 2.0)])
        elif until - end > 1e-9:
            events.append(Delay(until - end))
    npoints = int(protocol["sSpecPara.lVectorSize"])
    return SiemensSequence(
        events, name=protocol["tSequenceFileName"].strip('"').split("\\")[-1],
        f0_mhz=float(said["NOMINALFREQUENCY"]) * 1e-6, npoints=npoints,
        dwell_s=float(closes[readout] - opens[readout]) / npoints,
        pulses=tuple(name for _, _, name, _ in table))
