####################################################################################################
#                                               run.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: What a run's recordings say of the subject over the transients, read off any scan that  #
#          has them: the breathing and the heart from a physiological trace (a belt and a pulse   #
#          oximeter, as BIDS ships them), the head's motion from a 4D image series registered      #
#          volume by volume (PLAN 8.12: the same subject under an image and a spectrum).          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import datetime
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from mrsandbox.scan.calibrations import Calibration

#***************#
#   constants   #
#***************#
# the bands the rates are looked for in, in hertz: breathing 6 to 36 a minute, the heart 36 to 150
BANDS = {"respiratory": (0.1, 0.6), "cardiac": (0.6, 2.5)}
# a framewise displacement past which a volume is a motion event (Power et al. 2012's threshold), and the
# head's radius that turns a rotation into millimetres at the cortex
EVENT_MM, HEAD_MM = 0.5, 50.0


def _rate(trace, sampling_hz, band):
    """The dominant frequency of a trace inside `band`, the trace detrended by a cubic."""
    x = np.asarray(trace, dtype=float)
    x = x - np.polyval(np.polyfit(np.arange(x.size), x, 3), np.arange(x.size))
    frequencies = np.fft.rfftfreq(x.size, 1.0 / sampling_hz)
    power = np.abs(np.fft.rfft(x)) ** 2
    inside = (frequencies > band[0]) & (frequencies < band[1])
    if not inside.any():
        raise ValueError(f"the trace is too short for a rate between {band[0]} and {band[1]} Hz")
    return float(frequencies[inside][power[inside].argmax()])


def _conditions(header, dataset, citation, url, what):
    field = header.get("MagneticFieldStrength")
    if field is None and header.get("SpectrometerFrequency"):
        field = round(float(header["SpectrometerFrequency"]) / 42.5774785, 2)
    return dict(source={"dataset": dataset, "citation": citation, "url": url},
                scanner={"vendor": str(header.get("Manufacturer", "")).lower() or None, "model": header.get("ManufacturersModelName"),
                         "software": header.get("SoftwareVersions"), "field_t": None if field is None else float(field)},
                coil={"channels": (), "name": header.get("ReceiveCoilName")},
                sequence={"name": header.get("SequenceName") or what, "family": "", "te_s": header.get("EchoTime"), "tr_s": header.get("RepetitionTime")},
                date=datetime.date.today().strftime("%d/%m/%y"), carried_by="subject", note=f"one {what}; made by scan.run")


#**************************************************************************************************#
#                                       Class PhysioReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The breathing and the heart of a subject off a physiological recording made during a scan.       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class PhysioReading:
    """The breathing and the heart of a subject off a physiological recording made during a
    scan (a respiratory belt, a pulse oximeter): each rate the dominant frequency of its
    trace in its band. Any scanner's, any scan's: the subject breathes the same under an
    image and a spectrum.

    Attributes:
        resp_hz: The breathing rate in hertz (None without a respiratory trace).
        cardiac_hz: The heart rate in hertz (None without a cardiac trace).
        duration_s: How long the recording is.
        header: What the scan's sidecar says of the scanner and the sequence.
    """

    resp_hz: float
    cardiac_hz: float
    duration_s: float
    header: dict

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, traces, sampling_hz, header=None):
        """`traces` {"respiratory": ..., "cardiac": ...} (either may be missing) sampled at `sampling_hz`."""
        rates = {name: _rate(traces[name], sampling_hz, BANDS[name]) for name in BANDS if name in traces}
        length = max(len(trace) for trace in traces.values())
        return cls(resp_hz=rates.get("respiratory"), cardiac_hz=rates.get("cardiac"), duration_s=length / float(sampling_hz), header=dict(header or {}))

    @classmethod
    def from_bids(cls, path, sidecar=None, scan_sidecar=None):
        """A BIDS physio file (*_physio.tsv.gz) with its JSON (`sidecar`, else the file's own or the
        dataset's by inheritance) naming the columns and the sampling frequency; `scan_sidecar` the
        scan's JSON for the scanner."""
        path = pathlib.Path(path)
        if sidecar is None:
            sidecar = _sidecar(path, "_physio.json")
        meta = json.loads(pathlib.Path(sidecar).read_text())
        with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as file:
            data = np.loadtxt(file)
        columns = {name.lower(): data[:, k] for k, name in enumerate(meta["Columns"]) if k < data.shape[1]}
        traces = {name: columns[key] for name in BANDS for key in columns if key.startswith(name[:4])}
        header = json.loads(pathlib.Path(scan_sidecar).read_text()) if scan_sidecar else (
            json.loads(_sidecar(path, "_bold.json").read_text()) if _sidecar(path, "_bold.json") else {})
        return cls.from_arrays(traces, float(meta["SamplingFrequency"]), header)

    #*************#
    #   records   #
    #*************#
    def record_breathing(self, dataset, citation=None, url=None):
        """The Calibration of run.resp_freq_hz: the breathing rate, the prior a quarter either side."""
        value = float(round(self.resp_hz, 4))
        return Calibration(leaf="run.resp_freq_hz", value=value, prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
                           measured={"statistic": "the dominant frequency of the respiratory trace between 0.1 and 0.6 Hz (scan.run.PhysioReading)",
                                     "per_minute": round(60.0 * value, 1), "duration_s": round(self.duration_s, 1)},
                           acquisition={"sampling": "a respiratory belt during the scan"}, voxel={},
                           processing="the trace detrended by a cubic, its power spectrum's peak in the band",
                           **_conditions(self.header, dataset, citation, url, "physiological recording"))

    def record_heart(self, dataset, citation=None, url=None):
        """The Calibration of run.cardiac_hz: the heart rate, the prior a quarter either side."""
        value = float(round(self.cardiac_hz, 4))
        return Calibration(leaf="run.cardiac_hz", value=value, prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
                           measured={"statistic": "the dominant frequency of the cardiac trace between 0.6 and 2.5 Hz (scan.run.PhysioReading)",
                                     "per_minute": round(60.0 * value, 1), "duration_s": round(self.duration_s, 1)},
                           acquisition={"sampling": "a pulse oximeter during the scan"}, voxel={},
                           processing="the trace detrended by a cubic, its power spectrum's peak in the band",
                           **_conditions(self.header, dataset, citation, url, "physiological recording"))


#**************************************************************************************************#
#                                       Class MotionReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The head's motion over a scan off a 4D image series registered volume by volume.                 #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class MotionReading:
    """The head's motion over a scan off a 4D image series registered volume by volume (FSL
    mcflirt): the six rigid parameters per volume become a framewise displacement (Power et
    al. 2012: the translations' steps plus the rotations' steps at the cortex, 50 mm), a
    volume past half a millimetre is a motion event, and the events' sizes are how far the
    head moved.

    Attributes:
        displacement_mm: The framewise displacement per volume step.
        tr_s: The series' repetition time.
        header: What the series' sidecar says of the scanner and the sequence.
    """

    displacement_mm: np.ndarray
    tr_s: float
    header: dict

    #*********************#
    #   from parameters   #
    #*********************#
    @classmethod
    def from_parameters(cls, rotations_rad, translations_mm, tr_s, header=None):
        """The six rigid parameters per volume, (volumes, 3) each, as mcflirt's .par gives them."""
        rotations, translations = np.asarray(rotations_rad, dtype=float), np.asarray(translations_mm, dtype=float)
        displacement = np.abs(np.diff(translations, axis=0)).sum(axis=1) + HEAD_MM * np.abs(np.diff(rotations, axis=0)).sum(axis=1)
        return cls(displacement_mm=displacement, tr_s=float(tr_s), header=dict(header or {}))

    @classmethod
    def from_series(cls, path, tr_s=None, header=None):
        """A 4D NIfTI series registered to its first volume with mcflirt; `tr_s` from the header's
        RepetitionTime (the sidecar beside the file, or `header`) where not given."""
        path = pathlib.Path(path)
        header = dict(header or {})
        if not header and _sidecar(path, "_bold.json"):
            header = json.loads(_sidecar(path, "_bold.json").read_text())
        if tr_s is None:
            tr_s = float(header.get("RepetitionTime") or 0.0) or None
        if tr_s is None:
            import nibabel as nib

            tr_s = float(nib.load(str(path)).header.get_zooms()[3])
        tool = shutil.which("mcflirt") or shutil.which("mcflirt", path=os.path.expandvars("$FSLDIR/bin") + os.pathsep + os.path.expanduser("~/fsl/bin")
                                                       + os.pathsep + os.path.expanduser("~/fsl/share/fsl/bin"))
        if tool is None:
            raise RuntimeError("FSL's mcflirt is needed to register a series volume by volume (fsl.fmrib.ox.ac.uk)")
        with tempfile.TemporaryDirectory() as work:
            environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ", "FSLDIR": os.environ.get("FSLDIR", str(pathlib.Path(tool).parents[1]))}
            done = subprocess.run([tool, "-in", str(path), "-out", f"{work}/mc", "-plots", "-refvol", "0"], capture_output=True, text=True, env=environment)
            if done.returncode != 0:
                raise RuntimeError(f"mcflirt failed: {(done.stderr or done.stdout).strip()[-200:]}")
            parameters = np.loadtxt(f"{work}/mc.par")
        return cls.from_parameters(parameters[:, :3], parameters[:, 3:], tr_s, header)

    #*************#
    #   records   #
    #*************#
    @property
    def events(self):
        """The displacements of the volumes that are motion events."""
        return self.displacement_mm[self.displacement_mm > EVENT_MM]

    @property
    def minutes(self):
        return (self.displacement_mm.size + 1) * self.tr_s / 60.0

    def record_events(self, dataset, citation=None, url=None):
        """The Calibration of run.events_per_min: the motion events a minute, the prior a quarter either side."""
        value = float(round(self.events.size / self.minutes, 4))
        return Calibration(leaf="run.events_per_min", value=value, prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
                           measured={"statistic": "volumes whose framewise displacement passes 0.5 mm, a minute (scan.run.MotionReading)",
                                     "events": int(self.events.size), "volumes": int(self.displacement_mm.size + 1), "minutes": round(self.minutes, 2),
                                     "median_displacement_mm": round(float(np.median(self.displacement_mm)), 4)},
                           acquisition={"registration": "FSL mcflirt to the first volume", "tr_s": self.tr_s}, voxel={},
                           processing="framewise displacement (Power et al. 2012) from the six rigid parameters per volume",
                           **_conditions(self.header, dataset, citation, url, "image series"))

    def record_size(self, dataset, citation=None, url=None):
        """The Calibration of run.motion_mm: how far the head moves in an event, the median over the events
        with the prior their range (zero with none)."""
        events = self.events
        value = float(round(np.median(events), 4)) if events.size else 0.0
        prior = (float(round(events.min(), 4)), float(round(events.max(), 4))) if events.size else (0.0, 0.0)
        return Calibration(leaf="run.motion_mm", value=value, prior=prior,
                           measured={"statistic": "the framewise displacement of the motion events, median and range (scan.run.MotionReading)",
                                     "events": int(events.size), "largest_mm": round(float(self.displacement_mm.max()), 4)},
                           acquisition={"registration": "FSL mcflirt to the first volume", "tr_s": self.tr_s}, voxel={},
                           processing="framewise displacement (Power et al. 2012) from the six rigid parameters per volume",
                           **_conditions(self.header, dataset, citation, url, "image series"))


#**************************************************************************************************#
#                                        Class RunReading                                          #
#**************************************************************************************************#
#                                                                                                  #
# What a run's transients say of the scanner and the subject over it: drift, wander, breathing,     #
# jitter, events.                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class RunReading:
    """What a run's transients, kept apart, say of the scanner and the subject over it. Every
    transient is registered to the mean one on the first 100 ms of the FID (Near et al. 2015's
    spectral registration, here a weighted line fitted to the phase of transient times the
    mean's conjugate): a frequency, a phase and an amplitude per transient. Off those tracks:
    the drift is the frequency's slope in time; the breathing the frequency's amplitude in the
    0.15 to 0.5 Hz band (a sine's amplitude, root two times the band's rms); the wander what
    is left of the frequency after the line and the band; the phase jitter the phase's
    spread about its own trend; the amplitude jitter the amplitude's spread over its mean;
    an event a step of the frequency or the phase past four times its median step.

    Attributes:
        frequency_hz, phase_deg, amplitude: The tracks, one value per transient.
        tr_s: The time between transients.
        header: What the file says of the scanner and the sequence.
    """

    frequency_hz: np.ndarray
    phase_deg: np.ndarray
    amplitude: np.ndarray
    tr_s: float
    header: dict

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, data, dwell_s, tr_s, header=None, window_s=0.1):
        """`data` (npoints, coils, transients) raw FIDs, the coils combined as the readings are."""
        from mrsandbox.scan.calibrate import _combined

        x = _combined(np.asarray(data))                                             # (npoints, transients)
        if x.shape[1] < 8:
            raise ValueError("a run reading needs at least eight transients apart")
        n = min(x.shape[0], int(round(window_s / dwell_s)))
        t = np.arange(n) * dwell_s
        mean = x[:n].mean(axis=1)
        weight = np.abs(mean) ** 2
        frequencies, phases, amplitudes = [], [], []
        for k in range(x.shape[1]):
            cross = x[:n, k] * np.conj(mean)
            # the phase of the cross product is a line 2 pi f t + phi, unwrapped and fitted with the mean's power as weight
            angle = np.unwrap(np.angle(cross))
            slope, intercept = np.polyfit(t, angle, 1, w=np.sqrt(weight))
            frequencies.append(slope / (2.0 * np.pi))
            phases.append(np.degrees(intercept))
            amplitudes.append(float(np.abs(cross).sum() / weight.sum()))
        return cls(frequency_hz=np.asarray(frequencies), phase_deg=np.asarray(phases), amplitude=np.asarray(amplitudes), tr_s=float(tr_s),
                   header=dict(header or {}))

    @classmethod
    def from_nifti(cls, source, tr_s=None):
        """A NIfTI-MRS file with the coils and transients apart; `tr_s` from its header where not given."""
        from mrsandbox.scan.calibrate import _raw

        x, dwell_s, f0_mhz, header = _raw(source)
        return cls.from_arrays(x, dwell_s, tr_s if tr_s is not None else float(header.get("RepetitionTime") or 2.0),
                               {**header, "MagneticFieldStrength": round(f0_mhz / 42.5774785, 2)})

    #**************#
    #   the tracks #
    #**************#
    @property
    def minutes(self):
        return self.frequency_hz.size * self.tr_s / 60.0

    @property
    def times_s(self):
        return np.arange(self.frequency_hz.size) * self.tr_s

    def _line(self, track):
        return np.polyfit(self.times_s, track, 1)

    def _steady(self, track):
        """The track with the events' steps taken out (each step subtracted from what follows it), so the
        drift and the wander are the scanner's own and not the head's."""
        steady = np.array(track, dtype=float)
        for event in self.events:
            steady[event:] -= steady[event] - steady[event - 1]
        return steady

    @property
    def drift_hz_per_min(self):
        return float(60.0 * self._line(self._steady(self.frequency_hz))[0])

    def _band(self, low=0.15, high=0.5):
        """The frequency track's part in the breathing band, and the rest without the line and the events."""
        steady = self._steady(self.frequency_hz)
        residual = steady - np.polyval(self._line(steady), self.times_s)
        spectrum = np.fft.rfft(residual)
        f = np.fft.rfftfreq(residual.size, self.tr_s)
        inside = (f >= low) & (f <= high)
        breathing = np.fft.irfft(np.where(inside, spectrum, 0.0), residual.size)
        return breathing, residual - breathing

    @property
    def resp_amp_hz(self):
        breathing, _ = self._band()
        return float(np.sqrt(2.0) * breathing.std())

    @property
    def ar_sigma_hz(self):
        _, rest = self._band()
        return float(rest.std())

    @property
    def phase_jitter_deg(self):
        steady = self._steady(self.phase_deg)
        return float((steady - np.polyval(self._line(steady), self.times_s)).std())

    @property
    def amp_jitter_frac(self):
        return float(self.amplitude.std() / self.amplitude.mean())

    @property
    def events(self):
        """The transients where the frequency or the phase steps past four times its median step."""
        steps_f, steps_p = np.abs(np.diff(self.frequency_hz)), np.abs(np.diff(self.phase_deg))
        return np.flatnonzero((steps_f > 4.0 * np.median(steps_f) + 1e-9) | (steps_p > 4.0 * np.median(steps_p) + 1e-9)) + 1

    #*************#
    #   records   #
    #*************#
    def records(self, dataset, citation=None, url=None):
        """The Calibrations of the run's leaves this reading makes: drift, wander, breathing's amplitude, the phase
        and amplitude jitter, the events a minute; each the reading's value with a quarter either side."""
        values = {"run.drift_hz_per_min": (self.drift_hz_per_min, "the frequency track's slope, hertz a minute"),
                  "run.ar_sigma_hz": (self.ar_sigma_hz, "the frequency track's spread without its line and its breathing band"),
                  "run.resp_amp_hz": (self.resp_amp_hz, "the frequency track's amplitude in the 0.15 to 0.5 Hz band"),
                  "run.phase_jitter_deg": (self.phase_jitter_deg, "the phase track's spread about its line"),
                  "run.amp_jitter_frac": (self.amp_jitter_frac, "the amplitude track's spread over its mean"),
                  "run.events_per_min": (self.events.size / self.minutes, "steps of the frequency or the phase past four times the median step, a minute")}
        # the rule: a registration whose wander passes 10 Hz or whose phase jitter passes 60 degrees did not register the
        # transients (the phase wraps within the window, or the stored transients carry a receiver cycle): every row fails
        failed = ("" if self.ar_sigma_hz <= 10.0 and self.phase_jitter_deg <= 60.0 else
                  f"the transients did not register: wander {self.ar_sigma_hz:.1f} Hz, phase jitter {self.phase_jitter_deg:.0f} degrees")
        made = []
        for leaf, (value, statistic) in values.items():
            value = float(round(value, 6))
            made.append(Calibration(leaf=leaf, value=value, prior=tuple(sorted((round(0.75 * value, 6), round(1.25 * value, 6)))), failed=failed,
                                    measured={"statistic": f"{statistic} (scan.run.RunReading)", "transients": int(self.frequency_hz.size), "minutes": round(self.minutes, 2)},
                                    acquisition={"tr_s": self.tr_s, "registration": "each transient to the mean on the first 100 ms"}, voxel={},
                                    processing="the coils combined as the readings are; the phase of transient times the mean's conjugate fitted by a weighted line",
                                    **{**_conditions(self.header, dataset, citation, url, "run"),
                                       "carried_by": "scanner" if leaf in ("run.drift_hz_per_min", "run.ar_sigma_hz", "run.phase_jitter_deg", "run.amp_jitter_frac") else "subject"}))
        return tuple(made)


def _sidecar(path, ending):
    """The JSON beside a BIDS file, or the dataset's by inheritance (the same name without the subject's part,
    up the folders), or None."""
    name = path.name
    for suffix in (".tsv.gz", ".tsv", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = name.rsplit("_", 1)[0]
    candidates = [path.parent / f"{name}{ending}"]
    stripped = "_".join(part for part in name.split("_") if not part.startswith(("sub-", "ses-")))
    for parent in (path.parent, *path.parents[1:4]):
        candidates.append(parent / f"{stripped}{ending}")
    return next((c for c in candidates if c.exists()), None)
