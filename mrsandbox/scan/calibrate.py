####################################################################################################
#                                           calibrate.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: A scanner's own data in, a calibration record out: the single-transient NAA signal to   #
#          noise ratio off raw data and the noise level of the virtual scan that reproduces it,    #
#          and the noise covariance of a receive array's channels (scan.calibrations).             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import datetime

import numpy as np

from mrsandbox.scan.calibrations import APODIZED, Calibration
from mrsandbox.scan.voxel import noise_scale

# the apodization of the record's reading (scan.calibrations, the measure) and the NAA singlet
APODIZATION_HZ, NAA_PPM, WATER_PPM = 10.0, 2.008, 4.65


#***********#
#   read    #
#***********#
def _snr(fids, dwell_s, f0_mhz):
    """The consensus SNR of (npoints, transients) FIDs, plain and apodized, and the ppm
    the NAA singlet was found at: the real peak height over the standard deviation of a
    signal-free stretch after a linear detrend (Near et al. 2021), phased on the
    transients' mean, the median over transients. The floor is measured once, plain,
    where it is flat; under the apodization it is that floor times the root of the
    window's mean square, as for white noise and as the simulation has it: measured
    there, the broadened lines' feet and the edge of a cropped band would count as
    noise (up to a factor of two on oversampled data cropped to the band)."""
    n = fids.shape[0]
    window = np.exp(-np.pi * APODIZATION_HZ * np.arange(n) * dwell_s)
    hz = np.fft.fftshift(np.fft.fftfreq(n, dwell_s))
    mean = np.fft.fftshift(np.fft.fft(fids.mean(axis=1)))
    # the sign of the frequency axis differs between conventions: the one with NAA wins
    ppm = max((WATER_PPM + sign * hz / f0_mhz for sign in (1.0, -1.0)),
              key=lambda axis: np.abs(mean[np.abs(axis - NAA_PPM) < 0.15]).max())
    peak = np.argmax(np.abs(mean) * (np.abs(ppm - NAA_PPM) < 0.15))
    phase = np.exp(-1j * np.angle(mean[peak]))
    quiet, near = (ppm > -4.0) & (ppm < -1.0), np.abs(ppm - ppm[peak]) < 0.05
    plain = (np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0) * phase).real
    broad = (np.fft.fftshift(np.fft.fft(fids * window[:, None], axis=0), axes=0) * phase).real
    floors = np.array([(real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])).std()
                       for real in plain.T])
    return (float(np.median(plain[near].max(axis=0) / floors)),
            float(np.median(broad[near].max(axis=0) / (floors * np.sqrt((window ** 2).mean())))),
            float(ppm[peak]))


#**************************************************************************************************#
#                                        Class NoiseReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The single-transient NAA signal to noise ratio of one raw acquisition, and what it was read off. #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class NoiseReading:
    """The single-transient NAA signal to noise ratio of one raw acquisition, and what it
    was read off.

    The data are NIfTI-MRS with the coils and the transients apart (spec2nii writes it
    from TWIX, P-files, and data/list alike: `spec2nii twix -e image`, `spec2nii ge`,
    `spec2nii philips_dl`). The channels are whitened by the noise of the FIDs' last
    eighth and combined with the first singular vector of the mean transient's start;
    nothing else is done to the data, no fit enters (scan.calibrations, the measure).
    Averaged or combined data do not serve: a stored sum of excitations is not their
    ideal sum (Big GABA's GE sites read a quarter low that way), so `excitations` is
    for a sequence whose unit is more than one shot (SPECIAL's pair) and for nothing
    else.

    Args:
        snr, snr_apodized: The plain reading and the one under the 10 Hz apodization.
        naa_ppm: Where the singlet was found, against water at 4.65 ppm on resonance.
        transients, coils, npoints, bandwidth_hz, f0_mhz: What the data held.
        excitations: Excitations behind one stored transient.
        header: What the file says of the scanner, coil, and sequence.
    """

    snr: float
    snr_apodized: float
    naa_ppm: float
    transients: int
    coils: int
    npoints: int
    bandwidth_hz: float
    f0_mhz: float
    excitations: int
    header: dict

    HEADER = ("Manufacturer", "ManufacturersModelName", "SoftwareVersions", "ReceiveCoilName",
              "SequenceName", "ProtocolName", "EchoTime", "RepetitionTime")

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, data, dwell_s, f0_mhz, header=None, excitations=1):
        """Read raw FIDs, (npoints, coils, transients) complex, as they were stored (an
        oversampled receiver band is left as it is)."""
        x = np.asarray(data)
        n, coils, transients = x.shape
        covariance = np.atleast_2d(np.cov(x[-n // 8:].reshape(-1, coils).T))
        white = np.einsum("ck,nks->ncs", np.linalg.inv(np.linalg.cholesky(covariance)), x)
        weights = np.conj(np.linalg.svd(white.mean(axis=2)[:64], full_matrices=False)[2][0])
        plain, apodized, at = _snr(np.einsum("c,ncs->ns", weights, white), dwell_s, f0_mhz)
        return cls(snr=plain, snr_apodized=apodized, naa_ppm=at, transients=transients, coils=coils,
                   npoints=n, bandwidth_hz=1.0 / dwell_s, f0_mhz=float(f0_mhz),
                   excitations=int(excitations), header=dict(header or {}))

    #****************#
    #   from nifti   #
    #****************#
    @classmethod
    def from_nifti(cls, source, excitations=1):
        """Read a NIfTI-MRS file (a path, or a loaded image) with DIM_COIL and DIM_DYN."""
        import json

        import nibabel as nib

        image = nib.load(str(source)) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
        extension = json.loads(image.header.extensions[0].get_content())
        tags = [extension.get(f"dim_{k}") for k in (5, 6, 7)]
        data = np.asarray(image.dataobj)
        if data.shape[:3] != (1, 1, 1):
            raise ValueError("a single voxel is needed, not a lattice")
        data = data[0, 0, 0].reshape(data.shape[3:] + (1,) * (4 - (data.ndim - 3)))
        if "DIM_DYN" not in tags:
            raise ValueError("the transients are needed apart (DIM_DYN); averaged data do not serve")
        order = [0] + [1 + tags.index(tag) if tag in tags else None for tag in ("DIM_COIL", "DIM_DYN")]
        if order[1] is None:
            data, order[1] = data[..., None], data.ndim
        rest = [axis for axis in range(1, data.ndim) if axis not in order[1:]]
        if any(data.shape[axis] > 1 for axis in rest):
            raise ValueError("only coils and transients are read; select the other dimension first")
        x = np.transpose(data, order + rest).reshape(data.shape[0], data.shape[order[1]], data.shape[order[2]])
        header = {key: extension[key] for key in cls.HEADER if extension.get(key) not in (None, "")}
        return cls.from_arrays(x, float(image.header["pixdim"][4]), float(extension["SpectrometerFrequency"][0]),
                               header=header, excitations=excitations)

    #***********#
    #   sigma   #
    #***********#
    def sigma(self, voxel, seeds=12):
        """The reference noise level (the noise.sigma leaf) at which `voxel`, a Voxel of a
        scan set up as the acquisition was (field, sequence, timings, place, size),
        reproduces the reading: (plain, apodized), each the median over `seeds` drawn
        subjects. Nothing is drawn for the noise: white noise of sigma per real FID point
        leaves sigma times the root of the summed squared window in the real spectrum."""
        dwell_s = 1.0 / self.bandwidth_hz
        window = np.exp(-np.pi * APODIZATION_HZ * np.arange(self.npoints) * dwell_s)
        ppm = np.fft.fftshift(np.fft.fftfreq(self.npoints, dwell_s)) / voxel._f0_mhz()
        scale = voxel.noise_place() * noise_scale(voxel.volume_ml, voxel.scan.engine.b0_t, dwell_s,
                                                  self.excitations)
        found = []
        for seed in range(seeds):
            _, fid = voxel._clean_fid(voxel.sample(seed), self.npoints, dwell_s)
            row = []
            near = np.abs(ppm - NAA_PPM) < 0.05
            plain = np.fft.fftshift(np.fft.fft(np.asarray(fid)))
            phase = np.exp(-1j * np.angle(plain[near][np.argmax(np.abs(plain[near]))]))
            for apodization, measured in ((np.ones(self.npoints), self.snr), (window, self.snr_apodized)):
                height = (np.fft.fftshift(np.fft.fft(np.asarray(fid) * apodization)) * phase).real[near].max()
                row.append(height / (measured * np.sqrt((apodization ** 2).sum()) * scale))
            found.append(row)
        plain, apodized = np.median(np.array(found), axis=0)
        return float(plain), float(apodized)

    #************#
    #   record   #
    #************#
    def record(self, voxel, dataset, presets=(), citation=None, url=None, seeds=12):
        """The Calibration of noise.sigma this reading makes with `voxel` as its virtual
        twin: the value is the apodized reading's, the prior a quarter either side of it
        (one acquisition has no spread of its own; stated), the conditions the file's and
        the twin's."""
        plain, apodized = self.sigma(voxel, seeds)
        sequence = voxel.scan.sequence
        return Calibration(
            leaf="noise.sigma", value=round(apodized, 2),
            prior=(round(0.75 * apodized, 2), round(1.25 * apodized, 2)),
            measured={"statistic": "single-transient NAA SNR (scan.calibrate.NoiseReading)",
                      "median": round(self.snr, 1), "implied_value": round(plain, 2),
                      "naa_found_at_ppm": round(self.naa_ppm, 3),
                      "apodized": {"statistic": APODIZED, "median": round(self.snr_apodized, 1),
                                   "implied_value": round(apodized, 2)}},
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                     "model": self.header.get("ManufacturersModelName"),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or type(sequence).__name__,
                      "family": type(sequence).__name__, "te_s": self.header.get("EchoTime"),
                      "tr_s": self.header.get("RepetitionTime"),
                      "simulated_as": type(sequence).__name__},
            acquisition={"npoints": self.npoints, "bandwidth_hz": self.bandwidth_hz,
                         "transients": self.transients, "measured_on": self.excitations},
            voxel={"size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, SNR per transient, median",
            date=datetime.date.today().strftime("%d/%m/%y"), presets=tuple(presets),
            note="one acquisition; made by scan.calibrate",
        )


#**************************************************************************************************#
#                                     Class CovarianceReading                                      #
#**************************************************************************************************#
#                                                                                                  #
# The noise covariance of a receive array's channels, read off samples that hold no signal.        #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class CovarianceReading:
    """The noise covariance of a receive array's channels, read off samples that hold no
    signal: a noise-only scan, the noise adjustment a raw file carries, or the tail of
    long FIDs.

    What it says depends on the load: a head shares more noise between channels than a
    small phantom (scan.calibrations, the array noise correlation records), so a reading
    for in vivo use wants a head in the coil.

    Args:
        covariance: (coils, coils) complex, Hermitian.
        samples: Complex samples per channel behind it.
        header: What the file says of the scanner and coil.
    """

    covariance: np.ndarray
    samples: int
    header: dict

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, noise, header=None):
        """Read noise samples, (..., coils) complex: everything before the last axis is
        pooled."""
        x = np.asarray(noise).reshape(-1, np.shape(noise)[-1])
        x = x - x.mean(axis=0)
        return cls(covariance=(x.T @ x.conj()) / (x.shape[0] - 1), samples=x.shape[0],
                   header=dict(header or {}))

    #****************#
    #   from nifti   #
    #****************#
    @classmethod
    def from_nifti(cls, source, tail=0.125):
        """Read the last `tail` of the FIDs of a NIfTI-MRS file with a DIM_COIL (long
        acquisitions have decayed there; a residual signal would show as one strongly
        correlated mode)."""
        import json

        import nibabel as nib

        image = nib.load(str(source)) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
        extension = json.loads(image.header.extensions[0].get_content())
        tags = [extension.get(f"dim_{k}") for k in (5, 6, 7)]
        if "DIM_COIL" not in tags:
            raise ValueError("the channels are needed apart (DIM_COIL)")
        data = np.moveaxis(np.asarray(image.dataobj)[0, 0, 0], 1 + tags.index("DIM_COIL"), -1)
        start = int(round(data.shape[0] * (1.0 - tail)))
        header = {key: extension[key] for key in NoiseReading.HEADER if extension.get(key) not in (None, "")}
        return cls.from_arrays(data[start:], header=header)

    #******************#
    #   from ismrmrd   #
    #******************#
    @classmethod
    def from_ismrmrd(cls, path, lines=None):
        """Read an ISMRMRD (.mrd, .h5) file whose acquisitions hold noise only: all of
        them, or the acquisitions `lines` indexes. Needs h5py."""
        import re

        import h5py

        with h5py.File(str(path), "r") as file:
            table = file["dataset/data"]
            chosen = range(len(table)) if lines is None else lines
            total, count, coils = None, 0, None
            for index in chosen:
                row = table[index]
                coils = int(row["head"]["active_channels"])
                raw = row["data"].view(np.float32)
                x = (raw[0::2] + 1j * raw[1::2]).reshape(coils, -1)
                total = (x @ x.conj().T) if total is None else total + x @ x.conj().T
                count += x.shape[1]
            xml = file["dataset/xml"][0]
        xml = xml.decode() if isinstance(xml, bytes) else xml
        found = {tag: re.search(rf"<{tag}>(.*?)</{tag}>", xml) for tag in ("systemVendor", "systemModel", "systemFieldStrength_T")}
        header = {tag: match.group(1) for tag, match in found.items() if match}
        return cls(covariance=total.conj() / count, samples=count, header=header)

    #*****************#
    #   correlation   #
    #*****************#
    @property
    def coils(self):
        return self.covariance.shape[0]

    @property
    def correlation(self):
        """The magnitude of the channels' noise correlation, zero on the diagonal."""
        level = np.sqrt(np.real(np.diag(self.covariance)))
        return np.abs(self.covariance / np.outer(level, level)) - np.eye(self.coils)

    def statistics(self):
        """What the array noise records hold: the mean over channels of the strongest
        partner, the strongest pair, the median pair, and the spread of the channels'
        noise levels (largest over smallest standard deviation)."""
        c, level = self.correlation, np.sqrt(np.real(np.diag(self.covariance)))
        return {"strongest_partner_mean": float(c.max(axis=1).mean()), "strongest_pair": float(c.max()),
                "median_pair": float(np.median(c[np.triu_indices(self.coils, 1)])),
                "level_spread": float(level.max() / level.min())}

    #****************#
    #   resistance   #
    #****************#
    def resistance(self):
        """The covariance scaled to a mean diagonal of one: the `noise_resistance` of a
        hardware.CoilArray with as many channels, in the file's channel order (which a
        geometric array does not share; a mapped array of the same coil does)."""
        return self.covariance / np.real(np.diag(self.covariance)).mean()

    #***************#
    #   own noise   #
    #***************#
    def own_noise(self, array):
        """The share of a channel's noise that is its own, at which the geometric `array`
        (a hardware.CoilArray of loops) has this reading's mean strongest partner: the
        sample's share of the resistance comes from the geometry (`CoilArray.loaded`),
        the rest is the identity."""
        sample = array.loaded(own_noise=0.0).noise_resistance
        target = self.statistics()["strongest_partner_mean"]

        def strongest(share):
            r = (1.0 - share) * sample + share * np.eye(len(array))
            level = np.sqrt(np.diag(r))
            return float((np.abs(r / np.outer(level, level)) - np.eye(len(array))).max(axis=1).mean())

        low, high = 0.0, 0.999
        if strongest(low) <= target:
            return 0.0
        for _ in range(40):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if strongest(middle) > target else (low, middle)
        return float(0.5 * (low + high))

    #************#
    #   record   #
    #************#
    def record(self, array, dataset, load, citation=None, url=None):
        """The Calibration of coils.loaded.own_noise this reading makes with `array`;
        `load` says what was in the coil."""
        share = round(self.own_noise(array), 2)
        return Calibration(
            leaf="coils.loaded.own_noise", value=share, prior=(share, share),
            measured={"statistic": "magnitude of the noise correlation between channels "
                                   "(scan.calibrate.CovarianceReading.statistics)",
                      **{key: round(value, 3) for key, value in self.statistics().items()},
                      "samples": self.samples},
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", self.header.get("systemVendor", ""))).lower() or None,
                     "model": self.header.get("ManufacturersModelName", self.header.get("systemModel")),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": self.header.get("systemFieldStrength_T")},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": "noise", "family": ""}, acquisition={"channels": self.coils, "rf": "no signal"},
            voxel={"load": load}, processing="sample covariance of the channels over all samples",
            date=datetime.date.today().strftime("%d/%m/%y"),
            note="the modelled array stands in for the coil's unpublished geometry; made by scan.calibrate",
        )
