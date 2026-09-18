####################################################################################################
#                                           calibrate.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: A scanner's own data in, a calibration record out: the reading of the single-transient #
#          NAA signal to noise ratio off raw NIfTI-MRS (coils and transients apart), and the      #
#          noise level of the virtual scan that reproduces it (scan.calibrations).                 #
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
