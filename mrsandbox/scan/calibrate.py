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

from mrsandbox.phantom.solution import Phantom
from mrsandbox.scan.calibrations import Calibration
from mrsandbox.scan.voxel import noise_scale

# what a NIfTI-MRS header says of the scanner, coil, and sequence
HEADER = ("Manufacturer", "ManufacturersModelName", "SoftwareVersions", "ReceiveCoilName",
          "SequenceName", "ProtocolName", "EchoTime", "RepetitionTime")

# the apodization of the record's reading (scan.calibrations, the measure) and the NAA singlet
APODIZATION_HZ, NAA_PPM, WATER_PPM = 10.0, 2.008, 4.65
# the lines a noise reading is taken on: where the line lies with the carrier on water, and how far
# from there it is looked for (a phantom's water moves with its temperature)
LINES = {"NAA": (NAA_PPM, 0.05), "water": (WATER_PPM, 0.4)}
# the second reading of the noise: the plain single-transient NAA SNR follows the in vivo line, the apodized
# one the broad signal under the singlet; they err in opposite directions and each gives a sigma
APODIZED = ("the same single-transient NAA SNR after a 10 Hz exponential apodization of data and "
            "simulation alike")


#**************#
#   combine    #
#**************#
def _combined(data, noise=None, like=None):
    """(npoints, coils, transients) raw FIDs to (npoints, transients): the channels
    whitened by the noise of the FIDs' last eighth, or of `noise`, (samples, coils)
    measured with the RF off, where the FIDs still ring at their end (a phantom's
    unsuppressed water), and combined with the first singular vector of the mean
    transient's start. The combined noise has unit variance per complex point. A receiver
    phase cycle left in the stored transients (GE P-files alternate by a half turn) is
    undone first, in quarter turns against the first transient, or the mean would cancel;
    phase jitter stays as it is. `like`, another file of the same voxel and coils, lends
    its whitening and weights: a water reference of a few transients cannot carry its own."""
    x = np.asarray(data)
    n, coils, _ = x.shape
    source = x if like is None else np.asarray(like)
    quiet = source[-n // 8:].reshape(-1, coils) if noise is None else np.asarray(noise)
    covariance = np.atleast_2d(np.cov(quiet.T))
    whitening = np.linalg.inv(np.linalg.cholesky(covariance))

    def whitened(raw):
        w = np.einsum("ck,nks->ncs", whitening, raw)
        turns = np.round(np.angle(np.einsum("nc,ncs->s", np.conj(w[:64, :, 0]), w[:64])) / (np.pi / 2))
        return w * np.exp(-0.5j * np.pi * turns)

    white = whitened(x)
    lender = white if like is None else whitened(source)
    weights = np.conj(np.linalg.svd(lender.mean(axis=2)[:64], full_matrices=False)[2][0])
    return np.einsum("c,ncs->ns", weights, white)


def _axis(fid, dwell_s, f0_mhz, carrier_ppm):
    """(sign, spectrum, hertz) of a FID whose carrier sits at `carrier_ppm`: the spectrum
    sixteenfold zero filled, and the sign +1 where it holds NAA the engine's way round
    (ppm = carrier + hertz / f0), -1 where the axis runs the other way (a file stored as
    the conjugate of the engine's convention)."""
    n = fid.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(fid), 16 * n))
    hz = np.fft.fftshift(np.fft.fftfreq(16 * n, dwell_s))
    sign = max((1.0, -1.0), key=lambda s: np.abs(spectrum[np.abs(carrier_ppm + s * hz / f0_mhz - NAA_PPM) < 0.15]).max())
    return sign, spectrum, hz


def _lines(fid, dwell_s, f0_mhz, carrier_ppm):
    """The complex values of the NAA, creatine and water peaks of a FID whose carrier sits
    at `carrier_ppm`, each the mean over the bins within 1.5 Hz of its maximum on a
    sixteenfold zero-filled spectrum (one bin's value swings by degrees with where the
    bins fall on the line, which is a hundred microseconds of delay), and their
    places in hertz the ppm way round: {name: (value, hertz)}. The axis is the one of
    the two that holds NAA (`_axis`); a FID whose axis runs the other way has its values
    conjugated, so every phase reads in the engine's convention, a twin's included."""
    sign, spectrum, hz = _axis(fid, dwell_s, f0_mhz, carrier_ppm)
    ppm = carrier_ppm + sign * hz / f0_mhz
    found = {}
    for name, at in (("NAA", NAA_PPM), ("Cr", 3.03), ("water", WATER_PPM)):
        inside = np.flatnonzero(np.abs(ppm - at) < 0.08)
        k = inside[np.abs(spectrum[inside]).argmax()]
        near = np.abs(ppm - ppm[k]) * f0_mhz < 1.5
        value = complex(spectrum[near].mean())
        found[name] = (value if sign > 0 else np.conj(value), float(ppm[k] * f0_mhz))
    return found


def _exponentials(fid, dwell_s, order=20):
    """The complex exponentials in a FID's start, the state space way of HLSVD (Kung's
    Hankel SVD): (frequencies in Hz, dampings in 1/s, complex amplitudes at t = 0) of
    `order` components fitted to the first half of `fid` in a Hankel matrix; what the
    residual water records were read with (info/HANDOFF.md item 217), in the package."""
    x = np.asarray(fid, dtype=np.complex128)
    n = x.size
    m = n // 2
    hankel = np.lib.stride_tricks.sliding_window_view(x, n - m + 1)[:m]
    u = np.linalg.svd(hankel, full_matrices=False)[0][:, :order]
    poles = np.linalg.eigvals(np.linalg.pinv(u[:-1]) @ u[1:])
    poles = poles[np.abs(poles) > 0.0]
    basis = np.exp(np.outer(np.arange(n), np.log(poles)))
    amplitudes = np.linalg.lstsq(basis, x, rcond=None)[0]
    return (np.angle(poles) / (2.0 * np.pi * dwell_s), np.log(np.abs(poles)) / dwell_s, amplitudes)


def _eddy_transient(fid, dwell_s, f0_mhz, late_s=(0.06, 0.25), at_s=0.005):
    """The eddy phase transient at the FID's start, in radians `at_s` after the first
    sample: the water's components (within 0.4 ppm of the carrier, T2* above 5 ms) fitted
    on the late FID, where eddy currents have died and the water is still far above the
    metabolites, carried back to the start; what the early phase does against that model
    (info/HANDOFF.md item 236). Exact for transients dead by the window's start, a lower
    bound for slower ones (the model's frequencies take a slow phase into a ramp)."""
    x = np.asarray(fid, dtype=np.complex128)
    t = np.arange(x.size) * dwell_s
    a, b = int(late_s[0] / dwell_s), int(late_s[1] / dwell_s)
    freqs, damping, amplitudes = _exponentials(x[a:b], dwell_s, order=12)
    keep = (np.abs(freqs) < 0.4 * f0_mhz) & (damping > -200.0)
    rate = 2j * np.pi * freqs[keep] + damping[keep]
    back = amplitudes[keep] * np.exp(-rate * t[a])
    model = (back[:, None] * np.exp(rate[:, None] * t[None, :])).sum(axis=0)
    return float(np.angle(x * np.conj(model))[int(round(at_s / dwell_s))])


def _raw(source, water=False):
    """A NIfTI-MRS file (a path or a loaded image) as ((npoints, coils, transients),
    dwell_s, f0_mhz, header); the transients are needed apart. A run that interleaves
    unsuppressed water transients with its own (ds005917 stores one every 16th, some 300
    times the others at their start, where a suppressed run's transients stay within
    three times their median) gives its suppressed transients, or with `water` the
    interleaved water, the run's own reference (none: no transient)."""
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
    start = np.abs(x[:8]).mean(axis=(0, 1))
    loud = start > 20.0 * np.median(start)
    x = x[:, :, loud] if water else x[:, :, ~loud]
    header = {key: extension[key] for key in HEADER if extension.get(key) not in (None, "")}
    return x, float(image.header["pixdim"][4]), float(extension["SpectrometerFrequency"][0]), header


#***********#
#   read    #
#***********#
def _spectrum(fid, dwell_s, ppm_of, fill=4, apodization_hz=2.0):
    """The real spectrum of a FID phased on the NAA singlet and referenced to it, as
    (ppm ascending, real): zero filled `fill` times, lightly apodized; `ppm_of` maps
    hertz to ppm and the sign of the frequency axis is the one that holds NAA."""
    n = fid.shape[0]
    padded = np.concatenate([np.asarray(fid) * np.exp(-np.pi * apodization_hz * np.arange(n) * dwell_s),
                             np.zeros((fill - 1) * n, dtype=np.complex128)])
    spectrum = np.fft.fftshift(np.fft.fft(padded))
    hz = np.fft.fftshift(np.fft.fftfreq(fill * n, dwell_s))
    ppm = max((ppm_of(sign * hz) for sign in (1.0, -1.0)),
              key=lambda axis: np.abs(spectrum[np.abs(axis - NAA_PPM) < 0.15]).max())
    peak = np.argmax(np.abs(spectrum) * (np.abs(ppm - NAA_PPM) < 0.15))
    real = (spectrum * np.exp(-1j * np.angle(spectrum[peak]))).real
    order = np.argsort(ppm - (ppm[peak] - NAA_PPM))
    return (ppm - (ppm[peak] - NAA_PPM))[order], real[order]


def _shape(ppm, real, f0_mhz):
    """What a spectrum says without a fit: the NAA singlet's full width at half maximum
    over the level at 1.6 ppm (hertz), and the height of the 3.03 ppm creatine singlet
    over NAA's."""
    near = np.abs(ppm - NAA_PPM) < 0.12
    x, y = ppm[near], real[near]
    top = int(np.argmax(y))
    half = np.median(real[np.abs(ppm - 1.6) < 0.1]) + 0.5 * (y[top] - np.median(real[np.abs(ppm - 1.6) < 0.1]))
    left = top - int(np.argmax(y[top::-1] < half))
    right = top + int(np.argmax(y[top:] < half))
    low = np.interp(half, [y[left], y[left + 1]], [x[left], x[left + 1]])
    high = np.interp(half, [y[right], y[right - 1]], [x[right], x[right - 1]])
    creatine = real[np.abs(ppm - 3.03) < 0.04].max() / real[np.abs(ppm - NAA_PPM) < 0.04].max()
    return float((high - low) * f0_mhz), float(creatine)


#**************************************************************************************************#
#                                        Class NoiseReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The single-transient NAA signal to noise ratio of one raw acquisition, and what it was read off. #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class NoiseReading:
    """The single-transient signal to noise ratio of one raw acquisition, on the NAA
    singlet in vivo or on the water line of a phantom, and what it was read off.

    The data are NIfTI-MRS with the coils and the transients apart (spec2nii writes it
    from TWIX, P-files, and data/list alike: `spec2nii twix -e image`, `spec2nii ge`,
    `spec2nii philips_dl`). The channels are whitened by the noise of the FIDs' last
    eighth and combined with the first singular vector of the mean transient's start;
    nothing else is done to the data, no fit enters (scan.calibrations, the measure).
    Averaged or combined data do not serve: a stored sum of excitations is not their
    ideal sum (Big GABA's GE sites read a quarter low that way), so `excitations` is
    for a sequence whose unit is more than one shot (SPECIAL's pair) and for nothing
    else.

    A phantom has no NAA to speak of in one transient but all the water one could want:
    `line="water"` reads the unsuppressed water line, against a twin that is a
    phantom.Phantom with phantom.Unsuppressed water. Its FIDs ring to their end, so
    the noise comes from `noise`, the scanner's own noise samples (the NOI vectors of a
    data/list export, the noise label of lab/raw: io.philips), taken with the same
    receiver settings: whitened by them the floor is known and not measured. The
    record names the load; it is the receive chain's number with a phantom in the coil,
    a lower bound for a head (a phantom loads the coil less).

    Args:
        snr, snr_apodized: The plain reading and the one under the 10 Hz apodization.
        naa_ppm: Where the line was found, against water at 4.65 ppm on resonance.
        transients, coils, npoints, bandwidth_hz, f0_mhz: What the data held.
        excitations: Excitations behind one stored transient.
        header: What the file says of the scanner, coil, and sequence.
        line: The line the reading was taken on, "NAA" or "water".
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
    line: str = "NAA"

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, data, dwell_s, f0_mhz, header=None, excitations=1, line="NAA", noise=None):
        """Read raw FIDs, (npoints, coils, transients) complex, as they were stored (an
        oversampled receiver band is left as it is); `noise` is (samples, coils) of the
        receiver with the RF off, sampled and scaled as the FIDs are."""
        x = np.asarray(data)
        n, coils, transients = x.shape
        # the consensus SNR: the real peak height over the standard deviation of a signal-free
        # stretch after a linear detrend (Near et al. 2021), phased on the transients' mean, the
        # median over transients. The floor is measured once, plain, where it is flat; under
        # the apodization it is that floor times the root of the window's mean square, as for
        # white noise and as the simulation has it: measured there, the broadened lines' feet
        # and the edge of a cropped band would count as noise (up to a factor of two on
        # oversampled data cropped to the band).
        fids = _combined(x, noise)
        center = LINES[line][0]
        window = np.exp(-np.pi * APODIZATION_HZ * np.arange(n) * dwell_s)
        hz = np.fft.fftshift(np.fft.fftfreq(n, dwell_s))
        # a phantom's water line is as narrow as the grid and has no sure phase at its top: it is
        # found and phased under the apodization, here and on the twin
        mean = np.fft.fftshift(np.fft.fft(fids.mean(axis=1) * (window if line == "water" else 1.0)))
        # the sign of the frequency axis differs between conventions: the one with the line wins,
        # a line being what stands above its surroundings (a water residual's foot is tall and flat)
        ppm = max((WATER_PPM + sign * hz / f0_mhz for sign in (1.0, -1.0)),
                  key=lambda axis: (np.abs(mean[np.abs(axis - center) < 0.15]).max()
                                    - np.median(np.abs(mean[np.abs(axis - center) < 0.5]))))
        peak = np.argmax(np.abs(mean) * (np.abs(ppm - center) < 0.15))
        phase = np.exp(-1j * np.angle(mean[peak]))
        quiet, near = (ppm > -4.0) & (ppm < -1.0), np.abs(ppm - ppm[peak]) < 0.05
        spectra = (np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0) * phase).real
        broad = (np.fft.fftshift(np.fft.fft(fids * window[:, None], axis=0), axes=0) * phase).real
        # whitened by the scanner's own noise the floor is known: unit variance per complex FID
        # point, half of it in the real part
        floors = np.full(transients, np.sqrt(n / 2.0)) if noise is not None else np.array(
            [(real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])).std() for real in spectra.T])
        plain = float(np.median(spectra[near].max(axis=0) / floors))
        apodized = float(np.median(broad[near].max(axis=0) / (floors * np.sqrt((window ** 2).mean()))))
        return cls(snr=plain, snr_apodized=apodized, naa_ppm=float(ppm[peak]), transients=transients, coils=coils,
                   npoints=n, bandwidth_hz=1.0 / dwell_s, f0_mhz=float(f0_mhz),
                   excitations=int(excitations), header=dict(header or {}), line=line)

    #****************#
    #   from nifti   #
    #****************#
    @classmethod
    def from_nifti(cls, source, excitations=1):
        """Read a NIfTI-MRS file (a path, or a loaded image) with DIM_COIL and DIM_DYN."""
        x, dwell_s, f0_mhz, header = _raw(source)
        return cls.from_arrays(x, dwell_s, f0_mhz, header=header, excitations=excitations)

    #***********#
    #   sigma   #
    #***********#
    def sigma(self, voxel, seeds=12, gauss_hz=None):
        """The reference noise level (the noise.sigma leaf) at which `voxel`, a Voxel of a
        scan set up as the acquisition was (field, sequence, timings, place, size),
        reproduces the reading: (plain, apodized), each the median over `seeds` drawn
        subjects. Nothing is drawn for the noise: white noise of sigma per real FID point
        leaves sigma times the root of the summed squared window in the real spectrum.
        `gauss_hz` holds the twin's Gaussian broadening at the value read from the same
        data (SpectrumReading): a line's height goes with its width, so a twin as broad as
        the acquisition reads its noise, where a twin of the population's widths reads it
        only as well as this shim is typical."""
        dwell_s = 1.0 / self.bandwidth_hz
        window = np.exp(-np.pi * APODIZATION_HZ * np.arange(self.npoints) * dwell_s)
        ppm = np.fft.fftshift(np.fft.fftfreq(self.npoints, dwell_s)) / voxel._f0_mhz()
        scale = voxel.noise_place() * noise_scale(voxel.volume_ml, voxel.scan.engine.b0_t, dwell_s,
                                                  self.excitations)
        found = []
        for seed in range(seeds):
            truth = voxel.sample(seed)
            if gauss_hz is not None:
                dict.__setitem__(truth["lineshape"], "gauss_hz", float(gauss_hz))  # a realized leaf, as drawn
            _, fid = voxel._clean_fid(truth, self.npoints, dwell_s)
            row = []
            near = np.abs(ppm - LINES[self.line][0]) < LINES[self.line][1]
            # phased at the line's top as the data are, a phantom's water line under the apodization
            top = np.fft.fftshift(np.fft.fft(np.asarray(fid) * (window if self.line == "water" else 1.0)))
            phase = np.exp(-1j * np.angle(top[near][np.argmax(np.abs(top[near]))]))
            for apodization, measured in ((np.ones(self.npoints), self.snr), (window, self.snr_apodized)):
                height = (np.fft.fftshift(np.fft.fft(np.asarray(fid) * apodization)) * phase).real[near].max()
                row.append(height / (measured * np.sqrt((apodization ** 2).sum()) * scale))
            found.append(row)
        plain, apodized = np.median(np.array(found), axis=0)
        return float(plain), float(apodized)

    #************#
    #   record   #
    #************#
    def record(self, voxel, dataset, presets=(), citation=None, url=None, seeds=12, gauss_hz=None):
        """The Calibration of noise.sigma this reading makes with `voxel` as its virtual
        twin: the value is the apodized reading's, the prior a quarter either side of it
        (one acquisition has no spread of its own; stated), the conditions the file's and
        the twin's; `gauss_hz` as for `sigma`."""
        plain, apodized = self.sigma(voxel, seeds, gauss_hz)
        sequence = voxel.scan.sequence
        return Calibration(
            leaf="noise.sigma", value=round(apodized, 2),
            prior=(round(0.75 * apodized, 2), round(1.25 * apodized, 2)),
            measured={"statistic": f"single-transient {self.line} SNR (scan.calibrate.NoiseReading)",
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
            voxel={**({"load": voxel.scan.brain.name} if isinstance(voxel.scan.brain, Phantom) else {}),
                   "size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, SNR per transient, median",
            date=datetime.date.today().strftime("%d/%m/%y"), presets=tuple(presets),
            note="one acquisition; made by scan.calibrate",
        )


#**************************************************************************************************#
#                                      Class SpectrumReading                                       #
#**************************************************************************************************#
#                                                                                                  #
# What the averaged spectrum of one raw acquisition says without a fit: linewidth and creatine.    #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class SpectrumReading:
    """What the averaged spectrum of one raw acquisition says without a fit: the NAA
    singlet's width, and the 3.03 ppm creatine singlet's height against NAA's.

    The coils are combined as for the noise reading and the transients averaged as they
    are: no frequency alignment, so a drifting scanner widens the line (stated). The
    twin's spectrum is read the same way, noise off, so the two compare like for like:
    the width checks the lineshape prior (lineshape.gauss_hz), the ratio the creatine
    range (scan.calibrations CREATINE_THREE_DATASETS_3T).

    Args:
        naa_fwhm_hz: Full width at half maximum of the NAA singlet, under 2 Hz apodization.
        creatine_over_naa: Height of the 3.03 ppm singlet over the NAA singlet's.
        transients, coils, npoints, bandwidth_hz, f0_mhz: What the data held.
        header: What the file says of the scanner, coil, and sequence.
    """

    naa_fwhm_hz: float
    creatine_over_naa: float
    transients: int
    coils: int
    npoints: int
    bandwidth_hz: float
    f0_mhz: float
    header: dict

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, data, dwell_s, f0_mhz, header=None):
        """Read raw FIDs, (npoints, coils, transients) complex, the carrier on water."""
        x = np.asarray(data)
        ppm, real = _spectrum(_combined(x).mean(axis=1), dwell_s, lambda hz: WATER_PPM + hz / f0_mhz)
        width, creatine = _shape(ppm, real, f0_mhz)
        return cls(naa_fwhm_hz=width, creatine_over_naa=creatine, transients=x.shape[2], coils=x.shape[1],
                   npoints=x.shape[0], bandwidth_hz=1.0 / dwell_s, f0_mhz=float(f0_mhz),
                   header=dict(header or {}))

    @classmethod
    def from_nifti(cls, source):
        """Read a NIfTI-MRS file (a path, or a loaded image) with DIM_COIL and DIM_DYN."""
        x, dwell_s, f0_mhz, header = _raw(source)
        return cls.from_arrays(x, dwell_s, f0_mhz, header=header)

    #**********#
    #   twin   #
    #**********#
    def twin(self, voxel, seeds=12, gauss_hz=None):
        """The same two readings of `voxel`, the acquisition's virtual twin, noise off:
        {"naa_fwhm_hz": ..., "creatine_over_naa": ...}, each the median over `seeds`
        drawn subjects; `gauss_hz` holds the twin's Gaussian broadening at a value."""
        dwell_s, f0_mhz = 1.0 / self.bandwidth_hz, voxel._f0_mhz()
        rows = []
        for seed in range(seeds):
            truth = voxel.sample(seed)
            if gauss_hz is not None:
                dict.__setitem__(truth["lineshape"], "gauss_hz", float(gauss_hz))  # a realized leaf, as drawn
            _, fid = voxel._clean_fid(truth, self.npoints, dwell_s)
            rows.append(_shape(*_spectrum(np.asarray(fid), dwell_s, lambda hz: hz / f0_mhz), f0_mhz))
        width, creatine = np.median(np.array(rows), axis=0)
        return {"naa_fwhm_hz": float(width), "creatine_over_naa": float(creatine)}

    #**************#
    #   creatine   #
    #**************#
    def total_creatine_mm(self, voxel, seeds=12):
        """The total creatine (mM) at which the twin would show the measured ratio: the
        twin's drawn median scaled by measured over simulated (the singlet's height is
        linear in it, NAA held)."""
        drawn = np.median([float(voxel.resolve(truth.conc.Cr)) + float(voxel.resolve(truth.conc.PCr))
                           for truth in (voxel.sample(seed) for seed in range(seeds))])
        return float(drawn * self.creatine_over_naa / self.twin(voxel, seeds)["creatine_over_naa"])

    #************#
    #   record   #
    #************#
    def record(self, voxel, dataset, presets=(), citation=None, url=None, seeds=6):
        """The Calibration of lineshape.gauss_hz this reading makes with `voxel` as its
        virtual twin: the Gaussian broadening at which the twin's NAA singlet is as wide
        as the measured one, found by bisection (the twin's relaxation and sequence give
        the rest of the width), the prior a quarter either side (one acquisition has no
        spread of its own; stated). The shim of this scanner in this region of this
        subject: a number of the field, carried with its voxel's place."""
        low, high = 0.0, 4.0 * self.naa_fwhm_hz
        for _ in range(10):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if self.twin(voxel, seeds, middle)["naa_fwhm_hz"] < self.naa_fwhm_hz else (low, middle)
        value = round(0.5 * (low + high), 2)
        bare = self.twin(voxel, seeds, 0.0)["naa_fwhm_hz"]
        sequence = voxel.scan.sequence
        return Calibration(
            leaf="lineshape.gauss_hz", value=value, prior=(round(0.75 * value, 2), round(1.25 * value, 2)),
            measured={"statistic": "full width at half maximum of the NAA singlet in the averaged, coil-combined "
                                   "spectrum under 2 Hz apodization, in Hz (scan.calibrate.SpectrumReading)",
                      "naa_fwhm_hz": round(self.naa_fwhm_hz, 2),
                      "twin_without_broadening_hz": round(bare, 2)},
            # the rule: a twin already as wide as the line without any broadening leaves the bisection at its
            # floor, and the line's width was not read
            failed=("" if bare < self.naa_fwhm_hz else
                    f"the twin without broadening ({bare:.1f} Hz) is already as wide as the measured line ({self.naa_fwhm_hz:.1f} Hz)"),
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                     "model": self.header.get("ManufacturersModelName"),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or type(sequence).__name__,
                      "family": type(sequence).__name__, "te_s": self.header.get("EchoTime")},
            acquisition={"npoints": self.npoints, "bandwidth_hz": self.bandwidth_hz,
                         "transients": self.transients, "measured_on": "the average"},
            voxel={"size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the transients averaged as they "
                       "are (a drifting scanner widens the line)",
            date=datetime.date.today().strftime("%d/%m/%y"), presets=tuple(presets),
            note="one acquisition; made by scan.calibrate",
        )


#**************************************************************************************************#
#                                     Class RelaxationReading                                      #
#**************************************************************************************************#
#                                                                                                  #
# The T2 of a phantom's singlets, read off their heights over a series of echo times.              #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class RelaxationReading:
    """The T2 of a phantom's singlets, read off their heights over a series of echo times.

    A phantom's relaxation times are in no table (phantom.Phantom); its own scans give
    them. Each FID of the series, coils combined and transients averaged, is read as the
    other readings read a spectrum (phased on the NAA singlet and referenced to it, 2 Hz
    apodization), the height of every singlet taken at its place, and the T2 is the slope
    of the heights' logarithm against the echo time: two numbers per line from as many
    heights as there are echoes, no model of the spectrum. Singlets only: a coupled
    line's height swings with the echo time, and so does what lies under a singlet. On a
    virtual Braino the creatine singlet reads 3 percent short and NAA's 6 (glutamate
    beneath it), choline's 16 (myo-inositol beneath it): a bound for choline, a number
    for the other two. The T2 is the sequence's (a PRESS series carries its pulses'
    imperfections), as every published one is. In vitro: it sets a Phantom's t2_s, never
    a tissue leaf.

    Args:
        te_s: The echo times.
        heights: {line: heights over the echo times}, NAA's at the first echo being one.
        t2_s: {line: T2 in seconds}.
    """

    LINES = {"NAA": 2.008, "Cr": 3.03, "Cho": 3.19}

    te_s: tuple
    heights: dict
    t2_s: dict

    #*******************#
    #   from a series   #
    #*******************#
    @classmethod
    def from_series(cls, fids, dwell_s, f0_mhz, lines=None, water_ppm=WATER_PPM):
        """Read {echo time in seconds: FID (npoints,) complex}; `lines` is {name: ppm} of
        the singlets, `water_ppm` where the carrier sits (a cold phantom's water lies
        above 4.65 ppm)."""
        lines = dict(lines or cls.LINES)
        te_s = tuple(sorted(fids))
        heights = {name: [] for name in lines}
        for te in te_s:
            ppm, real = _spectrum(np.asarray(fids[te]), dwell_s, lambda hz: water_ppm + hz / f0_mhz)
            for name, at in lines.items():
                heights[name].append(real[np.abs(ppm - at) < 0.04].max())
        first = heights["NAA"][0] if "NAA" in heights else 1.0
        heights = {name: tuple(float(h / first) for h in values) for name, values in heights.items()}
        t2_s = {name: float(-1.0 / np.polyfit(te_s, np.log(values), 1)[0]) for name, values in heights.items()}
        return cls(te_s=te_s, heights=heights, t2_s=t2_s)


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
        header = {key: extension[key] for key in HEADER if extension.get(key) not in (None, "")}
        reading = cls.from_arrays(data[start:], header=header)
        # a signal that still rings there is one mode shared by every channel: an unsuppressed
        # phantom's water reads 1.00 where loaded arrays read 0.05 to 0.25
        if reading.coils > 2 and reading.statistics()["median_pair"] > 0.8:
            raise ValueError("the FIDs still ring at their end (the channels correlate as one): "
                             "give the scanner's own noise samples to from_arrays")
        return reading

    #******************#
    #   from ismrmrd   #
    #******************#
    @classmethod
    def from_ismrmrd(cls, path, lines=None):
        """Read the noise of an ISMRMRD (.mrd, .h5) file: the acquisitions `lines` indexes,
        else those flagged as a noise measurement (the pre-scan a scanner's converter keeps
        in the file), else all of them, for a file that holds noise only. Needs h5py."""
        import re

        import h5py

        with h5py.File(str(path), "r") as file:
            table = file["dataset/data"]
            if lines is None:
                flagged = np.flatnonzero(table["head"]["flags"] & np.uint64(1 << 18))  # ACQ_IS_NOISE_MEASUREMENT
                lines = flagged if flagged.size else range(len(table))
            total, count, coils = None, 0, None
            for index in lines:
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


#**************************************************************************************************#
#                                        Class WaterReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The residual water and the receiver delay of one suppressed acquisition against its reference.   #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class WaterReading:
    """What one suppressed acquisition and its water reference say of the scanner's
    water suppression and receiver, read the way the package's records were
    (info/HANDOFF.md items 217 and 219): the residual water as the exponentials within
    0.4 ppm of the carrier in the mean suppressed FID at t = 0 (`_exponentials`) over the
    reference's mean FID at t = 0, both combined with the suppressed file's own weights;
    the receiver delay as the phase of creatine against NAA over their distance, a
    first-order phase, read on creatine alone, BEYOND the phase the sequence itself
    gives creatine's peak (the coupled neighbours under 3.03 ppm evolve over the echo:
    a PRESS at TE 35 ms bends the peak by ten degrees with no delay at all; the twin
    says how much, `intrinsic_phase_deg`); and the residual's phase beyond the line
    through NAA and creatine, whose side (past a quarter turn: inverted) signs it.

    Attributes:
        residual_over_full: The residual water at t = 0 over the reference's, positive.
        residual_over_dry_max: The same over the maximum of the FID without its water.
        water_phase_deg: The residual's phase against the metabolites, the line taken out.
        creatine_phase_deg: Creatine's phase against NAA as measured.
        creatine_offset_hz: Creatine's distance from NAA, the ppm way round, in hertz.
        dry_amplitude: The FID without its water at t = 0, in the data's units, the twin's scale.
        residual_fid: The residual water as the data show it, its fitted components narrow and
            broad, turned so its line has `water_phase_deg` against the metabolites' frame.
        leak_over_full: The residual's broad part (T2* under 35 ms, the water outside the
            voxel the localization leaks) at t = 0 over the reference's; inside
            `residual_over_full`, which is the total (info/HANDOFF.md item 238).
        leak_t2star_s, leak_offset_hz: The heaviest broad part's decay and its frequency
            from the carrier.
        leak_phase_deg: The broad parts' phase against the narrow ones'.
        eddy_transient_rad: The reference's phase 5 ms after its first sample against its
            components fitted late (item 236).
    """

    residual_over_full: float
    residual_over_dry_max: float
    water_phase_deg: float
    creatine_phase_deg: float
    creatine_offset_hz: float
    dry_amplitude: float
    residual_fid: np.ndarray
    leak_over_full: float
    leak_t2star_s: float
    leak_offset_hz: float
    leak_phase_deg: float
    eddy_transient_rad: float
    transients: int
    reference_transients: int
    npoints: int
    coils: int
    dwell_s: float
    f0_mhz: float
    header: dict

    #*******************#
    #   from arrays     #
    #*******************#
    @classmethod
    def from_arrays(cls, data, reference, dwell_s, f0_mhz, header=None):
        """`data` and `reference` (npoints, coils, transients) of the same voxel, the
        suppressed run and its unsuppressed reference, in the standard's frame (the
        carrier at 4.65 ppm)."""
        suppressed = _combined(data).mean(axis=1)
        full = float(np.abs(_combined(reference, like=data).mean(axis=1)[0]))
        # the residual water: the decaying exponentials within 0.4 ppm of the carrier, at t = 0
        m = min(1024, suppressed.size)
        freqs, damping, amplitudes = _exponentials(suppressed[:m], dwell_s)
        water = (np.abs(freqs) < 0.4 * f0_mhz) & (damping < 0.0)          # within 0.4 ppm of the carrier, as the records
        residual = complex(amplitudes[water].sum())
        # the broad parts, the water the localization leaks; the narrow ones the voxel's own residual
        t2star = -1.0 / damping[water]
        broad = t2star < 0.035
        leak, narrow = complex(amplitudes[water][broad].sum()), complex(amplitudes[water][~broad].sum())
        heaviest = int(np.argmax(np.abs(amplitudes[water][broad]))) if broad.any() else None
        reference_fid = _combined(reference, like=data).mean(axis=1)
        t = np.arange(suppressed.size) * dwell_s
        fitted = sum(a * np.exp((1j * 2.0 * np.pi * f + d) * t) for f, d, a in zip(freqs[water], damping[water], amplitudes[water]))
        dry = suppressed - fitted
        # the first-order phase: the complex peaks of the mean spectrum, NAA phased to zero
        lines = _lines(suppressed, dwell_s, f0_mhz, WATER_PPM)
        (naa, f_naa), (creatine, f_cr), (residual_line, f_water) = lines["NAA"], lines["Cr"], lines["water"]
        phase_cr = float(np.angle(creatine / naa))
        slope = phase_cr / (f_cr - f_naa)                       # radians per hertz on creatine alone
        beyond = float(np.degrees(np.angle(np.exp(1j * (np.angle(residual_line / naa) - slope * (f_water - f_naa))))))
        return cls(residual_over_full=abs(residual) / full, residual_over_dry_max=abs(residual) / float(np.abs(dry).max()),
                   water_phase_deg=beyond, creatine_phase_deg=float(np.degrees(phase_cr)), creatine_offset_hz=float(f_cr - f_naa),
                   dry_amplitude=float(np.abs(dry[0])),
                   # in the engine's convention, as `_lines` reads the phases (a file stored as the conjugate: the FID
                   # conjugated), turned so its line has `beyond` against the metabolites' frame
                   residual_fid=(fitted if _axis(suppressed, dwell_s, f0_mhz, WATER_PPM)[0] > 0 else np.conj(fitted))
                   * np.exp(1j * (np.radians(beyond) - np.angle(residual_line))),
                   leak_over_full=abs(leak) / full,
                   leak_t2star_s=float(t2star[broad][heaviest]) if heaviest is not None else 0.0,
                   leak_offset_hz=float(freqs[water][broad][heaviest]) if heaviest is not None else 0.0,
                   leak_phase_deg=float(np.degrees(np.angle(leak / narrow))) if heaviest is not None and narrow != 0.0 else 0.0,
                   eddy_transient_rad=_eddy_transient(reference_fid, dwell_s, f0_mhz),
                   transients=int(np.asarray(data).shape[2]), reference_transients=int(np.asarray(reference).shape[2]),
                   npoints=int(suppressed.size), coils=int(np.asarray(data).shape[1]), dwell_s=float(dwell_s), f0_mhz=float(f0_mhz),
                   header=dict(header or {}))

    @classmethod
    def from_nifti(cls, source, reference=None):
        """From two NIfTI-MRS files with the transients apart: the suppressed run and its reference,
        or with no reference the water transients the run interleaves with its own (`_raw`)."""
        data, dwell_s, f0_mhz, header = _raw(source)
        water, dwell_ref, _, _ = _raw(source, water=True) if reference is None else _raw(reference)
        if water.shape[2] == 0:
            raise ValueError("the run interleaves no water transients; give its water reference")
        if abs(dwell_ref - dwell_s) > 1e-9:
            raise ValueError("the reference is sampled at another dwell than the scan")
        return cls.from_arrays(data, water, dwell_s, f0_mhz, header)

    #****************#
    #   conditions   #
    #****************#
    def _conditions(self, voxel, dataset, citation, url):
        sequence = voxel.scan.sequence
        return dict(
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                     "model": self.header.get("ManufacturersModelName"), "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or type(sequence).__name__, "family": type(sequence).__name__,
                      "te_s": self.header.get("EchoTime"), "tr_s": self.header.get("RepetitionTime"),
                      "simulated_as": type(sequence).__name__},
            acquisition={"npoints": self.npoints, "bandwidth_hz": 1.0 / self.dwell_s, "transients": self.transients,
                         "reference_transients": self.reference_transients, "water_suppressed": True},
            voxel={**({"load": voxel.scan.brain.name} if isinstance(voxel.scan.brain, Phantom) else {}),
                   "size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm)},
            date=datetime.date.today().strftime("%d/%m/%y"), note="one acquisition; made by scan.calibrate")

    #***************#
    #   the delay   #
    #***************#
    def intrinsic_phase_deg(self, voxel, gauss_hz=None, seeds=3):
        """The phase creatine's peak carries against NAA's with no delay: the twin's
        sources but its water, rendered at a zero delay (as broad as `gauss_hz` says,
        the acquisition's own width), plus the residual water as the data show it (its
        fitted components, the broad ones the localization leaks among them, at their
        phase against the metabolites, scaled by the FIDs without their water at t = 0),
        the median over `seeds` draws. What the sequence's own evolution, the
        macromolecules and the residual's wing put under 3.03 ppm, which a delay reading
        must take out. A residual taken as one narrow line of the reference's shape left
        the leak's broader wing in every delay row, where the console plays it again
        (info/HANDOFF.md item 244)."""
        t = np.arange(self.npoints) * self.dwell_s
        turn = np.exp(-2j * np.pi * WATER_PPM * voxel._f0_mhz() * t)     # the twin into the standard's frame
        phases = []
        for seed in range(seeds):
            truth = voxel.sample(seed)
            dict.__setitem__(truth.acquisition, "delay_s", 0.0)
            if gauss_hz is not None:
                dict.__setitem__(truth.lineshape, "gauss_hz", float(gauss_hz))
            parts = voxel._component_fids(truth, self.npoints, self.dwell_s)
            dry = sum(np.asarray(fid) for name, fid in parts.items() if name not in ("water", "oov")) * turn
            lines = _lines(dry + (abs(dry[0]) / self.dry_amplitude) * self.residual_fid, self.dwell_s, voxel._f0_mhz(), WATER_PPM)
            phases.append(np.degrees(np.angle(lines["Cr"][0] / lines["NAA"][0])))
        return float(np.median(phases))

    def delay_s(self, voxel, gauss_hz=None):
        """The receiver delay: creatine's measured phase beyond the twin's intrinsic one,
        over creatine's distance from NAA; negative when the first sample sits before
        the echo top."""
        return float(np.radians(self.creatine_phase_deg - self.intrinsic_phase_deg(voxel, gauss_hz))
                     / (2.0 * np.pi * self.creatine_offset_hz))

    #*************#
    #   records   #
    #*************#
    def record_residual(self, voxel, dataset, presets=(), citation=None, url=None):
        """The Calibration of water.adjusted_residual: the residual over the full water,
        signed by its side (inverted past a quarter turn), the prior a quarter either side."""
        value = self.residual_over_full * (-1.0 if abs(self.water_phase_deg) > 90.0 else 1.0)
        return Calibration(
            leaf="water.adjusted_residual", value=float(round(value, 6)),
            prior=tuple(sorted((round(0.75 * value, 6), round(1.25 * value, 6)))),
            measured={"statistic": "residual water at t = 0 over the water reference's, signed by its phase against "
                                   "the metabolites (scan.calibrate.WaterReading)",
                      "residual_over_full": round(self.residual_over_full, 6), "over_dry_maximum": round(self.residual_over_dry_max, 3),
                      "phase_deg": round(self.water_phase_deg, 1), "inverted": abs(self.water_phase_deg) > 90.0},
            processing="NIfTI-MRS, the suppressed file's noise-whitened SVD coil combination on both files, the exponentials "
                       "within 0.4 ppm of the carrier at t = 0 over the reference FID at t = 0",
            presets=tuple(presets), **self._conditions(voxel, dataset, citation, url))

    def record_leak(self, voxel, dataset, presets=(), citation=None, url=None):
        """The Calibration of oov.fraction: the residual's broad part over the full water, the
        prior a quarter either side (info/HANDOFF.md item 238)."""
        value = self.leak_over_full
        return Calibration(
            leaf="oov.fraction", value=float(round(value, 6)),
            prior=(round(0.75 * value, 6), round(1.25 * value, 6)),
            measured={"statistic": "the broad exponentials (T2* under 35 ms) of the suppressed mean FID within 0.4 ppm of the "
                                   "carrier at t = 0 over the reference's full water (scan.calibrate.WaterReading)",
                      "over_full": round(value, 6), "t2star_ms": round(1000.0 * self.leak_t2star_s, 1),
                      "offset_hz": round(self.leak_offset_hz, 2), "phase_vs_residual_deg": round(self.leak_phase_deg, 1)},
            processing="NIfTI-MRS, the suppressed file's noise-whitened SVD coil combination on both files, the exponentials "
                       "within 0.4 ppm of the carrier split at a T2* of 35 ms",
            # the rule: the split is a 3 T number; at 7 T the voxel's own water decays within it (ds005917's reference
            # water at 23 ms, its whole residual counted broad), so the broad part is not the leak there
            failed="" if self.f0_mhz < 4.0 * 42.5774785 else "the split at a T2* of 35 ms is set for 3 T: above 4 T the voxel's own water decays within it",
            presets=tuple(presets), **self._conditions(voxel, dataset, citation, url))

    def record_eddy(self, voxel, dataset, presets=(), citation=None, url=None):
        """The Calibration of acquisition.eddy_transient_rad: the reference's phase 5 ms after
        its first sample against its components fitted late, the prior a quarter either
        side of it (one subject's own; a population's record is a spread about zero); no
        fallback, this scanner's (info/HANDOFF.md item 236)."""
        value = self.eddy_transient_rad
        return Calibration(
            leaf="acquisition.eddy_transient_rad", value=float(round(value, 4)),
            prior=tuple(sorted((round(0.75 * value, 4), round(1.25 * value, 4)))),
            measured={"statistic": "the water reference's phase 5 ms after its first sample against its components "
                                   "fitted on 60 to 250 ms and carried back (scan.calibrate.WaterReading)",
                      "at_5ms_rad": round(value, 4)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the reference's mean FID; Hankel SVD on the late "
                       "window, the model carried back to the start",
            # the rules: a phase past a quarter turn 5 ms in is the late model landing a half turn off, not a transient; the
            # late window is set for 3 T, and at 7 T the water has decayed in it (T2* 23 ms on ds005917, 6 of 25 past the turn)
            failed=("the late window, 60 to 250 ms, is set for 3 T: above 4 T the water has decayed in it" if self.f0_mhz >= 4.0 * 42.5774785
                    else "" if abs(value) <= 0.5 * np.pi else f"the phase 5 ms in ({value:.2f} rad) is past a quarter turn: the late model's own wrap"),
            presets=tuple(presets), fallback=False, **self._conditions(voxel, dataset, citation, url))

    def record_delay(self, voxel, dataset, presets=(), citation=None, url=None, gauss_hz=None, seeds=3):
        """The Calibration of acquisition.delay_s: creatine's phase against NAA beyond the
        twin's intrinsic one (the twin as broad as `gauss_hz`, drawn `seeds` times), over
        their distance, a delay of the samples against the echo top; no fallback, this
        scanner's."""
        intrinsic = self.intrinsic_phase_deg(voxel, gauss_hz, seeds)
        delay = float(np.radians(self.creatine_phase_deg - intrinsic) / (2.0 * np.pi * self.creatine_offset_hz))
        return Calibration(
            leaf="acquisition.delay_s", value=float(round(delay, 7)),
            prior=tuple(sorted((round(0.75 * delay, 7), round(1.25 * delay, 7)))),
            measured={"statistic": "phase of creatine against NAA beyond the twin's intrinsic phase, over their frequency "
                                   "difference, as seconds (scan.calibrate.WaterReading)",
                      "delay_us": round(1e6 * delay, 1), "creatine_phase_deg": round(self.creatine_phase_deg, 1),
                      "intrinsic_phase_deg": round(intrinsic, 1), "creatine_offset_hz": round(self.creatine_offset_hz, 1)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the mean FID's spectrum fourfold zero filled, "
                       "phases at the peaks' maxima",
            presets=tuple(presets), fallback=False, **self._conditions(voxel, dataset, citation, url))

