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
#          the noise covariance of a receive array's channels, the tissue water's T2 off a series #
#          of water references over the echo time, and the macromolecules' level off the 0.9 ppm  #
#          peak of a short-echo run (scan.calibrations).                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import datetime
import functools

import numpy as np
import scipy.optimize
from keras import ops

from mrsandbox.hardware.receiver import Receiver
from mrsandbox.params.values import Tissues
from mrsandbox.phantom.macromolecules import _echo_time_s
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
# the second reading of the noise: the plain single-transient NAA SNR follows the in vivo line,
# the apodized one the broad signal under the singlet; they err in opposite directions and each
# gives a sigma
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
        turns = np.round(np.angle(np.einsum("nc,ncs->s", np.conj(w[:64, :, 0]), w[:64]))
                         / (np.pi / 2))
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
    sign = max((1.0, -1.0), key=lambda s: np.abs(
        spectrum[np.abs(carrier_ppm + s * hz / f0_mhz - NAA_PPM) < 0.15]).max())
    return sign, spectrum, hz


#****************#
#   registered   #
#****************#
def _registered(fids, dwell_s, f0_mhz, span_s=0.2):
    """(npoints, transients) FIDs, the carrier on water, turned onto their mean: each
    transient's frequency and phase fitted on the metabolites, 1.8 to 4.2 ppm of the FIDs'
    first `span_s` (the rest is mostly noise, which a trial frequency twists), by least
    squares (spectral registration, Near et al. 2015, Magn Reson Med 73:44). The readings of
    the line and the noise then hold the scanner's own, and the run's drift and jitter are
    left to the run's settings, which play them."""
    import scipy.optimize

    n = min(fids.shape[0], int(round(span_s / dwell_s)))
    t = np.arange(fids.shape[0]) * dwell_s
    sign, _, _ = _axis(fids.mean(axis=1), dwell_s, f0_mhz, WATER_PPM)
    hz = np.fft.fftshift(np.fft.fftfreq(n, dwell_s))
    band = np.abs(WATER_PPM + sign * hz / f0_mhz - 3.0) < 1.2
    mean = np.fft.fftshift(np.fft.fft(fids[:n].mean(axis=1)))[band]
    turned = []
    for fid in fids.T:
        def rest(p, fid=fid):
            d = np.fft.fft(fid[:n] * np.exp(-2j * np.pi * p[0] * t[:n] - 1j * p[1]))
            d = np.fft.fftshift(d)[band] - mean
            return np.concatenate([d.real, d.imag])

        shift, phase = scipy.optimize.least_squares(rest, [0.0, 0.0], x_scale="jac").x
        turned.append(fid * np.exp(-2j * np.pi * shift * t - 1j * phase))
    return np.stack(turned, axis=1)


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
    `order` components fitted to the first half of `fid` in a Hankel matrix; the residual
    water records are read with it."""
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
    metabolites, carried back to the start; what the early phase does against that model.
    Exact for transients dead by the window's start, a lower bound for slower ones (the
    model's frequencies take a slow phase into a ramp)."""
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
    unsuppressed water transients with its own (ds005917 stores one every 16th), found as
    the transients whose start stands far above the median, gives its suppressed
    transients, or with `water` the interleaved water, the run's own reference (none: no
    transient)."""
    import json

    import nibabel as nib

    image = (nib.load(str(source))
             if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source)
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
    x = np.transpose(data, order + rest).reshape(data.shape[0], data.shape[order[1]],
                                                 data.shape[order[2]])
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
    padded = np.concatenate([np.asarray(fid)
                             * np.exp(-np.pi * apodization_hz * np.arange(n) * dwell_s),
                             np.zeros((fill - 1) * n, dtype=np.complex128)])
    spectrum = np.fft.fftshift(np.fft.fft(padded))
    hz = np.fft.fftshift(np.fft.fftfreq(fill * n, dwell_s))
    ppm = max((ppm_of(sign * hz) for sign in (1.0, -1.0)),
              key=lambda axis: np.abs(spectrum[np.abs(axis - NAA_PPM) < 0.15]).max())
    peak = np.argmax(np.abs(spectrum) * (np.abs(ppm - NAA_PPM) < 0.15))
    real = (spectrum * np.exp(-1j * np.angle(spectrum[peak]))).real
    order = np.argsort(ppm - (ppm[peak] - NAA_PPM))
    return (ppm - (ppm[peak] - NAA_PPM))[order], real[order]


def _echo_top(parts, dwell_s):
    """The echo top of water line parts [(freqs, amps, lw_hz, tissue)] as the twin's
    FID would show it over points 2 to 8, where each part has decayed at its own width
    (the lineshape the parts share cancels in a series' ratios)."""
    t = np.arange(2, 8) * dwell_s
    return abs(sum(complex(np.sum(np.asarray(ops.convert_to_numpy(amps))))
                   * float(np.mean(np.exp(-np.pi * lw * t))) for _, amps, lw, _ in parts))


def _shape(ppm, real, f0_mhz):
    """What a spectrum says without a fit: the NAA singlet's full width at half maximum
    over the level at 1.6 ppm (hertz), not a number where the line does not fall to half
    its height on both sides within 0.12 ppm (broader than that, or lost in noise), and the
    height of the 3.03 ppm creatine singlet over NAA's."""
    near = np.abs(ppm - NAA_PPM) < 0.12
    x, y = ppm[near], real[near]
    top = int(np.argmax(y))
    half = (np.median(real[np.abs(ppm - 1.6) < 0.1])
            + 0.5 * (y[top] - np.median(real[np.abs(ppm - 1.6) < 0.1])))
    creatine = real[np.abs(ppm - 3.03) < 0.04].max() / real[np.abs(ppm - NAA_PPM) < 0.04].max()
    if not ((y[:top] < half).any() and (y[top + 1:] < half).any()):
        return float("nan"), float(creatine)
    left = top - int(np.argmax(y[top::-1] < half))
    right = top + int(np.argmax(y[top:] < half))
    low = np.interp(half, [y[left], y[left + 1]], [x[left], x[left + 1]])
    high = np.interp(half, [y[right], y[right - 1]], [x[right], x[right - 1]])
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
    eighth and combined with the first singular vector of the mean transient's start, and
    in vivo the transients are registered on the metabolites (`_registered`); nothing else
    is done to the data, no fit of a model enters (scan.calibrations, the measure).
    Averaged or combined data do not serve: a stored sum of excitations is not their
    ideal sum, so `excitations` is for a sequence whose unit is more than one shot
    (SPECIAL's pair) and for nothing else.

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
        # and the edge of a cropped band would count as noise.
        fids = _combined(x, noise)
        # registered on the metabolites in vivo; a phantom's water line has none to register on
        fids = _registered(fids, dwell_s, f0_mhz) if line == "NAA" else fids
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
            [(real[quiet] - np.polyval(np.polyfit(ppm[quiet], real[quiet], 1), ppm[quiet])).std()
             for real in spectra.T])
        plain = float(np.median(spectra[near].max(axis=0) / floors))
        apodized = float(np.median(broad[near].max(axis=0)
                                   / (floors * np.sqrt((window ** 2).mean()))))
        return cls(snr=plain, snr_apodized=apodized, naa_ppm=float(ppm[peak]),
                   transients=transients, coils=coils, npoints=n, bandwidth_hz=1.0 / dwell_s,
                   f0_mhz=float(f0_mhz), excitations=int(excitations), header=dict(header or {}),
                   line=line)

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
    def sigma(self, voxel, seeds=12, held=None):
        """The reference noise level (the noise.sigma leaf) at which `voxel`, a Voxel of a
        scan set up as the acquisition was (field, sequence, timings, place, size),
        reproduces the reading: (plain, apodized), each the median over `seeds` drawn
        subjects. Nothing is drawn for the noise: white noise of sigma per real FID point
        leaves sigma times the root of the summed squared window in the real spectrum.
        `held` {leaf: value} holds the twin's leaves at the values read off the same data
        (the width, SpectrumReading; the residual water, delay and eddy transient,
        WaterReading): a line's height goes with its width and with the residual's wing
        under it, so a twin at the acquisition's own readings reads its noise, where one
        drawn from the population's ranges reads it only as well as this run is typical."""
        dwell_s = 1.0 / self.bandwidth_hz
        window = np.exp(-np.pi * APODIZATION_HZ * np.arange(self.npoints) * dwell_s)
        ppm = np.fft.fftshift(np.fft.fftfreq(self.npoints, dwell_s)) / voxel._f0_mhz()
        scale = voxel.noise_place() * noise_scale(voxel.volume_ml, voxel.scan.engine.b0_t, dwell_s,
                                                  self.excitations)
        found = []
        for seed in range(seeds):
            truth = voxel.sample(seed)
            for leaf, value in (held or {}).items():
                branch, _, key = leaf.rpartition(".")
                # a realized leaf, as drawn
                dict.__setitem__(functools.reduce(lambda node, part: node[part],
                                                  branch.split("."), truth), key, float(value))
            _, fid = voxel._clean_fid(truth, self.npoints, dwell_s)
            row = []
            near = np.abs(ppm - LINES[self.line][0]) < LINES[self.line][1]
            # phased at the line's top as the data are, a phantom's water line under the apodization
            top = np.fft.fftshift(np.fft.fft(np.asarray(fid)
                                             * (window if self.line == "water" else 1.0)))
            phase = np.exp(-1j * np.angle(top[near][np.argmax(np.abs(top[near]))]))
            for apodization, measured in ((np.ones(self.npoints), self.snr),
                                          (window, self.snr_apodized)):
                height = (np.fft.fftshift(np.fft.fft(np.asarray(fid) * apodization))
                          * phase).real[near].max()
                row.append(height / (measured * np.sqrt((apodization ** 2).sum()) * scale))
            found.append(row)
        plain, apodized = np.median(np.array(found), axis=0)
        return float(plain), float(apodized)

    #************#
    #   record   #
    #************#
    def record(self, voxel, dataset, presets=(), citation=None, url=None, seeds=12, held=None):
        """The Calibration of noise.sigma this reading makes with `voxel` as its virtual
        twin: the value is the apodized reading's, the prior a quarter either side of it
        (one acquisition has no spread of its own; stated), the conditions the file's and
        the twin's; `held` as for `sigma`, kept with the reading."""
        plain, apodized = self.sigma(voxel, seeds, held)
        sequence = voxel.scan.sequence
        return Calibration(
            leaf="noise.sigma", value=round(apodized, 2),
            prior=(round(0.75 * apodized, 2), round(1.25 * apodized, 2)),
            measured={"statistic": f"single-transient {self.line} SNR "
                                   "(scan.calibrate.NoiseReading)",
                      "median": round(self.snr, 1), "implied_value": round(plain, 2),
                      "naa_found_at_ppm": round(self.naa_ppm, 3),
                      "apodized": {"statistic": APODIZED, "median": round(self.snr_apodized, 1),
                                   "implied_value": round(apodized, 2)},
                      "twin_held_at": dict(held or {})},
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
            voxel={**({"load": voxel.scan.brain.name}
                      if isinstance(voxel.scan.brain, Phantom) else {}),
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

    The coils are combined and the transients registered as for the noise reading, then
    averaged: a drifting run does not widen the line, its drift is the run's own. The
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
        ppm, real = _spectrum(_registered(_combined(x), dwell_s, f0_mhz).mean(axis=1), dwell_s,
                              lambda hz: WATER_PPM + hz / f0_mhz)
        width, creatine = _shape(ppm, real, f0_mhz)
        return cls(naa_fwhm_hz=width, creatine_over_naa=creatine, transients=x.shape[2],
                   coils=x.shape[1], npoints=x.shape[0], bandwidth_hz=1.0 / dwell_s,
                   f0_mhz=float(f0_mhz), header=dict(header or {}))

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
                # a realized leaf, as drawn
                dict.__setitem__(truth["lineshape"], "gauss_hz", float(gauss_hz))
            _, fid = voxel._clean_fid(truth, self.npoints, dwell_s)
            rows.append(_shape(*_spectrum(np.asarray(fid), dwell_s, lambda hz: hz / f0_mhz),
                               f0_mhz))
        width, creatine = np.nanmedian(np.array(rows), axis=0)
        return {"naa_fwhm_hz": float(width), "creatine_over_naa": float(creatine)}

    #**************#
    #   creatine   #
    #**************#
    def total_creatine_mm(self, voxel, seeds=12):
        """The total creatine (mM) at which the twin would show the measured ratio: the
        twin's drawn median scaled by measured over simulated (the singlet's height is
        linear in it, NAA held)."""
        drawn = np.median([float(voxel.resolve(truth.conc.Cr))
                           + float(voxel.resolve(truth.conc.PCr))
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
        subject: a number of the field, carried with its voxel's place. A line whose width
        could not be read gives a failed record."""
        unread = not np.isfinite(self.naa_fwhm_hz)
        low, high = 0.0, 0.0 if unread else 4.0 * self.naa_fwhm_hz
        for _ in range(10):
            middle = 0.5 * (low + high)
            low, high = ((middle, high)
                         if self.twin(voxel, seeds, middle)["naa_fwhm_hz"] < self.naa_fwhm_hz
                         else (low, middle))
        value = round(0.5 * (low + high), 2)
        bare = self.twin(voxel, seeds, 0.0)["naa_fwhm_hz"]
        sequence = voxel.scan.sequence
        return Calibration(
            leaf="lineshape.gauss_hz", value=value,
            prior=(round(0.75 * value, 2), round(1.25 * value, 2)),
            measured={"statistic": "full width at half maximum of the NAA singlet in the "
                                   "averaged, coil-combined spectrum under 2 Hz apodization, in "
                                   "Hz (scan.calibrate.SpectrumReading)",
                      "naa_fwhm_hz": None if unread else round(self.naa_fwhm_hz, 2),
                      "twin_without_broadening_hz": round(bare, 2)},
            # the rule: a twin already as wide as the line without any broadening leaves the
            # bisection at its floor, and the line's width was not read
            failed=("the NAA line does not fall to half its height within 0.12 ppm on both "
                    "sides: its width was not read" if unread else
                    "" if bare < self.naa_fwhm_hz else
                    f"the twin without broadening ({bare:.1f} Hz) is already as wide as the "
                    f"measured line ({self.naa_fwhm_hz:.1f} Hz)"),
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
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the transients averaged "
                       "as they are (a drifting scanner widens the line)",
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
    line's height swings with the echo time, and so does what lies under a singlet
    (glutamate under NAA, myo-inositol under choline), so choline's T2 is a bound and the
    other two are numbers. The T2 is the sequence's (a PRESS series carries its pulses'
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
        heights = {name: tuple(float(h / first) for h in values)
                   for name, values in heights.items()}
        t2_s = {name: float(-1.0 / np.polyfit(te_s, np.log(values), 1)[0])
                for name, values in heights.items()}
        return cls(te_s=te_s, heights=heights, t2_s=t2_s)


#**************************************************************************************************#
#                                     Class WaterSeriesReading                                     #
#**************************************************************************************************#
#                                                                                                  #
# The tissue water's T2 read off water references over the echo time, through the twin.            #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class WaterSeriesReading:
    """The tissue water's T2 read off a series of water references over the echo time,
    through the twin.

    Each reference's echo top (the mean magnitude of the combined mean FID's points 2
    to 8) falls with the echo time. The series is one scale when the tops fall as one
    exponential (`misfit`) at a brain water's rate (`apparent_t2_s` inside PLAUSIBLE_S);
    otherwise the series carries a scale between its scans (a reconstruction, a transmit
    setting) and the reading refuses (`failed`). The water's decay is not the tissue's
    T2 alone: the CSF's share, its own T2 and its diffusion through the crushers, the
    tissue water's diffusion, the T1 over the repetition and the pulses' relaxation all
    lie in it, and all of them in the twin. So the tissue's T2 is the value at which the
    twin's water, played through the same designs at the data's voxel (its own tissue
    fractions), falls as the data's (`tissue_t2_s`): the drawn grey and white T2s scaled
    together (the data cannot part them; the table's ratio stays), the CSF's as drawn.

    Args:
        te_s: The echo times in seconds, ascending.
        tops: The echo tops over the first.
        apparent_t2_s: One exponential through the tops.
        misfit: The rms of the tops' logarithms about that exponential.
    """

    # a brain water's T2 at any field lies inside; a series decaying faster or slower
    # carries a scale between its scans
    PLAUSIBLE_S = (0.04, 0.2)
    MISFIT = 0.05

    te_s: tuple
    tops: tuple
    apparent_t2_s: float
    misfit: float
    transients: tuple
    npoints: int
    coils: int
    dwell_s: float
    f0_mhz: float
    header: dict

    #*******************#
    #   from a series   #
    #*******************#
    @classmethod
    def from_arrays(cls, series, dwell_s, f0_mhz, header=None):
        """Read {echo time in seconds: (npoints, coils, transients)} water references of
        one voxel."""
        te_s = tuple(sorted(series))
        first = np.asarray(series[te_s[0]])
        # one whitening and one set of weights for the series, the first reference's: each
        # reference's own tail would put a few percent of its own on its top
        tops = np.array([float(np.abs(_combined(series[te], like=first).mean(axis=1)[2:8]).mean())
                         for te in te_s])
        slope, offset = np.polyfit(te_s, np.log(tops), 1)
        line = np.polyval((slope, offset), te_s)
        return cls(te_s=te_s, tops=tuple(float(top / tops[0]) for top in tops),
                   apparent_t2_s=float(-1.0 / slope),
                   misfit=float(np.sqrt(np.mean((np.log(tops) - line) ** 2))),
                   transients=tuple(int(np.asarray(series[te]).shape[2]) for te in te_s),
                   npoints=int(first.shape[0]), coils=int(first.shape[1]),
                   dwell_s=float(dwell_s), f0_mhz=float(f0_mhz), header=dict(header or {}))

    @classmethod
    def from_nifti(cls, sources):
        """From {echo time in seconds: a NIfTI-MRS water reference}, the transients apart."""
        series, dwells, f0s, headers = {}, set(), set(), {}
        for te, source in sources.items():
            series[te], dwell_s, f0_mhz, header = _raw(source)
            dwells.add(round(dwell_s, 12))
            f0s.add(round(f0_mhz, 6))
            headers = headers or header
        if len(dwells) > 1 or len(f0s) > 1:
            raise ValueError("the references are sampled at different dwells or frequencies")
        return cls.from_arrays(series, dwells.pop(), f0s.pop(), headers)

    #**************#
    #   the rule   #
    #**************#
    @property
    def failed(self):
        """Why the series is not one scale, by the stated rule; empty when it is."""
        low, high = self.PLAUSIBLE_S
        if not low <= self.apparent_t2_s <= high:
            return (f"the water falls as T2 {1e3 * self.apparent_t2_s:.0f} ms, outside a brain "
                    f"water's {1e3 * low:.0f} to {1e3 * high:.0f} ms: a scale between the scans, "
                    "not relaxation")
        if self.misfit > self.MISFIT:
            return (f"the echo tops do not fall as one exponential (rms misfit {self.misfit:.3f} "
                    f"of their logarithms, past {self.MISFIT}): a scale between the scans")
        return ""

    #**************#
    #   the twin   #
    #**************#
    def tissue_t2_s(self, voxels, seeds=3):
        """The tissue water's T2 as a Tissues record: {echo time: scan.Voxel} of the twin
        at the data's voxel with the water unsuppressed, one design per echo time; per
        draw, the factor on the drawn grey and white T2 at which the twin's echo tops
        fall as the data's (least squares of the logarithms), the median over `seeds`."""
        if tuple(sorted(voxels)) != self.te_s:
            raise ValueError(f"the twin's echo times {sorted(voxels)} are not the series' "
                             f"{self.te_s}")
        data = np.log(self.tops)
        found = []
        for seed in range(seeds):
            truth = voxels[self.te_s[0]].sample(seed)
            drawn = truth.water.t2_s
            if not isinstance(drawn, Tissues):
                raise ValueError("the twin's water T2 is not per tissue")

            def tops(factor):
                dict.__setitem__(truth.water, "t2_s", Tissues(gm=factor * drawn.gm,
                                                              wm=factor * drawn.wm, csf=drawn.csf))
                return np.array([_echo_top(voxel._water_parts(
                    truth, voxel.scan.engine, voxel.design(truth), voxel._f0_mhz(),
                    voxel._diffusion_factor(truth)), self.dwell_s)
                    for te, voxel in sorted(voxels.items())])

            def misfit(factor):
                twin = np.log(tops(factor))
                return float(np.sum((twin[1:] - twin[0] - data[1:] + data[0]) ** 2))

            factor = scipy.optimize.minimize_scalar(misfit, bounds=(0.3, 3.0), method="bounded").x
            found.append((factor * drawn.gm, factor * drawn.wm, drawn.csf))
        return Tissues(*(float(v) for v in np.median(found, axis=0)))

    #*************#
    #   records   #
    #*************#
    def records(self, voxels, dataset, presets=(), citation=None, url=None, seeds=3):
        """The Calibrations of water.t2_s.gm and water.t2_s.wm, the twin's reading of this
        series, the prior a quarter either side; carried by the subject; failed by the
        series' own rule."""
        t2 = self.tissue_t2_s(voxels, seeds)
        voxel = voxels[self.te_s[0]]
        designs = [type(v.scan.sequence).__name__ for _, v in sorted(voxels.items())]
        return tuple(Calibration(
            leaf=f"water.t2_s.{tissue}", value=float(round(value, 4)),
            prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
            measured={"statistic": "the tissue water's T2 at which the twin's water references "
                                   "fall over the echo time as the data's, the drawn grey and "
                                   "white T2 scaled together (scan.calibrate.WaterSeriesReading)",
                      "tops": tuple(round(top, 4) for top in self.tops),
                      "apparent_t2_ms": round(1e3 * self.apparent_t2_s, 1),
                      "misfit": round(self.misfit, 4),
                      "tissue_t2_ms": {"gm": round(1e3 * t2.gm, 1), "wm": round(1e3 * t2.wm, 1)}},
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                     "model": self.header.get("ManufacturersModelName"),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or "/".join(designs),
                      "family": "/".join(designs), "te_s": self.te_s,
                      "tr_s": self.header.get("RepetitionTime"), "simulated_as": "/".join(designs)},
            acquisition={"npoints": self.npoints, "bandwidth_hz": 1.0 / self.dwell_s,
                         "transients": self.transients, "water_suppressed": False},
            voxel={"size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm),
                   "fractions": tuple(round(float(f), 3) for f in voxel.fractions)},
            processing="the water references' echo tops (the mean magnitude of the combined mean "
                       "FID's points 2 to 8) over the first, the twin's water played through the "
                       "same designs at the voxel's own tissue fractions",
            date=datetime.date.today().strftime("%d/%m/%y"), carried_by="subject",
            presets=tuple(presets), failed=self.failed,
            note="one subject's series; made by scan.calibrate")
            for tissue, value in (("gm", t2.gm), ("wm", t2.wm)))


#**************************************************************************************************#
#                                 Class MetaboliteSeriesReading                                    #
#**************************************************************************************************#
#                                                                                                  #
# A metabolite's T2 read off its area over a series of echo times, through the twin.               #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class MetaboliteSeriesReading:
    """A metabolite's T2 read off its area in water-suppressed runs over the echo time,
    beside their water references, through the twin.

    Each echo's suppressed run gives the line's area (the NAA-phased real spectrum over
    the 0.3 to 0.5 ppm level, within the line's band) over its water reference's echo top
    (WaterSeriesReading's statistic, so each echo stands at its own export's scale), and
    the ratios fall over the echo time. What they fall with is not the T2 alone: what lies
    under the line and goes at its own rate (the macromolecules at a short echo, the
    coupled neighbours), the pulses' relaxation and the water's own decay all lie in them,
    and all of them in the twin. So the T2 is the value at which the twin's ratios, the
    same statistic at the data's voxel through the same designs, fall as the data's
    (`t2_s`): the metabolite's own lines swapped at the trial T2 (its systems alone
    replayed through each design, the rest of the twin's spectrum held), the factor on the
    drawn T2 fitted on the logarithms, the median over draws. Two lines: NAA's methyl, and
    creatine's at 3.03 ppm, which is creatine's and phosphocreatine's (the two scaled
    together). The reading refuses where the references are not one scale (the water
    series' own rule) or no T2 within BOUNDS of the drawn makes the twin fall as the data.

    Args:
        te_s: The echo times in seconds, ascending.
        ratios: {line: the area over the water top at each echo, over the first}.
        water: The references' WaterSeriesReading.
    """

    LINES = {"NAA": (("NAA",), (1.93, 2.09)), "Cr": (("Cr", "PCr"), (2.96, 3.10))}
    BOUNDS = (0.3, 3.0)

    te_s: tuple
    ratios: dict
    water: WaterSeriesReading
    transients: tuple
    npoints: int
    coils: int
    dwell_s: float
    f0_mhz: float
    header: dict

    #*******************#
    #   from a series   #
    #*******************#
    @classmethod
    def from_arrays(cls, series, references, dwell_s, f0_mhz, header=None):
        """Read {echo time in seconds: (npoints, coils, transients)} suppressed runs of one
        voxel beside their water references, {echo time: (npoints, coils, transients)}."""
        water = WaterSeriesReading.from_arrays(references, dwell_s, f0_mhz, header)
        if tuple(sorted(series)) != water.te_s:
            raise ValueError("the runs and the references are not at the same echo times")
        # one whitening and one set of weights for the runs, the first run's, as the references
        first = np.asarray(series[water.te_s[0]])
        areas = {te: cls._areas(_registered(_combined(series[te], like=first), dwell_s,
                                            f0_mhz).mean(axis=1), dwell_s, f0_mhz)
                 for te in water.te_s}
        ratios = {}
        for name in cls.LINES:
            over = np.array([areas[te][name] for te in water.te_s]) / np.array(water.tops)
            ratios[name] = tuple(float(r) for r in over / over[0])
        return cls(te_s=water.te_s, ratios=ratios, water=water,
                   transients=tuple(int(np.asarray(series[te]).shape[2]) for te in water.te_s),
                   npoints=int(first.shape[0]), coils=int(first.shape[1]), dwell_s=float(dwell_s),
                   f0_mhz=float(f0_mhz), header=dict(header or {}))

    @classmethod
    def from_nifti(cls, sources, references):
        """From {echo time in seconds: a NIfTI-MRS suppressed run} and {echo time: its water
        reference}, the transients apart."""
        series, water, dwells, f0s, header = {}, {}, set(), set(), {}
        for te in sources:
            series[te], dwell_s, f0_mhz, header = _raw(sources[te])
            water[te], _, _, _ = _raw(references[te])
            dwells.add(round(dwell_s, 12))
            f0s.add(round(f0_mhz, 6))
        if len(dwells) > 1 or len(f0s) > 1:
            raise ValueError("the runs are sampled at different dwells or frequencies")
        return cls.from_arrays(series, water, dwells.pop(), f0s.pop(), header)

    @classmethod
    def _areas(cls, fid, dwell_s, f0_mhz):
        """{line: its band's area over the 0.3 to 0.5 ppm level} of one FID whose carrier
        sits on the water."""
        ppm, real = _spectrum(fid, dwell_s, lambda hz: WATER_PPM + hz / f0_mhz)
        level = np.median(real[(ppm > 0.3) & (ppm < 0.5)])
        step = ppm[1] - ppm[0]
        return {name: float((real[(ppm > low) & (ppm < high)] - level).sum() * step)
                for name, (_, (low, high)) in cls.LINES.items()}

    #**************#
    #   the twin   #
    #**************#
    def factors(self, voxels, seeds=3):
        """Per draw the factor on each line's drawn T2 at which the twin's ratios fall as
        the data's: {echo time: scan.Voxel} of the suppressed twin at the data's voxel, one
        design per echo time; returns ({line: [factor per seed]}, {leaf: [drawn T2 per
        seed]})."""
        if tuple(sorted(voxels)) != self.te_s:
            raise ValueError(f"the twin's echo times {sorted(voxels)} are not the series' "
                             f"{self.te_s}")
        from mrsandbox.phantom.water import Unsuppressed

        ordered = [voxels[te] for te in self.te_s]
        n, dwell_s = self.npoints, self.dwell_s
        turn = np.exp(-2j * np.pi * WATER_PPM * ordered[0]._f0_mhz() * np.arange(n) * dwell_s)
        found = {name: [] for name in self.LINES}
        drawn = {leaf: [] for leaves, _ in self.LINES.values() for leaf in leaves}
        for seed in range(seeds):
            truth = ordered[0].sample(seed)
            # the twin's full water through each design, as a reference would show it
            tops = np.array([_echo_top(voxel._water_parts(
                truth, voxel.scan.engine, voxel.design(truth), voxel._f0_mhz(),
                voxel._diffusion_factor(truth), source=Unsuppressed()), dwell_s)
                for voxel in ordered])
            for name, (leaves, _) in self.LINES.items():
                own = {leaf: float(ordered[0].resolve(truth.t2[leaf])) for leaf in leaves}

                def misfit(factor):
                    areas = [self._areas(self._swapped(voxel, truth, own, factor, n, dwell_s)
                                         * turn, dwell_s, voxel._f0_mhz())[name]
                             for voxel in ordered]
                    twin = np.log(np.array(areas) / tops)
                    return float(np.sum((twin[1:] - twin[0] - np.log(self.ratios[name][1:])) ** 2))

                factor = scipy.optimize.minimize_scalar(misfit, bounds=self.BOUNDS,
                                                        method="bounded").x
                for leaf, value in own.items():
                    dict.__setitem__(truth.t2, leaf, value)
                found[name].append(float(factor))
                for leaf, value in own.items():
                    drawn[leaf].append(value)
        return found, drawn

    @staticmethod
    def _swapped(voxel, truth, own, factor, npoints, dwell_s):
        """The twin's clean FID on the grid with the lines of the metabolites in `own`
        ({leaf: drawn T2}) swapped at `factor` times their drawn T2: their systems alone
        replayed through the design, the other metabolites' lines as the twin holds them
        (left in the voxel's lines cache under the trial frame)."""
        # the twin's lines at the drawn frame (held), then the trial on the truth
        for leaf, value in own.items():
            dict.__setitem__(truth.t2, leaf, value)
        lines = dict(voxel._metabolite_lines(truth))
        for leaf, value in own.items():
            dict.__setitem__(truth.t2, leaf, factor * value)
        engine, design = voxel.scan.engine, voxel.design(truth)
        temperature_c = float(voxel.resolve(truth.water.temperature_c))
        for leaf, value in own.items():
            group = voxel.scan.systems(leaf, temperature_c)
            freqs, amps = engine.lines(
                design, *(s.with_relaxation(t1_s=float(voxel.resolve(truth.t1[leaf])),
                                            t2_s=factor * value) for s in group),
                b1_scale=voxel.b1_input(truth), offset_hz=voxel.offset_hz(truth),
                shift_mm=voxel.shift_mm)
            lines[leaf] = (np.asarray(ops.convert_to_numpy(freqs)),
                           np.asarray(ops.convert_to_numpy(amps)))
        key = (voxel.compiled, temperature_c) + voxel.frame(truth) + voxel._design_key(truth)
        voxel._lines_cache(truth)[key] = lines
        _, fid = voxel._clean_fid(truth, npoints, dwell_s)
        return np.asarray(fid)

    def t2_s(self, voxels, seeds=3):
        """{leaf: T2 in seconds} of t2.NAA, t2.Cr and t2.PCr: per draw the drawn T2 times
        the fitted factor, the median over `seeds`."""
        factors, drawn = self.factors(voxels, seeds)
        return {leaf: float(np.median([f * d for f, d in zip(factors[name], drawn[leaf])]))
                for name, (leaves, _) in self.LINES.items() for leaf in leaves}

    #*************#
    #   records   #
    #*************#
    def records(self, voxels, dataset, presets=(), citation=None, url=None, seeds=3):
        """The Calibrations of t2.NAA, t2.Cr and t2.PCr, the twin's reading of this series,
        the prior a quarter either side; carried by the subject; failed by the stated
        rules."""
        factors, drawn = self.factors(voxels, seeds)
        voxel = voxels[self.te_s[0]]
        designs = [type(v.scan.sequence).__name__ for _, v in sorted(voxels.items())]
        low, high = self.BOUNDS
        at_bound = [name for name, values in factors.items()
                    if not low * 1.01 < float(np.median(values)) < high / 1.01]
        failed = self.water.failed or ("" if not at_bound else (
            f"no T2 between {low} and {high} of the drawn makes the twin's "
            f"{' and '.join(at_bound)} fall as the data's: not relaxation"))
        found = []
        for name, (leaves, _) in self.LINES.items():
            for leaf in leaves:
                value = float(np.median([f * d for f, d in zip(factors[name], drawn[leaf])]))
                found.append(Calibration(
                    leaf=f"t2.{leaf}", value=float(round(value, 4)),
                    prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
                    measured={"statistic": f"the {name} area over the water reference's echo "
                                           "top at each echo, over the first; the T2 at which "
                                           "the twin's ratios fall as the data's, its own lines "
                                           "swapped (scan.calibrate.MetaboliteSeriesReading)",
                              "ratios": tuple(round(r, 4) for r in self.ratios[name]),
                              "factor": round(float(np.median(factors[name])), 3),
                              "t2_ms": round(1e3 * value, 1),
                              "water_apparent_t2_ms": round(1e3 * self.water.apparent_t2_s, 1)},
                    source={"dataset": dataset, "citation": citation, "url": url},
                    scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                             "model": self.header.get("ManufacturersModelName"),
                             "software": self.header.get("SoftwareVersions"),
                             "field_t": round(self.f0_mhz / 42.5774785, 2)},
                    coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
                    sequence={"name": self.header.get("SequenceName") or "/".join(designs),
                              "family": "/".join(designs), "te_s": self.te_s,
                              "tr_s": self.header.get("RepetitionTime"),
                              "simulated_as": "/".join(designs)},
                    acquisition={"npoints": self.npoints, "bandwidth_hz": 1.0 / self.dwell_s,
                                 "transients": self.transients, "water_suppressed": True},
                    voxel={"size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                           "simulated_position_mm": tuple(voxel.position_mm),
                           "fractions": tuple(round(float(f), 3) for f in voxel.fractions)},
                    processing="the combined mean FID's spectrum phased on NAA, 2 Hz apodization, "
                               "fourfold zero filled, the band's area over the 0.3 to 0.5 ppm "
                               "level, over the reference's echo top; the twin's clean spectrum "
                               "read the same way through the same designs at the voxel's own "
                               "tissue fractions",
                    date=datetime.date.today().strftime("%d/%m/%y"), carried_by="subject",
                    presets=tuple(presets), failed=failed,
                    note="one subject's series; made by scan.calibrate"))
        return tuple(found)


#**************************************************************************************************#
#                                   Class MacromoleculeReading                                     #
#**************************************************************************************************#
#                                                                                                  #
# The macromolecules' level read off the 0.9 ppm peak of a short-echo run, through the twin.       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class MacromoleculeReading:
    """The macromolecules' level read off the 0.9 ppm peak of a short-echo run, through
    the twin.

    The 0.9 ppm peak of a water-suppressed short-echo spectrum is the macromolecules'
    methyls (and the lipids' where the voxel reaches fat). Its height over the level at
    0.3 to 0.5 ppm, over NAA's over the same level, is the statistic: no fit, no model
    of what lies beneath. The twin's own reading of its clean spectrum at the same
    voxel through the same design carries everything else the peak holds (the sequence's
    echo decay at the macromolecules' T2s, the neighbours, the baseline), so the level
    is the scan's macromolecule leaf scaled by the data's reading over the twin's
    (`record`), the leaf being the one the scan's source scales with (MeasuredMM's
    level_mm, CoupledMM's kappa). The reading refuses where the twin's lipids make more
    than LIPID_SHARE of its peak (the voxel's edge in fat) or the echo time has taken the
    peak down (past LATEST_TE_S).

    Args:
        peak_over_naa: The 0.9 ppm peak's height over NAA's, both over the 0.3 to 0.5 ppm
            level.
    """

    LIPID_SHARE = 0.2
    LATEST_TE_S = 0.05

    peak_over_naa: float
    transients: int
    npoints: int
    coils: int
    dwell_s: float
    f0_mhz: float
    header: dict

    #*******************#
    #   from arrays     #
    #*******************#
    @classmethod
    def from_arrays(cls, data, dwell_s, f0_mhz, header=None):
        """`data` (npoints, coils, transients) of the suppressed run, in the standard's
        frame (the carrier at 4.65 ppm)."""
        fid = _combined(data).mean(axis=1)
        return cls(peak_over_naa=cls._peak_over_naa(fid, dwell_s, f0_mhz),
                   transients=int(np.asarray(data).shape[2]), npoints=int(fid.size),
                   coils=int(np.asarray(data).shape[1]), dwell_s=float(dwell_s),
                   f0_mhz=float(f0_mhz), header=dict(header or {}))

    @classmethod
    def from_nifti(cls, source):
        return cls.from_arrays(*_raw(source))

    @staticmethod
    def _peak_over_naa(fid, dwell_s, f0_mhz):
        ppm, real = _spectrum(fid, dwell_s, lambda hz: WATER_PPM + hz / f0_mhz)
        level = np.median(real[(ppm > 0.3) & (ppm < 0.5)])
        return float((real[(ppm > 0.85) & (ppm < 0.97)].max() - level)
                     / (real[np.abs(ppm - NAA_PPM) < 0.04].max() - level))

    #**************#
    #   the twin   #
    #**************#
    def twin(self, voxel, seeds=3):
        """The twin's own reading at `voxel`: per draw the clean spectrum's peak over NAA
        and the lipids' share of the peak (what the peak loses without them), as
        {"peak_over_naa": [per seed], "lipid_share": [per seed]}."""
        t = np.arange(self.npoints) * self.dwell_s
        turn = np.exp(-2j * np.pi * WATER_PPM * voxel._f0_mhz() * t)
        found = {"peak_over_naa": [], "lipid_share": []}
        for seed in range(seeds):
            parts = voxel._component_fids(voxel.sample(seed), self.npoints, self.dwell_s)
            total = sum(np.asarray(fid) for fid in parts.values()) * turn
            peak = self._peak_over_naa(total, self.dwell_s, voxel._f0_mhz())
            lean = total - np.asarray(parts.get("lipids", 0.0)) * turn
            found["peak_over_naa"].append(peak)
            found["lipid_share"].append(
                1.0 - self._peak_over_naa(lean, self.dwell_s, voxel._f0_mhz()) / peak)
        return found

    #************#
    #   record   #
    #************#
    def record(self, voxel, dataset, presets=(), citation=None, url=None, seeds=3):
        """The Calibration of the scan's macromolecule level leaf (mm.level_mm or mm.kappa):
        per draw the drawn leaf times the data's reading over the twin's, the median; the
        prior a quarter either side; carried by the subject; failed by the stated rules."""
        leaf = voxel.scan.mm.LEVEL_LEAF
        if leaf is None:
            raise ValueError(f"{type(voxel.scan.mm).__name__} scales with no mm leaf: nothing to record")
        twin = self.twin(voxel, seeds)
        drawn = [float(voxel.sample(seed).mm[leaf]) for seed in range(seeds)]
        values = [d * self.peak_over_naa / peak for d, peak in zip(drawn, twin["peak_over_naa"])]
        value = float(np.median(values))
        lipid, te_s = float(np.median(twin["lipid_share"])), _echo_time_s(voxel.scan.sequence)
        sequence = voxel.scan.sequence
        return Calibration(
            leaf=f"mm.{leaf}", value=float(round(value, 4)),
            prior=(round(0.75 * value, 4), round(1.25 * value, 4)),
            measured={"statistic": "the 0.9 ppm peak over NAA, both over the 0.3 to 0.5 ppm level, "
                                   "the data's over the twin's (scan.calibrate.MacromoleculeReading)",
                      "peak_over_naa": round(self.peak_over_naa, 4),
                      "twin_peak_over_naa": round(float(np.median(twin["peak_over_naa"])), 4),
                      "factor": round(self.peak_over_naa
                                      / float(np.median(twin["peak_over_naa"])), 3),
                      "twin_lipid_share": round(lipid, 3)},
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get("Manufacturer", "")).lower() or None,
                     "model": self.header.get("ManufacturersModelName"),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or type(sequence).__name__,
                      "family": type(sequence).__name__, "te_s": self.header.get("EchoTime", te_s),
                      "tr_s": self.header.get("RepetitionTime"),
                      "simulated_as": type(sequence).__name__},
            acquisition={"npoints": self.npoints, "bandwidth_hz": 1.0 / self.dwell_s,
                         "transients": self.transients, "water_suppressed": True},
            voxel={"size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm),
                   "fractions": tuple(round(float(f), 3) for f in voxel.fractions)},
            processing="the combined mean FID's spectrum phased on NAA, 2 Hz apodization, fourfold "
                       "zero filled; the twin's clean spectrum read the same way",
            date=datetime.date.today().strftime("%d/%m/%y"), carried_by="subject",
            presets=tuple(presets),
            failed=(f"the twin's lipids make {lipid:.2f} of the 0.9 ppm peak at this placement, past "
                    f"{self.LIPID_SHARE}: the voxel's edge in fat" if lipid > self.LIPID_SHARE
                    else f"the echo time {1e3 * te_s:.0f} ms has taken the peak down: read a short "
                         f"echo, up to {1e3 * self.LATEST_TE_S:.0f} ms" if te_s > self.LATEST_TE_S
                    else ""),
            note="one acquisition; made by scan.calibrate")


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

        image = (nib.load(str(source))
                 if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source)
        extension = json.loads(image.header.extensions[0].get_content())
        tags = [extension.get(f"dim_{k}") for k in (5, 6, 7)]
        if "DIM_COIL" not in tags:
            raise ValueError("the channels are needed apart (DIM_COIL)")
        data = np.moveaxis(np.asarray(image.dataobj)[0, 0, 0], 1 + tags.index("DIM_COIL"), -1)
        start = int(round(data.shape[0] * (1.0 - tail)))
        header = {key: extension[key] for key in HEADER if extension.get(key) not in (None, "")}
        reading = cls.from_arrays(data[start:], header=header)
        # a signal that still rings there is one mode shared by every channel (an unsuppressed
        # phantom's water), far above what a loaded array's noise shares
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
                # ACQ_IS_NOISE_MEASUREMENT
                flagged = np.flatnonzero(table["head"]["flags"] & np.uint64(1 << 18))
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
        found = {tag: re.search(rf"<{tag}>(.*?)</{tag}>", xml)
                 for tag in ("systemVendor", "systemModel", "systemFieldStrength_T")}
        header = {tag: match.group(1) for tag, match in found.items() if match}
        return cls(covariance=total.conj() / count, samples=count, header=header)

    #*****************#
    #   from kspace   #
    #*****************#
    @classmethod
    def from_kspace(cls, path, edge=0.125):
        """Read the noise of a multi-coil k-space as fastMRI stores it (an h5 file: kspace as
        slices, coils, readout, phase encodes, the readout oversampled twofold; the sampled
        phase encodes in `mask`; the ISMRMRD header): each sampled line brought to image
        space along the readout alone, whose outer `edge` on either side lies beyond the head
        (twice the prescribed field of view): noise, rolled off by the receiver's filter
        alike in every channel, whatever the undersampling. Needs h5py."""
        import re

        import h5py

        with h5py.File(str(path), "r") as file:
            kspace = file["kspace"]
            mask = (file["mask"][()].astype(bool) if "mask" in file
                    else np.ones(kspace.shape[-1], dtype=bool))
            n = kspace.shape[2]
            cut = int(round(edge * n))
            samples = []
            for index in range(kspace.shape[0]):     # a slice at a time: a volume holds a gigabyte
                lines = np.fft.fftshift(
                    np.fft.ifft(np.fft.ifftshift(kspace[index][..., mask], axes=1), axis=1), axes=1)
                samples.append(np.moveaxis(np.concatenate([lines[:, :cut], lines[:, n - cut:]],
                                                          axis=1), 0, -1))
            xml = file["ismrmrd_header"][()]
        xml = xml.decode() if isinstance(xml, bytes) else str(xml)
        found = {tag: re.search(rf"<{tag}>(.*?)</{tag}>", xml)
                 for tag in ("systemVendor", "systemModel", "systemFieldStrength_T")}
        header = {tag: match.group(1).strip() for tag, match in found.items() if match}
        if "systemFieldStrength_T" in header:
            header["systemFieldStrength_T"] = float(header["systemFieldStrength_T"])
        names = [name.split(":")[0] for name in re.findall(r"<coilName>(.*?)</coilName>", xml)]
        if names:
            header["ReceiveCoilName"] = max(set(names), key=names.count)
        return cls.from_arrays(np.stack(samples), header)

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
        return {"strongest_partner_mean": float(c.max(axis=1).mean()),
                "strongest_pair": float(c.max()),
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
        """The share of a channel's noise that is its own at which `array` has this
        reading's mean strongest partner (`share`)."""
        return self.share(array, self.statistics()["strongest_partner_mean"])

    @staticmethod
    def share(array, target):
        """The share of a channel's noise that is its own at which the geometric `array` (a
        hardware.CoilArray of loops) has `target` as its channels' mean strongest partner:
        the sample's share of the resistance comes from the geometry (`CoilArray.loaded`),
        the rest is the identity. The share is the geometry's as much as the data's (another
        stand-in geometry reads it otherwise), so it is taken against the array that plays
        it."""
        sample = array.loaded(own_noise=0.0).noise_resistance

        def strongest(share):
            r = (1.0 - share) * sample + share * np.eye(len(array))
            level = np.sqrt(np.diag(r))
            return float((np.abs(r / np.outer(level, level))
                          - np.eye(len(array))).max(axis=1).mean())

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
        return self._calibration("coils.loaded.own_noise", share, dataset, load, citation, url)

    def record_correlation(self, dataset, load, citation=None, url=None):
        """The Calibration of coils.partner_correlation: the channels' mean strongest partner,
        the data's own number whatever the coil's geometry, which the console turns into
        the own share against the array on the floor (`share`)."""
        return self._calibration("coils.partner_correlation",
                                 round(self.statistics()["strongest_partner_mean"], 3), dataset,
                                 load, citation, url)

    def _calibration(self, leaf, number, dataset, load, citation, url):
        return Calibration(
            leaf=leaf, value=number, prior=(number, number),
            measured={"statistic": "magnitude of the noise correlation between channels "
                                   "(scan.calibrate.CovarianceReading.statistics)",
                      **{key: round(value, 3) for key, value in self.statistics().items()},
                      "samples": self.samples},
            source={"dataset": dataset, "citation": citation, "url": url},
            scanner={"vendor": str(self.header.get(
                         "Manufacturer", self.header.get("systemVendor", ""))).lower() or None,
                     "model": self.header.get("ManufacturersModelName",
                                              self.header.get("systemModel")),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": self.header.get("systemFieldStrength_T")},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": "noise", "family": ""},
            acquisition={"channels": self.coils, "rf": "no signal"},
            voxel={"load": load}, processing="sample covariance of the channels over all samples",
            date=datetime.date.today().strftime("%d/%m/%y"),
            note="the modelled array stands in for the coil's unpublished geometry; made by "
                 "scan.calibrate",
        )


#**************************************************************************************************#
#                                       Class ReceiverReading                                      #
#**************************************************************************************************#
#                                                                                                  #
# The receiver's colour: the noise power across the band, and the filter that leaves it.          #
#                                                                                                  #
#**************************************************************************************************#
class ReceiverReading:
    """The receiver's colour: the noise power across the spectral band, read off samples that
    hold noise only, every channel and transient apart (the last quarter of every transient,
    or a noise-only scan's vectors at the spectral band), on FRACTIONS of the band from its
    centre over its median; and the hardware.Receiver whose filter and decimation leave the
    nearest colour. The central fifth of the band is left out of the comparison: there a
    noise-only vector keeps its slow drift and a transient's tail the signal still in it.

    Attributes:
        power: The measured power on FRACTIONS, over its median.
        vectors: How many noise vectors it averages.
    """

    FRACTIONS = np.linspace(-0.5, 0.5, 512, endpoint=False)
    SPANS = range(2, 65)

    #**********#
    #   init   #
    #**********#
    def __init__(self, power, vectors):
        self.power = np.asarray(power, dtype=np.float64)
        self.vectors = int(vectors)

    #*****************#
    #   from arrays   #
    #*****************#
    @classmethod
    def from_arrays(cls, noise):
        """Read (points, ...) of noise, every column a vector of its own."""
        x = np.asarray(noise).reshape(np.shape(noise)[0], -1)
        x = x - x.mean(axis=0, keepdims=True)
        spectra = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(x.shape[0])[:, None], axis=0),
                                         axes=0)) ** 2
        power = np.interp(cls.FRACTIONS, np.linspace(-0.5, 0.5, x.shape[0], endpoint=False),
                          spectra.mean(axis=1))
        return cls(power / np.median(power), x.shape[1])

    @classmethod
    def from_nifti(cls, source):
        """Read the last quarter of every transient of a NIfTI-MRS file with DIM_DYN (a path, or
        a loaded image), every channel apart."""
        x = _raw(source)[0]
        return cls.from_arrays(x[-x.shape[0] // 4:])

    #**************#
    #   receiver   #
    #**************#
    def miss(self, power):
        """The mean absolute log ratio of the measured colour to `power` over the band's
        outer four fifths, each over its median there."""
        kept = np.abs(self.FRACTIONS) >= 0.1
        measured, model = self.power[kept], np.asarray(power, dtype=np.float64)[kept]
        return float(np.mean(np.abs(np.log((measured / np.median(measured))
                                           / (model / np.median(model))))))

    def receiver(self, oversampling=8):
        """The hardware.Receiver of the span that leaves the nearest colour, or None where no
        filter does better than a flat band (a vendor whose files keep the band's middle, or
        whose roll-off the windowed sinc's family does not hold: compare `miss`)."""
        nearest = min(self.SPANS, key=lambda span: self.miss(
            Receiver(oversampling, span).power(self.FRACTIONS)))
        found = Receiver(oversampling, nearest)
        flat = self.miss(np.ones_like(self.FRACTIONS))
        return found if self.miss(found.power(self.FRACTIONS)) < flat else None


#**************************************************************************************************#
#                                         Class GIRFReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# One axis's gradient impulse response, read off a phantom's phase under test gradients.          #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class GIRFReading:
    """One axis's gradient impulse response, read off a phantom's phase under test
    gradients.

    The thin-slice, phase-encoded measurement (Vannesjo et al. 2013, MRM 69:583, in the
    form of Bacon's GIRF_PE_Python, which the package's Prisma response came from): a
    water phantom's signal at known positions (thin slices offset along the axis, phase
    encoded in plane) over a long readout while a test gradient plays on the axis
    (triangles of many amplitudes, both polarities, one run each), and the same
    positions' signal with no gradient, the reference. Per run and position the phase
    over the reference, unwrapped, has as its derivative the frequency the field at
    that place makes, over gammabar a field in millitesla; per time point the solid
    harmonics to first order are fitted over the positions ([1, x, y, z]: the B0 term
    in mT, the three gradient terms in mT/m, each per mT/m of input; a term the positions
    do not resolve, slices along one axis alone, comes out zero); per term the response
    is the runs' spectra against the inputs' by least squares over the runs, sum of
    O conj(I) over the sum of |I|^2 plus `regularizer` of the peak input power (the
    deposit's regularized division: where the inputs carry no power the response is
    nothing, not noise; the self term is fitted as its deviation from the nominal
    waveform and stored as the full response, one plus it, as resources.GIRF takes it).
    Three axes' readings make the scanner's response (`girf`).

    Args:
        axis: The input axis, "x", "y" or "z".
        freq_hz: The frequency grid, ascending, both signs.
        terms: {"b0", "x", "y", "z": the complex response on the grid}.
    """

    TERMS = ("b0", "x", "y", "z")

    axis: str
    freq_hz: np.ndarray
    terms: dict
    runs: int
    positions: int
    dwell_s: float

    #*******************#
    #   from arrays     #
    #*******************#
    @classmethod
    def from_arrays(cls, axis, inputs_mt_m, signals, references, positions_mm, dwell_s,
                    gammabar_hz_per_mt=42576.0, regularizer=1e-3):
        """`inputs_mt_m` (runs, points): each run's test waveform on the readout's grid,
        signed; `signals` (runs, positions, points) complex, the positions' signal under
        each run; `references` (positions, points), or (repeats, positions, points) to
        average, with no gradient; `positions_mm` (positions, 3)."""
        inputs = np.asarray(inputs_mt_m, dtype=np.float64)
        signals = np.asarray(signals)
        references = np.asarray(references)
        if references.ndim == 3:
            references = references.mean(axis=0)
        if inputs.shape != (signals.shape[0], signals.shape[2]):
            raise ValueError("the inputs are not one waveform per run on the signals' grid")
        # the phase each position gains over the reference, sample to sample: the field
        # there, averaged over the dwell (undone below as the dwell's sinc and half-sample
        # delay on the spectra, exact for the average)
        phase = np.unwrap(np.angle(signals * np.conj(references)[None]), axis=-1)
        field = np.diff(phase, axis=-1, prepend=phase[..., :1]) / (2.0 * np.pi * gammabar_hz_per_mt
                                                                   * dwell_s)
        basis = np.column_stack([np.ones(signals.shape[1]),
                                 np.asarray(positions_mm, dtype=np.float64) / 1e3])
        coefficients = np.einsum("mp,kpt->kmt", np.linalg.pinv(basis), field)
        f = np.fft.fftfreq(inputs.shape[1], dwell_s)
        spectra_in = np.fft.fft(inputs, axis=-1)
        spectra_out = np.fft.fft(coefficients, axis=-1) / (np.sinc(f * dwell_s)
                                                            * np.exp(-1j * np.pi * f * dwell_s))
        # the self term as its deviation from the nominal waveform (the regularizer then
        # weighs on the deviation, not on the one it rides on), the one put back after
        spectra_out[:, 1 + "xyz".index(axis)] -= spectra_in
        power = np.abs(spectra_in) ** 2
        denominator = power.sum(axis=0) + regularizer * power.max()
        order = np.fft.fftshift(np.arange(inputs.shape[1]))
        terms = {term: ((spectra_out[:, m] * np.conj(spectra_in)).sum(axis=0) / denominator)[order]
                 for m, term in enumerate(cls.TERMS)}
        terms[axis] = terms[axis] + 1.0
        return cls(axis=str(axis), freq_hz=np.fft.fftshift(np.fft.fftfreq(inputs.shape[1], dwell_s)),
                   terms=terms, runs=int(inputs.shape[0]), positions=int(signals.shape[1]),
                   dwell_s=float(dwell_s))

    #**************#
    #   the girf   #
    #**************#
    @staticmethod
    def girf(readings, name):
        """The scanner's response (resources.GIRF) from the three axes' readings, {axis:
        GIRFReading}, named `name`; GIRF.save keeps it for a profile."""
        from mrsandbox.resources.girf import GIRF

        if set(readings) != set("xyz"):
            raise ValueError(f"the three axes are needed, {sorted(readings)} given")
        return GIRF({axis: {term: (reading.freq_hz, response)
                            for term, response in reading.terms.items()}
                     for axis, reading in readings.items()}, name=name)


#**************************************************************************************************#
#                                        Class WaterReading                                        #
#**************************************************************************************************#
#                                                                                                  #
# The residual water and the receiver delay of one suppressed acquisition against its reference.   #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class WaterReading:
    """What one suppressed acquisition and its water reference say of the scanner's
    water suppression and receiver: the residual water as the exponentials within 0.4 ppm
    of the carrier in the mean suppressed FID at t = 0 (`_exponentials`) over the
    reference's mean FID at t = 0, both combined with the suppressed file's own weights;
    the receiver delay as the phase of creatine against NAA over their distance, a
    first-order phase, read on creatine alone, BEYOND the phase the sequence itself
    gives creatine's peak (the coupled neighbours under 3.03 ppm evolve over the echo;
    the twin says how much, `intrinsic_phase_deg`); and the residual's phase beyond the
    line through NAA and creatine, whose side (past a quarter turn: inverted) signs it.

    Attributes:
        residual_over_full: The residual water at t = 0 over the reference's, positive.
        residual_over_dry_max: The same over the maximum of the FID without its water.
        water_phase_deg: The residual's phase against the metabolites, the line taken out.
        creatine_phase_deg: Creatine's phase against NAA as measured.
        creatine_offset_hz: Creatine's distance from NAA, the ppm way round, in hertz.
        dry_amplitude: The FID without its water at t = 0, in the data's units, the twin's
            scale.
        residual_fid: The residual water as the data show it, its fitted components narrow and
            broad, turned so its line has `water_phase_deg` against the metabolites' frame.
        water_t2star_s: The voxel's own water's decay: the reference's parts fitted the same
            way, their amplitude-weighted T2*.
        leak_over_full: The residual's broad part (decaying faster than LEAK_BELOW of the
            voxel's own water, the water outside the voxel the localization leaks) at t = 0
            over the reference's; inside `residual_over_full`, which is the total.
        leak_t2star_s, leak_offset_hz: The heaviest broad part's decay and its frequency
            from the carrier.
        leak_phase_deg: The broad parts' phase against the narrow ones'.
        eddy_transient_rad: The reference's phase 5 ms after its first sample against its
            components fitted late.
    """

    residual_over_full: float
    residual_over_dry_max: float
    water_phase_deg: float
    creatine_phase_deg: float
    creatine_offset_hz: float
    dry_amplitude: float
    residual_fid: np.ndarray
    water_t2star_s: float
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

    # the water the localization leaks decays faster than the voxel's own: a part of the
    # residual below this share of the reference's T2* is leak (the voxel's own water spreads
    # to about there across the voxel's shim)
    LEAK_BELOW = 0.7

    #*******************#
    #   from arrays     #
    #*******************#
    @classmethod
    def from_arrays(cls, data, reference, dwell_s, f0_mhz, header=None):
        """`data` and `reference` (npoints, coils, transients) of the same voxel, the
        suppressed run and its unsuppressed reference, in the standard's frame (the
        carrier at 4.65 ppm)."""
        suppressed = _combined(data).mean(axis=1)
        reference_fid = _combined(reference, like=data).mean(axis=1)
        full = float(np.abs(reference_fid[0]))
        # the residual water: the decaying exponentials within 0.4 ppm of the carrier, at t = 0
        m = min(1024, suppressed.size)
        freqs, damping, amplitudes = _exponentials(suppressed[:m], dwell_s)
        # within 0.4 ppm of the carrier, as the records
        water = (np.abs(freqs) < 0.4 * f0_mhz) & (damping < 0.0)
        residual = complex(amplitudes[water].sum())
        # the voxel's own water: the reference's parts the same way, their weighted T2*
        own_freqs, own_damping, own_amplitudes = _exponentials(reference_fid[:m], dwell_s)
        own = (np.abs(own_freqs) < 0.4 * f0_mhz) & (own_damping < 0.0)
        weights = np.abs(own_amplitudes[own])
        own_t2star = float((weights / -own_damping[own]).sum() / weights.sum())
        # the broad parts, the water the localization leaks; the narrow ones the voxel's own
        # residual
        t2star = -1.0 / damping[water]
        broad = t2star < cls.LEAK_BELOW * own_t2star
        leak, narrow = (complex(amplitudes[water][broad].sum()),
                        complex(amplitudes[water][~broad].sum()))
        heaviest = int(np.argmax(np.abs(amplitudes[water][broad]))) if broad.any() else None
        t = np.arange(suppressed.size) * dwell_s
        fitted = sum(a * np.exp((1j * 2.0 * np.pi * f + d) * t)
                     for f, d, a in zip(freqs[water], damping[water], amplitudes[water]))
        dry = suppressed - fitted
        # the first-order phase: the complex peaks of the mean spectrum, NAA phased to zero
        lines = _lines(suppressed, dwell_s, f0_mhz, WATER_PPM)
        (naa, f_naa), (creatine, f_cr), (residual_line, f_water) = (lines["NAA"], lines["Cr"],
                                                                    lines["water"])
        phase_cr = float(np.angle(creatine / naa))
        slope = phase_cr / (f_cr - f_naa)                   # radians per hertz on creatine alone
        beyond = float(np.degrees(np.angle(np.exp(
            1j * (np.angle(residual_line / naa) - slope * (f_water - f_naa))))))
        return cls(residual_over_full=abs(residual) / full,
                   residual_over_dry_max=abs(residual) / float(np.abs(dry).max()),
                   water_phase_deg=beyond, creatine_phase_deg=float(np.degrees(phase_cr)),
                   creatine_offset_hz=float(f_cr - f_naa),
                   dry_amplitude=float(np.abs(dry[0])),
                   # in the engine's convention, as `_lines` reads the phases (a file stored as
                   # the conjugate: the FID conjugated), turned so its line has `beyond` against
                   # the metabolites' frame
                   residual_fid=(fitted if _axis(suppressed, dwell_s, f0_mhz, WATER_PPM)[0] > 0
                                 else np.conj(fitted))
                   * np.exp(1j * (np.radians(beyond) - np.angle(residual_line))),
                   water_t2star_s=own_t2star,
                   leak_over_full=abs(leak) / full,
                   leak_t2star_s=float(t2star[broad][heaviest]) if heaviest is not None else 0.0,
                   leak_offset_hz=(float(freqs[water][broad][heaviest]) if heaviest is not None
                                   else 0.0),
                   leak_phase_deg=(float(np.degrees(np.angle(leak / narrow)))
                                   if heaviest is not None and narrow != 0.0 else 0.0),
                   eddy_transient_rad=_eddy_transient(reference_fid, dwell_s, f0_mhz),
                   transients=int(np.asarray(data).shape[2]),
                   reference_transients=int(np.asarray(reference).shape[2]),
                   npoints=int(suppressed.size), coils=int(np.asarray(data).shape[1]),
                   dwell_s=float(dwell_s), f0_mhz=float(f0_mhz),
                   header=dict(header or {}))

    @classmethod
    def from_nifti(cls, source, reference=None):
        """From two NIfTI-MRS files with the transients apart: the suppressed run and its
        reference, or with no reference the water transients the run interleaves with its
        own (`_raw`)."""
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
                     "model": self.header.get("ManufacturersModelName"),
                     "software": self.header.get("SoftwareVersions"),
                     "field_t": round(self.f0_mhz / 42.5774785, 2)},
            coil={"channels": (self.coils,), "name": self.header.get("ReceiveCoilName")},
            sequence={"name": self.header.get("SequenceName") or type(sequence).__name__,
                      "family": type(sequence).__name__,
                      "te_s": self.header.get("EchoTime"),
                      "tr_s": self.header.get("RepetitionTime"),
                      "simulated_as": type(sequence).__name__},
            acquisition={"npoints": self.npoints, "bandwidth_hz": 1.0 / self.dwell_s,
                         "transients": self.transients,
                         "reference_transients": self.reference_transients,
                         "water_suppressed": True},
            voxel={**({"load": voxel.scan.brain.name}
                      if isinstance(voxel.scan.brain, Phantom) else {}),
                   "size_mm": voxel.size_mm, "simulated_brain": voxel.scan.brain.name,
                   "simulated_position_mm": tuple(voxel.position_mm)},
            date=datetime.date.today().strftime("%d/%m/%y"),
            note="one acquisition; made by scan.calibrate")

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
        must take out."""
        t = np.arange(self.npoints) * self.dwell_s
        # the twin into the standard's frame
        turn = np.exp(-2j * np.pi * WATER_PPM * voxel._f0_mhz() * t)
        phases = []
        for seed in range(seeds):
            truth = voxel.sample(seed)
            dict.__setitem__(truth.acquisition, "delay_s", 0.0)
            if gauss_hz is not None:
                dict.__setitem__(truth.lineshape, "gauss_hz", float(gauss_hz))
            parts = voxel._component_fids(truth, self.npoints, self.dwell_s)
            dry = sum(np.asarray(fid) for name, fid in parts.items()
                      if name not in ("water", "oov")) * turn
            lines = _lines(dry + (abs(dry[0]) / self.dry_amplitude) * self.residual_fid,
                           self.dwell_s, voxel._f0_mhz(), WATER_PPM)
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
            measured={"statistic": "residual water at t = 0 over the water reference's, signed "
                                   "by its phase against the metabolites "
                                   "(scan.calibrate.WaterReading)",
                      "residual_over_full": round(self.residual_over_full, 6),
                      "over_dry_maximum": round(self.residual_over_dry_max, 3),
                      "phase_deg": round(self.water_phase_deg, 1),
                      "inverted": abs(self.water_phase_deg) > 90.0},
            processing="NIfTI-MRS, the suppressed file's noise-whitened SVD coil combination on "
                       "both files, the exponentials within 0.4 ppm of the carrier at t = 0 over "
                       "the reference FID at t = 0",
            presets=tuple(presets), **self._conditions(voxel, dataset, citation, url))

    def record_leak(self, voxel, dataset, presets=(), citation=None, url=None):
        """The Calibration of oov.fraction: the residual's broad part over the full water, the
        prior a quarter either side."""
        value = self.leak_over_full
        return Calibration(
            leaf="oov.fraction", value=float(round(value, 6)),
            prior=(round(0.75 * value, 6), round(1.25 * value, 6)),
            measured={"statistic": "the broad exponentials of the suppressed mean FID within "
                                   "0.4 ppm of the carrier (decaying faster than 0.7 of the "
                                   "reference's own T2*) at t = 0 over the reference's full "
                                   "water (scan.calibrate.WaterReading)",
                      "over_full": round(value, 6),
                      "water_t2star_ms": round(1000.0 * self.water_t2star_s, 1),
                      "t2star_ms": round(1000.0 * self.leak_t2star_s, 1),
                      "offset_hz": round(self.leak_offset_hz, 2),
                      "phase_vs_residual_deg": round(self.leak_phase_deg, 1)},
            processing="NIfTI-MRS, the suppressed file's noise-whitened SVD coil combination on "
                       "both files, the exponentials within 0.4 ppm of the carrier split at 0.7 "
                       "of the reference's amplitude-weighted T2*",
            presets=tuple(presets), **self._conditions(voxel, dataset, citation, url))

    def record_eddy(self, voxel, dataset, presets=(), citation=None, url=None):
        """The Calibration of acquisition.eddy_transient_rad: the reference's phase 5 ms after
        its first sample against its components fitted late, the prior a quarter either
        side of it (one subject's own; a population's record is a spread about zero); no
        fallback, this scanner's."""
        value = self.eddy_transient_rad
        return Calibration(
            leaf="acquisition.eddy_transient_rad", value=float(round(value, 4)),
            prior=tuple(sorted((round(0.75 * value, 4), round(1.25 * value, 4)))),
            measured={"statistic": "the water reference's phase 5 ms after its first sample "
                                   "against its components fitted on 60 to 250 ms and carried "
                                   "back (scan.calibrate.WaterReading)",
                      "at_5ms_rad": round(value, 4)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the reference's mean "
                       "FID; Hankel SVD on the late window, the model carried back to the start",
            # the rules: a phase past a quarter turn 5 ms in is the late model landing a half
            # turn off, not a transient; the late window is set for 3 T, and at 7 T the water
            # has decayed in it
            failed=("the late window, 60 to 250 ms, is set for 3 T: above 4 T the water has "
                    "decayed in it" if self.f0_mhz >= 4.0 * 42.5774785
                    else "" if abs(value) <= 0.5 * np.pi
                    else f"the phase 5 ms in ({value:.2f} rad) is past a quarter turn: the late "
                         "model's own wrap"),
            presets=tuple(presets), fallback=False,
            **self._conditions(voxel, dataset, citation, url))

    def record_delay(self, voxel, dataset, presets=(), citation=None, url=None, gauss_hz=None,
                     seeds=3):
        """The Calibration of acquisition.delay_s: creatine's phase against NAA beyond the
        twin's intrinsic one (the twin as broad as `gauss_hz`, drawn `seeds` times), over
        their distance, a delay of the samples against the echo top; no fallback, this
        scanner's."""
        intrinsic = self.intrinsic_phase_deg(voxel, gauss_hz, seeds)
        delay = float(np.radians(self.creatine_phase_deg - intrinsic)
                      / (2.0 * np.pi * self.creatine_offset_hz))
        return Calibration(
            leaf="acquisition.delay_s", value=float(round(delay, 7)),
            prior=tuple(sorted((round(0.75 * delay, 7), round(1.25 * delay, 7)))),
            measured={"statistic": "phase of creatine against NAA beyond the twin's intrinsic "
                                   "phase, over their frequency difference, as seconds "
                                   "(scan.calibrate.WaterReading)",
                      "delay_us": round(1e6 * delay, 1),
                      "creatine_phase_deg": round(self.creatine_phase_deg, 1),
                      "intrinsic_phase_deg": round(intrinsic, 1),
                      "creatine_offset_hz": round(self.creatine_offset_hz, 1)},
            processing="NIfTI-MRS, noise-whitened SVD coil combination, the mean FID's spectrum "
                       "fourfold zero filled, phases at the peaks' maxima",
            # the rule: past a quarter turn from the twin's, creatine's phase cannot tell a delay
            # from an inverted or misplaced line (at 3 T about 2 ms, where the phase nears its wrap)
            failed=("" if abs(self.creatine_phase_deg - intrinsic) <= 90.0 else
                    "creatine's phase lies past a quarter turn from the twin's: an inverted or "
                    "misplaced line, not a delay"),
            presets=tuple(presets), fallback=False,
            **self._conditions(voxel, dataset, citation, url))

