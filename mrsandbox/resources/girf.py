####################################################################################################
#                                             girf.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: A measured gradient impulse response function: the Bacon, Jezzard, Clarke 2026 deposit  #
#          of a 3 T Prisma's response per gradient axis and spherical-harmonic term, served as     #
#          the response the eddy-current kernel of scan.eddy convolves a design's waveforms with.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import zipfile

import numpy as np
import scipy.fft


#*******************#
#   nws deposit     #
#*******************#
# Source: Bacon, J. B., Jezzard, P., Clarke, W. T. MR spectroscopy without water suppression
#         using the gradient impulse response function. MRM 96 (2026) 530-541.
#         doi:10.1002/mrm.70383. The GIRF was measured on a 3 T Siemens Prisma with the
#         phantom-based method of github.com/jbbacon/GIRF_PE_Python (Vannesjo et al. 2013,
#         MRM 69:583, in the thin-slice, phase-encoded form).
# Data:   Zenodo 10.5281/zenodo.18262881 (jbbacon/NWS_Spectroscopy 1.0.0), BSD 3-Clause;
#         the release archive is one checksummed resource (Zenodo's md5), the six GIRF
#         members are extracted on demand.
NWS_DEPOSIT = {
    "nws/release_zip": {
        "url": "https://zenodo.org/api/records/18262881/files/"
               "jbbacon/NWS_Spectroscopy-1.0.0.zip/content",
        "md5": "cedfcf6d5c5c94ebbc35299633e5986c",
        "filename": "nws/NWS_Spectroscopy-1.0.0.zip",
    },
}

#*********************#
#   reverse deposit   #
#*********************#
# Source: Malik, S. J. reVERSE-GIRF, release 1.0 (2015), doi 10.5281/zenodo.20127, MIT licence,
#         github.com/mriphysics/reVERSE-GIRF: "example GIRF data measured on 3T and 7T Philips
#         MRI systems" (the file names say London, 29/07/14, and Zurich, 02/03/15; the models
#         are not stated), measured for VERSE pulse design with gradient imperfections.
# Data:   per file `ff`, the frequencies in Hz, and `Hw` (points, 3), the self terms of the x, y,
#         and z axes, output over input. No B0 and no cross terms, and a grid of 76 Hz at 3 T
#         (49 Hz at 7 T): the response's first 13 ms (20 ms). It carries what a gradient does
#         to itself within a pulse or just after a crusher, not the slow eddy currents that
#         move a spectrum's lines, which need the B0 term and a finer grid (the deposit above).
REVERSE_DEPOSIT = {
    "reverse/philips_3t": {
        "url": "https://github.com/mriphysics/reVERSE-GIRF/releases/download/1.0/"
               "GIRF_3T_London_20140729.mat",
        "md5": "805ed8fb0ec114e7efe878225dea5d73",
        "filename": "reverse/GIRF_3T_London_20140729.mat",
    },
    "reverse/philips_7t": {
        "url": "https://github.com/mriphysics/reVERSE-GIRF/releases/download/1.0/"
               "GIRF_7T_Zurich_20150302.mat",
        "md5": "db03dd410462f69c87c3b32d4502c2a8",
        "filename": "reverse/GIRF_7T_Zurich_20150302.mat",
    },
}

# Source: the deposit's GIRF_measurements members and the paper's Methods:
#         per input axis j a SphericalHarmonics_{J}_1.npz with GIRF_FT (4, P, 1), the terms
#         in the order B0, x, y, z (the coefficients of the field the axis induces, in mT
#         for B0 and mT/m for the first order, per mT/m of input), on freqFull (P points,
#         plus and minus 100 kHz at the 5 us dwell of the measurement); Results39993 is
#         the 5 Hz resolution of the B0 and self terms, Results19999 the 10 Hz one the
#         first-order cross terms are represented at (the paper's Methods); the phase to
#         field conversion of the deposit used its params gammabar. The response's memory
#         is the measurement's own 200 ms.
# Units:  s, Hz, Hz per mT.
GIRF_MEASUREMENT = {
    "root": "jbbacon-NWS_Spectroscopy-33cb5fb/GIRF_measurements",
    "terms": ("b0", "x", "y", "z"),
    "resolution": {"self": "Results39993", "cross": "Results19999"},
    "dwell_s": 5e-6,
    "memory_s": 0.2,
    "gammabar_hz_per_mt": 42576.0,
}


#**************************************************************************************************#
#                                           Class GIRF                                             #
#**************************************************************************************************#
#                                                                                                  #
# A scanner's measured gradient impulse response, per input axis and spherical-harmonic term.      #
#                                                                                                  #
#**************************************************************************************************#
class GIRF:
    """A scanner's measured gradient impulse response, per input axis and
    spherical-harmonic term.

    `GIRF.load()` serves the deposit's 3 T Prisma measurement; an instance is the
    response scan.eddy convolves a design's gradient waveforms with: called as
    `girf(axis, waveform_mt_m, dt_s, bandwidth_hz)` it returns the fields one axis's
    waveform induces, {"b0": Hz, "x": mT/m, "y": mT/m, "z": mT/m}, the first-order
    terms as the deviation from the nominal waveform (the design's own gradient is in
    the spin simulation already), the cross terms as they are. The prediction is
    Equation 1 of the paper, O(w) = I(w) H(w) per term, band-limited to the receiver's
    `bandwidth_hz` (a cosine edge over the top fifth of the half band, on the
    deviation H - 1 for the self term so the nominal waveform's own edges never
    enter, stated): the measured response carries noise to 100 kHz that the
    receiver never sees, and a resonance beyond the readout's band would otherwise
    alias into it. The response is linear and time-invariant (the paper's
    assumption); a zero-frequency error of the self terms is a pre-emphasis residual
    the waveform's steps carry.

    Args:
        terms: {axis: {term: (freq_hz, response)}}, the complex response of each
            term on its own frequency grid.
        name: The measurement's name, part of every kernel cache key.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, terms, name):
        self.terms = terms
        self.name = name

    #*************#
    #   members   #
    #*************#
    @classmethod
    def file(cls, resolution, axis):
        """The local path of one GIRF member of the deposit, extracted from the release
        archive on first use (the archive itself is fetched and verified on first use)."""
        from mrsandbox.resources import Resources

        member = (f"{GIRF_MEASUREMENT['root']}/{resolution}/"
                  f"SphericalHarmonics_{axis.upper()}_1.npz")
        local = Resources.cache_dir() / "nws" / member
        if not local.exists():
            with zipfile.ZipFile(Resources.path("nws/release_zip")) as archive:
                archive.extract(member, Resources.cache_dir() / "nws")
        return local

    @classmethod
    def cached(cls):
        """True when the archive or every GIRF member is in the cache."""
        from mrsandbox.resources import Resources

        root = Resources.cache_dir() / "nws" / GIRF_MEASUREMENT["root"]
        return Resources.cached("nws/release_zip") or all(
            (root / resolution / f"SphericalHarmonics_{axis}_1.npz").exists()
            for resolution in GIRF_MEASUREMENT["resolution"].values() for axis in "XYZ")

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls):
        """The deposit's 3 T Prisma response: the B0 and self terms at 5 Hz, the
        first-order cross terms at 10 Hz, as the paper used them."""
        measurement = GIRF_MEASUREMENT
        terms = {}
        for axis in "xyz":
            terms[axis] = {}
            for k, term in enumerate(measurement["terms"]):
                which = "self" if term in ("b0", axis) else "cross"
                stored = np.load(cls.file(measurement["resolution"][which], axis))
                terms[axis][term] = (stored["freqFull"], stored["GIRF_FT"][k, :, 0])
        return cls(terms, name="nws/prisma_3t")

    @classmethod
    def philips(cls, field="3t"):
        """The self terms measured on a Philips 3 T ("3t") or 7 T ("7t") system
        (REVERSE_DEPOSIT): no B0 and no cross terms, the first 13 to 20 ms."""
        import scipy.io

        from mrsandbox.resources import Resources

        name = f"reverse/philips_{field}"
        stored = scipy.io.loadmat(Resources.path(name))
        grid = stored["ff"].ravel()
        return cls({axis: {axis: (grid, stored["Hw"][:, k])} for k, axis in enumerate("xyz")},
                   name=name)

    #**********#
    #   named  #
    #**********#
    @classmethod
    def named(cls, name):
        """The measurement a saved scan.profile names."""
        return cls.load() if name == "nws/prisma_3t" else cls.philips(name.rsplit("_", 1)[1])

    #**************#
    #   response   #
    #**************#
    def __call__(self, axis, waveform_mt_m, dt_s, bandwidth_hz):
        """The fields one axis's waveform induces, {"b0": Hz, "x", "y", "z": mT/m}."""
        waveform = np.asarray(waveform_mt_m, dtype=np.float64)
        n = scipy.fft.next_fast_len(
            waveform.size + int(round(GIRF_MEASUREMENT["memory_s"] / dt_s)))
        f = np.fft.rfftfreq(n, dt_s)
        half = 0.5 * bandwidth_hz
        edge = np.clip((f - 0.8 * half) / (0.2 * half), 0.0, 1.0)
        spectrum = np.fft.rfft(waveform, n) * 0.5 * (1.0 + np.cos(np.pi * edge))
        fields = {}
        for term, (grid, response) in self.terms[axis].items():
            h = np.interp(f, grid, response.real) + 1j * np.interp(f, grid, response.imag)
            if term == axis:
                h = h - 1.0  # the deviation from the nominal, band-limited as such
            field = np.fft.irfft(spectrum * h, n)[:waveform.size]
            if term == "b0":
                field = field * GIRF_MEASUREMENT["gammabar_hz_per_mt"]
            fields[term] = field
        return fields
