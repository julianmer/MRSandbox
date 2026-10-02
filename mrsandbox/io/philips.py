####################################################################################################
#                                            philips.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Readers for the three Philips spectroscopy formats: data/list (the raw export, channels #
#          and transients apart, noise vectors), lab/raw/sin (the scanner's own raw stream), and   #
#          sdat/spar (the reconstructed, combined spectra).                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os

import numpy as np


#*****************#
#   the sibling   #
#*****************#
def _sibling(path, extension):
    """The file of the same stem with `extension`, in the case it has on disk: every
    format is a pair or a triple of files."""
    stem = os.path.splitext(os.fspath(path))[0]
    for candidate in (stem + extension.lower(), stem + extension.upper()):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{stem}{extension} not found")


#**************************************************************************************************#
#                                          Class DataList                                          #
#**************************************************************************************************#
#                                                                                                  #
# The data/list raw export: one complex vector per channel and excitation, with its list row.      #
#                                                                                                  #
#**************************************************************************************************#
class DataList:
    """The data/list raw export: one complex vector per channel and excitation, with its
    list row.

    The .list file is text: general information (`# name : value` lines) and one row per
    vector (type, the loop counters, sign, size and offset in bytes); the .data file holds
    the vectors as little-endian float32 pairs. Vector types are STD (the data), NOI (one
    noise vector per channel, RF off), REJ, PHC, FRC, and NAV. The row layout follows the
    list file's own legend and spec2nii's reader (Clarke et al. 2022).

    Attributes:
        index: The list rows as a structured array, fields `typ` and COLUMNS.
        parameters: The general information, {name: text}.
    """

    COLUMNS = ("mix", "dyn", "card", "echo", "loca", "chan", "extr1", "extr2", "kx", "ky", "kz",
               "aver", "sign", "rf", "grad", "enc", "rtop", "rr", "size", "offset")

    #**********#
    #   init   #
    #**********#
    def __init__(self, index, parameters, raw):
        self.index = index
        self.parameters = parameters
        self._raw = raw

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """Read a data/list pair from either file's path or their common stem."""
        rows, parameters = [], {}
        with open(_sibling(path, ".list")) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) == 21 and fields[0].isalpha() and fields[0].isupper():
                    rows.append((fields[0],) + tuple(int(f) for f in fields[1:]))
                elif line.startswith(".") and ":" in line:
                    name, value = line[1:].split(":", 1)
                    parameters[" ".join(name.split()[3:])] = value.strip()
                elif line.startswith("#") and ":" in line:
                    name, value = line[1:].split(":", 1)
                    parameters[name.strip()] = value.strip()
        index = np.array(rows, dtype=[("typ", "U3")] + [(name, "i8") for name in cls.COLUMNS])
        return cls(index, parameters, np.fromfile(_sibling(path, ".data"), dtype="<f4"))

    #*************#
    #   vectors   #
    #*************#
    def vectors(self, typ="STD", mix=0, signed=True):
        """The vectors of a type and mix as (npoints, channels, vectors per channel)
        complex, each channel's vectors in the order of the file (averages, dynamics, and
        the extra loops as they were played; `rows` tells which is which). With `signed`
        the list's sign, the receiver's phase cycle, is applied."""
        rows = self.rows(typ, mix)
        pairs = [self._raw[row["offset"] // 4:(row["offset"] + row["size"]) // 4] for row in rows]
        vectors = np.stack([(pair[0::2] + 1j * pair[1::2])
                            * (row["sign"] if signed and typ == "STD" else 1)
                            for pair, row in zip(pairs, rows)], axis=-1)
        each = [vectors[:, rows["chan"] == channel] for channel in np.unique(rows["chan"])]
        # a scan stopped mid-average leaves some channels one short
        complete = min(channel.shape[1] for channel in each)
        return np.stack([channel[:, :complete] for channel in each], axis=1)

    def rows(self, typ="STD", mix=0):
        """The list rows of a type and mix, in file order."""
        chosen = self.index[self.index["typ"] == typ]
        return chosen if typ == "NOI" else chosen[chosen["mix"] == mix]


#**************************************************************************************************#
#                                          Class SdatSpar                                          #
#**************************************************************************************************#
#                                                                                                  #
# The reconstructed spectroscopy export: combined FIDs in VAX floats and their text parameters.    #
#                                                                                                  #
#**************************************************************************************************#
class SdatSpar:
    """The reconstructed spectroscopy export: combined FIDs in VAX floats and their text
    parameters.

    The channels are combined by the scanner and, unless the rows hold the dynamics, the
    transients averaged: good for a spectrum, not for a noise reading (scan.calibrate).
    """

    #**********#
    #   load   #
    #**********#
    @staticmethod
    def load(path):
        """(fids, parameters): the FIDs as (rows, samples) complex, stored conjugated as the
        scanner writes them, and the .spar entries as {name: text}."""
        parameters = {}
        with open(_sibling(path, ".spar"), errors="ignore") as handle:
            for line in handle:
                if ":" in line and not line.startswith("!"):
                    name, value = line.split(":", 1)
                    parameters[name.strip()] = value.strip()
        words = np.fromfile(_sibling(path, ".sdat"), dtype="<u4")
        # VAX F floats: the 16-bit halves swapped, an excess-128 exponent on a 0.1f mantissa
        words = (words << np.uint32(16)) | (words >> np.uint32(16))
        exponent = ((words >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64)
        value = np.ldexp(0.5 + (words & np.uint32(0x7FFFFF)) / 2.0 ** 24, exponent - 128)
        value = np.where(exponent == 0, 0.0, np.where(words >> np.uint32(31), -value, value))
        fids = value[0::2] + 1j * value[1::2]
        return fids.reshape(int(parameters["rows"]), int(parameters["samples"])), parameters


#**************************************************************************************************#
#                                           Class LabRaw                                           #
#**************************************************************************************************#
#                                                                                                  #
# The scanner's own raw stream: int16 samples of every channel per label, and the noise label.     #
#                                                                                                  #
#**************************************************************************************************#
class LabRaw:
    """The scanner's own raw stream: the samples of every channel per label, and the
    noise label.

    The .lab file holds one 64-byte label per block of the .raw file (its size, its
    control type, the loop counters, the receiver's phases), the .sin file the number of
    measured channels. Two generations exist. The older one (its .sin file names
    `enable_pda`) stores int16 samples as they are; after Welch's loadLABRAW.m with
    Boer's changes. The newer one stores int32 samples, bit-packed where the label's raw
    format is 4 or 6: chunks of groups of sixteen values, each group with its own bit
    width and shift, and the blocks follow each other by their coded size. Layout and
    decoder after the ISMRMRD project's converter (github.com/ismrmrd/philips_to_ismrmrd,
    main.cpp and philips.h, reverse engineered there). The samples are the receiver's,
    oversampled: no filter or decimation is applied here.

    Attributes:
        labels: {field: array over labels}, the parsed label fields.
        channels: Measured channels.
        packed: True for the newer generation.
        parameters: The .sin file's entries, {name: [text values]} (the first of a name kept).
    """

    NORMAL_DATA, NOISE_DATA, STANDARD_LABEL = 0, 16, 32513
    # (field, word of the label, shift, mask)
    FIELDS = (("size", 0, 0, 0xFFFFFFFF), ("coded_size", 1, 0, 0xFFFFFFFF),
              ("label_type", 3, 16, 0xFFFF), ("control", 4, 0, 0xFF), ("raw_format", 5, 8, 0xFF),
              ("measurement_phase", 4, 16, 0xFF), ("measurement_sign", 4, 24, 0xFF),
              ("mix", 6, 16, 0xFFFF), ("dynamic", 7, 0, 0xFFFF), ("cardiac_phase", 7, 16, 0xFFFF),
              ("echo", 8, 0, 0xFFFF), ("location", 8, 16, 0xFFFF), ("row", 9, 0, 0xFFFF),
              ("extra", 9, 16, 0xFFFF), ("measurement", 10, 0, 0xFFFF), ("e1", 10, 16, 0xFFFF),
              ("e2", 11, 0, 0xFFFF), ("e3", 11, 16, 0xFFFF), ("random_phase", 13, 16, 0xFFFF))

    #**********#
    #   init   #
    #**********#
    def __init__(self, labels, channels, packed, raw_path, parameters=None):
        self.labels = labels
        self.channels = channels
        self.packed = packed
        self.parameters = parameters or {}
        self._raw_path = raw_path
        # the blocks follow a 512-byte header one after another, by their stored size
        stored = labels["size"].astype(np.int64)
        if packed:  # a bit-packed block takes its coded size, every other one is stored as it is
            stored = np.where(np.isin(labels["raw_format"], (4, 6)),
                              labels["coded_size"].astype(np.int64), stored)
        self._offsets = 512 + np.concatenate([[0], np.cumsum(stored[:-1])])

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """Read a lab/raw/sin triple from any of the files' paths or their common stem."""
        words = np.fromfile(_sibling(path, ".lab"), dtype="<u4").reshape(-1, 16)
        labels = {name: (words[:, word] >> np.uint32(shift)) & np.uint32(mask)
                  for name, word, shift, mask in cls.FIELDS}
        with open(_sibling(path, ".sin")) as handle:
            sin = handle.read()
        channels = next(int(line.split()[-1]) for line in sin.splitlines()
                        if "nr_measured_channels" in line)
        parameters = {}
        for line in sin.splitlines():
            if line.count(":") >= 2:
                name, values = (part.strip() for part in line.split(":", 1)[1].split(":", 1))
                parameters.setdefault(name, values.split())
        return cls(labels, channels, "enable_pda" not in sin, _sibling(path, ".raw"), parameters)

    #************#
    #   blocks   #
    #************#
    def which(self, control=NORMAL_DATA):
        """The indices of the standard labels of a control type that hold data."""
        labels = self.labels
        return np.flatnonzero((labels["control"] == control) & (labels["size"] > 0)
                              & (labels["label_type"] == self.STANDARD_LABEL))

    def data(self, mix=0):
        """The normal data of a mix as (samples, channels, vectors) complex in the order
        they were measured (the labels at `which()` tell the loops), the receiver's random
        and measurement phases undone and reversed readouts turned."""
        chosen = [k for k in self.which() if self.labels["mix"][k] == mix]
        out = []
        for k in chosen:
            block = self._block(k)
            phase = (2.0 * float(self.labels["random_phase"][k]) / (2 ** 16 - 1)
                     + float(self.labels["measurement_phase"][k]) / 2.0)
            block = block * np.exp(-1j * np.pi * phase)
            out.append(block[::-1] if self.labels["measurement_sign"][k] else block)
        return np.stack(out, axis=-1)

    def noise(self):
        """The noise label's samples, (samples, channels) complex: the receiver with the
        RF off, measured before the scan."""
        return np.concatenate([self._block(k) for k in self.which(self.NOISE_DATA)], axis=0)

    #***********#
    #   nifti   #
    #***********#
    def nifti(self, f0_mhz):
        """The run as NIfTI-MRS with the coils and the transients apart, at the spectral band:
        the receiver's oversampled samples cropped to the band's middle in the spectrum (what
        the .sin's output resolution keeps; the scanner's own export, its SDAT, agrees with it
        but for its receiver filter), the water references of mix 1 after the run's own
        transients, as a run interleaving its reference stores them (scan.calibrate _raw finds
        them). The carrier is no entry of lab/raw/sin: `f0_mhz` gives it (a SPAR's synthesizer
        frequency, or the field's)."""
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
        from nifti_mrs.hdr_ext import Hdr_Ext

        mixes = [self.data(0)] + ([self.data(1)] if np.any(self.labels["mix"][self.which()] == 1)
                                  else [])
        x = np.concatenate(mixes, axis=-1)
        fids, fold = self._band(x), x.shape[0] // int(self.parameters["output_resolutions"][0])
        header = Hdr_Ext(float(f0_mhz), "1H")
        header.set_dim_info(0, "DIM_COIL")
        header.set_dim_info(1, "DIM_DYN")
        for key, name, scale in (("EchoTime", "spectro_echo_times", 1e-3),
                                 ("RepetitionTime", "spectro_rep_times", 1e-3)):
            if name in self.parameters:
                header.set_standard_def(key, float(self.parameters[name][0]) * scale)
        header.set_standard_def("Manufacturer", "Philips")
        header.set_standard_def("WaterSuppressed", True)
        dwell_s = float(self.parameters["sample_time_interval"][0]) * 1e-6 * fold
        return gen_nifti_mrs_hdr_ext(fids[None, None, None].astype(np.complex64), dwell_s, header)

    def band_noise(self):
        """The noise label at the spectral band as `nifti` crops the run, (samples, channels):
        what scan.calibrate CovarianceReading.from_arrays reads where the FIDs still ring."""
        # cropped readout by readout, at the length of the run's own
        noise, samples = self.noise(), self._block(self.which()[0]).shape[0]
        return np.concatenate([self._band(noise[start:start + samples])
                               for start in range(0, noise.shape[0] - samples + 1, samples)])

    def _band(self, x):
        """The receiver's oversampled samples (samples, ...) cropped to the band's middle in the
        spectrum, as many points as the .sin's output resolution keeps."""
        points = int(self.parameters["output_resolutions"][0])
        spectrum = np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)
        middle = spectrum.shape[0] // 2
        return np.fft.ifft(np.fft.ifftshift(spectrum[middle - points // 2:middle + points // 2],
                                            axes=0), axis=0) / (x.shape[0] // points)

    def _block(self, k):
        size, offset = int(self.labels["size"][k]), int(self._offsets[k])
        if not self.packed:
            values = np.fromfile(self._raw_path, dtype="<i2", count=size // 2, offset=offset)
        elif self.labels["raw_format"][k] not in (4, 6):
            values = np.fromfile(self._raw_path, dtype="<i4", count=size // 4, offset=offset)
        else:
            coded = np.fromfile(self._raw_path, dtype="u1", count=int(self.labels["coded_size"][k]),
                                offset=offset)
            values, at = np.zeros(size // 4, dtype=np.int64), 0
            while at < coded.size:
                # a chunk: decoded bytes, coded bytes, and where its values go; then 32-bit
                # words read from their top bit down
                decoded, length = (int(v) for v in coded[at:at + 4].view("<u2"))
                start = int(coded[at + 4:at + 8].view("<u4")[0]) // 4
                words = coded[at + 8:at + 8 + length]
                bits = np.unpackbits(
                    words[:words.size // 4 * 4].view("<u4").astype(">u4").view("u1"))
                bit, done, count = 0, 0, decoded // 4
                while done < count:
                    width, shift = (int(bits[b:b + 5] @ (16, 8, 4, 2, 1)) for b in (bit, bit + 5))
                    group = min(16, count - done)
                    field = (bits[bit + 10:bit + 10 + group * width].reshape(group, width)
                             .astype(np.int64))
                    number = (field @ (1 << np.arange(width - 1, -1, -1, dtype=np.int64)) if width
                              else np.zeros(group, np.int64))
                    # two's complement
                    number = number - (field[:, 0] << width) if width else number
                    values[start + done:start + done + group] = (
                        (number << shift) + ((1 << shift) >> 1))
                    bit, done = bit + 10 + group * width, done + group
                at += 8 + length
        pairs = values.astype(np.float32).reshape(self.channels, -1, 2)
        return (pairs[..., 0] + 1j * pairs[..., 1]).T
