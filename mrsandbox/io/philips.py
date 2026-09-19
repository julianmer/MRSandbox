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
        vectors = np.stack([(pair[0::2] + 1j * pair[1::2]) * (row["sign"] if signed and typ == "STD" else 1)
                            for pair, row in zip(pairs, rows)], axis=-1)
        return np.stack([vectors[:, rows["chan"] == channel] for channel in np.unique(rows["chan"])], axis=1)

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
    """The scanner's own raw stream: int16 samples of every channel per label, and the
    noise label.

    The .lab file holds one 64-byte label per block of the .raw file (its size, its
    control type, the loop counters, the receiver's phases), the .sin file the number of
    measured channels. After Welch's loadLABRAW.m with Boer's changes for release 5 (the
    block sizes count as stored, the channels come from the .sin file). The samples are
    the receiver's, oversampled: no filter or decimation is applied here. Newer releases
    store the blocks compressed (the .raw file is then smaller than its labels' sizes
    add up to); the codec is not public and such a file is refused: export data/list.

    Attributes:
        labels: {field: array over labels}, the parsed label fields.
        channels: Measured channels.
    """

    NORMAL_DATA, NOISE_DATA, STANDARD_LABEL = 0, 16, 32513
    # (field, word of the label, shift, mask)
    FIELDS = (("size", 0, 0, 0xFFFFFFFF), ("label_type", 3, 16, 0xFFFF), ("control", 4, 0, 0xFF),
              ("measurement_phase", 4, 16, 0xFF), ("measurement_sign", 4, 24, 0xFF),
              ("mix", 6, 16, 0xFFFF), ("dynamic", 7, 0, 0xFFFF), ("cardiac_phase", 7, 16, 0xFFFF),
              ("echo", 8, 0, 0xFFFF), ("location", 8, 16, 0xFFFF), ("row", 9, 0, 0xFFFF),
              ("extra", 9, 16, 0xFFFF), ("measurement", 10, 0, 0xFFFF), ("e1", 10, 16, 0xFFFF),
              ("e2", 11, 0, 0xFFFF), ("e3", 11, 16, 0xFFFF), ("random_phase", 13, 16, 0xFFFF))

    #**********#
    #   init   #
    #**********#
    def __init__(self, labels, channels, raw_path):
        self.labels = labels
        self.channels = channels
        self._raw_path = raw_path
        # the blocks follow a 512-byte header one after another
        self._offsets = 512 + np.concatenate([[0], np.cumsum(labels["size"][:-1].astype(np.int64))])

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
            channels = next(int(line.split()[-1]) for line in handle if "nr_measured_channels" in line)
        raw_path = _sibling(path, ".raw")
        if os.path.getsize(raw_path) < 512 + int(labels["size"].astype(np.int64).sum()):
            raise NotImplementedError("the .raw file is compressed; export the scan as data/list")
        return cls(labels, channels, raw_path)

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

    def _block(self, k):
        count = int(self.labels["size"][k]) // 2
        samples = np.fromfile(self._raw_path, dtype="<i2", count=count, offset=int(self._offsets[k]))
        pairs = samples.astype(np.float32).reshape(self.channels, -1, 2)
        return (pairs[..., 0] + 1j * pairs[..., 1]).T
