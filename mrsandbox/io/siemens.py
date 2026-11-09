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
#          the dwell, the carrier and what the header says of the scanner.                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import functools
import pathlib
import re
import struct
import tempfile

import numpy as np


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
