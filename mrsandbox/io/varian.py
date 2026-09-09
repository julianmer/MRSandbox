####################################################################################################
#                                            varian.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: A minimal reader for the Varian fid format (binary fid blocks plus the procpar text     #
#          header), enough to load the spectra the resource catalog points at.                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import struct

import numpy as np


#**************************************************************************************************#
#                                         Class VarianFid                                          #
#**************************************************************************************************#
#                                                                                                  #
# A minimal reader for the Varian fid format the consensus collection is stored in.                #
#                                                                                                  #
#**************************************************************************************************#
class VarianFid:
    """A minimal reader for the Varian fid format the consensus collection is stored in.

    Reads one float32 block (main header 32 bytes, block header 28 bytes, big endian
    interleaved real and imaginary points) and the procpar text parameters.
    """

    #**********#
    #   read   #
    #**********#
    @staticmethod
    def load(path):
        """The complex FID of the first block."""
        with open(path, "rb") as handle:
            raw = handle.read()
        nblocks, ntraces, npoints, ebytes, _, _ = struct.unpack(">6i", raw[:24])
        status = struct.unpack(">h", raw[26:28])[0]
        if not (status & 0x8 and ebytes == 4):
            raise IOError(f"unsupported Varian data layout (status {status:#x}, ebytes {ebytes})")
        data = np.frombuffer(raw[32 + 28:32 + 28 + npoints * ebytes], dtype=">f4")
        return (data[0::2] + 1j * data[1::2]).astype(np.complex64)

    @staticmethod
    def parameters(path, names=("sw", "sfrq", "np")):
        """Selected procpar values (first value of each named parameter)."""
        values = {}
        with open(path, errors="ignore") as handle:
            lines = handle.read().splitlines()
        for i, line in enumerate(lines):
            token = line.split(" ", 1)[0]
            if token in names and i + 1 < len(lines):
                fields = lines[i + 1].split()
                if len(fields) >= 2:
                    values[token] = float(fields[1])
        missing = [n for n in names if n not in values]
        if missing:
            raise IOError(f"procpar is missing {missing}")
        return values
