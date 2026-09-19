####################################################################################################
#                                            session.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The state of one console session, free of any interface: which scanner and subject are  #
#          in the room, where the voxel lies, the protocol card, and the scans that ran. The       #
#          console draws it and asks it to act; all physics stays in the package.                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import pathlib

import numpy as np

from mrsandbox.hardware.scanners import PRESETS
from mrsandbox.phantom import Brain, Phantom
from mrsandbox.scan import ScannerProfile
from mrsandbox.sequences import Press, SemiLaser, Steam

#**************#
#   subjects   #
#**************#
# who can lie on the table: a name for the picker and how to make them
SUBJECTS = {
    "Brain, ICBM152 average": lambda: Brain.load("icbm152"),
    "Phantom, SPECTRE sphere": lambda: Phantom.spectre(temperature_c=21.0),
    "Phantom, Braino sphere": lambda: Phantom.braino(temperature_c=21.0),
}

#***************#
#   sequences   #
#***************#
# the protocol library of the walking skeleton: a name and how a card's timings make the design
SEQUENCES = {
    "PRESS": lambda te_s, tr_s: Press(te1_s=min(0.011, te_s / 2.0), te2_s=te_s - min(0.011, te_s / 2.0), tr_s=tr_s),
    "STEAM": lambda te_s, tr_s: Steam(te_s=te_s, tm_s=0.010, tr_s=tr_s),
    "sLASER": lambda te_s, tr_s: SemiLaser(te_s=te_s, tr_s=tr_s),
}


#**************************************************************************************************#
#                                          Class Protocol                                          #
#**************************************************************************************************#
#                                                                                                  #
# One protocol card: the sequence and what an operator sets on it.                                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass
class Protocol:
    """One protocol card: the sequence and what an operator sets on it."""

    sequence: str = "PRESS"
    te_ms: float = 30.0
    tr_ms: float = 2000.0
    averages: int = 64
    npoints: int = 2048
    bandwidth_hz: float = 4000.0

    @property
    def scan_time_s(self):
        return self.averages * self.tr_ms / 1000.0


#**************************************************************************************************#
#                                          Class Session                                           #
#**************************************************************************************************#
#                                                                                                  #
# What is in the scanner room and what was scanned, without any interface.                         #
#                                                                                                  #
#**************************************************************************************************#
class Session:
    """What is in the scanner room and what was scanned, without any interface.

    Attributes:
        scanner: Name of the scanner preset on the floor.
        subject: Name of the subject on the table (a key of SUBJECTS).
        position_mm, size_mm: The planned voxel, world millimetres.
        protocol: The protocol card being edited.
        scans: What ran, newest last: {"name", "protocol", "data" (NIfTI-MRS), "ppm", "spectrum"}.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        self.scanner = "siemens_prisma_3t"
        self.subject = "Brain, ICBM152 average"
        self.position_mm = [0.0, -45.0, 35.0]
        self.size_mm = 20.0
        self.protocol = Protocol()
        self.scans = []
        self._bodies = {}

    #************#
    #   choices  #
    #************#
    @staticmethod
    def scanners():
        return sorted(PRESETS)

    #**********#
    #   body   #
    #**********#
    def body(self):
        """The subject on the table, made once per name."""
        if self.subject not in self._bodies:
            self._bodies[self.subject] = SUBJECTS[self.subject]()
        return self._bodies[self.subject]

    #************#
    #   planes   #
    #************#
    def planes(self):
        """The three planning planes through the voxel's centre as {"tra", "sag", "cor"}:
        each (image, rectangle), an 8-bit array drawn with its first row at the top
        (anterior up in the transverse plane, superior up in the other two) and the
        voxel's outline (left, top, width, height) in the image's pixels."""
        body = self.body()
        tissue = 0.85 * body.gm + 1.0 * body.wm + 0.35 * body.csf
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        i, j, k = (int(round(c)) for c in (np.asarray(self.position_mm) - origin) / step)
        nx, ny, nz = tissue.shape
        i, j, k = min(max(i, 0), nx - 1), min(max(j, 0), ny - 1), min(max(k, 0), nz - 1)
        half = 0.5 * self.size_mm / step
        side = max(nx, ny, nz)

        def framed(plane, left, top, width, height):
            """A plane shaded to 8 bits and centred in a square of the largest grid side, all three
            planes then drawing alike; the voxel's outline moves with it."""
            shade = np.uint8(255 * np.clip(plane / max(float(tissue.max()), 1e-6), 0.0, 1.0))
            rows, columns = (side - shade.shape[0]) // 2, (side - shade.shape[1]) // 2
            square = np.zeros((side, side), dtype=np.uint8)
            square[rows:rows + shade.shape[0], columns:columns + shade.shape[1]] = shade
            return square, (left + columns, top + rows, width, height)

        return {
            "tra": framed(tissue[:, ::-1, k].T, i - half[0], ny - 1 - j - half[1], 2 * half[0], 2 * half[1]),
            "sag": framed(tissue[i, :, ::-1].T, j - half[1], nz - 1 - k - half[2], 2 * half[1], 2 * half[2]),
            "cor": framed(tissue[:, j, ::-1].T, i - half[0], nz - 1 - k - half[2], 2 * half[0], 2 * half[2]),
        }

    def place(self, plane, column, row):
        """Move the voxel to the point clicked in a plane, (column, row) in image pixels;
        the coordinate the plane does not show stays."""
        body = self.body()
        step, origin = np.diag(body.affine)[:3], body.affine[:3, 3]
        nx, ny, nz = body.gm.shape
        side = max(nx, ny, nz)
        # the planes are centred in squares of the largest grid side (`planes`)
        px, py, pz = ((side - n) // 2 for n in (nx, ny, nz))
        index = {"tra": (column - px, ny - 1 - (row - py), None), "sag": (None, column - py, nz - 1 - (row - pz)),
                 "cor": (column - px, None, nz - 1 - (row - pz))}[plane]
        for axis, value in enumerate(index):
            if value is not None:
                self.position_mm[axis] = float(origin[axis] + step[axis] * value)

    #***********#
    #   voxel   #
    #***********#
    def voxel(self):
        """The planned voxel of a scan of this room under the card's sequence."""
        card = self.protocol
        sequence = SEQUENCES[card.sequence](card.te_ms / 1000.0, card.tr_ms / 1000.0)
        scan = ScannerProfile.preset(self.scanner).scan(self.body(), sequence)
        return scan.voxel(tuple(self.position_mm), self.size_mm)

    def outlook(self):
        """What the card promises before it runs: tissue in the voxel, the scan time, and
        why the design would not play on this scanner (empty when it does)."""
        voxel = self.voxel()
        fractions = voxel.fractions
        return {"gm": float(fractions.gm), "wm": float(fractions.wm), "csf": float(fractions.csf),
                "scan_time_s": self.protocol.scan_time_s,
                "violations": tuple(voxel.scan.sequence.violations(voxel.scan.scanner))}

    #*********#
    #   run   #
    #*********#
    def run(self, seed=None):
        """Acquire the card at the planned voxel and keep the scan; returns it."""
        card = self.protocol
        npoints = int(card.npoints)
        data = self.voxel().acquire(seed=seed, npoints=npoints, dwell_s=1.0 / card.bandwidth_hz,
                                    averages=int(card.averages))
        fid = np.asarray(data[:]).reshape(-1)[:npoints]
        # the display a console gives: light apodization, zero filled, phased on its tallest line upfield of water
        window = np.exp(-np.pi * 2.0 * np.arange(npoints) / card.bandwidth_hz)
        spectrum = np.fft.fftshift(np.fft.fft(fid * window, 2 * npoints))
        f0_mhz = float(data.hdr_ext["SpectrometerFrequency"][0])
        ppm = np.fft.fftshift(np.fft.fftfreq(2 * npoints, 1.0 / card.bandwidth_hz)) / f0_mhz
        upfield = (ppm > 0.2) & (ppm < 4.2)
        top = np.flatnonzero(upfield)[np.abs(spectrum[upfield]).argmax()]
        spectrum = spectrum * np.exp(-1j * np.angle(spectrum[top]))
        scan = {"name": f"{len(self.scans) + 1} {card.sequence} TE {card.te_ms:g}", "protocol": dataclasses.replace(card),
                "data": data, "ppm": ppm[upfield], "spectrum": spectrum.real[upfield]}
        self.scans.append(scan)
        return scan

    #************#
    #   export   #
    #************#
    def export(self, scan, folder):
        """Write a scan as NIfTI-MRS into `folder`; returns the path."""
        folder = pathlib.Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (scan["name"].replace(" ", "_").replace(".", "p") + ".nii.gz")
        scan["data"].save(path)
        return path
