####################################################################################################
#                                            profile.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The virtual copy of one scanner: its limits, its receive chain, and the calibration     #
#          records measured on its own raw data, bundled so a scan of it is one call and a         #
#          profile of a new scanner is read off a raw file.                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import json

from mrsandbox.hardware import CoilArray, Loop, Receiver, Scanner
from mrsandbox.resources.girf import GIRF
from mrsandbox.scan.calibrate import CovarianceReading, NoiseReading, SpectrumReading
from mrsandbox.scan.calibrations import Calibration, calibrations
from mrsandbox.scan.virtual_scan import VirtualScan

#***************#
#   constants   #
#***************#
# the GIRFs the package fetches (resources.girf); a saved profile names its own
GIRF_NAMES = ("nws/prisma_3t", "reverse/philips_3t", "reverse/philips_7t")


#**************************************************************************************************#
#                                       Class ScannerProfile                                       #
#**************************************************************************************************#
#                                                                                                  #
# One scanner as the package replicates it: limits, receive chain, and its own records.            #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class ScannerProfile:
    """One scanner as the package replicates it: limits, receive chain, and its own records.

    A record of the profile outranks the package's record of the same leaf
    (scan.calibrations), in the priors of every scan and in its header; a leaf the
    profile has no record of keeps the package's, with its departures named.

    Args:
        scanner: The hardware.Scanner: field and limits.
        records: The Calibrations measured on this scanner's own raw data.
        coils: Its hardware.CoilArray; None gives one ideal channel.
        girf: Its measured gradient impulse response, a resources.GIRF.
        receiver: Its hardware.Receiver.
    """

    scanner: Scanner
    records: tuple = ()
    coils: CoilArray = None
    girf: GIRF = None
    receiver: Receiver = None

    #*****************#
    #   from preset   #
    #*****************#
    @classmethod
    def preset(cls, name):
        """The profile of a named scanner preset, served by the package's records."""
        return cls(Scanner.preset(name))

    #*********************#
    #   from a raw file   #
    #*********************#
    @classmethod
    def read(cls, source, voxel, dataset, array=None, load="a head", excitations=1):
        """The profile one raw acquisition gives: `source` is NIfTI-MRS with the coils and
        the transients apart, `voxel` its virtual twin (a Voxel of a scan set up as the
        acquisition was: field, sequence, timings, place, size; placed with
        `compiled=True` it is drawn several times faster, and the readings draw it some
        eighty times). The single-transient
        NAA SNR gives the noise.sigma record (scan.calibrate NoiseReading), the averaged
        spectrum's NAA width the lineshape.gauss_hz one (SpectrumReading). With `array`,
        the hardware.CoilArray standing in for the coil's geometry, the FIDs' tails give
        the share of the channels' noise that is their own (CovarianceReading; `load` says
        what was in the coil), and the profile's coils are the array so loaded, its
        combined noise the drawn sigma at the twin's place."""
        noise = NoiseReading.from_nifti(source, excitations).record(voxel, dataset)
        records, coils = (noise, SpectrumReading.from_nifti(source).record(voxel, dataset)), None
        if array is not None:
            share = dataclasses.replace(CovarianceReading.from_nifti(source).record(array, dataset, load=load),
                                        scanner=noise.scanner)
            records += (share,)
            coils = array.loaded(own_noise=share.value).calibrated(voxel.position_mm, voxel.size_mm)
        return cls(voxel.scan.scanner, records, coils=coils)

    #**********#
    #   save   #
    #**********#
    def save(self, path):
        """Write the profile as one JSON file, readable as it stands: the scanner, the
        records with their conditions, the coils as loops with their noise resistance and
        gain, the receiver, and the GIRF by name. A mapped array and a GIRF other than the
        package's deposits live in files of their own and are refused."""
        if self.coils is not None and not self.coils.loops:
            raise ValueError("only an array of loops is saved; a mapped array lives in its maps")
        if self.girf is not None and self.girf.name not in GIRF_NAMES:
            raise ValueError(f"only the package's GIRFs ({', '.join(GIRF_NAMES)}) are saved, by name")
        coils = None if self.coils is None else {
            "loops": [dataclasses.asdict(loop) for loop in self.coils.loops],
            "noise_resistance": self.coils.noise_resistance.tolist(), "gain": self.coils.gain}
        with open(path, "w") as file:
            json.dump({"scanner": dataclasses.asdict(self.scanner),
                       "records": [dataclasses.asdict(record) for record in self.records],
                       "coils": coils, "girf": None if self.girf is None else self.girf.name,
                       "receiver": None if self.receiver is None else dataclasses.asdict(self.receiver)},
                      file, indent=2)

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """The profile `save` wrote."""
        with open(path) as file:
            stored = json.load(file)
        records = tuple(Calibration(**{**record, "prior": tuple(record["prior"]),
                                       "presets": tuple(record["presets"])}) for record in stored["records"])
        coils = stored["coils"] and CoilArray(
            [Loop(loop["radius_mm"], tuple(loop["center_mm"]), tuple(loop["normal"])) for loop in stored["coils"]["loops"]],
            stored["coils"]["noise_resistance"], gain=stored["coils"]["gain"])
        return cls(Scanner(**stored["scanner"]), records, coils=coils,
                   girf=stored["girf"] and GIRF.named(stored["girf"]),
                   receiver=stored["receiver"] and Receiver(**stored["receiver"]))

    #******************#
    #   calibrations   #
    #******************#
    def calibrations(self):
        """{leaf: Calibration} in force: the profile's own, else the package's."""
        return calibrations(self.scanner, self.records)

    #**********#
    #   scan   #
    #**********#
    def scan(self, brain, sequence, **kwargs):
        """A VirtualScan of this scanner; `kwargs` are VirtualScan's."""
        return VirtualScan(brain, self.scanner, sequence, coils=self.coils, girf=self.girf,
                           receiver=self.receiver, records=self.records, **kwargs)
