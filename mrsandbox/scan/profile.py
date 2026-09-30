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

import numpy as np

from mrsandbox.hardware import CoilArray, Loop, Receiver, Scanner
from mrsandbox.resources.girf import GIRF
from mrsandbox.scan.calibrate import (CovarianceReading, NoiseReading, SpectrumReading,
                                      WaterReading, _raw)
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
    def preset(cls, name, cohort=None):
        """The profile of a named scanner preset, the hardware alone: its scans take the
        package's rows pooled under its conditions (scan.cohort). With `cohort`, the name
        of one of the package's cohorts, that cohort's rows alone are its records (one
        dataset as the scanner's calibration). The vendor's receiver where the raw data
        showed one (hardware.receiver RECEIVERS)."""
        from mrsandbox.hardware.receiver import RECEIVERS
        from mrsandbox.scan import cohort as cohorts

        scanner = Scanner.preset(name)
        records = ()
        if cohort is not None:
            records = tuple(record for record in (cohorts.pooled(leaf, scanner, within=(cohort,))
                                                  for leaf in cohorts.leaves())
                            if record is not None)
        return cls(scanner, records, receiver=RECEIVERS.get(name))

    #*********************#
    #   from a raw file   #
    #*********************#
    @classmethod
    def read(cls, source, voxel, dataset, array=None, load="a head", excitations=1,
             reference=None, seeds=None):
        """The profile one raw acquisition gives: `source` is NIfTI-MRS with the coils and
        the transients apart, `voxel` its virtual twin (a Voxel of a scan set up as the
        acquisition was: field, sequence, timings, place, size; placed with `compiled=True`
        it draws faster, and the readings draw it many times). The single-transient NAA SNR
        gives the noise.sigma record (scan.calibrate NoiseReading), the averaged spectrum's
        NAA width the lineshape.gauss_hz one (SpectrumReading). With `array`, the
        hardware.CoilArray standing in for the coil's geometry, the FIDs' tails give the
        share of the channels' noise that is their own (CovarianceReading; `load` says what
        was in the coil), and the profile's coils are the array so loaded, its combined
        noise the drawn sigma at the twin's place. With `reference`, the same voxel's
        unsuppressed water (NIfTI-MRS, transients apart), or the water transients the run
        interleaves with its own, the residual water the scanner's suppression leaves and
        its receiver delay become records too (WaterReading)."""
        interleaved = reference is None and _raw(source, water=True)[0].shape[2] > 0
        return cls.from_readings(
            SpectrumReading.from_nifti(source), NoiseReading.from_nifti(source, excitations),
            voxel, dataset,
            covariance=None if array is None else CovarianceReading.from_nifti(source),
            array=array, load=load,
            water=(WaterReading.from_nifti(source, reference)
                   if reference is not None or interleaved else None),
            seeds=seeds)

    #*********************#
    #   from readings     #
    #*********************#
    @classmethod
    def from_readings(cls, spectrum, noise, voxel, dataset, covariance=None, array=None,
                      load="a head", water=None, seeds=None):
        """The profile the readings of one acquisition give (`read` makes them from files; a
        cohort's builder from arrays): the spectrum's width first, with `water` the
        residual, the receiver delay, the leaked water and the eddy transient, then the noise
        against a twin held at those readings (the leak aside, which the twin holds calm);
        with `covariance` and `array` the channels' own noise and the array so loaded.
        `seeds` draws the twin that many times per reading in place of the readers' own
        counts (fewer for a twin that plays shaped pulses)."""
        few = {} if seeds is None else {"seeds": seeds}
        shim = spectrum.record(voxel, dataset, **few)
        read = ()
        if water is not None:
            delay = water.record_delay(voxel, dataset, gauss_hz=shim.value, **few)
            if shim.failed:
                delay = dataclasses.replace(delay, failed="read against a failed width: "
                                                          + shim.failed)
            read = (water.record_residual(voxel, dataset), delay, water.record_leak(voxel, dataset),
                    water.record_eddy(voxel, dataset))
        # the twin at this run's own readings, not the population's ranges (a failed one drawn)
        held = {record.leaf: record.value for record in (shim,) + read
                if not record.failed and record.leaf != "oov.fraction"}
        sigma = noise.record(voxel, dataset, held=held, **few)
        if shim.failed:                          # what is read against the width fails with it
            sigma = dataclasses.replace(sigma, failed="read against a failed width: " + shim.failed)
        records, coils = (sigma, shim), None
        # the channels' own number, whatever the geometry
        if covariance is not None and covariance.coils > 1:
            records += (dataclasses.replace(covariance.record_correlation(dataset, load=load),
                                            scanner=sigma.scanner),)
        if array is not None:
            share = dataclasses.replace(covariance.record(array, dataset, load=load),
                                        scanner=sigma.scanner)
            records += (share,)
            coils = array.loaded(own_noise=share.value).calibrated(voxel.position_mm, voxel.size_mm)
        records += tuple(dataclasses.replace(record, scanner=sigma.scanner) for record in read)
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
            raise ValueError(f"only the package's GIRFs ({', '.join(GIRF_NAMES)}) "
                             "are saved, by name")
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)

    def to_dict(self):
        """The profile as plain data, what `save` writes."""
        # JSON holds no complex: a complex matrix (the modes, the resistance they mix) as its
        # real and imaginary parts
        pair = lambda matrix: None if matrix is None else (
            np.stack([np.real(matrix), np.imag(matrix)]).tolist()
            if np.iscomplexobj(matrix) else matrix.tolist())
        coils = None if self.coils is None else {
            "loops": [dataclasses.asdict(loop) for loop in self.coils.loops],
            "noise_resistance": pair(self.coils.noise_resistance), "gain": self.coils.gain,
            "modes": pair(self.coils.modes)}
        return {"scanner": dataclasses.asdict(self.scanner),
                "records": [dataclasses.asdict(record) for record in self.records],
                "coils": coils, "girf": None if self.girf is None else self.girf.name,
                "receiver": None if self.receiver is None else dataclasses.asdict(self.receiver)}

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path):
        """The profile `save` wrote."""
        with open(path) as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_dict(cls, stored):
        """The profile `to_dict` gave."""
        records = tuple(Calibration(**{**record, "prior": tuple(record["prior"]),
                                       "presets": tuple(record["presets"])})
                        for record in stored["records"])
        # a matrix as its real and imaginary parts (an older file holds a real resistance alone)
        matrix = lambda stored: None if stored is None else (
            np.asarray(stored)[0] + 1j * np.asarray(stored)[1]
            if np.asarray(stored).ndim == 3 else np.asarray(stored))
        coils = stored["coils"] and CoilArray(
            [Loop(loop["radius_mm"], tuple(loop["center_mm"]), tuple(loop["normal"]))
             for loop in stored["coils"]["loops"]],
            matrix(stored["coils"]["noise_resistance"]), gain=stored["coils"]["gain"],
            modes=matrix(stored["coils"].get("modes")))
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
