####################################################################################################
#                                         calibrations.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The calibrations against in vivo data, each typed with the exact conditions it was      #
#          measured under (scanner, coil, sequence, acquisition, voxel, processing), so a number   #
#          never leaves its conditions unnoticed: a scan names the calibrations it used and the    #
#          conditions it departs from.                                                             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses


#**************************************************************************************************#
#                                        Class Calibration                                         #
#**************************************************************************************************#
#                                                                                                  #
# One number measured on in vivo data, with the conditions of the measurement.                     #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Calibration:
    """One number measured on in vivo data, with the conditions of the measurement.

    A calibration holds for its conditions. A physical law of the simulator (for the
    noise: volume, field, bandwidth, averages; scan.voxel noise_scale) carries it to
    other conditions, and that is an extrapolation until a second calibration under
    those conditions confirms the law. `departures` names them for a scan.

    Args:
        leaf: The parameter the calibration sets, as its path in the Params tree.
        value: The calibrated value at the simulator's reference conditions.
        prior: The (low, high) range of the prior.
        measured: What was measured on the data: statistic, value, spread, subjects.
        source: Dataset, citation, and where the data live.
        scanner: Vendor, model, software, field in tesla.
        coil: The receive array as far as the data tell.
        sequence: Name and timings in seconds; `family` is the part of this package's
            class names the sequence goes by where the name is not.
        acquisition: Points, bandwidth, transients, and how many enter the measurement.
        voxel: Place, size, and the simulator's stand-in for the place.
        processing: How the measured number was read off the data.
        date: When the calibration was made.
        presets: The hardware.Scanner presets the record serves; none makes it the
            fallback of its leaf or a check.
        note: What the record is good for and what it is not.
    """

    leaf: str
    value: float
    prior: tuple
    measured: dict
    source: dict
    scanner: dict
    coil: dict
    sequence: dict
    acquisition: dict
    voxel: dict
    processing: str
    date: str
    presets: tuple = ()
    note: str = ""

    #****************#
    #   departures   #
    #****************#
    def departures(self, scan):
        """The conditions under which `scan` departs from the calibration, as
        {condition: (calibrated, scan's)}: there the number is extrapolated."""
        found = {}
        if abs(scan.engine.b0_t - self.scanner["field_t"]) > 0.2:
            found["field_t"] = (self.scanner["field_t"], float(scan.engine.b0_t))
        vendor = getattr(scan.scanner, "name", "").split("_")[0]
        if vendor in ("siemens", "ge", "philips") and vendor != self.scanner["vendor"]:
            found["vendor"] = (self.scanner["vendor"], vendor)
        name = type(scan.sequence).__name__
        family = self.sequence.get("family", self.sequence["name"])
        if family.lower() not in name.lower():
            found["sequence"] = (family, name)
        return found

    #************#
    #   record   #
    #************#
    def record(self, scan):
        """What a scan's header says of the calibration: its source, its conditions in
        short, and the scan's departures."""
        return {"value": self.value, "source": self.source["dataset"], "date": self.date,
                "scanner": self.scanner, "sequence": self.sequence["name"],
                "extrapolated": {k: list(v) for k, v in self.departures(scan).items()}}


#********************#
#   noise, 3 tesla   #
#********************#
NOISE_BIG_GABA_SIEMENS_3T = Calibration(
    leaf="noise.sigma",
    value=6.3,
    prior=(5.3, 8.0),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization)",
              "median": 36.4, "site_medians": (38.5, 33.8, 36.7), "p5_p95": (28.8, 43.6),
              "subjects": 36, "simulated_at_value": 36.6},
    source={"dataset": "Big GABA, PRESS arm, sites S1, S5, S6",
            "citation": "Mikkelsen et al. 2017, NeuroImage 159:32",
            "url": "https://www.nitrc.org/projects/biggaba/"},
    scanner={"vendor": "siemens", "model": "TrioTim", "software": "syngo MR B17", "field_t": 2.89},
    coil={"channels": (32, 4, 32), "name": None},
    sequence={"name": "PRESS", "product": "svs_se", "te_s": 0.035, "tr_s": 2.0,
              "simulated_as": "Press(te1_s=0.011, te2_s=0.024, tr_s=2.0)"},
    acquisition={"npoints": 2048, "bandwidth_hz": 2000.0, "transients": 64, "measured_on": 1,
                 "water_suppressed": True},
    voxel={"region": "medial parietal lobe", "size_mm": (30.0, 30.0, 30.0),
           "simulated_brain": "icbm152", "simulated_position_mm": (0.0, -45.0, 35.0)},
    processing="raw TWIX, receiver oversampling removed, noise-whitened SVD coil combination "
               "(weights from the mean transient), SNR per transient, median per subject",
    date="18/09/26",
)


#***************************#
#   noise, prisma, slaser   #
#***************************#
# serves the siemens_prisma_3t preset. The 3 T prior above, carried here by noise_scale,
# gives an SNR of 14.3 where the 29 spectra give 21.1: the Prisma with its 32-channel
# array and semi-LASER is 1.5 times quieter than the Trio sites with PRESS. The narrower
# in vivo line (4.6 to 5.5 Hz by region against the simulated 5.8) is a part of it and is
# absorbed into the value (stated). By region: prefrontal 23.7, occipital 22.2, posterior
# frontal near the vertex 16.3, where the simulated places give 14.0, 15.0, 14.0: the
# place under the array, which the pooled value does not carry.
NOISE_COWS_PRISMA_3T = Calibration(
    leaf="noise.sigma",
    value=4.3,
    prior=(3.4, 6.4),
    measured={"statistic": NOISE_BIG_GABA_SIEMENS_3T.measured["statistic"], "median": 21.1,
              "p5_p95": (14.1, 26.7), "region_medians": (23.7, 22.2, 16.3), "subjects": 10,
              "spectra": 29, "naa_fwhm_hz": (4.9, 5.6, 4.6), "simulated_naa_fwhm_hz": 5.8,
              "simulated_at_big_gaba_value": 14.3, "averaged_over_single": 5.55},
    source={"dataset": "COWS, OpenNeuro ds006812, the VAPOR metabolite scans of all subjects "
                       "and regions (one occipital file unreadable)",
            "citation": "Igwe et al. 2025, Magn Reson Med, doi 10.1002/mrm.30550",
            "url": "https://openneuro.org/datasets/ds006812"},
    scanner={"vendor": "siemens", "model": "Prisma", "software": "syngo MR E11", "field_t": 2.89},
    coil={"channels": (32,), "name": "Siemens 32-channel head and neck (the paper)"},
    sequence={"name": "sLASER", "family": "SemiLaser", "product": "svs_slaser", "te_s": 0.026,
              "tr_s": 2.0,
              "te_note": "26 ms in the headers and the dataset's README, 20 ms in the paper",
              "simulated_as": "SemiLaser(te_s=0.026, tr_s=2.0)"},
    acquisition={"npoints": 2048, "bandwidth_hz": 4000.0, "transients": 32, "measured_on": 1,
                 "water_suppressed": True},
    voxel={"region": "prefrontal, occipital, posterior frontal", "size_mm": (20.0, 20.0, 20.0),
           "size_note": "20 mm in the headers, 25 mm in the dataset's README",
           "simulated_brain": "icbm152",
           "simulated_position_mm": ((0.0, 50.0, 25.0), (0.0, -75.0, 10.0), (0.0, -10.0, 55.0))},
    processing=NOISE_BIG_GABA_SIEMENS_3T.processing,
    date="18/09/26",
    presets=("siemens_prisma_3t",),
    note="the places are stand-ins; the prior is the spectra's 5th to 95th percentile",
)


#*********************************#
#   noise, 3 tesla, one channel   #
#*********************************#
# not in use either: a transmit-receive birdcage in place of an array. The 3 T prior
# gives 19.2 where the 15 subjects give 13.0 (11.1 to 15.5): the birdcage hears a
# medial frontal voxel 1.5 times worse than the arrays of Big GABA hear a parietal one,
# the order an array's gain at that depth has. With COWS above (1.9 the other way, a
# newer scanner and array) it says what the drawn sigma is: a property of the receive
# chain, scanner and coil, that one prior per field does not carry.
NOISE_FMRS_PAIN_ACHIEVA_3T = Calibration(
    leaf="noise.sigma",
    value=9.3,
    prior=(9.3, 9.3),
    measured={"statistic": NOISE_BIG_GABA_SIEMENS_3T.measured["statistic"], "median": 13.0,
              "range": (11.1, 15.5), "subjects": 15, "simulated_at_big_gaba_value": 19.2},
    source={"dataset": "fMRS in pain, the functional runs",
            "citation": "Archibald et al. 2020, Sci Rep 10:19218",
            "url": "https://www.nitrc.org/projects/fmrs_2020"},
    scanner={"vendor": "philips", "model": "Achieva", "software": None, "field_t": 3.0},
    coil={"channels": (1,), "name": "transmit-receive head coil (the paper)"},
    sequence={"name": "PRESS", "product": None, "te_s": 0.022, "tr_s": 4.0,
              "simulated_as": "Press(te1_s=0.009, te2_s=0.013, tr_s=4.0), the split a stand-in"},
    acquisition={"npoints": 2048, "bandwidth_hz": 2000.0, "transients": 320, "measured_on": 1,
                 "water_suppressed": True,
                 "bandwidth_note": "not in the files or the paper; fixed by the creatine to NAA "
                                   "spacing in the data, 1.02 ppm at 2 kHz"},
    voxel={"region": "anterior cingulate", "size_mm": (25.0, 30.0, 15.0),
           "simulated_brain": "icbm152", "simulated_position_mm": (0.0, 35.0, 20.0)},
    processing="raw .data/.list, one channel, the list's sign applied per phase cycle step, "
               "SNR per transient, median per subject",
    date="18/09/26",
    note="a check of the scaling law across vendor and coil; the voxel's place a stand-in; not in use",
)


#*****************************#
#   array noise correlation   #
#*****************************#
# the share of a channel's noise that is its own, hardware.CoilArray.loaded(own_noise):
# read off the noise adjustment measurement in the COWS file, the array on a head
COIL_OWN_NOISE_PRISMA_3T = Calibration(
    leaf="coils.loaded.own_noise",
    value=0.3,
    prior=(0.3, 0.3),
    measured={"statistic": "magnitude of the noise correlation between channels: the mean over "
                           "channels of the strongest partner, the strongest pair, the median pair",
              "values": (0.41, 0.51, 0.15), "samples": 65536, "subjects": 1,
              "simulated_at_value": (0.41, 0.48, 0.12), "simulated_without": (0.59, 0.69, 0.16)},
    source=NOISE_COWS_PRISMA_3T.source,
    scanner=NOISE_COWS_PRISMA_3T.scanner,
    coil=NOISE_COWS_PRISMA_3T.coil,
    sequence={"name": "noise adjustment", "product": "the first measurement of the TWIX file"},
    acquisition={"channels": 32, "rf": "off"},
    voxel={"load": "a head"},
    processing="sample covariance of the 32 channels over all samples",
    date="18/09/26",
    note="the modelled array is CoilArray.helmet(), not this coil's geometry (not published)",
)
#***************#
#   catalogue   #
#***************#
# the calibrations in use. A record serves the scanner presets it names; the first record
# of a leaf without presets is the fallback for every other scanner, and a scan's header
# says where it departs from the record it got (`Calibration.departures`)
CALIBRATIONS = (NOISE_BIG_GABA_SIEMENS_3T, NOISE_COWS_PRISMA_3T)


def calibrations(scanner=None):
    """{leaf: Calibration} for a hardware.Scanner (or a preset name): the record that
    names its preset, else the leaf's fallback."""
    name = getattr(scanner, "name", scanner)
    chosen = {}
    for record in CALIBRATIONS:
        if name in record.presets or (not record.presets and record.leaf not in chosen):
            chosen[record.leaf] = record
    return chosen
# records read the same way under other conditions, kept to test what carries a number there
CHECKS = (NOISE_FMRS_PAIN_ACHIEVA_3T)
# numbers of the hardware models, outside the Params tree
HARDWARE = {c.leaf: c for c in (COIL_OWN_NOISE_PRISMA_3T,)}
