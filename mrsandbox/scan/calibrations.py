####################################################################################################
#                                         calibrations.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/09/26                                                                                #
#                                                                                                  #
# Purpose: The calibrations against in vivo data, each with the exact conditions it was measured   #
#          under (scanner, coil, sequence, acquisition, voxel, processing), so a number never      #
#          leaves its conditions unnoticed: a scan names the calibrations it used and the          #
#          conditions it departs from. The in vivo leaves are pooled from the package's cohorts    #
#          (scan.cohort); typed here is what no cohort row covers.                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

from mrsandbox.phantom.solution import Phantom


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
        carried_by: "scanner" for a number of the hardware, whose departures a scan names;
            "subject" for one of the tissue, which no scanner or sequence changes.
        presets: The hardware.Scanner presets the record serves; none makes it the
            fallback of its leaf or a check.
        fallback: Whether the record may serve scanners it does not name (the first
            such record of a leaf does); False keeps a vendor's own number to its presets.
        note: What the record is good for and what it is not.
        failed: Why the reading is not to be trusted, by the reader's own stated rule
            (empty when it is): a failed record is never pooled or drawn.
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
    carried_by: str = "scanner"
    presets: tuple = ()
    fallback: bool = True
    note: str = ""
    failed: str = ""

    #****************#
    #   departures   #
    #****************#
    def departures(self, scan):
        """The conditions under which `scan` departs from the calibration, as
        {condition: (calibrated, scan's)}: there the number is extrapolated."""
        found = {}
        if self.carried_by != "scanner":
            return found
        if abs(scan.engine.b0_t - self.scanner["field_t"]) > 0.2:
            found["field_t"] = (self.scanner["field_t"], float(scan.engine.b0_t))
        vendor = getattr(scan.scanner, "name", "").split("_")[0]
        if vendor in ("siemens", "ge", "philips") and vendor != self.scanner["vendor"]:
            found["vendor"] = (self.scanner["vendor"], vendor)
        # a phantom loads a coil less than the head a record was measured on
        if isinstance(scan.brain, Phantom) and "load" not in self.voxel:
            found["load"] = ("a head", scan.brain.name)
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


#*****************************#
#   array noise correlation   #
#*****************************#
# the array's own noise share on a head, read off the noise adjustment measurement of a COWS
# file by scan.calibrate CovarianceReading
COIL_OWN_NOISE_PRISMA_3T = Calibration(
    leaf="coils.loaded.own_noise",
    value=0.25,
    prior=(0.25, 0.25),
    measured={"statistic": "magnitude of the noise correlation between channels: the mean over "
                           "channels of the strongest partner, the strongest pair, the median pair",
              "values": (0.41, 0.51, 0.15), "samples": 65536, "subjects": 1,
              "simulated_at_value": (0.41, 0.50, 0.10), "simulated_without": (0.61, 0.79, 0.13)},
    source={"dataset": "COWS, OpenNeuro ds006812, the noise adjustment measurement of one file",
            "citation": "Igwe et al. 2025, Magn Reson Med, doi 10.1002/mrm.30550",
            "url": "https://openneuro.org/datasets/ds006812"},
    scanner={"vendor": "siemens", "model": "Prisma", "software": None, "field_t": 2.89},
    coil={"channels": (32,), "name": None},
    sequence={"name": "noise adjustment", "product": "the first measurement of the TWIX file"},
    acquisition={"channels": 32, "rf": "off"},
    voxel={"load": "a head"},
    processing="sample covariance of the 32 channels over all samples",
    date="18/09/26",
    note="the modelled array is CoilArray.helmet(), not this coil's geometry (not published)",
)
# the same statistic on four Philips 3 T systems with 32-channel head arrays and a phantom in the
# coil, read off the noise vectors (RF off) of their data/list exports (io.philips.DataList): a
# check that the default of CoilArray.loaded holds across vendors, not in use
COIL_OWN_NOISE_PHILIPS_3T = Calibration(
    leaf="coils.loaded.own_noise",
    value=0.26,
    prior=(0.22, 0.35),
    measured={"statistic": COIL_OWN_NOISE_PRISMA_3T.measured["statistic"],
              "sites": {"P01a, Achieva, R5.6.1": (0.41, 0.52, 0.08, 0.24),
                        "P05a, Achieva dStream, R5.7.1": (0.34, 0.51, 0.14, 0.35),
                        "P08a, Ingenia Elition, R5.6.1": (0.43, 0.51, 0.07, 0.22),
                        "P17a, Ingenia CX, R5.6.1": (0.40, 0.54, 0.13, 0.27)},
              "sites_note": "per site the three statistics and the share they give",
              "samples": 2048, "subjects": 0,
              "other_arrays": {"P07a, Ingenia, 24 channels": (0.39, 0.61, 0.12),
                               "P16a, Ingenia Elition X, 15 channels": (0.42, 0.66, 0.20)}},
    source={"dataset": "Big Drift, the Philips sites' raw exports of the first day's scan before "
                       "the echo-planar run (CC BY-NC-SA 4.0)",
            "citation": "Hui et al. 2021, NeuroImage 241:118430 (the models from its Table 2)",
            "url": "https://www.nitrc.org/projects/bigdrift/"},
    scanner={"vendor": "philips", "model": "Achieva, Achieva dStream, Ingenia Elition, Ingenia CX",
             "software": "R5.6.1 and R5.7.1", "field_t": 3.0},
    coil={"channels": (32, 32, 32, 32), "name": None},
    sequence={"name": "noise vectors of a PRESS export", "product": "RawExp - PRESS"},
    acquisition={"channels": 32, "rf": "off"},
    voxel={"load": "a phantom"},
    processing="data/list, the NOI vector of every channel, sample covariance over its 2048 "
               "samples",
    date="19/09/26",
    note="a check across vendors; the FIDs' tails do not serve here (unsuppressed phantom water "
         "rings to the end and the channels correlate to one)",
)

# and on two Siemens systems of the same study, read off the noise adjustment of their raw files
# with a phantom in the coil (a phantom of a head's size loads the array as a head does); a
# check, not in use
COIL_OWN_NOISE_SIEMENS_PHANTOM_3T = Calibration(
    leaf="coils.loaded.own_noise",
    value=0.26,
    prior=(0.26, 0.26),
    measured={"statistic": COIL_OWN_NOISE_PRISMA_3T.measured["statistic"],
              "sites": {"S01a, Prisma_fit, syngo MR E11, Head_32": (0.40, 0.48, 0.17, 0.26),
                        "S02a, MAGNETOM Vida, syngo MR XA20, HeadNeck_20_TCS, 16 channels":
                            (0.43, 0.47, 0.24, None)},
              "sites_note": "per site the three statistics and, for 32 channels, the share they "
                            "give",
              "samples": 65536, "subjects": 0},
    source=COIL_OWN_NOISE_PHILIPS_3T.source | {"dataset": "Big Drift, two Siemens sites' raw "
                                                          "files of the first day's scan before "
                                                          "the echo-planar run (CC BY-NC-SA 4.0)"},
    scanner={"vendor": "siemens", "model": "Prisma_fit and MAGNETOM Vida",
             "software": "syngo MR E11 and XA20", "field_t": 2.89},
    coil={"channels": (32, 16), "name": "Head_32 and HeadNeck_20_TCS"},
    sequence={"name": "noise adjustment", "product": "the first measurement of the TWIX file"},
    acquisition={"channels": 32, "rf": "off"},
    voxel={"load": "a phantom"},
    processing="sample covariance of the channels over all samples of the noise adjustment",
    date="19/09/26",
    note="a check; a Verio file of the same study (syngo MR B17) holds no noise adjustment",
)

#**************#
#   creatine   #
#**************#
# the creatine range against three datasets' averaged spectra: de Graaf's creatine and
# phosphocreatine ranges drawn side by side put the 3.03 ppm singlet above its measured height
# over NAA's, so the creatine range is the phosphocreatine one, 3.0 to 5.5 mM (a total of 6 to
# 11 mM, half of it phosphorylated in the middle; the split hardly shows at 3 T, stated)
CREATINE_THREE_DATASETS_3T = Calibration(
    leaf="conc.Cr",
    value=4.25,
    prior=(3.0, 5.5),
    measured={"statistic": "height of the 3.03 ppm singlet over the NAA singlet's in the median "
                           "averaged spectrum (2 Hz apodization, NAA phased)",
              "big_gaba_siemens": 0.616, "cows_prefrontal": 0.560, "fmrs_in_pain": 0.625,
              "simulated_with_de_graaf": (0.792, 0.847, 0.762),
              "simulated_total_creatine_over_naa": 1.19, "implied_total_creatine_over_naa": 0.90},
    source={"dataset": "Big GABA sites S1, S5, S6; COWS prefrontal; fMRS in pain (the noise "
                       "records above)", "citation": None, "url": None},
    scanner={"vendor": "siemens and philips", "model": "TrioTim, Prisma, Achieva", "software": None,
             "field_t": 3.0},
    coil={"channels": (32, 4, 1), "name": None},
    sequence={"name": "PRESS and sLASER", "family": "", "te_s": (0.035, 0.026, 0.022)},
    acquisition={"subjects": (36, 10, 15), "measured_on": "the average"},
    voxel={"region": "medial parietal, prefrontal, anterior cingulate"},
    processing="as the noise records, the transients averaged, the subjects' median spectrum",
    date="18/09/26",
    carried_by="subject",
    note="sets the creatine range only; NAA, NAAG, and phosphocreatine keep the de Graaf ranges",
)


#********************#
#   macromolecules   #
#********************#
# the peak anchor of phantom.MeasuredMM, the consensus 3 T macromolecule spectrum: its level rests
# on a stated prior (mm.level_mm) through the area of its whole FID, which also holds everything
# broad beneath the peaks, so the 0.9 ppm peak's height over NAA's is anchored to raw data, the
# ratio measured over twin read once (the source's PEAK_ANCHOR). It is not tuned until twin and
# data agree: NAA's own height grows with the background beneath it, so that would be a fit.
MM_MEASURED_LEVEL_3T = Calibration(
    leaf="mm.measured.peak_anchor",
    value=3.3,
    prior=(2.0, 3.5),
    measured={"statistic": "height of the 0.9 ppm peak (0.85 to 0.97 ppm) above the 0.3 to 0.5 "
                           "ppm baseline in the subjects' median spectrum, NAA height 1, 2 Hz "
                           "apodization: measured over the MeasuredMM twin at the prior's level",
              "fmrs_in_pain": (0.157, 0.051), "cows_prefrontal": (0.177, 0.054),
              "big_gaba_siemens": (0.195, 0.046),
              "twin_with_the_anchor": (0.151, 0.113, 0.125)},
    source=CREATINE_THREE_DATASETS_3T.source,
    scanner=CREATINE_THREE_DATASETS_3T.scanner,
    coil=CREATINE_THREE_DATASETS_3T.coil,
    sequence=CREATINE_THREE_DATASETS_3T.sequence,
    acquisition=CREATINE_THREE_DATASETS_3T.acquisition,
    voxel=CREATINE_THREE_DATASETS_3T.voxel,
    processing=CREATINE_THREE_DATASETS_3T.processing,
    date="19/09/26",
    carried_by="subject",
    note="the median of the three ratios; Big GABA's 27 mL PRESS voxel may hold lipid at 0.9 ppm",
)

#***************#
#   catalogue   #
#***************#
# the typed records in use: what no cohort row of the package covers (scan.cohort pooled serves
# every leaf the cohorts hold readings of). A record serves the scanner presets it names; the
# first record of a leaf is the fallback for every other scanner, and a scan's header says where
# it departs from the record it got (`Calibration.departures`)
CALIBRATIONS = (CREATINE_THREE_DATASETS_3T,)



def calibrations(scanner=None, records=(), channels=None):
    """{leaf: Calibration} for a hardware.Scanner (or a preset name): for every leaf the
    package's cohorts hold readings of, the rows pooled under the scanner's conditions
    (scan.cohort.pooled; `channels` the receive array's count); for the others the typed
    record that names its preset, else the leaf's fallback. `records` are the scanner's
    own (scan.profile) and outrank all."""
    from mrsandbox.hardware.scanners import Scanner
    from mrsandbox.scan import cohort

    name = getattr(scanner, "name", scanner)
    chosen = {}
    for record in CALIBRATIONS:
        if record.fallback:
            chosen.setdefault(record.leaf, record)
    for record in CALIBRATIONS:
        if name in record.presets:
            chosen[record.leaf] = record
    if scanner is not None:
        hardware = scanner if isinstance(scanner, Scanner) else Scanner.preset(scanner)
        for leaf in cohort.leaves():
            record = cohort.pooled(leaf, hardware, channels)
            if record is not None:
                chosen[leaf] = record
    chosen.update({record.leaf: record for record in records})
    return chosen




# the Prisma under conditions far from the COWS rows' (the unedited sub-experiment of HERMES at
# TE 80 in a lateral temporal voxel): the reading leans on the NAA T2 prior there and the twin is
# an ideal PRESS, so the record is a check of what carries, which a scan's departures name
NOISE_DS007806_PRISMA_3T = Calibration(
    leaf="noise.sigma",
    value=8.0,
    prior=(7.2, 9.3),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the standard deviation of the real noise floor), median over subjects",
              "median": 17.6, "implied_value": 8.80, "subjects": 4,
              "apodized": {"statistic": "the same single-transient NAA SNR after a 10 Hz "
                                        "exponential apodization of data and simulation alike",
                           "median": 54.7, "implied_value": 7.26}},
    source={"dataset": "OpenNeuro ds007806, SPfMRS social processing, the first sessions of four "
                       "subjects, the sub-experiment with the editing pulse at 7.5 ppm (CC0)",
            "citation": "Ford et al. 2026, bioRxiv, doi 10.64898/2025.12.03.691778",
            "url": "https://openneuro.org/datasets/ds007806"},
    scanner={"vendor": "siemens", "model": "Prisma_fit", "software": "syngo MR E11",
             "field_t": 2.89},
    coil={"channels": (32,), "name": "Head_32"},
    sequence={"name": "HERMES", "family": "Press",
              "product": "mgs_svs_ed (universal editing sequence)", "te_s": 0.080, "tr_s": 2.0,
              "simulated_as": "Press(te1_s=0.03, te2_s=0.05, tr_s=2.0)"},
    acquisition={"npoints": 4096, "bandwidth_hz": 4000.0, "transients": 195, "measured_on": 1,
                 "water_suppressed": True},
    voxel={"region": "right temporal, functional MRS during a task", "size_mm": (25.0, 40.0, 25.0),
           "simulated_brain": "icbm152", "simulated_position_mm": (52.0, -30.0, 8.0),
           "place_note": "read off the scanner coordinates of one subject, a stand-in"},
    processing="NIfTI-MRS from TWIX (spec2nii), noise-whitened SVD coil combination, SNR per "
               "transient, median per subject",
    date="19/09/26",
    note="a check of what the Prisma's record carries: not to TE 80 in a lateral voxel",
)

# records read the same way under other conditions, kept to test what carries a number there
CHECKS = (COIL_OWN_NOISE_PHILIPS_3T, COIL_OWN_NOISE_SIEMENS_PHANTOM_3T,
          NOISE_DS007806_PRISMA_3T)
# numbers of the hardware models, outside the Params tree
HARDWARE = {c.leaf: c for c in (COIL_OWN_NOISE_PRISMA_3T,)}
# numbers of the signal models, outside the Params tree
MODELS = {c.leaf: c for c in (MM_MEASURED_LEVEL_3T,)}
