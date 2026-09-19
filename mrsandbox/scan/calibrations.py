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
        carried_by: "scanner" for a number of the hardware, whose departures a scan names;
            "subject" for one of the tissue, which no scanner or sequence changes.
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
    carried_by: str = "scanner"
    presets: tuple = ()
    note: str = ""

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


#*****************#
#   the measure   #
#*****************#
# The noise records read the single-transient NAA SNR twice, through scan.calibrate
# NoiseReading: plain (Near et al. 2021) and under a 10 Hz exponential apodization of data
# and simulation alike, and each reading gives the sigma at which the virtual twin
# reproduces it. The plain height follows the in vivo linewidth and lineshape (shim, eddy
# currents), the apodized one the broad signal under the singlet (the macromolecule at
# 2.04 ppm the default model is short of): neither belongs to the receive chain, they err
# in opposite directions, and they agree within 25 percent, mostly within 7. The value of a
# record is the median over subjects of the two sigmas' mean, the prior its 5th to 95th
# percentile. (A first round read the apodized floor off oversampled data cropped to the
# receiver's band, where the window sat at the band's edge and read up to twice the
# floor; the reader now carries the plain floor to the apodized spectrum as for white
# noise. Values of that round: 6.3, 3.9, 10.1, 5.3, 2.5.)
APODIZED = ("the same single-transient NAA SNR after a 10 Hz exponential apodization of data and "
            "simulation alike")


#********************#
#   noise, 3 tesla   #
#********************#
NOISE_BIG_GABA_SIEMENS_3T = Calibration(
    leaf="noise.sigma",
    value=5.5,
    prior=(4.3, 6.9),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over subjects",
              "median": 36.5, "implied_value": 6.16,
              "site_medians": ((39.5, 149.4), (34.5, 166.7), (36.8, 125.1)), "subjects": 36,
              "apodized": {"statistic": APODIZED, "median": 143.3, "implied_value": 4.78}},
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
    presets=("siemens_trio_3t",),
    note="the first record of its leaf: the fallback of every scanner without its own",
)


#***************************#
#   noise, prisma, slaser   #
#***************************#
# serves the siemens_prisma_3t preset: the Prisma with its 32-channel array is a tenth
# quieter than the Trio sites (4.9 against 5.5), the two readings a few percent apart. By
# region the SNR runs prefrontal 23.7, occipital 22.3, posterior frontal near the vertex
# 16.3 (apodized 64.4, 57.1, 41.0): the place under the array, which the pooled value does
# not carry (hardware.CoilArray.helmet is checked against it).
NOISE_COWS_PRISMA_3T = Calibration(
    leaf="noise.sigma",
    value=4.9,
    prior=(3.7, 6.9),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over subjects",
              "median": 21.3, "implied_value": 5.07,
              "region_medians": ((23.7, 64.4), (22.3, 57.1), (16.3, 41.0)), "subjects": 10, "spectra": 29,
              "naa_fwhm_hz": (4.9, 5.6, 4.6), "simulated_naa_fwhm_hz": 5.8, "averaged_over_single": 5.55,
              "apodized": {"statistic": APODIZED, "median": 57.1, "implied_value": 4.75}},
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
           "simulated_position_mm": ((-1.0, 51.0, 18.0), (0.0, -75.0, 21.0), (0.0, 4.0, 50.0)),
           "registered_note": "the headers' voxel centres carried through an affine registration "
                              "of each subject's T1 to the ICBM152 brain, medians over subjects; "
                              "the file named Parietal of subject 1 lies occipital"},
    processing=NOISE_BIG_GABA_SIEMENS_3T.processing,
    date="18/09/26",
    presets=("siemens_prisma_3t",),
    note="simulated at the registered places; the prior is the spectra's 5th to 95th "
         "percentile, apodized",
)


#*********************************#
#   noise, 3 tesla, one channel   #
#*********************************#
# serves the philips_achieva_3t preset: a transmit-receive birdcage in place of an array,
# twice as noisy as the arrays above for a medial frontal voxel (10.3 against 4.9 to 5.5),
# the order an array's gain at that depth has; both readings within two percent. The
# drawn sigma is a property of the receive chain, scanner and coil, that one prior per
# field does not carry.
NOISE_FMRS_PAIN_ACHIEVA_3T = Calibration(
    leaf="noise.sigma",
    value=10.3,
    prior=(8.8, 11.4),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over subjects",
              "median": 13.0, "implied_value": 10.08, "subjects": 15,
              "apodized": {"statistic": APODIZED, "median": 44.9, "implied_value": 10.24}},
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
    presets=("philips_achieva_3t",),
    note="the voxel's place and the echo's split are stand-ins; the prior is the subjects' 5th "
         "to 95th percentile, apodized",
)


#**********************#
#   noise, ge, press   #
#**********************#
# serves the ge_mr750_3t preset: the two GE sites of Big GABA whose P-files hold the 64
# transients apart (psd jpress), an 8-channel array: 7.0 against the 32-channel Siemens
# sites' 5.5 under the same protocol. The readings stand furthest apart here (8.2 plain,
# 5.9 apodized): the GE lines are broad and misshapen before an eddy current correction,
# which pulls the plain height down. The four other GE sites store frames of 2 or 8 summed
# excitations; divided by the square root of that they read a quarter below these two: a
# frame is not its excitations' ideal sum, and no record is built on one.
NOISE_BIG_GABA_GE_3T = Calibration(
    leaf="noise.sigma",
    value=7.0,
    prior=(6.0, 10.3),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over subjects",
              "median": 27.7, "implied_value": 8.21,
              "site_medians": ((28.9, 128.7), (27.7, 120.0)), "subjects": 24,
              "apodized": {"statistic": APODIZED, "median": 120.5, "implied_value": 5.93}},
    source={"dataset": "Big GABA, PRESS arm, sites G7, G8",
            "citation": "Mikkelsen et al. 2017, NeuroImage 159:32",
            "url": "https://www.nitrc.org/projects/biggaba/"},
    scanner={"vendor": "ge", "model": None, "software": "P-file revision 24; the model is not in the "
             "files (in database.scanners OPENNEURO_SURVEY the release 24 LX, DV24.0, belongs to "
             "the Discovery MR750)", "field_t": 3.0},
    coil={"channels": (8, 8), "name": "8HRBRAIN"},
    sequence={"name": "PRESS", "product": "jpress", "te_s": 0.035, "tr_s": 2.0,
              "simulated_as": "Press(te1_s=0.011, te2_s=0.024, tr_s=2.0)"},
    acquisition={"npoints": 2048, "bandwidth_hz": 2000.0, "transients": 64, "measured_on": 1,
                 "water_suppressed": True},
    voxel=NOISE_BIG_GABA_SIEMENS_3T.voxel,
    processing="P-files through spec2nii, noise-whitened SVD coil combination (weights from the "
               "mean transient), SNR per transient, median per subject",
    date="18/09/26",
    presets=("ge_mr750_3t",),
    note="the prior is the subjects' 5th to 95th percentile, apodized",
)


#****************************#
#   noise, ge, 32 channels   #
#****************************#
# serves the ge_mr750_32ch_3t preset: the MR750 with its 32-channel head array under sLASER,
# 13 subjects scanned twice (the sessions of a subject agree within a tenth). It reads 7.9
# where the 8-channel array above reads 7.0: the voxel lies deep on the midline, where many
# small loops gain little over few large ones, and the twin is an ideal sLASER, which keeps
# signal a played one loses (four adiabatic pulses at a body coil's peak B1), so part of the
# value is the sequence's and not the receive chain's (stated). The files keep the receiver's
# phase cycle in the stored transients, which scan.calibrate undoes.
NOISE_DS006444_GE_32CH_3T = Calibration(
    leaf="noise.sigma",
    value=7.9,
    prior=(6.9, 8.7),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over acquisitions",
              "median": 32.3, "implied_value": 8.56, "subjects": 13, "acquisitions": 26,
              "apodized": {"statistic": APODIZED, "median": 117.9, "implied_value": 7.32}},
    source={"dataset": "OpenNeuro ds006444, the sLASER scans of both sessions (CC0)",
            "citation": "doi 10.18112/openneuro.ds006444.v1.0.9",
            "url": "https://openneuro.org/datasets/ds006444"},
    scanner={"vendor": "ge", "model": "Discovery MR750", "software": "DV26.0_R01_1725.a",
             "field_t": 3.0},
    coil={"channels": (32,), "name": "32Ch Head"},
    sequence={"name": "sLASER", "family": "SemiLaser", "product": "slaser", "te_s": 0.035, "tr_s": 2.0,
              "simulated_as": "SemiLaser(te_s=0.035, tr_s=2.0)"},
    acquisition={"npoints": 4096, "bandwidth_hz": 5000.0, "transients": 64, "measured_on": 1,
                 "water_suppressed": True},
    voxel=NOISE_BIG_GABA_SIEMENS_3T.voxel,
    processing="NIfTI-MRS from P-files (spec2nii), the stored phase cycle undone, noise-whitened "
               "SVD coil combination, SNR per transient, median per acquisition",
    date="19/09/26",
    presets=("ge_mr750_32ch_3t",),
    note="the prior is the acquisitions' 5th to 95th percentile of the two readings' mean",
)


#********************#
#   noise, 7 tesla   #
#********************#
# serves the generic_7t preset, and is the first test of the field term of noise_scale
# (sigma over B0, a signal-to-noise ratio linear in the field). The value is in the units of
# the 3 T reference: were the law exact, a 32-channel array of the Prisma's generation
# would read 4.9 here too; it reads 3.6, which a field exponent of about 1.4 in place of 1
# would give (the literature's is 1.65: Pohmann et al. 2016, Magn Reson Med 75:801). One
# 7 T site cannot part the field from its coil, so the law stays linear and the record
# carries the difference (stated). Both readings within three percent.
NOISE_NEUROMET_7T = Calibration(
    leaf="noise.sigma",
    value=3.6,
    prior=(3.3, 5.1),
    measured={"statistic": "single-transient NAA SNR (Near et al. 2021: real NAA peak height over "
                           "the noise standard deviation of a signal-free stretch after a linear "
                           "detrend, no apodization), median over subjects",
              "median": 60.3, "implied_value": 3.66, "subjects": 9, "unit": "one add-subtract pair, two excitations",
              "apodized": {"statistic": APODIZED, "median": 218.3, "implied_value": 3.57}},
    source={"dataset": "NeuroMET SPECIAL MRS reproducibility, the first hyperbolic secant scan of "
                       "each subject's first session (CC BY-NC-SA 4.0)",
            "citation": "Riemann et al. 2022, Magn Reson Med 87:1119, doi 10.1002/mrm.29034",
            "url": "https://zenodo.org/record/5500320"},
    scanner={"vendor": "siemens", "model": "Magnetom 7T (Investigational_Device_7T)",
             "software": "syngo MR B17", "field_t": 6.98},
    coil={"channels": (32,), "name": "Nova Medical, birdcage transmit, 32 receive (the paper)"},
    sequence={"name": "SPECIAL", "product": "lr_2special (customer sequence)", "te_s": 0.009,
              "tr_s": 6.5, "simulated_as": "Press(te1_s=0.0045, te2_s=0.0045, tr_s=6.5) with "
              "averages=2: an ideal echo of that time, the pair as two averaged excitations"},
    acquisition={"npoints": 2048, "bandwidth_hz": 4000.0, "transients": 64, "measured_on": 2,
                 "water_suppressed": True},
    voxel={"region": "posterior cingulate cortex, on the midline (the paper)",
           "size_mm": (20.0, 20.0, 20.0), "simulated_brain": "icbm152",
           "simulated_position_mm": (0.0, -45.0, 35.0)},
    processing="raw TWIX streamed member by member from the archive, receiver oversampling "
               "removed, pairs summed, noise-whitened SVD coil combination, SNR per pair, "
               "median per subject",
    date="18/09/26",
    presets=("generic_7t",),
    note="the prior is the subjects' range, apodized; one site",
)


#*****************************#
#   array noise correlation   #
#*****************************#
# the share of a channel's noise that is its own, hardware.CoilArray.loaded(own_noise):
# read off the noise adjustment measurement in the COWS file, the array on a head, by
# scan.calibrate CovarianceReading. The tails of the FIDs of a Big GABA Trio subject (a
# 32-channel head array too, no noise adjustment in those files) give 0.27 the same way.
COIL_OWN_NOISE_PRISMA_3T = Calibration(
    leaf="coils.loaded.own_noise",
    value=0.25,
    prior=(0.25, 0.25),
    measured={"statistic": "magnitude of the noise correlation between channels: the mean over "
                           "channels of the strongest partner, the strongest pair, the median pair",
              "values": (0.41, 0.51, 0.15), "samples": 65536, "subjects": 1,
              "simulated_at_value": (0.41, 0.50, 0.10), "simulated_without": (0.61, 0.79, 0.13)},
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
#**************#
#   creatine   #
#**************#
# the de Graaf ranges give creatine 4.5 to 10.5 and phosphocreatine 3.0 to 5.5 mM; drawn side
# by side they put total creatine at 11.7 mM in the middle, as much as NAA plus NAAG, and the
# 3.03 ppm singlet at 0.76 to 0.85 of the NAA singlet's height where three datasets show 0.56
# to 0.62 (sequence, timings, and voxel matched; relaxation moves the ratio by 0.97 only,
# the creatine lines' greater width by 0.66). Matched, the data put total creatine at 0.8
# to 1.0 of NAA, about 8.5 mM, the cortical value of the literature. The creatine range is
# therefore the phosphocreatine one, 3.0 to 5.5: a total of 6 to 11 mM, half of it
# phosphorylated in the middle (the split hardly shows at 3 T; stated).
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


#***************#
#   catalogue   #
#***************#
# the calibrations in use. A record serves the scanner presets it names; the first record
# of a leaf is the fallback for every other scanner, and a scan's header
# says where it departs from the record it got (`Calibration.departures`)
CALIBRATIONS = (NOISE_BIG_GABA_SIEMENS_3T, NOISE_COWS_PRISMA_3T, NOISE_FMRS_PAIN_ACHIEVA_3T,
                NOISE_BIG_GABA_GE_3T, NOISE_DS006444_GE_32CH_3T, NOISE_NEUROMET_7T,
                CREATINE_THREE_DATASETS_3T)


def calibrations(scanner=None, records=()):
    """{leaf: Calibration} for a hardware.Scanner (or a preset name): the record that
    names its preset, else the leaf's fallback. `records` are the scanner's own
    (scan.profile) and outrank both."""
    name = getattr(scanner, "name", scanner)
    chosen = {}
    for record in CALIBRATIONS:
        chosen.setdefault(record.leaf, record)
    for record in CALIBRATIONS:
        if name in record.presets:
            chosen[record.leaf] = record
    chosen.update({record.leaf: record for record in records})
    return chosen


#***************#
#   linewidth   #
#***************#
# a check of the stated lineshape prior (scan.defaults lineshape.gauss_hz, 2 to 6 Hz at 3 T):
# the NAA singlet's width in the averaged spectra of the two Siemens datasets above against
# the simulated one. The prior covers both; it is not fitted to them.
LINEWIDTH_SIEMENS_3T = Calibration(
    leaf="lineshape.gauss_hz",
    value=4.0,
    prior=(2.0, 6.0),
    measured={"statistic": "full width at half maximum of the NAA singlet in the averaged, "
                           "coil-combined spectrum, eightfold zero filled, no apodization, in Hz",
              "big_gaba_siemens_median": 6.1, "cows_region_medians": (4.9, 5.6, 4.6),
              "simulated_median": 5.8, "simulated_range": (3.4, 7.5)},
    source={"dataset": "Big GABA sites S1, S5, S6 and COWS (the records above)",
            "citation": None, "url": None},
    scanner={"vendor": "siemens", "model": "TrioTim and Prisma", "software": None, "field_t": 2.89},
    coil={"channels": (32, 4), "name": None},
    sequence={"name": "PRESS", "family": "", "te_s": (0.035, 0.026)},
    acquisition={"transients": (64, 32), "measured_on": "the average"},
    voxel={"size_mm": ((30.0, 30.0, 30.0), (20.0, 20.0, 20.0))},
    processing="as the noise records, then the average over transients; no frequency alignment, "
               "so drift widens the measured line (stated)",
    date="18/09/26",
    note="a check, not a fit; other vendors' lines are not measured yet",
)

# a second 7 T site read the same way, kept as a check beside NeuroMET and not as a value:
# at an echo time of 69 ms the reading leans on the 7 T NAA T2 prior (a fifth off in T2 is a
# tenth in sigma), and the echo's split and the voxel's place are stand-ins. It reads 2.1
# where NeuroMET reads 3.6: quieter, by how much the stand-ins do not let one say.
NOISE_DS005917_7T = Calibration(
    leaf="noise.sigma",
    value=2.1,
    prior=(1.8, 2.8),
    measured={"statistic": NOISE_NEUROMET_7T.measured["statistic"], "median": 32.8,
              "implied_value": 2.04, "subjects": 25,
              "apodized": {"statistic": APODIZED, "median": 92.9, "implied_value": 2.18}},
    source={"dataset": "OpenNeuro ds005917, NIMH ketamine mechanism of action study, the baseline "
                       "sessions (CC0)", "citation": "doi 10.18112/openneuro.ds005917.v1.1.0",
            "url": "https://openneuro.org/datasets/ds005917"},
    scanner={"vendor": "siemens", "model": None, "software": "syngo MR B17", "field_t": 6.98},
    coil={"channels": (32,), "name": "32Ch_Head_7T"},
    sequence={"name": "PRESS", "product": "press_glx1 (customer sequence, J-suppressed)", "te_s": 0.069,
              "tr_s": 2.5, "simulated_as": "Press(te1_s=0.0345, te2_s=0.0345, tr_s=2.5), the split a "
              "stand-in"},
    acquisition={"npoints": 2048, "bandwidth_hz": 4000.0, "transients": 136, "measured_on": 1,
                 "water_suppressed": True},
    voxel={"region": "perigenual anterior cingulate", "size_mm": (20.0, 20.0, 20.0),
           "simulated_brain": "icbm152", "simulated_position_mm": (0.0, 40.0, 8.0)},
    processing="NIfTI-MRS from TWIX (spec2nii), noise-whitened SVD coil combination, SNR per "
               "transient, median per subject",
    date="19/09/26",
    note="a check: patients and controls at baseline; the place and the echo's split are stand-ins",
)

# records read the same way under other conditions, kept to test what carries a number there
CHECKS = (LINEWIDTH_SIEMENS_3T, NOISE_DS005917_7T)
# numbers of the hardware models, outside the Params tree
HARDWARE = {c.leaf: c for c in (COIL_OWN_NOISE_PRISMA_3T,)}
