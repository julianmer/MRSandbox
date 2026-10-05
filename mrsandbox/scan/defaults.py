####################################################################################################
#                                           defaults.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/09/26                                                                                #
#                                                                                                  #
# Purpose: The default prior catalogue of a VirtualScan: concentration and relaxation priors for   #
#          every metabolite the database and de Graaf both know, the background signal inherent    #
#          to the voxel (residual water, lipids, macromolecules; the Kreis 2021 terminology        #
#          consensus, NMR Biomed e4347, section 2.4), the diffusion attenuation, and the           #
#          instrumental imperfections (noise, transmit and receiver, spurious echo).               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

from mrsandbox.database import Database, concentrations
from mrsandbox.database.downfield import OZDEMIR
from mrsandbox.database.macromolecules import CONSENSUS_T1
from mrsandbox.database.tissue import DEELCHAND_2018
from mrsandbox.database.transfer import CREATINE_METHYL_RATE_HZ, NO_TRANSFER, STANISZ_2005_3T
from mrsandbox.database.gudmundson import PooledValue
from mrsandbox.database.relaxation import BOUDREAU, WATER, WATER_FIELDS, WATER_T2_3T
from mrsandbox.hardware.scanners import Scanner
from mrsandbox.scan.calibrations import calibrations
from mrsandbox.params import (Clipped, Empirical, Fixed, Mixture, Normal, Params, PerTissue, Priors,
                              Sourced, Uniform)


#****************#
#   field lock   #
#****************#
# the typed fields; a scan at any other field takes the nearest one (stated)
FIELDS_T = (1.5, 3.0, 7.0)
TISSUES = ("gm", "wm", "csf")
WATER_CITED = ("Boudreau et al. 2024 (the ISMRM T1 challenge, 3 T cortex and genu), "
               "Rooney et al. 2007 (CSF); T2 the median of five stimulated-echo-corrected "
               "studies at 3 T, their spread the range (the WATER_T2_3T card); at 1.5 and 7 T "
               "Ethofer et al. 2003, Xin et al. 2013, Marjanska et al. 2012 (database.relaxation)")
RUN_CITED = {"drift": "Hui et al. 2021, NeuroImage 241:118430 (Big Drift): about 0.3 Hz a minute "
                      "at 3 T, either sign; Augmentrum's envelope",
             "breathing": "van de Moortele et al. 2002, Magn Reson Med 47:888: breathing at about "
                          "0.3 Hz modulates the field; the range stated",
             "jitter": "Near et al. 2015, Magn Reson Med 73:44: the phase partly follows the "
                       "frequency; the spreads stated",
             "events": "Rowland et al. 2017 and Hess et al. 2011: motion as sparse events; "
                       "the rate stated",
             "stated": "a stated range (Augmentrum's TransientSynthesizer default)"}
TRANSFER_CITED = {"creatine": "MacMillan, Boesch, Kreis 2013, MRM 70:916: water to creatine "
                                "methyl 0.39 and 0.30 per s in two human muscles at 3 T "
                                "(database.transfer)",
                  "none": "no transfer to NAA, NAAG or choline in human brain: Kruiskamp et al. "
                          "2001, Helms and Frahm 1999, Meyerhoff 1999 (database.transfer)"}
CREATINE = ("de Graaf, In Vivo NMR Spectroscopy, 3rd ed.: total creatine 4.5 to 10.5 mM with the "
            "phosphocreatine 3.0 to 5.5 mM inside it taken out, the free creatine "
            "(database.concentrations DE_GRAAF)")

# the consensus MM T1 study per field beyond 3 T and the default components it covers
# (the only human rows the consensus table carries off 3 T; the rest keep the 3 T values)
MM_T1_STUDIES = {1.5: "kreis_2005", 7.0: "xin_2013"}
MM_T1_PEAKS = {"mm091": "M0.94", "mm121": "M1.22"}


#***************#
#   adc ratio   #
#***************#
def _adc_ratio(name):
    """A metabolite's measured ADC over the mean of the five measured ones (Deelchand
    et al. 2018); one for a metabolite without a row."""
    table = DEELCHAND_2018["adc_um2_ms"]
    for total, members in DEELCHAND_2018["members"].items():
        if name in members:
            return table[total] / (sum(table.values()) / len(table))
    return 1.0


#********************#
#   gm to wm ratio   #
#********************#
# constituents inherit their compound total's ratio, the documented approximation
_TISSUE_ROW = {"Ins": "mIns", "sIns": "sIno", "Cr": "tCr", "PCr": "tCr",
               "GPC": "tCho", "PCho": "tCho"}


#**********************#
#   default catalogue  #
#**********************#
def default_params(database=None, b0_t=3.0, scanner=None, records=(), channels=None,
                   subject=None):
    """Concentration priors and pooled literature relaxation for every metabolite that
    has both a spin system and a de Graaf range. Total choline is covered by GPC and
    PCho, its two constituents, so there is no separate Cho entry.

    The relaxation priors follow the scanner field: metabolite T2 pooled from the
    Gudmundson database at the nearest typed field (1.5, 3, or 7 T), metabolite T1
    from the typed tables of that field (Database.T1_SOURCES), water and the
    macromolecule T1 from the per-field cards, each falling back to the 3 T value
    where a field has no rows (stated). Every leaf stays a belief the caller or the
    closed loop overrides at will.

    Args:
        database: Database instance; defaults to the packaged one.
        b0_t: The scanner field in tesla.
        scanner: The hardware.Scanner (or its preset name); a leaf calibrated on in vivo
            data takes the package's readings pooled under the scanner's conditions, else
            the typed record that serves the preset, else the leaf's fallback
            (scan.calibrations); without a scanner, a nameless one at `b0_t`.
        records: The scanner's own Calibrations (scan.profile), which outrank those.
        channels: The receive array's count, one of the conditions the readings are pooled by.
        subject: The person's traits the readings they carry are pooled by (scan.cohort
            pooled): "age_y", "sex", "health"; None is a healthy adult.
    """
    db = database if database is not None else Database()
    tesla = min(FIELDS_T, key=lambda f: abs(f - float(b0_t)))
    names = [name for name in db.metabolites() if name in concentrations.DE_GRAAF]
    conc = Priors.de_graaf(metabolites=names, database=db)
    # creatine: de Graaf's 4.5 to 10.5 mM is the total creatine and his 3.0 to 5.5 the
    # phosphocreatine inside it; stacked on the PCr leaf the total range would count the
    # phosphocreatine twice, so the creatine leaf's literature is the total with the
    # phosphocreatine taken out, 3.0 to 5.5 in the ranges' own numbers (a total of 6 to 11 mM);
    # the measured source is scan.calibrations CREATINE_THREE_DATASETS_3T
    served = calibrations(scanner if scanner is not None else Scanner("unnamed", float(b0_t)),
                          records, channels, subject)  # every leaf a record serves
    if "Cr" in conc:
        conc = conc | Params(Cr=_sourced("conc.Cr", served, Uniform, literature=Uniform(3.0, 5.5),
                                         cited=CREATINE))
    t2 = Priors.gudmundson_t2(names, tesla=tesla)  # pooled where the literature has rows
    literature_t1, literature_t2 = _water_relaxation(tesla)
    # the 3 T T2 studies each a voice of the mixed source, where their range is the literature
    # (at 1.5 T too, the 3 T values reused, stated); each drawn about its value with its own
    # spread between subjects, else the spread pooled over the studies that print one
    studies = {tissue: tuple(
        Clipped(Normal(value, sd if sd is not None else PooledValue.within(rows)), low=0.02)
        for value, sd, _ in rows) for tissue, rows in WATER_T2_3T.items()
        if "t2_s" not in WATER_FIELDS.get(tesla, {})}
    water_t1 = _per_tissue("water.t1_s", served, literature_t1)
    water_t2 = _per_tissue("water.t2_s", served, literature_t2, studies)
    # background signal inherent to the voxel: residual water, lipids, and the
    # macromolecules are signal, never baseline or nuisance (Kreis et al. 2021)
    background = Params(
        # water: the residual left by suppression. Per-tissue leaves are weighted
        # by each voxel's composition at render time.
        #   residual_fraction: a stated assumption.
        #   t1_s, t2_s: per field (_water_relaxation): at 3 T the T1 draws from the
        #     crowdsourced reproducibility challenge (the BOUDREAU table: cortical
        #     GM, and the genu as the challenge's cleanest WM ROI; the
        #     intersubmission sd is a realistic site-to-site spread) and T2 is the
        #     cited WATER table; at 1.5 and 7 T the WATER_FIELDS cards (Ethofer,
        #     Xin, Marjanska) replace what they cover. Draws clip at a stated 0.2 s
        #     floor; CSF keeps the field-independent cited value.
        #   temperature_c: the tissue temperature; the water line moves against the
        #     metabolites by the thermometry calibration (tissue.CADY_1995, -1 / 94
        #     ppm per degree from 37 C, where the shifts are typed), and the metabolites
        #     take their shifts and couplings at the same drawn temperature
        #     (VirtualScan.systems, a quarter of a degree a step). Fixed at the scan's
        #     temperature; a range is the user's belief about the subject.
        #   adjusted_residual: what the scanner's water suppression adjustment leaves
        #     of the full water when a train is played (a record per scanner, read
        #     against the water references: scan.calibrations); the console's prescan
        #     lands the train there instead of on zero.
        water=dict(residual_fraction=(1e-4, 3e-3), t1_s=water_t1, t2_s=water_t2,
                   temperature_c=37.0,
                   adjusted_residual=_sourced("water.adjusted_residual", served, Uniform)),
        # lipids: subcutaneous fat at the skull boundary.
        #   amplitude_mm: adipose protons are water-like (about 60 to 70 M), which
        #     is why a voxel clipping the skull drowns in fat; the range is stated.
        #   t1_s, t2_s: stated assumptions.
        lipids=dict(amplitude_mm=(30000.0, 70000.0), t2_s=0.05, t1_s=0.3),
        # oov: the water outside the voxel that the localization leaks (the refocusing
        #   pulses' stopbands, the crushers' incomplete dephasing), one broad line at the
        #   water's frequency at the FID's start, read on the raw as the broad part of
        #   the residual (the package's cohorts, scan.cohort).
        #   fraction: of the full water at t = 0 after the echo's decay, the record's.
        #   t2star_s, offset_hz: its decay and its frequency from the water's, the 10th
        #     to 90th percentile read off the Trio cohort's subjects.
        #   phase_deg: against the voxel's own residual at t = 0, the Trio subjects'
        #     median and a stated spread.
        oov=dict(fraction=_sourced("oov.fraction", served, Uniform) or Fixed(0.0),
                 t2star_s=(0.015, 0.028),
                 offset_hz=(-3.2, 7.3), phase_deg=Normal(-153.0, 45.0)),
        # mm: the macromolecule background.
        #   kappa: scales the challenge tissue ratios times the drawn creatine
        #     (the CoupledMM default). The challenge's own 0.4 normalizes its
        #     measured basis, not concentration units, so kappa is retyped (stated)
        #     to where the chains read raw short-echo data through the twin beside
        #     the measured shape (the 1.2 to 1.7 ppm heights over NAA). It carries
        #     the factor 11.75 / 8.5 by which the calibrated creatine range lowers
        #     the middle of total creatine, so the background does not move with
        #     the creatine range.
        #   level_mm: the level of the measured shape (MeasuredMM) and of the Gaussian
        #     components (ParameterizedMM): a stated range, the readings of the package's
        #     cohorts (the 0.9 ppm peak over NAA through the twin) pooled beside it.
        #   t1, t2: per component, typed from the consensus Appendix S1 3 T human
        #     columns (T2: Landheer 2020, else Hoefemann 2020; T1: Hoefemann 2020,
        #     fit-limit rows excluded; the database.macromolecules CONSENSUS
        #     tables). Components map by peak identity (challenge M0.91 is
        #     consensus M0.94 and so on; positions vary 0.01 to 0.04 ppm between
        #     papers). Printed errors draw as Clipped Normals with a stated 5 ms
        #     floor; values without errors or without a 3 T entry stay fixed. Off
        #     3 T the T1 of the components with a human row at that field (Xin 2013
        #     at 7 T, Kreis 2005 at 1.5 T; M0.94 and M1.22 only) replaces the 3 T
        #     value; T2 has no human rows off 3 T and keeps them (stated).
        mm=dict(level_mm=_sourced("mm.level_mm", served, Uniform, literature=Uniform(1.5, 4.0),
                                  cited="a stated range: the level of the measured 3 T shape"),
                kappa=_sourced("mm.kappa", served, Uniform, literature=Uniform(20.1, 34.7),
                               cited="a stated range: the coupled chains' anchor"),
                t1={**dict(mm091=0.290, mm121=0.309, mm143=0.400, mm167=0.225,
                           mm226=0.263, mm321=0.400, mm377=0.400),
                    **{component: CONSENSUS_T1["t1_ms"][peak][MM_T1_STUDIES[tesla]][0] / 1000.0
                       for component, peak in MM_T1_PEAKS.items()
                       if MM_T1_STUDIES.get(tesla) in CONSENSUS_T1["t1_ms"][peak]}},
                t2=dict(mm091=Clipped(Normal(0.027, 0.002), low=0.005),
                        mm121=Clipped(Normal(0.039, 0.013), low=0.005),
                        mm143=Clipped(Normal(0.018, 0.004), low=0.005),
                        mm167=Clipped(Normal(0.017, 0.003), low=0.005),
                        mm226=0.024,
                        mm321=Clipped(Normal(0.018, 0.005), low=0.005),
                        mm377=0.030)),
        # downfield: the nine 6.83 to 8.49 ppm resonances (protein amides, the NAA
        # amide, ATP, homocarnosine; Ozdemir 2023 MRM 90:814, Povazan 2022 MRM
        # 87:1661, the database.downfield cards).
        #   level: per peak, the reported institutional units treated as
        #     concentration equivalent mM (stated).
        #   width_scale: every peak renders at its digitized 3 T basis width
        #     (database.downfield POVAZAN_WIDTHS, 14 to 32 Hz) times this drawn
        #     scale; the 20 percent spread is the subject scatter of the reported
        #     linewidths (12.0 +- 2.4 Hz, Povazan 2022 Table S2), clipped at a stated
        #     floor.
        #   presaturated: water presaturation transfers saturation onto the
        #     exchanging protons, and each peak retains R1 / (R1 + kex) with the
        #     MacMillan 2011 3 T medians. True matches the default suppressed-water
        #     world; set False for water-unsuppressed or metabolite-cycled
        #     acquisitions.
        downfield=dict(level={name: peak["level_iu"] for name, peak in OZDEMIR.items()},
                       width_scale=Clipped(Normal(1.0, 0.2), low=0.5),
                       presaturated=True),
        # lineshape: the microscopic field distribution below the field map's
        #   resolution (the map's own spread dephases the voxel through its kernel),
        #   a Gaussian envelope exp(-t^2 / TG^2) on every spin-system component
        #   (metabolites, water, lipids; the measured MM and downfield widths already
        #   contain it), the T2' term every synthesis peer carries.
        #   gauss_hz: its FWHM, a stated range that lands the default 3 T water
        #     linewidth in the excellent-to-good grades of Oz et al. 2021 (5 to 10 Hz,
        #     16 to 19 at 7 T) on top of the Lorentzian 1 / (pi T2); susceptibility
        #     broadening scales with the field, so the range scales by B0 / 3 T. The
        #     width read on the package's cohorts stands beside it as its measured source.
        lineshape=dict(gauss_hz=_sourced("lineshape.gauss_hz", served, Uniform,
                                         literature=Uniform(2.0 * tesla / 3.0, 6.0 * tesla / 3.0),
                                         cited="a stated range: the water linewidth grades of "
                                               "Oz et al. 2021 over the Lorentzian part")),
    )
    # instrumental and physiological imperfections and artifact signals
    imperfections = Params(
        # noise: complex Gaussian per FID point of ONE transient of the reference
        #   acquisition, a 20 mm cube (8 mL) at 3 T sampled at 5 kHz (scan.voxel
        #   NOISE_REFERENCE); every acquisition scales it by its volume, field,
        #   receiver bandwidth, and averages. The range is calibrated on raw in vivo
        #   data, coils and transients apart, and belongs to the receive chain: the
        #   package's cohorts hold the readings with their exact conditions, pooled per
        #   scanner (scan.cohort), scan.calibrate makes such a record from a scanner's
        #   own data, and every acquisition's header names the record it got and the
        #   conditions it departs from. `Voxel.fid` is referenced to 0 ppm, a line at
        #   f hertz is exp(+i 2 pi f t): the plain transform puts NAA at 2.01 ppm.
        #   tests/scan/test_scan TestNoiseModel gates it.
        noise=dict(sigma=_sourced("noise.sigma", served, Uniform)),
        # b1_scale: every pulse flip is multiplied by the drawn transmit scale;
        #   a few percent calibration scatter at 3 T is a stated assumption.
        b1_scale=Normal(1.0, 0.03),
        # acquisition: receiver-side imperfections, stated assumptions.
        #   shift_hz, phase_deg: scanner frequency error (which also runs the
        #     sequence off resonance) and an uncorrected zero-order phase.
        #   eddy_std_rad, eddy_lp_cut_hz, eddy_seed: the residual eddy-current
        #     phase is Augmentrum's synthetic trajectory (EddyCurrent: low-pass
        #     filtered Gaussian phase noise with its linear trend removed, its
        #     defaults 0.6 rad and 25 Hz as the range ceilings, stated); the seed
        #     is drawn so the trajectory regenerates from the truth.
        #   delay_s: the receiver delay, every sample taken that long after the
        #     readout's start (the first-order phase of the spectrum, exactly): a
        #     record per scanner where one was read (scan.calibrations; negative when
        #     the first sample comes before the echo top), zero for a scanner without
        #     one.
        #   eddy_transient_rad, eddy_transient_tau_s: the phase transient at the
        #     FID's start that the water references show (the package's cohorts,
        #     scan.cohort), as the phase 5 ms after the first sample, decaying as
        #     exp(-t / tau) from there and grown back to the start: a Normal about
        #     zero whose sd is the record's, zero for a scanner without one; the time
        #     constant a stated range.
        acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0),
                         delay_s=_sourced("acquisition.delay_s", served, Uniform) or Fixed(0.0),
                         eddy_std_rad=(0.0, 0.6), eddy_lp_cut_hz=(10.0, 40.0),
                         eddy_seed=(0.0, 2.0 ** 31 - 1.0),
                         eddy_transient_rad=_sourced("acquisition.eddy_transient_rad", served,
                                                     _transient_prior) or Fixed(0.0),
                         eddy_transient_tau_s=(0.005, 0.02)),
        # gradients: the eddy currents the design's own gradients induce (scan.eddy),
        #   two exponential terms per step of the gradient, shared by the three
        #   axes (stated): a short one of a few to a hundred milliseconds and a long
        #   one of a fraction of a second to seconds (the residuals of a pre-emphasized
        #   system, de Graaf 2019 section 10.5; the constants are stated ranges,
        #   unverified against a measured response), each with a uniform
        #   B0 term in hertz per mT/m of step and a fraction of the step opposing it
        #   along the axis. Their integrals before the readout phase the coherences
        #   and unbalance the crusher pairs (scan.eddy.induced_sequence, a zero-order
        #   phase growing with the voxel's distance from the isocentre), over the
        #   readout they phase the FID (the Klose-type lineshape distortion) and
        #   dephase the voxel box; the ranges are stated so the strong end distorts
        #   a line visibly and the weak end is clean; the B0 terms carry either sign,
        #   as the eddy phase on the cohorts' water references does (those rows set
        #   the acquisition's phase transient, which these terms cannot reach at any
        #   physical size). A scan with a measured response (VirtualScan girf) plays
        #   its gradients through that instead and leaves these terms unread.
        gradients=dict(tau_short_s=(0.002, 0.100), tau_long_s=(0.3, 3.0),
                       b0_short_hz_per_mt_m=(-0.5, 0.5), b0_long_hz_per_mt_m=(-0.2, 0.2),
                       linear_short=(0.0, 0.005), linear_long=(0.0, 0.002)),
        # spurious: the out-of-voxel echo that imperfect crushing or cycling fails
        #   to remove, rephasing mid-acquisition (the mechanism is demonstrated by
        #   the crusher gates in tests/sequences). The form is the AGNOSTIC OOV
        #   synthesis (Gudmundson et al. 2023, Imaging Neuroscience, imag_a_00025):
        #   a Gaussian envelope echo a exp(-W (t - tau)^2) exp(i (2 pi f t + phi)),
        #   rendered by Augmentrum's SpuriousEchoes (scan.Voxel), its amplitude
        #   relative to the maximum of the voxel's FID without its water. The ranges
        #   are read off raw data with the transients apart (Big GABA PRESS and COWS
        #   sLASER runs; an echo is a late bump of the residual that the average does
        #   not carry). A run's echo keeps its timing and place, so tau, width and
        #   ppm are drawn once per subject and exam, and the transients carry it or
        #   not (scan.Multiplex.transients): rate is the share of transients that
        #   carry it, spread the half-width of a carrying transient's size about the
        #   run's amplitude (a factor of 1 - spread to 1 + spread), amplitude the
        #   run's typical size, tau and the width (W = 4 ln 2 / FWHM^2) per run, and
        #   the phase free. The place is the lipid band (0.9 to 1.7 ppm), where most
        #   runs put it; echoes elsewhere are not modelled. amplitude=0.0 switches
        #   the echo off; a single acquisition carries it whole.
        spurious=dict(amplitude=(0.01, 0.07), tau_s=(0.020, 0.150),
                      width_s2=(2000.0, 12000.0), ppm=(0.9, 1.7),
                      phase_deg=(0.0, 360.0),
                      rate=Clipped(Normal(0.15, 0.3), low=0.0, high=1.0),
                      spread=(0.2, 0.4)),
        # transients: the instabilities across repeated transients (drift, breathing,
        #   jitter, motion) come from Augmentrum's TransientSynthesizer, configured
        #   per acquisition in scan.Multiplex.transients, not from this tree.
    )
    # diffusion: every component attenuates with the b-value the sequence's own
    # gradient events build up (sequences.diffusion; consensus Box 2 of Ligneul
    # et al. 2024, MRM 91:860), so a standard sequence is only as diffusion
    # weighted as its crushers make it. ADCs in um^2/ms:
    #   model: the metabolite decay law. "sticks" is the consensus minimal model
    #     (section 5.1: randomly oriented infinitely long cylinders of zero
    #     radius; the radius needs the gradient timings the b-value alone does not
    #     carry, stated), whose powder average is non-monoexponential above b of
    #     3 to 5 ms/um^2 as measured; "gaussian" is the monoexponential exp(-b ADC)
    #     representation. Water, lipids, and the macromolecules stay Gaussian.
    #   adc: metabolites follow the human long-diffusion-time summary the
    #     consensus defers to (Valette et al. 2018, Front Neurosci 12:2: about
    #     0.10 in grey and 0.15 in white matter, weighted per voxel); under the
    #     sticks model this low-b ADC is D_intra / 3 (consensus section 5.1), so
    #     the intracellular diffusivity is 3 ADC, 0.3 to 0.45 um^2/ms, inside the
    #     0.2 to 0.5 range the consensus simulates. The five major metabolites
    #     scale both tissues by their measured ADC over the five's mean (Deelchand
    #     et al. 2018, database.tissue DEELCHAND_2018: Glu fastest at 1.17, mIns
    #     slowest at 0.86); the tissue contrast per metabolite is not measured, so
    #     one ratio serves grey and white matter, and the others stay at 1 (stated).
    #   water: per tissue, each water part fading with its own (phantom.water): the
    #     tissues' stated; the CSF's is free water at 37 C, 3.02 um^2/ms on the
    #     Arrhenius line through Holz, Heil and Sacco 2000 (PCCP 2:4740: 2.895 at
    #     35 C, 3.222 at 40 C), which the crushers of an editing design fade as the
    #     water references of Big GABA's Philips series fade.
    #   mm: the consensus's own range. lipids: stated.
    diffusion = Params(diffusion=dict(
        model="sticks",
        adc={name: PerTissue(gm=0.10 * _adc_ratio(name), wm=0.15 * _adc_ratio(name),
                             csf=0.10 * _adc_ratio(name)) for name in names},
        water=PerTissue(gm=0.75, wm=0.75, csf=3.0),
        mm=(0.005, 0.01),
        lipids=0.0))
    # contrast: the GM to WM concentration ratio per metabolite, typed from the
    # challenge Table 1 per-tissue values (database.concentrations MRSI_CHALLENGE;
    # compound totals lend their ratio to their constituents, metabolites without a
    # row stay at 1.0, both stated). The render splits the drawn concentration as
    # gm 2r/(1+r) + wm 2/(1+r), so an equal-mix voxel keeps exactly the drawn value
    # and pure tissues realize the published contrast.
    rows = {name: concentrations.MRSI_CHALLENGE.get(_TISSUE_ROW.get(name, name), (1.0, 1.0))
            for name in names}
    # run: what happens over the transients of a run (Augmentrum's TransientSynthesizer plays
    #   them, scan.Multiplex): the scanner's frequency drift (Hui et al. 2021: about 0.3 Hz a
    #   minute at 3 T, either sign), its slow wander (an AR process, a stated spread), the
    #   subject's breathing (its rate, and the field modulation it makes, van de Moortele et al.
    #   2002), the receiver's phase and amplitude jitter (Near et al. 2015), broadening over the
    #   run, motion events (Rowland et al. 2017) and how far the head moves in one (zero: the
    #   events kick the frequency, phase and amplitude alone). Each takes the package's readings
    #   where a dataset or a scan of the operator's gives them (scan.run: breathing, motion; the
    #   spectra's transients: the rest).
    run = Params(run=dict(
        drift_hz_per_min=_sourced("run.drift_hz_per_min", served, Uniform,
                                  literature=Normal(0.0, 0.3), cited=RUN_CITED["drift"]),
        ar_sigma_hz=_sourced("run.ar_sigma_hz", served, Uniform,
                             literature=Uniform(0.05, 0.3), cited=RUN_CITED["stated"]),
        resp_freq_hz=_sourced("run.resp_freq_hz", served, Uniform,
                              literature=Uniform(0.2, 0.35), cited=RUN_CITED["breathing"]),
        resp_amp_hz=_sourced("run.resp_amp_hz", served, Uniform,
                             literature=Uniform(0.05, 0.3), cited=RUN_CITED["breathing"]),
        phase_jitter_deg=_sourced("run.phase_jitter_deg", served, Uniform,
                                  literature=Uniform(0.5, 3.0), cited=RUN_CITED["jitter"]),
        amp_jitter_frac=_sourced("run.amp_jitter_frac", served, Uniform,
                                 literature=Uniform(0.005, 0.02), cited=RUN_CITED["jitter"]),
        broaden_hz=_sourced("run.broaden_hz", served, Uniform,
                            literature=Uniform(0.0, 2.0), cited=RUN_CITED["stated"]),
        events_per_min=_sourced("run.events_per_min", served, Uniform,
                                literature=Uniform(0.1, 0.5), cited=RUN_CITED["events"]),
        motion_mm=_sourced("run.motion_mm", served, Uniform) or Fixed(0.0)))
    contrast = Params(contrast={name: gm / wm if gm > 0.0 and wm > 0.0 else 1.0
                                for name, (gm, wm) in rows.items()})
    relaxation = Params(t2=t2, t1=Priors.literature_t1(names, database=db, tesla=tesla))
    # magnetization transfer (phantom.transfer): every tissue's bound pool (Stanisz et al. 2005 at
    # 3 T, the one field typed, stated elsewhere; CSF holds none) and the rate at which each
    # metabolite's protons exchange magnetization with it, creatine's literature the muscle rates,
    # the others' none; the measured source the package's readings of Bacon et al.'s spectra
    # (scan.calibrations TRANSFER_BACON_2026)
    pool = STANISZ_2005_3T
    transfer = Params(mt=dict(
        **{key: PerTissue(gm=pool["gm"][key], wm=pool["wm"][key], csf=0.0 if key != "t2_s"
                          else pool["wm"][key])
           for key in ("pool_fraction", "exchange_hz", "t2_s")},
        rate={name: _sourced(f"mt.rate.{name}", served, Uniform,
                             literature=(Uniform(*CREATINE_METHYL_RATE_HZ)
                                         if name not in NO_TRANSFER else Fixed(0.0)),
                             cited=TRANSFER_CITED["none" if name in NO_TRANSFER else "creatine"])
              for name in ("Cr", "PCr", *NO_TRANSFER) if name in names}))
    tree = (Params(conc=conc) | relaxation | contrast | diffusion | background | imperfections | run
            | transfer)
    # a scanner's own record of any leaf of the tree is its own source (the operator's scan) or its
    # dataset source (a cohort subject's reading), chosen: a reading made on a given scan outranks
    # the pool; a record of the hardware outside the tree (the array's noise share) is the
    # hardware's to read
    for record in records:
        *branch, leaf = record.leaf.split(".")
        nodes = [tree]
        for key in branch:
            nodes.append(nodes[-1].get(key, {}))
        node = nodes[-1]
        measured = (Uniform(*record.prior) if record.prior[0] < record.prior[1]
                    else Fixed(record.value))
        # a cohort's subject names itself (Member.fixed)
        slot = "dataset" if str(record.source.get("dataset", "")).startswith("subject ") else "own"
        if leaf in TISSUES and isinstance(node, (PerTissue, Sourced)):
            # one tissue of a PerTissue leaf (a map read per tissue, scan.maps): the slot's
            # PerTissue takes the record's tissue, its others as the leaf gives them (a tissue
            # already read into the slot stays)
            held = getattr(node, slot, None) if isinstance(node, Sourced) else None
            measured = dataclasses.replace(
                held if held is not None else node.current if isinstance(node, Sourced) else node,
                **{leaf: measured})
            leaf, node = branch.pop(), nodes[-2]
        if leaf in node:
            told = {slot: _cited(record)}
            sourced = (dataclasses.replace(node[leaf], **{slot: measured}, chosen=slot,
                                           provenance={**node[leaf].provenance, **told})
                       if isinstance(node[leaf], Sourced)
                       else Sourced(literature=node[leaf], **{slot: measured}, chosen=slot,
                                    provenance={"literature": "the package's stated prior",
                                                **told}))
            override = {leaf: sourced}
            for key in reversed(branch):
                override = {key: override}
            tree = tree | Params(override)
    return tree


#***************#
#   transient   #
#***************#
def _transient_prior(low, high):
    """The eddy transient's prior from its record: a population's record spans zero and is
    a Normal about it with that spread; one subject's own lies on a side and is its range."""
    return Normal(0.0, high) if low < 0.0 < high else Uniform(low, high)


#*************#
#   sourced   #
#*************#
def _sourced(leaf, served, make, literature=None, cited=None):
    """A Sourced leaf: `literature` its published value (None where none is published: a
    zero for want of a number is not literature) with `cited` as its provenance, the
    pooled source made by `make` from the record that serves the leaf, its provenance the
    record's dataset and date. The literature is chosen wherever it exists, the pooled one
    otherwise; a leaf with neither is None."""
    record = served.get(leaf)
    if literature is None and record is None:
        return None
    both = literature is not None and record is not None
    provenance = {**({"literature": cited} if literature is not None else {}),
                  **({"pooled": _cited(record)} if record is not None else {}),
                  **({"mixed": f"{cited}; {_cited(record)}: each an equal voice"} if both else {})}
    return Sourced(literature=literature, pooled=None if record is None else _spread(record, make),
                   mixed=_mixed((literature,), record, make) if both else None,
                   chosen="literature" if literature is not None else "pooled",
                   provenance=provenance)


def _spread(record, make):
    """The Value a record's population is drawn from: a pooled record (scan.cohort) carries
    its rows' values and their weights, and every draw is one subject's own, its dataset
    drawn by its share (the rows fall into sites and vendors, not about one middle); any
    other record gives `make` of its prior."""
    measured = record.measured if isinstance(record.measured, dict) else {}
    values = measured.get("values")
    return Empirical(values, measured.get("weights")) if values else make(*record.prior)


def _mixed(studies, record, make):
    """The literature and the measured datasets as equal voices (a Mixture): each study of
    `studies` (the literature's Values, one per study, or the literature as one) and each
    dataset of the pooled `record`, its own subjects; a record that is no pool is one voice,
    `make` of its prior."""
    measured = record.measured if isinstance(record.measured, dict) else {}
    if not measured.get("values"):
        return Mixture(tuple(studies) + (make(*record.prior),))
    by_dataset = {}
    for value, dataset in zip(measured["values"], measured["datasets"]):
        by_dataset.setdefault(dataset, []).append(value)
    return Mixture(tuple(studies) + tuple(Empirical(values) for values in by_dataset.values()))


def _cited(record):
    return f"{record.source['dataset']}, {record.date}"


#****************#
#   per tissue   #
#****************#
def _per_tissue(leaf, served, literature, studies=None):
    """A PerTissue leaf with its pooled source where the package's rows read it per tissue (a
    quantitative map on its subject's tissue maps, scan.maps): `leaf`.gm, .wm and .csf each
    pooled, a tissue without rows as the literature gives it; its mixed source the
    literature's studies (`studies`, {tissue: Values, one per study}, else the literature as
    one) and the datasets as equal voices; the literature chosen. Without rows the
    literature alone."""
    records = {tissue: served.get(f"{leaf}.{tissue}") for tissue in TISSUES}
    if not any(records.values()):
        return literature
    pooled = PerTissue(**{tissue: getattr(literature, tissue) if record is None
                          else _spread(record, Uniform) for tissue, record in records.items()})
    mixed = PerTissue(**{tissue: getattr(literature, tissue) if record is None
                         else _mixed((studies or {}).get(tissue, (getattr(literature, tissue),)),
                                     record, Uniform)
                         for tissue, record in records.items()})
    held = "; ".join(f"{tissue}: {_cited(record)}" for tissue, record in records.items() if record)
    return Sourced(literature=literature, pooled=pooled, mixed=mixed, chosen="literature",
                   provenance={"literature": WATER_CITED, "pooled": held,
                               "mixed": f"the literature's studies and {held}, each an equal "
                                        "voice"})


#**********************#
#   water relaxation   #
#**********************#
def _water_relaxation(tesla):
    """The per-tissue water T1 and T2 leaves of a typed field: the 3 T tables with the
    field's WATER_FIELDS entries replacing what they cover; a (mean, sd) pair draws as
    a Clipped Normal with the stated floor, 0.2 s for T1, 0.02 s for T2 (no brain water
    lies near either). The 3 T grey and white T2 are the reference range of the
    WATER_T2_3T studies (database.gudmundson PooledValue.reference)."""
    field = WATER_FIELDS.get(tesla, {})
    t1 = {"gm": BOUDREAU["t1_s"]["gm_cortex"], "wm": BOUDREAU["t1_s"]["wm_genu"],
          "csf": WATER["t1_s"]["csf"], **field.get("t1_s", {})}
    t1 = {tissue: (Clipped(Normal(*value), low=0.2) if isinstance(value, tuple) else value)
          for tissue, value in t1.items()}
    t2 = {**{tissue: tuple(PooledValue.reference(rows)[:2])
             for tissue, rows in WATER_T2_3T.items()},
          **WATER["t2_s"], **field.get("t2_s", {})}
    t2 = {tissue: (Clipped(Normal(*value), low=0.02) if isinstance(value, tuple) else value)
          for tissue, value in t2.items()}
    return PerTissue(**t1), PerTissue(**t2)
