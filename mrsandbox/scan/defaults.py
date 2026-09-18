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
from mrsandbox.database import Database, concentrations
from mrsandbox.database.downfield import OZDEMIR
from mrsandbox.database.macromolecules import CONSENSUS_T1
from mrsandbox.database.relaxation import BOUDREAU, WATER, WATER_FIELDS
from mrsandbox.params import Clipped, Normal, Params, PerTissue, Priors, Uniform


#****************#
#   field lock   #
#****************#
# the typed fields; a scan at any other field takes the nearest one (stated)
FIELDS_T = (1.5, 3.0, 7.0)

# the consensus MM T1 study per field beyond 3 T and the default components it covers
# (the only human rows the consensus table carries off 3 T; the rest keep the 3 T values)
MM_T1_STUDIES = {1.5: "kreis_2005", 7.0: "xin_2013"}
MM_T1_PEAKS = {"mm091": "M0.94", "mm121": "M1.22"}


#********************#
#   gm to wm ratio   #
#********************#
# constituents inherit their compound total's ratio, the documented approximation
_TISSUE_ROW = {"Ins": "mIns", "sIns": "sIno", "Cr": "tCr", "PCr": "tCr",
               "GPC": "tCho", "PCho": "tCho"}


def _gm_wm_ratio(name):
    """The challenge GM to WM concentration ratio of a metabolite; 1.0 where the
    table has no entry or a zero tissue value (stated fallback)."""
    gm, wm = concentrations.MRSI_CHALLENGE.get(_TISSUE_ROW.get(name, name), (1.0, 1.0))
    return gm / wm if gm > 0.0 and wm > 0.0 else 1.0


#**********************#
#   default catalogue  #
#**********************#
def default_params(database=None, b0_t=3.0):
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
    """
    db = database if database is not None else Database()
    tesla = min(FIELDS_T, key=lambda f: abs(f - float(b0_t)))
    names = [name for name in db.metabolites() if name in concentrations.DE_GRAAF]
    conc = Priors.de_graaf(metabolites=names, database=db)
    t2 = Priors.gudmundson_t2(names, tesla=tesla)  # pooled where the literature has rows
    water_t1, water_t2 = _water_relaxation(tesla)
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
        #     ppm per degree from 37 C, where the shifts are typed). Fixed at 37; a
        #     range is the user's belief about the subject (stated: the metabolites'
        #     own shifts, fourteen times less sensitive, stay).
        water=dict(residual_fraction=(1e-4, 3e-3), t1_s=water_t1, t2_s=water_t2,
                   temperature_c=37.0),
        # lipids: subcutaneous fat at the skull boundary.
        #   amplitude_mm: adipose protons are water-like (about 60 to 70 M), which
        #     is why a voxel clipping the skull drowns in fat; the range is stated.
        #   t1_s, t2_s: stated assumptions.
        lipids=dict(amplitude_mm=(30000.0, 70000.0), t2_s=0.05, t1_s=0.3),
        # mm: the macromolecule background.
        #   kappa: scales the challenge tissue ratios times the drawn creatine
        #     (the CoupledMM default). The challenge's own 0.4 normalizes its
        #     measured basis, not concentration units, so kappa is recalibrated
        #     (stated) to land the short-TE MM area fraction in the consensus
        #     10 to 25 percent ballpark.
        #   level_mm: drives the ParameterizedMM alternative.
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
        mm=dict(level_mm=(1.5, 4.0), kappa=(5.5, 9.5),
                t1={**dict(mm091=0.290, mm121=0.309, mm143=0.400, mm167=0.225,
                           mm226=0.263, mm321=0.400, mm377=0.400),
                    **_mm_t1(tesla)},
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
        #     broadening scales with the field, so the range scales by B0 / 3 T.
        lineshape=dict(gauss_hz=(2.0 * tesla / 3.0, 6.0 * tesla / 3.0)),
    )
    # instrumental and physiological imperfections and artifact signals
    imperfections = Params(
        # noise: complex Gaussian per FID point of ONE transient of the reference
        #   acquisition, a 20 mm cube (8 mL) at 3 T sampled at 5 kHz (scan.voxel
        #   NOISE_REFERENCE); every acquisition scales it by its volume, field,
        #   receiver bandwidth, and averages. Calibrated
        #   17/09/26 against one in vivo 3 T dataset (Siemens Prisma, 64-channel head
        #   coil, PRESS TE 35 ms, TR 2 s, a 30 mm cube, 4096 points at 5 kHz, 32
        #   transients with the coils kept: dataset 28 of the BasisREMY REMY test
        #   set, by its own NIfTI-MRS header): the single-transient NAA SNR by the
        #   consensus definition (Near et al. 2021: the real NAA peak height over the
        #   noise standard deviation of a signal-free region, here -5 to -1 ppm after
        #   a linear detrend, no apodization) is 6.7 +- 1.2 over the 32 transients
        #   after an SVD coil combination, 28.1 for their aligned average; the
        #   simulator at the same voxel, timings, and grid (ICBM mixed tissue, the
        #   default lineshape, whose NAA width of 6 to 7 Hz brackets the measured 5)
        #   reaches it at a reference sigma of 47 (median SNR 6.6 over 32 subjects,
        #   IQR 5.8 to 8.1, the spread being the concentration prior). Recalibrated
        #   18/09/26: the first calibration (sigma 5.2) read the simulated spectrum
        #   the way the NIfTI-MRS file of the dataset is read, conjugated and with a
        #   4.65 ppm carrier, while `Voxel.fid` is referenced to 0 ppm with a line at
        #   f hertz as exp(+i 2 pi f t); its "NAA" was the multiplets near 2.64 ppm,
        #   a ninth as tall, and the noise came out that much too low. The range
        #   around it is stated: 0.7 to 1.4 times, the site-to-site SNR scatter of
        #   multi-site 3 T studies (tests/scan/test_scan TestNoiseModel gates it).
        noise=dict(sigma=(33.0, 66.0)),
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
        #     readout's start (the first-order phase of the spectrum, exactly);
        #     zero by default, a range is the user's to set.
        acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0),
                         delay_s=0.0,
                         eddy_std_rad=(0.0, 0.6), eddy_lp_cut_hz=(10.0, 40.0),
                         eddy_seed=(0.0, 2.0 ** 31 - 1.0)),
        # gradients: the eddy currents the design's own gradients induce (scan.eddy),
        #   two exponential terms per step of the gradient, shared by the three
        #   axes (stated): a short one of a few to a hundred milliseconds and a long
        #   one of a fraction of a second to seconds (the residuals of a pre-emphasized
        #   system, de Graaf 2019 section 10.5; the constants are the PLAN's stated
        #   ranges, unverified against a measured response), each with a uniform
        #   B0 term in hertz per mT/m of step and a fraction of the step opposing it
        #   along the axis. Their integrals before the readout phase the coherences
        #   and unbalance the crusher pairs (scan.eddy.induced_sequence, a zero-order
        #   phase growing with the voxel's distance from the isocentre), over the
        #   readout they phase the FID (the Klose-type lineshape distortion) and
        #   dephase the voxel box; the ranges are stated so the strong end distorts
        #   a line visibly and the weak end is clean. A scan with a measured
        #   response (VirtualScan girf) plays its gradients through that instead
        #   and leaves these terms unread.
        gradients=dict(tau_short_s=(0.002, 0.100), tau_long_s=(0.3, 3.0),
                       b0_short_hz_per_mt_m=(0.0, 0.5), b0_long_hz_per_mt_m=(0.0, 0.2),
                       linear_short=(0.0, 0.005), linear_long=(0.0, 0.002)),
        # spurious: the out-of-voxel echo that imperfect crushing or cycling fails
        #   to remove, rephasing mid-acquisition (the mechanism is demonstrated by
        #   the crusher gates in tests/sequences). Form and every range are typed
        #   from the AGNOSTIC OOV synthesis (Gudmundson et al. 2023, Imaging
        #   Neuroscience, imag_a_00025): a Gaussian envelope echo
        #   a exp(-W (t - tau)^2) exp(i (2 pi f t + phi)), tau 10-400 ms,
        #   W 500-8000 1/s^2 (FWHM 18-74 ms), position 1-4 ppm, amplitude 0.1-20%
        #   of the FID maximum, free phase. The source adds one echo to 85% of its
        #   examples; here it is always on, and amplitude=0.0 switches it off. The
        #   echo is rendered by Augmentrum's SpuriousEchoes (scan.Voxel).
        spurious=dict(amplitude=(0.001, 0.2), tau_s=(0.010, 0.400),
                      width_s2=(500.0, 8000.0), ppm=(1.0, 4.0),
                      phase_deg=(0.0, 360.0)),
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
    #     0.2 to 0.5 range the consensus simulates.
    #   mm: the consensus's own range. water, lipids: stated.
    diffusion = Params(diffusion=dict(
        model="sticks",
        adc={name: PerTissue(gm=0.10, wm=0.15, csf=0.10) for name in names},
        water=0.75,
        mm=(0.005, 0.01),
        lipids=0.0))
    # contrast: the GM to WM concentration ratio per metabolite, typed from the
    # challenge Table 1 per-tissue values (database.concentrations MRSI_CHALLENGE;
    # compound totals lend their ratio to their constituents, metabolites without a
    # row stay at 1.0, both stated). The render splits the drawn concentration as
    # gm 2r/(1+r) + wm 2/(1+r), so an equal-mix voxel keeps exactly the drawn value
    # and pure tissues realize the published contrast.
    contrast = Params(contrast={name: _gm_wm_ratio(name) for name in names})
    relaxation = Params(t2=t2, t1=Priors.literature_t1(names, database=db, tesla=tesla))
    return Params(conc=conc) | relaxation | contrast | diffusion | background | imperfections


#**********************#
#   water relaxation   #
#**********************#
def _water_relaxation(tesla):
    """The per-tissue water T1 and T2 leaves of a typed field: the 3 T tables with the
    field's WATER_FIELDS entries replacing what they cover; a (mean, sd) pair draws as
    a Clipped Normal with the stated 0.2 s floor."""
    field = WATER_FIELDS.get(tesla, {})
    t1 = {"gm": BOUDREAU["t1_s"]["gm_cortex"], "wm": BOUDREAU["t1_s"]["wm_genu"],
          "csf": WATER["t1_s"]["csf"], **field.get("t1_s", {})}
    t1 = {tissue: (Clipped(Normal(*value), low=0.2) if isinstance(value, tuple) else value)
          for tissue, value in t1.items()}
    return PerTissue(**t1), PerTissue(**{**WATER["t2_s"], **field.get("t2_s", {})})


#*************************#
#   macromolecule t1      #
#*************************#
def _mm_t1(tesla):
    """The consensus MM T1 (seconds) of the components with a human row at the field,
    empty at 3 T and where the table has none."""
    study = MM_T1_STUDIES.get(tesla)
    if study is None:
        return {}
    return {component: CONSENSUS_T1["t1_ms"][peak][study][0] / 1000.0
            for component, peak in MM_T1_PEAKS.items()
            if study in CONSENSUS_T1["t1_ms"][peak]}
