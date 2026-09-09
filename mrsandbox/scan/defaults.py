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
#          instrumental imperfections (noise, transmit and receiver, spurious echo, transient      #
#          instabilities).                                                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.database import Database, concentrations
from mrsandbox.database.downfield import OZDEMIR
from mrsandbox.database.relaxation import BOUDREAU, WATER
from mrsandbox.params import Clipped, Normal, Params, PerTissue, Priors, Uniform


#**********************#
#   default catalogue  #
#**********************#
def default_params(database=None):
    """Concentration priors and pooled literature relaxation for every metabolite that
    has both a spin system and a de Graaf range. Total choline is covered by GPC and
    PCho, its two constituents, so there is no separate Cho entry.
    """
    db = database if database is not None else Database()
    names = [name for name in db.metabolites() if name in concentrations.DE_GRAAF]
    conc = Priors.de_graaf(metabolites=names, database=db)
    t2 = Priors.gudmundson_t2(names)  # pooled where the literature has rows, Fixed fallback
    # background signal inherent to the voxel: residual water, lipids, and the
    # macromolecules are signal, never baseline or nuisance (Kreis et al. 2021)
    background = Params(
        # water: the residual left by suppression. Per-tissue leaves are weighted
        # by each voxel's composition at render time.
        #   residual_fraction: a stated assumption.
        #   t1_s: drawn from the crowdsourced 3 T reproducibility challenge (the
        #     BOUDREAU table: cortical GM, and the genu as the challenge's
        #     cleanest WM ROI; the intersubmission sd is a realistic site-to-site
        #     spread), clipped at a stated 0.2 s floor. CSF keeps the cited WATER
        #     value.
        #   t2_s: the cited WATER table.
        water=dict(residual_fraction=(1e-4, 3e-3),
                   t1_s=PerTissue(
                       gm=Clipped(Normal(*BOUDREAU["t1_s"]["gm_cortex"]), low=0.2),
                       wm=Clipped(Normal(*BOUDREAU["t1_s"]["wm_genu"]), low=0.2),
                       csf=WATER["t1_s"]["csf"]),
                   t2_s=PerTissue(**WATER["t2_s"])),
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
        #     floor; values without errors or without a 3 T entry stay fixed.
        mm=dict(level_mm=(1.5, 4.0), kappa=(5.5, 9.5),
                t1=dict(mm091=0.290, mm121=0.309, mm143=0.400, mm167=0.225,
                        mm226=0.263, mm321=0.400, mm377=0.400),
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
        #   gauss_hz: a stated common width, pending the per-peak basis values of
        #     the Povazan supporting information; the observed 3 T linewidths were
        #     about 10 Hz and the amide region is broader.
        #   presaturated: water presaturation transfers saturation onto the
        #     exchanging protons, and each peak retains R1 / (R1 + kex) with the
        #     MacMillan 2011 3 T medians. True matches the default suppressed-water
        #     world; set False for water-unsuppressed or metabolite-cycled
        #     acquisitions.
        downfield=dict(level={name: peak["level_iu"] for name, peak in OZDEMIR.items()},
                       gauss_hz=(10.0, 20.0),
                       presaturated=True),
    )
    # instrumental and physiological imperfections and artifact signals
    imperfections = Params(
        # noise: complex Gaussian per FID point, acquisition side; range stated.
        noise=dict(sigma=(0.5, 3.0)),
        # b1_scale: every pulse flip is multiplied by the drawn transmit scale;
        #   a few percent calibration scatter at 3 T is a stated assumption.
        b1_scale=Normal(1.0, 0.03),
        # acquisition: receiver-side imperfections, stated assumptions.
        #   shift_hz, phase_deg: scanner frequency error and an uncorrected
        #     zero-order phase.
        #   eddy_deg, eddy_tau_s: a residual eddy current as a decaying phase on
        #     the FID (the leading term of the multi-exponential gradient response,
        #     de Graaf 2019; in vivo the bulk is removed by Klose 1990 ECC, so the
        #     residual stays modest).
        acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0),
                         eddy_deg=Uniform(-30.0, 30.0), eddy_tau_s=(0.01, 0.1)),
        # spurious: the out-of-voxel echo that imperfect crushing or cycling fails
        #   to remove, rephasing mid-acquisition (the mechanism is demonstrated by
        #   the crusher gates in tests/sequences). Form and every range are typed
        #   from the AGNOSTIC OOV synthesis (Gudmundson et al. 2023, Imaging
        #   Neuroscience, imag_a_00025): a Gaussian envelope echo
        #   a exp(-W (t - tau)^2) exp(i (2 pi f t + phi)), tau 10-400 ms,
        #   W 500-8000 1/s^2 (FWHM 18-74 ms), position 1-4 ppm, amplitude 0.1-20%
        #   of the FID maximum, free phase. The source adds one echo to 85% of its
        #   examples; here it is always on, and amplitude=0.0 switches it off.
        spurious=dict(amplitude=(0.001, 0.2), tau_s=(0.010, 0.400),
                      width_s2=(500.0, 8000.0), ppm=(1.0, 4.0),
                      phase_deg=(0.0, 360.0)),
        # transients: scanner and subject instabilities across repeated transients.
        #   drift_hz_per_min: grounded in Hui et al. 2021 (NeuroImage 241:118430,
        #     80 sites, 99 scanners at 3 T): median 0.29 Hz/min before fMRI
        #     heating with an IQR-derived spread, 0.43 after.
        #   jitter, phase, amplitude, motion: the shot-to-shot scatter and the
        #     motion event model are stated assumptions, every leaf overridable.
        transients=dict(
            drift_hz_per_min=Normal(0.29, 0.18),
            jitter_hz=0.2,
            phase_jitter_deg=4.0,
            amplitude_sd=0.02,
            motion_rate_per_min=(0.0, 0.5),
            motion_shift_hz=(1.0, 8.0),
            motion_phase_deg=(-45.0, 45.0),
        ),
    )
    # diffusion: every component fades by exp(-b ADC), with b built from the
    # sequence's own gradient events (sequences.diffusion; consensus Box 2 of
    # Ligneul et al. 2024, MRM 91:860), so a standard sequence is only as
    # diffusion weighted as its crushers make it. ADCs in um^2/ms:
    #   adc: metabolites follow the human long-diffusion-time summary the
    #     consensus defers to (Valette et al. 2018, Front Neurosci 12:2: about
    #     0.10 in grey and 0.15 in white matter, weighted per voxel).
    #   mm: the consensus's own range. water, lipids: stated.
    diffusion = Params(diffusion=dict(
        adc={name: PerTissue(gm=0.10, wm=0.15, csf=0.10) for name in names},
        water=0.75,
        mm=(0.005, 0.01),
        lipids=0.0))
    relaxation = Params(t2=t2, t1=Priors.literature_t1(names))
    return Params(conc=conc) | relaxation | diffusion | background | imperfections
