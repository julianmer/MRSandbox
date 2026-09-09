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
        # residual fraction after suppression is a stated assumption; the per-tissue
        # water relaxation is weighted by each voxel's composition at render time.
        # T1 draws from the crowdsourced 3 T reproducibility challenge (the typed
        # BOUDREAU table: cortical GM, and the genu for WM, the challenge's
        # cleanest ROI; the intersubmission sd is a realistic site-to-site spread,
        # clipped at a stated 0.2 s floor); CSF T1 and the T2 values keep the
        # cited WATER table entries
        water=dict(residual_fraction=(1e-4, 3e-3),
                   t1_s=PerTissue(
                       gm=Clipped(Normal(*BOUDREAU["t1_s"]["gm_cortex"]), low=0.2),
                       wm=Clipped(Normal(*BOUDREAU["t1_s"]["wm_genu"]), low=0.2),
                       csf=WATER["t1_s"]["csf"]),
                   t2_s=PerTissue(**WATER["t2_s"])),
        # adipose proton concentration is water-like (about 60 to 70 M), which is why
        # a voxel clipping the skull drowns in fat; the range is a stated assumption
        lipids=dict(amplitude_mm=(30000.0, 70000.0), t2_s=0.05, t1_s=0.3),
        # macromolecules: the coupled default scales as kappa times the challenge
        # tissue ratios times the drawn creatine. The challenge kappa of 0.4 applies
        # to its own measured basis normalization, not to concentration units, so
        # kappa here is recalibrated (stated) to land the short echo time MM area
        # fraction in the consensus 10 to 25 percent ballpark; level_mm drives the
        # ParameterizedMM alternative. Per-component relaxation is typed from the
        # consensus Appendix S1 3 T human columns (T2: Landheer 2020, else
        # Hoefemann 2020; T1: Hoefemann 2020, fit-limit rows excluded; the
        # database.macromolecules CONSENSUS tables), components mapped by peak
        # identity (challenge M0.91 is consensus M0.94 and so on, positions vary
        # 0.01 to 0.04 ppm between papers). Where the table prints an error the
        # leaf draws from it, clipped at a 5 ms floor (a stated plausibility
        # bound); values without errors and components without a 3 T value stay
        # fixed at the typed value or the stated fallback
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
    )
    # instrumental and physiological imperfections and artifact signals
    imperfections = Params(
        # complex Gaussian noise per FID point, acquisition side; range is an assumption
        noise=dict(sigma=(0.5, 3.0)),
        # transmit field scale: every pulse flip is multiplied by the drawn value;
        # a few percent calibration scatter at 3 T is a stated assumption
        b1_scale=Normal(1.0, 0.03),
        # receiver-side imperfections, stated assumptions: scanner frequency error
        # and drift as a global shift, an uncorrected zero-order phase, and a
        # residual eddy current as a decaying phase on the FID (the leading term of
        # the multi-exponential gradient response, de Graaf 2019; in vivo the bulk
        # is removed by Klose 1990 ECC, so the residual amplitude stays modest)
        acquisition=dict(shift_hz=Normal(0.0, 2.0), phase_deg=Uniform(-20.0, 20.0),
                         eddy_deg=Uniform(-30.0, 30.0), eddy_tau_s=(0.01, 0.1)),
        # spurious echo: out-of-voxel magnetization that imperfect crushing or phase
        # cycling fails to remove and that rephases mid-acquisition (the mechanism
        # is demonstrated physically by the crusher gates in tests/sequences). The
        # form and every range are typed from the AGNOSTIC OOV synthesis
        # (Gudmundson et al. 2023, Imaging Neuroscience, imag_a_00025): a Gaussian
        # envelope echo a exp(-W (t - tau)^2) exp(i (2 pi f t + phi)) with tau
        # 10-400 ms, W 500-8000 1/s^2 (FWHM 18-74 ms), position 1-4 ppm, amplitude
        # 0.1-20% of the FID maximum, free phase; the source adds one echo to 85%
        # of examples, here it is always on and amplitude=0.0 switches it off
        spurious=dict(amplitude=(0.001, 0.2), tau_s=(0.010, 0.400),
                      width_s2=(500.0, 8000.0), ppm=(1.0, 4.0),
                      phase_deg=(0.0, 360.0)),
        # scanner and subject instabilities across repeated transients. The drift
        # rate is grounded in Hui et al. 2021 (NeuroImage 241:118430, 80 sites,
        # 99 scanners at 3 T): median 0.29 Hz/min before fMRI heating with an
        # IQR-derived spread, 0.43 after. The shot-to-shot jitter, phase scatter,
        # amplitude instability, and motion event model are stated assumptions,
        # every leaf overridable.
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
    # diffusion: every component attenuates by exp(-b ADC) with b built from
    # the sequence's own gradient events (sequences.diffusion, the consensus
    # Box 2 of Ligneul et al. 2024, MRM 91:860), so a standard sequence is
    # only as diffusion weighted as its crushers make it. ADCs in um^2/ms:
    # metabolites follow the human long-diffusion-time summary the consensus
    # defers to (Valette et al. 2018, Front Neurosci 12:2: about 0.10 in grey
    # and 0.15 in white matter, weighted by the voxel composition); the MM
    # range is the consensus's own 0.005 to 0.01; water and lipids are stated
    diffusion = Params(diffusion=dict(
        adc={name: PerTissue(gm=0.10, wm=0.15, csf=0.10) for name in names},
        water=0.75,
        mm=(0.005, 0.01),
        lipids=0.0))
    relaxation = Params(t2=t2, t1=Priors.literature_t1(names))
    return Params(conc=conc) | relaxation | diffusion | background | imperfections
