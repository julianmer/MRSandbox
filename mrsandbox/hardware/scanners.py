####################################################################################################
#                                           scanners.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Scanner descriptions: the field strength and the hardware limits a design must stay     #
#          within (peak B1, maximum gradient, slew rate). "Select scanner" grows here.             #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses


#**************************************************************************************************#
#                                          Class Scanner                                           #
#**************************************************************************************************#
#                                                                                                  #
# A scanner description: a name, a field strength, and the hardware limits.                        #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Scanner:
    """A scanner description: a name, a field strength, and the hardware limits.

    The limits are what `Sequence.feasible` checks a design against: the peak
    transmit field bounds every pulse amplitude and sets the shortest hard pulse, the
    gradient amplitude and slew rate set the shortest crusher and slice gradient. The
    presets carry stated numbers per field class until measured ones arrive: the
    gradient limits of the common whole-body systems, and the peak B1 of Oz et al.
    2021 (NMR Biomed 34:e4236, adiabatic pulses: 24 to 26 uT with a body coil at 3 T,
    26 to 45 uT with a head transmit coil at 7 T).

    Attributes:
        name: Preset name.
        b0_t: Field strength in tesla.
        peak_b1_ut: Peak transmit field in microtesla.
        max_grad_mt_m: Maximum gradient amplitude in millitesla per metre.
        max_slew_t_m_s: Maximum slew rate in tesla per metre per second.
        vendor, model: What the raw data's headers call the system (lowercase vendor), by
            which the package's measured rows are pooled for it (scan.cohort pooled);
            None for a generic scanner, which pools every system at its field.
        cited: Where the numbers come from: the vendor's sheet or the paper, or "stated".
    """

    name: str
    b0_t: float
    peak_b1_ut: float = 25.0
    max_grad_mt_m: float = 40.0
    max_slew_t_m_s: float = 150.0
    vendor: str = None
    model: str = None
    cited: str = "stated"

    #*****************#
    #   from preset   #
    #*****************#
    @classmethod
    def preset(cls, name):
        """Look up a named preset."""
        try:
            return PRESETS[name]
        except KeyError:
            raise KeyError(f"unknown scanner preset {name!r}; known: {', '.join(sorted(PRESETS))}") from None


PRESETS = {
    "generic_1p5t": Scanner("generic_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=40.0,
                            max_slew_t_m_s=150.0,
                            cited="stated: the common whole-body limits at 1.5 T"),
    "generic_3t": Scanner("generic_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=40.0,
                          max_slew_t_m_s=200.0,
                          cited="stated: the common whole-body limits at 3 T, the peak B1 of Oz et al. 2021 (NMR Biomed 34:e4236)"),
    "generic_7t": Scanner("generic_7t", 7.0, peak_b1_ut=30.0, max_grad_mt_m=70.0,
                          max_slew_t_m_s=200.0,
                          cited="stated: the common 7 T limits, the peak B1 of Oz et al. 2021 (NMR Biomed 34:e4236, a head transmit coil)"),
    # the field of a "3 T" system is what its raw data state: the Siemens systems run at
    # 2.895 T (123.25 MHz in the Big GABA Trio headers of sites S1 and S5, 123.26 in the
    # COWS Prisma's), the GE MR750 at 3.001 T (127.77 MHz in Big GABA's GE files); the
    # Philips systems keep the nominal 3.0 T, no raw header of theirs read (stated)
    # the three 3 T systems of the across-vendor semi-LASER (Deelchand et al. 2021, NMR
    # Biomed 34:e4218): the body coil peak B1 as the paper states it (GE and Siemens
    # between 20 and 25 uT, the upper end typed; Philips 13.5 uT by default, raised to
    # 15 uT for the study, 22 uT on later releases), the gradients from the vendors'
    # specifications (Prisma XR 80/200, MR750 XRM 50/200, Elition Vega HP 45/220)
    "siemens_prisma_3t": Scanner("siemens_prisma_3t", 2.895, peak_b1_ut=25.0, max_grad_mt_m=80.0,
                                 max_slew_t_m_s=200.0, vendor="siemens", model="Prisma",
                                 cited="the body coil's peak B1 as Deelchand et al. 2021 (NMR Biomed 34:e4218) state it, the gradients from the vendor's specification (Prisma XR 80 mT/m, 200 T/m/s); the field from the COWS raw headers, 123.26 MHz"),
    "ge_mr750_3t": Scanner("ge_mr750_3t", 3.001, peak_b1_ut=25.0, max_grad_mt_m=50.0,
                           max_slew_t_m_s=200.0, vendor="ge", model="MR750",
                           cited="the body coil's peak B1 as Deelchand et al. 2021 (NMR Biomed 34:e4218) state it, the gradients from the vendor's specification (MR750 XRM 50 mT/m, 200 T/m/s); the field from Big GABA's GE files, 127.77 MHz"),
    "philips_elition_3t": Scanner("philips_elition_3t", 3.0, peak_b1_ut=15.0, max_grad_mt_m=45.0,
                                  max_slew_t_m_s=220.0, vendor="philips", model="Elition",
                                  cited="the body coil's peak B1 as Deelchand et al. 2021 (NMR Biomed 34:e4218) state it, the gradients from the vendor's specification (Elition Vega HP 45 mT/m, 220 T/m/s); the field nominal, no raw header read"),
    # the two systems the noise is calibrated on besides the Prisma (scan.calibrations): the
    # Trio of Big GABA's Siemens sites with its head arrays, and the Achieva of the fMRS in
    # pain data with its one-channel transmit-receive head coil, which is part of what the
    # preset stands for. Peak B1 as the generic 3 T and the Philips default above, gradients
    # from the vendors' specifications (Trio TQ 40/200, Achieva Quasar Dual 40/200); stated.
    # the MR 7700 with its XP gradients: "amplitudes of 65 mT/m and high slew rates of 220 T/m/s simultaneously, on
    # each cartesian axis" (Philips, MR 7700 brochure, read 19/09/26:
    # philips.com/c-dam/b2bhc/master/resource-catalog/landing/mr-7700/brochure-mr7700.pdf); the peak B1 as the
    # Philips study value above, which is stated and found in no sheet of this system
    "philips_mr7700_3t": Scanner("philips_mr7700_3t", 3.0, peak_b1_ut=15.0, max_grad_mt_m=65.0,
                                 max_slew_t_m_s=220.0, vendor="philips", model="MR 7700",
                                 cited="the gradients from the Philips MR 7700 brochure (philips.com/c-dam/b2bhc/master/resource-catalog/landing/mr-7700/brochure-mr7700.pdf, read 19/09/26: 65 mT/m at 220 T/m/s on each axis); the peak B1 the Philips study value of Deelchand et al. 2021, stated; the field nominal"),
    "siemens_trio_3t": Scanner("siemens_trio_3t", 2.895, peak_b1_ut=25.0, max_grad_mt_m=40.0,
                               max_slew_t_m_s=200.0, vendor="siemens", model="TrioTim",
                               cited="the gradients from the vendor's specification (Trio TQ 40 mT/m, 200 T/m/s), the peak B1 as the generic 3 T; the field from Big GABA's Trio headers, 123.25 MHz"),
    "siemens_avanto_1p5t": Scanner("siemens_avanto_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                     vendor="siemens", model="Avanto", cited="SQ gradients 45 mT/m at 200 T/m/s (MRN, mrn.org/collaborate/mobile-mri-system, read 27/09/26); the peak B1 stated"),
    "siemens_aera_1p5t": Scanner("siemens_aera_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                   vendor="siemens", model="Aera", cited="XQ gradients 45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "siemens_sola_1p5t": Scanner("siemens_sola_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                   vendor="siemens", model="Sola", cited="XQ gradients 45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "siemens_verio_3t": Scanner("siemens_verio_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                  vendor="siemens", model="Verio", cited="45 mT/m at 200 T/m/s (Block Imaging, Siemens 3T MRI scanners compared (blockimaging.com/siemens-3t-mri-scanners-compared, read 27/09/26))"),
    "siemens_skyra_3t": Scanner("siemens_skyra_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                  vendor="siemens", model="Skyra", cited="XQ gradients 45 mT/m, 200 T/m/s (the slew from Block Imaging, Siemens 3T MRI scanners compared (blockimaging.com/siemens-3t-mri-scanners-compared, read 27/09/26), the amplitude nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26))"),
    "siemens_spectra_3t": Scanner("siemens_spectra_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=33.0, max_slew_t_m_s=125.0,
                                    vendor="siemens", model="Spectra", cited="33 mT/m at 125 T/m/s (Block Imaging, Siemens 3T MRI scanners compared (blockimaging.com/siemens-3t-mri-scanners-compared, read 27/09/26))"),
    "siemens_vida_3t": Scanner("siemens_vida_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=60.0, max_slew_t_m_s=200.0,
                                 vendor="siemens", model="Vida", cited="XT gradients 60 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "siemens_lumina_3t": Scanner("siemens_lumina_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=36.0, max_slew_t_m_s=200.0,
                                   vendor="siemens", model="Lumina", cited="XK gradients 36 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "siemens_terra_7t": Scanner("siemens_terra_7t", 7.0, peak_b1_ut=30.0, max_grad_mt_m=80.0, max_slew_t_m_s=200.0,
                                  vendor="siemens", model="Terra", cited="XR gradients 80 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 of a head transmit coil, Oz et al. 2021"),
    "ge_signa_hdxt_3t": Scanner("ge_signa_hdxt_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=150.0,
                                  vendor="ge", model="Signa HDxt", cited="45 mT/m at 150 T/m/s (Signa Pulse, signapulse.gehealthcare.com, gradient systems explained, read 27/09/26)"),
    "ge_mr750w_3t": Scanner("ge_mr750w_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=44.0, max_slew_t_m_s=200.0,
                              vendor="ge", model="MR750w", cited="44 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "ge_signa_pioneer_3t": Scanner("ge_signa_pioneer_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=36.0, max_slew_t_m_s=150.0,
                                     vendor="ge", model="Signa Pioneer", cited="36 mT/m at 150 T/m/s (the Pioneer specification sheet as posted on scribd, read 27/09/26)"),
    "ge_signa_architect_3t": Scanner("ge_signa_architect_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=44.0, max_slew_t_m_s=200.0,
                                       vendor="ge", model="Signa Architect", cited="44 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "ge_signa_premier_3t": Scanner("ge_signa_premier_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=80.0, max_slew_t_m_s=200.0,
                                     vendor="ge", model="Signa Premier", cited="SuperG gradients 80 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "ge_signa_artist_1p5t": Scanner("ge_signa_artist_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=44.0, max_slew_t_m_s=200.0,
                                      vendor="ge", model="Signa Artist", cited="44 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "ge_signa_explorer_1p5t": Scanner("ge_signa_explorer_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=33.0, max_slew_t_m_s=120.0,
                                        vendor="ge", model="Signa Explorer", cited="33 mT/m at 120 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "philips_ingenia_1p5t": Scanner("philips_ingenia_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                      vendor="philips", model="Ingenia", cited="Omega HP gradients 45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "philips_ambition_1p5t": Scanner("philips_ambition_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                       vendor="philips", model="Ambition", cited="45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "philips_ingenia_3t": Scanner("philips_ingenia_3t", 3.0, peak_b1_ut=15.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                    vendor="philips", model="Ingenia", cited="Omega HP gradients 45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 the Philips study value of Deelchand et al. 2021"),
    "philips_achieva_7t": Scanner("philips_achieva_7t", 7.0, peak_b1_ut=30.0, max_grad_mt_m=40.0, max_slew_t_m_s=200.0,
                                    vendor="philips", model="Achieva", cited="40 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 of a head transmit coil, Oz et al. 2021"),
    "bruker_biospec_7t": Scanner("bruker_biospec_7t", 7.0, peak_b1_ut=60.0, max_grad_mt_m=660.0, max_slew_t_m_s=4570.0,
                                   vendor="bruker", model="BioSpec", cited="preclinical: BGA-12S HP gradients 660 mT/m at 4570 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 of a small volume coil, stated"),
    "bruker_biospec_9p4t": Scanner("bruker_biospec_9p4t", 9.4, peak_b1_ut=60.0, max_grad_mt_m=660.0, max_slew_t_m_s=4570.0,
                                     vendor="bruker", model="BioSpec", cited="preclinical: BGA-12S HP gradients 660 mT/m at 4570 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 of a small volume coil, stated"),
    "united_umr_570_1p5t": Scanner("united_umr_570_1p5t", 1.5, peak_b1_ut=25.0, max_grad_mt_m=33.0, max_slew_t_m_s=120.0,
                                     vendor="united", model="uMR 570", cited="33 mT/m at 120 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "united_umr_780_3t": Scanner("united_umr_780_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=65.0, max_slew_t_m_s=200.0,
                                   vendor="united", model="uMR 780", cited="65 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "united_umr_890_5t": Scanner("united_umr_890_5t", 5.0, peak_b1_ut=30.0, max_grad_mt_m=120.0, max_slew_t_m_s=200.0,
                                   vendor="united", model="uMR Jupiter", cited="5 T: 120 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26); the peak B1 stated"),
    "canon_galan_3t": Scanner("canon_galan_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=45.0, max_slew_t_m_s=200.0,
                                vendor="canon", model="Vantage Galan", cited="45 mT/m at 200 T/m/s, nominal, as the vendor quotes it and the trade compares it; the sheet itself not read (27/09/26)"),
    "philips_achieva_3t": Scanner("philips_achieva_3t", 3.0, peak_b1_ut=13.5, max_grad_mt_m=40.0,
                                  max_slew_t_m_s=200.0, vendor="philips", model="Achieva",
                                  cited="the gradients from the vendor's specification (Achieva Quasar Dual 40 mT/m, 200 T/m/s), the peak B1 the Philips default of Deelchand et al. 2021; the field nominal"),
}
