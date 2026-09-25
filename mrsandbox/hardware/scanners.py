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
    """

    name: str
    b0_t: float
    peak_b1_ut: float = 25.0
    max_grad_mt_m: float = 40.0
    max_slew_t_m_s: float = 150.0

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
                            max_slew_t_m_s=150.0),
    "generic_3t": Scanner("generic_3t", 3.0, peak_b1_ut=25.0, max_grad_mt_m=40.0,
                          max_slew_t_m_s=200.0),
    "generic_7t": Scanner("generic_7t", 7.0, peak_b1_ut=30.0, max_grad_mt_m=70.0,
                          max_slew_t_m_s=200.0),
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
                                 max_slew_t_m_s=200.0),
    "ge_mr750_3t": Scanner("ge_mr750_3t", 3.001, peak_b1_ut=25.0, max_grad_mt_m=50.0,
                           max_slew_t_m_s=200.0),
    # the same MR750 with its 32-channel head array, which has a noise record of its own
    # (scan.calibrations): the coil is part of what the preset stands for
    "ge_mr750_32ch_3t": Scanner("ge_mr750_32ch_3t", 3.001, peak_b1_ut=25.0, max_grad_mt_m=50.0,
                                max_slew_t_m_s=200.0),
    "philips_elition_3t": Scanner("philips_elition_3t", 3.0, peak_b1_ut=15.0, max_grad_mt_m=45.0,
                                  max_slew_t_m_s=220.0),
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
                                 max_slew_t_m_s=220.0),
    "siemens_trio_3t": Scanner("siemens_trio_3t", 2.895, peak_b1_ut=25.0, max_grad_mt_m=40.0,
                               max_slew_t_m_s=200.0),
    "philips_achieva_3t": Scanner("philips_achieva_3t", 3.0, peak_b1_ut=13.5, max_grad_mt_m=40.0,
                                  max_slew_t_m_s=200.0),
}
