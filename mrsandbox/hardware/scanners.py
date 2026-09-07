####################################################################################################
#                                           scanners.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Scanner descriptions. Tiny on purpose: field strength first; gradient/slew/B1/raster    #
#          limits join as simulations start consuming them. "Select scanner" grows here.           #
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
# A scanner description; today that is a name and a field strength.                                #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Scanner:
    """A scanner description; today that is a name and a field strength.

    Attributes:
        name: Preset name.
        b0_t: Field strength in tesla.
    """

    name: str
    b0_t: float

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
    "generic_1p5t": Scanner("generic_1p5t", 1.5),
    "generic_3t": Scanner("generic_3t", 3.0),
    "generic_7t": Scanner("generic_7t", 7.0),
}
