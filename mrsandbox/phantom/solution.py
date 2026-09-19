####################################################################################################
#                                           solution.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: The in vitro subject: a sphere of metabolite solution of known contents, pH, and        #
#          temperature that takes a brain's place in a scan. No tissue, no macromolecules, no      #
#          lipids; what it tells belongs to the hardware and the spin physics.                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox.database import concentrations
from mrsandbox.params import Params
from mrsandbox.phantom.brain import Brain


#**************************************************************************************************#
#                                          Class Phantom                                           #
#**************************************************************************************************#
#                                                                                                  #
# A sphere of metabolite solution with known contents: the in vitro subject of a scan.             #
#                                                                                                  #
#**************************************************************************************************#
class Phantom(Brain):
    """A sphere of metabolite solution with known contents: the in vitro subject of a scan.

    It stands where a Brain stands. A VirtualScan of it takes no macromolecules, lipids,
    or downfield peaks, runs at the phantom's temperature and pH (the shifts and the
    water line follow, database.shifts), and composes `params()` over the in vivo
    defaults: the contents fixed, everything else absent, one relaxation for all
    metabolites, the lines as narrow as the relaxation makes them.

    The relaxation times are NOT in vivo numbers and no table gives them: a doped
    solution's depend on its dose. The defaults are read off one SPECTRE session (Philips
    Ingenia CX, 3 T, 21.7 C: the singlets' widths of 0.6 to 1.1 Hz bound the
    metabolite T2 from below at 0.3 to 0.5 s, the water line's 1.2 Hz its T2 at 0.25 s);
    both T1 are stated assumptions. Override them with the phantom's own.

    A phantom is in vitro: a scan of it may calibrate what belongs to the hardware
    (receiver noise, array correlation, drift, the digital filter, the gradients'
    response), with the load named, since a phantom loads a coil less than a head. It
    never sets a tissue leaf.

    Args:
        contents: {metabolite: mM}, names as in the Database.
        diameter_mm: Outer diameter of the sphere.
        ph, temperature_c: Of the solution, as scanned.
        t1_s, t2_s: Relaxation of every metabolite.
        water_t1_s, water_t2_s: Relaxation of the (doped) water.
        name: Identifier stored in provenance.
        cell_mm: Grid step of the sphere's map.
    """

    # a solution is water throughout
    water_content = {"gm": 1.0, "wm": 1.0, "csf": 1.0}

    #**********#
    #   init   #
    #**********#
    def __init__(self, contents, diameter_mm=180.0, ph=7.0, temperature_c=20.0, t1_s=1.5, t2_s=0.5,
                 water_t1_s=0.4, water_t2_s=0.25, name="phantom", cell_mm=4.0):
        half = np.ceil(diameter_mm / 2.0 / cell_mm) * cell_mm + cell_mm
        axis = np.arange(-half, half + cell_mm / 2.0, cell_mm)
        radius = np.linalg.norm(np.stack(np.meshgrid(axis, axis, axis, indexing="ij")), axis=0)
        inside = (radius <= diameter_mm / 2.0).astype(np.float32)
        affine = np.diag([cell_mm, cell_mm, cell_mm, 1.0])
        affine[:3, 3] = -half
        # the solution counts as the first tissue class; no fat anywhere
        super().__init__(gm=inside, wm=0.0 * inside, csf=0.0 * inside, affine=affine, name=name, fat=0.0 * inside)
        self.contents = dict(contents)
        self.diameter_mm = float(diameter_mm)
        self.ph, self.temperature_c = float(ph), float(temperature_c)
        self.t1_s, self.t2_s = float(t1_s), float(t2_s)
        self.water_t1_s, self.water_t2_s = float(water_t1_s), float(water_t2_s)

    #*************#
    #   spectre   #
    #*************#
    @classmethod
    def spectre(cls, temperature_c=20.0, pico=False, **kwargs):
        """The SPECTRE phantom of Gold Standard Phantoms (database.concentrations
        SPECTRE): seven metabolites at pH 7.4, 180 mm across, 110 mm as the Pico."""
        return cls(concentrations.SPECTRE, diameter_mm=110.0 if pico else 180.0, ph=7.4,
                   temperature_c=temperature_c, name="spectre_pico" if pico else "spectre", **kwargs)

    #************#
    #   params   #
    #************#
    def params(self, defaults):
        """The overrides a scan of the phantom composes over its in vivo `defaults`: the
        contents fixed and every other metabolite absent, one relaxation for all, no
        tissue contrast, the water's own relaxation, and no broadening beyond the
        relaxation's. A content outside the in vivo tree (free choline) gets the leaves
        the tree lacks; the diffusion coefficients stay the in vivo ones (stated: a free
        solution's are larger, which a design without diffusion weighting does not see)."""
        names = list(defaults.conc) + [name for name in self.contents if name not in defaults.conc]
        adc = defaults.diffusion.adc
        return Params(
            conc={name: float(self.contents.get(name, 0.0)) for name in names},
            t1={name: self.t1_s for name in names},
            t2={name: self.t2_s for name in names},
            contrast={name: 1.0 for name in names},
            diffusion=dict(adc={name: adc[name] if name in adc else adc.NAA for name in names}),
            water=dict(t1_s=self.water_t1_s, t2_s=self.water_t2_s),
            lineshape=dict(gauss_hz=0.0),
        )
