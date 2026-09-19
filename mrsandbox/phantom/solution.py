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
import dataclasses

import numpy as np

from mrsandbox.database import phantoms
from mrsandbox.params import Mapped, Params
from mrsandbox.phantom.brain import Brain
from mrsandbox.phantom.volume import Volume


#**************************************************************************************************#
#                                            Class Vial                                            #
#**************************************************************************************************#
#                                                                                                  #
# A sphere of its own solution inside a phantom; what it leaves unset is the phantom's.            #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Vial:
    """A sphere of its own solution inside a phantom; what it leaves unset is the phantom's.

    Args:
        center_mm: Centre, millimetres from the phantom's.
        diameter_mm: Inner diameter.
        contents: {metabolite: mM}.
        t1_s, t2_s: Metabolite relaxation, one value or {metabolite: value}.
        water_t1_s, water_t2_s: Relaxation of its water.
        water: The share of its water that is H2O (the rest D2O): its proton density.
    """

    center_mm: tuple
    diameter_mm: float
    contents: dict = None
    t1_s: object = None
    t2_s: object = None
    water_t1_s: float = None
    water_t2_s: float = None
    water: float = 1.0


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
    defaults: the contents fixed, everything else absent, the phantom's relaxation, the
    lines as narrow as the relaxation makes them. `vials` are spheres of other
    solutions inside it; a leaf a vial changes becomes a map (params.Mapped) that every
    voxel reads at its own place, as it reads a brain's quantitative maps.

    The relaxation times are NOT in vivo numbers: a doped solution's depend on its dose
    and the gadolinium relaxes the metabolites unequally (database.phantoms
    BRAINO_RELAXATION: choline's T1 a third of the NAA singlet's at 7 T). The defaults:
    the water T1 of 0.4 s is what two groups measured in a Braino near 3 T (0.39 and
    0.45 s); the metabolite T2 of 0.5 s and the water T2 of 0.5 s are bounds read off
    the linewidths of one SPECTRE session (Philips Ingenia CX, 3 T, 21.7 C: singlets
    0.6 to 1.1 Hz wide, the water line 0.7 Hz); the metabolite T1 of 1.5 s is a stated
    assumption. Give the phantom's own where they are known, per metabolite if need be.

    A phantom is in vitro: a scan of it may calibrate what belongs to the hardware
    (receiver noise, array correlation, drift, the digital filter, the gradients'
    response), with the load named, since a phantom loads a coil less than a head. It
    never sets a tissue leaf.

    Args:
        contents: {metabolite: mM}, names as in the Database.
        diameter_mm: Inner diameter of the sphere.
        ph, temperature_c: Of the solution, as scanned.
        t1_s, t2_s: Metabolite relaxation, one value for all or {metabolite: value}
            (a metabolite the dict leaves out keeps the default).
        water_t1_s, water_t2_s: Relaxation of the (doped) water.
        vials: Vials inside the sphere.
        name: Identifier stored in provenance.
        cell_mm: Grid step of the phantom's maps; a vial needs a few cells across.
    """

    # a solution is water throughout
    water_content = {"gm": 1.0, "wm": 1.0, "csf": 1.0}
    T1_S, T2_S = 1.5, 0.5

    #**********#
    #   init   #
    #**********#
    def __init__(self, contents, diameter_mm=180.0, ph=7.0, temperature_c=20.0, t1_s=T1_S, t2_s=T2_S,
                 water_t1_s=0.4, water_t2_s=0.5, vials=(), name="phantom", cell_mm=4.0):
        half = np.ceil(diameter_mm / 2.0 / cell_mm) * cell_mm + cell_mm
        axis = np.arange(-half, half + cell_mm / 2.0, cell_mm)
        self._grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        self.vials = tuple(vials)
        # the solution counts as the first tissue class, a vial's D2O as its absence; no fat
        solution = (np.linalg.norm(self._grid, axis=-1) <= diameter_mm / 2.0).astype(np.float32)
        for vial in self.vials:
            solution[self._inside(vial)] = vial.water
        affine = np.diag([cell_mm, cell_mm, cell_mm, 1.0])
        affine[:3, 3] = -half
        super().__init__(gm=solution, wm=0.0 * solution, csf=0.0 * solution, affine=affine, name=name,
                         fat=0.0 * solution)
        self.contents = dict(contents)
        self.diameter_mm = float(diameter_mm)
        self.ph, self.temperature_c = float(ph), float(temperature_c)
        self.t1_s, self.t2_s = t1_s, t2_s
        self.water_t1_s, self.water_t2_s = float(water_t1_s), float(water_t2_s)

    def _inside(self, vial):
        return np.linalg.norm(self._grid - np.asarray(vial.center_mm, dtype=float), axis=-1) <= vial.diameter_mm / 2.0

    #*************#
    #   recipes   #
    #*************#
    @classmethod
    def spectre(cls, temperature_c=20.0, pico=False, **kwargs):
        """The SPECTRE phantom of Gold Standard Phantoms (database.phantoms SPECTRE):
        seven metabolites at pH 7.4, 180 mm across, 110 mm as the Pico."""
        return cls(phantoms.SPECTRE, diameter_mm=110.0 if pico else 180.0, ph=7.4,
                   temperature_c=temperature_c, name="spectre_pico" if pico else "spectre", **kwargs)

    @classmethod
    def braino(cls, temperature_c=20.0, **kwargs):
        """GE's Braino sphere as the literature reports it (database.phantoms BRAINO): six
        metabolites, 169 mm across; its pH is in no source and stays at 7 (stated)."""
        return cls(phantoms.BRAINO, diameter_mm=169.0, temperature_c=temperature_c, name="braino", **kwargs)

    @classmethod
    def nist(cls, b0_t=3.0, commercial=False, temperature_c=20.0, cell_mm=2.5):
        """The ISMRM/NIST system phantom (Stupic et al. 2021): water, no metabolites, and
        three plates of fourteen spheres with their measured T1 and T2, the proton density
        spheres with their share of H2O. The prototype's tables hold 1.5 and 3 T at 20 C
        (database.phantoms NIST_SYSTEM); `commercial` takes the T1 and T2 arrays of the
        model 130 from its NIST certificate (NIST_MODEL_130: 3 T only, 16 to 26 C, the
        nearest typed temperature; two of its MnCl2 spheres are flagged there). The fill
        water takes its own measured rates at 3 T and 20 C throughout (stated); the
        fiducial spheres, the plates, and the resolution inset are not modelled."""
        table, field = phantoms.NIST_SYSTEM, 3 if b0_t > 2.25 else 2
        relaxation = {array: [row[field] for row in table[array]] for array in ("t1_array", "t2_array", "pd_array")}
        if commercial:
            if field == 2:
                raise ValueError("the model 130's certificate holds 3 T only")
            for array, rows in phantoms.NIST_MODEL_130.items():
                relaxation[array] = [values[min(values, key=lambda t: abs(t - temperature_c))] for _, values in rows]
        vials = [Vial(row[0], table["sphere_inner_diameter_mm"], water_t1_s=t1_ms / 1000.0, water_t2_s=t2_ms / 1000.0,
                      water=1.0 - row[1] / 100.0 if array == "pd_array" else 1.0)
                 for array in relaxation for row, (t1_ms, t2_ms) in zip(table[array], relaxation[array])]
        r1, r2 = table["water_r1_r2_3t"]
        return cls({}, diameter_mm=table["inner_diameter_mm"], temperature_c=temperature_c, water_t1_s=1.0 / r1,
                   water_t2_s=1.0 / r2, vials=vials, name="nist_model_130" if commercial else "nist_system", cell_mm=cell_mm)

    #************#
    #   params   #
    #************#
    def params(self, defaults):
        """The overrides a scan of the phantom composes over its in vivo `defaults`: the
        contents fixed and every other metabolite absent, the phantom's relaxation, no
        tissue contrast, the water's own relaxation, and no broadening beyond the
        relaxation's; maps where a vial differs. A content outside the in vivo tree
        (free choline) gets the leaves the tree lacks; the diffusion coefficients stay
        the in vivo ones (stated: a free solution's are larger, which a design without
        diffusion weighting does not see)."""
        names = list(defaults.conc)
        names += [name for holder in (self, *self.vials) for name in (holder.contents or {}) if name not in names]
        adc = defaults.diffusion.adc

        def leaf(of, fallback):
            """One leaf over the phantom: `of(holder)` is a holder's value or None."""
            here = of(self) if of(self) is not None else fallback
            changed = [(vial, of(vial)) for vial in self.vials if of(vial) is not None and of(vial) != here]
            if not changed:
                return float(here)
            values = np.full(self.gm.shape, here, dtype=np.float32)
            for vial, value in changed:
                values[self._inside(vial)] = value
            return Mapped(Volume(values, self.affine, name=self.name))

        per = lambda attribute, name: lambda holder: (
            getattr(holder, attribute).get(name) if isinstance(getattr(holder, attribute), dict) else getattr(holder, attribute))
        return Params(
            conc={name: leaf(lambda h, name=name: None if h.contents is None else h.contents.get(name, 0.0), 0.0)
                  for name in names},
            t1={name: leaf(per("t1_s", name), self.T1_S) for name in names},
            t2={name: leaf(per("t2_s", name), self.T2_S) for name in names},
            contrast={name: 1.0 for name in names},
            diffusion=dict(adc={name: adc[name] if name in adc else adc.NAA for name in names}),
            water=dict(t1_s=leaf(lambda h: h.water_t1_s, None), t2_s=leaf(lambda h: h.water_t2_s, None)),
            lineshape=dict(gauss_hz=0.0),
        )
