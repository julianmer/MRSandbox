####################################################################################################
#                                         macromolecules.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: The macromolecule background as a swappable source, like water and lipids. The          #
#          parameterized source renders the LCModel-convention Gaussians from the database with    #
#          a sampled overall level; the shape is fixed by the parameterization, the amount is a    #
#          drawn parameter, exactly how linear combination fitting treats it.                      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc

import numpy as np

from mrsandbox.database import macromolecules
from mrsandbox.io import VarianFid


#**************************************************************************************************#
#                                          Class MMSource                                          #
#**************************************************************************************************#
#                                                                                                  #
# How the voxel's macromolecule background enters the acquisition.                                 #
#                                                                                                  #
#**************************************************************************************************#
class MMSource(abc.ABC):
    """How the voxel's macromolecule background enters the acquisition.

    A source contributes either parameterized lines (`lines`) or a direct time domain
    signal (`fid`); the default for each is silence, a subclass implements one.
    """

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, fractions, truth, f0_mhz):
        """Return (freqs_hz, amps, gauss_hz), or None when line-silent. Sources built
        from static parameterizations may ignore the engine and sequence; sources built
        from spin systems run them through both, so TE behaviour and editing emerge."""
        return None

    #*********#
    #   fid   #
    #*********#
    def fid(self, fractions, truth, f0_mhz, t_s):
        """Return the complex time domain contribution on `t_s`, or None when silent."""
        return None


#**************************************************************************************************#
#                                            Class NoMM                                            #
#**************************************************************************************************#
#                                                                                                  #
# A macromolecule-free voxel (a metabolite-nulled acquisition, or plain simplification).           #
#                                                                                                  #
#**************************************************************************************************#
class NoMM(MMSource):
    """A macromolecule-free voxel (a metabolite-nulled acquisition, or plain simplification)."""


#**************************************************************************************************#
#                                       Class ParameterizedMM                                      #
#**************************************************************************************************#
#                                                                                                  #
# The LCModel-convention Gaussian components (Osprey values), scaled by the drawn mm level.        #
#                                                                                                  #
#**************************************************************************************************#
class ParameterizedMM(MMSource):
    """The LCModel-convention Gaussian components (Osprey values), scaled by the drawn
    mm level.

    Each component is a Gaussian at its published position and FWHM; its amplitude is
    the published proton area times the drawn level (concentration equivalent mM per
    proton area) times the tissue presence of the voxel. The parameterization describes
    short echo time spectra; at long echo times lower the level.

    Args:
        components: Names to include; defaults to the MM components (lipids stay with
            the lipid source).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, components=("MM09", "MM12", "MM14", "MM17", "MM20")):
        self.components = tuple(components)

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, fractions, truth, f0_mhz):
        tissue = fractions.gm + fractions.wm
        if tissue <= 0.0:
            return None
        freqs, amps, gauss = [], [], []
        for name in self.components:
            for _, ppm, fwhm_ppm, protons in macromolecules.OSPREY[name]:
                freqs.append(ppm * f0_mhz)
                amps.append(protons * truth.mm.level_mm * tissue)
                gauss.append(fwhm_ppm * f0_mhz)
        return (np.asarray(freqs, dtype=np.float32),
                np.asarray(amps, dtype=np.complex64),
                np.asarray(gauss, dtype=np.float32))


#**************************************************************************************************#
#                                         Class MeasuredMM                                         #
#**************************************************************************************************#
#                                                                                                  #
# The measured consensus macromolecule spectrum as the background: real shapes, not Gaussians.     #
#                                                                                                  #
#**************************************************************************************************#
class MeasuredMM(MMSource):
    """The measured consensus macromolecule spectrum as the background: real in vivo
    shapes instead of Gaussians.

    The 3 T human metabolite-nulled STEAM spectrum from the consensus data collection
    (fetched on first use) is mapped into the simulation frame: conjugated to our sign
    convention, time scaled from the measured 123.245 MHz to the simulated field, and
    remodulated from its water centred transmitter frame (4.7 ppm, verified against the
    consensus peak positions) to the 0 ppm reference frame. The signal is normalized so
    the drawn mm.level_mm leaf keeps the ParameterizedMM meaning: total MM area equals
    the level times the parameterization's summed proton areas.
    """

    TOTAL_PROTON_AREAS = 11.39  # the ParameterizedMM MM components summed

    #**********#
    #   init   #
    #**********#
    def __init__(self, center_ppm=4.7):
        self.center_ppm = center_ppm
        self._measured = None

    #*********#
    #   fid   #
    #*********#
    def fid(self, fractions, truth, f0_mhz, t_s):
        tissue = fractions.gm + fractions.wm
        if tissue <= 0.0:
            return None
        measured, sw_hz, sfrq_mhz = self._load()
        t = np.asarray(t_s, dtype=np.float64)
        # sample the measured FID at times scaled to the simulated field, ppm-faithful
        t_measured = t * (f0_mhz / sfrq_mhz)
        grid = np.arange(measured.size) / sw_hz
        resampled = (np.interp(t_measured, grid, measured.real, right=0.0)
                     + 1j * np.interp(t_measured, grid, measured.imag, right=0.0))
        # conjugate into our +ppm convention, shift the transmitter frame to 0 ppm
        signal = np.conj(resampled) * np.exp(2j * np.pi * self.center_ppm * f0_mhz * t)
        scale = truth.mm.level_mm * tissue * self.TOTAL_PROTON_AREAS / np.conj(measured[0])
        return (signal * scale).astype(np.complex64)

    #*************#
    #   loading   #
    #*************#
    def _load(self):
        if self._measured is None:
            from mrsandbox.resources import Resources

            fid = VarianFid.load(Resources.path("mm_consensus/3t_human_steam_fid"))
            procpar = VarianFid.parameters(Resources.path("mm_consensus/3t_human_steam_procpar"))
            self._measured = (fid, procpar["sw"], procpar["sfrq"])
        return self._measured


#**************************************************************************************************#
#                                         Class CoupledMM                                          #
#**************************************************************************************************#
#                                                                                                  #
# Macromolecules as coupled mini spin systems: sequence behaviour and co-editing emerge.           #
#                                                                                                  #
#**************************************************************************************************#
class CoupledMM(MMSource):
    """Macromolecules as coupled mini spin systems run through the engine, so echo time
    behaviour and co-editing emerge from the physics instead of being modelled.

    Chains carry the consensus Table 1 couplings (CH3 to CH doublets for the 0.91, 1.21,
    and 1.43 components; the CH2 to CH2 lysine chain linking 1.67 to 2.99 that the
    consensus flags as the GABA-relevant background); components without a listed
    partner stay uncoupled. Amplitudes follow the MRSI challenge model: kappa times the
    tissue-weighted component ratio times the drawn total creatine, so the background
    scales with the subject. Chain partners at 2.04 and 2.99 are emitted by their chains
    (matching the challenge ratios within about 20 percent); Gaussian widths come from
    the Osprey values at matching positions, stated where no label matches. The proton
    counts of the chains (3:1 and 2:2) and the mm t1_s and t2_s leaves are stated
    assumptions.
    """

    # (name, groups as (ppm, protons), couplings as (a, b, j_hz), gauss width ppm)
    SYSTEMS = (
        ("mm091", ((0.91, 3), (2.04, 1)), ((0, 1, 7.7),), 0.14),
        ("mm121", ((1.21, 3), (4.24, 1)), ((0, 1, 6.6),), 0.15),
        ("mm143", ((1.43, 3), (4.32, 1)), ((0, 1, 7.7),), 0.17),
        ("mm167", ((1.67, 2), (2.99, 2)), ((0, 1, 7.7),), 0.15),
        ("mm226", ((2.26, 1),), (), 0.20),
        ("mm321", ((3.21, 1),), (), 0.15),
        ("mm377", ((3.77, 1),), (), 0.15),
    )

    # challenge component carrying each system's amplitude anchor
    ANCHORS = {"mm091": "M0.91", "mm121": "M1.21", "mm143": "M1.43", "mm167": "M1.67",
               "mm226": "M2.26", "mm321": "M3.21", "mm377": "M3.77"}

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        self._compiled = {}

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, fractions, truth, f0_mhz):
        from mrsandbox.spins import SpinSystem

        tissue = fractions.gm + fractions.wm
        if tissue <= 0.0:
            return None
        components = macromolecules.MRSI_CHALLENGE["components"]
        c_tcr = truth.conc.Cr + truth.conc.PCr
        freqs, amps, gauss = [], [], []
        for name, groups, couplings, width_ppm in self.SYSTEMS:
            anchor = components[self.ANCHORS[name]]
            weighted = fractions.gm * anchor["gm"] + fractions.wm * anchor["wm"]
            weight = truth.mm.kappa * weighted * c_tcr / groups[0][1]
            key = (id(engine), id(sequence), name)
            if key not in self._compiled:
                shifts = tuple(ppm for ppm, protons in groups for _ in range(protons))
                spins = len(shifts)
                j = [[0.0] * spins for _ in range(spins)]
                offsets = [sum(protons for _, protons in groups[:k]) for k in range(len(groups))]
                for a, b, j_hz in couplings:
                    for i in range(offsets[a], offsets[a] + groups[a][1]):
                        for k in range(offsets[b], offsets[b] + groups[b][1]):
                            j[i][k] = j[k][i] = j_hz
                system = SpinSystem(name=name, nucleus="1H", shifts_ppm=shifts,
                                    j_hz=tuple(map(tuple, j)), scale=1.0,
                                    gamma_mhz_t=f0_mhz / engine.b0_t)
                self._compiled[key] = engine.compiled(sequence, system)
            from keras import ops
            fn = self._compiled[key]
            line_f, line_a = fn(ops.convert_to_tensor(np.float32(truth.mm.t1_s)),
                                ops.convert_to_tensor(np.float32(truth.mm.t2_s)),
                                ops.convert_to_tensor(np.float32(truth.b1_scale)))
            line_f = np.asarray(ops.convert_to_numpy(line_f))
            line_a = np.asarray(ops.convert_to_numpy(line_a)) * weight
            keep = np.abs(line_a) > 1e-9
            freqs.append(line_f[keep])
            amps.append(line_a[keep])
            gauss.append(np.full(int(keep.sum()), width_ppm * f0_mhz, dtype=np.float32))
        return (np.concatenate(freqs).astype(np.float32),
                np.concatenate(amps).astype(np.complex64),
                np.concatenate(gauss))
