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
from keras import ops

from mrsandbox.database import macromolecules
from mrsandbox.io import VarianFid
from mrsandbox.spins import SpinSystem


#*****************#
#   nearest peak  #
#*****************#
def _nearest(positions, ppm):
    """The index of the nearest of the sorted `positions` for every value of `ppm`."""
    return np.searchsorted(0.5 * (positions[1:] + positions[:-1]), ppm)


#*****************#
#   echo timing   #
#*****************#
def _echo_time_s(sequence):
    """Transverse (echo) time of a sequence, for the measured-shape TE decay: the te_s
    attribute where the design carries one (STEAM's mixing time is longitudinal and
    excluded), te1_s plus te2_s for PRESS designs, the tau sum for MEGA designs, and
    the sum of Delay events as the fallback (exact for gradient-crushed echoes)."""
    if hasattr(sequence, "te_s"):
        return float(sequence.te_s)
    if hasattr(sequence, "te1_s"):
        return float(sequence.te1_s) + float(sequence.te2_s)
    if hasattr(sequence, "taus_s"):
        return float(sum(sequence.taus_s))
    from mrsandbox.sequences.events import Delay

    return float(sum(float(event.t_s) for event in sequence.events if isinstance(event, Delay)))


#*******************#
#   peak position   #
#*******************#
def _peak_ppm(label):
    """Position of a consensus peak from its frequency label (M3.4-3.6 to the midpoint)."""
    parts = label.lstrip("M").split("-")
    return sum(float(part) for part in parts) / len(parts)


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
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        """Return (freqs_hz, amps, gauss_hz), or None when line-silent. The shared
        source signature: the placed scan.Voxel (fractions and geometry), the realized
        truth, and the spectrometer frequency in MHz. Sources built from static
        parameterizations may ignore the engine and sequence; sources built from spin
        systems run them through both, so TE behaviour and editing emerge."""
        return None

    #*********#
    #   fid   #
    #*********#
    def fid(self, voxel, truth, f0_mhz, t_s, sequence=None):
        """Return the complex time domain contribution on the uniform grid `t_s`, or
        None when silent. `sequence` is the scan's sequence, so a measured source can
        carry echo time behaviour."""
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
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        tissue = voxel.fractions.gm + voxel.fractions.wm
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
    shapes instead of Gaussians, decaying with the sequence's echo time.

    The 3 T human metabolite-nulled STEAM spectrum from the consensus data collection
    (fetched on first use) is mapped into the simulation frame: conjugated to our sign
    convention, time scaled from the measured 123.245 MHz to the simulated field, and
    remodulated from its water centred transmitter frame (4.7 ppm, verified against the
    consensus peak positions) to the 0 ppm reference frame. The signal is normalized so
    the drawn mm.level_mm leaf keeps the ParameterizedMM meaning: total MM area equals
    the level times the parameterization's summed proton areas.

    The measurement was acquired at TE 8 ms; for any other sequence every frequency bin
    is attenuated by exp(-(TE - 8 ms) / T2(ppm)), with T2(ppm) the value of the nearest
    peak in the consensus 3 T human column (Landheer 2020 in the typed Appendix S1
    Table S3, the 3 T study reporting uncertainties) and the echo time read from the
    sequence's own timings. With a TR every bin is scaled by the steady-state factor
    of the nearest component: a one-spin probe at the component's position with the
    drawn mm.t1 and mm.t2 runs the sequence itself with its TR and at a TR of a
    thousand seconds, and the ratio of the two is the factor, so partial recovery and
    the inversion of the recovered magnetization by refocusing pulses follow exactly
    the law the metabolites obey (Sequence.run). Stated approximations: the T2 and T1
    maps are nearest-peak step functions, and the editing-band response of the
    measured shapes is not modelled (couplings are not resolvable in a measured sum;
    CoupledMM carries co-editing).
    """

    TOTAL_PROTON_AREAS = 11.39  # the ParameterizedMM MM components summed
    MEASURED_TE_S = 0.008  # the consensus 3 T human STEAM acquisition

    #**********#
    #   init   #
    #**********#
    def __init__(self, center_ppm=4.7):
        self.center_ppm = center_ppm
        self._measured = None
        self._t2_map = None
        self._repetition = None

    #*********#
    #   fid   #
    #*********#
    def fid(self, voxel, truth, f0_mhz, t_s, sequence=None):
        tissue = voxel.fractions.gm + voxel.fractions.wm
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
        if sequence is not None:
            signal = self._sequence_response(signal, t, f0_mhz, sequence, voxel, truth)
        scale = truth.mm.level_mm * tissue * self.TOTAL_PROTON_AREAS / np.conj(measured[0])
        return (signal * scale).astype(np.complex64)

    #*************************#
    #   sequence response     #
    #*************************#
    def _sequence_response(self, signal, t, f0_mhz, sequence, voxel, truth):
        """Every frequency bin scaled by what the sequence does beyond the measurement,
        via one FFT round trip: the echo decay exp(-(TE - 8 ms) / T2(ppm)) with the
        typed T2 of the nearest consensus 3 T peak, and with a TR the steady-state
        factor of the nearest component (`_repetition_factors`)."""
        dwell = float(t[1] - t[0])
        ppm = (np.fft.fftfreq(t.size, d=dwell) % (1.0 / dwell)) / f0_mhz
        factor = np.ones(t.size)
        delta = _echo_time_s(sequence) - self.MEASURED_TE_S
        if abs(delta) >= 1e-9:
            if self._t2_map is None:
                table = macromolecules.CONSENSUS_T2["t2_ms"]
                rows = sorted((_peak_ppm(peak), value[0] / 1000.0)
                              for peak, entries in table.items()
                              if (value := entries.get("landheer_2020")) is not None)
                self._t2_map = (np.asarray([ppm for ppm, _ in rows]),
                                np.asarray([t2 for _, t2 in rows]))
            positions, t2_s = self._t2_map
            factor = factor * np.exp(-delta / t2_s[_nearest(positions, ppm)])
        if sequence.tr_s is not None:
            positions, factors = self._repetition_factors(sequence, voxel, truth, f0_mhz)
            factor = factor * factors[_nearest(positions, ppm)]
        if np.all(factor == 1.0):
            return signal
        return np.fft.ifft(np.fft.fft(signal) * factor)

    def _repetition_factors(self, sequence, voxel, truth, f0_mhz):
        """The steady-state factor of every component under the sequence's TR, as
        (positions ppm, complex factors): a one-spin probe at each component's
        position with the drawn mm.t1 and mm.t2 runs the design at its TR and at a
        TR of a thousand seconds (the same compiled graph, TR being a timing), and
        the ratio of its detected amplitudes is the factor. Cached per truth and per
        engine, sequence, and transmit frame, like CoupledMM's lines."""
        from mrsandbox.sequences.events import Sequence

        engine = voxel.scan.engine
        b1_scale, offset_hz = voxel.b1_input(truth), voxel.offset_hz(truth)
        cached = self._repetition
        if cached is None or cached[0] is not truth:
            cached = self._repetition = (truth, {})
        key = (engine, sequence, voxel.frame(truth)[2], offset_hz)
        if key not in cached[1]:
            relaxed = Sequence(*sequence.events, tr_s=1000.0, cycles=sequence.cycles)
            positions, factors = [], []
            for name, groups, _, _ in CoupledMM.SYSTEMS:
                probe = SpinSystem(name=name, nucleus="1H", shifts_ppm=(groups[0][0],),
                                   j_hz=((0.0,),), scale=1.0, gamma_mhz_t=f0_mhz / engine.b0_t)
                args = tuple(ops.convert_to_tensor(np.float32(v)) for v in
                             (truth.mm.t1[name], truth.mm.t2[name], b1_scale, offset_hz))
                _, at_tr = engine.compiled(sequence, probe)(*args)
                _, at_infinity = engine.compiled(relaxed, probe)(*args)
                positions.append(groups[0][0])
                factors.append(complex(ops.convert_to_numpy(ops.sum(at_tr)))
                               / complex(ops.convert_to_numpy(ops.sum(at_infinity))))
            order = np.argsort(positions)
            cached[1][key] = (np.asarray(positions)[order], np.asarray(factors)[order])
        return cached[1][key]

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
#                                        Class ChallengeMM                                         #
#**************************************************************************************************#
#                                                                                                  #
# The macromolecule basis of the 2024 MRSI challenge: nine component FIDs on its amplitude model.  #
#                                                                                                  #
#**************************************************************************************************#
class ChallengeMM(MeasuredMM):
    """The macromolecule basis of the 2024 MRSI challenge: nine component FIDs on its
    amplitude model, b = kappa x r x c_tCr per component with the challenge's own
    kappa of 0.4 and the tissue-weighted ratios r (database.macromolecules
    MRSI_CHALLENGE), the drawn creatine plus phosphocreatine as c_tCr.

    The basis is the public example dataset of the challenge (LaMaster et al.,
    arXiv:2609.02637; basis_macromol_3T, the mrsi_challenge resources): one FID
    per component, 32768 points of 80.389 us, linewidths adjusted to typical in vivo
    MM spectra at 3 T, scaled in the proton units of the challenge's metabolite basis
    (a first point of three for the creatine methyl), which are the engine's. The
    files carry upfield as negative frequency, our sign, at the 123.24 MHz of their
    own headers (the dataset's readme gives 127.73 MHz for the metabolite basis; the
    nine positions fit 123.24 to 0.001 ppm and 127.73 not at all); the time axis is
    scaled to the simulated field and the frame moves from the basis reference,
    4.647 ppm as those positions place it, to 0 ppm. The basis is a TE 0 FID-MRSI
    shape; the sequence acts on it as on the measured spectrum (the echo decay per
    frequency bin from TE 0 and the steady state of a TR, `MeasuredMM`), with the
    same stated approximations. mm.kappa and mm.level_mm are not read.
    """

    MEASURED_TE_S = 0.0
    BASIS_MHZ = 123.24
    BASIS_DWELL_S = 8.03892e-05
    REFERENCE_PPM = 4.647
    FILES = {"M0.91": "mrs_mm_0_9", "M1.21": "mrs_mm_1_21", "M1.43": "mrs_mm_1_43",
             "M1.67": "mrs_mm_1_67", "M2.04": "mrs_mm_2_04", "M2.26": "mrs_mm_2_26",
             "M2.99": "mrs_mm_2_99", "M3.21": "mrs_mm_3_21", "M3.77": "mrs_mm_3_77"}

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        super().__init__(center_ppm=self.REFERENCE_PPM)

    #*********#
    #   fid   #
    #*********#
    def fid(self, voxel, truth, f0_mhz, t_s, sequence=None):
        fractions = voxel.fractions
        if fractions.gm + fractions.wm <= 0.0:
            return None
        basis = self._load()
        t = np.asarray(t_s, dtype=np.float64)
        # sample the basis at times scaled to the simulated field, ppm-faithful
        t_basis = t * (f0_mhz / self.BASIS_MHZ)
        grid = np.arange(basis.shape[1]) * self.BASIS_DWELL_S
        c_tcr = voxel.resolve(truth.conc.Cr) + voxel.resolve(truth.conc.PCr)
        signal = np.zeros(t.size, dtype=np.complex128)
        for row, name in zip(basis, self.FILES):
            ratio = macromolecules.MRSI_CHALLENGE["components"][name]
            weight = fractions.gm * ratio["gm"] + fractions.wm * ratio["wm"]
            signal += weight * (np.interp(t_basis, grid, row.real, right=0.0)
                                + 1j * np.interp(t_basis, grid, row.imag, right=0.0))
        signal *= macromolecules.MRSI_CHALLENGE["kappa"] * c_tcr
        signal = signal * np.exp(2j * np.pi * self.center_ppm * f0_mhz * t)
        if sequence is not None:
            signal = self._sequence_response(signal, t, f0_mhz, sequence, voxel, truth)
        return signal.astype(np.complex64)

    #**********#
    #   load   #
    #**********#
    def _load(self):
        if self._measured is None:
            import nibabel

            from mrsandbox.resources import Resources

            self._measured = np.stack([
                np.asarray(nibabel.load(Resources.path(f"mrsi_challenge/{stem}")).dataobj).squeeze()
                for stem in self.FILES.values()])
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
    counts of the chains (3:1 and 2:2) are stated assumptions. Each system relaxes with
    its own mm.t1 and mm.t2 leaves, typed from the consensus Appendix S1 3 T human
    columns where the tables carry a value (see default_params) and stated otherwise,
    so TE behaviour is component-faithful.
    """

    # (name, groups as (ppm, protons), coupled group pairs, gauss width ppm); the J of
    # each pair is read from the cited CONSENSUS_COUPLINGS table by peak position
    SYSTEMS = (
        ("mm091", ((0.91, 3), (2.04, 1)), ((0, 1),), 0.14),
        ("mm121", ((1.21, 3), (4.24, 1)), ((0, 1),), 0.15),
        ("mm143", ((1.43, 3), (4.32, 1)), ((0, 1),), 0.17),
        ("mm167", ((1.67, 2), (2.99, 2)), ((0, 1),), 0.15),
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
        self._lines = None

    #***********#
    #   lines   #
    #***********#
    def lines(self, engine, sequence, voxel, truth, f0_mhz):
        fractions = voxel.fractions
        tissue = fractions.gm + fractions.wm
        if tissue <= 0.0:
            return None
        components = macromolecules.MRSI_CHALLENGE["components"]
        c_tcr = voxel.resolve(truth.conc.Cr) + voxel.resolve(truth.conc.PCr)
        freqs, amps, gauss = [], [], []
        for name, groups, couplings, width_ppm in self.SYSTEMS:
            anchor = components[self.ANCHORS[name]]
            weighted = fractions.gm * anchor["gm"] + fractions.wm * anchor["wm"]
            weight = truth.mm.kappa * weighted * c_tcr / groups[0][1]
            line_f, line_a = self._unit_lines(engine, sequence, voxel, truth, f0_mhz,
                                              name, groups, couplings)
            line_a = line_a * weight
            keep = np.abs(line_a) > 1e-9
            freqs.append(line_f[keep])
            amps.append(line_a[keep])
            gauss.append(np.full(int(keep.sum()), width_ppm * f0_mhz, dtype=np.float32))
        return (np.concatenate(freqs).astype(np.float32),
                np.concatenate(amps).astype(np.complex64),
                np.concatenate(gauss))

    #****************#
    #   coupling j   #
    #****************#
    @staticmethod
    def _coupling_j_hz(ppm_a, ppm_b):
        """The consensus Table 1 J of the chain whose partner positions lie nearest
        (papers place the peaks 0.01 to 0.04 ppm apart), read from the cited
        CONSENSUS_COUPLINGS table so it and the simulated couplings cannot drift."""
        a, b, j_hz = min(macromolecules.CONSENSUS_COUPLINGS,
                         key=lambda row: abs(row[0] - ppm_a) + abs(row[1] - ppm_b))
        if abs(a - ppm_a) + abs(b - ppm_b) > 0.1:
            raise ValueError(f"no consensus coupling near ({ppm_a}, {ppm_b}) ppm")
        return j_hz

    #****************#
    #   unit lines   #
    #****************#
    def _unit_lines(self, engine, sequence, voxel, truth, f0_mhz, name, groups, couplings):
        """One system's unweighted numpy lines for one truth, cached per truth and per
        transmit frame: the draw fixes relaxation, the voxel its b1 and offset, and the
        position otherwise only scales amplitudes, so one truth serving many voxels
        replays each system once per distinct frame. Compilation itself is cached on
        the engine, keyed by the sequence and the structural system."""
        cached = self._lines
        if cached is None or cached[0] is not truth:
            cached = self._lines = (truth, {})
        b1_scale, offset_hz = voxel.b1_input(truth), voxel.offset_hz(truth)
        shift = None if voxel.shift_mm is None else tuple(voxel.shift_mm.values())
        key = (engine, sequence, name, voxel.frame(truth)[2], offset_hz, shift)
        if key in cached[1]:
            return cached[1][key]
        shifts = tuple(ppm for ppm, protons in groups for _ in range(protons))
        spins = len(shifts)
        j = [[0.0] * spins for _ in range(spins)]
        offsets = [sum(protons for _, protons in groups[:k]) for k in range(len(groups))]
        for a, b in couplings:
            j_hz = self._coupling_j_hz(groups[a][0], groups[b][0])
            for i in range(offsets[a], offsets[a] + groups[a][1]):
                for k in range(offsets[b], offsets[b] + groups[b][1]):
                    j[i][k] = j[k][i] = j_hz
        system = SpinSystem(name=name, nucleus="1H", shifts_ppm=shifts,
                            j_hz=tuple(map(tuple, j)), scale=1.0,
                            gamma_mhz_t=f0_mhz / engine.b0_t)
        fn = engine.compiled(sequence, system, shift_mm=voxel.shift_mm)
        line_f, line_a = fn(*(ops.convert_to_tensor(np.float32(v)) for v in
                              (truth.mm.t1[name], truth.mm.t2[name], b1_scale, offset_hz)))
        lines = (np.asarray(ops.convert_to_numpy(line_f)),
                 np.asarray(ops.convert_to_numpy(line_a)))
        cached[1][key] = lines
        return lines
