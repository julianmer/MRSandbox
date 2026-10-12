####################################################################################################
#                                            systems.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Spin systems and their operator algebra on the product basis. A SpinSystem is          #
#          immutable and hashable (so operator stacks cache outside jit) and carries its gamma,    #
#          making it self-contained; its Hamiltonian uses FULL (non-secular) scalar coupling       #
#          between spins of its nucleus, so strong-coupling behaviour is exact at any field, and   #
#          the secular term to a spin of another nucleus. Constants are numpy complex64; convert   #
#          to backend tensors at the point of use.                                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import functools

import numpy as np

from mrsandbox.database import Database
from mrsandbox.database.nuclei import HETERONUCLEI, SPIN


#**************************************************************************************************#
#                                         Class SpinSystem                                         #
#**************************************************************************************************#
#                                                                                                  #
# One coupled spin group with everything needed to simulate it at any field.                       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class SpinSystem:
    """One coupled spin group with everything needed to simulate it at any field.

    Attributes:
        name: Group name, e.g. 'Lac.backbone'.
        nucleus: Resonant nucleus, e.g. '1H'.
        shifts_ppm: Chemical shift per spin (tuple, length n).
        j_hz: Full n-by-n coupling matrix in Hz (tuple of tuples, symmetric).
        scale: Signal scale (equivalent uncoupled spins folded into one).
        gamma_mhz_t: Gyromagnetic ratio of the nucleus, MHz/T.
        spin: Nuclear spin quantum number I of the nucleus (1/2 for protons, 1 for
            deuterium, 3/2 for sodium).
        nuclei: The nucleus of every spin (tuple, length n), or None when every spin is
            `nucleus`. A spin of another nucleus (the 14N and 31P of the choline
            compounds) is never pulsed nor detected: it sits in its own rotating frame,
            couples to the resonant spins by the secular J Iz Sz alone (the frames are
            megahertz apart), and carries its own spin quantum number (14N spin 1, its
            quadrupolar relaxation left out, stated).
        t1_s, t2_s: Optional relaxation times; when set, Delay events relax the system
            (T2 decays every coherence, T1 recovers the populations toward equilibrium).
            Excluded from equality and hash: relaxation never changes the operator
            algebra, so relaxed variants share one cached operator stack.
    """

    name: str
    nucleus: str
    shifts_ppm: tuple
    j_hz: tuple
    scale: float
    gamma_mhz_t: float
    spin: float = 0.5
    nuclei: tuple = None
    t1_s: float = dataclasses.field(default=None, compare=False)
    t2_s: float = dataclasses.field(default=None, compare=False)

    #*******************#
    #   from database   #
    #*******************#
    @classmethod
    def from_database(cls, metabolite, database=None, source="newest", temperature_c=37.0,
                      ph=7.0):
        """All coupled groups of `metabolite` as SpinSystems: simulate each, sum the lines. `source`
        names the table ("newest", every metabolite from the newest publication that has it, its
        gaps from the next, Database.NEWEST_FIRST; a typed table such as "degraaf_2019" or
        "govindaraju", FID-A's; "dmi" for deuterium); the nucleus comes from the record, a group's
        heteronuclei from its "nuclei", and the Govindaraju set's nitrogen and phosphorus spins from
        nuclei.HETERONUCLEI. Away from 37 C and pH 7, where the values are typed, every proton moves
        by the published change of its group and every coupling the table holds by its own
        (Database.shift_change and coupling_change)."""
        db = database if database is not None else Database()
        record = db.spin_system(metabolite, source=source)
        nucleus = record.get("nucleus", "1H")
        tags = HETERONUCLEI.get(metabolite, {}) if source == "govindaraju" else {}
        systems = []
        for group in record["groups"]:
            labels = list(group["spins"])
            kinds = [group.get("nuclei", tags.get(group["name"], {})).get(label, nucleus)
                     for label in labels]
            index = {label: i for i, label in enumerate(labels)}
            j = np.zeros((len(labels), len(labels)))
            for a, b, value in group.get("j", ()):
                j[index[a], index[b]] = j[index[b], index[a]] = value
            typed = [float(group["spins"][label]) for label in labels]
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    if j[a, b] != 0.0:
                        j[a, b] = j[b, a] = j[a, b] + db.coupling_change(
                            metabolite, typed[a], typed[b], temperature_c, ph)
            shifts = [shift + db.shift_change(metabolite, shift, temperature_c, ph)
                      for shift in typed]
            # spins that no coupling connects evolve on their own, and their signals
            # add: every J-connected part is its own system (the dimension is 2^n),
            # and uncoupled spins at one shift are one spin counted that many times
            parts, singles = [], {}
            for part in cls._connected(j):
                if len(part) == 1:
                    singles[shifts[part[0]]] = singles.get(shifts[part[0]], 0) + 1
                else:
                    parts.append((part, 1))
            parts += [([shifts.index(shift)], count) for shift, count in singles.items()]
            for k, (part, count) in enumerate(parts):
                suffix = "" if len(parts) == 1 else f".{k}"
                others = any(kinds[i] != nucleus for i in part)
                systems.append(
                    cls(
                        name=f"{metabolite}.{group['name']}{suffix}",
                        nucleus=nucleus,
                        shifts_ppm=tuple(shifts[i] for i in part),
                        j_hz=tuple(map(tuple, j[np.ix_(part, part)])),
                        scale=float(group.get("scale", 1.0)) * count,
                        gamma_mhz_t=db.gamma(nucleus),
                        spin=db.spin(nucleus),
                        nuclei=tuple(kinds[i] for i in part) if others else None,
                        t1_s=record.get("t1_s"),
                        t2_s=record.get("t2_s"),
                    )
                )
        return tuple(systems)

    @staticmethod
    def _connected(j):
        """The J-connected parts of a coupling matrix, each a sorted index list, in
        order of their first spin."""
        n, seen, parts = len(j), set(), []
        for start in range(n):
            if start in seen:
                continue
            stack, part = [start], []
            while stack:
                a = stack.pop()
                if a not in seen:
                    seen.add(a)
                    part.append(a)
                    stack += [b for b in range(n) if j[a, b] != 0.0 and b not in seen]
            parts.append(sorted(part))
        return parts

    #***********#
    #   sizes   #
    #***********#
    @property
    def n(self):
        """Number of spins."""
        return len(self.shifts_ppm)

    @property
    def resonant(self):
        """Per spin, whether it is of the system's nucleus (pulsed and detected)."""
        if self.nuclei is None:
            return (True,) * self.n
        return tuple(kind == self.nucleus for kind in self.nuclei)

    @property
    def levels(self):
        """Per spin, its number of levels 2 I + 1."""
        return tuple(int(round(2 * (self.spin if own else SPIN[kind]) + 1))
                     for own, kind in zip(self.resonant, self.nuclei or (self.nucleus,) * self.n))

    @property
    def dim(self):
        """Hilbert-space dimension, the product of every spin's 2 I + 1."""
        return int(np.prod(self.levels))

    #******************#
    #   relaxation     #
    #******************#
    def with_relaxation(self, t1_s=None, t2_s=None):
        """The same system with relaxation times attached; Delay events then relax it."""
        return dataclasses.replace(self, t1_s=t1_s, t2_s=t2_s)

    #***************#
    #   operators   #
    #***************#
    def operators(self):
        """The cached operator stack of this system."""
        return Operators.build(self)

    #*****************#
    #   hamiltonian   #
    #*****************#
    def hamiltonian(self, b0_t):
        """Static Hamiltonian at field `b0_t` (tesla), in Hz, numpy complex64 (d, d).

        H = sum_i nu_i Iz_i + sum_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j),
        with nu_i = shift_i * gamma_mhz_t * b0_t (the frame offset is a Context input).
        A spin of another nucleus sits in its own frame (nu_i = 0), and two spins of
        different nuclei couple by J_ij Iz_i Iz_j alone.
        """
        op = self.operators()
        resonant = np.asarray(self.resonant)
        kinds = self.nuclei or (self.nucleus,) * self.n
        nu = np.asarray(self.shifts_ppm) * self.gamma_mhz_t * float(b0_t) * resonant
        h = np.tensordot(nu.astype(np.complex64), op.iz, axes=1)
        j = np.asarray(self.j_hz)
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if j[a, b] != 0.0:
                    coupling = op.iz[a] @ op.iz[b]
                    if kinds[a] == kinds[b]:
                        coupling = coupling + op.ix[a] @ op.ix[b] + op.iy[a] @ op.iy[b]
                    h = h + np.complex64(j[a, b]) * coupling
        return h

    #*******************#
    #   thermal state   #
    #*******************#
    def thermal_rho(self):
        """Thermal equilibrium density operator, high temperature, up to one global constant.

        rho is proportional to (1 + beta Fz) / d; the identity is silent, so the signal
        bearing part is 2 Fz / d (Fz / 2**(n - 1) for spins 1/2), which makes the per
        spin amplitude invariant to the system size (a 6 spin creatine and a 1 spin
        singlet weigh their protons alike). Across nuclei the amplitude per spin then
        scales as I (I + 1), the trace of Fx^2 per spin; the polarization and the
        detection gain of a nucleus are global factors outside the engine (stated).
        """
        return np.complex64(2.0 * self.scale / self.dim) * self.operators().fz


#**************************************************************************************************#
#                                         Class Operators                                          #
#**************************************************************************************************#
#                                                                                                  #
# The cached operator constants of one spin system on the product basis.                           #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Operators:
    """The cached operator constants of one spin system on the product basis.

    Attributes:
        ix, iy, iz: Per-spin operators, (n, d, d) complex64.
        fx, fy, fz: Total-spin operators of the resonant spins (another nucleus is
            never pulsed nor detected), (d, d) complex64.
        fplus: Detection operator Fx + i Fy, (d, d) complex64; its support lies on
            coherence order +1, so Tr(F+ rho) reads out the -1 coherences of rho.
        coherence_order: Integer matrix q[a, b] = m_a - m_b of the resonant spins,
            (d, d); a coherence filter is a boolean mask on it.
        mz: Per-spin magnetic quantum numbers on the product basis, (n, d) float32: the
            diagonals of iz, which are also the eigenvalues of ix in the hadamard basis;
            a spin of another nucleus has a zero row, so no pulse sum turns it.
        hadamard: The transverse eigenbasis H, (d, d) complex64, the Kronecker product
            of the per-spin eigenvector matrices of Ix ordered like Iz (for spins 1/2
            the n-fold Hadamard matrix; the identity on another nucleus): H Iz_i H^H =
            Ix_i for every resonant spin, so any sum of their Ix_i is diagonalized by H
            with the matching sum of mz rows as eigenvalues, and transverse rotations
            need no eigendecomposition.
    """

    ix: np.ndarray
    iy: np.ndarray
    iz: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray
    fplus: np.ndarray
    coherence_order: np.ndarray
    mz: np.ndarray
    hadamard: np.ndarray

    #***********#
    #   build   #
    #***********#
    @classmethod
    @functools.lru_cache(maxsize=256)
    def build(cls, system):
        """Build (and cache) the operator stack of a hashable SpinSystem; relaxed
        variants of one structural system hit the same entry."""
        # the x, y, z operators of one spin on its levels, m descending from I: Iz = diag(m) and
        # the ladder I+ with <m + 1|I+|m> = sqrt(I (I + 1) - m (m + 1))
        singles = []
        for count in system.levels:
            spin = (count - 1) / 2.0
            m = spin - np.arange(count)
            plus = np.diag(np.sqrt(spin * (spin + 1.0) - m[1:] * (m[1:] + 1.0)), k=1)
            singles.append({"x": ((plus + plus.T) / 2.0).astype(np.complex64),
                            "y": ((plus - plus.T) / 2.0j).astype(np.complex64),
                            "z": np.diag(m).astype(np.complex64)})
        ix, iy, iz = (
            np.stack([cls._embed(singles[i][axis], i, system.levels) for i in range(system.n)])
            for axis in "xyz"
        )
        resonant = np.asarray(system.resonant, dtype=np.float32)
        fx, fy, fz = (np.tensordot(resonant.astype(np.complex64), op, axes=1)
                      for op in (ix, iy, iz))
        m = np.real(np.diag(fz))
        order = np.rint(m[:, None] - m[None, :]).astype(np.int32)
        mz = (np.real(np.diagonal(iz, axis1=1, axis2=2)) * resonant[:, None]).astype(np.float32)
        # the eigenvectors of Ix, one column per m in the order of Iz (eigh returns
        # them ascending; the column phases are free, a diagonal one cancels in
        # V diag(m) V^H): the n-fold Hadamard for spins 1/2; another nucleus keeps its basis
        hadamard = np.array([[1.0]], dtype=np.complex64)
        for single, own in zip(singles, system.resonant):
            basis = np.eye(single["x"].shape[0], dtype=np.complex64)
            if own:
                basis = np.linalg.eigh(single["x"])[1][:, ::-1].astype(np.complex64)
            hadamard = np.kron(hadamard, basis)
        return cls(ix=ix, iy=iy, iz=iz, fx=fx, fy=fy, fz=fz, fplus=fx + 1j * fy,
                   coherence_order=order, mz=mz, hadamard=hadamard)


    #***************#
    #   embedding   #
    #***************#
    @staticmethod
    def _embed(op, i, levels):
        """Kronecker-embed a single-spin operator at position `i` of the product space of
        spins with `levels` levels each."""
        out = np.array([[1.0]], dtype=np.complex64)
        for k, count in enumerate(levels):
            out = np.kron(out, op if k == i else np.eye(count, dtype=np.complex64))
        return out
