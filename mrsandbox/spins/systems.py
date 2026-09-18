####################################################################################################
#                                            systems.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Spin systems and their operator algebra on the 2^n product basis. A SpinSystem is       #
#          immutable and hashable (so operator stacks cache outside jit) and carries its gamma,    #
#          making it self-contained; its Hamiltonian uses FULL (non-secular) scalar coupling, so   #
#          strong-coupling behaviour is exact at any field. Constants are numpy complex64; convert #
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
            deuterium, 3/2 for sodium); every spin of a system is one nucleus.
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
    t1_s: float = dataclasses.field(default=None, compare=False)
    t2_s: float = dataclasses.field(default=None, compare=False)

    #*******************#
    #   from database   #
    #*******************#
    @classmethod
    def from_database(cls, metabolite, database=None, source="govindaraju", temperature_c=37.0,
                      ph=7.0):
        """All coupled groups of `metabolite` as SpinSystems - simulate each, sum the
        lines. `source` names the table (the 1H Govindaraju set, or "dmi" for the
        deuterium set); the nucleus comes from the record. Away from 37 C and pH 7,
        where the values are typed, every proton moves by the published change of its
        group and every coupling the table holds by its own (Database.shift_change and
        coupling_change)."""
        db = database if database is not None else Database()
        record = db.spin_system(metabolite, source=source)
        nucleus = record.get("nucleus", "1H")
        systems = []
        for group in record["groups"]:
            labels = list(group["spins"])
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
                systems.append(
                    cls(
                        name=f"{metabolite}.{group['name']}{suffix}",
                        nucleus=nucleus,
                        shifts_ppm=tuple(shifts[i] for i in part),
                        j_hz=tuple(map(tuple, j[np.ix_(part, part)])),
                        scale=float(group.get("scale", 1.0)) * count,
                        gamma_mhz_t=db.gamma(nucleus),
                        spin=db.spin(nucleus),
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
    def dim(self):
        """Hilbert-space dimension (2 I + 1)**n."""
        return int(round(2 * self.spin + 1)) ** self.n

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
        """
        op = self.operators()
        nu = np.asarray(self.shifts_ppm) * self.gamma_mhz_t * float(b0_t)
        h = np.tensordot(nu.astype(np.complex64), op.iz, axes=1)
        j = np.asarray(self.j_hz)
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if j[a, b] != 0.0:
                    h = h + np.complex64(j[a, b]) * (
                        op.ix[a] @ op.ix[b] + op.iy[a] @ op.iy[b] + op.iz[a] @ op.iz[b]
                    )
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
        fx, fy, fz: Total-spin operators, (d, d) complex64.
        fplus: Detection operator Fx + i Fy, (d, d) complex64 - its support lies on
            coherence order +1, so Tr(F+ rho) reads out the -1 coherences of rho.
        coherence_order: Integer matrix q[a, b] = m_a - m_b, (d, d) - a coherence filter
            is a boolean mask on it.
        mz: Per-spin magnetic quantum numbers on the product basis, (n, d) float32: the
            diagonals of iz, which are also the eigenvalues of ix in the hadamard basis.
        hadamard: The transverse eigenbasis H, (d, d) complex64, the Kronecker product
            of the per-spin eigenvector matrices of Ix ordered like Iz (for spins 1/2
            the n-fold Hadamard matrix): H Iz_i H^H = Ix_i for every spin, so any sum
            of Ix_i is diagonalized by H with the matching sum of mz rows as
            eigenvalues, and transverse rotations need no eigendecomposition.
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
        single = cls._single(system.spin)
        ix, iy, iz = (
            np.stack([cls._embed(single[axis], i, system.n) for i in range(system.n)])
            for axis in "xyz"
        )
        fx, fy, fz = ix.sum(axis=0), iy.sum(axis=0), iz.sum(axis=0)
        m = np.real(np.diag(fz))
        order = np.rint(m[:, None] - m[None, :]).astype(np.int32)
        mz = np.real(np.diagonal(iz, axis1=1, axis2=2)).astype(np.float32)
        # the eigenvectors of Ix, one column per m in the order of Iz (eigh returns
        # them ascending; the column phases are free, a diagonal one cancels in
        # V diag(m) V^H): the n-fold Hadamard for spins 1/2
        _, basis = np.linalg.eigh(single["x"])
        basis = basis[:, ::-1].astype(np.complex64)
        hadamard = np.array([[1.0]], dtype=np.complex64)
        for _ in range(system.n):
            hadamard = np.kron(hadamard, basis)
        return cls(ix=ix, iy=iy, iz=iz, fx=fx, fy=fy, fz=fz, fplus=fx + 1j * fy, coherence_order=order,
                   mz=mz, hadamard=hadamard)

    #*******************#
    #   single spin     #
    #*******************#
    @staticmethod
    def _single(spin):
        """The x, y, z operators of one spin I on its 2 I + 1 levels, m descending:
        Iz = diag(m) and the ladder I+ with <m + 1|I+|m> = sqrt(I (I + 1) - m (m + 1))."""
        m = spin - np.arange(int(round(2 * spin + 1)))
        plus = np.diag(np.sqrt(spin * (spin + 1.0) - m[1:] * (m[1:] + 1.0)), k=1)
        return {
            "x": ((plus + plus.T) / 2.0).astype(np.complex64),
            "y": ((plus - plus.T) / 2.0j).astype(np.complex64),
            "z": np.diag(m).astype(np.complex64),
        }

    #***************#
    #   embedding   #
    #***************#
    @staticmethod
    def _embed(op, i, n):
        """Kronecker-embed a single-spin operator at position `i` of an n-spin product space."""
        out = np.array([[1.0]], dtype=np.complex64)
        for k in range(n):
            out = np.kron(out, op if k == i else np.eye(op.shape[0], dtype=np.complex64))
        return out
