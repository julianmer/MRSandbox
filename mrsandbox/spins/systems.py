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
        t1_s, t2_s: Optional relaxation times; when set, Delay events relax the system
            (T2 decays every coherence, T1 recovers the populations toward equilibrium).
    """

    name: str
    nucleus: str
    shifts_ppm: tuple
    j_hz: tuple
    scale: float
    gamma_mhz_t: float
    t1_s: float = None
    t2_s: float = None

    #*******************#
    #   from database   #
    #*******************#
    @classmethod
    def from_database(cls, metabolite, database=None):
        """All coupled groups of `metabolite` as SpinSystems - simulate each, sum the lines."""
        db = database if database is not None else Database()
        record = db.spin_system(metabolite)
        nucleus = record.get("nucleus", "1H")
        systems = []
        for group in record["groups"]:
            labels = list(group["spins"])
            index = {label: i for i, label in enumerate(labels)}
            j = np.zeros((len(labels), len(labels)))
            for a, b, value in group.get("j", ()):
                j[index[a], index[b]] = j[index[b], index[a]] = value
            systems.append(
                cls(
                    name=f"{metabolite}.{group['name']}",
                    nucleus=nucleus,
                    shifts_ppm=tuple(float(group["spins"][label]) for label in labels),
                    j_hz=tuple(map(tuple, j)),
                    scale=float(group.get("scale", 1.0)),
                    gamma_mhz_t=db.gamma(nucleus),
                )
            )
        return tuple(systems)

    #***********#
    #   sizes   #
    #***********#
    @property
    def n(self):
        """Number of spins."""
        return len(self.shifts_ppm)

    @property
    def dim(self):
        """Hilbert-space dimension 2**n."""
        return 2 ** self.n

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
    def hamiltonian(self, b0_t, offset_hz=0.0, ref_ppm=0.0):
        """Static Hamiltonian at field `b0_t` (tesla), in Hz, numpy complex64 (d, d).

        H = sum_i nu_i Iz_i + sum_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j),
        with nu_i = (shift_i - ref_ppm) * gamma_mhz_t * b0_t - offset_hz.
        """
        op = self.operators()
        nu = (np.asarray(self.shifts_ppm) - ref_ppm) * self.gamma_mhz_t * float(b0_t) - offset_hz
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
        bearing part is Fz / 2**(n - 1), which makes the per proton amplitude invariant to
        the system size (a 6 spin creatine and a 1 spin singlet weigh their protons alike).
        """
        return np.complex64(self.scale / 2 ** (self.n - 1)) * self.operators().fz


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
    """

    ix: np.ndarray
    iy: np.ndarray
    iz: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray
    fplus: np.ndarray
    coherence_order: np.ndarray

    #***********#
    #   build   #
    #***********#
    @classmethod
    @functools.cache
    def build(cls, system):
        """Build (and cache) the operator stack of a hashable SpinSystem."""
        pauli = {
            "x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex64),
            "y": np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex64),
            "z": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex64),
        }
        ix, iy, iz = (
            np.stack([cls._embed(pauli[axis], i, system.n) for i in range(system.n)])
            for axis in "xyz"
        )
        fx, fy, fz = ix.sum(axis=0), iy.sum(axis=0), iz.sum(axis=0)
        m = np.real(np.diag(fz))
        order = np.rint(m[:, None] - m[None, :]).astype(np.int32)
        return cls(ix=ix, iy=iy, iz=iz, fx=fx, fy=fy, fz=fz, fplus=fx + 1j * fy, coherence_order=order)

    #***************#
    #   embedding   #
    #***************#
    @staticmethod
    def _embed(op, i, n):
        """Kronecker-embed a single-spin operator at position `i` of an n-spin product space."""
        out = np.array([[1.0]], dtype=np.complex64)
        for k in range(n):
            out = np.kron(out, op if k == i else np.eye(2, dtype=np.complex64))
        return out
