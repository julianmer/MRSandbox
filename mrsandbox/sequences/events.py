####################################################################################################
#                                            events.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Sequences as event lists. An Event knows how to apply itself to a density matrix        #
#          (polymorphic execution: pulses rotate, delays evolve, filters mask coherence orders),   #
#          so ANY ideal-pulse design runs through the same propagator - and every timing is a free #
#          tensor, differentiable by construction.                                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import abc
import math

import numpy as np

from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.backend import Backend


#**************************************************************************************************#
#                                           Class Event                                            #
#**************************************************************************************************#
#                                                                                                  #
# One step of a pulse sequence, applied to a density matrix in a system context.                   #
#                                                                                                  #
#**************************************************************************************************#
class Event(abc.ABC):
    """One step of a pulse sequence, applied to a density matrix in a system context."""

    #***********#
    #   apply   #
    #***********#
    @abc.abstractmethod
    def apply(self, rho, ctx):
        """Return the density matrix after this event.

        Args:
            rho: Density matrix, complex64 (d, d).
            ctx: The Simulator-built context of one spin system (Hamiltonian eigensystem,
                total-spin operators, coherence orders).
        """

    #**************#
    #   sandwich   #
    #**************#
    @staticmethod
    def _sandwich(u, rho):
        """U rho U^H."""
        return ops.einsum("...ij,...jk,...lk->...il", u, rho, ops.conj(u))


#**************************************************************************************************#
#                                           Class Pulse                                            #
#**************************************************************************************************#
#                                                                                                  #
# Ideal (hard) pulse: a rotation by flip degrees about an axis in the transverse plane.            #
#                                                                                                  #
#**************************************************************************************************#
class Pulse(Event):
    """Ideal (hard) pulse: a rotation by flip degrees about an axis in the transverse plane.

    Args:
        flip_deg: Flip angle in degrees.
        phase_deg: Axis phase in degrees (0 = x, 90 = y).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg, phase_deg=0.0):
        self.flip_deg = flip_deg
        self.phase_deg = phase_deg

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        phase = math.radians(self.phase_deg)
        axis = math.cos(phase) * ctx.fx + math.sin(phase) * ctx.fy
        u = Backend.expm_herm(axis, ctx.b1_scale * self.flip_deg / 360.0)
        return self._sandwich(u, rho)


#**************************************************************************************************#
#                                       Class SelectivePulse                                       #
#**************************************************************************************************#
#                                                                                                  #
# Ideal frequency-selective pulse: rotates only the spins inside its band.                         #
#                                                                                                  #
#**************************************************************************************************#
class SelectivePulse(Event):
    """Ideal frequency-selective pulse: rotates only the spins inside its band(s).

    The ideal-selective approximation: a spin is rotated in full when its resonance lies
    within bandwidth_hz around the pulse offset, and untouched otherwise. This is the
    primitive behind water suppression trains and, later, spectral editing pulses. Offset
    and bandwidth may be traced tensors, so a network can emit them. A tuple of offsets
    is the ideal multi-band pulse: one band per centre, the validated idealization of
    the dual-lobe cosine-modulated waveforms of Hadamard-encoded editing.

    Args:
        flip_deg: Flip angle in degrees.
        offset_ppm: Pulse centre in ppm (water sits at 4.65), or a tuple of centres.
        bandwidth_hz: Full width of each band in Hz.
        phase_deg: Axis phase in degrees (0 = x, 90 = y).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg, offset_ppm, bandwidth_hz, phase_deg=0.0):
        self.flip_deg = flip_deg
        self.offset_ppm = tuple(offset_ppm) if isinstance(offset_ppm, (tuple, list)) else (offset_ppm,)
        self.bandwidth_hz = bandwidth_hz
        self.phase_deg = phase_deg

    #*************#
    #   apply     #
    #*************#
    def apply(self, rho, ctx):
        half_width = ops.convert_to_tensor(self.bandwidth_hz) / 2.0
        inside = None
        for offset in self.offset_ppm:
            offset_hz = ops.convert_to_tensor(offset) * ctx.f0_mhz
            band = ops.abs(ctx.nu_hz - offset_hz) <= half_width
            inside = band if inside is None else ops.logical_or(inside, band)
        inside = ops.cast(inside, "float32")
        phase = math.radians(self.phase_deg)
        wx = nops.cast(inside * math.cos(phase), "complex64")
        wy = nops.cast(inside * math.sin(phase), "complex64")
        axis = ops.einsum("i,ijk->jk", wx, ctx.ix) + ops.einsum("i,ijk->jk", wy, ctx.iy)
        u = Backend.expm_herm(axis, ctx.b1_scale * self.flip_deg / 360.0)
        return self._sandwich(u, rho)


#**************************************************************************************************#
#                                        Class ShapedPulse                                         #
#**************************************************************************************************#
#                                                                                                  #
# A real RF waveform: piecewise-constant propagation of the full Hamiltonian plus the RF term.     #
#                                                                                                  #
#**************************************************************************************************#
class ShapedPulse(Event):
    """A real RF waveform: piecewise-constant propagation of the full Hamiltonian plus
    the RF term, the standard GAMMA/denmatsim treatment. Chemical shift and J evolve
    DURING the pulse, so finite-pulse effects and the excitation profile are physical
    rather than the ideal-band approximation of SelectivePulse.

    The envelope is any series of relative amplitudes (a traced tensor works, so a
    network can emit the shape); it is calibrated so an on-resonance uncoupled spin is
    flipped by exactly flip_deg. A carrier at offset_ppm is carried as an RF phase ramp
    in the reference frame, which keeps detection frame-free.

    With `gradient_hz_per_mm` set, the pulse plays under a slice-selection gradient:
    every isochromat sees its resonance shifted by gradient times position, so the
    excited slab, its profile, and the chemical shift displacement of off-resonance
    spins all emerge. Pair it with a refocusing Gradient of minus half the pulse's
    gradient moment, as on a scanner.

    Args:
        flip_deg: On-resonance flip angle in degrees.
        duration_s: Total pulse duration in seconds.
        envelope: Relative amplitude samples, any positive scale.
        offset_ppm: Carrier position in ppm (water sits at 4.65).
        phase_deg: RF phase at t = 0 in degrees (0 = x, 90 = y).
        gradient_hz_per_mm: Slice gradient strength; 0 plays without a gradient.
        gradient_axis: The slice gradient axis, "x", "y", or "z".
        extent_mm: Isochromat span when a gradient is set.
        frequency_hz: Optional per-step frequency modulation (adiabatic sweeps).
        peak_hz: Optional peak RF amplitude in Hz; when set the envelope is scaled to
            it directly instead of being flip calibrated (adiabatic pulses are defined
            by their peak B1, not a nutation angle; flip_deg is ignored then).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg, duration_s, envelope, offset_ppm=0.0, phase_deg=0.0,
                 gradient_hz_per_mm=0.0, gradient_axis="x", extent_mm=20.0,
                 frequency_hz=None, peak_hz=None):
        if gradient_axis not in ("x", "y", "z"):
            raise ValueError(f"unknown gradient axis {gradient_axis!r}; known: x, y, z")
        self.flip_deg = flip_deg
        self.duration_s = duration_s
        self.envelope = envelope
        self.offset_ppm = offset_ppm
        self.phase_deg = phase_deg
        self.gradient_hz_per_mm = gradient_hz_per_mm
        self.gradient_axis = gradient_axis
        self.extent_mm = extent_mm
        self.frequency_hz = frequency_hz
        self.peak_hz = peak_hz

    #***************#
    #   gaussian    #
    #***************#
    @classmethod
    def gaussian(cls, flip_deg, duration_s, offset_ppm=0.0, phase_deg=0.0, steps=64,
                 truncation=0.01, **kwargs):
        """A Gaussian envelope truncated at `truncation` of its peak (midpoint sampled).

        The de Graaf (2019, Eq. 5.15) Gaussian: exp(-beta (2t/T)^2) with the truncation
        level setting beta = -ln(truncation). Extra keyword arguments (the slice
        gradient among them) pass through to the constructor.
        """
        tau = ((np.arange(steps, dtype=np.float32) + 0.5) / steps) * 2.0 - 1.0
        envelope = np.exp(np.log(truncation) * tau * tau).astype(np.float32)
        return cls(flip_deg, duration_s, envelope, offset_ppm=offset_ppm, phase_deg=phase_deg,
                   **kwargs)

    #***************#
    #   hermite     #
    #***************#
    @classmethod
    def hermite(cls, flip_deg, duration_s, offset_ppm=0.0, phase_deg=0.0, steps=64, **kwargs):
        """The Hermite envelope (Warren 1984), coefficients per nutation angle as in
        de Graaf's PulseWizard: (1 - 6.003 tau^2) exp(-9 tau^2) for 90 degrees and
        (1 - 8.604 tau^2) exp(-9 tau^2) for 180 degrees, tau in [-1, 1]. Only these
        two angles carry published coefficients.
        """
        coefficients = {90.0: 6.003, 180.0: 8.604}
        try:
            beta = coefficients[float(flip_deg)]
        except KeyError:
            raise ValueError("hermite coefficients are published for 90 and 180 degrees only") from None
        tau = ((np.arange(steps, dtype=np.float32) + 0.5) / steps) * 2.0 - 1.0
        envelope = ((1.0 - beta * tau * tau) * np.exp(-9.0 * tau * tau)).astype(np.float32)
        return cls(flip_deg, duration_s, envelope, offset_ppm=offset_ppm, phase_deg=phase_deg,
                   **kwargs)

    #**********************#
    #   hyperbolic secant  #
    #**********************#
    @classmethod
    def hyperbolic_secant(cls, duration_s, bandwidth_hz, peak_hz, beta=5.3, power=1,
                          steps=256, **kwargs):
        """An adiabatic full passage: offset independent adiabaticity pulses built on
        hyperbolic secants raised to a power (power 1 is the classic HS pulse).

        Constructed as in de Graaf's PulseWizard (citing Silver, Joseph, Hoult, J Magn
        Reson 59 (1984) 347, and Tannus, Garwood, J Magn Reson A 120 (1996) 133): half
        the pulse has amplitude sech(beta (1 - tau)^power), the frequency follows the
        normalized running integral of the squared amplitude scaled to half the
        bandwidth, and the second half mirrors both; beta 5.3 truncates at one percent.
        The inversion is adiabatic: insensitive to peak_hz above threshold.
        """
        half = steps // 2
        tau = np.arange(half, dtype=np.float64) / max(half - 1, 1)
        amplitude = 1.0 / np.cosh(beta * (1.0 - tau) ** power)
        amplitude = amplitude / amplitude.max()
        sweep = np.cumsum(amplitude * amplitude)
        sweep = (1.0 - sweep / sweep.max()) * (bandwidth_hz / 2.0)
        envelope = np.concatenate([amplitude, amplitude[::-1]]).astype(np.float32)
        frequency = np.concatenate([sweep, -sweep[::-1]]).astype(np.float32)
        return cls(None, duration_s, envelope, frequency_hz=frequency, peak_hz=peak_hz,
                   **kwargs)

    #***********#
    #   sinc    #
    #***********#
    @classmethod
    def sinc(cls, flip_deg, duration_s, offset_ppm=0.0, phase_deg=0.0, steps=128, lobes=3,
             **kwargs):
        """A sinc envelope truncated at the nth zero crossing on each side.

        The de Graaf (2019, Eq. 5.14) sinc: sin(2 pi n t / T) / (2 pi n t / T); lobes=3
        is the common five-lobe pulse. Side lobes are negative, which the RF term
        handles as a sign on the amplitude.
        """
        tau = ((np.arange(steps, dtype=np.float32) + 0.5) / steps) * 2.0 - 1.0
        x = np.pi * lobes * tau
        envelope = np.where(np.abs(x) < 1e-6, 1.0, np.sin(x) / np.where(x == 0.0, 1.0, x))
        return cls(flip_deg, duration_s, envelope.astype(np.float32),
                   offset_ppm=offset_ppm, phase_deg=phase_deg, **kwargs)

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        env = ops.cast(ops.convert_to_tensor(self.envelope), "float32")
        steps = int(env.shape[0])
        dt = ops.convert_to_tensor(self.duration_s, dtype="float32") / steps
        if self.peak_hz is not None:
            # adiabatic style: the envelope is scaled to the peak B1 directly
            f1 = env * ops.convert_to_tensor(ctx.b1_scale * self.peak_hz, dtype="float32")
        else:
            # calibration: 360 * integral of f1 dt equals flip_deg for an on-resonance
            # spin, scaled by the transmit field like any real RF amplitude
            flip = ops.convert_to_tensor(ctx.b1_scale * self.flip_deg, dtype="float32")
            f1 = env * (flip / 360.0) / (ops.sum(env) * dt)
        h_free = ops.einsum("ij,j,kj->ik", ctx.v, nops.cast(ctx.lam, "complex64"), ops.conj(ctx.v))
        if self.gradient_hz_per_mm:
            slice_hz = self.gradient_hz_per_mm * ops.convert_to_tensor(
                getattr(ctx, "gradient_" + self.gradient_axis), dtype="float32")
            if ops.ndim(slice_hz) > 0:  # a vector of positions batches the ensemble
                slice_hz = ops.reshape(slice_hz, (-1, 1, 1))
            h_free = h_free + nops.cast(slice_hz, "complex64") * ctx.fz
        offset_hz = ops.convert_to_tensor(self.offset_ppm, dtype="float32") * ctx.f0_mhz
        t_mid = (ops.arange(0, steps, dtype="float32") + 0.5) * dt
        phase = math.radians(self.phase_deg) + 2.0 * math.pi * offset_hz * t_mid
        if self.frequency_hz is not None:
            sweep = ops.cast(ops.convert_to_tensor(self.frequency_hz), "float32")
            phase = phase + 2.0 * math.pi * ops.cumsum(sweep) * dt
        # all step Hamiltonians at once: one batched eigendecomposition replaces the
        # step loop, then the step propagators collapse into the total by an ordered
        # pairwise product (log2(steps) batched matmuls), and one sandwich applies it;
        # with an in-pulse gradient over an isochromat vector the batch gains that axis
        shape = (-1, 1, 1, 1) if ops.ndim(h_free) == 3 else (-1, 1, 1)
        wx = nops.cast(f1 * ops.cos(phase), "complex64")
        wy = nops.cast(f1 * ops.sin(phase), "complex64")
        h = (ops.expand_dims(h_free, 0)
             + ops.reshape(wx, shape) * ctx.fx
             + ops.reshape(wy, shape) * ctx.fy)
        u = Backend.expm_herm(h, dt)
        while int(u.shape[0]) > 1:
            n = int(u.shape[0])
            pairs = ops.matmul(u[1:n:2], u[0:n - n % 2:2])
            u = pairs if n % 2 == 0 else ops.concatenate([pairs, u[n - 1:]], axis=0)
        return self._sandwich(u[0], rho)


#**************************************************************************************************#
#                                           Class Delay                                            #
#**************************************************************************************************#
#                                                                                                  #
# Free evolution under the system Hamiltonian for a duration in seconds.                           #
#                                                                                                  #
#**************************************************************************************************#
class Delay(Event):
    """Free evolution under the system Hamiltonian for a duration in seconds.

    When the system carries relaxation times, the phenomenological model applies after
    the coherent evolution: every coherence decays with exp(-t/T2) (zero-quantum terms
    included, the standard approximation), and the populations recover toward thermal
    equilibrium with exp(-t/T1). Without relaxation times the delay stays purely
    coherent, exactly as before.

    Args:
        t_s: Duration in seconds - may be a traced tensor, so sequence timings stay
            free, sweepable, and differentiable.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, t_s):
        self.t_s = t_s

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        u = Backend.propagator_from_eigh(ctx.lam, ctx.v, self.t_s)
        rho = self._sandwich(u, rho)
        if ctx.t2_s is None:
            return rho
        t = ops.convert_to_tensor(self.t_s)
        e2 = nops.cast(ops.exp(-t / ctx.t2_s), "complex64")
        one = nops.cast(ops.ones_like(t), "complex64")
        e1 = nops.cast(ops.exp(-t / ctx.t1_s), "complex64") if ctx.t1_s is not None else one
        off = (rho - rho * ctx.eye) * e2
        equilibrium = ctx.rho_eq * ctx.eye
        return off + equilibrium + (rho * ctx.eye - equilibrium) * e1


#**************************************************************************************************#
#                                          Class Gradient                                          #
#**************************************************************************************************#
#                                                                                                  #
# A gradient moment: ensemble dephasing over the voxel, exact per coherence order.                 #
#                                                                                                  #
#**************************************************************************************************#
class Gradient(Event):
    """A gradient moment applied along one axis: every density matrix element of
    coherence order q accumulates the position dependent phase 2 pi q k x. The engine
    averages the detected lines over an isochromat ensemble spanning the voxel, so the
    dephasing stays coherent through the sequence: a balanced pair around a refocusing
    pulse rephases the inverted pathway exactly and crushes everything else, and an
    imperfect pulse leaves the physical spurious residue a CoherenceFilter idealizes
    away (the unsuppressed FID artifact).

    Args:
        moment_per_mm: Gradient moment k in cycles per millimetre (gamma times area).
        extent_mm: Voxel extent along the gradient axis (sets the ensemble span).
        axis: The gradient axis, "x", "y", or "z"; kicks on different axes dephase
            independently and never rephase each other.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, moment_per_mm, extent_mm=20.0, axis="x"):
        if axis not in ("x", "y", "z"):
            raise ValueError(f"unknown gradient axis {axis!r}; known: x, y, z")
        self.moment_per_mm = moment_per_mm
        self.extent_mm = extent_mm
        self.axis = axis

    #*************#
    #   crusher   #
    #*************#
    @classmethod
    def crusher(cls, amplitude_mt_m, duration_s, extent_mm=20.0, gamma_mhz_t=42.5774785,
                axis="x"):
        """A crusher from hardware terms: k = gamma x amplitude x duration."""
        moment = gamma_mhz_t * 1e6 * amplitude_mt_m * 1e-3 * duration_s * 1e-3
        return cls(moment_per_mm=moment, extent_mm=extent_mm, axis=axis)

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        # the sign matches the in-pulse gradient evolution, so a slice pulse rephases
        # with the scanner-style lobe of minus half its gradient moment; a vector of
        # positions broadcasts the whole isochromat ensemble through one application
        x = ops.convert_to_tensor(getattr(ctx, "gradient_" + self.axis), dtype="float32")
        if ops.ndim(x) > 0:
            x = ops.reshape(x, (-1, 1, 1))
        angle = -2.0 * math.pi * self.moment_per_mm * x * ops.cast(ctx.q, "float32")
        factor = ops.exp(nops.complex_from(ops.zeros_like(angle), angle))
        return rho * factor


#**************************************************************************************************#
#                                      Class CoherenceFilter                                       #
#**************************************************************************************************#
#                                                                                                  #
# Keep only the selected coherence order(s) - the tensor-mask replacement for phase cycling and    #
# crushers.                                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class CoherenceFilter(Event):
    """Keep only the selected coherence order(s) - the tensor-mask replacement for phase
    cycling and crushers.

    Args:
        orders: A coherence order, or a tuple of orders, to retain.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, orders):
        self.orders = (orders,) if isinstance(orders, int) else tuple(orders)

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        keep = ops.equal(ctx.q, self.orders[0])
        for order in self.orders[1:]:
            keep = ops.logical_or(keep, ops.equal(ctx.q, order))
        return ops.where(keep, rho, ops.zeros_like(rho))


#**************************************************************************************************#
#                                          Class Sequence                                          #
#**************************************************************************************************#
#                                                                                                  #
# An ordered event list; subclasses are named designs, instances are concrete timings.             #
#                                                                                                  #
#**************************************************************************************************#
class Sequence:
    """An ordered event list; subclasses are named designs, instances are concrete timings."""

    #**********#
    #   init   #
    #**********#
    def __init__(self, *events):
        self.events = tuple(events)

    #*********#
    #   run   #
    #*********#
    def run(self, rho, ctx):
        """Apply every event in order and return the pre-readout density matrix."""
        for event in self.events:
            rho = event.apply(rho, ctx)
        return rho

    #*******************#
    #   composition     #
    #*******************#
    def __add__(self, other):
        """Concatenate two sequences, e.g. Chess(3) + Press(te1, te2)."""
        combined = Sequence(*self.events, *other.events)
        return combined
