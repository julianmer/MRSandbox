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
import copy
import math
import numbers

import numpy as np

from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.backend import Backend


#*****************#
#   hashable      #
#*****************#
def _hashable(value):
    """A structural attribute as a dict key: arrays by their bytes, the rest as is."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.tobytes())
    return value


#*****************#
#   trapezoid     #
#*****************#
def trapezoid(area_t_s_m, max_grad_t_m, max_slew_t_m_s):
    """The shortest trapezoid of a signed gradient area (T s / m) under a gradient and
    slew limit, as (amplitude T/m, ramp s, flat s): a triangle while the area fits
    under one, a flat top at the maximum gradient beyond."""
    magnitude = abs(area_t_s_m)
    if magnitude <= max_grad_t_m ** 2 / max_slew_t_m_s:
        amplitude = math.sqrt(magnitude * max_slew_t_m_s)
        return math.copysign(amplitude, area_t_s_m), amplitude / max_slew_t_m_s, 0.0
    return (math.copysign(max_grad_t_m, area_t_s_m), max_grad_t_m / max_slew_t_m_s,
            (magnitude - max_grad_t_m ** 2 / max_slew_t_m_s) / max_grad_t_m)


#**************************************************************************************************#
#                                           Class Event                                            #
#**************************************************************************************************#
#                                                                                                  #
# One step of a pulse sequence, applied to a density matrix in a system context.                   #
#                                                                                                  #
#**************************************************************************************************#
class Event(abc.ABC):
    """One step of a pulse sequence, applied to a density matrix in a system context.

    Every event splits into its structure (the type and the attributes that shape the
    graph: axes, coherence orders) and its timings (the attributes `free` names:
    durations, flips, phases, moments, waveforms), so a compiled graph is keyed by
    the structure and replays with the timings as inputs; a free attribute left None
    counts as structure.
    """

    free = ()

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

    #***********************#
    #   structure, timings  #
    #***********************#
    def structure(self):
        """The hashable key of everything but the timings."""
        fixed = tuple((name, _hashable(value)) for name, value in sorted(vars(self).items())
                      if name not in self.free or value is None)
        return (type(self).__name__, fixed)

    def timings(self):
        """The free values in `free` order, None ones left out."""
        return tuple(getattr(self, name) for name in self.free if getattr(self, name) is not None)

    def bound(self, values):
        """A copy of this event with its timings replaced, in `timings` order."""
        event = copy.copy(self)
        names = [name for name in self.free if getattr(self, name) is not None]
        for name, value in zip(names, values):
            setattr(event, name, value)
        return event

    #**************#
    #   sandwich   #
    #**************#
    @staticmethod
    def _sandwich(u, rho):
        """U rho U^H."""
        return ops.einsum("...ij,...jk,...lk->...il", u, rho, ops.conj(u))

    #**************************#
    #   transverse rotation    #
    #**************************#
    @staticmethod
    def _transverse_rotation(ctx, lam, phase_deg, turns):
        """exp(-i 2 pi turns A) for a transverse axis A = cos(phase) Ax + sin(phase) Ay,
        where Ax is any sum of per-spin Ix and `lam` the matching sum of mz rows.

        Ax = H Az H with H the Hadamard basis and Az diagonal with eigenvalues `lam`, and
        the phase is the z rotation D = exp(-i phase Fz), diagonal on the product basis,
        so U = (D H) exp(-i 2 pi turns Az) (D H)^H: two matmuls and no eigendecomposition
        in the graph. `turns` may be a traced tensor (the transmit scale times the flip).
        """
        phase = ops.convert_to_tensor(phase_deg, dtype="float32") * (math.pi / 180.0)
        angle = -phase * ops.sum(ctx.mz, axis=0)
        d = ops.exp(nops.complex_from(ops.zeros_like(angle), angle))
        v = ops.expand_dims(d, 1) * ctx.hadamard
        return Backend.propagator_from_eigh(lam, v, turns)


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

    free = ("flip_deg", "phase_deg")

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
        u = self._transverse_rotation(ctx, ops.sum(ctx.mz, axis=0), self.phase_deg,
                                      ctx.b1_scale * self.flip_deg / 360.0)
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
    primitive behind water suppression trains and spectral editing pulses. Offset
    and bandwidth may be traced tensors, so a network can emit them. A tuple of offsets
    is the ideal multi-band pulse: one band per centre, the idealization of
    the dual-lobe cosine-modulated waveforms of Hadamard-encoded editing.

    Args:
        flip_deg: Flip angle in degrees.
        offset_ppm: Pulse centre in ppm (water sits at 4.65), or a tuple of centres.
        bandwidth_hz: Full width of each band in Hz.
        phase_deg: Axis phase in degrees (0 = x, 90 = y).
    """

    free = ("flip_deg", "offset_ppm", "bandwidth_hz", "phase_deg")

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg, offset_ppm, bandwidth_hz, phase_deg=0.0):
        self.flip_deg = flip_deg
        self.offset_ppm = (tuple(offset_ppm) if isinstance(offset_ppm, (tuple, list))
                           else (offset_ppm,))
        self.bandwidth_hz = bandwidth_hz
        self.phase_deg = phase_deg

    #*************#
    #   apply     #
    #*************#
    def apply(self, rho, ctx):
        half_width = ops.convert_to_tensor(self.bandwidth_hz) / 2.0
        # every spin appears at its resonance minus the frame offset
        seen_hz = ctx.nu_hz - ops.convert_to_tensor(ctx.offset_hz, dtype="float32")
        inside = None
        for offset in self.offset_ppm:
            offset_hz = ops.convert_to_tensor(offset) * ctx.f0_mhz
            band = ops.abs(seen_hz - offset_hz) <= half_width
            inside = band if inside is None else ops.logical_or(inside, band)
        inside = ops.cast(inside, "float32")
        # the axis is the in-band sum of per-spin Ix, whose eigenvalues in the Hadamard
        # basis are the same sum of the per-spin magnetic quantum numbers
        u = self._transverse_rotation(ctx, ops.einsum("i,ij->j", inside, ctx.mz),
                                      self.phase_deg, ctx.b1_scale * self.flip_deg / 360.0)
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
        gradient_shape: Optional per-step modulation of the slice gradient, relative
            to gradient_hz_per_mm (gradient-modulated pulses); None plays it flat.
        extent_mm: Isochromat span when a gradient is set.
        frequency_hz: Optional per-step frequency modulation (adiabatic sweeps).
        peak_hz: Optional peak RF amplitude in Hz; when set the envelope is scaled to
            it directly instead of being flip calibrated (adiabatic pulses are defined
            by their peak B1, not a nutation angle; flip_deg is ignored then).
        induced: Optional (4, steps + 1) track of the phase an induced field builds up
            from the pulse's start, as an InducedKick counts it: the uniform cycles
            and the moments along x, y, z in cycles per millimetre, at the step
            midpoints and, last, at the pulse's end. The spins then run off resonance
            during the pulse by that field (scan.eddy fills it).
    """

    free = ("flip_deg", "duration_s", "envelope", "offset_ppm", "frequency_hz", "peak_hz",
            "phase_deg", "induced", "gradient_shape")

    # complex elements of step propagators held at once (2**24 is 128 MB in complex64)
    STEP_BLOCK_ELEMENTS = 2 ** 24

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg, duration_s, envelope, offset_ppm=0.0, phase_deg=0.0,
                 gradient_hz_per_mm=0.0, gradient_axis="x", extent_mm=20.0,
                 frequency_hz=None, peak_hz=None, induced=None, gradient_shape=None):
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
        self.induced = induced
        self.gradient_shape = gradient_shape

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
            raise ValueError("hermite coefficients are published for 90 and 180 degrees "
                             "only") from None
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

    #*****************#
    #   goia wurst    #
    #*****************#
    @classmethod
    def goia_wurst(cls, duration_s, bandwidth_hz, peak_hz, n=16, m=4, modulation=0.85, steps=256,
                   **kwargs):
        """A gradient-modulated adiabatic full passage, GOIA-W(n, m): the bandwidth of
        a much stronger pulse at a peak B1 a body coil delivers.

        Andronesi et al. (J Magn Reson 203 (2010) 283), Eq. 9 and 4: the amplitude is
        1 - |sin(pi/2 (2t/T - 1))|^n, the slice gradient (1 - f) + f |sin(...)|^m
        relative to its maximum (`modulation` is f: the gradient dips to 1 - f at the
        pulse's centre, where the B1 peaks), and the frequency follows the gradient
        times the running integral of B1^2 / G, scaled so the sweep spans the
        bandwidth at the full gradient; the adiabatic factor is then the same at
        every position of the slab. The defaults are the orders and the modulation of
        the across-vendor 3 T semi-LASER (Deelchand et al. 2021, NMR Biomed 34:e4218:
        4.5 ms, 10 kHz, 15 uT). Give gradient_hz_per_mm (the bandwidth over the slab
        thickness) and the pulse plays under its modulated gradient; without one
        only the sweep remains.
        """
        tau = np.abs(np.sin(0.5 * np.pi * (2.0 * (np.arange(steps) + 0.5) / steps - 1.0)))
        amplitude = 1.0 - tau ** n
        gradient = (1.0 - modulation) + modulation * tau ** m
        sweep = np.cumsum(amplitude * amplitude / gradient)
        sweep = sweep - 0.5 * amplitude * amplitude / gradient  # to the step midpoints
        frequency = bandwidth_hz * gradient * (sweep / sweep[-1] - 0.5)
        shape = gradient.astype(np.float32) if kwargs.get("gradient_hz_per_mm") else None
        return cls(None, duration_s, amplitude.astype(np.float32),
                   frequency_hz=frequency.astype(np.float32), peak_hz=peak_hz,
                   gradient_shape=shape, **kwargs)

    #**********#
    #   slr    #
    #**********#
    @classmethod
    def slr(cls, flip_deg, duration_s, time_bandwidth, kind, offset_ppm=0.0, phase_deg=0.0,
            steps=256, ripples=(0.01, 0.01), minimum_phase=False, **kwargs):
        """A Shinnar-Le Roux pulse, linear phase unless told, the design behind the
        slice-selective pulses scanners play (Pauly, Le Roux, Nishimura, Macovski, IEEE Trans Med
        Imaging 10 (1991) 53): the profile is an equiripple low-pass filter (Parks and
        McClellan), here its beta polynomial, of `time_bandwidth` (the bandwidth is
        that over the duration, between the half-amplitude points); the minimum-phase
        alpha polynomial completes it, and the inverse transform peels the hard
        pulses off one by one. `kind` sets how the passband and stopband `ripples`
        of the magnetization map onto the filter's: "excitation" (a 90, rephased by
        half its gradient moment) or "refocusing" (a spin-echo 180 between crushers).
        `minimum_phase` makes the beta polynomial the minimum-phase half of a filter
        twice as long, played backwards, so the main lobe comes late (at 0.8 of the
        pulse): the asymmetric excitation of short echo times, rephased by about 0.23
        of its moment instead of 0.52, at the price of a phase that is no longer linear
        across the slab.
        """
        from scipy import signal

        d1, d2 = ripples
        if kind == "excitation":
            scale, d1, d2 = math.sqrt(0.5), math.sqrt(d1 / 2.0), d2 / math.sqrt(2.0)
        elif kind == "refocusing":
            scale, d1, d2 = 1.0, d1 / 4.0, math.sqrt(d2)
        else:
            raise ValueError(f"unknown kind {kind!r}; known: excitation, refocusing")
        # the transition width of an equiripple filter for these ripples (their Eq. 23)
        l1, l2 = math.log10(d1), math.log10(d2)
        d_inf = ((5.309e-3 * l1 * l1 + 7.114e-2 * l1 - 4.761e-1) * l2
                 - (2.66e-3 * l1 * l1 + 5.941e-1 * l1 + 4.278e-1))
        width = d_inf / time_bandwidth
        edges = np.array([0.0, (1.0 - width) * time_bandwidth / 2.0,
                          (1.0 + width) * time_bandwidth / 2.0, steps / 2.0]) / steps
        if minimum_phase:
            # the prototype carries the squared response: its stopband ripple squares
            prototype = signal.remez(2 * steps - 1, edges, [1.0, 0.0],
                                     weight=[1.0, 2.0 * d1 / (0.5 * d2 * d2)], fs=1.0)
            beta = scale * signal.minimum_phase(prototype, method="homomorphic", half=True)[::-1]
        else:
            beta = scale * signal.remez(steps, edges, [1.0, 0.0], weight=[1.0, d1 / d2], fs=1.0)
        # the minimum-phase alpha with |alpha|^2 + |beta|^2 = 1, by the cepstrum
        pad = 16 * steps
        response = np.fft.fft(beta, pad)
        response = response / max(1.0, np.abs(response).max() + 1e-7)
        log_mag = np.fft.fft(0.5 * np.log(1.0 - np.abs(response) ** 2))
        log_mag[1:pad // 2] *= 2.0
        log_mag[pad // 2 + 1:] = 0.0
        alpha = (np.fft.fft(np.exp(np.fft.ifft(log_mag))) / pad)[:steps][::-1]
        # the inverse transform: each step's rotation read off the leading coefficients
        a, b = alpha.astype(np.complex128), beta.astype(np.complex128)
        envelope = np.zeros(steps)
        for k in range(steps - 1, -1, -1):
            ratio = b[k] / a[k]
            c = 1.0 / math.sqrt(1.0 + abs(ratio) ** 2)
            sj = np.conj(c * ratio)
            envelope[k] = 2.0 * math.atan2(abs(sj), c) * np.sign(sj.real if sj.real else 1.0)
            a, b = (c * a + sj * b)[1:k + 1], (-np.conj(sj) * a + c * b)[:k]
        # the hard pulses are the design: played at their own amplitude (rotations of
        # `envelope` radians per step), not rescaled to a nominal flip, which the
        # passband ripple at the centre would turn into a wrong one
        peak = np.abs(envelope).max()
        return cls(flip_deg, duration_s, (envelope / peak).astype(np.float32),
                   offset_ppm=offset_ppm, phase_deg=phase_deg,
                   peak_hz=peak / (2.0 * math.pi * duration_s / steps), **kwargs)

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
    def apply(self, rho, ctx, propagator=None):
        """Play the pulse; `propagator` is a `propagator(ctx)` result to reuse."""
        propagator, total = propagator if propagator is not None else self.propagator(ctx)
        rho = self._sandwich(propagator, rho)
        frame = -2.0 * math.pi * total
        if ops.ndim(total) > 0:
            frame = ops.reshape(frame, (-1, 1, 1))
        frame = frame * ops.cast(ctx.q, "float32")
        return rho * ops.exp(nops.complex_from(ops.zeros_like(frame), frame))

    #**********#
    #   peak   #
    #**********#
    @property
    def played_peak_hz(self):
        """The peak RF the pulse plays, in Hz: its own peak where it has one (an adiabatic
        pulse), else what its flip asks of its envelope."""
        if self.peak_hz is not None:
            return float(self.peak_hz)
        envelope = np.asarray(self.envelope, dtype=np.float64)
        dt = float(self.duration_s) / envelope.size
        return float(envelope.max() * (float(self.flip_deg) / 360.0) / (envelope.sum() * dt))

    #****************#
    #   propagator   #
    #****************#
    def propagator(self, ctx):
        """The pulse's propagator in a context, per isochromat where an ensemble
        plays, and the frame term's cycles over the pulse: everything the pulse costs,
        and nothing of the state, so a sequence that plays one pulse object several
        times computes it once per run (`Sequence.run`)."""
        env = ops.cast(ops.convert_to_tensor(self.envelope), "float32")
        steps = int(env.shape[0])
        dt = ops.convert_to_tensor(self.duration_s, dtype="float32") / steps
        # the transmit scale: one value, or one per isochromat (a transmit field
        # varying across the voxel), which batches the step Hamiltonians like the
        # positions batch the frame
        b1 = ops.convert_to_tensor(ctx.b1_scale, dtype="float32")
        if ops.ndim(b1) > 0:
            env = ops.expand_dims(env, 1)
        if self.peak_hz is not None:
            # adiabatic style: the envelope is scaled to the peak B1 directly
            f1 = env * (b1 * ops.convert_to_tensor(self.peak_hz, dtype="float32"))
        else:
            # calibration: 360 * integral of f1 dt equals flip_deg for an on-resonance
            # spin, scaled by the transmit field like any real RF amplitude
            flip = b1 * ops.convert_to_tensor(self.flip_deg, dtype="float32")
            f1 = env * (flip / 360.0) / (ops.sum(env, axis=0) * dt)
        h_free = ops.einsum("ij,j,kj->ik", ctx.v, nops.cast(ctx.lam, "complex64"), ops.conj(ctx.v))
        # the frame: the offset and the slice gradient are one z term delta Fz, with
        # delta a scalar or one value per isochromat
        delta = -ops.convert_to_tensor(ctx.offset_hz, dtype="float32")
        if self.gradient_hz_per_mm and self.gradient_shape is None:
            delta = delta + self.gradient_hz_per_mm * ops.convert_to_tensor(
                getattr(ctx, "gradient_" + self.gradient_axis), dtype="float32")
        offset_hz = ops.convert_to_tensor(self.offset_ppm, dtype="float32") * ctx.f0_mhz
        t_mid = (ops.arange(0, steps, dtype="float32") + 0.5) * dt
        phase = (ops.convert_to_tensor(self.phase_deg, dtype="float32") * (math.pi / 180.0)
                 + 2.0 * math.pi * offset_hz * t_mid)
        if self.frequency_hz is not None:
            sweep = ops.cast(ops.convert_to_tensor(self.frequency_hz), "float32")
            # the sweep phase at the same step midpoints as the carrier (the running
            # integral to the end of each step would lead the envelope by half a step
            # and skew the inversion profile)
            phase = phase + 2.0 * math.pi * (ops.cumsum(sweep) - sweep / 2.0) * dt
        # every step Hamiltonian is a z rotation of H_free + f1 Fx: the RF phase and,
        # in the interaction frame of delta Fz, the frame term itself become one
        # angle theta = phase - 2 pi delta t per step (the frame term sampled at the
        # step midpoints, the approximation the carrier ramp already makes), so one
        # batched eigendecomposition over the steps serves every isochromat, carrier,
        # and sweep, and the rotation D = exp(-i theta Fz) is diagonal on the product
        # basis: U_n = D W_n D^H, a phase per row times a phase per column. The step
        # propagators collapse into the total by an ordered pairwise product
        # (log2(steps) batched matmuls), one sandwich applies it, and the frame
        # rotation over the whole pulse follows as a phase per coherence order,
        # exactly as a Delay applies the offset
        h = ops.expand_dims(ops.expand_dims(nops.cast(f1, "complex64"), -1), -1) * ctx.fx
        h = h + h_free  # (steps, d, d), or (steps, isochromats, d, d) with a vector b1
        w = Backend.expm_herm(h, dt)
        if ops.ndim(b1) == 0:
            w = ops.expand_dims(w, 1)
        m = ops.sum(ctx.mz, axis=0)
        # the frame term in cycles, at the step midpoints and over the whole pulse
        if ops.ndim(delta) > 0:  # a vector of positions batches the ensemble
            turns = ops.expand_dims(delta, 0) * ops.expand_dims(t_mid, 1)
        else:
            turns = ops.expand_dims(delta * t_mid, 1)
        total = ops.convert_to_tensor(self.duration_s, dtype="float32") * delta
        if self.gradient_hz_per_mm and self.gradient_shape is not None:
            # a modulated slice gradient: its running integral at the step midpoints
            shape = ops.cast(ops.convert_to_tensor(self.gradient_shape), "float32")
            x = self.gradient_hz_per_mm * ops.convert_to_tensor(
                getattr(ctx, "gradient_" + self.gradient_axis), dtype="float32")
            run = (ops.cumsum(shape) - shape / 2.0) * dt
            turns = turns + ops.expand_dims(run, 1) * ops.expand_dims(x, 0)
            total = total + ops.sum(shape) * dt * x
        if self.induced is not None:
            # an induced field is one more frame term, its cycles given per step
            induced = ops.cast(ops.convert_to_tensor(self.induced), "float32")
            turns = turns + ops.expand_dims(induced[0, :-1], 1)
            total = total + induced[0, -1]
            for k, axis in enumerate("xyz"):
                x = ops.convert_to_tensor(getattr(ctx, "gradient_" + axis), dtype="float32")
                if ops.ndim(x) > 0:
                    turns = turns + ops.expand_dims(induced[k + 1, :-1], 1) * ops.expand_dims(x, 0)
                    total = total + induced[k + 1, -1] * x
        theta = ops.expand_dims(phase, 1) - 2.0 * math.pi * turns
        angle = -ops.expand_dims(theta, -1) * m  # (steps, isochromats or 1, d)
        d = ops.exp(nops.complex_from(ops.zeros_like(angle), angle))
        # the step propagators of an ensemble of a large system do not fit at once
        # (steps x isochromats x d x d): they are formed and collapsed in blocks of
        # steps that hold STEP_BLOCK_ELEMENTS, and the blocks multiply in order
        size = int(d.shape[1]) * int(d.shape[2]) ** 2
        block = max(1, min(steps, self.STEP_BLOCK_ELEMENTS // size))
        propagator = None
        for start in range(0, steps, block):
            rows = d[start:start + block]
            u = (ops.expand_dims(rows, -1) * w[start:start + block]
                 * ops.expand_dims(ops.conj(rows), -2))
            if ops.ndim(total) == 0 and ops.ndim(b1) == 0:
                u = u[:, 0]
            while int(u.shape[0]) > 1:
                n = int(u.shape[0])
                pairs = ops.matmul(u[1:n:2], u[0:n - n % 2:2])
                u = pairs if n % 2 == 0 else ops.concatenate([pairs, u[n - 1:]], axis=0)
            propagator = u[0] if propagator is None else ops.matmul(u[0], propagator)
        return propagator, total


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
    equilibrium with exp(-t/T1). A missing partner is treated as infinite, so a
    T1-only system still recovers and a T2-only system still dephases. Without any
    relaxation times the delay stays purely coherent.

    The frame offset of the context evolves alongside: a uniform offset commutes with
    the Hamiltonian, so its evolution exp(+i 2 pi offset Fz t) is a phase per
    coherence order, applied like a gradient kick; refocused by spin echoes, seen by
    the selective and shaped pulses in between.

    Args:
        t_s: Duration in seconds - may be a traced tensor, so sequence timings stay
            free, sweepable, and differentiable.
    """

    free = ("t_s",)

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
        angle = (2.0 * math.pi * ops.convert_to_tensor(ctx.offset_hz, dtype="float32")
                 * ops.convert_to_tensor(self.t_s, dtype="float32") * ops.cast(ctx.q, "float32"))
        rho = rho * ops.exp(nops.complex_from(ops.zeros_like(angle), angle))
        if ctx.t1_s is None and ctx.t2_s is None:
            return rho
        t = ops.convert_to_tensor(self.t_s)
        one = nops.cast(ops.ones_like(t), "complex64")
        e2 = nops.cast(ops.exp(-t / ctx.t2_s), "complex64") if ctx.t2_s is not None else one
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

    free = ("moment_per_mm",)

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
#                                        Class InducedKick                                         #
#**************************************************************************************************#
#                                                                                                  #
# The phase an induced field accumulates over an interval: uniform cycles and moments per axis.    #
#                                                                                                  #
#**************************************************************************************************#
class InducedKick(Event):
    """The phase an induced field accumulates over an interval: uniform cycles and
    moments per axis.

    What the scanner's gradient response adds to a design before the readout
    (scan.eddy.induced_sequence): the field's integral over the interval it follows,
    as a uniform part in cycles applied per coherence order (the B0 term plus the
    first-order terms at the voxel centre) and a moment per axis in cycles per
    millimetre applied over the isochromat ensemble (the first-order terms across the
    box, unbalancing the crushers), both with a gradient kick's sign: a positive field
    advances the spins as faster precession does, the sign the readout's kernel
    carries on (a delay's frame offset is the scanner's frequency error and has the
    other sign). Every value is a timing, so one compiled design replays at every
    voxel.

    Args:
        cycles: The uniform phase in cycles.
        moment_x, moment_y, moment_z: The moments in cycles per millimetre.
    """

    free = ("cycles", "moment_x", "moment_y", "moment_z")

    #**********#
    #   init   #
    #**********#
    def __init__(self, cycles, moment_x=0.0, moment_y=0.0, moment_z=0.0):
        self.cycles = cycles
        self.moment_x = moment_x
        self.moment_y = moment_y
        self.moment_z = moment_z

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx):
        angle = -2.0 * math.pi * ops.convert_to_tensor(self.cycles, dtype="float32")
        for axis in "xyz":
            x = ops.convert_to_tensor(getattr(ctx, "gradient_" + axis), dtype="float32")
            if ops.ndim(x) > 0:
                x = ops.reshape(x, (-1, 1, 1))
            moment = ops.convert_to_tensor(getattr(self, "moment_" + axis), dtype="float32")
            angle = angle - 2.0 * math.pi * moment * x
        angle = angle * ops.cast(ctx.q, "float32")
        return rho * ops.exp(nops.complex_from(ops.zeros_like(angle), angle))


#**************************************************************************************************#
#                                         Class Projection                                         #
#**************************************************************************************************#
#                                                                                                  #
# Average the density matrix over the isochromat ensemble: one axis of a localization is done.     #
#                                                                                                  #
#**************************************************************************************************#
class Projection(Event):
    """Average the density matrix over the isochromat ensemble: one axis of a
    localization is done.

    The one-dimensional projection of Zhang, An, and Shen (Med Phys 44 (2017) 4169):
    a slice-selective pulse depends on the position along its own axis alone, so the
    sum over a three-dimensional voxel factorizes, sum_y P(y) [sum_x P(x) rho P(x)^H]
    P(y)^H, and the density matrix may be averaged over an axis as soon as the pulses
    on that axis (a refocusing pair counts as one) have played and their unwanted
    coherences are gone. The cost of a voxel of n^3 positions falls to 3 n with the
    same result. It holds while nothing later addresses that axis again and nothing
    before addressed another one, so coherence filters select the pathway here, not
    crushers on the other axes. The one thing before a localization that addresses
    its axes, outer-volume saturation slabs, is separable too, and Sequence folds
    each slab into the projection of its own axis (`saturation`): what the slab
    leaves of each isochromat's longitudinal magnetization weighs it in the average.

    Args:
        gain: The factor on the average: the ensemble's extent over the nominal slab
            thickness makes the result the signal per nominal voxel, one for an ideal
            profile.
    """

    free = ("gain",)

    #**********#
    #   init   #
    #**********#
    def __init__(self, gain=1.0):
        self.gain = gain

    #***********#
    #   apply   #
    #***********#
    def apply(self, rho, ctx, saturation=(), propagators=None):
        if ops.ndim(rho) == 3:
            if saturation:
                left = ctx.rho_eq
                for event in saturation:
                    if isinstance(event, ShapedPulse):
                        if id(event) not in propagators:
                            propagators[id(event)] = event.propagator(ctx)
                        left = event.apply(left, ctx, propagators[id(event)])
                    else:
                        left = event.apply(left, ctx)
                fz = nops.cast(ctx.fz, "complex64")
                weights = (ops.real(ops.einsum("iab,ba->i", left, fz))
                           / ops.real(ops.einsum("ab,ba->", nops.cast(ctx.rho_eq, "complex64"), fz)))
                rho = ops.mean(ops.reshape(nops.cast(weights, "complex64"), (-1, 1, 1)) * rho,
                               axis=0)
            else:
                rho = ops.mean(rho, axis=0)
        return rho * nops.cast(ops.convert_to_tensor(self.gain, dtype="float32"), "complex64")


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
    """An ordered event list; subclasses are named designs, instances are concrete timings.

    Args:
        *events: The events in order.
        tr_s: Repetition time. None runs the events once from thermal equilibrium, a
            fully relaxed acquisition. With a TR the run reaches the spoiled steady
            state first: `cycles` repetitions of the events, each followed by the
            readout's crushing (only order zero survives the spoilers) and the
            recovery over the rest of the repetition (a Delay of TR minus the
            sequence's own duration, relaxing under T1 and T2), then the events run
            once more for the readout. So partial recovery, the inversion of the
            recovered magnetization by refocusing pulses, and the per-metabolite T1
            weighting emerge from the same density matrix; the readout and its
            spoiling fall inside the recovery time. One cycle is exact for a complete
            excitation (nothing longitudinal survives an ideal 90), and every further
            cycle multiplies the residual of an imperfect one by another cos(flip)
            exp(-TR / T1). TR is a timing, so a compiled design replays at any TR.
            One idealization to know: a CoherenceFilter keeps only its orders, so the
            longitudinal magnetization that recovers between excitation and readout
            (and that refocusing pulses invert) is dropped by a filter on +1 or -1,
            an O(TE / T1) term of the steady state; physical Gradient crushers keep
            it and reproduce the inversion law exactly.
        cycles: Steady-state repetitions before the readout run.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, *events, tr_s=None, cycles=1):
        self.events = tuple(events)
        self.tr_s = tr_s
        self.cycles = cycles
        if isinstance(tr_s, numbers.Real) and isinstance(self.duration_s, numbers.Real) \
                and tr_s < self.duration_s:
            raise ValueError(f"tr_s {tr_s} s is shorter than the sequence's {self.duration_s} s")

    #*********#
    #   run   #
    #*********#
    def run(self, rho, ctx):
        """Apply every event in order and return the pre-readout density matrix, from
        the spoiled steady state when the sequence carries a TR."""
        propagators = {}
        if self.tr_s is not None:
            crush, recover = CoherenceFilter(0), Delay(self.tr_s - self.duration_s)
            for _ in range(self.cycles):
                rho = recover.apply(crush.apply(self._pass(rho, ctx, propagators), ctx), ctx)
        return self._pass(rho, ctx, propagators)

    def _pass(self, rho, ctx, propagators):
        # a shaped pulse object played more than once (a refocusing pair, the passes
        # of a steady state) is computed once: its propagator does not see the state
        held, folded = self._slabs()
        for k, event in enumerate(self.events):
            if k in held:
                continue
            if isinstance(event, ShapedPulse):
                if id(event) not in propagators:
                    propagators[id(event)] = event.propagator(ctx)
                rho = event.apply(rho, ctx, propagators[id(event)])
            elif isinstance(event, Projection):
                rho = event.apply(rho, ctx, folded.get(k, ()), propagators)
            else:
                rho = event.apply(rho, ctx)
        return rho

    def _slabs(self):
        """The saturation slabs a projected design folds into its projections: before the
        localization, a slice-selective pulse followed by a filter on order zero alone is
        an outer-volume suppression slab. Played in place, a slab along y would be averaged
        into the x projection as a loss over the whole voxel, so each slab is held back and
        weighed into the projection of its own axis instead (Projection.apply), which is
        where the factorization puts it. A slab along an axis no projection localizes plays
        in place. Returns (the indices held back, {projection index: its slab events})."""
        events = self.events
        if not any(isinstance(event, Projection) for event in events):
            return set(), {}
        first = next(k for k, event in enumerate(events) if isinstance(event, Projection))
        slabs = [k for k in range(first - 1)
                 if isinstance(events[k], ShapedPulse) and events[k].gradient_hz_per_mm
                 and isinstance(events[k + 1], CoherenceFilter) and events[k + 1].orders == (0,)]
        held, folded, since = set(), {}, set()
        for k, event in enumerate(events):
            if isinstance(event, ShapedPulse) and event.gradient_hz_per_mm and k not in slabs:
                since.add(event.gradient_axis)
            elif isinstance(event, Projection):
                mine = [j for j in slabs if events[j].gradient_axis in since and j not in held]
                held.update(j + i for j in mine for i in (0, 1))
                folded[k] = tuple(events[j + i] for j in mine for i in (0, 1))
                since = set()
        return held, folded

    #**************#
    #   duration   #
    #**************#
    @property
    def duration_s(self):
        """The time the events take: every delay and every shaped pulse (ideal pulses,
        kicks, and filters are instantaneous)."""
        total = 0.0
        for event in self.events:
            if isinstance(event, Delay):
                total = total + event.t_s
            elif isinstance(event, ShapedPulse):
                total = total + event.duration_s
        return total

    #*****************#
    #   feasibility   #
    #*****************#
    def feasible(self, scanner, gamma_mhz_t=42.5774785):
        """Whether the design plays on `scanner`; `violations` says why not."""
        return not self.violations(scanner, gamma_mhz_t)

    def violations(self, scanner, gamma_mhz_t=42.5774785):
        """Why the design cannot play on `scanner` (a hardware.Scanner), as a tuple of
        reasons, empty when it can.

        Two kinds of limit. Amplitudes: a shaped pulse's peak RF must stay under the
        scanner's peak B1 and its slice gradient under the maximum gradient. Time:
        the events that are instantaneous in the simulation take time on hardware,
        and the delays around them must absorb it, exactly as the Pulseq export
        borrows it (io.pulseq.write_seq), so the shortfall reported is what the echo
        time must grow by. The hardware durations, all stated: a hard pulse is the
        shortest block pulse the peak B1 allows, flip / (360 gamma B1); an ideal
        selective band is the 1 percent truncated Gaussian of that FWHM bandwidth, so
        its length is the de Graaf Table 5.1 bandwidth factor (2.70 for excitation,
        1.53 for inversion) over the bandwidth; a gradient kick is the shortest
        trapezoid of its moment under the maximum gradient and slew rate, and a
        slice gradient adds its two ramps; a coherence filter is an idealization and
        takes no time (the physical design carries crushers). A trailing shortfall
        means the readout would start inside the last event.
        """
        peak_hz = gamma_mhz_t * scanner.peak_b1_ut  # MHz/T times uT is Hz
        reasons, owed = [], 0.0
        for event in self.events:
            if isinstance(event, Delay):
                owed = max(owed - float(event.t_s), 0.0)
            elif isinstance(event, ShapedPulse):
                peak = event.played_peak_hz
                if peak > peak_hz:
                    reasons.append(f"shaped pulse peak {peak / gamma_mhz_t:.1f} uT exceeds the "
                                   f"scanner's {scanner.peak_b1_ut:.1f} uT")
                if event.gradient_hz_per_mm:
                    amplitude = abs(float(event.gradient_hz_per_mm)) / gamma_mhz_t  # mT/m
                    if amplitude > scanner.max_grad_mt_m:
                        reasons.append(f"slice gradient {amplitude:.1f} mT/m exceeds the "
                                       f"scanner's {scanner.max_grad_mt_m:.1f} mT/m")
                    owed += 2.0 * amplitude * 1e-3 / scanner.max_slew_t_m_s
            elif isinstance(event, SelectivePulse):
                factor = 1.53 if float(event.flip_deg) >= 135.0 else 2.70
                owed += factor / float(event.bandwidth_hz)
            elif isinstance(event, Pulse):
                owed += (float(event.flip_deg) / 360.0) / peak_hz
            elif isinstance(event, Gradient):
                _, ramp, flat = trapezoid(float(event.moment_per_mm) * 1e3 / (gamma_mhz_t * 1e6),
                                          scanner.max_grad_mt_m * 1e-3, scanner.max_slew_t_m_s)
                owed += 2.0 * ramp + flat
        if owed > 1e-9:
            reasons.append(f"the delays are {owed * 1e3:.3f} ms too short to absorb the pulse "
                           f"and gradient durations")
        return tuple(reasons)

    #***********************#
    #   structure, timings  #
    #***********************#
    def structure(self):
        """The hashable key of the design: every event's structure in order, the
        cycle count, and whether a TR is set. Two sequences of one structure share
        one compiled graph and differ only in their timings."""
        return (tuple(event.structure() for event in self.events), self.cycles, self.tr_s is None)

    def timings(self):
        """The free values of every event, one tuple per event, then the TR as a last
        tuple (empty without one)."""
        return (tuple(event.timings() for event in self.events)
                + ((self.tr_s,) if self.tr_s is not None else (),))

    def bound(self, timings):
        """The same design at other timings (as `timings` lays them out), as a plain
        Sequence; this is what a compiled graph rebuilds from its inputs."""
        # one event object at several places stays one object (its first timings)
        bound = {}
        events = [bound.setdefault(id(event), event.bound(values))
                  for event, values in zip(self.events, timings[:-1])]
        tr_s = timings[-1][0] if self.tr_s is not None else None
        return Sequence(*events, tr_s=tr_s, cycles=self.cycles)

    #**************#
    #   confined   #
    #**************#
    def confined(self, extent_mm):
        """The same design with its isochromat ensemble confined to a box of `extent_mm`
        (x, y, z; None leaves an axis as it is) and the `Projection` of every confined
        axis a plain average (gain one): what the design leaves of a small part of
        the object, a cell of a lattice inside the box it selects, wherever the
        engine's shift places that part. A Projection belongs to the axis of the
        gradient played last before it. The timings stay."""
        extent = dict(zip("xyz", (None if e is None else float(e) for e in extent_mm)))
        events, last = [], None
        for event in self.events:
            if isinstance(event, Projection):
                event = Projection(1.0) if extent.get(last) is not None else event
            elif isinstance(event, Gradient):
                last = event.axis
                if extent[last] is not None:
                    event = copy.copy(event)
                    event.extent_mm = extent[last]
            elif isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
                last = event.gradient_axis
                if extent[last] is not None:
                    event = copy.copy(event)
                    event.extent_mm = extent[last]
            events.append(event)
        design = copy.copy(self)
        design.events = tuple(events)
        return design

    #*******************#
    #   composition     #
    #*******************#
    TIMINGS = ("te_s", "te1_s", "te2_s", "taus_s", "tm_s")

    def __add__(self, other):
        """Concatenate two sequences, e.g. Chess(3) + Press(te1, te2); the right
        operand's TR and echo timings (the localization's) carry, else the left one's,
        so a train before a design leaves the design's echo time readable."""
        tr_s = other.tr_s if other.tr_s is not None else self.tr_s
        cycles = other.cycles if other.tr_s is not None else self.cycles
        joined = Sequence(*self.events, *other.events, tr_s=tr_s, cycles=cycles)
        timed = other if any(hasattr(other, name) for name in self.TIMINGS) else self
        for name in self.TIMINGS:
            if hasattr(timed, name):
                setattr(joined, name, getattr(timed, name))
        return joined
