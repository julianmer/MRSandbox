####################################################################################################
#                                             core.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The GAMMA-class engine core: run any sequence on any spin systems and read the result   #
#          out as transition lines - which then render to a spectrum or FID at ANY grid for free,  #
#          so readout resolution is a rendering parameter, not a simulation cost. All tensor math  #
#          is keras.ops; complex64 throughout.                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import math

import numpy as np

from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.backend import Backend


#**************************************************************************************************#
#                                          Class Context                                           #
#**************************************************************************************************#
#                                                                                                  #
# The tensor-side view of one spin system at one field, handed to sequence events.                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Context:
    """The tensor-side view of one spin system at one field, handed to sequence events.

    Attributes:
        lam, v: Eigenvalues (Hz) and eigenvectors of the system Hamiltonian.
        fx, fy: Total-spin operators (pulse rotation axes).
        ix, iy: Per-spin operators (selective rotation axes).
        nu_hz: Per-spin resonance frequencies in Hz.
        f0_mhz: Spectrometer frequency in MHz (Hz per ppm).
        q: Integer coherence-order matrix (filter masks).
        rho_eq, eye: Thermal equilibrium state and the identity mask, for relaxation.
        fplus: The detection operator F+.
        t1_s, t2_s: The system's relaxation times, or None for coherent-only delays.
        b1_scale: Transmit field scale; every pulse flip is multiplied by it.
        gradient_x: Isochromat position along the gradient axis in millimetres.
        fz: The longitudinal total-spin operator (gradient offsets act through it).
    """

    lam: object
    v: object
    fx: object
    fy: object
    ix: object
    iy: object
    nu_hz: object
    f0_mhz: float
    q: object
    rho_eq: object
    eye: object
    fplus: object
    t1_s: object
    t2_s: object
    b1_scale: object
    fz: object = None
    gradient_x: object = 0.0


#**************************************************************************************************#
#                                         Class Engine                                             #
#**************************************************************************************************#
#                                                                                                  #
# Runs sequences on spin systems at a freely choosable field and renders the outcome.              #
#                                                                                                  #
#**************************************************************************************************#
class Engine:
    """Runs sequences on spin systems at a freely choosable field and renders the outcome.

    Args:
        b0: Field strength in tesla, or anything with a `b0_t` attribute (e.g. a
            hardware.Scanner preset) - the field stays a free parameter either way.
        receiver_phase_deg: Receiver phase applied to all detected lines. The default of
            90 makes a simple 90x pulse-acquire render as a positive real absorption
            spectrum (the convention every fitting tool expects); 0 gives raw physics.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, b0, receiver_phase_deg=90.0):
        self.b0_t = getattr(b0, "b0_t", b0)
        self.receiver_phase_deg = receiver_phase_deg

    #*************#
    #   context   #
    #*************#
    def context(self, system, b1_scale=1.0):
        """Build the tensor Context of one system at this field."""
        op = system.operators()
        lam, v = Backend.eigh_herm(ops.convert_to_tensor(system.hamiltonian(self.b0_t)))
        f0_mhz = system.gamma_mhz_t * self.b0_t
        nu = np.asarray(system.shifts_ppm, dtype=np.float32) * f0_mhz
        return Context(
            lam=lam,
            v=v,
            fx=ops.convert_to_tensor(op.fx),
            fy=ops.convert_to_tensor(op.fy),
            ix=ops.convert_to_tensor(op.ix),
            iy=ops.convert_to_tensor(op.iy),
            nu_hz=ops.convert_to_tensor(nu),
            f0_mhz=f0_mhz,
            q=ops.convert_to_tensor(op.coherence_order),
            rho_eq=ops.convert_to_tensor(system.thermal_rho()),
            eye=ops.convert_to_tensor(np.eye(system.dim, dtype=np.complex64)),
            fplus=ops.convert_to_tensor(op.fplus),
            fz=ops.convert_to_tensor(op.fz),
            t1_s=system.t1_s,
            t2_s=system.t2_s,
            b1_scale=b1_scale,
        )

    #***********#
    #   lines   #
    #***********#
    def lines(self, sequence, *systems, b1_scale=1.0, isochromats=32):
        """Transition lines of `sequence` run on each system, concatenated.

        The pre-readout density matrix is projected onto the Hamiltonian eigenbasis, where
        every element is one exponential: s(t) = sum_k amp_k * exp(+i 2 pi freq_k t).
        When the sequence contains Gradient events the lines are averaged over an
        isochromat ensemble of `isochromats` positions spanning the gradient extent, so
        crusher pairs rephase coherently and unbalanced moments dephase physically.

        Returns:
            (freqs_hz, amps): flat complex64-compatible tensors over all d*d elements of
            every system (silent elements carry zero amplitude).
        """
        from mrsandbox.sequences.events import Gradient, ShapedPulse

        extents = [event.extent_mm for event in sequence.events if isinstance(event, Gradient)]
        extents += [event.extent_mm for event in sequence.events
                    if isinstance(event, ShapedPulse) and event.gradient_hz_per_mm]
        all_freqs, all_amps = [], []
        for system in systems:
            ctx = self.context(system, b1_scale=b1_scale)
            if not extents:
                freqs, amps = self._system_lines(sequence, ctx)
            else:
                span = max(extents)
                offsets = (np.arange(isochromats) + 0.5) / isochromats - 0.5
                stack = []
                for x in offsets * span:
                    freqs, amps = self._system_lines(
                        sequence, dataclasses.replace(ctx, gradient_x=float(x))
                    )
                    stack.append(amps)
                amps = ops.mean(ops.stack(stack, axis=0), axis=0)
            all_freqs.append(freqs)
            all_amps.append(amps)
        return ops.concatenate(all_freqs), ops.concatenate(all_amps)

    def _system_lines(self, sequence, ctx):
        """The (freqs, amps) of one system context, receiver phase applied."""
        rho = sequence.run(ctx.rho_eq, ctx)
        rho_e = ops.einsum("ai,ab,bj->ij", ops.conj(ctx.v), rho, ctx.v)
        det_e = ops.einsum("ai,ab,bj->ij", ops.conj(ctx.v), ctx.fplus, ctx.v)
        freqs = ops.expand_dims(ctx.lam, 1) - ops.expand_dims(ctx.lam, 0)
        phase = math.radians(self.receiver_phase_deg)
        receiver = nops.complex_from(
            ops.convert_to_tensor(math.cos(phase), dtype="float32"),
            ops.convert_to_tensor(math.sin(phase), dtype="float32"),
        )
        return ops.reshape(freqs, (-1,)), receiver * ops.reshape(det_e * ops.transpose(rho_e), (-1,))

    #****************#
    #   compiled     #
    #****************#
    def compiled(self, sequence, system):
        """A jit-compiled (t1_s, t2_s, b1_scale) -> (freqs, amps) for one system and sequence.

        The context (eigendecomposition, operators) is built once and closed over; only
        the relaxation times remain inputs, passed as backend tensors so the trace is
        reused across draws. This is what makes per-draw relaxation sampling cheap: the
        first call compiles, every further call replays the compiled graph.
        """
        from nifti_mrs_plus.backend import jit

        ctx0 = self.context(system)

        def fn(t1_s, t2_s, b1_scale):
            ctx = dataclasses.replace(ctx0, t1_s=t1_s, t2_s=t2_s, b1_scale=b1_scale)
            return self._system_lines(sequence, ctx)

        return jit(fn)

    #**************#
    #   spectrum   #
    #**************#
    def spectrum(self, sequence, *systems, npoints=4096, sw_hz=4000.0, lw_hz=1.0):
        """Complex spectrum on a freely choosable grid, rendered analytically from the lines.

        Args:
            sequence: The Sequence to run.
            *systems: SpinSystems to simulate and sum.
            npoints: Spectral points.
            sw_hz: Spectral width in Hz; the grid spans [0, sw_hz).
            lw_hz: Lorentzian linewidth (FWHM) in Hz.

        Returns:
            (ppm, spec): ascending ppm axis (float) and the complex spectrum.
        """
        freqs, amps = self.lines(sequence, *systems)
        return self.render_spectrum(freqs, amps, self.f0_mhz(*systems), npoints=npoints, sw_hz=sw_hz, lw_hz=lw_hz)

    #*********#
    #   fid   #
    #*********#
    def fid(self, sequence, *systems, npoints=2048, dwell_s=1.0 / 4000.0, lw_hz=1.0):
        """Complex FID on a freely choosable time grid: s(t) = sum_k amp_k e^{(i 2 pi f_k - r) t}."""
        freqs, amps = self.lines(sequence, *systems)
        return self.render_fid(freqs, amps, npoints=npoints, dwell_s=dwell_s, lw_hz=lw_hz)

    #***********************#
    #   render from lines   #
    #***********************#
    def render_spectrum(self, freqs, amps, f0_mhz, npoints=4096, sw_hz=4000.0, lw_hz=1.0):
        """Render any line list to a complex spectrum on a freely choosable grid.

        This is what makes weighted compositions cheap: scale or concatenate line lists
        (metabolite amplitudes, tissue weights) and render once, at any resolution.
        """
        grid = ops.arange(0, npoints, dtype=ops.dtype(freqs)) * (sw_hz / npoints)
        r = math.pi * ops.convert_to_tensor(lw_hz, dtype=ops.dtype(freqs)) * ops.ones_like(freqs)
        detuning = 2.0 * math.pi * (ops.expand_dims(grid, 0) - ops.expand_dims(freqs, 1))
        denominator = nops.complex_from(ops.expand_dims(r, 1) * ops.ones_like(detuning), detuning)
        spec = ops.sum(ops.expand_dims(amps, 1) / denominator, axis=0)
        return grid / f0_mhz, spec

    def render_fid(self, freqs, amps, npoints=2048, dwell_s=1.0 / 4000.0, lw_hz=1.0, gauss_hz=0.0):
        """Render any line list to a complex FID: s(t) = sum_k amp_k e^{(i 2 pi f_k - r) t - g t^2}.

        `lw_hz` and `gauss_hz` are scalars, or one value per line, so heterogeneous
        compositions render in a single call. `lw_hz` is the Lorentzian FWHM; `gauss_hz`
        an additional Gaussian FWHM (g = (pi gauss_hz)^2 / (4 ln 2)), together a Voigt
        line, the shape macromolecule parameterizations are stated in.
        """
        t = ops.arange(0, npoints, dtype=ops.dtype(freqs)) * dwell_s
        r = math.pi * ops.convert_to_tensor(lw_hz, dtype=ops.dtype(freqs)) * ops.ones_like(freqs)
        g = (math.pi * ops.convert_to_tensor(gauss_hz, dtype=ops.dtype(freqs))) ** 2 / (
            4.0 * math.log(2.0)
        ) * ops.ones_like(freqs)
        decay = (-ops.expand_dims(r, 1) * ops.expand_dims(t, 0)
                 - ops.expand_dims(g, 1) * ops.expand_dims(t * t, 0))
        phase = nops.complex_from(
            decay,
            2.0 * math.pi * ops.expand_dims(freqs, 1) * ops.expand_dims(t, 0),
        )
        return t, ops.sum(ops.expand_dims(amps, 1) * ops.exp(phase), axis=0)

    #*******************#
    #   ppm reference   #
    #*******************#
    def f0_mhz(self, *systems):
        """Spectrometer frequency in MHz (= Hz per ppm) of the systems' shared nucleus."""
        nuclei = {system.nucleus for system in systems}
        if len(nuclei) != 1:
            raise ValueError(f"systems span multiple nuclei {sorted(nuclei)}; simulate them separately")
        return systems[0].gamma_mhz_t * self.b0_t
