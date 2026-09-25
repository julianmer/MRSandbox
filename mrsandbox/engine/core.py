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
import numbers

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
        offset_hz: The frequency offset the sequence runs at (the scanner's frequency
            error): every spin appears at its resonance minus this in the pulse and
            delay frame, so selective bands, shaped profiles, and suppression respond
            to it. The readout stays in the nominal frame (the scan shifts it).
        gradient_x, gradient_y, gradient_z: Isochromat positions along the three
            gradient axes in millimetres (scalars, or vectors batching the ensemble).
        fz: The longitudinal total-spin operator (gradient offsets act through it).
        mz, hadamard: Per-spin magnetic quantum numbers (n, d) and the transverse
            eigenbasis H (d, d) of spins.Operators, so ideal pulses rotate without an
            eigendecomposition in the graph.
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
    mz: object = None
    hadamard: object = None
    offset_hz: object = 0.0
    gradient_x: object = 0.0
    gradient_y: object = 0.0
    gradient_z: object = 0.0


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
        isochromats: Default ensemble size for gradient-bearing sequences (positions
            spanning each gradient extent); raise it for dense multi-axis schemes.
            Per-call arguments override it.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, b0, receiver_phase_deg=90.0, isochromats=32):
        self.b0_t = getattr(b0, "b0_t", b0)
        self.receiver_phase_deg = receiver_phase_deg
        self.isochromats = isochromats
        self._compiled = {}
        self._timings = {}
        self._renderers = {}

    #*************#
    #   context   #
    #*************#
    def context(self, system, b1_scale=1.0, offset_hz=0.0):
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
            mz=ops.convert_to_tensor(op.mz),
            hadamard=ops.convert_to_tensor(op.hadamard),
            t1_s=system.t1_s,
            t2_s=system.t2_s,
            b1_scale=b1_scale,
            offset_hz=offset_hz,
        )

    #***********#
    #   lines   #
    #***********#
    def lines(self, sequence, *systems, b1_scale=1.0, offset_hz=0.0, isochromats=None,
              shift_mm=None):
        """Transition lines of `sequence` run on each system, concatenated.

        The pre-readout density matrix is projected onto the Hamiltonian eigenbasis, where
        every element is one exponential: s(t) = sum_k amp_k * exp(+i 2 pi freq_k t).
        When the sequence contains Gradient events the lines are averaged over an
        isochromat ensemble of `isochromats` positions (the engine default when None)
        spanning the gradient extent, so crusher pairs rephase coherently and
        unbalanced moments dephase physically. `offset_hz` runs the sequence off
        resonance (see Context); the returned frequencies stay in the nominal frame.
        `shift_mm` ({axis: mm}) moves the ensemble off the design's centre.

        Returns:
            (freqs_hz, amps): flat complex64-compatible tensors over all d*d elements of
            every system (silent elements carry zero amplitude).
        """
        isochromats = isochromats if isochromats is not None else self.isochromats
        all_freqs, all_amps = [], []
        for system in systems:
            ctx = self._ensemble_context(
                sequence, self.context(system, b1_scale=b1_scale, offset_hz=offset_hz),
                isochromats, shift_mm)
            freqs, amps = self._system_lines(sequence, ctx)
            all_freqs.append(freqs)
            all_amps.append(amps)
        return ops.concatenate(all_freqs), ops.concatenate(all_amps)

    def positions(self, sequence, isochromats=None):
        """The isochromat positions of a gradient-bearing design, {axis: (isochromats,)
        mm from the voxel centre} over the axes its gradients use, empty for a design
        without gradients.

        Every used axis gets the exact uniform grid over its largest extent, so all
        per-axis dephasing laws hold exactly; with several axes the later grids are
        reordered by deterministic coprime strides (successive golden-ratio fractions of
        the count, nudged to the nearest coprime), which decorrelates the axes so
        cross-axis moments dephase instead of spuriously refocusing, while each axis's
        own marginal stays the exact uniform grid. Cross-axis combinations are thereby
        sampled along a torus line, a stated approximation; raise `isochromats` for
        dense multi-axis schemes."""
        isochromats = isochromats if isochromats is not None else self.isochromats
        extents = self.extents(sequence)
        grid = (np.arange(isochromats) + 0.5) / isochromats - 0.5
        positions = {}
        for order, axis in enumerate(sorted(extents)):
            target = max(1, int(round(isochromats * 0.6180339887 ** order)))
            nearby = (target + sign * offset for offset in range(isochromats) for sign in (1, -1))
            stride = 1 if order == 0 else next(
                (c for c in nearby if 1 <= c < isochromats and math.gcd(c, isochromats) == 1), 1)
            positions[axis] = (grid[(np.arange(isochromats) * stride) % isochromats]
                               * extents[axis]).astype(np.float32)
        return positions

    @staticmethod
    def extents(sequence):
        """The span of the isochromat ensemble per axis in mm, {axis: extent} over the
        axes a design's gradients use: the largest extent its events name."""
        from mrsandbox.sequences.events import Gradient, ShapedPulse

        extents = {}
        for event in sequence.events:
            if isinstance(event, Gradient):
                extents[event.axis] = max(extents.get(event.axis, 0.0), event.extent_mm)
            elif isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
                extents[event.gradient_axis] = max(extents.get(event.gradient_axis, 0.0),
                                                   event.extent_mm)
        return extents

    #*************#
    #   profile   #
    #*************#
    def profile(self, sequence, system, axis, positions_mm, b1_scale=1.0, offset_hz=0.0):
        """What a projected design leaves of a system at single positions along one
        axis, the other axes at the voxel centre: (freqs, amps) with amps of shape
        (positions, lines), nothing averaged (the design's `Projection` events are
        left out). For an uncoupled spin the response of a localization is the product
        of its three axis profiles over the centre's, which is how pulse design papers
        build a voxel from one-dimensional Bloch profiles; for coupled spins the
        profile along one axis is exact and the product is not."""
        from mrsandbox.sequences.events import Projection, Sequence

        played = Sequence(*(e for e in sequence.events if not isinstance(e, Projection)),
                          tr_s=sequence.tr_s, cycles=sequence.cycles)
        ctx = dataclasses.replace(
            self.context(system, b1_scale=b1_scale, offset_hz=offset_hz),
            **{"gradient_" + axis: ops.convert_to_tensor(np.asarray(positions_mm, dtype=np.float32))})
        rho = played.run(ctx.rho_eq, ctx)
        rho_e = ops.einsum("ai,...ab,bj->...ij", ops.conj(ctx.v), rho, ctx.v)
        det_e = ops.einsum("ai,ab,bj->ij", ops.conj(ctx.v), ctx.fplus, ctx.v)
        freqs = ops.expand_dims(ctx.lam, 1) - ops.expand_dims(ctx.lam, 0)
        phase = math.radians(self.receiver_phase_deg)
        receiver = complex(math.cos(phase), math.sin(phase))
        amps = np.asarray(ops.convert_to_numpy(det_e * ops.einsum("...ij->...ji", rho_e)))
        return (np.asarray(ops.convert_to_numpy(ops.reshape(freqs, (-1,)))),
                receiver * amps.reshape(len(positions_mm), -1))

    def _ensemble_context(self, sequence, ctx, isochromats, shift_mm=None):
        """The context with the isochromat position vectors a gradient-bearing sequence
        needs (the events broadcast them as one shared batch axis; `positions`,
        moved by `shift_mm` per axis), or the context untouched."""
        positions = self.positions(sequence, isochromats)
        if not positions:
            return ctx
        shift = dict(shift_mm or {})
        return dataclasses.replace(ctx, **{
            "gradient_" + axis: ops.convert_to_tensor(values + np.float32(shift.get(axis, 0.0)))
            for axis, values in positions.items()})

    def _system_lines(self, sequence, ctx):
        """The (freqs, amps) of one system context, receiver phase applied; a vector
        gradient_x runs the whole isochromat ensemble as one batch, and the detected
        amplitudes average over it."""
        rho = sequence.run(ctx.rho_eq, ctx)
        rho_e = ops.einsum("ai,...ab,bj->...ij", ops.conj(ctx.v), rho, ctx.v)
        det_e = ops.einsum("ai,ab,bj->ij", ops.conj(ctx.v), ctx.fplus, ctx.v)
        freqs = ops.expand_dims(ctx.lam, 1) - ops.expand_dims(ctx.lam, 0)
        phase = math.radians(self.receiver_phase_deg)
        receiver = nops.complex_from(
            ops.convert_to_tensor(math.cos(phase), dtype="float32"),
            ops.convert_to_tensor(math.sin(phase), dtype="float32"),
        )
        amps = det_e * ops.einsum("...ij->...ji", rho_e)
        if ops.ndim(amps) == 3:
            amps = ops.mean(amps, axis=0)
        return ops.reshape(freqs, (-1,)), receiver * ops.reshape(amps, (-1,))

    #**********#
    #   graph  #
    #**********#
    def graph(self, sequence, groups, isochromats=None, pruned=False):
        """The jit-compiled graph of a sequence design on many systems:
        (t1_s, t2_s, b1_scale, offset_hz, timings) -> (freqs, amps), with t1_s and
        t2_s one value per group, `timings` laid out as Sequence.timings (backend
        tensors; `tensors` converts), and the lines of every system concatenated in
        group order. Returns the function and the element bounds of the groups in
        the output, so a caller slices its metabolite's lines without another call.

        `groups` is a tuple of tuples of structural systems, each group sharing one
        relaxation entry (a metabolite and its coupled groups). The contexts
        (eigendecomposition, operators, the isochromat ensemble of a gradient-bearing
        design) are built once and closed over; the relaxation times, the transmit
        scale, the frequency offset, and every timing of the design are inputs, so
        the first call compiles and every further call replays, for any timings.
        Graphs are cached on the engine, keyed by the sequence structure, the
        structural systems, and the ensemble size: two instances of one design (a TE
        series, a candidate set of a closed loop) share one compilation, and a
        gradient with respect to the timings flows through `value_and_grad`. The
        graph takes a last input, the (3,) shift of the ensemble off the design's
        centre in mm, for content that sits elsewhere (the fat layer beyond the
        brain, a cell of an MRSI lattice inside a selected box): one compilation
        serves every place.

        `pruned` keeps the output to the elements that ever carry signal: the
        coherence filters silence most of the d*d transition elements exactly,
        whatever the draw, so a static index found on generic replays (`_signal_index`)
        is gathered inside the graph and a replay transfers and renders the few
        thousand lines that matter instead of every element. The bounds then count the
        kept elements per group. The complete layout stays the default.
        """
        from nifti_mrs_plus.backend import jit

        isochromats = isochromats if isochromats is not None else self.isochromats
        key = (sequence.structure(), groups, isochromats, pruned)
        if key not in self._compiled and pruned:
            full, bounds = self.graph(sequence, groups, isochromats)
            index, kept = self._signal_index(full, sequence, bounds)

            def gathered(t1_s, t2_s, b1_scale, offset_hz, timings, shift):
                freqs, amps = full(t1_s, t2_s, b1_scale, offset_hz, timings, shift)
                return ops.take(freqs, index), ops.take(amps, index)

            self._compiled[key] = (jit(gathered), kept)
        if key not in self._compiled:
            contexts = [(index, self._ensemble_context(sequence, self.context(system), isochromats))
                        for index, group in enumerate(groups) for system in group]
            axes = sorted(self.extents(sequence))
            bounds = np.cumsum([0] + [sum(system.dim ** 2 for system in group)
                                      for group in groups])

            def fn(t1_s, t2_s, b1_scale, offset_hz, timings, shift):
                design = sequence.bound(timings)
                freqs, amps = [], []
                for index, ctx0 in contexts:
                    moved = {"gradient_" + axis: getattr(ctx0, "gradient_" + axis) + shift["xyz".index(axis)]
                             for axis in axes}
                    ctx = dataclasses.replace(ctx0, t1_s=t1_s[index], t2_s=t2_s[index],
                                              b1_scale=b1_scale, offset_hz=offset_hz, **moved)
                    f, a = self._system_lines(design, ctx)
                    freqs.append(f)
                    amps.append(a)
                return ops.concatenate(freqs), ops.concatenate(amps)

            self._compiled[key] = (jit(fn), bounds)
        return self._compiled[key]

    #******************#
    #   signal index   #
    #******************#
    def _signal_index(self, full, sequence, bounds):
        """The elements of a graph's output that carry signal on generic replays
        (relaxation 1.5 and 0.2 s, transmit scales 0.8 to 1.25, offsets 0 and 7 Hz,
        the sequence's own timings), as an index tensor and the groups' bounds in it.
        An element under 1e-9 of its group's maximum in every replay is a structural
        zero (a coherence filter's), two orders below float32 itself: a null one
        timing happens to sit on cannot reach it."""
        timings, shift = self.tensors(sequence.timings()), ops.zeros((3,), dtype="float32")
        n = len(bounds) - 1
        keep = np.zeros(int(bounds[-1]), dtype=bool)
        for b1 in (0.8, 1.0, 1.25):
            for offset in (0.0, 7.0):
                _, amps = full(ops.convert_to_tensor(np.full(n, 1.5, np.float32)), ops.convert_to_tensor(np.full(n, 0.2, np.float32)),
                               ops.convert_to_tensor(np.float32(b1)), ops.convert_to_tensor(np.float32(offset)), timings, shift)
                amps = np.abs(np.asarray(ops.convert_to_numpy(amps)))
                for g in range(n):
                    part = amps[bounds[g]:bounds[g + 1]]
                    keep[bounds[g]:bounds[g + 1]] |= part > 1e-9 * part.max()
        kept = np.concatenate([[0], np.cumsum([int(keep[bounds[g]:bounds[g + 1]].sum()) for g in range(n)])])
        return ops.convert_to_tensor(np.flatnonzero(keep)), kept

    @staticmethod
    def tensors(timings):
        """Sequence timings as float32 backend tensors, nesting kept."""
        if isinstance(timings, tuple):
            return tuple(Engine.tensors(value) for value in timings)
        return ops.convert_to_tensor(timings, dtype="float32")

    #****************#
    #   compiled     #
    #****************#
    def compiled(self, sequence, system, isochromats=None, batched=False, shift_mm=None):
        """(t1_s, t2_s, b1_scale, offset_hz) -> (freqs, amps) for one system, replaying
        the design's graph (`graph`) with this sequence's own timings: the first call
        of a design compiles, every further call, at any timings, replays.

        `batched` maps the replay over draws: t1_s and t2_s are (B,) vectors, one
        relaxation pair per draw, b1_scale and offset_hz stay one frame, and the
        amplitudes come back as (B, L) over the one line list (Backend.vmap: one
        graph over the whole batch on jax, a loop elsewhere), the rows a
        `render_fids` call turns into B FIDs at once. `shift_mm` moves the
        ensemble (see `graph`)."""
        fn, _ = self.compiled_batch(sequence, ((system,),), isochromats, shift_mm)
        row = lambda value: ops.expand_dims(ops.convert_to_tensor(value, dtype="float32"), -1)

        def replay(t1_s, t2_s, b1_scale, offset_hz):
            return fn(row(t1_s), row(t2_s), b1_scale, offset_hz)

        if not batched:
            return replay
        mapped = Backend.vmap(replay, in_axes=(0, 0, None, None))

        def replay_batch(t1_s, t2_s, b1_scale, offset_hz):
            freqs, amps = mapped(ops.convert_to_tensor(t1_s, dtype="float32"),
                                 ops.convert_to_tensor(t2_s, dtype="float32"), b1_scale, offset_hz)
            return freqs[0], amps

        return replay_batch

    #********************#
    #   compiled batch   #
    #********************#
    def compiled_batch(self, sequence, groups, isochromats=None, shift_mm=None, pruned=False):
        """(t1_s, t2_s, b1_scale, offset_hz) -> (freqs, amps) for many systems (t1_s
        and t2_s one value per group) plus the group bounds, replaying the design's
        graph with this sequence's own timings; see `graph` (`pruned` keeps the
        output to the elements that carry signal)."""
        fn, bounds = self.graph(sequence, groups, isochromats, pruned)
        # a design's timings become tensors once: every draw replays the same design (a
        # profile showed the conversion of twenty events' values per replay)
        held = self._timings.get(id(sequence))
        if held is None or held[0] is not sequence:
            held = self._timings[id(sequence)] = (sequence, self.tensors(sequence.timings()))
        timings = held[1]
        shift = ops.convert_to_tensor(np.asarray(
            [float((shift_mm or {}).get(axis, 0.0)) for axis in "xyz"], dtype=np.float32))
        return (lambda t1_s, t2_s, b1_scale, offset_hz: fn(t1_s, t2_s, b1_scale, offset_hz, timings,
                                                           shift),
                bounds)

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

        Lorentzian lines (`gauss_hz` the scalar 0) render through the factorized
        graph: with the point index split as n = P p + q, the sum s(n) = sum_k a_k z_k^n
        is one complex matmul of the two small power tables a_k z_k^q and (z_k^P)^p,
        so the cost is a BLAS product instead of L x N exponentials. The phases of
        both tables come from cycle counts folded on the host in float64, f dwell mod
        1 and f P dwell mod 1, so inside the float32 graph a folded phase only ever
        multiplies a small integer: the tail phase error stays near 5e-5 rad at 8192
        points, where any float32 product of f and t (the dense fold, or a power
        table built in the graph) sits at 3e-3. The frequencies are therefore
        constants to the graph (they are the Hamiltonian's, never a timing's).
        Voigt lines take the dense graph. Both replay jitted graphs cached per
        (npoints, dwell); the callers' stable line-bucket shapes keep retracing
        rare, so training draws render at replay speed.
        """
        voigt = not (isinstance(gauss_hz, numbers.Real) and gauss_hz == 0.0)
        key = (npoints, float(dwell_s), voigt)
        if key not in self._renderers:
            self._renderers[key] = self._fid_renderer(npoints, dwell_s, voigt)
        dtype = ops.dtype(freqs)
        lw = ops.convert_to_tensor(lw_hz, dtype=dtype) * ops.ones_like(freqs)
        if not voigt:
            cycles = np.asarray(ops.convert_to_numpy(freqs), dtype=np.float64) * dwell_s
            within = ops.convert_to_tensor(np.mod(cycles, 1.0).astype(np.float32))
            across = ops.convert_to_tensor(
                np.mod(cycles * self._block(npoints), 1.0).astype(np.float32))
            return self._renderers[key](amps, lw, within, across)
        gauss = ops.convert_to_tensor(gauss_hz, dtype=dtype) * ops.ones_like(freqs)
        return self._renderers[key](freqs, amps, lw, gauss)

    def render_fids(self, freqs, amps, npoints=2048, dwell_s=1.0 / 4000.0, lw_hz=1.0):
        """Render K amplitude rows over one Lorentzian line list to K FIDs in one call:
        `amps` is (K, L), the result (K, npoints). The factorized graph of `render_fid`
        with a batched matmul; what a multi-voxel acquisition uses to render every
        metabolite's unit FID once per truth."""
        key = (npoints, float(dwell_s), "many")
        if key not in self._renderers:
            self._renderers[key] = self._fid_renderer(npoints, dwell_s, False, many=True)
        dtype = ops.dtype(freqs)
        lw = ops.convert_to_tensor(lw_hz, dtype=dtype) * ops.ones_like(freqs)
        cycles = np.asarray(ops.convert_to_numpy(freqs), dtype=np.float64) * dwell_s
        within = ops.convert_to_tensor(np.mod(cycles, 1.0).astype(np.float32))
        across = ops.convert_to_tensor(np.mod(cycles * self._block(npoints), 1.0).astype(np.float32))
        return self._renderers[key](amps, lw, within, across)

    @staticmethod
    def _block(npoints):
        """The block length P of the factorized render: the largest power of two at or
        below sqrt(N), so both power tables stay near sqrt(N) wide."""
        return 1 << int(math.log2(math.sqrt(npoints)))

    def _fid_renderer(self, npoints, dwell_s, voigt, many=False):
        """A jitted (amps, lw_hz, within, across) -> (t, fid) for the factorized
        Lorentzian graph (`within` and `across` the folded cycle counts f dwell mod 1
        and f P dwell mod 1 per line; with `many` the amplitudes are (K, L) rows and
        the result (K, npoints)), or a jitted (freqs, amps, lw_hz, gauss_hz) ->
        (t, fid) for the dense Voigt one, for one time grid."""
        from nifti_mrs_plus.backend import jit

        if not voigt:
            block = self._block(npoints)
            blocks = -(-npoints // block)

            def lorentzian(amps, lw_hz, within, across):
                t = ops.arange(0, npoints, dtype=ops.dtype(lw_hz)) * dwell_s
                q = ops.arange(0, block, dtype=ops.dtype(lw_hz))
                p = ops.arange(0, blocks, dtype=ops.dtype(lw_hz))
                decay = -math.pi * lw_hz * dwell_s
                # z^q inside a block and (z^P)^p block by block, each (L, count)
                inside = ops.exp(nops.complex_from(
                    ops.expand_dims(decay, 1) * q, 2.0 * math.pi * ops.expand_dims(within, 1) * q))
                stride = ops.exp(nops.complex_from(
                    ops.expand_dims(decay * block, 1) * p,
                    2.0 * math.pi * ops.expand_dims(across, 1) * p))
                if many:
                    table = ops.einsum("lm,kl,lp->kmp", stride, amps, inside)
                    return t, ops.reshape(table, (amps.shape[0], -1))[:, :npoints]
                table = ops.matmul(ops.transpose(stride), ops.expand_dims(amps, 1) * inside)
                return t, ops.reshape(table, (-1,))[:npoints]

            return jit(lorentzian)

        def fn(freqs, amps, lw_hz, gauss_hz):
            n = ops.arange(0, npoints, dtype=ops.dtype(freqs))
            t = n * dwell_s
            r = math.pi * lw_hz
            g = (math.pi * gauss_hz) ** 2 / (4.0 * math.log(2.0))
            decay = (-ops.expand_dims(r, 1) * ops.expand_dims(t, 0)
                     - ops.expand_dims(g, 1) * ops.expand_dims(t * t, 0))
            # the oscillation f t runs to thousands of cycles on long FIDs, where
            # float32 representation alone costs ~1e-3 rad; folding (f dwell) n modulo
            # one cycle before the 2 pi scaling keeps the final phase at wrapped
            # magnitude, leaving only the single-product float32 floor (fully removing
            # it needs 64-bit floats, which jax disables by default)
            cycles = ops.mod(ops.expand_dims(freqs * dwell_s, 1) * ops.expand_dims(n, 0), 1.0)
            phase = nops.complex_from(decay, 2.0 * math.pi * cycles)
            return t, ops.sum(ops.expand_dims(amps, 1) * ops.exp(phase), axis=0)

        return jit(fn)

    #*******************#
    #   ppm reference   #
    #*******************#
    def f0_mhz(self, *systems):
        """Spectrometer frequency in MHz (= Hz per ppm) of the systems' shared nucleus."""
        nuclei = {system.nucleus for system in systems}
        if len(nuclei) != 1:
            raise ValueError(f"systems span multiple nuclei {sorted(nuclei)}; simulate them separately")
        return systems[0].gamma_mhz_t * self.b0_t
