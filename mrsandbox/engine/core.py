####################################################################################################
#                                              core.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The GAMMA-class engine core: run any sequence on any spin systems and read the result   #
#          out as transition lines - which then render to a spectrum or FID at ANY grid for free,  #
#          so readout resolution is a rendering parameter, not a simulation cost. All tensor math  #
#          is keras.ops, in the set precision (backend.precision).                                 #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import hashlib
import math
import numbers
import os
import pathlib

import numpy as np

from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.backend import Backend, precision


#**************************************************************************************************#
#                                           Class Context                                          #
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
#                                           Class Engine                                           #
#**************************************************************************************************#
#                                                                                                  #
# Runs sequences on spin systems at a freely choosable field and renders the outcome.              #
#                                                                                                  #
#**************************************************************************************************#
class Engine:
    """Runs sequences on spin systems at a freely choosable field and renders the outcome.

    Args:
        b0: Field strength in tesla, or anything with a `b0_t` attribute (e.g. a
            hardware.Scanner preset); the field stays a free parameter either way.
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
        self._draws = {}
        self._renderers = {}

    #*************#
    #   context   #
    #*************#
    def context(self, system, b1_scale=1.0, offset_hz=0.0):
        """Build the tensor Context of one system at this field."""
        op = system.operators()
        lam, v = Backend.eigh_herm(ops.convert_to_tensor(system.hamiltonian(self.b0_t)))
        f0_mhz = system.gamma_mhz_t * self.b0_t
        nu = np.asarray(system.shifts_ppm, dtype=precision.REAL) * f0_mhz
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
            eye=ops.convert_to_tensor(np.eye(system.dim, dtype=precision.COMPLEX)),
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
            freqs, amps = self._chunked_lines(sequence, ctx, system.dim)
            all_freqs.append(freqs)
            all_amps.append(amps)
        return ops.concatenate(all_freqs), ops.concatenate(all_amps)

    def _chunked_lines(self, sequence, ctx, dim):
        """`_system_lines` over the ensemble in chunks of at most ShapedPulse's
        STEP_BLOCK_ELEMENTS density matrix elements, the amplitudes averaged by the chunks'
        sizes: a whole voxel's ensemble of a large system does not fit at once."""
        from mrsandbox.sequences.events import ShapedPulse

        batched = {name: value for name in ("gradient_x", "gradient_y", "gradient_z", "b1_scale")
                   if np.ndim(value := getattr(ctx, name)) > 0}
        size = max((int(np.shape(value)[0]) for value in batched.values()), default=1)
        chunk = max(1, ShapedPulse.STEP_BLOCK_ELEMENTS // dim ** 2)
        if size <= chunk:
            return self._system_lines(sequence, ctx)
        total = 0.0
        for start in range(0, size, chunk):
            part = dataclasses.replace(ctx, **{name: value[start:start + chunk]
                                               for name, value in batched.items()})
            freqs, amps = self._system_lines(sequence, part)
            total = total + amps * float(min(chunk, size - start))
        return freqs, total / float(size)

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
        dense multi-axis schemes. A design played as a whole (a tilted slab before its
        localization, Sequence.unfolded) takes the full grid instead, `isochromats` along
        every axis and all their combinations: without the tilt it sums exactly what the
        projections factorize."""
        isochromats = isochromats if isochromats is not None else self.isochromats
        extents = self.extents(sequence)
        grid = (np.arange(isochromats) + 0.5) / isochromats - 0.5
        if getattr(sequence, "unfolded", False):
            axes = sorted(extents)
            full = np.meshgrid(*(grid * extents[axis] for axis in axes), indexing="ij")
            return {axis: values.ravel().astype(precision.REAL) for axis, values in zip(axes, full)}
        positions = {}
        for order, axis in enumerate(sorted(extents)):
            target = max(1, int(round(isochromats * 0.6180339887 ** order)))
            nearby = (target + sign * offset for offset in range(isochromats) for sign in (1, -1))
            stride = 1 if order == 0 else next(
                (c for c in nearby if 1 <= c < isochromats and math.gcd(c, isochromats) == 1), 1)
            positions[axis] = (grid[(np.arange(isochromats) * stride) % isochromats]
                               * extents[axis]).astype(precision.REAL)
        return positions

    @staticmethod
    def extents(sequence):
        """The span of the isochromat ensemble per axis in mm, {axis: extent} over the
        axes a design's gradients use, its played spoiler's among them: the largest extent
        its events name (a tilted slab names all three)."""
        from mrsandbox.sequences.events import Gradient, ShapedPulse

        extents = {}
        for event in sequence.events + (getattr(sequence, "spoiler", None) or ()):
            if isinstance(event, Gradient):
                extents[event.axis] = max(extents.get(event.axis, 0.0), event.extent_mm)
            elif isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
                for axis in event.gradient_axis if event.gradient_normal is None else "xyz":
                    extents[axis] = max(extents.get(axis, 0.0), event.extent_mm)
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
            **{"gradient_" + axis:
               ops.convert_to_tensor(np.asarray(positions_mm, dtype=precision.REAL))})
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
            "gradient_" + axis: ops.convert_to_tensor(
                values + np.asarray(shift.get(axis, 0.0), precision.REAL))
            for axis, values in positions.items()})

    def _system_lines(self, sequence, ctx, propagators=None):
        """The (freqs, amps) of one system context, receiver phase applied; a vector
        gradient_x runs the whole isochromat ensemble as one batch, and the detected
        amplitudes average over it. `propagators` are shaped pulses' built ahead
        (Sequence.run)."""
        rho = sequence.run(ctx.rho_eq, ctx, propagators)
        rho_e = ops.einsum("ai,...ab,bj->...ij", ops.conj(ctx.v), rho, ctx.v)
        det_e = ops.einsum("ai,ab,bj->ij", ops.conj(ctx.v), ctx.fplus, ctx.v)
        freqs = ops.expand_dims(ctx.lam, 1) - ops.expand_dims(ctx.lam, 0)
        phase = math.radians(self.receiver_phase_deg)
        receiver = nops.complex_from(
            ops.convert_to_tensor(math.cos(phase), dtype=precision.REAL),
            ops.convert_to_tensor(math.sin(phase), dtype=precision.REAL),
        )
        amps = det_e * ops.einsum("...ij->...ji", rho_e)
        if ops.ndim(amps) == 3:
            amps = ops.mean(amps, axis=0)
        return ops.reshape(freqs, (-1,)), receiver * ops.reshape(amps, (-1,))

    #***********#
    #   graph   #
    #***********#
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
        whatever the draw, so a static index found on generic replays of the full graph
        (`_signal_index`) gathers its output, and a replay renders the few thousand lines
        that matter instead of every element; the full graph is the one compiled, so a
        pruned graph costs no compilation of its own. The bounds then count the kept
        elements per group. The complete layout stays the default.

        On jax with the persistent compilation cache configured, the graph is also
        kept as a serialized jax.export beside that cache (`_export_path`): a later
        session deserializes it and never traces the python at all, and its one XLA
        compilation lands in the persistent cache like any other. A call under a jax
        transformation falls back to the live graph (`_wrap_export`).
        """
        from nifti_mrs_plus.backend import jit

        isochromats = isochromats if isochromats is not None else self.isochromats
        # the timing shapes join the structure in the key: a tuple-valued timing (a tau list,
        # a waveform) can change its tree or length under one structure, and a stored export,
        # unlike a jit, cannot retrace for it
        key = (sequence.structure(), self._timing_shapes(sequence.timings()), groups,
               isochromats, pruned, precision.REAL)
        if key in self._compiled:
            return self._compiled[key]
        if pruned:
            full, bounds = self.graph(sequence, groups, isochromats)
            index, kept = self._signal_index(full, sequence, groups, isochromats, bounds)

            def gathered(*args):
                freqs, amps = full(*args)
                return ops.take(freqs, index), ops.take(amps, index)

            self._compiled[key] = (gathered, kept)
            return self._compiled[key]
        path = self._export_path("graph", sequence, groups, isochromats, False)
        stored = self._load_export(path)
        if stored is not None:
            fn = self._wrap_export(stored[0], lambda: self._raw_graph(sequence, groups,
                                                                      isochromats, False)[0])
        else:
            raw, _ = self._raw_graph(sequence, groups, isochromats, False)
            if path is not None:
                fn = self._wrap_export(self._store_export(path, raw,
                                                          self._specs(sequence, groups)),
                                       lambda: raw)
            else:
                fn = jit(raw)
        self._compiled[key] = (fn, self._group_bounds(groups))
        return self._compiled[key]

    #***************#
    #   raw graph   #
    #***************#
    def _raw_graph(self, sequence, groups, isochromats, pruned):
        """`graph`'s plain python function and bounds, unjitted and uncached: what a
        jit, an export, or a transformation fallback traces."""
        if pruned:
            full, bounds = self._raw_graph(sequence, groups, isochromats, False)
            # the index off the full graph as `graph` compiles it, the one compilation
            index, kept = self._signal_index(self.graph(sequence, groups, isochromats)[0],
                                             sequence, groups, isochromats, bounds)

            def gathered(t1_s, t2_s, b1_scale, offset_hz, timings, shift):
                freqs, amps = full(t1_s, t2_s, b1_scale, offset_hz, timings, shift)
                return ops.take(freqs, index), ops.take(amps, index)

            return gathered, kept
        contexts = [(index, self._ensemble_context(sequence, self.context(system), isochromats))
                    for index, group in enumerate(groups) for system in group]
        axes = sorted(self.extents(sequence))

        def fn(t1_s, t2_s, b1_scale, offset_hz, timings, shift):
            design = sequence.bound(timings)
            freqs, amps = [], []
            for index, ctx0 in contexts:
                moved = {"gradient_" + axis:
                         getattr(ctx0, "gradient_" + axis) + shift["xyz".index(axis)]
                         for axis in axes}
                ctx = dataclasses.replace(ctx0, t1_s=t1_s[index], t2_s=t2_s[index],
                                          b1_scale=b1_scale, offset_hz=offset_hz, **moved)
                f, a = self._system_lines(design, ctx)
                freqs.append(f)
                amps.append(a)
            return ops.concatenate(freqs), ops.concatenate(amps)

        return fn, self._group_bounds(groups)

    @staticmethod
    def _group_bounds(groups):
        """Element bounds of the groups in the full output, by their dimensions."""
        return np.cumsum([0] + [sum(system.dim ** 2 for system in group) for group in groups])

    @staticmethod
    def _timing_shapes(timings):
        """The shape of every timing, nesting kept: the part of a sequence's timings that
        shapes its graph beyond the structure."""
        if isinstance(timings, tuple):
            return tuple(Engine._timing_shapes(value) for value in timings)
        return np.shape(timings)

    #******************#
    #   signal index   #
    #******************#
    def _signal_index(self, full, sequence, groups, isochromats, bounds):
        """The elements of a graph's output that carry signal on generic replays
        (relaxation 1.5 and 0.2 s, the sequence's own timings, two frames: transmit
        scale 0.8 on resonance and 1.25 at 7 Hz off), as an index tensor and the
        groups' bounds in it. An element under 1e-9 of its group's maximum in both
        replays is a structural zero (a coherence filter's, zero at every draw), two
        orders below float32 itself: an element that carries signal sits on a null
        at one frame only by coincidence, and the other frame lifts it.

        The index is a pure function of the graph, so where graphs persist as exports
        it is kept beside them, under the export's own key (`_export_path`: the code,
        the field, the design's structure and timing shapes, the systems, the ensemble;
        the probe recipe is code), and a later session reads it back instead of
        replaying the full graph."""
        relax, frames, floor = (1.5, 0.2), ((0.8, 0.0), (1.25, 7.0)), 1e-9
        timings, shift = self.tensors(sequence.timings()), ops.zeros((3,), dtype=precision.REAL)
        n = len(bounds) - 1

        def probe(b1, offset):
            return (ops.convert_to_tensor(np.full(n, relax[0], precision.REAL)),
                    ops.convert_to_tensor(np.full(n, relax[1], precision.REAL)),
                    ops.convert_to_tensor(np.asarray(b1, precision.REAL)),
                    ops.convert_to_tensor(np.asarray(offset, precision.REAL)), timings, shift)

        path = self._export_path("index", sequence, groups, isochromats, False)
        path = None if path is None else path.parent.with_name("signal_index") / path.name
        if path is not None and path.exists():
            stored = np.load(path)
            return ops.convert_to_tensor(stored["index"]), stored["kept"]
        keep = np.zeros(int(bounds[-1]), dtype=bool)
        for b1, offset in frames:
            _, amps = full(*probe(b1, offset))
            amps = np.abs(np.asarray(ops.convert_to_numpy(amps)))
            for g in range(n):
                part = amps[bounds[g]:bounds[g + 1]]
                keep[bounds[g]:bounds[g + 1]] |= part > floor * part.max()
        kept = np.concatenate([[0], np.cumsum([int(keep[bounds[g]:bounds[g + 1]].sum())
                                               for g in range(n)])])
        index = np.flatnonzero(keep)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh = path.with_suffix(f".{os.getpid()}.npz")
            np.savez(fresh, index=index, kept=kept)
            os.replace(fresh, path)
        return ops.convert_to_tensor(index), kept

    #*************#
    #   exports   #
    #*************#
    _code_salt = None

    @classmethod
    def _export_salt(cls):
        """One hash over the physics sources and the library versions: any change
        invalidates every stored export, since a serialized graph cannot see code."""
        if cls._code_salt is None:
            import jax
            import keras

            # the physics folders by path, never as package attributes: a concurrent import
            # on another thread (the console's) leaves mrsandbox.engine unbound for a moment
            root = pathlib.Path(__file__).parents[1]
            folders = [root / name for name in ("engine", "sequences", "spins", "backend")]
            folders.append(pathlib.Path(nops.__file__).parent)
            digest = hashlib.sha256((jax.__version__ + keras.__version__).encode())
            for folder in folders:
                for file in sorted(folder.glob("*.py")):
                    digest.update(file.name.encode() + file.read_bytes())
            cls._code_salt = digest.hexdigest()
        return cls._code_salt

    def _export_path(self, kind, sequence, groups, isochromats, pruned):
        """Where a graph's serialized export persists, or None to stay live: only on the
        jax backend with the persistent compilation cache configured. The key holds the
        code salt, the engine's field, receiver phase and ensemble, the design's
        structure with its timing shapes, and every structural number of the systems;
        never a looser key, a stale export would silently play other physics."""
        import keras

        if keras.config.backend() != "jax":
            return None
        try:
            import flatbuffers  # noqa: F401  (jax.export's serialization needs it)
        except ImportError:
            return None
        import jax

        directory = jax.config.jax_compilation_cache_dir
        if directory is None:
            return None
        shapes = self._timing_shapes(sequence.timings())
        systems = tuple(tuple((s.name, s.nucleus, s.shifts_ppm, s.j_hz, s.scale, s.gamma_mhz_t,
                               s.spin, s.nuclei) for s in group) for group in groups)
        text = repr((self._export_salt(), self.b0_t, self.receiver_phase_deg, kind,
                     sequence.structure(), shapes, systems, isochromats, pruned,
                     precision.REAL))
        digest = hashlib.sha256(text.encode()).hexdigest()
        return pathlib.Path(directory) / "exports" / (digest + ".npz")

    def _specs(self, sequence, groups, draws=None):
        """ShapeDtypeStructs of a graph's inputs (t1_s, t2_s, b1_scale, offset_hz,
        timings, shift), float32 as every caller converts them; with `draws` (a
        symbolic dimension) the first four take it as their leading axis."""
        import jax

        lead = () if draws is None else (draws,)
        spec = lambda shape: jax.ShapeDtypeStruct(shape, precision.REAL)
        timings = jax.tree_util.tree_map(lambda value: spec(np.shape(value)),
                                         sequence.timings())
        return (spec(lead + (len(groups),)), spec(lead + (len(groups),)), spec(lead),
                spec(lead), timings, spec((3,)))

    def _load_export(self, path):
        """(the deserialized jax.export.Exported, the stored kept bounds or None), or
        None to build live: a missing file, or one another jax version wrote
        (deserialize refuses it), rebuilds and rewrites."""
        if path is None or not path.exists():
            return None
        from jax import export

        try:
            stored = np.load(path)
            loaded = export.deserialize(stored["export"].tobytes())
            kept = stored["kept"] if "kept" in stored.files else None
        except Exception:
            return None
        return loaded, kept

    def _store_export(self, path, fn, specs, kept=None):
        """Serialize `fn` at `specs` (traced and lowered, never run) to `path` and
        return its own deserialization, so every session compiles the same wrapped
        module and jax's persistent cache serves it."""
        import jax
        from jax import export

        data = np.frombuffer(export.export(jax.jit(fn))(*specs).serialize(), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = path.with_suffix(f".{os.getpid()}.npz")
        np.savez(fresh, export=data, **({} if kept is None else {"kept": kept}))
        os.replace(fresh, path)
        return export.deserialize(data.tobytes())

    @staticmethod
    def _wrap_export(exported, build):
        """The loaded export behind the live graph's interface: a call whose leaves fit
        the export's own signature runs the serialized module (each leaf made its own
        input's dtype, float32 but for held propagators), and any other call falls back
        to the live python graph, built once on first need: a tracer among the leaves
        (vmap has no rule for a serialized call, a gradient would need its VJP), or a
        shape the export was not made for (a vector transmit scale over the ensemble),
        which a jit quietly retraces for and a serialized module cannot. A symbolic
        dimension fits any size."""
        import jax

        from nifti_mrs_plus.backend import jit

        call = jax.jit(exported.call)
        shapes = tuple(aval.shape for aval in exported.in_avals)
        live = []

        def fits(leaf, shape):
            got = np.shape(leaf)
            return (not isinstance(leaf, jax.core.Tracer) and len(got) == len(shape)
                    and all(want == have for want, have in zip(shape, got)
                            if isinstance(want, int)))

        def wrapped(*args):
            leaves = jax.tree_util.tree_leaves(args)
            if len(leaves) != len(shapes) or not all(map(fits, leaves, shapes)):
                if not live:
                    live.append(jit(build()))
                return live[0](*args)
            return call(*jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(args),
                [ops.convert_to_tensor(leaf, dtype=aval.dtype)
                 for leaf, aval in zip(leaves, exported.in_avals)]))

        return wrapped

    @staticmethod
    def tensors(timings):
        """Sequence timings as float32 backend tensors, nesting kept."""
        if isinstance(timings, tuple):
            return tuple(Engine.tensors(value) for value in timings)
        return ops.convert_to_tensor(timings, dtype=precision.REAL)

    #**************#
    #   compiled   #
    #**************#
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
        row = lambda value: ops.expand_dims(ops.convert_to_tensor(value, dtype=precision.REAL), -1)

        def replay(t1_s, t2_s, b1_scale, offset_hz):
            return fn(row(t1_s), row(t2_s), b1_scale, offset_hz)

        if not batched:
            return replay
        mapped = Backend.vmap(replay, in_axes=(0, 0, None, None))

        def replay_batch(t1_s, t2_s, b1_scale, offset_hz):
            freqs, amps = mapped(ops.convert_to_tensor(t1_s, dtype=precision.REAL),
                                 ops.convert_to_tensor(t2_s, dtype=precision.REAL), b1_scale,
                                 offset_hz)
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
        # read anew each call: an event set by hand replays its new physics
        timings = self.tensors(sequence.timings())
        shift = ops.convert_to_tensor(np.asarray(
            [float((shift_mm or {}).get(axis, 0.0)) for axis in "xyz"], dtype=precision.REAL))
        return (lambda t1_s, t2_s, b1_scale, offset_hz: fn(t1_s, t2_s, b1_scale, offset_hz, timings,
                                                           shift),
                bounds)

    #******************#
    #   replay draws   #
    #******************#
    def replay_draws(self, sequence, groups, isochromats=None, shift_mm=None):
        """`compiled_batch`, pruned, mapped over draws: (t1_s, t2_s, b1_scale, offset_hz)
        with a leading draw axis on every input, t1_s and t2_s (draws, groups), the lines
        back as (draws, kept); plus the groups' bounds in the kept lines. One graph runs
        the whole batch (Backend.vmap); the mapped graph is keyed like `graph` and reads
        the design's timings at each call like `compiled_batch` (a design induced anew
        for every draw, scan.eddy, shares it). Where graphs persist as exports
        (`graph`), the mapped graph keeps its own, with the draws axis a symbolic
        dimension (a serialized call cannot go under vmap), so a session replays draws
        untraced too."""
        chromats = isochromats if isochromats is not None else self.isochromats
        key = (sequence.structure(), self._timing_shapes(sequence.timings()), groups, chromats,
               "draws", precision.REAL)
        if key not in self._draws:
            path = self._export_path("draws", sequence, groups, chromats, True)
            stored = self._load_export(path)
            if stored is None and path is not None:
                import jax
                from jax import export

                raw, kept = self._raw_graph(sequence, groups, chromats, True)
                stored = (self._store_export(
                    path, jax.vmap(raw, in_axes=(0, 0, 0, 0, None, None)),
                    self._specs(sequence, groups, draws=export.symbolic_shape("b")[0]),
                    kept=kept), kept)
            if stored is not None:
                import jax

                exported, kept = stored
                call = self._wrap_export(exported, lambda: jax.vmap(
                    self._raw_graph(sequence, groups, chromats, True)[0],
                    in_axes=(0, 0, 0, 0, None, None)))
                self._draws[key] = (call, kept)
            else:
                fn, bounds = self.graph(sequence, groups, chromats, pruned=True)
                self._draws[key] = (Backend.vmap(fn, in_axes=(0, 0, 0, 0, None, None)), bounds)
        call, kept = self._draws[key]
        timings = self.tensors(sequence.timings())
        shift = ops.convert_to_tensor(np.asarray(
            [float((shift_mm or {}).get(axis, 0.0)) for axis in "xyz"], dtype=precision.REAL))
        return (lambda t1_s, t2_s, b1_scale, offset_hz: call(t1_s, t2_s, b1_scale, offset_hz,
                                                             timings, shift),
                kept)

    #************#
    #   framed   #
    #************#
    def framed(self, sequence, groups, isochromats=None, shift_mm=None):
        """The design's pruned graph split at the transmit frame: `frame(b1_scale,
        offset_hz)` plays every shaped pulse once (their propagators, which of a draw
        only the transmit scale and the frequency offset decide), `lines(t1_s, t2_s,
        b1_scale, offset_hz, frame)` runs the state through them at a draw's
        relaxation, and `draws` is `lines` mapped over a leading draw axis of t1_s and
        t2_s with one frame shared; plus the groups' bounds in the kept lines, as
        `replay_draws` gives them. A batch of draws at one transmit frame pays the
        pulses once and replays the pass alone, most of a played design's cost; the
        lines equal the full graph's to float32. Keyed like `graph`, the index of
        the kept lines the full graph's; the functions read the design's timings at
        each call like `compiled_batch`. Where graphs persist as exports (`graph`),
        `frame` and `draws` keep their own (the draws axis a symbolic dimension), so a
        later session plays a batch untraced; `lines` stays a live jit."""
        from nifti_mrs_plus.backend import jit

        from mrsandbox.sequences.events import ShapedPulse

        isochromats = isochromats if isochromats is not None else self.isochromats
        key = (sequence.structure(), self._timing_shapes(sequence.timings()), groups,
               isochromats, "framed", precision.REAL)
        if key not in self._compiled:
            index, kept = self._signal_index(
                lambda *args: self.graph(sequence, groups, isochromats)[0](*args),
                sequence, groups, isochromats, self._group_bounds(groups))
            built = []

            def build():
                # the python of `frame` and `lines`, made on first need: a session the
                # exports serve never builds the contexts
                if built:
                    return built[0]
                contexts = [(group, self._ensemble_context(sequence, self.context(system),
                                                           isochromats))
                            for group, members in enumerate(groups) for system in members]
                axes = sorted(self.extents(sequence))

                def placed(ctx0, shift, **values):
                    moved = {"gradient_" + axis:
                             getattr(ctx0, "gradient_" + axis) + shift["xyz".index(axis)]
                             for axis in axes}
                    return dataclasses.replace(ctx0, **moved, **values)

                def pulses(design):
                    # each shaped pulse object once, in order: what `frame` plays and `lines` zips
                    seen = {}
                    for event in design.events:
                        if isinstance(event, ShapedPulse):
                            seen.setdefault(id(event), event)
                    return tuple(seen.values())

                def frame(b1_scale, offset_hz, timings, shift):
                    design = sequence.bound(timings)
                    return tuple(tuple(pulse.propagator(placed(ctx0, shift, b1_scale=b1_scale,
                                                               offset_hz=offset_hz))
                                       for pulse in pulses(design)) for _, ctx0 in contexts)

                def lines(t1_s, t2_s, b1_scale, offset_hz, timings, shift, frame):
                    design = sequence.bound(timings)
                    freqs, amps = [], []
                    for (group, ctx0), held in zip(contexts, frame):
                        ctx = placed(ctx0, shift, t1_s=t1_s[group], t2_s=t2_s[group],
                                     b1_scale=b1_scale, offset_hz=offset_hz)
                        f, a = self._system_lines(design, ctx,
                                                  dict(zip(map(id, pulses(design)), held)))
                        freqs.append(f)
                        amps.append(a)
                    return (ops.take(ops.concatenate(freqs), index),
                            ops.take(ops.concatenate(amps), index))

                built.append((frame, lines))
                return built[0]

            axes = (0, 0, None, None, None, None, None)
            lines = jit(lambda *args: build()[1](*args))
            frame_path, draws_path = (self._export_path(kind, sequence, groups, isochromats, True)
                                      for kind in ("frame", "framed draws"))
            if frame_path is None:
                # the contexts made here, not inside the first trace: a tensorflow graph's
                # constants belong to that graph, and the other graphs read them too
                build()
                self._compiled[key] = (jit(lambda *args: build()[0](*args)), lines,
                                       Backend.vmap(lines, in_axes=axes), kept)
            else:
                import jax
                from jax import export

                specs = self._specs(sequence, groups)
                stored = self._load_export(frame_path)
                if stored is None:
                    stored = (self._store_export(frame_path, build()[0], specs[2:]), None)
                frame = self._wrap_export(stored[0], lambda: build()[0])
                mapped = lambda: jax.vmap(build()[1], in_axes=axes)
                stored = self._load_export(draws_path)
                if stored is None:
                    rows = self._specs(sequence, groups, draws=export.symbolic_shape("b")[0])
                    held = jax.eval_shape(build()[0], *specs[2:])
                    stored = (self._store_export(draws_path, mapped(),
                                                 rows[:2] + specs[2:] + (held,)), None)
                self._compiled[key] = (frame, lines, self._wrap_export(stored[0], mapped), kept)
        frame, lines, draws, kept = self._compiled[key]
        timings = self.tensors(sequence.timings())
        shift = ops.convert_to_tensor(np.asarray(
            [float((shift_mm or {}).get(axis, 0.0)) for axis in "xyz"], dtype=precision.REAL))
        return (lambda b1_scale, offset_hz: frame(b1_scale, offset_hz, timings, shift),
                lambda t1_s, t2_s, b1_scale, offset_hz, held: lines(
                    t1_s, t2_s, b1_scale, offset_hz, timings, shift, held),
                lambda t1_s, t2_s, b1_scale, offset_hz, held: draws(
                    t1_s, t2_s, b1_scale, offset_hz, timings, shift, held),
                kept)

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
        return self.render_spectrum(freqs, amps, self.f0_mhz(*systems), npoints=npoints,
                                    sw_hz=sw_hz, lw_hz=lw_hz)

    #*****************#
    #   convergence   #
    #*****************#
    def convergence(self, sequence, *systems, lw_hz=4.0):
        """How far a design's spectrum on `systems` is from converged in each numerical
        choice of this engine: the largest difference of the complex spectrum (4096 points
        over 4 kHz, `lw_hz` lines) over its peak, against the spectrum as set, with twice
        the isochromats, with every shaped pulse at twice its steps (each sample held for
        two, the same waveform: what changes is the step discretization), and in float64.
        Returns {"isochromats": ..., "steps": ..., "float64": ...}. The design is refined as
        built: a pulse induced with a gradient response (its track per step) is not."""
        import copy

        from mrsandbox.sequences.events import Sequence, ShapedPulse

        def spectrum(design, isochromats):
            engine = Engine(self.b0_t, self.receiver_phase_deg, isochromats)
            _, spec = engine.spectrum(design, *systems, npoints=4096, sw_hz=4000.0, lw_hz=lw_hz)
            return np.asarray(ops.convert_to_numpy(spec)).astype(np.complex128)

        def refined(design):
            events = []
            for event in design.events:
                if isinstance(event, Sequence):
                    event = refined(event)
                elif isinstance(event, ShapedPulse):
                    event = copy.copy(event)
                    for name in ("envelope", "frequency_hz", "gradient_shape"):
                        if getattr(event, name) is not None:
                            setattr(event, name, np.repeat(np.asarray(getattr(event, name)), 2))
                events.append(event)
            design = copy.copy(design)
            design.events = tuple(events)
            return design

        reference = spectrum(sequence, self.isochromats)
        before = precision.REAL
        precision.set_precision("float64")
        try:
            double = spectrum(sequence, self.isochromats)
        finally:
            precision.set_precision(before)
        return {name: float(np.abs(other - reference).max() / np.abs(reference).max())
                for name, other in (("isochromats", spectrum(sequence, 2 * self.isochromats)),
                                    ("steps", spectrum(refined(sequence), self.isochromats)),
                                    ("float64", double))}

    #*********#
    #   fid   #
    #*********#
    def fid(self, sequence, *systems, npoints=2048, dwell_s=1.0 / 4000.0, lw_hz=1.0):
        """Complex FID on a freely choosable time grid:
        s(t) = sum_k amp_k e^{(i 2 pi f_k - r) t}."""
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
        """Render any line list to a complex FID:
        s(t) = sum_k amp_k e^{(i 2 pi f_k - r) t - g t^2}.

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
        multiplies a small integer, and the tail phase stays far more accurate than any
        float32 product of f and t (the dense fold, or a power table built in the
        graph). The frequencies are therefore constants to the graph (they are the
        Hamiltonian's, never a timing's).
        Voigt lines take the dense graph. Both replay jitted graphs cached per
        (npoints, dwell); the callers' stable line-bucket shapes keep retracing
        rare, so training draws render at replay speed.
        """
        voigt = not (isinstance(gauss_hz, numbers.Real) and gauss_hz == 0.0)
        key = (npoints, float(dwell_s), voigt, precision.REAL)
        if key not in self._renderers:
            self._renderers[key] = self._fid_renderer(npoints, dwell_s, voigt)
        dtype = ops.dtype(freqs)
        lw = ops.convert_to_tensor(lw_hz, dtype=dtype) * ops.ones_like(freqs)
        if not voigt:
            cycles = np.asarray(ops.convert_to_numpy(freqs), dtype=np.float64) * dwell_s
            within = ops.convert_to_tensor(np.mod(cycles, 1.0).astype(precision.REAL))
            across = ops.convert_to_tensor(
                np.mod(cycles * self._block(npoints), 1.0).astype(precision.REAL))
            return self._renderers[key](amps, lw, within, across)
        gauss = ops.convert_to_tensor(gauss_hz, dtype=dtype) * ops.ones_like(freqs)
        return self._renderers[key](freqs, amps, lw, gauss)

    def render_fids(self, freqs, amps, npoints=2048, dwell_s=1.0 / 4000.0, lw_hz=1.0,
                    gauss_hz=None):
        """Render K amplitude rows over one line list to K FIDs in one call: `amps` is
        (K, L), the result (K, npoints). Lorentzian lines take the factorized graph of
        `render_fid` with a batched matmul; what a multi-voxel acquisition uses to render
        every metabolite's unit FID once per truth. `lw_hz` is a scalar, one width per
        line, or (K, L), one per line and row: a batch of draws, each with its own
        widths. With `gauss_hz` (K, L) the lines are Voigt and take the dense graph, eight
        rows at a time (its phase table is rows x lines x points)."""
        if gauss_hz is not None:
            key = (npoints, float(dwell_s), "voigt rows", precision.REAL)
            if key not in self._renderers:
                self._renderers[key] = self._fid_renderer(npoints, dwell_s, True, rows=True)
            lw = (ops.convert_to_tensor(lw_hz, dtype=ops.dtype(freqs))
                  * ops.ones_like(gauss_hz, dtype=ops.dtype(freqs)))
            fids = [self._renderers[key](freqs, amps[first:first + 8], lw[first:first + 8],
                                         ops.convert_to_tensor(gauss_hz[first:first + 8],
                                                               dtype=ops.dtype(freqs)))[1]
                    for first in range(0, int(amps.shape[0]), 8)]
            return None, ops.concatenate(fids, axis=0)
        rows = np.ndim(lw_hz) == 2
        key = (npoints, float(dwell_s), "rows" if rows else "many", precision.REAL)
        if key not in self._renderers:
            self._renderers[key] = self._fid_renderer(npoints, dwell_s, False, many=True, rows=rows)
        dtype = ops.dtype(freqs)
        lw = ops.convert_to_tensor(lw_hz, dtype=dtype) * (
            ops.ones_like(amps, dtype=dtype) if rows else ops.ones_like(freqs))
        cycles = np.asarray(ops.convert_to_numpy(freqs), dtype=np.float64) * dwell_s
        within = ops.convert_to_tensor(np.mod(cycles, 1.0).astype(precision.REAL))
        across = ops.convert_to_tensor(
            np.mod(cycles * self._block(npoints), 1.0).astype(precision.REAL))
        return self._renderers[key](amps, lw, within, across)

    @staticmethod
    def _block(npoints):
        """The block length P of the factorized render: the largest power of two at or
        below sqrt(N), so both power tables stay near sqrt(N) wide."""
        return 1 << int(math.log2(math.sqrt(npoints)))

    def _fid_renderer(self, npoints, dwell_s, voigt, many=False, rows=False):
        """A jitted (amps, lw_hz, within, across) -> (t, fid) for the factorized
        Lorentzian graph (`within` and `across` the folded cycle counts f dwell mod 1
        and f P dwell mod 1 per line; with `many` the amplitudes are (K, L) rows and
        the result (K, npoints), with `rows` the widths are (K, L) too, each row's own),
        or a jitted (freqs, amps, lw_hz, gauss_hz) -> (t, fid) for the dense Voigt one,
        for one time grid."""
        from nifti_mrs_plus.backend import jit

        if not voigt:
            block = self._block(npoints)
            blocks = -(-npoints // block)

            def lorentzian(amps, lw_hz, within, across):
                t = ops.arange(0, npoints, dtype=ops.dtype(lw_hz)) * dwell_s
                q = ops.arange(0, block, dtype=ops.dtype(lw_hz))
                p = ops.arange(0, blocks, dtype=ops.dtype(lw_hz))
                decay = -math.pi * lw_hz * dwell_s
                if rows:
                    # every row its own decay: the tables are (K, L, count) and each row
                    # multiplies its own
                    inside = ops.exp(nops.complex_from(
                        ops.expand_dims(decay, 2) * q,
                        2.0 * math.pi * ops.expand_dims(ops.expand_dims(within, 0), 2) * q))
                    stride = ops.exp(nops.complex_from(
                        ops.expand_dims(decay * block, 2) * p,
                        2.0 * math.pi * ops.expand_dims(ops.expand_dims(across, 0), 2) * p))
                    table = ops.einsum("klm,kl,klp->kmp", stride, amps, inside)
                    return t, ops.reshape(table, (amps.shape[0], -1))[:, :npoints]
                # z^q inside a block and (z^P)^p block by block, each (L, count)
                inside = ops.exp(nops.complex_from(
                    ops.expand_dims(decay, 1) * q, 2.0 * math.pi * ops.expand_dims(within, 1) * q))
                stride = ops.exp(nops.complex_from(
                    ops.expand_dims(decay * block, 1) * p,
                    2.0 * math.pi * ops.expand_dims(across, 1) * p))
                if many:
                    table = ops.einsum("lm,kl,lp->kmp", stride, amps, inside)
                    return t, ops.reshape(table, (amps.shape[0], -1))[:, :npoints]
                table = ops.transpose(stride) @ (ops.expand_dims(amps, 1) * inside)
                return t, ops.reshape(table, (-1,))[:npoints]

            return jit(lorentzian)

        def fn(freqs, amps, lw_hz, gauss_hz):
            n = ops.arange(0, npoints, dtype=ops.dtype(freqs))
            t = n * dwell_s
            r = math.pi * lw_hz
            g = (math.pi * gauss_hz) ** 2 / (4.0 * math.log(2.0))
            if rows:
                # (K, L) widths: the table is (K, L, N) and each row sums its own lines
                decay = (-ops.expand_dims(r, 2) * t - ops.expand_dims(g, 2) * (t * t))
                cycles = ops.mod(ops.expand_dims(freqs * dwell_s, 1) * ops.expand_dims(n, 0), 1.0)
                phase = nops.complex_from(decay, 2.0 * math.pi * ops.expand_dims(cycles, 0))
                return t, ops.sum(ops.expand_dims(amps, 2) * ops.exp(phase), axis=1)
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
            raise ValueError(f"systems span multiple nuclei {sorted(nuclei)}; "
                             "simulate them separately")
        return systems[0].gamma_mhz_t * self.b0_t
