####################################################################################################
#                                             mrsi.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: The spatial dimension: an MRSI slice as a lattice of voxels over the virtual brain.     #
#          One realized truth serves the whole acquisition; spatial contrast emerges from the      #
#          physics (tissue fractions, the lipid ring at the boundary, per-voxel B0 dephasing,      #
#          per-position coil sensitivities), and the result is NIfTI-MRS with real spatial         #
#          dimensions and a world affine.                                                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numbers

import numpy as np

from mrsandbox.scan.voxel import draw, noise_scale, receiver_noise


#**************************************************************************************************#
#                                            Class MRSI                                            #
#**************************************************************************************************#
#                                                                                                  #
# One MRSI slice acquisition: a grid of voxels sharing one realized parameter tree.                #
#                                                                                                  #
#**************************************************************************************************#
class MRSI:
    """One MRSI slice acquisition: a grid of voxels sharing one realized parameter tree.

    The lattice lies in the axial plane at the given centre; every grid point is a
    full Voxel, so tissue weighting, water, lipids, macromolecules, B0 dephasing, and
    coil reception all apply per position. With `oversampling` above one, the object is
    simulated on a finer source grid and phase encoded through k-space: the acquisition
    keeps only the central matrix of k, which imprints the real point spread function
    (Gibbs ringing, the lipid ring bleeding inward).

    The k-space conventions, exactly: after the spatial FFT the acquired grid holds the
    centred k indices p = i - n//2 per axis, so DC sits at index n//2 and an even
    matrix carries its single Nyquist row at p = -n/2. `weighting="hamming"` multiplies
    k by the DC-centred window w(p) = 0.54 + 0.46 cos(pi p / (n/2)) per axis (w(0) = 1,
    so a uniform object keeps its amplitude; w = 0.08 at the Nyquist edge). Note that
    numpy's window peaks between samples on an even grid and would both attenuate DC
    and shift the point spread function by half a k step. `sampling="elliptical"`
    measures only the encodes with (2 px / nx)^2 + (2 py / ny)^2 <= 1, edge inclusive,
    the standard reduced phase encoding (Maudsley et al. 1994): the corners (about
    1 - pi/4, one fifth of the encodes) are never measured and stay zero, buying
    acquisition time for a slightly broader response function. `sampling` also accepts
    an explicit (nx, ny) array multiplying k, ordered like the acquired grid with DC at
    (nx//2, ny//2): booleans give an arbitrary measured-point mask, floats an arbitrary
    acquisition density. Amplitudes are normalized so full sampling of a uniform object
    is exactly the identity.

    The slab is shaped in z by `profile`: the given weights are the relative excitation
    at the centres of that many equal sub-slabs spanning thickness_mm, every source
    cell renders one voxel per sub-slab, and the cell signal is their weighted mean.
    The ideal box slab is the all-ones profile (the default, one sub-slab); a soft
    profile both tapers what the slab edges see and loses the total signal a real soft
    excitation loses. "gaussian" gives the truncated Gaussian of the pulse factories
    (de Graaf 2019 Eq 5.15 at 1 percent truncation, the small-tip shape of a Gaussian
    excitation, seven sub-slabs); an explicit array supplies any measured or computed
    profile, e.g. one read off the engine's own slice-profile machinery. Rendering cost
    scales with the number of sub-slabs.

    With `averages`, the weighting becomes a real acquisition-weighted averaging scheme
    (Pohmann and von Kienlin 2001): every encode is acquired NA(k) = round(averages
    times W(k)) times, W being the product the sampling and weighting knobs already
    define (all ones bare). The signal then carries the realized quantized weighting
    NA(k)/NA(0), receiver noise enters per encode in k-space with standard deviation
    proportional to sqrt(NA(k)) (encodes rounded to zero are never measured and
    contribute neither signal nor noise), and the header records the realized map and
    the total transient count, the scan time currency. Without `averages` the
    weighting stays the ideal continuous multiplication with white per-voxel noise.

    A three-long `matrix` makes the acquisition a 3D slab: nz phase-encoded partitions
    of thickness_mm each, spanning nz times thickness_mm around the centre, output
    (nx, ny, nz, npoints). The sampling, weighting, and averages maps then extend over
    kz (the ellipse becomes the inscribed ellipsoid, Maudsley's spherical encoding, and
    the Hamming window multiplies separably along every encoded axis), while the
    excitation profile weights the partitions across the slab instead of sub-slabs
    (an explicit profile must then carry one weight per partition). The z axis is
    phase encoded at the acquired resolution: oversampling refines in plane only.

    `trajectory` replaces the one-encode-per-excitation CSI readout with an echo
    planar traversal (EPSI): within every spectral dwell the ADC sweeps the whole kx
    axis (one sample per kx line), so each k line is sampled at its own visit time
    inside the period. The samples are exact FID values at those times, which is the
    whole physics: off-resonance phase accrues during the traversal, T2 star decays
    across it, and "epsi-bidirectional" (alternating sweep direction, no flyback)
    shows the classic odd-even ghost at half the spectral width, while "epsi" flies
    back between sweeps. `dead_time_s` is the finite flyback (or turnaround) time at
    the end of every sweep: the nx visits then compress into dwell_s - dead, spaced
    dwell_s / (nx + n_dead) with the dead time quantized to n_dead of those fine
    steps (the realized value is recorded); zero idealizes it away. ky stays phase
    encoded, and a slab additionally phase encodes its kz partitions, every partition
    sharing the kx visit clock. The header records the kx visit order so
    reconstruction can be scored against it; the standard reconstruction here ignores
    the visit times, as an uncorrected recon would. Any 2-D trajectory of Augmentrum's
    registry (rosette_2d, concentric_rings_2d, spiral_2d, ...) plays the same way:
    Augmentrum generates the shots for this lattice's geometry (`trajectory_params`
    are its parameters, e.g. n_shots or samples_per_ring), every shot traverses its
    samples once per spectral dwell (so every shot must carry the same sample count,
    the visit clock being shared), each sample reads the source at its visit time
    through Augmentrum's gridding NUFFT (the forward transform of the fine-grid
    image at that slot, Kaiser-Bessel), and the acquired grid is the density
    compensated adjoint of all samples, ignoring the visit times, scaled so a uniform
    object keeps its amplitude (one scalar gridding gain, stated). A trajectory
    composes with the z profile and with slab partitions, but not with oversampling,
    sampling, weighting, or averages.

    Stated limits: oversampling and its truncation PSF are in-plane only; the z profile
    is an excitation weighting, not a per-metabolite run under the slice gradient;
    trajectories run idealized gradients (the sweep timing is the model, not the
    gradient waveform).

    Args:
        scan: The VirtualScan to acquire with.
        center_mm: World centre of the slice or slab (x, y, z).
        matrix: Acquired grid, (nx, ny) for a slice or (nx, ny, nz) for a slab.
        voxel_mm: Nominal in-plane voxel edge (also the acquired grid spacing).
        thickness_mm: Slice thickness, or partition thickness of a slab; defaults to
            voxel_mm.
        oversampling: Source grid refinement per in-plane dimension; 1 gives ideal box
            voxels.
        weighting: None, or "hamming" k-space acquisition weighting.
        sampling: None (full Cartesian), "elliptical" corner-free coverage, or an
            explicit mask or density array matching the encoded grid.
        profile: None (an ideal box slab), "gaussian", or explicit excitation weights
            along z (per sub-slab for a slice, per partition for a slab).
        averages: None (one ideal acquisition per encode), or the transient count at
            the k-space centre of the acquisition-weighted scheme.
        trajectory: None (conventional CSI), "epsi" (flyback), "epsi-bidirectional"
            (alternating sweep direction), or the name of a 2-D Augmentrum trajectory.
        trajectory_params: Parameters of the Augmentrum trajectory.
        dead_time_s: Flyback or turnaround time at the end of every trajectory sweep,
            quantized to fine-grid samples; 0 idealizes it away.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, scan, center_mm=(0.0, 0.0, 0.0), matrix=(16, 16), voxel_mm=10.0,
                 thickness_mm=None, oversampling=1, weighting=None, sampling=None,
                 profile=None, averages=None, trajectory=None, trajectory_params=None,
                 dead_time_s=0.0):
        if weighting not in (None, "hamming"):
            raise ValueError(f"unknown weighting {weighting!r}; known: None, hamming")
        if averages is not None and (not isinstance(averages, numbers.Real)
                                     or int(averages) != averages or averages < 1):
            raise ValueError(f"averages must be a positive integer, got {averages!r}")
        self.averages = int(averages) if averages is not None else None
        self.scan = scan
        self.center_mm = tuple(float(x) for x in center_mm)
        self.matrix = tuple(int(n) for n in matrix)
        if len(self.matrix) not in (2, 3):
            raise ValueError(f"matrix must be (nx, ny) or (nx, ny, nz), got {matrix!r}")
        self.partitions = self.matrix[2] if len(self.matrix) == 3 else 1
        self.voxel_mm = float(voxel_mm)
        self.thickness_mm = float(thickness_mm) if thickness_mm is not None else self.voxel_mm
        self.oversampling = int(oversampling)
        self.weighting = weighting
        # z render voxels per cell: sub-slabs of one slice, or the slab's partitions
        renders = self.partitions if self.partitions > 1 else None
        if profile is None:
            self.profile = None
        elif isinstance(profile, str):
            if profile != "gaussian":
                raise ValueError(f"unknown profile {profile!r}; known: None, gaussian, or an array")
            steps = renders if renders is not None else 7
            tau = ((np.arange(steps) + 0.5) / steps) * 2.0 - 1.0
            self.profile = np.exp(np.log(0.01) * tau * tau)
        else:
            self.profile = np.asarray(profile, dtype=np.float64)
            if self.profile.ndim != 1 or self.profile.size < 1:
                raise ValueError("profile must be a one dimensional array of weights")
            if renders is not None and self.profile.size != renders:
                raise ValueError(f"a slab profile needs one weight per partition: "
                                 f"{self.profile.size} given for {renders} partitions")
        if sampling is None or (isinstance(sampling, str) and sampling == "elliptical"):
            self.sampling = sampling
        elif isinstance(sampling, str):
            raise ValueError(f"unknown sampling {sampling!r}; known: None, elliptical, or an array")
        else:
            mask = np.asarray(sampling, dtype=np.float32)
            if mask.shape != self.dims:
                raise ValueError(f"sampling mask shape {mask.shape} does not match the "
                                 f"encoded grid {self.dims}")
            self.sampling = mask
        nx, ny = self.source_matrix
        source_mm = self.voxel_mm / self.oversampling
        # in-plane the source grid refines with the oversampling; along z every source
        # cell holds its render voxels spanning the excited slab: the profile's
        # sub-slabs of the single slice, or one full-thickness box per slab partition
        span = self.thickness_mm * self.partitions
        nzr = renders if renders is not None else (
            self.profile.size if self.profile is not None else 1)
        sub = (source_mm, source_mm, span / nzr)

        def cell(i, j):
            x, y, z0 = self.source_position(i, j)
            centres = z0 + ((np.arange(nzr) + 0.5) / nzr - 0.5) * span
            return [scan.voxel((x, y, float(z)), sub, compiled=True) for z in centres]

        self.voxels = [[cell(i, j) for j in range(ny)] for i in range(nx)]
        self.trajectory_params = dict(trajectory_params or {})
        self.shots, self.trajectory_meta = None, None
        if trajectory in (None, "epsi", "epsi-bidirectional"):
            if self.trajectory_params:
                raise ValueError("trajectory_params belong to an Augmentrum trajectory")
        elif isinstance(trajectory, str):
            self.shots, self.trajectory_meta = self._augmentrum_shots(trajectory)
        else:
            raise ValueError(f"unknown trajectory {trajectory!r}; known: None, epsi, "
                             f"epsi-bidirectional, or an Augmentrum trajectory name")
        self.trajectory = trajectory
        if trajectory is not None and (
                self.oversampling != 1 or self.weighting is not None
                or self.sampling is not None or self.averages is not None):
            raise ValueError("a trajectory composes only with the z profile and slab "
                             "partitions: oversampling, sampling, weighting, and "
                             "averages stay conventional CSI")
        if dead_time_s and trajectory is None:
            raise ValueError("dead_time_s needs a trajectory: conventional CSI has no sweep")
        if dead_time_s < 0.0:
            raise ValueError(f"dead_time_s must be non-negative, got {dead_time_s!r}")
        self.dead_time_s = float(dead_time_s)

    #**************#
    #   geometry   #
    #**************#
    @property
    def dims(self):
        """The encoded k grid, (nx, ny) for a slice or (nx, ny, nz) for a slab."""
        return self.matrix if len(self.matrix) == 3 else self.matrix[:2]

    @property
    def source_matrix(self):
        """The simulated in-plane source grid, `oversampling` times finer than the
        acquisition."""
        return (self.matrix[0] * self.oversampling, self.matrix[1] * self.oversampling)

    @property
    def samples_per_period(self):
        """Readout samples inside one spectral dwell: the kx lines of an EPSI sweep,
        or the samples of every shot of an Augmentrum trajectory."""
        return self.matrix[0] if self.shots is None else self.shots.shape[1]

    def _augmentrum_shots(self, name):
        """The shots of a named Augmentrum trajectory for this lattice, as normalized
        coordinates (n_shots, samples, 2) in [-0.5, 0.5) per axis, and its metadata."""
        try:
            from augmentrum.sampling.kspace_sampling import TrajectoryRegistry
        except ImportError as error:
            raise ImportError("non-Cartesian trajectories need Augmentrum (the "
                              "mrsandbox[augment] extra)") from error
        nx, ny = self.matrix[0], self.matrix[1]
        geometry = {"matrix": (nx, ny, self.partitions),
                    "fov_mm": (nx * self.voxel_mm, ny * self.voxel_mm,
                               self.partitions * self.thickness_mm)}
        shots, meta = TrajectoryRegistry.create(name, **self.trajectory_params).generate(geometry)
        counts = {int(np.asarray(shot).shape[0]) for shot in shots}
        if len(counts) != 1:
            raise ValueError(f"every shot must carry the same sample count per spectral dwell "
                             f"(the shared visit clock); {name!r} gave {sorted(counts)}: set "
                             f"its per-shot sample parameter in trajectory_params")
        kmax = np.asarray(meta["kmax"], dtype=np.float64)[:2]
        stacked = np.stack([np.asarray(shot, dtype=np.float64)[:, :2] for shot in shots])
        return (stacked / (2.0 * kmax)).astype(np.float32), meta

    def position(self, i, j, l=0):
        """World position of acquired grid point (i, j, l), the lattice centred on
        center_mm; l indexes the slab partition (0 for a slice)."""
        nx, ny = self.matrix[0], self.matrix[1]
        return (self.center_mm[0] + (i - (nx - 1) / 2.0) * self.voxel_mm,
                self.center_mm[1] + (j - (ny - 1) / 2.0) * self.voxel_mm,
                self.center_mm[2] + (l - (self.partitions - 1) / 2.0) * self.thickness_mm)

    def source_position(self, i, j):
        """World in-plane position of source grid point (i, j), at the slab centre."""
        nx, ny = self.source_matrix
        spacing = self.voxel_mm / self.oversampling
        return (self.center_mm[0] + (i - (nx - 1) / 2.0) * spacing,
                self.center_mm[1] + (j - (ny - 1) / 2.0) * spacing,
                self.center_mm[2])

    @property
    def affine(self):
        """Voxel-to-world matrix of the acquisition grid; the third axis steps one
        partition (or the slice thickness)."""
        affine = np.diag([self.voxel_mm, self.voxel_mm, self.thickness_mm, 1.0])
        affine[:3, 3] = self.position(0, 0, 0)
        return affine

    #*****************#
    #   acquisition   #
    #*****************#
    def acquire(self, seed=None, npoints=512, dwell_s=1.0 / 4000.0):
        """Acquire the slice and return NIfTI-MRS carrying the full provenance.

        One parameter draw serves every voxel (a subject has one set of concentrations).
        The clean source grid is phase encoded (spatial FFT), truncated to the central
        acquisition matrix of k with the optional weighting, and decoded back; receiver
        noise is drawn per acquired voxel and per coil, or, with `averages`, per encode
        in k-space with the acquisition-weighted scheme's per-count scaling. With a
        CoilArray on the scan the data gains a trailing DIM_COIL dimension.
        """
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext

        seed, truth, rng = draw(self.scan.params, seed)
        data, _, n_dead = self._decoded(truth, npoints, dwell_s)
        noise = self._noise(truth, rng, npoints, dwell_s)
        if noise is not None:
            data = data + noise
        hdr = self._header(seed, truth, dwell_s, n_dead)
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr, affine=self.affine)

    def acquire_components(self, seed=None, npoints=512, dwell_s=1.0 / 4000.0):
        """Acquire the slice component by component and return NIfTI-MRS on DIM_USER_0
        (after DIM_COIL with a CoilArray): the same subject, noise, header, and
        provenance as `acquire` with the same arguments, every cell split into the
        signal components named in the header's SimulationComponents (the union over
        the cells in order of appearance, noise last), each encoded and decoded on
        its own, so the components sum to the acquisition exactly (the encoding is
        linear). The second file of a training pair (Voxel.acquire_components)."""
        from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext

        seed, truth, rng = draw(self.scan.params, seed)
        data, names, n_dead = self._decoded(truth, npoints, dwell_s, components=True)
        noise = self._noise(truth, rng, npoints, dwell_s)
        if noise is None:
            noise = np.zeros(data.shape[:-1], dtype=np.complex64)
        data = np.concatenate([data, noise[..., None]], axis=-1)
        names = names + ["noise"]
        hdr = self._header(seed, truth, dwell_s, n_dead)
        hdr.set_dim_info(0 if self.scan.coils is None else 1, "DIM_USER_0",
                         info="MRSandbox: the signal components named in SimulationComponents.")
        hdr.set_user_def("SimulationComponents", {"names": names},
                         "MRSandbox: the components on DIM_USER_0, in order; they sum to "
                         "the acquisition of the same seed.")
        return gen_nifti_mrs_hdr_ext(data, dwell_s, hdr, affine=self.affine)

    def _decoded(self, truth, npoints, dwell_s, components=False):
        """The clean decoded lattice of one truth, (nx, ny, nz, npoints[, coils]) or with
        `components` a trailing axis over the signal components, as (data, the
        component names, the dead points per period)."""
        coils = self.scan.coils
        nx, ny = self.matrix[0], self.matrix[1]
        nz = self.partitions
        sx, sy = self.source_matrix
        # a trajectory samples every kx line once per spectral dwell, so the sources
        # render on the fine ADC grid and the sampler picks each line's visit times;
        # a finite dead time appends n_dead unread fine samples to every period
        n_dead = self._dead_points(dwell_s) if self.trajectory is not None else 0
        period = self.samples_per_period + n_dead
        fine = npoints if self.trajectory is None else npoints * period
        step = dwell_s if self.trajectory is None else dwell_s / period
        tail = (npoints,) + ((len(coils),) if coils is not None else ())
        weights = (self.profile if self.profile is not None
                   else np.ones(len(self.voxels[0][0])))
        # every cell's FIDs, {component: fid} per voxel of the cell (one component,
        # the sum, without `components`), the metabolites rendered once per truth
        # (and resolved frame) as unit FIDs every voxel forms its own from
        cells, names = {}, []
        for i in range(sx):
            for j in range(sy):
                cells[i, j] = []
                for voxel in self.voxels[i][j]:
                    basis = voxel.metabolite_basis(truth, fine, step)
                    if components:
                        fids = voxel._component_fids(truth, fine, step, basis=basis)
                        names += [n for n in fids if n not in names]
                    else:
                        fids = {"": voxel._clean_fid(truth, fine, step, basis=basis)[1]}
                    cells[i, j].append(fids)
        components_names = names
        names = names or [""]
        zero = np.zeros(fine, dtype=np.complex64)
        source = np.zeros((sx, sy) + ((nz,) if nz > 1 else ()) + (fine,) + tail[1:]
                          + (len(names),), dtype=np.complex64)
        for (i, j), cell in cells.items():
            for k, name in enumerate(names):
                fids = [f.get(name, zero) for f in cell]
                if nz == 1:
                    if coils is None:
                        source[i, j, ..., k] = (sum(w * fid for w, fid in zip(weights, fids))
                                                / weights.size)
                    else:
                        channels = sum(
                            w * coils.sensitivities(voxel.position_mm, voxel.size_mm)[:, None]
                            * fid[None, :]
                            for w, voxel, fid in zip(weights, self.voxels[i][j], fids)
                        ) / weights.size
                        source[i, j, ..., k] = np.transpose(channels)
                else:
                    for l, (w, voxel, fid) in enumerate(zip(weights, self.voxels[i][j], fids)):
                        channel = w * fid if coils is None else np.transpose(
                            w * coils.sensitivities(voxel.position_mm, voxel.size_mm)[:, None]
                            * fid[None, :])
                        source[i, j, l, ..., k] = channel

        if self.trajectory is None:
            encoded = self._encode(source)
        elif self.shots is None:
            encoded = self._sample_trajectory(source, npoints, n_dead)
        else:
            encoded = self._sample_nonuniform(source, npoints, n_dead)
        data = np.zeros((nx, ny, nz) + tail + (len(names),), dtype=np.complex64)
        if nz == 1:
            data[:, :, 0] = encoded
        else:
            data[:] = encoded
        return (data if components else data[..., 0]), components_names, n_dead

    def _noise(self, truth, rng, npoints, dwell_s):
        """The receiver noise of one acquisition, (nx, ny, nz, npoints[, coils]), or
        None when the drawn sigma is zero."""
        coils = self.scan.coils
        nx, ny = self.matrix[0], self.matrix[1]
        nz = self.partitions
        tail = (npoints,) + ((len(coils),) if coils is not None else ())
        # the drawn sigma scaled to the acquired cell (its in-plane size times the
        # slab thickness), the field, and the spectral bandwidth; the averages enter
        # per encode below
        sigma = truth.noise.sigma * noise_scale(
            self.voxel_mm ** 2 * self.thickness_mm / 1000.0, self.scan.engine.b0_t, dwell_s)
        if sigma <= 0.0:
            return None
        noise = np.zeros((nx, ny, nz) + tail, dtype=np.complex64)
        if self.averages is None:
            for index in np.ndindex(nx, ny, nz):
                if coils is None:
                    noise[index] = receiver_noise(rng, sigma, npoints)
                else:
                    noise[index] = np.transpose(coils.noise(rng, sigma, npoints))
            return noise
        # per-encode noise, summed over the NA(k) transients and normalized by
        # the centre count like the signal: std sigma sqrt(prod(dims)) sqrt(NA)
        # over NA(0) in the centred k frame, nothing where never measured
        counts = self._averages_map()
        centre = tuple(n // 2 for n in self.dims)
        scale = sigma * np.sqrt(float(np.prod(self.dims))) * np.sqrt(counts) / counts[centre]
        if coils is None:
            noise_k = scale[..., None] * (
                rng.standard_normal(self.dims + (npoints,))
                + 1j * rng.standard_normal(self.dims + (npoints,)))
        else:
            noise_k = np.zeros(self.dims + (npoints, len(coils)), dtype=np.complex64)
            for index in np.ndindex(*self.dims):
                if counts[index] > 0:
                    noise_k[index] = np.transpose(
                        coils.noise(rng, float(scale[index]), npoints))
        axes = tuple(range(len(self.dims)))
        image = np.fft.ifftn(np.fft.ifftshift(noise_k, axes=axes), axes=axes)
        return image.reshape((nx, ny, nz) + tail).astype(np.complex64)

    def _header(self, seed, truth, dwell_s, n_dead):
        """The header extension every acquisition of the lattice carries."""
        from nifti_mrs.hdr_ext import Hdr_Ext

        period = self.samples_per_period + n_dead
        hdr = Hdr_Ext(self.voxels[0][0][0]._f0_mhz(), "1H")
        if self.scan.coils is not None:
            hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_user_def(
            "SimulationGroundTruth",
            {"seed": seed, "params": truth.to_dict()},
            "MRSandbox: the realized parameter tree of this acquisition.",
        )
        hdr.set_user_def(
            "SimulationGrid",
            {"brain": self.scan.brain.name, "center_mm": list(self.center_mm),
             "matrix": list(self.matrix), "voxel_mm": self.voxel_mm,
             "thickness_mm": self.thickness_mm, "oversampling": self.oversampling,
             "weighting": self.weighting,
             "sampling": self.sampling.tolist() if isinstance(self.sampling, np.ndarray)
             else self.sampling,
             "profile": self.profile.tolist() if self.profile is not None else None,
             "averages": self.averages, "trajectory": self.trajectory,
             **({"dead_time_s": n_dead * dwell_s / period}
                if self.trajectory is not None else {}),
             **({"visit_order": self._visit_order(0).tolist(),
                 "visit_order_odd": self._visit_order(1).tolist()}
                if self.trajectory is not None and self.shots is None else {}),
             **({"trajectory_params": self.trajectory_params,
                 "shots": int(self.shots.shape[0]),
                 "samples_per_shot": int(self.shots.shape[1]),
                 "kmax_per_m": [float(k) for k in self.trajectory_meta["kmax"][:2]]}
                if self.shots is not None else {}),
             **({"transients": int(self._averages_map().sum()),
                 "averages_map": self._averages_map().tolist()}
                if self.averages is not None else {})},
            "MRSandbox: the virtual brain and the slice lattice.",
        )
        return hdr

    #**************#
    #   encoding   #
    #**************#
    def _encode(self, source):
        """Phase encode the source grid and decode the central k-space matrix.

        With oversampling 1, no weighting, and full sampling this is exactly the
        identity (ideal box voxels). Otherwise the truncation to the acquired matrix
        imprints the point spread function. The forward FFT runs over the oversampled
        source grid and the inverse over the acquired grid, so dividing by
        oversampling^2 makes a uniform object keep its amplitude exactly; the sampling
        mask and the DC-centred weighting (w(0) = 1) both preserve that identity.
        """
        if (self.oversampling == 1 and self.weighting is None and self.sampling is None
                and self.averages is None):
            return source
        axes = tuple(range(len(self.dims)))
        k = np.fft.fftshift(np.fft.fftn(source, axes=axes), axes=axes)
        nx, ny = self.matrix[0], self.matrix[1]
        sx, sy = self.source_matrix
        cx, cy = sx // 2, sy // 2
        k = k[cx - nx // 2:cx - nx // 2 + nx, cy - ny // 2:cy - ny // 2 + ny]
        pad = [1] * (k.ndim - len(self.dims))
        if self.averages is not None:
            counts = self._averages_map()
            centre = tuple(n // 2 for n in self.dims)
            k = k * (counts / counts[centre]).reshape(*self.dims, *pad)
        else:
            weight = self._weight_map()
            if weight is not None:
                k = k * weight.reshape(*self.dims, *pad)
        decoded = np.fft.ifftn(np.fft.ifftshift(k, axes=axes), axes=axes)
        return (decoded / self.oversampling ** 2).astype(np.complex64)

    def _sampling_mask(self):
        """The sampling mask on the encoded k grid as float32, or None for full
        Cartesian: the inscribed ellipse sum((2 p / n)^2) <= 1 over the encoded axes
        (edge inclusive; the ellipsoid of a slab) for "elliptical", or the user's
        explicit array verbatim."""
        if self.sampling is None:
            return None
        if isinstance(self.sampling, np.ndarray):
            return self.sampling
        radius = np.zeros(self.dims, dtype=np.float64)
        for axis, n in enumerate(self.dims):
            u = (np.arange(n) - n // 2) / (n / 2.0)
            shape = [1] * len(self.dims)
            shape[axis] = n
            radius = radius + (u ** 2).reshape(shape)
        return (radius <= 1.0 + 1e-9).astype(np.float32)

    def _weight_map(self):
        """The combined sampling times weighting map on the encoded k grid, or None
        when both knobs are absent (full uniform Cartesian)."""
        weight = self._sampling_mask()
        if self.weighting == "hamming":
            # the k-space Hamming centred on DC, one factor per encoded axis:
            # w(p) = 0.54 + 0.46 cos(pi p / (n/2)), w(0) exactly 1; numpy's np.hamming
            # peaks between samples on an even grid and would attenuate DC and shift
            # the point spread function half a k step
            window = np.ones(self.dims, dtype=np.float64)
            for axis, n in enumerate(self.dims):
                w = 0.54 + 0.46 * np.cos(np.pi * (np.arange(n) - n // 2) / (n / 2.0))
                shape = [1] * len(self.dims)
                shape[axis] = n
                window = window * w.reshape(shape)
            window = window.astype(np.float32)
            weight = window if weight is None else weight * window
        return weight

    def _averages_map(self):
        """The realized per-encode transient counts NA(k) = round(averages x W(k)),
        the quantized acquisition weighting; encodes rounded to zero are unmeasured."""
        weight = self._weight_map()
        if weight is None:
            weight = np.ones(self.dims, dtype=np.float32)
        return np.rint(self.averages * weight).astype(int)

    #****************#
    #   trajectory   #
    #****************#
    def _dead_points(self, dwell_s):
        """The dead time quantized to fine-grid samples: the nx visits occupy
        dwell - dead, so the fine step is dwell / (nx + n_dead) and the realized dead
        time is n_dead of those steps (recorded in the header)."""
        if not self.dead_time_s < dwell_s:
            raise ValueError(f"dead_time_s must stay below the spectral dwell: got "
                             f"{self.dead_time_s} at dwell {dwell_s}")
        samples = self.samples_per_period
        return int(round(self.dead_time_s * samples / (dwell_s - self.dead_time_s)))

    def _visit_order(self, parity):
        """Sample slot of every centred kx index inside one spectral period: the sweep
        runs from low to high kx; bidirectional periods of odd parity reverse it."""
        order = np.arange(self.matrix[0])
        if self.trajectory == "epsi-bidirectional" and parity == 1:
            return order[::-1].copy()
        return order

    def _sample_trajectory(self, source, npoints, n_dead=0):
        """Sample the EPSI readout and decode it the standard way.

        The fine-grid sources are spatially Fourier transformed, every kx line is
        taken at its visit time inside each spectral period (exact FID values, no
        interpolation; a finite dead time occupies the last n_dead fine samples of
        every period, so the visits compress into its front), and the k data is
        decoded ignoring the visit times, exactly as an uncorrected reconstruction
        would. A slab phase encodes ky and kz per excitation, so every partition
        shares the kx visit clock."""
        nx, ny = self.matrix[0], self.matrix[1]
        nz = self.partitions
        axes = tuple(range(len(self.dims)))
        k = np.fft.fftshift(np.fft.fftn(source, axes=axes), axes=axes)
        period = nx + n_dead
        kdata = np.empty(self.dims + (npoints,) + source.shape[len(axes) + 1:],
                         dtype=np.complex64)
        rows = np.arange(nx)
        cols = np.arange(ny)
        for n in range(npoints):
            slots = n * period + self._visit_order(n % 2)
            if nz == 1:
                kdata[:, :, n] = k[rows[:, None], cols[None, :], slots[:, None]]
            else:
                kdata[:, :, :, n] = k[rows[:, None, None], cols[None, :, None],
                                      np.arange(nz)[None, None, :], slots[:, None, None]]
        decoded = np.fft.ifftn(np.fft.ifftshift(kdata, axes=axes), axes=axes)
        return decoded.astype(np.complex64)

    def _sample_nonuniform(self, source, npoints, n_dead=0):
        """Sample an Augmentrum trajectory and reconstruct it the standard way.

        Slot s of every spectral period is the fine sample every shot reads at its
        s-th coordinate: the fine-grid sources at those times form one image whose
        channels are the spectral points (and partitions, and coils), read off at
        the shots' s-th coordinates by the gridding NUFFT's forward transform, so
        every sample is the exact FID value at its visit time. The acquired grid is
        the density compensated (Pipe) adjoint of all samples, decoded ignoring the
        visit times as an uncorrected reconstruction would, and divided by the
        gridding gain of a uniform object (one scalar, stated)."""
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

        nx, ny = self.matrix[0], self.matrix[1]
        n_shots, samples = self.shots.shape[0], self.shots.shape[1]
        period = samples + n_dead
        nufft = GriddingNUFFT((nx, ny))
        # the fine time axis is the third (slice) or fourth (slab) axis of the source
        time_axis = 2 if self.partitions == 1 else 3
        kdata = None
        for s in range(samples):
            slot = np.take(source, np.arange(npoints) * period + s, axis=time_axis)
            image = np.moveaxis(slot, (0, 1), (-2, -1))  # (..., nx, ny)
            lead = image.shape[:-2]
            read = nufft.forward(image.reshape((-1, nx, ny)), self.shots[:, s, :])  # (C, shots)
            if kdata is None:
                kdata = np.empty((read.shape[0], n_shots, samples), dtype=np.complex64)
            kdata[:, :, s] = np.asarray(read)
        coords = self.shots.reshape(-1, 2)
        weights = nufft.density_weights(coords)[None, :]
        image = np.asarray(nufft.adjoint(kdata.reshape(kdata.shape[0], -1) * weights, coords))
        # the gain of a uniform unit object through the same sampling and reconstruction
        ones = np.ones((1, nx, ny), dtype=np.complex64)
        gain = np.asarray(nufft.adjoint(np.asarray(nufft.forward(ones, coords)) * weights, coords))
        image = image / np.abs(gain).mean()
        decoded = np.moveaxis(image.reshape(lead + (nx, ny)), (-2, -1), (0, 1))
        return decoded.astype(np.complex64)
