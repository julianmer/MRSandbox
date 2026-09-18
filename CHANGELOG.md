# Changelog

The format follows Keep a Changelog. Versions follow `mrsandbox.__version__`.

## Unreleased

### Changed, and it changes seeded output

* The noise prior is calibrated against Big GABA: the single-transient NAA SNR of 36 subjects at
  its three Siemens sites, measured on the raw coils and transients (median 36.4), is reproduced
  at a reference sigma of 6.3. `noise.sigma` is now Uniform(5.3, 8.0), was (3.5, 7.5). The first
  calibration had read the simulated spectrum conjugated with a 4.65 ppm carrier (`Voxel.fid` is
  referenced to 0 ppm, a line at f hertz is exp(+i 2 pi f t)) and used one unusually noisy
  dataset; the two errors nearly cancelled.
* The receiver bandwidth enters the noise: the per-point sigma grows with the square root of the
  bandwidth, referenced to the 5 kHz of the calibration.
* The ideal `Steam` keeps the stimulated echo pathway (+1, 0, -1). It had kept the anti-echo,
  whose amplitude is right and whose shifts never refocus.
* The induced field before the readout takes the sign of the readout kernel, and the exponential
  eddy current terms reach the steady state of a train when the sequence has a TR.
* The `SemiLaser` default fits the 3 T body coil: 6 ms, 1.5 kHz, 1000 Hz (23.5 uT).
* Spin systems split into their J-connected parts (creatine was one 64-dimensional system).
  Spectra are unchanged, names gain a suffix where a group splits.

### Added

* Localization played, summed by one-dimensional projection (Zhang, An, Shen 2017):
  `LocalizedPress`, `LocalizedSteam`, `LocalizedSemiLaser`, `LocalizedMegaPress` (one or two
  editing bands, `scheme()` for HERMES and HERCULES), `LocalizedMegaSemiLaser`; the `Projection`
  event; `Sequence.confined`; `Engine.profile`.
* Pulses: `ShapedPulse.slr` (linear and minimum phase), `ShapedPulse.goia_wurst`, a modulated
  slice gradient (`gradient_shape`), the induced field during a shaped pulse (`induced`).
* MRSI inside the box a localized design selects (`box_center_mm`), and the odd sweep
  imperfections of bidirectional EPSI (`odd_sweeps`).
* The field inside the pulses: `VirtualScan(field_groups=N)` and `FieldMap.groups`;
  `FieldMap.from_susceptibility` (the dipole model) and the QSM challenge head's field.
* The QSM challenge 2.0 head as a third brain (`Brain.load("qsm_rc2")`, placed by hand, never
  redistributed), its maps (`resources.QSMHead`), and its scalp as a fat layer for ICBM152 and
  BigBrain-MR (`Brain.load(name, fat="qsm_rc2")`). Lipids run through a localized design's voxel
  profile cell by cell.
* The digital receiver (`hardware.Receiver`): oversampling, decimation filter, first-point
  corruption, coloured noise; the receiver delay (`acquisition.delay_s`).
* Scanner presets `siemens_prisma_3t`, `ge_mr750_3t`, `philips_elition_3t`; the measured Prisma
  gradient impulse response (`resources.GIRF`).
* Temperature and pH: `VirtualScan(temperature_c, ph)` moves the metabolite shifts and couplings
  by the published functions (Wermter et al. 2026); `water.temperature_c` moves the water line.
* `ChallengeMM`, the macromolecule basis of the 2024 MRSI challenge (in the repository's `data/`
  folder, never in the wheel).
* Transients: head displacement, rotation, motion events coupled to Augmentrum's, and the shim a
  move spoils.
* Components files for `Multiplex` and `MRSI`; resources placed by hand (`page` entries) and
  resources served from the repository (`repo` entries).

### Fixed

* The shift of the isochromat ensemble is an input of the compiled graph: one compilation
  serves every place.
* Shaped pulses played more than once are computed once per run, and their step propagators
  form in blocks (peak memory of a localized scan fell from 9.6 to 4.2 GB).
