####################################################################################################
#                                         virtual_scan.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: The composition layer and public face: VirtualScan binds a brain, a scanner, and a      #
#          sequence under one Params tree and places Voxels, MRSI lattices, and training           #
#          generators on it. The prior catalogue lives in defaults.py, the voxel in voxel.py.      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.database import Database, tissue
from mrsandbox.engine import Engine
from mrsandbox.params import Normal, Params
from mrsandbox.phantom.downfield import NoDownfield, ParameterizedDownfield
from mrsandbox.phantom.lipids import NoLipids, Skull
from mrsandbox.phantom.macromolecules import CoupledMM, MeasuredMM, NoMM
from mrsandbox.phantom.solution import Phantom
from mrsandbox.phantom.water import Suppressed
from mrsandbox.scan.defaults import default_params
from mrsandbox.scan.voxel import Voxel
from mrsandbox.spins import SpinSystem


#**************************************************************************************************#
#                                        Class VirtualScan                                         #
#**************************************************************************************************#
#                                                                                                  #
# A brain, a scanner, and a sequence, configured by one Params tree.                               #
#                                                                                                  #
#**************************************************************************************************#
class VirtualScan:
    """A brain, a scanner, and a sequence, configured by one Params tree.

    Args:
        brain: A phantom.Brain, or a phantom.Phantom, the in vitro subject: the scan then
            takes no macromolecules, lipids, or downfield peaks, the phantom's temperature
            and pH, and its `params()` over the defaults (each still overridable here).
        scanner: A hardware.Scanner, or a field strength in tesla.
        sequence: A sequences.Sequence.
        params: Params overriding the defaults, composed as defaults | params.
        database: Database instance; defaults to the packaged one.
        spins: The spin table the metabolites play (Database.SPIN_SYSTEMS): one source
            name for all, or {metabolite: source} with the rest on "newest" (the newest
            publication per metabolite, the default).
        coils: A hardware.CoilArray for multi-channel reception; None gives one ideal
            channel.
        noise_map: Where the scan is quiet and where it is not, for an acquisition
            without channels: a hardware.CoilArray (its optimum combination's noise,
            `noise_gain`, one where it was `calibrated`), or a phantom.Volume of a
            measured noise level (a noise-only scan without RF, reconstructed through
            the scanner's own combination; Kellman and McVeigh 2005), read as its
            box mean over its box mean at `noise_reference_mm`. The drawn sigma
            belongs to the reference place; every voxel scales it by the map. With
            `coils` the channels carry their own noise and the map is not used.
        noise_reference_mm: The place the noise prior was calibrated at (the
            parietal midline voxel of scan.defaults).
        field: A phantom.FieldMap for B0 inhomogeneity; None gives a uniform field.
        field_groups: How many groups of the voxel's post-shim field distribution play
            the sequence at their own offset. One (the default) lets the field act on
            the FID alone; more put it into the pulses too, so an editing or a
            suppression pulse sits off its band in the part of the voxel where the
            field is off, at that many times the simulation per voxel.
        water_groups: How many groups of the voxel's field distribution the WATER alone plays
            the sequence at, each at its own offset with its own dephasing (the other sources
            keep `field_groups`): a suppression pulse misses the water where the field is off
            its band, which the voxel's mean offset cannot show, and the microscopic spread
            (lineshape.gauss_hz) goes into the groups rather than onto the result, so a
            train a few hertz wide leaves what the spread puts in or out of it. One (the
            default) plays the water at the mean like everything else; the water is one
            spin, so more costs little.
        transmit: A phantom.Volume of the relative transmit field B1+ (1.0 nominal);
            every voxel runs its pulses at the drawn b1_scale times its local value.
            None gives the nominal field everywhere.
        mm: A phantom.MMSource; defaults to the measured 3 T shape (MeasuredMM, the
            console's, which carries the whole background where the coupled systems fall
            short), and to the coupled systems (CoupledMM) for an editing design, whose
            pulses act on them where the measured shape carries no editing response; none
            in vitro.
        downfield: A phantom.DownfieldSource; defaults to the nine-peak
            parameterization, presaturated along with the default suppressed water.
        girf: A resources.GIRF, the scanner's measured gradient impulse response;
            every readout then carries the fields the design's own gradients induce
            through it (scan.eddy) in place of the drawn exponential terms of the
            gradients leaves. None draws those terms.
        temperature_c, ph: The tissue's temperature and pH, 37 C and 7 for a brain, a
            phantom's own. The metabolite shifts are typed at 37 C and pH 7 and move by
            their published functions of both (database.shifts, Wermter et al. 2026).
            The temperature is the default of the leaf water.temperature_c, and that
            leaf is the tissue's temperature of every draw: a range there (fever,
            hypothermia, a phantom's room) moves the water line and the metabolites of
            each draw together.
        temperature_spread: True draws a brain's temperature about `temperature_c` with
            the spread between healthy adults that MRS thermometry measures (0.4 C sd,
            tissue.RZECHORZEK_2022); False (the default) keeps it fixed.
        receiver: A hardware.Receiver, the digital receiver every FID passes through
            (oversampled, filtered, decimated: the first points carry the filter's
            onset, the noise its band edges). None samples the ideal FID.
        isochromats: Ensemble size for gradient-bearing sequences, handed to the
            engine; raise it for dense multi-axis gradient schemes.
        records: The scanner's own Calibrations (a scan.profile ScannerProfile hands them
            over); they outrank the package's records in the priors and the header.
        subject: The person scanned, as a scanner registers them: {"age_y": years, "sex":
            "F" or "M", "health": "healthy" or the condition by name}; the measured rows they
            carry (the water's relaxation, breathing, motion) are pooled over people like
            them (scan.cohort.pooled). None is a healthy adult.
        within: The population the pooled readings come from: cohort names and (cohort,
            subject) pairs (scan.cohort pooled); empty is every row the conditions keep.
        engine: An Engine to share (its compiled graphs and renderers) between scans
            of one scanner, as a Multiplex does; None builds one for this scan.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequence, params=None, database=None, water=None,
                 lipids=None, coils=None, field=None, transmit=None, mm=None, downfield=None,
                 girf=None, receiver=None, isochromats=32, engine=None, temperature_c=None,
                 ph=None, field_groups=1, water_groups=1, noise_map=None,
                 noise_reference_mm=(0.0, -45.0, 35.0), records=(), subject=None, within=(),
                 spins=None, temperature_spread=False):
        self.brain = brain
        self.scanner = scanner
        self.sequence = sequence
        self.database = database if database is not None else Database()
        self.spins = spins
        self.engine = engine if engine is not None else Engine(scanner, isochromats=isochromats)
        self.water = water if water is not None else Suppressed()
        in_vitro = isinstance(brain, Phantom)
        self.lipids = lipids if lipids is not None else NoLipids() if in_vitro else Skull()
        self.mm = (mm if mm is not None else NoMM() if in_vitro
                   else CoupledMM() if hasattr(sequence, "edit_ppm") else MeasuredMM())
        self.downfield = (downfield if downfield is not None
                          else NoDownfield() if in_vitro else ParameterizedDownfield())
        self.coils = coils
        self.field = field
        self.field_groups = int(field_groups)
        self.water_groups = int(water_groups)
        self.noise_map = noise_map
        self.noise_reference_mm = tuple(float(x) for x in noise_reference_mm)
        self.transmit = transmit
        self.girf = girf
        self.receiver = receiver
        self.records = tuple(records)
        self.subject = subject
        self.within = tuple(within)
        self.temperature_c = float(temperature_c if temperature_c is not None
                                   else brain.temperature_c if in_vitro else 37.0)
        self.ph = float(ph if ph is not None else brain.ph if in_vitro else 7.0)
        defaults = default_params(self.database, b0_t=self.engine.b0_t, scanner=scanner,
                                  records=self.records,
                                  channels=None if coils is None else len(coils),
                                  subject=subject, within=self.within)
        self.params = (defaults
                       | (brain.params(defaults) if in_vitro else Params())
                       | Params(water=dict(temperature_c=Normal(
                           self.temperature_c, tissue.RZECHORZEK_2022["healthy_sd_c"])
                           if temperature_spread and not in_vitro else self.temperature_c))
                       | (params if params is not None else Params()))
        # a metabolite added through params needs its relaxation, contrast and ADC leaves too
        for path, node in (("t1", self.params.t1), ("t2", self.params.t2),
                           ("contrast", self.params.contrast),
                           ("diffusion.adc", self.params.diffusion.adc)):
            missing = [name for name in self.params.conc if name not in node]
            if missing:
                raise ValueError(f"{', '.join(missing)} in conc but not in {path}: a metabolite "
                                 f"needs its t1, t2, contrast and diffusion.adc leaves")
        self._lines = {}  # each truth's rendered FIDs (Voxel.prepare)
        # the metabolite lines by resolved frame, shared by every draw at it
        self._frame_lines = {}
        # the pulses of the latest transmit frame under a played design (Voxel._framed_lines)
        self._propagators = {}
        # magnetization transfer: the design's pulses' responses (Voxel.transfer)
        self._transfer = {}
        self._basis = None
        self._eddy = None
        self._systems = {}
        self._b_value = None

    #********************#
    #   spin systems     #
    #********************#
    def systems(self, metabolite, temperature_c=None):
        """The spin systems of a metabolite from this scan's database, from the table `spins` names
        for it, built once and cached: every draw and every voxel reuses the same structural
        systems. A draw's own tissue temperature (the water.temperature_c leaf) gives the systems of
        that temperature, in steps of a quarter of a degree (the shifts move by 0.0001 to 0.001 ppm
        a degree; database.shifts), each step built once; a compiled scan compiles each step it
        meets."""
        temperature_c = round(4.0 * float(self.temperature_c if temperature_c is None
                                          else temperature_c)) / 4.0
        key = (metabolite, temperature_c)
        if key not in self._systems:
            source = (self.spins if isinstance(self.spins, str)
                      else (self.spins or {}).get(metabolite, "newest"))
            self._systems[key] = SpinSystem.from_database(
                metabolite, database=self.database, source=source, temperature_c=temperature_c,
                ph=self.ph)
        return self._systems[key]

    #*************************#
    #   diffusion weighting   #
    #*************************#
    @property
    def b_value_ms_um2(self):
        """The b-value the sequence's own gradient events build up, cached; see
        sequences.diffusion for the consensus formulation."""
        if self._b_value is None:
            from mrsandbox.sequences.diffusion import b_value_ms_um2

            self._b_value = b_value_ms_um2(self.sequence)
        return self._b_value

    #************************#
    #   default parameters   #
    #************************#
    default_params = staticmethod(default_params)

    #***********#
    #   voxel   #
    #***********#
    def voxel(self, position_mm, size_mm, compiled=False, box_center_mm=None, box_axes="xyz",
              shim_at_mm=None, turn=None):
        """Place a voxel (a size_mm cube, or an (x, y, z) box, turned by `turn`, its rotation
        from its own frame to the world's: phantom.turning) and get its acquisition handle.

        With `compiled`, metabolite lines run through jitted per-system functions cached
        on the scan: the first draw compiles, every further draw replays. Worth it for
        many draws from one scan (the training mode turns it on); a one-shot look is
        faster plain. `box_center_mm` makes the voxel a cell of the box a localized
        design selects about that centre (an MRSI lattice places its cells so);
        `shim_at_mm` keeps the shim fitted for that place (a head that moved since).
        """
        return Voxel(self, position_mm, size_mm, compiled=compiled, box_center_mm=box_center_mm,
                     box_axes=box_axes, shim_at_mm=shim_at_mm, turn=turn)

    #***********#
    #   mrsi    #
    #***********#
    def mrsi(self, center_mm=(0.0, 0.0, 0.0), matrix=(16, 16), voxel_mm=10.0,
             thickness_mm=None, oversampling=1, weighting=None, sampling=None,
             profile=None, averages=None, trajectory=None, trajectory_params=None,
             dead_time_s=0.0, odd_sweeps=None, box_center_mm=None):
        """Place an MRSI slice lattice and get its acquisition handle; see scan.mrsi.MRSI."""
        from mrsandbox.scan.mrsi import MRSI

        return MRSI(self, center_mm=center_mm, matrix=matrix, voxel_mm=voxel_mm,
                    thickness_mm=thickness_mm, oversampling=oversampling,
                    weighting=weighting, sampling=sampling, profile=profile,
                    averages=averages, trajectory=trajectory,
                    trajectory_params=trajectory_params, dead_time_s=dead_time_s,
                    odd_sweeps=odd_sweeps, box_center_mm=box_center_mm)

    #*****************#
    #   training      #
    #*****************#
    def generator(self, position_mm, size_mm, batch_size=16, seed=None, npoints=2048,
                  dwell_s=1.0 / 4000.0):
        """A seeded on-the-fly training stream for this voxel; see scan.generator.Generator."""
        from mrsandbox.scan.generator import Generator

        return Generator(self.voxel(position_mm, size_mm, compiled=True), batch_size=batch_size,
                         seed=seed, npoints=npoints, dwell_s=dwell_s)
