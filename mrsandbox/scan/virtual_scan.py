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
from mrsandbox.database import Database
from mrsandbox.engine import Engine
from mrsandbox.params import Params
from mrsandbox.phantom.downfield import ParameterizedDownfield
from mrsandbox.phantom.lipids import Skull
from mrsandbox.phantom.macromolecules import CoupledMM
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
        brain: A phantom.Brain.
        scanner: A hardware.Scanner, or a field strength in tesla.
        sequence: A sequences.Sequence.
        params: Params overriding the defaults, composed as defaults | params.
        database: Database instance; defaults to the packaged one.
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
        transmit: A phantom.Volume of the relative transmit field B1+ (1.0 nominal);
            every voxel runs its pulses at the drawn b1_scale times its local value.
            None gives the nominal field everywhere.
        mm: A phantom.MMSource; defaults to the LCModel-convention parameterization.
        downfield: A phantom.DownfieldSource; defaults to the nine-peak
            parameterization, presaturated along with the default suppressed water.
        girf: A resources.GIRF, the scanner's measured gradient impulse response;
            every readout then carries the fields the design's own gradients induce
            through it (scan.eddy) in place of the drawn exponential terms of the
            gradients leaves. None draws those terms.
        temperature_c, ph: The tissue's temperature and pH. The metabolite shifts are
            typed at 37 C and pH 7 and move by their published functions of both
            (database.shifts, Wermter et al. 2026); the default of the water leaf
            water.temperature_c follows the temperature.
        receiver: A hardware.Receiver, the digital receiver every FID passes through
            (oversampled, filtered, decimated: the first points carry the filter's
            onset, the noise its band edges). None samples the ideal FID.
        isochromats: Ensemble size for gradient-bearing sequences, handed to the
            engine; raise it for dense multi-axis gradient schemes.
        records: The scanner's own Calibrations (a scan.profile ScannerProfile hands them
            over); they outrank the package's records in the priors and the header.
        engine: An Engine to share (its compiled graphs and renderers) between scans
            of one scanner, as a Multiplex does; None builds one for this scan.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, brain, scanner, sequence, params=None, database=None, water=None,
                 lipids=None, coils=None, field=None, transmit=None, mm=None, downfield=None,
                 girf=None, receiver=None, isochromats=32, engine=None, temperature_c=37.0,
                 ph=7.0, field_groups=1, noise_map=None, noise_reference_mm=(0.0, -45.0, 35.0),
                 records=()):
        self.brain = brain
        self.scanner = scanner
        self.sequence = sequence
        self.database = database if database is not None else Database()
        self.engine = engine if engine is not None else Engine(scanner, isochromats=isochromats)
        self.water = water if water is not None else Suppressed()
        self.lipids = lipids if lipids is not None else Skull()
        self.mm = mm if mm is not None else CoupledMM()
        self.downfield = downfield if downfield is not None else ParameterizedDownfield()
        self.coils = coils
        self.field = field
        self.field_groups = int(field_groups)
        self.noise_map = noise_map
        self.noise_reference_mm = tuple(float(x) for x in noise_reference_mm)
        self.transmit = transmit
        self.girf = girf
        self.receiver = receiver
        self.records = tuple(records)
        self.temperature_c = float(temperature_c)
        self.ph = float(ph)
        self.params = (default_params(self.database, b0_t=self.engine.b0_t, scanner=scanner,
                                      records=self.records)
                       | Params(water=dict(temperature_c=self.temperature_c))
                       | (params if params is not None else Params()))
        self._lines = None
        self._basis = None
        self._eddy = None
        self._systems = {}
        self._b_value = None

    #********************#
    #   spin systems     #
    #********************#
    def systems(self, metabolite):
        """The spin systems of a metabolite from this scan's database, built once and
        cached: every draw and every voxel reuses the same structural systems."""
        if metabolite not in self._systems:
            self._systems[metabolite] = SpinSystem.from_database(
                metabolite, database=self.database, temperature_c=self.temperature_c, ph=self.ph)
        return self._systems[metabolite]

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
              shim_at_mm=None):
        """Place a voxel (a size_mm cube, or an (x, y, z) box) and get its acquisition
        handle.

        With `compiled`, metabolite lines run through jitted per-system functions cached
        on the scan: the first draw compiles, every further draw replays. Worth it for
        many draws from one scan (the training mode turns it on); a one-shot look is
        faster plain. `box_center_mm` makes the voxel a cell of the box a localized
        design selects about that centre (an MRSI lattice places its cells so);
        `shim_at_mm` keeps the shim fitted for that place (a head that moved since).
        """
        return Voxel(self, position_mm, size_mm, compiled=compiled, box_center_mm=box_center_mm,
                     box_axes=box_axes, shim_at_mm=shim_at_mm)

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
    def generator(self, position_mm, size_mm, batch_size=16, seed=None, npoints=2048, dwell_s=1.0 / 4000.0):
        """A seeded on-the-fly training stream for this voxel; see scan.generator.Generator."""
        from mrsandbox.scan.generator import Generator

        return Generator(self.voxel(position_mm, size_mm, compiled=True), batch_size=batch_size,
                         seed=seed, npoints=npoints, dwell_s=dwell_s)
