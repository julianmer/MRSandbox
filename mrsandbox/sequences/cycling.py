####################################################################################################
#                                            cycling.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: Phase cycling as physics: a design run step by step with its pulse and receiver phases  #
#          following a table and the steps averaged, so the pathways a scheme selects are what     #
#          survive the sum, and a per-step phase error leaks what the ideal cycle cancels.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
from keras import ops
from nifti_mrs_plus import ops as nops

from mrsandbox.sequences.events import Pulse, SelectivePulse, Sequence, ShapedPulse, _hashable


#**************************************************************************************************#
#                                           Class Cycled                                           #
#**************************************************************************************************#
#                                                                                                  #
# A design run as an explicit phase cycle, the steps averaged.                                     #
#                                                                                                  #
#**************************************************************************************************#
class Cycled(Sequence):
    """A design run as an explicit phase cycle, the steps averaged.

    The events of `sequence` run once per row of `table`, with the phase of its k-th
    pulse (hard, selective, and shaped pulses in order) raised by the row's k-th entry
    and the receiver phase set to the row of `receiver_deg`, and the density matrices
    of the steps are averaged: exactly what the scanner sums, so the coherence
    pathways the scheme selects are what survive, and the ones it cancels leak by
    the imbalance of the steps. `jitter_deg`, a (steps, pulses) array of per-step
    phase errors (drawn instability), is a timing of the design, so an instability
    replays through the compiled graph. The receiver phase rotates each step's
    density matrix about z by its coherence order, the same as phasing the detected
    signal. With a TR the steady state is reached per step at that step's phases
    (a stated approximation of the cycle's own steady state).

    Args:
        sequence: The design (its CoherenceFilters kept if any).
        table: (steps, pulses) pulse phase increments in degrees.
        receiver_deg: (steps,) receiver phases in degrees.
        jitter_deg: (steps, pulses) per-step phase errors in degrees; None for none.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, sequence, table, receiver_deg, jitter_deg=None):
        super().__init__(*sequence.events, tr_s=sequence.tr_s, cycles=sequence.cycles)
        self.table = np.asarray(table, dtype=np.float32)
        self.receiver_deg = np.asarray(receiver_deg, dtype=np.float32)
        pulses = [k for k, event in enumerate(self.events)
                  if isinstance(event, (Pulse, SelectivePulse, ShapedPulse))]
        if self.table.ndim != 2 or self.table.shape[1] != len(pulses) \
                or self.receiver_deg.shape != (self.table.shape[0],):
            raise ValueError(f"the table needs one row per step and one column per pulse "
                             f"({len(pulses)} pulses), the receiver one phase per step")
        self.pulses = tuple(pulses)
        self.jitter_deg = jitter_deg

    #*********#
    #   run   #
    #*********#
    def run(self, rho, ctx):
        """The average over the steps of the design run at each step's phases, each
        step's result rotated by its receiver phase."""
        total = None
        jitter = None if self.jitter_deg is None else ops.convert_to_tensor(self.jitter_deg, dtype="float32")
        for step in range(self.table.shape[0]):
            events = list(self.events)
            for column, k in enumerate(self.pulses):
                event = self.events[k]
                phase = event.phase_deg + float(self.table[step, column])
                if jitter is not None:
                    phase = phase + jitter[step, column]
                events[k] = event.bound(tuple(phase if name == "phase_deg" else getattr(event, name)
                                              for name in event.free
                                              if getattr(event, name) is not None))
            stepped = Sequence(*events, tr_s=self.tr_s, cycles=self.cycles).run(rho, ctx)
            angle = (-math.radians(float(self.receiver_deg[step]))) * ops.cast(ctx.q, "float32")
            stepped = stepped * ops.exp(nops.complex_from(ops.zeros_like(angle), angle))
            total = stepped if total is None else total + stepped
        return total / float(self.table.shape[0])

    #***********************#
    #   structure, timings  #
    #***********************#
    def structure(self):
        return (super().structure(), _hashable(self.table), _hashable(self.receiver_deg),
                self.jitter_deg is None)

    def timings(self):
        return super().timings() + (((self.jitter_deg,),) if self.jitter_deg is not None else ())

    def bound(self, timings):
        base = super().bound(timings[:-1] if self.jitter_deg is not None else timings)
        jitter = timings[-1][0] if self.jitter_deg is not None else None
        return Cycled(base, self.table, self.receiver_deg, jitter)
