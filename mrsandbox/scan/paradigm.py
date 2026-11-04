####################################################################################################
#                                            paradigm.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/11/26                                                                                #
#                                                                                                  #
# Purpose: When the task is on over a functional run, and how the levels follow it: events of     #
#          onsets and durations (a block design is a few long events), the task as a function of  #
#          time, and the response the metabolites' courses ride, a first-order rise and decay.    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses

import numpy as np


#**************************************************************************************************#
#                                          Class Paradigm                                          #
#**************************************************************************************************#
#                                                                                                  #
# When the task is on over a run: events of onsets and durations in seconds.                       #
#                                                                                                  #
#**************************************************************************************************#
@dataclasses.dataclass(frozen=True)
class Paradigm:
    """When the task is on over a run: events of onsets and durations in seconds; a block
    design is a few long events (`blocks`). The levels follow the task through `response`,
    which Multiplex.transients plays and scan.calibrate TaskReading reads."""

    onsets_s: tuple
    durations_s: tuple

    #************#
    #   blocks   #
    #************#
    @classmethod
    def blocks(cls, rest_s, task_s, cycles=1, lead_s=None):
        """`cycles` blocks of `task_s` seconds, each after `rest_s` of rest, the first after
        `lead_s` (the rest itself when None)."""
        lead = rest_s if lead_s is None else lead_s
        return cls(onsets_s=tuple(float(lead + k * (rest_s + task_s)) for k in range(cycles)),
                   durations_s=(float(task_s),) * cycles)

    #**********#
    #   task   #
    #**********#
    def task(self, times_s):
        """One where the task is on at each of `times_s`, zero elsewhere."""
        t = np.asarray(times_s, dtype=np.float64)
        on = np.zeros(t.shape)
        for onset, duration in zip(self.onsets_s, self.durations_s):
            on = np.maximum(on, ((t >= onset) & (t < onset + duration)).astype(np.float64))
        return on

    #**************#
    #   response   #
    #**************#
    def response(self, times_s, rise_s, decay_s):
        """How far the levels have followed the task at each of `times_s` (ascending), zero
        to one: a first-order course, rising toward one with `rise_s` while the task is on
        and falling toward zero with `decay_s` while it is off, from zero at the start."""
        t = np.asarray(times_s, dtype=np.float64)
        on, course = self.task(t), np.zeros(t.shape)
        for k in range(1, t.size):
            tau = rise_s if on[k - 1] > 0.0 else decay_s
            course[k] = on[k - 1] + (course[k - 1] - on[k - 1]) * np.exp(-(t[k] - t[k - 1]) / tau)
        return course

    def to_dict(self):
        return {"onsets_s": list(self.onsets_s), "durations_s": list(self.durations_s)}
