####################################################################################################
#                                           transfer.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Magnetization transfer: what a sequence's saturation pulses take from the metabolites   #
#          and the water through the tissue's bound (semi-solid) pool, its longitudinal water,     #
#          bound and metabolite magnetizations played through the design's own pulses and delays.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from keras import ops

from mrsandbox.sequences.events import (CoherenceFilter, Delay, Pulse, SelectivePulse, Sequence,
                                        ShapedPulse, _hashable)


#**********************#
#   superlorentzian    #
#**********************#
def superlorentzian(delta_hz, t2_s, floor_hz=1000.0):
    """The bound pool's absorption line G(delta) in s (Morrison and Henkelman, MRM 33 (1995)
    475), per offset in `delta_hz`: the integral over u from 0 to 1 of sqrt(2 / pi) T2 / |3u^2 -
    1| exp(-2 (2 pi delta T2 / (3u^2 - 1))^2). It diverges as delta goes to 0, so below
    `floor_hz` it is held at its value there, the convention of quantitative MT (stated)."""
    delta = np.maximum(np.abs(np.asarray(delta_hz, dtype=np.float64)), floor_hz)[..., None]
    u = (np.arange(4000) + 0.5) / 4000.0
    spread = np.abs(3.0 * u * u - 1.0)
    return np.mean(np.sqrt(2.0 / np.pi) * t2_s / spread
                   * np.exp(-2.0 * (2.0 * np.pi * delta * t2_s / spread) ** 2), axis=-1)


#**************************************************************************************************#
#                                          Class Transfer                                          #
#**************************************************************************************************#
#                                                                                                  #
# The longitudinal water, bound pool and metabolites through a design, in steady state over TR.   #
#                                                                                                  #
#**************************************************************************************************#
class Transfer:
    """Magnetization transfer between the free water, the tissue's bound (semi-solid) pool and
    the metabolites, through a design's pulses and delays in steady state over its TR.

    The magnetizations, each over its own equilibrium, follow
        dw/dt = R1w (1 - w) + k_ws (s - w)
        ds/dt = R1s (1 - s) + k_sw (w - s) - W(t) s
        dm/dt = R1m (1 - m) + k_m (s - m)
    with k_ws = f k_sw for a bound pool of f of the water's size: water and bound pool exchange,
    and a metabolite's methyl and methylene protons relax toward the bound pool at their own rate
    (cross-relaxation, the mechanism of the upfield signal loss; Kruiskamp et al., NMR Biomed 14
    (2001) 1; Hoefemann et al., MRM 85 (2021) 1766). Every shaped pulse saturates the bound pool
    at W = pi w1(t)^2 G(delta(t)), its own power at its offset from the water at the voxel's
    centre (`superlorentzian`); ideal pulses carry no power. The water takes every pulse as the
    engine plays it on a water singlet at the voxel's centre; a metabolite takes the
    localization's pulses as the water does (both lie in their bands) and none of the saturation
    pulses, whose direct effect on the metabolites the engine's own play of the design holds.

    A saturation pulse is one whose transverse magnetization a filter on order zero alone spoils
    right after: the water suppression trains' pulses and the outer-volume slabs. A metabolite's
    factor is its magnetization at the excitation with them over without them (each replaced by a
    delay of its length), what they take through the bound pool; the water's is its own with the
    bound pool's saturation over without it, and only where no saturation pulse hits the water
    itself (a suppressed water's residual is the prescan's to set).

    Args:
        pool_fraction: f, the bound pool's size over the free water's.
        exchange_hz: k_sw, the bound pool's exchange rate toward the water, 1/s.
        t2_s: The bound pool's T2.
        r1_s: The bound pool's longitudinal rate, 1/s.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, pool_fraction, exchange_hz, t2_s=9e-6, r1_s=1.0):
        self.pool_fraction = float(pool_fraction)
        self.exchange_hz = float(exchange_hz)
        self.t2_s = float(t2_s)
        self.r1_s = float(r1_s)

    #*************#
    #   factors   #
    #*************#
    def factors(self, engine, sequence, water, water_t1_s, rates, t1_s, b1_scale=1.0, kept=None):
        """{metabolite: factor} for `rates` ({name: k_m in 1/s}) with their T1s (`t1_s`), and
        "water": the water's factor; `water` is the water's spin system, `engine` plays it.
        `kept`, a dict, holds the pulses' responses between calls on one design."""
        kept = {} if kept is None else kept
        events = list(sequence.events)
        saturating = {k for k, event in enumerate(events[:-1])
                      if isinstance(event, (Pulse, SelectivePulse, ShapedPulse))
                      and isinstance(events[k + 1], CoherenceFilter)
                      and events[k + 1].orders == (0,)}
        pulses = {}
        for k, event in enumerate(events):
            if isinstance(event, (Pulse, SelectivePulse, ShapedPulse)):
                # the pulse itself (a design induced anew per draw, scan.eddy, plays the same
                # pulses; the induced field is hertz against a pulse's kilohertz, stated)
                key = (event.structure(), tuple(_hashable(getattr(event, name))
                                                for name in event.free if name != "induced"),
                       b1_scale, self.t2_s)
                if key not in kept:
                    kept[key] = self._pulse(engine, event, water, b1_scale)
                pulses[k] = kept[key]
        names = list(rates)
        rate = np.array([float(rates[name]) for name in names])
        r1 = np.array([1.0 / float(t1_s[name]) for name in names])
        first = next((k for k in sorted(pulses) if k not in saturating), None)
        if first is None:
            return {**{name: 1.0 for name in names}, "water": 1.0}
        recovery = (None if sequence.tr_s is None
                    else max(float(sequence.tr_s) - float(sequence.duration_s), 0.0))

        def excited(plays, saturates):
            """The magnetizations (w, s, m...) just before the excitation, the first pulse that is
            no saturation: in steady state over the TR, or from equilibrium without one."""
            def through(steps, scale, shift):
                for k in steps:
                    if k in pulses:
                        water_left, bound_left, length = pulses[k]
                        keep = np.ones(2 + len(names))
                        if k not in saturating or plays:
                            keep[0] = water_left
                            keep[1] = bound_left if saturates else 1.0
                        if k not in saturating:
                            keep[2:] = water_left
                        scale, shift = keep[:, None] * scale, keep * shift
                        scale, shift = self._relax(length, water_t1_s, rate, r1, scale, shift)
                    elif isinstance(events[k], Delay):
                        scale, shift = self._relax(float(events[k].t_s), water_t1_s, rate, r1,
                                                   scale, shift)
                return scale, shift

            n = 2 + len(names)
            if recovery is None:
                scale, shift = through(range(first), np.eye(n), np.zeros(n))
                return scale @ np.ones(n) + shift
            scale, shift = through(range(first, len(events)), np.eye(n), np.zeros(n))
            scale, shift = self._relax(recovery, water_t1_s, rate, r1, scale, shift)
            scale, shift = through(range(first), scale, shift)
            # the steady state: the state the whole cycle maps onto itself
            return np.linalg.solve(np.eye(n) - scale, shift)

        with_all, without = excited(True, True), excited(False, True)
        found = {name: float(with_all[2 + i] / without[2 + i]) for i, name in enumerate(names)}
        hits_water = any(pulses[k][0] < 0.9 for k in saturating)
        found["water"] = 1.0 if hits_water else float(with_all[0] / excited(True, False)[0])
        return found

    #***********#
    #   relax   #
    #***********#
    def _relax(self, t_s, water_t1_s, rate, r1, scale, shift):
        """The affine map (scale, shift) followed by `t_s` of relaxation and exchange: x -> e^{At}
        x + (e^{At} - 1) x_eq, exact for the linear equations about their equilibrium (all ones)."""
        if t_s <= 0.0:
            return scale, shift
        n = 2 + len(rate)
        k_ws = self.pool_fraction * self.exchange_hz
        a = np.zeros((n, n))
        a[0, 0], a[0, 1] = -1.0 / water_t1_s - k_ws, k_ws
        a[1, 0], a[1, 1] = self.exchange_hz, -self.r1_s - self.exchange_hz
        for i in range(len(rate)):
            a[2 + i, 2 + i], a[2 + i, 1] = -r1[i] - rate[i], rate[i]
        values, vectors = np.linalg.eig(a)
        step = np.real(vectors @ np.diag(np.exp(values * t_s)) @ np.linalg.inv(vectors))
        return step @ scale, step @ shift + (np.eye(n) - step) @ np.ones(n)

    #***********#
    #   pulse   #
    #***********#
    def _pulse(self, engine, event, water, b1_scale):
        """(the water's longitudinal share after the pulse at the voxel's centre, the bound
        pool's, the pulse's length)."""
        read = Pulse(90.0)
        _, alone = engine.lines(Sequence(read), water, b1_scale=b1_scale)
        played = Sequence(event, CoherenceFilter(0), read)
        if isinstance(event, ShapedPulse) and event.gradient_hz_per_mm:
            _, after = engine.profile(played, water, event.gradient_axis, [0.0], b1_scale)
            after = after[0]
        else:
            _, after = engine.lines(played, water, b1_scale=b1_scale)
        alone, after = (np.asarray(ops.convert_to_numpy(v)) for v in (alone, after))
        line = np.argmax(np.abs(alone))
        water_left = float(np.real(after[line] / alone[line]))
        if not isinstance(event, ShapedPulse):
            return water_left, 1.0, 0.0
        envelope = np.asarray(event.envelope, dtype=np.float64)
        dt = float(event.duration_s) / envelope.size
        hz = (envelope / np.abs(envelope).max() * float(event.peak_hz) if event.peak_hz is not None
              else envelope * (float(event.flip_deg) / 360.0) / (envelope.sum() * dt)) * b1_scale
        f0_mhz = float(water.gamma_mhz_t) * engine.b0_t
        offset = (float(event.offset_ppm) - 4.65) * f0_mhz
        sweep = (np.zeros(envelope.size) if event.frequency_hz is None
                 else np.asarray(event.frequency_hz, dtype=np.float64))
        absorbed = np.sum(np.pi * (2.0 * np.pi * hz) ** 2
                          * superlorentzian(offset + sweep, self.t2_s) * dt)
        return water_left, float(np.exp(-absorbed)), float(event.duration_s)
