####################################################################################################
#                                            library.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/09/26                                                                                #
#                                                                                                  #
# Purpose: Named, citable sequence designs built from the event primitives. Coherence-filter       #
#          patterns follow the community convention (detection lives on order -1): PulseAcquire    #
#          keeps -1; PRESS selects -1, +1, -1; STEAM selects -1, 0, -1.                            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from mrsandbox.database import suppression
from mrsandbox.sequences.events import (CoherenceFilter, Delay, Pulse, SelectivePulse,
                                        Sequence, ShapedPulse)


#**************************************************************************************************#
#                                        Class PulseAcquire                                        #
#**************************************************************************************************#
#                                                                                                  #
# Hard 90x followed immediately by acquisition.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class PulseAcquire(Sequence):
    """Hard 90x followed immediately by acquisition."""

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        super().__init__(Pulse(90.0), CoherenceFilter(-1))


#**************************************************************************************************#
#                                           Class Press                                            #
#**************************************************************************************************#
#                                                                                                  #
# Ideal PRESS: 90x - te1/2 - 180y - (te1+te2)/2 - 180y - te2/2 - acquire.                          #
#                                                                                                  #
#**************************************************************************************************#
class Press(Sequence):
    """Ideal PRESS: 90x - te1/2 - 180y - (te1+te2)/2 - 180y - te2/2 - acquire.

    Args:
        te1_s: First echo time in seconds (may be a traced tensor).
        te2_s: Second echo time in seconds (may be a traced tensor).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te1_s, te2_s):
        self.te1_s = te1_s
        self.te2_s = te2_s
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(te1_s / 2.0),
            Pulse(180.0, phase_deg=90.0),
            CoherenceFilter(+1),
            Delay((te1_s + te2_s) / 2.0),
            Pulse(180.0, phase_deg=90.0),
            CoherenceFilter(-1),
            Delay(te2_s / 2.0),
        )


#**************************************************************************************************#
#                                           Class Steam                                            #
#**************************************************************************************************#
#                                                                                                  #
# Ideal STEAM: 90x - te/2 - 90x - tm - 90x - te/2 - acquire, stimulated-echo pathway only.         #
#                                                                                                  #
#**************************************************************************************************#
class Steam(Sequence):
    """Ideal STEAM: 90x - te/2 - 90x - tm - 90x - te/2 - acquire, stimulated-echo pathway only.

    Args:
        te_s: Echo time in seconds.
        tm_s: Mixing time in seconds.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, tm_s):
        self.te_s = te_s
        self.tm_s = tm_s
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(te_s / 2.0),
            Pulse(90.0),
            CoherenceFilter(0),
            Delay(tm_s),
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(te_s / 2.0),
        )


#**************************************************************************************************#
#                                     Class SuppressionTrain                                       #
#**************************************************************************************************#
#                                                                                                  #
# A generic water suppression module: frequency-selective pulses, each followed by a crusher.      #
#                                                                                                  #
#**************************************************************************************************#
class SuppressionTrain(Sequence):
    """A generic water suppression module: frequency-selective pulses, each followed by a
    crusher (kept longitudinal magnetization only), with optional interpulse delays.

    This is the machinery behind every named scheme. CHESS is a train of equal 90s; Wet
    and Vapor below are trains with their published flip and delay series, typed from
    the sources in database.suppression.

    Args:
        flips_deg: One flip angle per suppression pulse.
        offset_ppm: Pulse centre in ppm (water sits at 4.65).
        bandwidth_hz: Full selective bandwidth in Hz.
        delays_s: Optional delay after each pulse (same length as flips_deg).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flips_deg, offset_ppm=4.65, bandwidth_hz=60.0, delays_s=None):
        self.flips_deg = tuple(flips_deg)
        self.offset_ppm = offset_ppm
        self.bandwidth_hz = bandwidth_hz
        self.delays_s = tuple(delays_s) if delays_s is not None else (0.0,) * len(self.flips_deg)
        events = []
        for flip, delay in zip(self.flips_deg, self.delays_s):
            events.append(SelectivePulse(flip, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz))
            events.append(CoherenceFilter(0))
            if delay:
                events.append(Delay(delay))
        super().__init__(*events)


#**************************************************************************************************#
#                                           Class Chess                                            #
#**************************************************************************************************#
#                                                                                                  #
# CHESS water suppression: repeated selective excitation and crushing (Haase et al. 1985).         #
#                                                                                                  #
#**************************************************************************************************#
class Chess(SuppressionTrain):
    """CHESS water suppression: repeated selective excitation and crushing (Haase et al.
    1985). Each cycle leaves cos(flip) of the longitudinal water, so n ideal 90s
    suppress completely and a flip error leaves cos(flip)**n as the residual.

    Args:
        n: Number of suppression cycles.
        flip_deg: Flip angle per cycle.
        offset_ppm: Pulse centre in ppm.
        bandwidth_hz: Full selective bandwidth in Hz.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, n=3, flip_deg=90.0, offset_ppm=4.65, bandwidth_hz=60.0):
        self.n = n
        self.flip_deg = flip_deg
        super().__init__([flip_deg] * n, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz)


#**************************************************************************************************#
#                                            Class Wet                                             #
#**************************************************************************************************#
#                                                                                                  #
# WET water suppression: the numerically optimized flip series of Ogg et al. 1994.                 #
#                                                                                                  #
#**************************************************************************************************#
class Wet(SuppressionTrain):
    """WET water suppression: the numerically optimized flip series of Ogg et al. 1994
    (81.4, 101.4, 69.3, 161.0 degrees), each pulse followed by a crusher. The series
    nulls the longitudinal water insensitively to T1 and B1; the paper optimizes flips
    for the sequence timing at hand, so the interpulse delay stays a free timing here.

    Args:
        delay_s: Delay after each pulse in seconds.
        offset_ppm: Pulse centre in ppm (water sits at 4.65).
        bandwidth_hz: Full selective bandwidth in Hz.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, delay_s=0.060, offset_ppm=4.65, bandwidth_hz=60.0):
        self.delay_s = delay_s
        flips = suppression.WET["flips_deg"]
        super().__init__(flips, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz,
                         delays_s=(delay_s,) * len(flips))


#**************************************************************************************************#
#                                           Class Vapor                                            #
#**************************************************************************************************#
#                                                                                                  #
# VAPOR water suppression: variable pulse power and optimized relaxation delays (Tkac 1999).       #
#                                                                                                  #
#**************************************************************************************************#
class Vapor(SuppressionTrain):
    """VAPOR water suppression: variable pulse power and optimized relaxation delays
    (Tkac et al. 1999). Built from the complete eight pulse table of Tkac and Gruetter
    2005: power offsets in dB from the nominal pulse power with per-field delay series.
    A dB offset scales the flip of a fixed-shape pulse by 10 ** (db / 20); the scheme
    is designed to null the water over a wide band of nominal flips around 90 degrees.

    Args:
        field: Delay series to use, "4t" or "7t".
        nominal_flip_deg: Flip of the nominal-power pulses (models a B1 miscalibration).
        offset_ppm: Pulse centre in ppm.
        bandwidth_hz: Full selective bandwidth in Hz.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, field="4t", nominal_flip_deg=90.0, offset_ppm=4.65, bandwidth_hz=60.0):
        self.field = field
        self.nominal_flip_deg = nominal_flip_deg
        flips = tuple(nominal_flip_deg * 10.0 ** (db / 20.0) for db in suppression.VAPOR["power_db"])
        super().__init__(flips, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz,
                         delays_s=suppression.VAPOR["delays_s"][field])


#**************************************************************************************************#
#                                         Class MegaPress                                          #
#**************************************************************************************************#
#                                                                                                  #
# MEGA-PRESS J-difference editing: PRESS with two frequency-selective editing pulses.              #
#                                                                                                  #
#**************************************************************************************************#
class MegaPress(Sequence):
    """MEGA-PRESS J-difference editing (Mescher et al. 1998): ideal PRESS with two
    frequency-selective 180s that refocus the J-evolution of spins inside the editing
    band. The canonical timing is typed from the FID-A simulation (Simpson et al. 2017,
    run_simMegaPressShaped): 90 - 5 - 180 - 17 - edit - 17 - 180 - 17 - edit - 12 ms,
    TE 68 ms, edit ON at 1.88 ppm and OFF at 7.5 ppm. The difference ON minus OFF
    isolates the coupled partners of the edited spins (GABA at 3.01 ppm being the
    classic target); uncoupled singlets subtract away.

    Args:
        edit_ppm: Editing pulse centre (1.88 for ON, 7.5 for OFF).
        taus_s: The five inter-pulse delays in seconds; their sum is the echo time.
        bandwidth_hz: Full selective bandwidth of the editing pulses.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm=1.88, taus_s=(0.005, 0.017, 0.017, 0.017, 0.012), bandwidth_hz=60.0):
        self.edit_ppm = edit_ppm
        self.taus_s = tuple(taus_s)
        self.bandwidth_hz = bandwidth_hz
        t1, t2, t3, t4, t5 = self.taus_s
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(t1),
            Pulse(180.0, phase_deg=90.0),
            CoherenceFilter(+1),
            Delay(t2),
            SelectivePulse(180.0, offset_ppm=edit_ppm, bandwidth_hz=bandwidth_hz),
            Delay(t3),
            Pulse(180.0, phase_deg=90.0),
            CoherenceFilter(-1),
            Delay(t4),
            SelectivePulse(180.0, offset_ppm=edit_ppm, bandwidth_hz=bandwidth_hz),
            Delay(t5),
        )

    #**********#
    #   pair   #
    #**********#
    @classmethod
    def pair(cls, on_ppm=1.88, off_ppm=7.5, **kwargs):
        """The ON and OFF conditions of the editing experiment, as a named dict."""
        return {"on": cls(edit_ppm=on_ppm, **kwargs), "off": cls(edit_ppm=off_ppm, **kwargs)}


#**************************************************************************************************#
#                                         Class SemiLaser                                          #
#**************************************************************************************************#
#                                                                                                  #
# Semi-LASER: 90 excitation refocused by two pairs of adiabatic full passage pulses.               #
#                                                                                                  #
#**************************************************************************************************#
class SemiLaser(Sequence):
    """Semi-LASER: a 90 excitation refocused by two pairs of adiabatic full passage
    pulses (hyperbolic secant), the consensus-recommended localization. Paired AFPs
    cancel each other's nonlinear phase, and because adiabatic inversion is insensitive
    to the transmit field, the refocusing chain keeps its efficiency where PRESS loses
    it; only the excitation retains B1 dependence.

    CPMG-style spacing: te/8 either side of the chain, te/4 between refocusing pulses.

    Args:
        te_s: Echo time in seconds.
        afp_duration_s: Duration of each adiabatic pulse.
        afp_bandwidth_hz: Sweep bandwidth of each adiabatic pulse.
        afp_peak_hz: Peak RF amplitude; must sit above the adiabatic threshold.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, afp_duration_s=0.005, afp_bandwidth_hz=2000.0, afp_peak_hz=1500.0):
        self.te_s = te_s
        self.afp_duration_s = afp_duration_s
        self.afp_bandwidth_hz = afp_bandwidth_hz
        self.afp_peak_hz = afp_peak_hz

        def afp():
            return ShapedPulse.hyperbolic_secant(afp_duration_s, afp_bandwidth_hz, afp_peak_hz,
                                                 offset_ppm=4.65)

        tau = te_s / 8.0
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(tau),
            afp(), CoherenceFilter(+1), Delay(2.0 * tau),
            afp(), CoherenceFilter(-1), Delay(2.0 * tau),
            afp(), CoherenceFilter(+1), Delay(2.0 * tau),
            afp(), CoherenceFilter(-1), Delay(tau),
        )
