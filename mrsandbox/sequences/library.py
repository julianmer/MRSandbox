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
import numpy as np

from mrsandbox.database import pulses, suppression
from mrsandbox.sequences.events import (CoherenceFilter, Delay, Gradient, Projection, Pulse,
                                        SelectivePulse, Sequence, ShapedPulse)


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
    def __init__(self, tr_s=None):
        super().__init__(Pulse(90.0), CoherenceFilter(-1), tr_s=tr_s)


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
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te1_s, te2_s, tr_s=None):
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
            tr_s=tr_s,
        )


#**************************************************************************************************#
#                                           Class Steam                                            #
#**************************************************************************************************#
#                                                                                                  #
# Ideal STEAM: 90x - te/2 - 90x - tm - 90x - te/2 - acquire, stimulated-echo pathway only.         #
#                                                                                                  #
#**************************************************************************************************#
class Steam(Sequence):
    """Ideal STEAM: 90x - te/2 - 90x - tm - 90(-x) - te/2 - acquire, stimulated-echo pathway
    only: order +1 in the first half, stored along z through the mixing time, order -1
    in the second half, so the shifts refocus at the echo (the pathway -1, 0, -1 has the
    same half amplitude and never refocuses).

    Args:
        te_s: Echo time in seconds.
        tm_s: Mixing time in seconds.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, tm_s, tr_s=None):
        self.te_s = te_s
        self.tm_s = tm_s
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(+1),
            Delay(te_s / 2.0),
            Pulse(90.0),
            CoherenceFilter(0),
            Delay(tm_s),
            Pulse(90.0, phase_deg=180.0),  # the echo upright, in phase with a plain excitation
            CoherenceFilter(-1),
            Delay(te_s / 2.0),
            tr_s=tr_s,
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
#                                       Class LocalizedPress                                       #
#**************************************************************************************************#
#                                                                                                  #
# PRESS with its localization played: three slice-selective sinc pulses on three axes.             #
#                                                                                                  #
#**************************************************************************************************#
class LocalizedPress(Sequence):
    """PRESS with its localization played: a slice-selective 90 on x and slice-selective
    180s on y and z, so the voxel's profile and, above all, the chemical shift
    displacement of the narrow refocusing pulses emerge: coupled partners whose
    slabs do not coincide are refocused in part of the voxel only, which is what
    costs lactate its inverted doublet at TE 144 ms at 3 T and above.

    The pulses are linear-phase Shinnar-Le Roux designs (`ShapedPulse.slr`), as on a
    scanner: a 90 of 2.6 ms and 2.3 kHz (time-bandwidth 6, 14 uT) rephased by 0.52
    of its gradient moment (where its slab integral peaks), and 180s of 6.5 ms and
    0.96 kHz (time-bandwidth 6.24, 24.5 uT: what a 25 uT body coil allows; 1.2 kHz in
    5.2 ms would take 31 uT). Each axis is averaged as soon as its pulse has played
    (`Projection`), and coherence filters on BOTH sides of every refocusing pulse
    select the pathway, as its crusher pair does: only what was transverse before the
    pulse may pass, not the magnetization that recovered during the delay and that
    the pulse's edges excite. Every pulse's phase
    sets its carrier, a ramp from the pulse's start, to read its nominal phase at the
    pulse's centre. The echo times run between the pulse centres.

    Args:
        te1_s, te2_s: The two echo times in seconds, as for Press.
        f0_mhz: The spectrometer frequency (the carrier phases need it).
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        excitation_s, refocusing_s: The durations of the 90 and of each 180; the
            bandwidths are the time-bandwidth products over them.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    #**********#
    #   init   #
    #**********#
    EXCITATION_TBW, REFOCUSING_TBW, REPHASE = 6.0, 6.24, 0.52

    def __init__(self, te1_s, te2_s, f0_mhz, size_mm=20.0, reference_ppm=3.0, excitation_s=0.0026,
                 refocusing_s=0.0065, span=2.5, tr_s=None):
        self.te1_s = te1_s
        self.te2_s = te2_s
        edges = (float(size_mm),) * 3 if np.ndim(size_mm) == 0 else tuple(float(s) for s in size_mm)
        delays = (te1_s / 2.0 - excitation_s / 2.0 - refocusing_s / 2.0,
                  (te1_s + te2_s) / 2.0 - refocusing_s, te2_s / 2.0 - refocusing_s / 2.0)
        if min(delays) <= 0.0:
            raise ValueError(f"te1_s {te1_s} and te2_s {te2_s} are too short for the pulses")

        def slab(flip_deg, duration_s, axis, edge, phase_deg):
            kind, tbw = (("excitation", self.EXCITATION_TBW) if flip_deg == 90
                         else ("refocusing", self.REFOCUSING_TBW))
            slope = tbw / duration_s / edge  # Hz per mm
            carrier = -180.0 * reference_ppm * f0_mhz * duration_s
            return slope, ShapedPulse.slr(float(flip_deg), duration_s, tbw, kind, steps=128,
                                          offset_ppm=reference_ppm, phase_deg=phase_deg + carrier,
                                          gradient_hz_per_mm=slope, gradient_axis=axis,
                                          extent_mm=span * edge)

        slope, excitation = slab(90, excitation_s, "x", edges[0], 0.0)
        super().__init__(
            excitation, Gradient(-self.REPHASE * slope * excitation_s, span * edges[0], axis="x"),
            CoherenceFilter(-1), Projection(span), Delay(delays[0]),
            CoherenceFilter(-1), slab(180, refocusing_s, "y", edges[1], 90.0)[1],
            CoherenceFilter(+1), Projection(span), Delay(delays[1]),
            CoherenceFilter(+1), slab(180, refocusing_s, "z", edges[2], 90.0)[1],
            CoherenceFilter(-1), Projection(span), Delay(delays[2]),
            tr_s=tr_s,
        )


#**************************************************************************************************#
#                                     Class LocalizedMegaPress                                     #
#**************************************************************************************************#
#                                                                                                  #
# MEGA-PRESS with its localization and its editing pulses played.                                  #
#                                                                                                  #
#**************************************************************************************************#
class LocalizedMegaPress(Sequence):
    """MEGA-PRESS with its localization and its editing pulses played: the
    slice-selective Shinnar-Le Roux pulses of `LocalizedPress` on x, y, and z, and two
    Gaussian editing 180s of finite length between them, so the edited multiplet
    carries the voxel's profile, the chemical shift displacement between the coupled
    partners (GABA's 3.0 and 1.9 ppm slabs do not coincide under 0.96 kHz pulses), and
    the editing pulse's real inversion profile with what it does to its neighbours.

    The timing is `MegaPress`'s, between pulse centres: 90 - 5 - 180 - 17 - edit - 17 -
    180 - 17 - edit - 12 ms, TE 68 ms. The editing pulses are 14 ms Gaussians (the
    length in common use at 3 T, Mullins et al. 2014, NeuroImage 86:43; de Graaf
    Table 5.1 gives them 88 Hz at half maximum of the inversion), played without a
    gradient on the voxel already averaged over the axes before them. Coherence
    filters on both sides of every refocusing and editing pulse select what the
    crusher pairs select: the order turns over at a refocusing pulse and stays
    across an editing pulse, so spins inside the editing band, which it turns over
    too, are dephased, as on a scanner (the NAA methyl leaves the ON spectrum).

    Args:
        edit_ppm: Editing pulse centre (1.88 for ON, 7.5 for OFF).
        f0_mhz: The spectrometer frequency (the carrier phases need it).
        taus_s: The five delays between pulse centres in seconds; their sum is TE.
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        edit_s: The editing pulse's duration.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    EXCITATION_S, REFOCUSING_S = 0.0026, 0.0065

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm, f0_mhz, taus_s=(0.005, 0.017, 0.017, 0.017, 0.012), size_mm=20.0,
                 reference_ppm=3.0, edit_s=0.014, span=2.5, tr_s=None):
        self.edit_ppm = edit_ppm
        self.taus_s = tuple(taus_s)
        edges = (float(size_mm),) * 3 if np.ndim(size_mm) == 0 else tuple(float(s) for s in size_mm)
        t90, t180 = self.EXCITATION_S, self.REFOCUSING_S
        t1, t2, t3, t4, t5 = self.taus_s
        delays = (t1 - t90 / 2.0 - t180 / 2.0, t2 - t180 / 2.0 - edit_s / 2.0,
                  t3 - edit_s / 2.0 - t180 / 2.0, t4 - t180 / 2.0 - edit_s / 2.0, t5 - edit_s / 2.0)
        if min(delays) <= 0.0:
            raise ValueError(f"taus_s {self.taus_s} are too short for the pulses between them")
        press = LocalizedPress

        def slab(flip_deg, duration_s, axis, edge, phase_deg):
            kind, tbw = (("excitation", press.EXCITATION_TBW) if flip_deg == 90
                         else ("refocusing", press.REFOCUSING_TBW))
            slope = tbw / duration_s / edge  # Hz per mm
            carrier = -180.0 * reference_ppm * f0_mhz * duration_s
            return slope, ShapedPulse.slr(float(flip_deg), duration_s, tbw, kind, steps=128,
                                          offset_ppm=reference_ppm, phase_deg=phase_deg + carrier,
                                          gradient_hz_per_mm=slope, gradient_axis=axis,
                                          extent_mm=span * edge)

        # one editing pulse object played twice: computed once per run
        edit = ShapedPulse.gaussian(180.0, edit_s, offset_ppm=edit_ppm,
                                    phase_deg=-180.0 * edit_ppm * f0_mhz * edit_s)
        slope, excitation = slab(90, t90, "x", edges[0], 0.0)
        super().__init__(
            excitation, Gradient(-press.REPHASE * slope * t90, span * edges[0], axis="x"),
            CoherenceFilter(-1), Projection(span), Delay(delays[0]),
            CoherenceFilter(-1), slab(180, t180, "y", edges[1], 90.0)[1],
            CoherenceFilter(+1), Projection(span), Delay(delays[1]),
            CoherenceFilter(+1), edit, CoherenceFilter(+1), Delay(delays[2]),
            CoherenceFilter(+1), slab(180, t180, "z", edges[2], 90.0)[1],
            CoherenceFilter(-1), Projection(span), Delay(delays[3]),
            CoherenceFilter(-1), edit, CoherenceFilter(-1), Delay(delays[4]),
            tr_s=tr_s,
        )

    #**********#
    #   pair   #
    #**********#
    @classmethod
    def pair(cls, f0_mhz, on_ppm=1.88, off_ppm=7.5, **kwargs):
        """The ON and OFF conditions of the editing experiment, as a named dict."""
        return {"on": cls(on_ppm, f0_mhz, **kwargs), "off": cls(off_ppm, f0_mhz, **kwargs)}


#**************************************************************************************************#
#                                       Class LocalizedSteam                                       #
#**************************************************************************************************#
#                                                                                                  #
# STEAM with its localization played: three slice-selective Shinnar-Le Roux 90s on three axes.     #
#                                                                                                  #
#**************************************************************************************************#
class LocalizedSteam(Sequence):
    """STEAM with its localization played: slice-selective Shinnar-Le Roux 90s on x, y,
    and z (2.6 ms, 2.3 kHz, 14 uT: every pulse has the bandwidth of an excitation,
    which is why STEAM displaces less than PRESS and reaches shorter echo times).

    The stimulated-echo pathway +1, 0, -1 is selected by coherence filters on both
    sides of the second and third pulse, as the crushers of the two echo halves and
    of the mixing time do; zero-quantum coherences of coupled spins pass the mixing
    time, as they do on a scanner. The first pulse is rephased after it, the second,
    which stores, before it (the same pulse run backwards), the third after it, each
    by 0.52 of its gradient moment, and each axis is averaged as soon as its pulse
    has played (`Projection`). The times run between pulse centres; the carriers
    read their nominal phase at the pulse centres.

    Args:
        te_s: Echo time in seconds.
        tm_s: Mixing time in seconds.
        f0_mhz: The spectrometer frequency (the carrier phases need it).
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    PULSE_S, PULSE_TBW, REPHASE = 0.0026, 6.0, 0.52

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, tm_s, f0_mhz, size_mm=20.0, reference_ppm=3.0, span=2.5, tr_s=None):
        self.te_s = te_s
        self.tm_s = tm_s
        edges = (float(size_mm),) * 3 if np.ndim(size_mm) == 0 else tuple(float(s) for s in size_mm)
        t = self.PULSE_S
        if te_s / 2.0 <= t or tm_s <= t:
            raise ValueError(f"te_s {te_s} and tm_s {tm_s} are too short for the {t * 1e3:.1f} ms pulses")
        carrier = -180.0 * reference_ppm * f0_mhz * t

        def slab(axis, edge, phase_deg):
            slope = self.PULSE_TBW / t / edge  # Hz per mm
            pulse = ShapedPulse.slr(90.0, t, self.PULSE_TBW, "excitation", steps=128,
                                    offset_ppm=reference_ppm, phase_deg=phase_deg + carrier,
                                    gradient_hz_per_mm=slope, gradient_axis=axis,
                                    extent_mm=span * edge)
            return pulse, Gradient(-self.REPHASE * slope * t, span * edge, axis=axis)

        (first, after_x), (second, before_y), (third, after_z) = (
            slab("x", edges[0], 0.0), slab("y", edges[1], 0.0), slab("z", edges[2], 180.0))
        super().__init__(
            first, after_x, CoherenceFilter(+1), Projection(span), Delay(te_s / 2.0 - t),
            CoherenceFilter(+1), before_y, second, CoherenceFilter(0), Projection(span),
            Delay(tm_s - t),
            CoherenceFilter(0), third, after_z, CoherenceFilter(-1), Projection(span),
            Delay(te_s / 2.0 - t / 2.0),
            tr_s=tr_s,
        )


#**************************************************************************************************#
#                                     Class LocalizedSemiLaser                                     #
#**************************************************************************************************#
#                                                                                                  #
# Semi-LASER with its localization played: a slice-selective 90 and GOIA-WURST pairs on two axes.  #
#                                                                                                  #
#**************************************************************************************************#
class LocalizedSemiLaser(Sequence):
    """Semi-LASER with its localization played: a slice-selective 90 and GOIA-WURST
    pairs on two axes, so the voxel's profile, the chemical shift displacement, and
    the transmit field's hold on the excitation emerge.

    The design of the across-vendor 3 T semi-LASER (Deelchand et al. 2021, NMR Biomed
    34:e4218): GOIA-W(16,4) refocusing pulses of 4.5 ms and 10 kHz at 15 uT; the
    excitation is a Shinnar-Le Roux 90 of their 2.6 ms and 2.6 kHz (linear phase
    here, asymmetric in theirs, stated) under its slice gradient on x, rephased by
    0.52 of its moment, the pairs play on y and z. Each axis is averaged over its isochromats as
    soon as its pulses have played (`Projection`, the one-dimensional projection of
    Zhang et al. 2017), coherence filters on both sides of every refocusing pulse
    select the echo pathway as its crusher pair does (what recovered during a delay
    does not pass), and the ensemble spans `span` voxel
    edges per axis so the profiles' feet and the displaced slabs stay inside it. The
    echo time runs from the excitation's centre: the delays are what te_s leaves
    after the four refocusing pulses, the first one shortened by the half of the
    excitation the shifts already evolve through.

    Args:
        te_s: Echo time in seconds.
        f0_mhz: The spectrometer frequency: the excitation's phase is set so its
            carrier, a phase ramp from the pulse's start, reads zero at the pulse's
            centre, as a receiver locked to the transmitter sees it.
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    EXCITATION_S, EXCITATION_TBW, REPHASE = 0.0026, 6.76, 0.52
    AFP_S, AFP_BANDWIDTH_HZ, AFP_PEAK_HZ = 0.0045, 10000.0, 15.0 * 42.5774785

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, f0_mhz, size_mm=20.0, reference_ppm=3.0, span=2.5, tr_s=None):
        self.te_s = te_s
        edges = (float(size_mm),) * 3 if np.ndim(size_mm) == 0 else tuple(float(s) for s in size_mm)
        tau = (te_s - 4.0 * self.AFP_S) / 8.0
        if tau <= self.EXCITATION_S / 2.0:
            raise ValueError(f"te_s {te_s} is too short for the excitation and four "
                             f"{self.AFP_S * 1e3:.1f} ms refocusing pulses")
        slope = self.EXCITATION_TBW / self.EXCITATION_S / edges[0]  # Hz per mm

        def afp(axis, edge):
            # 128 steps: within 5e-4 of 512 on lactate, glutamate, and glucose
            return ShapedPulse.goia_wurst(self.AFP_S, self.AFP_BANDWIDTH_HZ, self.AFP_PEAK_HZ,
                                          steps=128, offset_ppm=reference_ppm, extent_mm=span * edge,
                                          gradient_hz_per_mm=self.AFP_BANDWIDTH_HZ / edge,
                                          gradient_axis=axis)

        # one pulse object per pair: its propagator is computed once per run
        afp_y, afp_z = afp("y", edges[1]), afp("z", edges[2])
        super().__init__(
            ShapedPulse.slr(90.0, self.EXCITATION_S, self.EXCITATION_TBW, "excitation", steps=128,
                            offset_ppm=reference_ppm,
                            phase_deg=-180.0 * reference_ppm * f0_mhz * self.EXCITATION_S,
                            gradient_hz_per_mm=slope, gradient_axis="x", extent_mm=span * edges[0]),
            Gradient(-self.REPHASE * slope * self.EXCITATION_S, span * edges[0], axis="x"),
            CoherenceFilter(-1), Projection(span),
            Delay(tau - self.EXCITATION_S / 2.0),
            CoherenceFilter(-1), afp_y, CoherenceFilter(+1), Delay(2.0 * tau),
            CoherenceFilter(+1), afp_y, CoherenceFilter(-1), Projection(span), Delay(2.0 * tau),
            CoherenceFilter(-1), afp_z, CoherenceFilter(+1), Delay(2.0 * tau),
            CoherenceFilter(+1), afp_z, CoherenceFilter(-1), Projection(span), Delay(tau),
            tr_s=tr_s,
        )


#**************************************************************************************************#
#                                  Class OuterVolumeSuppression                                    #
#**************************************************************************************************#
#                                                                                                  #
# Outer-volume suppression as physics: slab-selective saturation of the regions around the voxel.  #
#                                                                                                  #
#**************************************************************************************************#
class OuterVolumeSuppression(Sequence):
    """Outer-volume suppression as physics: slab-selective saturation of the regions
    around the voxel, played before the localization (`Ovs(...) + Press(...)`).

    Every band is a three-lobe sinc 90 under a slice gradient on its axis, its
    carrier set so the slab of `thickness_mm` sits at `centre_mm` from the voxel
    centre for spins at `reference_ppm` (other shifts displace by their chemical
    shift over the gradient, the same displacement the localization suffers),
    followed by a coherence filter keeping the longitudinal magnetization only: the
    idealization of the crusher (a scanner's crusher moment aliases on the discrete
    ensemble, and a smaller one a later slice lobe can undo, so the filter states the
    dephasing exactly; the spurious echo of a band is not modelled). The voxel's own
    isochromats span `extent_mm` about its centre and a band outside it leaves them
    alone; the fat layer, rendered where it sits (phantom.lipids.Skull), is what a
    band covers, and what survives (the profile's transition band, an imperfect
    flip, T1 recovery before the excitation) is the physical residual, not a typed
    factor.

    Args:
        bands: ((axis, centre_mm, thickness_mm), ...), each axis "x", "y", or "z".
        f0_mhz: The spectrometer frequency, which sets the carriers in ppm.
        duration_s: Each saturation pulse's length.
        extent_mm: The isochromat span of the voxel (its size along the axes).
        reference_ppm: The scanner reference the slab positions hold at (water).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, bands, f0_mhz, duration_s=0.003, extent_mm=20.0, reference_ppm=4.65):
        self.bands = tuple((axis, float(centre), float(thickness)) for axis, centre, thickness in bands)
        bandwidth_hz = pulses.DE_GRAAF[("sinc", 90)]["r_bw"] / duration_s
        events = []
        for axis, centre_mm, thickness_mm in self.bands:
            gradient = bandwidth_hz / thickness_mm  # Hz per mm
            events.append(ShapedPulse.sinc(90.0, duration_s,
                                           offset_ppm=reference_ppm + gradient * centre_mm / f0_mhz,
                                           gradient_hz_per_mm=gradient, gradient_axis=axis,
                                           extent_mm=extent_mm))
            events.append(CoherenceFilter(0))
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
        edit_ppm: Editing pulse centre (1.88 for ON, 7.5 for OFF), or a tuple of
            centres for a multi-band editing pulse (Hadamard-encoded schemes).
        taus_s: The five inter-pulse delays in seconds; their sum is the echo time.
        bandwidth_hz: Full selective bandwidth of the editing pulses.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm=1.88, taus_s=(0.005, 0.017, 0.017, 0.017, 0.012), bandwidth_hz=60.0,
                 tr_s=None):
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
            tr_s=tr_s,
        )

    #**********#
    #   pair   #
    #**********#
    @classmethod
    def pair(cls, on_ppm=1.88, off_ppm=7.5, **kwargs):
        """The ON and OFF conditions of the editing experiment, as a named dict."""
        return {"on": cls(edit_ppm=on_ppm, **kwargs), "off": cls(edit_ppm=off_ppm, **kwargs)}


#**************************************************************************************************#
#                                      Class HadamardEditing                                       #
#**************************************************************************************************#
#                                                                                                  #
# The machinery of Hadamard-encoded editing: MEGA sub-experiments with multi-band editing pulses   #
# plus the combination weights that reconstruct each edited difference.                            #
#                                                                                                  #
#**************************************************************************************************#
class HadamardEditing(MegaPress):
    """The machinery of Hadamard-encoded editing (Chan et al. 2016): MEGA sub-experiments
    whose editing pulses carry one or two bands, encoded so every target is ON in half
    the sub-experiments and OFF in the other half. Named subclasses type `bands` (the
    editing band centres of each condition) and `combinations` (per-condition weights);
    `scheme()` builds the named condition dict that Multiplex acquires, and the weights
    reconstruct each edited difference from the acquired stack.

    The published schemes below run at TE 80 ms with 20 ms editing pulses. The papers
    type no delay table, so the default taus place the editing pulse centres at TE/4
    and 3TE/4 (their TE/2 spacing refocuses the J evolution of the edited spins at TE)
    and keep the FID-A convention of 5 ms from excitation to the first refocusing
    pulse, which then sits TE/2 before the second: a stated assumption.

    Args:
        edit_ppm: Editing band centre(s) in ppm of one sub-experiment.
        taus_s: The five inter-pulse delays in seconds; their sum is the echo time.
        bandwidth_hz: Full selective bandwidth of each editing band.
    """

    bands = {}
    combinations = {}

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm, taus_s=(0.005, 0.015, 0.025, 0.015, 0.020), bandwidth_hz=60.0,
                 tr_s=None):
        super().__init__(edit_ppm=edit_ppm, taus_s=taus_s, bandwidth_hz=bandwidth_hz, tr_s=tr_s)

    #************#
    #   scheme   #
    #************#
    @classmethod
    def scheme(cls, **kwargs):
        """The named sub-experiments of this scheme, as Multiplex consumes them."""
        return {name: cls(edit_ppm=centres, **kwargs) for name, centres in cls.bands.items()}


#**************************************************************************************************#
#                                           Class Hermes                                           #
#**************************************************************************************************#
#                                                                                                  #
# HERMES for GABA and GSH: four Hadamard-encoded sub-experiments at TE 80 ms.                      #
#                                                                                                  #
#**************************************************************************************************#
class Hermes(HadamardEditing):
    """HERMES for GABA and GSH (Saleh et al. 2016): four sub-experiments at TE 80 ms
    with 20 ms editing pulses at 1.9 ppm (ON GABA), 4.56 ppm (ON GSH), both at once (a
    dual-lobe pulse), or 7.22 ppm (the published non-interacting position of the
    OFF-both scan). The gaba combination a + b - c - d sums the ON GABA scans minus
    the OFF GABA scans (the 3.01 ppm multiplet); gsh is a - b + c - d (the 2.95 ppm
    cysteinyl resonance); sum recovers the unedited spectrum.
    """

    bands = {"a": (1.9, 4.56), "b": (1.9,), "c": (4.56,), "d": (7.22,)}
    combinations = {
        "sum": {"a": +1.0, "b": +1.0, "c": +1.0, "d": +1.0},
        "gaba": {"a": +1.0, "b": +1.0, "c": -1.0, "d": -1.0},
        "gsh": {"a": +1.0, "b": -1.0, "c": +1.0, "d": -1.0},
    }


#**************************************************************************************************#
#                                          Class Hercules                                          #
#**************************************************************************************************#
#                                                                                                  #
# HERCULES: four Hadamard-encoded sub-experiments editing seven metabolites at once.               #
#                                                                                                  #
#**************************************************************************************************#
class Hercules(HadamardEditing):
    """HERCULES (Oeltzschner et al. 2019): four sub-experiments at TE 80 ms with 20 ms
    editing pulses at 4.58 and 1.9 ppm, 4.18 and 1.9 ppm, 4.58 ppm, or 4.18 ppm,
    editing seven low-concentration metabolites at once. The gaba combination
    a + b - c - d carries the 1.9 ppm targets (GABA, lactate among them); gsh, as
    a - b + c - d, carries the targets that distinguish 4.58 from 4.18 ppm (GSH,
    aspartate, NAA, NAAG); sum recovers the unedited spectrum.
    """

    bands = {"a": (4.58, 1.9), "b": (4.18, 1.9), "c": (4.58,), "d": (4.18,)}
    combinations = {
        "sum": {"a": +1.0, "b": +1.0, "c": +1.0, "d": +1.0},
        "gaba": {"a": +1.0, "b": +1.0, "c": -1.0, "d": -1.0},
        "gsh": {"a": +1.0, "b": -1.0, "c": +1.0, "d": -1.0},
    }


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
        tr_s: Repetition time; None is fully relaxed (see Sequence).

    The defaults (6 ms, 1.5 kHz, 1000 Hz or 23.5 uT) stay within the 20 to 25 uT a 3 T
    body coil delivers (Deelchand et al. 2021, NMR Biomed 34:e4218) and keep the echo
    down to 70 percent of the nominal B1; the plain hyperbolic secant pays for that
    with its bandwidth (the gradient-modulated pulses of that paper reach 10 kHz at
    15 uT).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te_s, afp_duration_s=0.006, afp_bandwidth_hz=1500.0, afp_peak_hz=1000.0,
                 tr_s=None):
        self.te_s = te_s
        self.afp_duration_s = afp_duration_s
        self.afp_bandwidth_hz = afp_bandwidth_hz
        self.afp_peak_hz = afp_peak_hz

        # one pulse object played four times: its propagator is computed once per run
        afp = ShapedPulse.hyperbolic_secant(afp_duration_s, afp_bandwidth_hz, afp_peak_hz,
                                            offset_ppm=4.65)
        tau = te_s / 8.0
        super().__init__(
            Pulse(90.0),
            CoherenceFilter(-1),
            Delay(tau),
            afp, CoherenceFilter(+1), Delay(2.0 * tau),
            afp, CoherenceFilter(-1), Delay(2.0 * tau),
            afp, CoherenceFilter(+1), Delay(2.0 * tau),
            afp, CoherenceFilter(-1), Delay(tau),
            tr_s=tr_s,
        )
