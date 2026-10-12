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

from mrsandbox.database import suppression
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

    Coherence filters select the double spin echo, the idealization of a crusher scheme
    that works. `crushers` plays the scheme in their place: a pair of gradient kicks
    around each refocusing pulse, dephasing on the isochromat ensemble whatever the
    pulse did not refocus. With perfect pulses and moments of whole cycles over the
    ensemble the two agree exactly; with a transmit field off its nominal value a weak
    pair lets through what a scanner's weak crushers let through (the spurious signal
    of unrefocused coherences), and a strong pair removes it. The localized designs
    keep their filters: their one-dimensional projection (events.Projection) holds
    only while no gradient on another axis selects the pathway.

    Args:
        te1_s: First echo time in seconds (may be a traced tensor).
        te2_s: Second echo time in seconds (may be a traced tensor).
        tr_s: Repetition time; None is fully relaxed (see Sequence).
        crushers: None for coherence filters, or the moments of the two pairs in cycles
            per millimetre over the 20 mm ensemble, (0.15, 0.25) being three and five
            whole cycles (they must differ, or the second pair refocuses what the first
            dephased).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, te1_s, te2_s, tr_s=None, crushers=None):
        self.te1_s = te1_s
        self.te2_s = te2_s
        refocus = Pulse(180.0, phase_deg=90.0)
        if crushers is None:
            events = (Pulse(90.0), CoherenceFilter(-1), Delay(te1_s / 2.0), refocus,
                      CoherenceFilter(+1), Delay((te1_s + te2_s) / 2.0), refocus,
                      CoherenceFilter(-1), Delay(te2_s / 2.0))
        else:
            first, second = Gradient(crushers[0], 20.0), Gradient(crushers[1], 20.0)
            events = (Pulse(90.0), Delay(te1_s / 2.0), first, refocus, first,
                      Delay((te1_s + te2_s) / 2.0), second, refocus, second, Delay(te2_s / 2.0))
        super().__init__(*events, tr_s=tr_s)


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
    def __init__(self, te_s, tm_s, tr_s=None, crushers=None):
        self.te_s = te_s
        self.tm_s = tm_s
        last = Pulse(90.0, phase_deg=180.0)  # the echo upright, in phase with a plain excitation
        if crushers is None:
            events = (Pulse(90.0), CoherenceFilter(+1), Delay(te_s / 2.0), Pulse(90.0),
                      CoherenceFilter(0), Delay(tm_s), last, CoherenceFilter(-1),
                      Delay(te_s / 2.0))
        else:
            # `crushers` plays the scheme in the filters' place (see Press): the same kick in both
            # echo halves, which the stimulated echo's orders +1 and -1 undo, and a spoiler in the
            # mixing time, which the stored order 0 does not feel; moments in cycles per
            # millimetre over the 20 mm ensemble, (echo halves, mixing time)
            half, mixing = Gradient(crushers[0], 20.0), Gradient(crushers[1], 20.0)
            events = (Pulse(90.0), half, Delay(te_s / 2.0), Pulse(90.0), mixing, Delay(tm_s),
                      last, half, Delay(te_s / 2.0))
        super().__init__(*events, tr_s=tr_s)


#********************#
#   gaussian width   #
#********************#
def gaussian_for(bandwidth_hz):
    """The duration of a 10 percent cutoff Gaussian that saturates `bandwidth_hz` at half
    (1 - Mz): the COWS pulse of Igwe et al. 2025 does 68 Hz in 17.92 ms, and the width
    scales as one over the duration. For the trains whose sources hold a band and no
    pulse."""
    return 0.01792 * 68.0 / float(bandwidth_hz)


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

    The pulses are ideal bands taking no time, or, with a `pulse_s`, pulses played in
    time (a Gaussian cut at `truncation` of its peak, or a sinc of `lobes` zero crossings
    a side): then the profile, the spins' evolution during the pulse and the time it takes
    are physical. A delay is the gap after a pulse, before the next pulse or the
    excitation; a played pulse adds its own duration before its gap.

    Args:
        flips_deg: One flip angle per suppression pulse.
        offset_ppm: Pulse centre in ppm (water sits at 4.65).
        bandwidth_hz: Full selective bandwidth in Hz of the ideal band, and the stated
            width the closed forms keep for a played pulse.
        delays_s: Optional gap after each pulse (same length as flips_deg).
        pulse_s: Optional pulse duration in seconds; None plays ideal bands.
        shape: "gaussian" or "sinc" for a played pulse.
        truncation: Where the Gaussian is cut, as a fraction of its peak.
        lobes: The sinc's zero crossings a side.
        adjusted: Whether the scanner's water suppression adjustment scales this train
            per voxel (a product's does; a scheme played at its nominal flips is not).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flips_deg, offset_ppm=4.65, bandwidth_hz=60.0, delays_s=None, pulse_s=None,
                 shape="gaussian", truncation=0.1, lobes=2, adjusted=True):
        self.flips_deg = tuple(flips_deg)
        self.offset_ppm = offset_ppm
        self.bandwidth_hz = bandwidth_hz
        self.delays_s = tuple(delays_s) if delays_s is not None else (0.0,) * len(self.flips_deg)
        self.pulse_s, self.shape, self.truncation, self.lobes = pulse_s, shape, truncation, lobes
        self.adjusted = adjusted
        events = []
        for flip, delay in zip(self.flips_deg, self.delays_s):
            if pulse_s is None:
                events.append(SelectivePulse(flip, offset_ppm=offset_ppm,
                                             bandwidth_hz=bandwidth_hz))
            elif shape == "gaussian":
                events.append(ShapedPulse.gaussian(flip, pulse_s, offset_ppm=offset_ppm,
                                                   truncation=truncation))
            else:
                events.append(ShapedPulse.sinc(flip, pulse_s, offset_ppm=offset_ppm, lobes=lobes))
            events.append(CoherenceFilter(0))
            if delay > 0.0:
                events.append(Delay(delay))
        super().__init__(*events)

    #**************#
    #   adjusted   #
    #**************#
    def scaled(self, factor):
        """This train with every flip times `factor`: what the scanner's water suppression
        adjustment does to the pulses' power."""
        return SuppressionTrain(tuple(flip * factor for flip in self.flips_deg),
                                offset_ppm=self.offset_ppm, bandwidth_hz=self.bandwidth_hz,
                                delays_s=self.delays_s, pulse_s=self.pulse_s, shape=self.shape,
                                truncation=self.truncation, lobes=self.lobes,
                                adjusted=self.adjusted)


#*****************#
#   voxel edges   #
#*****************#
def voxel_edges(size_mm):
    """The (x, y, z) edges in mm of a voxel given as one edge or as its three."""
    return (float(size_mm),) * 3 if np.ndim(size_mm) == 0 else tuple(float(s) for s in size_mm)


#****************#
#   slab pulse   #
#****************#
def slab_pulse(flip_deg, duration_s, time_bandwidth, axis, edge_mm, span, reference_ppm, f0_mhz,
               phase_deg=0.0):
    """One slice-selective Shinnar-Le Roux pulse of a localized design and its gradient
    in Hz per mm: the slab of `edge_mm` for spins at `reference_ppm` (the bandwidth,
    the time-bandwidth product over the duration, over the edge), the ensemble
    spanning `span` edges, and the carrier, a phase ramp from the pulse's start,
    reading `phase_deg` at the pulse's centre."""
    slope = time_bandwidth / duration_s / edge_mm
    kind = "excitation" if flip_deg == 90 else "refocusing"
    pulse = ShapedPulse.slr(float(flip_deg), duration_s, time_bandwidth, kind, steps=128,
                            offset_ppm=reference_ppm,
                            phase_deg=phase_deg - 180.0 * reference_ppm * f0_mhz * duration_s,
                            gradient_hz_per_mm=slope, gradient_axis=axis, extent_mm=span * edge_mm)
    return pulse, slope


#*******************#
#   editing pulse   #
#*******************#
def editing_pulse(edit_ppm, f0_mhz, duration_s):
    """The Gaussian editing 180 of the localized editing designs, its carrier reading
    zero phase at the pulse's centre: one band at `edit_ppm`, or for two centres a
    dual band, the Gaussian cosine-modulated about their middle at twice the single
    band's amplitude (each band then turns its spins by 180 degrees).

    The Gaussian is truncated at 5.71 percent of its peak, where it inverts a band 1.24
    over its duration wide at half maximum: 82.7 Hz at 15 ms and 62.0 Hz at 20 ms, the
    scanners' own (82.5 and 61.9 Hz on Philips, 81.7 and 61.3 Hz on GE, Mikkelsen et al.
    2017, NeuroImage 159:32), and 88.6 Hz at 14 ms (de Graaf Table 5.1: 88 Hz)."""
    centres = tuple(edit_ppm) if isinstance(edit_ppm, (tuple, list)) else (edit_ppm,)
    if len(centres) > 2:
        raise ValueError(f"an editing pulse carries one or two bands, got {len(centres)}")
    middle = sum(centres) / len(centres)
    pulse = ShapedPulse.gaussian(180.0, duration_s, offset_ppm=middle, truncation=0.0571,
                                 phase_deg=-180.0 * middle * f0_mhz * duration_s)
    if len(centres) == 1:
        return pulse
    envelope = np.asarray(pulse.envelope, dtype=np.float64)
    t = ((np.arange(envelope.size) + 0.5) / envelope.size - 0.5) * duration_s
    split = np.cos(np.pi * (centres[0] - centres[1]) * f0_mhz * t)
    single = 0.5 / (envelope.sum() * duration_s / envelope.size)  # a 180's peak in Hz
    return ShapedPulse(None, duration_s, (envelope / envelope.max() * split).astype(np.float32),
                       offset_ppm=middle, phase_deg=pulse.phase_deg,
                       peak_hz=2.0 * single * envelope.max())


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
    the pulse's edges excite. Every pulse's phase sets its carrier, a ramp from the
    pulse's start, to read its nominal phase at the pulse's centre. The echo times run
    between the pulse centres.

    Args:
        te1_s, te2_s: The two echo times in seconds, as for Press.
        f0_mhz: The spectrometer frequency (the carrier phases need it).
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        excitation_s, refocusing_s: The durations of the 90 and of each 180; the
            bandwidths are the time-bandwidth products over them.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
        axes: Which axis the 90 selects and which the two 180s, in that order ("xyz" by
            default): the narrow 180s displace a shift the most, so the order decides
            along which two axes of the subject that happens.
    """

    #**********#
    #   init   #
    #**********#
    EXCITATION_TBW, REFOCUSING_TBW, REPHASE = 6.0, 6.24, 0.52

    def __init__(self, te1_s, te2_s, f0_mhz, size_mm=20.0, reference_ppm=3.0,
                 excitation_s=0.0026, refocusing_s=0.0065, span=2.5, tr_s=None, axes="xyz"):
        if sorted(axes) != ["x", "y", "z"]:
            raise ValueError(f"axes must order x, y, and z, got {axes!r}")
        self.te1_s = te1_s
        self.te2_s = te2_s
        self.axes = axes
        edges = [voxel_edges(size_mm)["xyz".index(axis)] for axis in axes]
        delays = (te1_s / 2.0 - excitation_s / 2.0 - refocusing_s / 2.0,
                  (te1_s + te2_s) / 2.0 - refocusing_s, te2_s / 2.0 - refocusing_s / 2.0)
        if min(delays) <= 0.0:
            raise ValueError(f"te1_s {te1_s} and te2_s {te2_s} are too short for the pulses")

        frame = (span, reference_ppm, f0_mhz)
        excitation, slope = slab_pulse(90, excitation_s, self.EXCITATION_TBW, axes[0], edges[0],
                                       *frame)
        refocus_y, _ = slab_pulse(180, refocusing_s, self.REFOCUSING_TBW, axes[1], edges[1],
                                  *frame, 90.0)
        refocus_z, _ = slab_pulse(180, refocusing_s, self.REFOCUSING_TBW, axes[2], edges[2],
                                  *frame, 90.0)
        super().__init__(
            excitation,
            Gradient(-self.REPHASE * slope * excitation_s, span * edges[0], axis=axes[0]),
            CoherenceFilter(-1), Projection(span), Delay(delays[0]),
            CoherenceFilter(-1), refocus_y, CoherenceFilter(+1), Projection(span), Delay(delays[1]),
            CoherenceFilter(+1), refocus_z, CoherenceFilter(-1), Projection(span), Delay(delays[2]),
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
    too, are dephased, as on a scanner (the NAA methyl leaves the ON spectrum). The
    editing filters carry the moment of the crushers they stand for, so the design
    weights by diffusion as a scanner's does (sequences.diffusion): about 9.7 cycles
    per mm a side, read off the water references of Big GABA's Philips series, where
    the CSF's water fades through the pairs as the data's; the refocusing pulses'
    crushers stay unplayed, a few s/mm^2 (stated).

    Two editing centres make a dual-band pulse, the Gaussian cosine-modulated about
    their middle at twice the single band's amplitude (each band then turns its
    spins by 180 degrees; the bands of the Hadamard schemes lie hundreds of hertz
    apart against a width of tens, so they do not disturb each other), and
    `scheme` builds the sub-experiments of a Hadamard scheme with it.

    Args:
        edit_ppm: Editing pulse centre (1.88 for ON, 7.5 for OFF), or a tuple of one
            or two centres.
        f0_mhz: The spectrometer frequency (the carrier phases and a dual band's
            modulation need it).
        taus_s: The five delays between pulse centres in seconds; their sum is TE.
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        edit_s: The editing pulse's duration.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    EXCITATION_S, REFOCUSING_S = 0.0026, 0.0065
    EDIT_CRUSHER_PER_MM = 9.7

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm, f0_mhz, taus_s=(0.005, 0.017, 0.017, 0.017, 0.012), size_mm=20.0,
                 reference_ppm=3.0, edit_s=0.014, span=2.5, tr_s=None):
        self.edit_ppm = edit_ppm
        self.taus_s = tuple(taus_s)
        edges = voxel_edges(size_mm)
        t90, t180 = self.EXCITATION_S, self.REFOCUSING_S
        t1, t2, t3, t4, t5 = self.taus_s
        delays = (t1 - t90 / 2.0 - t180 / 2.0, t2 - t180 / 2.0 - edit_s / 2.0,
                  t3 - edit_s / 2.0 - t180 / 2.0, t4 - t180 / 2.0 - edit_s / 2.0, t5 - edit_s / 2.0)
        if min(delays) <= 0.0:
            raise ValueError(f"taus_s {self.taus_s} are too short for the pulses between them")
        press, frame = LocalizedPress, (span, reference_ppm, f0_mhz)
        excitation, slope = slab_pulse(90, t90, press.EXCITATION_TBW, "x", edges[0], *frame)
        refocus_y, _ = slab_pulse(180, t180, press.REFOCUSING_TBW, "y", edges[1], *frame, 90.0)
        refocus_z, _ = slab_pulse(180, t180, press.REFOCUSING_TBW, "z", edges[2], *frame, 90.0)
        # one editing pulse object played twice: computed once per run
        edit = editing_pulse(edit_ppm, f0_mhz, edit_s)
        m = self.EDIT_CRUSHER_PER_MM
        super().__init__(
            excitation, Gradient(-press.REPHASE * slope * t90, span * edges[0], axis="x"),
            CoherenceFilter(-1), Projection(span), Delay(delays[0]),
            CoherenceFilter(-1), refocus_y, CoherenceFilter(+1), Projection(span), Delay(delays[1]),
            CoherenceFilter(+1, m), edit, CoherenceFilter(+1, -m), Delay(delays[2]),
            CoherenceFilter(+1), refocus_z, CoherenceFilter(-1), Projection(span), Delay(delays[3]),
            CoherenceFilter(-1, m), edit, CoherenceFilter(-1, -m), Delay(delays[4]),
            tr_s=tr_s,
        )

    #**********#
    #   pair   #
    #**********#
    @classmethod
    def pair(cls, f0_mhz, on_ppm=1.88, off_ppm=7.5, **kwargs):
        """The ON and OFF conditions of the editing experiment, as a named dict."""
        return {"on": cls(on_ppm, f0_mhz, **kwargs), "off": cls(off_ppm, f0_mhz, **kwargs)}

    #************#
    #   scheme   #
    #************#
    @classmethod
    def scheme(cls, editing, f0_mhz, taus_s=(0.005, 0.015, 0.025, 0.015, 0.020), edit_s=0.020,
               **kwargs):
        """The sub-experiments of a Hadamard scheme (`Hermes`, `Hercules`) with the
        localization played, as Multiplex consumes them beside `editing.combinations`:
        the scheme's bands at its TE 80 ms timing and 20 ms editing pulses."""
        return {name: cls(centres, f0_mhz, taus_s=taus_s, edit_s=edit_s, **kwargs)
                for name, centres in editing.bands.items()}


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
        edges = voxel_edges(size_mm)
        t = self.PULSE_S
        if te_s / 2.0 <= t or tm_s <= t:
            raise ValueError(f"te_s {te_s} and tm_s {tm_s} are too short for the "
                             f"{t * 1e3:.1f} ms pulses")

        def slab(axis, edge, phase_deg):
            pulse, slope = slab_pulse(90, t, self.PULSE_TBW, axis, edge, span, reference_ppm,
                                      f0_mhz, phase_deg)
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
    0.52 of its moment, the pairs play on y and z. Each axis is averaged over its
    isochromats as soon as its pulses have played (`Projection`, the one-dimensional
    projection of Zhang et al. 2017), coherence filters on both sides of every
    refocusing pulse select the echo pathway as its crusher pair does (what recovered
    during a delay does not pass), and the ensemble spans `span` voxel edges per axis
    so the profiles' feet and the displaced slabs stay inside it. The echo time runs
    from the excitation's centre: the delays are what te_s leaves after the four
    refocusing pulses, the first one shortened by the half of the excitation the
    shifts already evolve through.

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
        edges = voxel_edges(size_mm)
        tau = (te_s - 4.0 * self.AFP_S) / 8.0
        if tau <= self.EXCITATION_S / 2.0:
            raise ValueError(f"te_s {te_s} is too short for the excitation and four "
                             f"{self.AFP_S * 1e3:.1f} ms refocusing pulses")
        excitation, slope = slab_pulse(90, self.EXCITATION_S, self.EXCITATION_TBW, "x", edges[0],
                                       span, reference_ppm, f0_mhz)

        def afp(axis, edge):
            # 128 steps: finer steps leave coupled spectra as they are
            return ShapedPulse.goia_wurst(self.AFP_S, self.AFP_BANDWIDTH_HZ, self.AFP_PEAK_HZ,
                                          steps=128, offset_ppm=reference_ppm,
                                          extent_mm=span * edge,
                                          gradient_hz_per_mm=self.AFP_BANDWIDTH_HZ / edge,
                                          gradient_axis=axis)

        # one pulse object per pair: its propagator is computed once per run
        afp_y, afp_z = afp("y", edges[1]), afp("z", edges[2])
        super().__init__(
            excitation,
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
#                                  Class LocalizedMegaSemiLaser                                    #
#**************************************************************************************************#
#                                                                                                  #
# MEGA editing on the semi-LASER localization, every pulse played.                                 #
#                                                                                                  #
#**************************************************************************************************#
class LocalizedMegaSemiLaser(Sequence):
    """MEGA editing on the semi-LASER localization, every pulse played: the
    Shinnar-Le Roux excitation and the GOIA-WURST pairs of `LocalizedSemiLaser`, with
    the Gaussian editing 180s after the first pair and inside the second (90 - AFP -
    AFP - edit - AFP - edit - AFP): an odd number of refocusing pulses lies between
    the two, so they meet the coherence with opposite signs, as in MEGA-PRESS, and
    their Bloch-Siegert phases cancel (with an even number both add, and the editing
    fails). The 10 kHz refocusing slabs move the coupled partners apart by two percent
    of the voxel where PRESS's move them by fifteen, so the spatial editing loss of
    `LocalizedMegaPress` all but goes.

    The pulse centres follow from two conditions: the shifts refocus at TE when the
    two pair spacings sum to TE/2, and the editing pulses refocus the J evolution of
    the edited spins when they sit TE/2 apart. The first pair plays as early and as
    tight as the pulses allow, the first editing pulse right after it, the second
    TE/2 later with the last adiabatic pulse right behind it; the third follows from
    the spacings (TE 80 ms: AFPs at 3.8, 8.6, 32.3, 67.6 ms, editing pulses at 18.1
    and 58.1 ms; a stated schedule, the vendors' differ in the margins). Filters on
    both sides of every adiabatic and editing pulse select what the crushers select,
    the editing ones carrying the crushers' moment as in `LocalizedMegaPress`; each
    axis is averaged after its pair.

    Args:
        edit_ppm: Editing pulse centre, or a tuple of one or two centres.
        f0_mhz: The spectrometer frequency.
        te_s: Echo time in seconds.
        size_mm: The voxel edge, or its (x, y, z) edges.
        reference_ppm: The shift the three slabs are centred for; others displace.
        edit_s: The editing pulse's duration.
        span: The ensemble's extent per axis in voxel edges.
        tr_s: Repetition time; None is fully relaxed (see Sequence).
    """

    MARGIN_S = 0.00025

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm, f0_mhz, te_s=0.080, size_mm=20.0, reference_ppm=3.0,
                 edit_s=0.014, span=2.5, tr_s=None):
        self.edit_ppm = edit_ppm
        self.te_s = te_s
        laser = LocalizedSemiLaser
        t90, afp_s, gap = laser.EXCITATION_S, laser.AFP_S, self.MARGIN_S
        # the pulse centres from the excitation's centre
        a = t90 / 2.0 + afp_s / 2.0 + gap
        b = a + afp_s + gap
        e1 = b + afp_s / 2.0 + edit_s / 2.0 + gap
        e2 = e1 + te_s / 2.0
        d = e2 + edit_s / 2.0 + afp_s / 2.0 + gap
        c = d - (te_s / 2.0 - (b - a))
        self.taus_s = (a, b - a, e1 - b, c - e1, e2 - c, d - e2, te_s - d)
        lengths = (t90 / 2.0 + afp_s / 2.0, afp_s, (afp_s + edit_s) / 2.0, (afp_s + edit_s) / 2.0,
                   (afp_s + edit_s) / 2.0, (afp_s + edit_s) / 2.0, afp_s / 2.0)
        delays = [tau - length for tau, length in zip(self.taus_s, lengths)]
        if min(delays) < -1e-9:
            raise ValueError(f"te_s {te_s} is too short for four {afp_s * 1e3:.1f} ms adiabatic "
                             f"and two {edit_s * 1e3:.0f} ms editing pulses")
        delays = [max(delay, 0.0) for delay in delays]
        edges = voxel_edges(size_mm)
        excitation, slope = slab_pulse(90, t90, laser.EXCITATION_TBW, "x", edges[0], span,
                                       reference_ppm, f0_mhz)

        def afp(axis, edge):
            return ShapedPulse.goia_wurst(afp_s, laser.AFP_BANDWIDTH_HZ, laser.AFP_PEAK_HZ,
                                          steps=128, offset_ppm=reference_ppm,
                                          extent_mm=span * edge,
                                          gradient_hz_per_mm=laser.AFP_BANDWIDTH_HZ / edge,
                                          gradient_axis=axis)

        afp_y, afp_z, edit = (afp("y", edges[1]), afp("z", edges[2]),
                              editing_pulse(edit_ppm, f0_mhz, edit_s))
        m = LocalizedMegaPress.EDIT_CRUSHER_PER_MM
        super().__init__(
            excitation, Gradient(-laser.REPHASE * slope * t90, span * edges[0], axis="x"),
            CoherenceFilter(-1), Projection(span), Delay(delays[0]),
            CoherenceFilter(-1), afp_y, CoherenceFilter(+1), Delay(delays[1]),
            CoherenceFilter(+1), afp_y, CoherenceFilter(-1), Projection(span), Delay(delays[2]),
            CoherenceFilter(-1, m), edit, CoherenceFilter(-1, -m), Delay(delays[3]),
            CoherenceFilter(-1), afp_z, CoherenceFilter(+1), Delay(delays[4]),
            CoherenceFilter(+1, m), edit, CoherenceFilter(+1, -m), Delay(delays[5]),
            CoherenceFilter(+1), afp_z, CoherenceFilter(-1), Projection(span), Delay(delays[6]),
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
#                                  Class OuterVolumeSuppression                                    #
#**************************************************************************************************#
#                                                                                                  #
# Outer-volume suppression as physics: slab-selective saturation of the regions around the voxel.  #
#                                                                                                  #
#**************************************************************************************************#
class OuterVolumeSuppression(Sequence):
    """Outer-volume suppression as physics: slab-selective saturation of the regions
    around the voxel, played before the localization (`Ovs(...) + Press(...)`).

    Every band is a Shinnar-Le Roux 90 of `time_bandwidth` under a slice gradient on
    its axis (11 over 3 ms: 3.7 kHz at 23 uT, the most an amplitude-modulated 90 reaches
    under a 25 uT transmitter; the swept pulses real sequences play, hyperbolic secants
    of 9 kHz in 4.6 ms (Oz and Tkac, MRM 65 (2011) 901), are beyond it), its carrier
    set so the slab of `thickness_mm` sits at `centre_mm` from the voxel centre for
    spins at `reference_ppm` (other shifts displace by their chemical shift over the
    gradient, the same displacement the localization suffers; the wider the band, the
    less), followed by a coherence filter keeping the longitudinal magnetization only: the
    idealization of the crusher (a scanner's crusher moment aliases on the discrete
    ensemble, and a smaller one a later slice lobe can undo, so the filter states the
    dephasing exactly; the spurious echo of a band is not modelled). The voxel's own
    isochromats span `extent_mm` about its centre and a band outside it leaves them
    alone; the fat layer, rendered where it sits (phantom.lipids.Skull), is what a
    band covers, and what survives (the profile's transition band, an imperfect
    flip, T1 recovery before the excitation) is the physical residual, not a typed
    factor.

    Args:
        bands: ((axis, centre_mm, thickness_mm), ...), each axis "x", "y", or "z"; a fourth
            item, a unit (x, y, z) vector, tilts that slab off its axis (its centre and
            thickness then run along the vector; None leaves it on its axis).
        f0_mhz: The spectrometer frequency, which sets the carriers in ppm.
        duration_s: Each saturation pulse's length; a transmitter short of its peak
            plays it longer at the same time-bandwidth (narrower), as the console does.
        time_bandwidth: The pulse's time-bandwidth product, its bandwidth over its length.
        extent_mm: The isochromat span of the voxel (its size along the axes).
        reference_ppm: The scanner reference the slab positions hold at (water).
    """

    TIME_BANDWIDTH = 11.0

    #**********#
    #   init   #
    #**********#
    def __init__(self, bands, f0_mhz, duration_s=0.003, time_bandwidth=TIME_BANDWIDTH,
                 extent_mm=20.0, reference_ppm=4.65):
        self.bands = tuple((axis, float(centre), float(thickness), *normal)
                           for axis, centre, thickness, *normal in bands)
        bandwidth_hz = time_bandwidth / duration_s
        events = []
        for axis, centre_mm, thickness_mm, *normal in self.bands:
            gradient = bandwidth_hz / thickness_mm  # Hz per mm
            events.append(ShapedPulse.slr(90.0, duration_s, time_bandwidth, "excitation",
                                          offset_ppm=reference_ppm + gradient * centre_mm / f0_mhz,
                                          gradient_hz_per_mm=gradient, gradient_axis=axis,
                                          gradient_normal=normal[0] if normal else None,
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
    def __init__(self, n=3, flip_deg=90.0, offset_ppm=4.65, bandwidth_hz=60.0, real=False):
        self.n = n
        self.flip_deg = flip_deg
        super().__init__([flip_deg] * n, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz,
                         pulse_s=gaussian_for(bandwidth_hz) if real else None)


#**************************************************************************************************#
#                                        Class SiemensChess                                        #
#**************************************************************************************************#
#                                                                                                  #
# Siemens' product water saturation: three pulses of 50 Hz, the third overtipping, as svs_se       #
# plays it.                                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class SiemensChess(SuppressionTrain):
    """Siemens' product water saturation: three pulses of 50 Hz, the third overtipping, as
    svs_se plays it. The amplitudes stand in the ratio 1 : 0.935 : 1.803 and the band is
    50 Hz wide (database.suppression SIEMENS_SVS_SE, read from Big GABA headers); like
    WET's last pulse the third one turns the water past zero so that what recovers through
    the last gap crosses zero at the excitation, whatever the T1. The scanner's adjustment
    scales all three at once (`scaled`). The spacing is a stated 30 ms. Played real, the
    pulses are Gaussians of 24.4 ms: the header holds no shape or duration, and that is
    what a 10 percent Gaussian takes for the header's 50 Hz (`gaussian_for`), stated.

    Args:
        flip_deg: The first pulse's flip; the other two follow the ratios.
        offset_ppm: Pulse centre in ppm (water sits at 4.65).
        spacing_s: The gap after each pulse; None takes the table's stated value.
        real: Whether the pulses are played in time (Gaussians) or as ideal bands.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, flip_deg=90.0, offset_ppm=4.65, spacing_s=None, real=False):
        table = suppression.SIEMENS_SVS_SE
        self.flip_deg = flip_deg
        spacing = table["spacing_s"] if spacing_s is None else spacing_s
        super().__init__(tuple(flip_deg * ratio for ratio in table["amplitude_ratios"]),
                         offset_ppm=offset_ppm, bandwidth_hz=table["bandwidth_hz"],
                         delays_s=(spacing,) * len(table["amplitude_ratios"]),
                         pulse_s=gaussian_for(table["bandwidth_hz"]) if real else None)


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
    def __init__(self, delay_s=0.060, offset_ppm=4.65, bandwidth_hz=60.0, real=False):
        self.delay_s = delay_s
        flips = suppression.WET["flips_deg"]
        super().__init__(flips, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz,
                         delays_s=(delay_s,) * len(flips),
                         pulse_s=gaussian_for(bandwidth_hz) if real else None)


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
    Played real, the pulses are the three-lobe sincs of 35 ms of Tkac and Gruetter (ISMRM
    2000, abstract 1856: under 5 percent saturation outside 0.4 ppm at 4 T), symmetric
    here where theirs were asymmetrically truncated, stated; that source plays VAPOR
    without any water suppression adjustment (the ratio to the localization's power
    holds), so the train is not `adjusted`.

    Args:
        field: Delay series to use, "4t" or "7t".
        nominal_flip_deg: Flip of the nominal-power pulses (models a B1 miscalibration).
        offset_ppm: Pulse centre in ppm.
        bandwidth_hz: Full selective bandwidth in Hz.
        real: Whether the pulses are played in time (sincs) or as ideal bands.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, field="4t", nominal_flip_deg=90.0, offset_ppm=4.65, bandwidth_hz=60.0,
                 real=False):
        self.field = field
        self.nominal_flip_deg = nominal_flip_deg
        flips = tuple(nominal_flip_deg * 10.0 ** (db / 20.0)
                      for db in suppression.VAPOR["power_db"])
        pulse = suppression.VAPOR["pulse"]
        super().__init__(flips, offset_ppm=offset_ppm, bandwidth_hz=bandwidth_hz,
                         delays_s=suppression.VAPOR["delays_s"][field],
                         pulse_s=pulse["duration_s"] if real else None, shape="sinc",
                         lobes=pulse["lobes"], adjusted=False)


#**************************************************************************************************#
#                                            Class Cows                                            #
#**************************************************************************************************#
#                                                                                                  #
# The water suppression schemes of Igwe et al. 2025: COWS(7;236), COWS(12;626) and their VAPOR7.   #
#                                                                                                  #
#**************************************************************************************************#
class Cows(SuppressionTrain):
    """The water suppression schemes of Igwe et al. 2025, typed from its Table 1
    (database.suppression COWS): COWS(7;236) and COWS(12;626), flips and delays constrained
    optimized to null the water over T1 0.5 to 5 s and a B1 factor of 0.75 to 1.25 in 236
    and 626 ms, and VAPOR7, the seven pulse VAPOR the paper held them against. The delays
    run from each pulse's peak to the next and the last to the excitation's; the pulses
    are 10 percent cutoff Gaussians of 17.92 ms saturating 68 Hz at half, played in time
    with `real`. The study's scanner played the table's flips as they stand (no
    adjustment per voxel: the dataset's headers), so the train is not `adjusted`.

    Args:
        scheme: "COWS(7;236)", "COWS(12;626)" or "VAPOR7".
        offset_ppm: Pulse centre in ppm (water sits at 4.65).
        real: Whether the pulses are played in time (Gaussians) or as ideal bands.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, scheme="COWS(7;236)", offset_ppm=4.65, real=False):
        self.scheme = scheme
        table, pulse = suppression.COWS[scheme], suppression.COWS["pulse_duration_s"]
        # from each pulse's peak to the next, the last to the excitation's: gaps when played
        delays = table["delays_s"]
        if real:
            delays = (*(d - pulse for d in delays[:-1]), delays[-1] - pulse / 2.0)
        super().__init__(table["flips_deg"], offset_ppm=offset_ppm,
                         bandwidth_hz=suppression.COWS["pulse_bandwidth_hz"],
                         delays_s=delays, pulse_s=pulse if real else None,
                         truncation=suppression.COWS["pulse_truncation"], adjusted=False)


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
        crushers: True (the scanner's case) plays the pathway the crusher gradients around
            the pulses select, which dephases the own signal of every spin inside the
            editing band (an in-band singlet leaves the ON spectrum, so NAA at 2.01 ppm
            shows negative in the difference); False keeps every pathway, the convention of
            FID-A's sim_megapress, where an in-band singlet is refocused and stays. Without
            crushers a coupled partner's ON multiplet depends on the editing pulses' phase
            convention (stated).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm=1.88, taus_s=(0.005, 0.017, 0.017, 0.017, 0.012), bandwidth_hz=60.0,
                 tr_s=None, crushers=True):
        self.edit_ppm = edit_ppm
        self.taus_s = tuple(taus_s)
        self.bandwidth_hz = bandwidth_hz
        self.crushers = crushers
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
            *((CoherenceFilter(-1),) if crushers else ()),
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

    The editing bands here are ideal top hats: right for the edited targets, wrong on the
    flanks. They co-edit glutamate at 3.75 ppm too strongly (its 2.1 ppm partner lies
    inside the hat, where a Gaussian turns it in part) and leave out part of the GABA+
    (the macromolecules' partners on a Gaussian's skirts). Take the played scheme,
    `LocalizedMegaPress.scheme(Hermes, ...)`, where co-editing matters.

    Args:
        edit_ppm: Editing band centre(s) in ppm of one sub-experiment.
        taus_s: The five inter-pulse delays in seconds; their sum is the echo time.
        bandwidth_hz: Full selective bandwidth of each editing band.
        crushers: As MegaPress.
    """

    bands = {}
    combinations = {}

    #**********#
    #   init   #
    #**********#
    def __init__(self, edit_ppm, taus_s=(0.005, 0.015, 0.025, 0.015, 0.020), bandwidth_hz=60.0,
                 tr_s=None, crushers=True):
        super().__init__(edit_ppm=edit_ppm, taus_s=taus_s, bandwidth_hz=bandwidth_hz, tr_s=tr_s,
                         crushers=crushers)

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

    CPMG-style spacing, the pulses counted into the echo time: with tau an eighth of what
    te_s leaves beside the four pulses, tau either side of the chain and 2 tau between the
    refocusing pulses.

    Args:
        te_s: Echo time in seconds, at least the four pulses' durations.
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
                 tr_s=None, crushers=None):
        self.te_s = te_s
        self.afp_duration_s = afp_duration_s
        self.afp_bandwidth_hz = afp_bandwidth_hz
        self.afp_peak_hz = afp_peak_hz

        # one pulse object played four times: its propagator is computed once per run
        afp = ShapedPulse.hyperbolic_secant(afp_duration_s, afp_bandwidth_hz, afp_peak_hz,
                                            offset_ppm=4.65)
        tau = (te_s - 4.0 * afp_duration_s) / 8.0
        if tau < 0.0:
            raise ValueError(f"te_s {te_s} is too short for four {afp_duration_s * 1e3:.1f} ms "
                             "refocusing pulses")
        if crushers is None:
            events = (Pulse(90.0), CoherenceFilter(-1), Delay(tau),
                      afp, CoherenceFilter(+1), Delay(2.0 * tau),
                      afp, CoherenceFilter(-1), Delay(2.0 * tau),
                      afp, CoherenceFilter(+1), Delay(2.0 * tau),
                      afp, CoherenceFilter(-1), Delay(tau))
        else:
            # `crushers` plays a gradient pair around each of the four pulses in the filters'
            # place (see Press): four moments in cycles per millimetre over the 20 mm ensemble,
            # no sum of which with any signs vanishes, (0.15, 0.25, 0.35, 0.55) being 3, 5, 7,
            # and 11 cycles
            pairs = [Gradient(moment, 20.0) for moment in crushers]
            delays = (tau, 2.0 * tau, 2.0 * tau, 2.0 * tau, tau)
            events = [Pulse(90.0), Delay(delays[0])]
            for pair, delay in zip(pairs, delays[1:]):
                events += [pair, afp, pair, Delay(delay)]
        super().__init__(*events, tr_s=tr_s)
