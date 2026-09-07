####################################################################################################
#                                        test_gradients.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 06/09/26                                                                                #
#                                                                                                  #
# Purpose: Gradient gate: a single crusher dephases by exactly sinc(q k L), a balanced pair        #
#          around a perfect refocusing pulse reproduces the idealized coherence mask, an           #
#          imperfect pulse leaves the physical spurious FID a mask cannot show, and strong         #
#          crushers suppress that artifact the way scanners rely on.                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest
from keras import ops

from mrsandbox import CoherenceFilter, Delay, Engine, Gradient, Pulse, Sequence, SpinSystem
from mrsandbox.phantom.water import WaterSource

TE = 0.03


#*************#
#   helpers   #
#*************#
@pytest.fixture(scope="module")
def engine():
    return Engine(3.0)


def amps(engine, sequence, system, b1=1.0):
    _, a = engine.lines(sequence, system, b1_scale=b1, isochromats=64)
    return np.asarray(ops.convert_to_numpy(a))


def masked_echo():
    return Sequence(Pulse(90.0), CoherenceFilter(+1), Delay(TE / 2),
                    Pulse(180.0, phase_deg=90.0), CoherenceFilter(-1), Delay(TE / 2))


def crushed_echo(moment):
    crusher = Gradient(moment, 20.0)
    return Sequence(Pulse(90.0), Delay(TE / 2), crusher,
                    Pulse(180.0, phase_deg=90.0), crusher, Delay(TE / 2))


#**************************************************************************************************#
#                                        Class TestGradient                                        #
#**************************************************************************************************#
class TestGradient:

    #****************#
    #   sinc law     #
    #****************#
    def test_single_crusher_follows_the_sinc_law(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        full = np.abs(amps(engine, Sequence(Pulse(90.0)), water)).sum()
        for kl in (0.3, 0.7, 1.5):
            crushed = np.abs(amps(engine, Sequence(Pulse(90.0), Gradient(kl, 1.0)), water)).sum()
            assert abs(crushed / full - abs(np.sinc(kl))) < 1e-3

    #**********************#
    #   pair equals mask   #
    #**********************#
    def test_balanced_pair_reproduces_the_coherence_mask(self, engine):
        for system in SpinSystem.from_database("Lac"):
            a = amps(engine, masked_echo(), system)
            b = amps(engine, crushed_echo(3.7), system)
            corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert corr > 0.99999
            assert abs(np.abs(b).sum() / np.abs(a).sum() - 1.0) < 1e-3

    #********************#
    #   spurious fid     #
    #********************#
    def test_imperfect_pulse_leaves_the_physical_spurious_fid(self, engine):
        # with a miscalibrated transmit both pulses err: longitudinal magnetization
        # survives the excitation and the imperfect 180 tips it into a spurious FID
        # that only the second crusher touches; a weak crusher shows it, the mask never
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        ideal = np.abs(amps(engine, masked_echo(), water, b1=0.6)).sum()
        weak = np.abs(amps(engine, crushed_echo(0.02), water, b1=0.6)).sum()
        strong = np.abs(amps(engine, crushed_echo(3.7), water, b1=0.6)).sum()
        assert weak > 1.1 * ideal  # the artifact exists and the mask cannot show it
        assert abs(strong / ideal - 1.0) < 0.01  # strong crushers suppress it, as on scanners


#**************************************************************************************************#
#                                     Class TestSliceSelection                                     #
#**************************************************************************************************#
#                                                                                                  #
# A shaped pulse under a gradient excites a slab; the refocus lobe rephases it; off-resonance      #
# spins are displaced (chemical shift displacement).                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestSliceSelection:

    #*************#
    #   helpers   #
    #*************#
    @staticmethod
    def fraction(engine, gradient_hz_mm, system, refocus=-0.5, duration=0.004):
        from mrsandbox import ShapedPulse

        pulse = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65,
                                     gradient_hz_per_mm=gradient_hz_mm, extent_mm=20.0)
        lobe = Gradient(refocus * gradient_hz_mm * duration, 20.0)
        sliced = np.abs(amps(engine, Sequence(pulse, lobe, CoherenceFilter(-1)), system)).sum()
        plain = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65)
        full = np.abs(amps(engine, Sequence(plain, CoherenceFilter(-1)), system)).sum()
        return sliced / full

    #*****************#
    #   slab + lobe   #
    #*****************#
    def test_slab_fraction_and_refocus_lobe(self, engine):
        water = WaterSource.system(gamma_mhz_t=42.5774785)
        got = self.fraction(engine, 100.0, water)
        # gaussian R = 2.70: bandwidth 675 Hz over 100 Hz/mm = 6.75 mm of the 20 mm voxel
        assert abs(got - 2.70 / 0.004 / 100.0 / 20.0) < 0.05
        assert self.fraction(engine, 100.0, water, refocus=0.0) < 0.2 * got  # no lobe, dephased

    #***********************************#
    #   chemical shift displacement     #
    #***********************************#
    def test_chemical_shift_displaces_the_slab(self, engine):
        """The slab centre of an off-carrier spin sits at (carrier - nu) / g: the NAA
        acetyl singlet, 337 Hz from the water carrier, must land 11 mm away."""
        import dataclasses

        from mrsandbox import ShapedPulse

        g, duration = 30.0, 0.004
        pulse = ShapedPulse.gaussian(90.0, duration, offset_ppm=4.65,
                                     gradient_hz_per_mm=g, extent_mm=20.0)
        sequence = Sequence(pulse, Gradient(-0.5 * g * duration, 20.0), CoherenceFilter(-1))
        positions = np.linspace(-30.0, 30.0, 61)

        def slab_centre(system):
            ctx = engine.context(system)
            profile = []
            for x in positions:
                _, a = engine._system_lines(
                    sequence, dataclasses.replace(ctx, gradient_x=float(x)))
                profile.append(float(np.abs(ops.convert_to_numpy(a)).sum()))
            profile = np.asarray(profile)
            return float((positions * profile).sum() / profile.sum())

        water = WaterSource.system(gamma_mhz_t=42.5774785)
        (acetyl,) = [s for s in SpinSystem.from_database("NAA") if "acetyl" in s.name.lower()]
        displacement = slab_centre(acetyl) - slab_centre(water)
        expected = (4.65 - 2.008) * 127.73 / g
        assert abs(displacement - expected) < 1.0  # millimetres
