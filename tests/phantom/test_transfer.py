####################################################################################################
#                                         test_transfer.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Magnetization transfer gate: the bound pool's line is normalized as its saturation      #
#          rate asks, a design without saturation pulses takes nothing, a water suppression train  #
#          takes from a coupled metabolite as much more as it couples and nothing from an          #
#          uncoupled one, and a voxel without a bound pool takes nothing.                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox import Brain, Engine, Params, Press, Scanner, VirtualScan
from mrsandbox.phantom.transfer import Transfer, superlorentzian
from mrsandbox.phantom.water import WaterSource
from mrsandbox.sequences.library import Vapor

GAMMA = 42.5774785


#**************************************************************************************************#
#                                        Class TestTransfer                                        #
#**************************************************************************************************#
class TestTransfer:

    #****************#
    #   line shape   #
    #****************#
    def test_the_bound_pools_line_is_normalized_over_angular_frequency(self):
        # W = pi w1^2 G asks a line normalized over w: 1 / (2 pi) over Hz; it falls off the centre
        offsets = np.linspace(-2e6, 2e6, 200001)
        line = np.concatenate([superlorentzian(part, 9.1e-6, floor_hz=0.0)
                               for part in np.array_split(offsets, 50)])
        assert abs(np.trapezoid(line, offsets) * 2.0 * np.pi - 1.0) < 0.01
        assert superlorentzian(1e3, 9.1e-6) > superlorentzian(5e3, 9.1e-6) > 0.0

    #**************#
    #   factors    #
    #**************#
    def test_a_train_takes_from_a_coupled_metabolite_only(self):
        engine, water = Engine(3.0), WaterSource.system(gamma_mhz_t=GAMMA)
        press = Press(te1_s=0.015, te2_s=0.015, tr_s=2.0)
        model = Transfer(0.05, 40.0)
        rates, t1 = {"Cr": 0.0, "PCr": 0.2, "NAA": 0.4}, {"Cr": 1.3, "PCr": 1.3, "NAA": 1.3}
        # nothing saturates: nothing is taken
        alone = model.factors(engine, press, water, 1.6, rates, t1)
        assert all(abs(value - 1.0) < 1e-9 for value in alone.values())
        # the water saturated before the excitation: the faster a metabolite couples, the more it
        # loses; the uncoupled one keeps all, and the water is the train's (no factor of its own)
        trained = model.factors(engine, Vapor() + press, water, 1.6, rates, t1)
        assert trained["Cr"] == 1.0 and 1.0 > trained["PCr"] > trained["NAA"] > 0.6
        assert trained["water"] == 1.0

    #************#
    #   voxel    #
    #************#
    def test_a_voxel_without_a_bound_pool_takes_nothing(self):
        brain = Brain.from_maps(gm=np.ones((20, 20, 20)), wm=np.zeros((20, 20, 20)),
                                csf=np.zeros((20, 20, 20)), affine=np.diag([2.0, 2.0, 2.0, 1.0]))
        design = Vapor() + Press(te1_s=0.015, te2_s=0.015, tr_s=2.0)
        place = (20.0, 20.0, 20.0)
        scan = VirtualScan(brain, Scanner.preset("generic_3t"), design)
        voxel = scan.voxel(place, 10.0)
        taken = voxel.transfer(voxel.sample(seed=1))
        assert taken["Cr"] < 1.0 and taken["NAA"] == 1.0
        empty = VirtualScan(brain, Scanner.preset("generic_3t"), design,
                            params=Params(mt=dict(pool_fraction=0.0))).voxel(place, 10.0)
        assert empty.transfer(empty.sample(seed=1)) == {}
