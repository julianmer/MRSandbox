####################################################################################################
#                                              fmrs.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/11/26                                                                                #
#                                                                                                  #
# Purpose: What a task does to the metabolite levels and to the lines over a functional MRS run,   #
#          one dict per source with the citation beside it: the fractional changes at the plateau   #
#          of a stimulus, how fast they come and go, and the BOLD effect on the line widths; the    #
#          priors the tree draws from them at the end (scan.defaults fmrs, scan.paradigm).         #
#                                                                                                  #
####################################################################################################

#*****************#
#   mangia 2007   #
#*****************#
# Source: Mangia, S., Tkac, I., Gruetter, R., Van de Moortele, P. F., Maraviglia, B., Ugurbil, K.
#         Sustained neuronal activation raises oxidative metabolism to a new steady-state level:
#         evidence from 1H NMR spectroscopy in the human visual cortex. J Cereb Blood Flow Metab 27
#         (2007) 1055-1063, doi 10.1038/sj.jcbfm.9600401: 7 T, visual stimulation of 5.3 and 10.6
#         min; the changes as percent of the level (mean, sd); lactate at its new steady state
#         within the first minute of activation, back to baseline only after the stimulus ended.
MANGIA_2007 = {"field_t": 7.0, "stimulus": "visual", "change": {"Lac": (0.23, 0.05),
                                                                "Glu": (0.03, 0.01),
                                                                "Asp": (-0.15, 0.06)},
               "lactate_steady_within_s": 60.0}

#*******************#
#   bednarik 2015   #
#*******************#
# Source: Bednarik, P., Tkac, I., Giove, F., DiNuzzo, M., Deelchand, D. K., Emir, U. E., Eberly,
#         L. E., Mangia, S. Neurochemical and BOLD responses during neuronal activation measured
#         in the human visual cortex at 7 Tesla. J Cereb Blood Flow Metab 35 (2015) 601-610,
#         doi 10.1038/jcbfm.2014.233: 12 volunteers, a block design of visual stimulation; the
#         changes in umol/g (mean, sd) and as the paper's percent of the level.
BEDNARIK_2015 = {"field_t": 7.0, "stimulus": "visual", "subjects": 12,
                 "change_umol_g": {"Lac": (0.26, 0.06), "Glu": (0.28, 0.03), "Asp": (-0.20, 0.04),
                                   "Glc": (-0.19, 0.03)},
                 "change": {"Lac": 0.30, "Glu": 0.03, "Asp": -0.05, "Glc": -0.16}}

#**************#
#   lin 2012   #
#**************#
# Source: Lin, Y., Stephenson, M. C., Xin, L., Napolitano, A., Morris, P. G. Investigating the
#         metabolic changes due to visual stimulation using functional proton magnetic resonance
#         spectroscopy at 7 T. J Cereb Blood Flow Metab 32 (2012) 1484-1495,
#         doi 10.1038/jcbfm.2012.33: single and repeated visual stimulation; increases of 2 to 11
#         percent in lactate, glutamate and glutathione, decreases in aspartate, glutamine and
#         glycine; glutathione's course like glutamate's.
LIN_2012 = {"field_t": 7.0, "stimulus": "visual", "increase_range": (0.02, 0.11),
            "increased": ("Lac", "Glu", "GSH"), "decreased": ("Asp", "Gln", "Gly")}

#****************#
#   koush 2022   #
#****************#
# Source: Koush, Y., Rothman, D. L., Behar, K. L., de Graaf, R. A., Hyder, F. Human brain
#         functional MRS reveals interplay of metabolites implicated in neurotransmission and
#         neuroenergetics. J Cereb Blood Flow Metab 42 (2022) 911-934,
#         doi 10.1177/0271678X221076570: the ranges across the visual and motor studies it
#         reviews, as fractions of the level (GABA from the one visual study reporting it and a
#         second reporting a 5 percent decrease beside a 7 percent lactate rise).
KOUSH_2022 = {"range": {"Glu": (0.015, 0.05), "Lac": (0.09, 0.30), "Asp": (-0.15, -0.05),
                        "Glc": (-0.16, -0.12), "GABA": (-0.06, -0.05)}}

#********************#
#   archibald 2020   #
#********************#
# Source: Archibald, J., MacMillan, E. L., Graf, C., Kozlowski, P., Laule, C., Kramer, J. L. K.
#         Metabolite activity in the anterior cingulate cortex during a painful stimulus using
#         functional MRS. Sci Rep 10 (2020) 19218, doi 10.1038/s41598-020-76263-3: 3 T Philips
#         Achieva, PRESS TE 22 ms TR 4 s, the anterior cingulate, 15 volunteers, capsaicin and heat
#         on the forearm from 9 min for 4.4 min of a 22 min run; the change at the onset of pain
#         against the block before it (fraction, Cohen's d), transient, no sustained rise through
#         the tonic heat. The package's reading of the deposited runs: scan.calibrate TaskReading.
ARCHIBALD_2020 = {"field_t": 3.0, "stimulus": "heat pain", "subjects": 15,
                  "onset_change": {"Glu": (0.0838, 0.74), "Glx": (0.0725, 0.60),
                                   "tCr": (0.060, 0.83), "NAA": (0.0281, 0.28)},
                  "sustained": False, "paradigm_s": {"rest": 540.0, "task": 264.0}}

#*******************#
#   zhu chen 2001   #
#*******************#
# Source: Zhu, X. H., Chen, W. Observed BOLD effects on cerebral metabolite resonances in human
#         visual cortex during visual stimulation: a functional 1H MRS study at 4 T. Magn Reson
#         Med 46 (2001) 841-847, doi 10.1002/mrm.1267: the line widths narrow and the peak
#         heights rise with the BOLD effect, as fractions.
ZHU_CHEN_2001 = {"field_t": 4.0, "linewidth_change": {"NAA": -0.017, "Cr": -0.018,
                                                      "water": -0.023},
                 "height_change": {"NAA": 0.025, "Cr": 0.031, "water": 0.031}}

#**************#
#   priors     #
#**************#
# what the tree draws (scan.defaults fmrs): each metabolite's fractional change at the plateau of
# a stimulus, Koush et al.'s ranges over the 7 T visual and motor studies (stated for any
# stimulus and field); the rise and decay time constants, stated ranges about Mangia et al.'s
# lactate steady within the first minute and back to baseline over minutes after the stimulus;
# the BOLD narrowing as a fraction of the lines' width (Zhu and Chen's NAA and creatine at 4 T,
# the field dependence left out, stated) and the heights' rise beyond it, stated
EFFECTS = dict(KOUSH_2022["range"])
RISE_S, DECAY_S = (20.0, 60.0), (30.0, 180.0)
BOLD_NARROWING, BOLD_AMPLITUDE = (0.017, 0.023), (0.0, 0.01)
CITED = {"effect": "Koush et al. 2022, J Cereb Blood Flow Metab 42:911: the range over the 7 T "
                   "visual and motor studies (Mangia 2007, Lin 2012, Bednarik 2015 among them)",
         "time": "Mangia et al. 2007, J Cereb Blood Flow Metab 27:1055: lactate steady within "
                 "the first minute, back after the stimulus; the ranges stated",
         "bold": "Zhu and Chen 2001, Magn Reson Med 46:841: the line widths 1.7 to 2.3 percent "
                 "narrower at 4 T; the heights' rise beyond it stated"}
