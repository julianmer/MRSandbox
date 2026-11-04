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
#         (2007) 1055-1063, doi 10.1038/sj.jcbfm.9600401: 7 T STEAM TE 6 ms TR 5 s, 12 volunteers,
#         flickering checkerboard in blocks of 5.3 min on and off twice and of 10.6 min; the
#         changes as fractions of the level (mean, sd; Table 1 in umol/g), lactate at its new
#         steady state within the first minute (Figure 2), glutamate delayed by minutes and
#         aspartate its mirror (Figure 3), glucose falling through the stimulus without a plateau;
#         back to baseline over about three minutes after the stimulus; the BOLD effect raised the
#         creatine height about 2 percent through the stimulus (Figure 2B) and a 0.4 Hz broadening
#         matched the stimulus spectrum's lines to rest (Figure 4).
MANGIA_2007 = {"field_t": 7.0, "stimulus": "visual", "subjects": 12,
               "blocks_s": (318.0, 636.0),
               "change": {"Lac": (0.23, 0.05), "Glu": (0.03, 0.01), "Asp": (-0.15, 0.06),
                          "Glc": (-0.12, 0.07)},
               "change_umol_g": {"Lac": (0.24, 0.01), "Glu": (0.23, 0.01), "Asp": (-0.14, 0.02),
                                 "Glc": (-0.13, 0.02)},
               "lactate_steady_within_s": 60.0, "glutamate_delayed": True,
               "back_within_s": 180.0, "bold_narrowing_hz": 0.4, "bold_height_rise": 0.02}

#**************#
#   lin 2012   #
#**************#
# Source: Lin, Y., Stephenson, M. C., Xin, L., Napolitano, A., Morris, P. G. Investigating the
#         metabolic changes due to visual stimulation using functional proton magnetic resonance
#         spectroscopy at 7 T. J Cereb Blood Flow Metab 32 (2012) 1484-1495,
#         doi 10.1038/jcbfm.2012.33: 7 T STEAM TE 15 ms TR 3 s, 10 volunteers, moving wedges; one
#         stimulus of 13.2 min after 6.6 min of rest and 19.8 min of recovery, and two of 9.9 min
#         after rests of 9.9 min; Table 1's changes as fractions (mean, sd): the single stimulus
#         over its whole length, lactate +0.26 +-0.07 in its first 6.6 min and falling toward
#         baseline while the stimulus went on, no lactate rise to the second stimulus; glutamate
#         and glutathione recovering slowly, still raised 9.9 min after; the BOLD narrowing about
#         0.5 Hz at 7 T.
LIN_2012 = {"field_t": 7.0, "stimulus": "visual", "subjects": 10,
            "blocks_s": (792.0, 594.0),
            "change": {"Glu": (0.02, 0.01), "Asp": (-0.09, 0.08), "Lac": (0.10, 0.06),
                       "GSH": (0.07, 0.04), "Gln": (-0.08, 0.05), "Gly": (-0.11, 0.06),
                       "Glc": (-0.22, 0.18)},
            "lactate_first_block": (0.26, 0.07), "lactate_peak_within_s": 99.0,
            "lactate_transient": True, "bold_narrowing_hz": 0.5}

#*******************#
#   bednarik 2015   #
#*******************#
# Source: Bednarik, P., Tkac, I., Giove, F., DiNuzzo, M., Deelchand, D. K., Emir, U. E., Eberly,
#         L. E., Mangia, S. Neurochemical and BOLD responses during neuronal activation measured
#         in the human visual cortex at 7 Tesla. J Cereb Blood Flow Metab 35 (2015) 601-610,
#         doi 10.1038/jcbfm.2014.233: 7 T semi-LASER TE 26 ms TR 5 s, 12 volunteers, two blocks of
#         5.3 min between rests of 5.3 min; Table 1 after the line widths were matched, as
#         fractions of the level (mean) and in umol/g (mean, sem); lactate and glutamate at their
#         plateaus within about a minute and two (Figure 2), back within about two after the
#         stimulus; the creatine line 0.46 Hz narrower through the stimulus (the water's 0.1 to
#         1.0 Hz per subject, with the BOLD-fMRI amplitude 2.4 +-1.2 percent).
BEDNARIK_2015 = {"field_t": 7.0, "stimulus": "visual", "subjects": 12, "blocks_s": (318.0,),
                 "change": {"Glu": 0.033, "Lac": 0.296, "Asp": -0.054, "Glc": -0.160,
                            "GABA": 0.037},
                 "change_umol_g": {"Glu": (0.28, 0.03), "Lac": (0.26, 0.06), "Asp": (-0.20, 0.04),
                                   "Glc": (-0.19, 0.03)},
                 "plateau_within_s": {"Lac": 60.0, "Glu": 120.0}, "back_within_s": 120.0,
                 "bold_narrowing_hz": 0.46, "bold_fmri": (0.024, 0.012)}

#****************#
#   koush 2022   #
#****************#
# Source: Koush, Y., Rothman, D. L., Behar, K. L., de Graaf, R. A., Hyder, F. Human brain
#         functional MRS reveals interplay of metabolites implicated in neurotransmission and
#         neuroenergetics. J Cereb Blood Flow Metab 42 (2022) 911-934,
#         doi 10.1177/0271678X221076570: the ranges over the 7 T visual studies (page 917), the
#         other stimuli and fields of its Table 1 as fractions, and the J-difference reading of a
#         flashing checkerboard at 4 T (Figure 2, mean, sd).
KOUSH_2022 = {"visual_7t": {"Glu": (0.015, 0.05), "Lac": (0.09, 0.30), "Asp": (-0.15, -0.05),
                            "Glc": (-0.16, -0.12), "GABA": (-0.06, -0.06)},
              "motor_7t": {"Lac": 0.17, "Glu": 0.02}, "motor_learning_gaba": -0.20,
              "visual_3t": {"Glu": 0.12, "Lac": 0.50},
              "pain_acc_3t": {"Glx": 0.22, "GABA": -0.15}, "pain_insula_3t": {"Glu": 0.18},
              "stroop_dacc": {"Glu": 0.03},
              "visual_4t_edited": {"GABA": (-0.057, 0.033), "Glx": (0.030, 0.021),
                                   "Lac": (0.078, 0.054)}}

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
#         Med 46 (2001) 841-847, doi 10.1002/mrm.1267: 4 T PRESS TE 23 ms, six subjects, a
#         checkerboard in blocks of 6.4 and 12.8 min; Table 1's heights and widths as fractions
#         (mean, sem), the widths 0.2 to 0.3 Hz narrower (Figure 4), the course following the
#         stimulus within tens of seconds (Figure 4).
ZHU_CHEN_2001 = {"field_t": 4.0, "stimulus": "visual", "subjects": 6, "blocks_s": (384.0, 768.0),
                 "linewidth_change": {"NAA": (-0.017, 0.005), "Cr": (-0.018, 0.005),
                                      "water": (-0.023, 0.003)},
                 "height_change": {"NAA": (0.025, 0.006), "Cr": (0.031, 0.007),
                                   "water": (0.030, 0.004)},
                 "narrowing_hz": (0.2, 0.3)}

#**************#
#   priors     #
#**************#
# what the tree draws (scan.defaults fmrs): each metabolite's fractional change at the plateau of
# a stimulus, the ranges Koush et al. review over the 7 T visual studies, Lin et al.'s
# glutathione, glutamine and glycine beside them (stated for any stimulus and field); the rise
# and decay time constants per metabolite, lactate's about its rise within the first minute and
# its return over one to three (Mangia, Lin, Bednarik), glutamate's and glutathione's slower
# courses, the others lactate's (stated); the BOLD narrowing in hertz per tesla (0.2 to 0.3 Hz
# at 4 T, 0.4 to 0.5 at 7 T), its own course within tens of seconds (stated about Zhu and Chen's
# Figure 4) and the heights' rise beyond the narrowing (Zhu and Chen's heights less widths)
EFFECTS = {**KOUSH_2022["visual_7t"], "GSH": (0.07, 0.11), "Gln": (-0.08, -0.06),
           "Gly": (-0.12, -0.11)}
RISE_S = {"Lac": (20.0, 40.0), "Glu": (60.0, 120.0), "GSH": (60.0, 120.0)}
DECAY_S = {"Lac": (40.0, 90.0), "Glu": (120.0, 300.0), "GSH": (120.0, 300.0)}
RISE_OTHERS_S, DECAY_OTHERS_S = (20.0, 40.0), (40.0, 90.0)
BOLD_HZ_PER_T, BOLD_S, BOLD_AMPLITUDE = (0.055, 0.075), (5.0, 15.0), (0.005, 0.010)
CITED = {"effect": "Koush et al. 2022, J Cereb Blood Flow Metab 42:911 (the 7 T visual studies' "
                   "ranges); Lin et al. 2012, J Cereb Blood Flow Metab 32:1484 (glutathione, "
                   "glutamine, glycine)",
         "time": "Mangia et al. 2007, Lin et al. 2012, Bednarik et al. 2015 (J Cereb Blood Flow "
                 "Metab 27:1055, 32:1484, 35:601): lactate within the first minute and back over "
                 "one to three, glutamate over minutes; the ranges stated",
         "bold": "Zhu and Chen 2001, Magn Reson Med 46:841 (0.2 to 0.3 Hz at 4 T); Mangia 2007, "
                 "Lin 2012, Bednarik 2015 (0.4 to 0.5 Hz at 7 T); the course and the heights' "
                 "rise beyond the narrowing stated"}
