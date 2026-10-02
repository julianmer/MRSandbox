####################################################################################################
#                                           transfer.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: Published magnetization transfer numbers: the brain's bound (semi-solid) pool at 3 T    #
#          and the rates at which metabolites exchange magnetization with the water, one dict per  #
#          source, stored as published with the citation beside each.                              #
#                                                                                                  #
####################################################################################################

#****************#
#   bound pool   #
#****************#
# Source: Stanisz, G. J., Odrobina, E. E., Pun, J., Escaravage, M., Graham, S. J., Bronskill,
#         M. J., Henkelman, R. M. T1, T2 relaxation and magnetization transfer in tissue at 3T.
#         MRM 54 (2005) 507-512, as the pulseq-CEST library transcribes it
#         (github.com/kherz/pulseq-cest-library, sim-library/Stanisz_3T, the 2-pool files).
# Units:  pool_fraction the bound pool's size over the free water's (M0B / M0A); exchange_hz
#         its exchange rate toward the water pool, 1/s; t2_s its transverse relaxation time
#         (super-Lorentzian); its R1 the library's assumed 1/s.
STANISZ_2005_3T = {
    "gm": {"pool_fraction": 0.050, "exchange_hz": 40.0, "t2_s": 9.1e-6},
    "wm": {"pool_fraction": 0.139, "exchange_hz": 23.0, "t2_s": 10.0e-6},
    "r1_s": 1.0,
}

#*****************************#
#   water to metabolite rates  #
#*****************************#
# Source: MacMillan, E. L., Boesch, C., Kreis, R. Magnetization exchange observed in human
#         skeletal muscle by non-water-suppressed proton magnetic resonance spectroscopy. MRM 70
#         (2013) 916-924: magnetization transfer rates from water to the creatine methyl
#         resonance of 0.39 (soleus) and 0.30 s-1 (tibialis anterior), 3 T, metabolite cycled;
#         and Roell, S. A., Dreher, W., Leibfritz, D. MRM 42 (1999) 222-227: an exchange rate of
#         0.36 +- 0.04 s-1 for total creatine in rat brain (their two-pool model).
# Units:  1/s, the range the two muscles span.
# Notes:  muscle and rat, not human brain. NAA, NAAG and the choline compounds show no transfer
#         in human brain, so their rates are 0: Kruiskamp, M. J., de Graaf, R. A., van der Grond,
#         J., Lamerichs, R., Nicolay, K. NMR Biomed 14 (2001) 1-4 (inversion transfer, 1.5 T:
#         creatine coupled, choline and NAA not); Helms, G., Frahm, J. NMR Biomed 12 (1999)
#         490-494 (off-resonance saturation, 2.0 T: creatine 0.87 +- 0.05, NAA, NAAG, Glu, Gln,
#         choline and mI unattenuated); Meyerhoff, D. J. MRM 42 (1999) 417-420 (creatine 12 to
#         18 percent attenuated, NAA and choline not).
CREATINE_METHYL_RATE_HZ = (0.30, 0.39)
NO_TRANSFER = ("NAA", "NAAG", "PCho", "GPC")
