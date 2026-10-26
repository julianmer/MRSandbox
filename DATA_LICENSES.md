# The measured cohorts and their sources

The cohorts in `data/cohorts/` hold per-subject readings (noise, line width, residual water,
receiver delay, eddy transient, drift, jitters, relaxation, age, sex, health) that MRSandbox's
readers made from public datasets. No raw data is kept. Each cohort derives from the dataset below
under that dataset's terms, as the package's resource modules state them.

| cohort | dataset | terms |
|:--|:--|:--|
| cows_prisma | COWS, OpenNeuro ds006812 (Igwe et al. 2025) | CC0 |
| ds001168_siemens_7t | OpenNeuro ds001168 | CC0 |
| ds004611_prisma_3t | Developmental relaxometry, OpenNeuro ds004611 | CC0 |
| ds005917_siemens_7t | NIMH ketamine study, OpenNeuro ds005917 | CC0 |
| ds006444_ge_mr750 | OpenNeuro ds006444 | CC0 |
| ixi_guys_1p5t, ixi_hh_3t | IXI, Imperial College London | CC BY-SA 3.0 |
| radial_mese_prisma_3t | 3D radial multi-echo spin echo, Zenodo 10908743 | CC BY 4.0 |
| neuromet_siemens_7t | NeuroMET, Zenodo 5500320 | CC BY-NC-SA 4.0 |
| big_gaba_ge, big_gaba_philips, big_gaba_trio | Big GABA, NITRC (Mikkelsen et al. 2017 and 2019, NeuroImage; NIH R01 EB016089) | CC BY-NC 4.0 (the download agreement; cite both papers and the grant) |
| fmrs_pain_achieva | fMRS in pain, NITRC fmrs_2020 (Archibald et al. 2020, Sci Rep; doi 10.25790/bml0cm.159) | a Creative Commons licence, the variant not stated on NITRC (to confirm with the authors) |
| hmri_demo_prisma_3t | the hMRI toolbox demo dataset (github.com/hMRI-group/hMRI-toolbox-demodata) | CC BY 4.0 |
| uk7t_oxford_7t | the UK7T Network travelling head, Oxford (ora.ox.ac.uk, Clarke et al. 2020) | ODC-By |

Share-alike terms (IXI, NeuroMET) apply to those cohort files; the non-commercial terms of NeuroMET
and Big GABA apply to their cohort files.
