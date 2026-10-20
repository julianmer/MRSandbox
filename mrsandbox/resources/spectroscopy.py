####################################################################################################
#                                          spectroscopy.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/09/26                                                                                #
#                                                                                                  #
# Purpose: The open spectroscopy datasets the scanner's rows are read off, each a cohort of its    #
#          runs: a local copy of the dataset in, one Cohort out, every run read through the        #
#          package's readers against one virtual twin of the dataset's acquisition, its subjects'  #
#          traits attached.                                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import dataclasses
import glob
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from mrsandbox.hardware import CoilArray
from mrsandbox.io.philips import DataList, SdatSpar
from mrsandbox.io.siemens import read_twix
from mrsandbox.params import Params
from mrsandbox.phantom import Brain
from mrsandbox.resources import participants
from mrsandbox.scan.calibrate import (CovarianceReading, NoiseReading, SpectrumReading,
                                      WaterReading, _raw)
from mrsandbox.scan.cohort import Cohort, Member
from mrsandbox.scan.profile import ScannerProfile
from mrsandbox.scan.run import RunReading
from mrsandbox.sequences import Press, SemiLaser

#***************#
#   constants   #
#***************#
# the twin's own room held calm (no spurious echoes, no shifts, no eddy spread, no leaked water),
# so a reading sets the run against its acquisition alone
CALM = Params(spurious=dict(amplitude=0.0),
              acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0),
              oov=dict(fraction=0.0))
PRESS_35 = Press(te1_s=0.011, te2_s=0.024, tr_s=2.0)


#************#
#   cohort   #
#************#
def _cohort(name, dataset, cited, twins, runs, people=None, forced=None, excitations=1,
            seeds=None, run_rows=True):
    """One cohort. `twins` {key: (preset, sequence, place, size)}, each the package's preset
    playing the sequence, ideal, in the ICBM head at the dataset's voxel; `runs` [(subject,
    twin key, load, load_reference)], each loader giving (x, dwell_s, f0_mhz, header), the
    reference's None where there is none; `forced` header entries the dataset's own
    documentation states over its files'; `excitations` how many excitations one stored
    transient sums (a number, or a function of the data and header), by which the noise
    reading carries the stored SNR to one excitation; `seeds` draws the twin fewer times
    per reading (a semi-LASER twin plays its adiabatic pulses slowly). Every run gives its
    profile (ScannerProfile.from_readings: the width, the noise, the channels' covariance
    against the helmet for 32 channels, the residual water, the delay, the leak, the eddy
    transient) and, with `run_rows`, the run's own rows (RunReading), cited `cited`
    (citation, url); the cohort is checked and its subjects' traits attached (`people`,
    else what participants states of the dataset)."""
    voxels = {key: ScannerProfile.preset(preset).scan(Brain.load("icbm152"), sequence,
                                                      params=CALM).voxel(place, size, compiled=True)
              for key, (preset, sequence, place, size) in twins.items()}
    citation, url = cited
    members = []
    for subject, key, load, load_reference in runs:
        x, dwell, f0, header = load()
        header = {**header, **(forced or {})}
        array = CoilArray.helmet() if x.shape[1] == 32 else None
        tail = np.moveaxis(x, 1, -1)[int(round(x.shape[0] * 0.875)):]
        water = (None if load_reference is None
                 else WaterReading.from_arrays(x, load_reference()[0], dwell, f0, header))
        profile = ScannerProfile.from_readings(
            SpectrumReading.from_arrays(x, dwell, f0, header),
            NoiseReading.from_arrays(x, dwell, f0, header, excitations=excitations(x, header)
                                     if callable(excitations) else excitations),
            voxels[key], dataset,
            covariance=None if array is None else CovarianceReading.from_arrays(tail, header),
            array=array, load="a head", water=water, seeds=seeds)
        records = tuple(dataclasses.replace(r, source={**r.source, "citation": citation,
                                                       "url": url}) for r in profile.records)
        if run_rows:
            # the field off the carrier
            records += RunReading.from_arrays(
                x, dwell, twins[key][1].tr_s,
                {**header, "MagneticFieldStrength": round(f0 / 42.5774785, 2)}).records(
                dataset, citation, url)
        _, _, place, size = twins[key]
        members.append(Member(subject, dataclasses.replace(profile, records=records),
                              tuple(place), tuple(size) if isinstance(size, tuple) else (size,) * 3,
                              "icbm152"))
    return participants.attach(Cohort(name, tuple(members)).checked(), people)


def _found(pattern):
    """The files a glob pattern finds ("**" any folders deep), sorted by their paths as text."""
    return sorted(glob.glob(str(pattern), recursive=True))


#**************************************************************************************************#
#                                           Class BigGABA                                          #
#**************************************************************************************************#
#                                                                                                  #
# Big GABA's PRESS arm: one parietal PRESS run per subject at 3 T on each vendor's systems.        #
#                                                                                                  #
#**************************************************************************************************#
class BigGABA:
    """Big GABA's PRESS arm: one parietal PRESS run per subject at 3 T on each vendor's systems.

    Source: Mikkelsen, M., et al. Big GABA: Edited MR spectroscopy at 24 research sites.
            NeuroImage 159 (2017) 32; the data on NITRC (www.nitrc.org/projects/biggaba).
    Data:   PRESS TE 35 ms, TR 2 s, a 30 mm parietal voxel; the Siemens sites (TrioTim,
            32-channel head array) as TWIX with an H2O reference each, the GE sites as P-files
            holding their water frames, the Philips sites as the scanner's own average (SDAT,
            coil-combined, 64 excitations) with its water reference.
    """

    CITED = ("Mikkelsen et al. 2017, NeuroImage 159:32", "https://www.nitrc.org/projects/biggaba/")
    TWIN = ((0.0, -35.0, 20.0), 30.0)

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root, vendor, folder=None):
        """The big_gaba_<vendor> cohort ("trio", "ge", "philips") from the dataset's local copy
        at `root`; the GE P-files are converted to NIfTI-MRS by spec2nii into `folder`
        (root/PRESS_ge_nifti unless given), and the files there are reused."""
        root = pathlib.Path(root)
        people = participants.big_gaba(root / "demographics.csv")
        if vendor == "trio":
            preset = "siemens_trio_3t"
            what = "Siemens sites S1, S5, S6 (each run with its H2O reference)"
            runs = [(f"{f.split('/')[-3]}/{f.split('/')[-2]}", (lambda f=f: read_twix(f)),
                     (lambda f=f: read_twix(f.replace("_PRESS_35.dat", "_PRESS_35_H2O.dat"))))
                    for f in _found(root / "PRESS" / "S*_P" / "S*" / "S*_PRESS_35.dat")]
            options = {}
        elif vendor == "ge":
            preset = "ge_mr750_3t"
            what = "GE sites G1, G4 to G8 (each run with its water reference)"
            tool = (shutil.which("spec2nii")
                    or str(pathlib.Path(sys.executable).parent / "spec2nii"))
            folder = pathlib.Path(folder) if folder is not None else root / "PRESS_ge_nifti"
            folder.mkdir(parents=True, exist_ok=True)
            runs = []
            for f in _found(root / "PRESS" / "G*_P" / "S*" / "*.7"):
                stem = folder / f"{f.split('/')[-3]}_{f.split('/')[-2]}"
                if not stem.with_name(f"{stem.name}.nii.gz").exists():
                    subprocess.run([tool, "ge", "-o", str(folder), "-f", stem.name, f],
                                   check=True, capture_output=True)
                runs.append((stem.name.replace("_P_", "/"),
                             (lambda s=stem: _raw(f"{s}.nii.gz")),
                             (lambda s=stem: _raw(f"{s}_ref.nii.gz"))))
            # 64 excitations stored in frames of 1, 2 or 8 (the model is not in the P-file)
            options = {"excitations": lambda x, header: 64 // x.shape[2]}
        else:
            preset = "philips_achieva_3t"
            what = ("Philips sites P1 to P10 (the scanner's averaged export with its water "
                    "reference)")
            runs = [(f"{f.split('/')[-3]}/{f.split('/')[-2]}", (lambda f=f: cls._sdat(f)),
                     (lambda f=f: cls._sdat(f.replace("_act.SDAT", "_ref.SDAT"))))
                    for f in _found(root / "PRESS" / "P*_P" / "S*" / "S*_PRESS_35_act.SDAT")]
            # one stored average: its statistics carried by the stated count, no run of its own
            options = {"excitations": lambda x, header: header["averages"], "run_rows": False}
        return _cohort(f"big_gaba_{vendor}", f"Big GABA, PRESS arm, {what}", cls.CITED,
                       {"parietal": (preset, PRESS_35) + cls.TWIN},
                       [(subject, "parietal", *loaders) for subject, *loaders in runs],
                       people, **options)

    @staticmethod
    def _sdat(path):
        """A Philips SDAT/SPAR pair as one stored average, (points, 1, 1), its dwell, carrier and
        header (the averages it sums)."""
        fids, spar = SdatSpar.load(path)
        header = {"Manufacturer": "Philips", "SequenceName": "PRESS",
                  "EchoTime": float(spar["spectrum_echo_time"]) * 1e-3,
                  "RepetitionTime": float(spar["repetition_time"]) * 1e-3,
                  "averages": int(spar["averages"])}
        return (fids.reshape(-1, 1, 1), 1.0 / float(spar["sample_frequency"]),
                float(spar["synthesizer_frequency"]) * 1e-6, header)


#**************************************************************************************************#
#                                            Class COWS                                            #
#**************************************************************************************************#
#                                                                                                  #
# COWS: semi-LASER at 3 T in three regions under three water suppression schemes.                  #
#                                                                                                  #
#**************************************************************************************************#
class COWS:
    """COWS: semi-LASER at 3 T in three regions under three water suppression schemes.

    Source: Igwe, K., et al. Constrained optimized water suppression for 1H MRS. Magn Reson Med
            (2025), doi 10.1002/mrm.30550; OpenNeuro ds006812 (CC0).
    Data:   ten healthy adults on a Siemens Prisma (32-channel head array), semi-LASER TE 26
            ms, TR 2 s, 20 mm voxels in the prefrontal, occipital and parietal cortex, each
            metabolite scan with its water reference stored in the same file (Phs 0).
    """

    CITED = ("Igwe et al. 2025, Magn Reson Med, doi 10.1002/mrm.30550",
             "https://openneuro.org/datasets/ds006812")
    # the voxels by region (the prefrontal files say PFL or PFC)
    PLACES = {"PF": (-1.0, 51.0, 18.0), "Occipital": (0.0, -75.0, 21.0),
              "Parietal": (0.0, 4.0, 50.0)}
    # a file of the deposit cut short (four bytes under its siblings), which mapVBVD does not read
    SHORT = ("sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat",)

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The cows_prisma cohort from the dataset's local copy at `root`: every subject's
        metabolite scans of every scheme and region, but a file cut short (SHORT)."""
        root = pathlib.Path(root)
        return _cohort(
            "cows_prisma", "COWS, OpenNeuro ds006812, the metabolite scans of all subjects, "
                           "schemes and regions (the water reference in the same file)",
            cls.CITED, {region: ("siemens_prisma_3t", SemiLaser(te_s=0.026, tr_s=2.0), place, 20.0)
                        for region, place in cls.PLACES.items()},
            [(f"{f.split('/')[-4]} {f.split('_svs_slaser_')[1][:-4].replace('_metab_', ' ')}",
              next(region for region in cls.PLACES if region in f),
              (lambda f=f: read_twix(f, phs=1)), (lambda f=f: read_twix(f, phs=0)))
             for f in _found(root / "sub-*" / "mrs" / "sourcedata" / "*_metab_*.dat")
             if pathlib.Path(f).name not in cls.SHORT],
            participants.bids(root, health="healthy"), seeds=3)


#**************************************************************************************************#
#                                          Class FMRSPain                                          #
#**************************************************************************************************#
#                                                                                                  #
# fMRS in pain: functional PRESS runs in the anterior cingulate at 3 T, raw per excitation.        #
#                                                                                                  #
#**************************************************************************************************#
class FMRSPain:
    """fMRS in pain: functional PRESS runs in the anterior cingulate at 3 T, raw per excitation.

    Source: Archibald, J., et al. Metabolite activity in the anterior cingulate cortex during a
            painful stimulus using functional MRS. Sci Rep 10 (2020) 19218; the data on NITRC
            (www.nitrc.org/projects/fmrs_2020).
    Data:   a Philips Achieva 3 T with a one-channel transmit-receive head coil, PRESS TE 22 ms,
            TR 4 s, a 25 x 30 x 15 mm voxel in the anterior cingulate, raw data/list files with
            the water reference as the second mix; the receiver's band, in neither the files nor
            the paper, is 2 kHz by the creatine to NAA spacing in the data (stated).
    """

    CITED = ("Archibald et al. 2020, Sci Rep 10:19218", "https://www.nitrc.org/projects/fmrs_2020")
    F0_MHZ, DWELL_S = 127.76, 1.0 / 2000.0

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The fmrs_pain_achieva cohort from the dataset's local copy at `root`: each subject's
        functional run (its largest list file) with its water reference."""
        header = {"Manufacturer": "Philips", "ManufacturersModelName": "Achieva",
                  "ReceiveCoilName": "transmit-receive head coil", "SequenceName": "PRESS",
                  "EchoTime": 0.022, "RepetitionTime": 4.0}

        def load(folder, mix):
            largest = max(pathlib.Path(folder).glob("*.list"),
                          key=lambda p: sum(1 for _ in open(p)))
            return DataList.load(largest).vectors("STD", mix), cls.DWELL_S, cls.F0_MHZ, header
        return _cohort(
            "fmrs_pain_achieva", "fMRS in pain, the functional runs (with their water reference)",
            cls.CITED, {"acc": ("philips_achieva_3t", Press(te1_s=0.009, te2_s=0.013, tr_s=4.0),
                                (0.0, 35.0, 20.0), (25.0, 30.0, 15.0))},
            [(folder[-3:], "acc", (lambda f=folder: load(f, 0)), (lambda f=folder: load(f, 1)))
             for folder in _found(pathlib.Path(root) / "SUBJECTS" / "RAW" / "GLUPI_S*")])


#**************************************************************************************************#
#                                     Class TissueSegmentation                                     #
#**************************************************************************************************#
#                                                                                                  #
# The tissue segmentation dataset: semi-LASER in the medial parietal cortex at 3 T, two sessions.  #
#                                                                                                  #
#**************************************************************************************************#
class TissueSegmentation:
    """The tissue segmentation dataset: semi-LASER in the medial parietal cortex at 3 T, two
    sessions.

    Source: The impact and reliability of tissue segmentation on in vivo MRS metabolite
            quantification, OpenNeuro ds006444 (CC0), doi 10.18112/openneuro.ds006444.v1.0.9.
    Data:   sixteen healthy adults (its README) on a GE Discovery MR750, 32-channel head array,
            semi-LASER TE 35 ms, TR 2 s, a 30 mm medial parietal voxel, two sessions each, as
            NIfTI-MRS with the transients apart and no water reference.
    """

    CITED = ("doi 10.18112/openneuro.ds006444.v1.0.9", "https://openneuro.org/datasets/ds006444")

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The ds006444_ge_mr750 cohort from the dataset's local copy at `root`."""
        root = pathlib.Path(root)
        return _cohort(
            "ds006444_ge_mr750", "OpenNeuro ds006444, the sLASER scans of both sessions (CC0)",
            cls.CITED, {"parietal": ("ge_mr750_3t", SemiLaser(te_s=0.035, tr_s=2.0),
                                     (0.0, -45.0, 35.0), 30.0)},
            [(f.split("/")[-1].split("_acq")[0], "parietal", (lambda f=f: _raw(f)), None)
             for f in _found(root / "**" / "*_acq-slaser_svs.nii.gz")],
            participants.bids(root, health="healthy"),
            forced={"ManufacturersModelName": "Discovery MR750", "ReceiveCoilName": "32Ch Head"},
            seeds=3)


#**************************************************************************************************#
#                                          Class Ketamine                                          #
#**************************************************************************************************#
#                                                                                                  #
# The NIMH ketamine study: J-suppressed PRESS in the pregenual cingulate at 7 T.                   #
#                                                                                                  #
#**************************************************************************************************#
class Ketamine:
    """The NIMH ketamine study: J-suppressed PRESS in the pregenual cingulate at 7 T.

    Source: NIMH Ketamine Mechanism of Action Study, OpenNeuro ds005917 (CC0), doi
            10.18112/openneuro.ds005917.v1.1.0.
    Data:   patients with major depressive or bipolar disorder and healthy controls (its group
            column) on a Siemens 7 T with a 32-channel head array, J-suppressed PRESS TE 69 ms,
            TR 2.5 s, a 20 mm pregenual anterior cingulate voxel, the baseline sessions as
            NIfTI-MRS; every 16th transient is unsuppressed water, the run's own reference.
    """

    CITED = ("doi 10.18112/openneuro.ds005917.v1.1.0", "https://openneuro.org/datasets/ds005917")

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The ds005917_siemens_7t cohort from the dataset's local copy at `root`."""
        root = pathlib.Path(root)
        return _cohort(
            "ds005917_siemens_7t", "OpenNeuro ds005917, NIMH ketamine mechanism of action study, "
                                   "the baseline sessions (CC0)",
            cls.CITED, {"pgacc": ("generic_7t", Press(te1_s=0.0345, te2_s=0.0345, tr_s=2.5),
                                  (0.0, 40.0, 8.0), 20.0)},
            [(f.split("/")[-1].split("_ses")[0], "pgacc", (lambda f=f: _raw(f)),
              (lambda f=f: _raw(f, water=True)))
             for f in _found(root / "**" / "*_ses-b0_svs.nii.gz")],
            participants.bids(root), forced={"ReceiveCoilName": "32Ch_Head_7T"})


#**************************************************************************************************#
#                                          Class NeuroMET                                          #
#**************************************************************************************************#
#                                                                                                  #
# NeuroMET: SPECIAL in the posterior cingulate at 7 T, raw TWIX.                                   #
#                                                                                                  #
#**************************************************************************************************#
class NeuroMET:
    """NeuroMET: SPECIAL in the posterior cingulate at 7 T, raw TWIX.

    Source: Riemann, L. T., et al. Assessment of measurement precision in single-voxel
            spectroscopy at 7 T. Magn Reson Med 87 (2022) 1119, doi 10.1002/mrm.29034; the data
            on Zenodo, record 5500320 (CC BY-NC-SA 4.0).
    Data:   nine healthy volunteers on a Siemens 7 T, SPECIAL with hyperbolic secant pulses, TE
            9 ms, TR 6.5 s, a 20 mm posterior cingulate voxel; the first hyperbolic secant scan of
            each subject's first session, p<k>_T1_1_HS.dat taken out of the deposit's archive.
    """

    CITED = ("Riemann et al. 2022, Magn Reson Med 87:1119, doi 10.1002/mrm.29034",
             "https://zenodo.org/record/5500320")

    #************#
    #   cohort   #
    #************#
    @classmethod
    def cohort(cls, root):
        """The neuromet_siemens_7t cohort from the nine files at `root`: the pairs of the phase
        cycle summed, each a stored transient of two excitations; no water reference."""
        def load(path):
            x, dwell, f0, header = read_twix(path)
            return x[:, :, 0::2] + x[:, :, 1::2], dwell, f0, header
        return _cohort(
            "neuromet_siemens_7t", "NeuroMET SPECIAL MRS reproducibility, the first hyperbolic "
                                   "secant scan of each subject's first session (CC BY-NC-SA 4.0)",
            cls.CITED, {"pcc": ("generic_7t", Press(te1_s=0.0045, te2_s=0.0045, tr_s=6.5),
                                (0.0, -45.0, 35.0), 20.0)},
            [(f"p{k}", "pcc", (lambda k=k: load(pathlib.Path(root) / f"p{k}_T1_1_HS.dat")), None)
             for k in range(1, 10)], excitations=2, run_rows=False)
