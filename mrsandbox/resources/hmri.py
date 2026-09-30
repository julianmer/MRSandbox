####################################################################################################
#                                             hmri.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/09/26                                                                                #
#                                                                                                  #
# Purpose: The hMRI toolbox demonstration subject: one healthy volunteer, one 3 T Prisma session,  #
#          raw multi-echo FLASH (PDw, T1w, MTw), an SE/STE B1+ series, and a dual-echo B0 map,     #
#          registered as fetch-on-demand resources, and the quantitative maps derived from them    #
#          once in cited numpy (the hMRI toolbox formulas, never the toolbox) and cached: the      #
#          coherent one-subject 3 T stack of R1, PD, MT saturation, R2*, relative B1+, and B0.     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import itertools

import numpy as np

from mrsandbox.phantom.volume import Volume


#******************#
#   hmri demo      #
#******************#
# Source: Callaghan, M. F., et al. Example dataset for the hMRI toolbox. Data in Brief 25
#         (2019) 104132. doi:10.1016/j.dib.2019.104132.
# Data:   https://github.com/hMRI-group/hMRI-toolbox-demodata (raw-data), CC BY 4.0. The
#         repository publishes no checksums; the md5 sums below were computed on 16/09/26
#         from the downloaded files. One subject, Siemens Prisma 3 T, 64-channel head coil.
# Units:  NIfTI uint16 magnitudes (arbitrary), 0.8 mm MPM on an oblique grid, 4 mm B1+
#         series, 3 x 3 x 2 mm B0 pair; the B0 phase difference is stored scaled.
BASE = "https://raw.githubusercontent.com/hMRI-group/hMRI-toolbox-demodata/master/raw-data/"
HMRI_DEMO = {
    "hmri/pdw_e1": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-00224-1.nii",
        "md5": "c21e5b3580522351999e6e2d4ff2c13b",
        "filename": "hmri/pdw_e1.nii",
    },
    "hmri/pdw_e2": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-00448-2.nii",
        "md5": "a754fdf72af90213316413b2114c1f6f",
        "filename": "hmri/pdw_e2.nii",
    },
    "hmri/pdw_e3": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-00672-3.nii",
        "md5": "9c798fa1e72fb62dcfa62f096a0e6882",
        "filename": "hmri/pdw_e3.nii",
    },
    "hmri/pdw_e4": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-00896-4.nii",
        "md5": "c98daffc171bb6c540ffd326e1cadc38",
        "filename": "hmri/pdw_e4.nii",
    },
    "hmri/pdw_e5": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-01120-5.nii",
        "md5": "ca7bbd47aa27938ec664c5836b246aa1",
        "filename": "hmri/pdw_e5.nii",
    },
    "hmri/pdw_e6": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-01344-6.nii",
        "md5": "3c2c443bd714cd2800767c8b55d68ef2",
        "filename": "hmri/pdw_e6.nii",
    },
    "hmri/pdw_e7": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-01568-7.nii",
        "md5": "a8a2415a0fcd53ca1ccac32d2644d4f1",
        "filename": "hmri/pdw_e7.nii",
    },
    "hmri/pdw_e8": {
        "url": BASE + "pdw_mfc_3dflash_v1i_R4_0009/anon_s2018-02-28_18-26-185345-00001-01792-8.nii",
        "md5": "5c00b16f1c52750f2951926c4a5f4078",
        "filename": "hmri/pdw_e8.nii",
    },
    "hmri/t1w_e1": {
        "url": BASE + "t1w_mfc_3dflash_v1i_R4_0015/anon_s2018-02-28_18-26-190921-00001-00224-1.nii",
        "md5": "050f75469f79d4b4f197f8376d107a2c",
        "filename": "hmri/t1w_e1.nii",
    },
    "hmri/mtw_e1": {
        "url": BASE + "mtw_mfc_3dflash_v1i_R4_0012/anon_s2018-02-28_18-26-190132-00001-00224-1.nii",
        "md5": "ffc7424b53694f5f1a226172eca8767f",
        "filename": "hmri/mtw_e1.nii",
    },
    "hmri/b1_se_01": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184837-00001-00001-1.nii",
        "md5": "7d8a2b42d444ceefccf4338ac09f6439",
        "filename": "hmri/b1_se_01.nii",
    },
    "hmri/b1_ste_01": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184837-00001-00049-2.nii",
        "md5": "4d31f51564c205bba99ebce6922d1b65",
        "filename": "hmri/b1_ste_01.nii",
    },
    "hmri/b1_se_02": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184849-00002-00097-1.nii",
        "md5": "9ca8506376f50d22a33713fdd237e6aa",
        "filename": "hmri/b1_se_02.nii",
    },
    "hmri/b1_ste_02": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184849-00002-00145-2.nii",
        "md5": "5c08213a7381d960e22695ec065f7b7b",
        "filename": "hmri/b1_ste_02.nii",
    },
    "hmri/b1_se_03": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184901-00003-00193-1.nii",
        "md5": "37782426e2ea0dbe5be330329a2554d6",
        "filename": "hmri/b1_se_03.nii",
    },
    "hmri/b1_ste_03": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184901-00003-00241-2.nii",
        "md5": "70132d4730c0dde9c545bce02ad9c0a3",
        "filename": "hmri/b1_ste_03.nii",
    },
    "hmri/b1_se_04": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184913-00004-00289-1.nii",
        "md5": "83edd81b6bbbf01c6c76af5049c9065d",
        "filename": "hmri/b1_se_04.nii",
    },
    "hmri/b1_ste_04": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184913-00004-00337-2.nii",
        "md5": "51ea0bc56ea6ec71db6f4ca8c84e3274",
        "filename": "hmri/b1_ste_04.nii",
    },
    "hmri/b1_se_05": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184925-00005-00385-1.nii",
        "md5": "024e1b5b5e26e178e9364274ebbfd585",
        "filename": "hmri/b1_se_05.nii",
    },
    "hmri/b1_ste_05": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184925-00005-00433-2.nii",
        "md5": "da2b822d97fa152a01fbcf93931cebd8",
        "filename": "hmri/b1_ste_05.nii",
    },
    "hmri/b1_se_06": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184937-00006-00481-1.nii",
        "md5": "fdac893816911be119e62e91c9fc7470",
        "filename": "hmri/b1_se_06.nii",
    },
    "hmri/b1_ste_06": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184937-00006-00529-2.nii",
        "md5": "548d0bae4f70e9dd3ac11e8f12dc73a6",
        "filename": "hmri/b1_ste_06.nii",
    },
    "hmri/b1_se_07": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184949-00007-00577-1.nii",
        "md5": "a065153d79ea887d26b38ab11c05b4f1",
        "filename": "hmri/b1_se_07.nii",
    },
    "hmri/b1_ste_07": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-184949-00007-00625-2.nii",
        "md5": "c6a4640f628e1d5bd95729d19c30fe3d",
        "filename": "hmri/b1_ste_07.nii",
    },
    "hmri/b1_se_08": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185001-00008-00673-1.nii",
        "md5": "5141d2fec8e23e6b478c72b1f650ce0a",
        "filename": "hmri/b1_se_08.nii",
    },
    "hmri/b1_ste_08": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185001-00008-00721-2.nii",
        "md5": "bbd22f2516db77a5500eb3be5d7f9218",
        "filename": "hmri/b1_ste_08.nii",
    },
    "hmri/b1_se_09": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185013-00009-00769-1.nii",
        "md5": "9b812bf465ea5b5d716394ed6998550b",
        "filename": "hmri/b1_se_09.nii",
    },
    "hmri/b1_ste_09": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185013-00009-00817-2.nii",
        "md5": "a00429eb9839fa7b12ab204a006d67bd",
        "filename": "hmri/b1_ste_09.nii",
    },
    "hmri/b1_se_10": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185025-00010-00865-1.nii",
        "md5": "9904212cb957a031e63edd72b6425207",
        "filename": "hmri/b1_se_10.nii",
    },
    "hmri/b1_ste_10": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185025-00010-00913-2.nii",
        "md5": "737d60ccd3f22dbdad949ffc4e746106",
        "filename": "hmri/b1_ste_10.nii",
    },
    "hmri/b1_se_11": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185037-00011-00961-1.nii",
        "md5": "8c3f3014def13d6ffec7d9d4ae3f6e2c",
        "filename": "hmri/b1_se_11.nii",
    },
    "hmri/b1_ste_11": {
        "url": BASE + "mfc_seste_b1map_v1e_0004/anon_s2018-02-28_18-26-185037-00011-01009-2.nii",
        "md5": "e6d5059e1f5d1f6acd1ddfb3605e3480",
        "filename": "hmri/b1_ste_11.nii",
    },
    "hmri/b0_mag_e1": {
        "url": BASE + "gre_field_mapping_1acq_rl_0005/"
                      "anon_s2018-02-28_18-26-185100-00001-00001-1.nii",
        "md5": "48ec224055e62226264e9ee7d129d690",
        "filename": "hmri/b0_mag_e1.nii",
    },
    "hmri/b0_mag_e2": {
        "url": BASE + "gre_field_mapping_1acq_rl_0005/"
                      "anon_s2018-02-28_18-26-185101-00001-00001-2.nii",
        "md5": "e2fa4d6b53a1289408ad83b632d40660",
        "filename": "hmri/b0_mag_e2.nii",
    },
    "hmri/b0_phase": {
        "url": BASE + "gre_field_mapping_1acq_rl_0006/"
                      "anon_s2018-02-28_18-26-185101-00001-00001-2.nii",
        "md5": "cfa0ee99a8a25b25601795b629dee27e",
        "filename": "hmri/b0_phase.nii",
    },
}

# Source: the DICOM sidecars of the files above (acqpar: RepetitionTime, FlipAngle,
#         EchoTime); for the SE/STE series the nominal angles and T1 of the hMRI toolbox
#         standard defaults (config/hmri_b1_standard_defaults.m, i3D_EPI: beta 115:-5:65
#         degrees, T1 1192 ms), and the mixing time of the series' own sidecar
#         (sWipMemBlock.alFree(14), 33800 us), which the toolbox reads before its default.
# Units:  s, degrees; the B0 phase difference is stored 0..4095 with the file's scale factors
#         (slope 2, intercept -4096, which the NIfTI reader applies) mapping it onto
#         -4096..4094, so -pi..pi is the scaled value over 4096.
HMRI_ACQUISITION = {
    "flash": {"tr_s": 0.025, "te_s": tuple(0.0023 * k for k in range(1, 9)),
              "flip_deg": {"pdw": 6.0, "t1w": 21.0, "mtw": 6.0}},
    "b1": {"beta_deg": tuple(float(115 - 5 * k) for k in range(11)), "tm_s": 0.0338,
           "t1_s": 1.192, "pairs": 5, "fwhm_mm": 8.0},
    "b0": {"te_s": (0.010, 0.01246), "phase_range": 4096.0},
    # the head support of each grid: magnitude above this fraction of the 99th
    # percentile (stated), so the maps are zero where the images are noise
    "support": 0.15,
}


#*****************#
#   flip term     #
#*****************#
def _flip_term(flip_rad, small_angle):
    """The flip angle as the rational approximation takes it: alpha, or 2 tan(alpha / 2)."""
    return flip_rad if small_angle else 2.0 * np.tan(flip_rad / 2.0)


#*****************#
#   finite maps   #
#*****************#
def _finite(values, valid):
    """A map as float32 with everything outside `valid` or non-finite set to zero."""
    return np.where(valid & np.isfinite(values), values, 0.0).astype(np.float32)


#**************************************************************************************************#
#                                           Class HMRI                                             #
#**************************************************************************************************#
#                                                                                                  #
# The demo subject's quantitative maps, derived once from the raw scans and cached.                #
#                                                                                                  #
#**************************************************************************************************#
class HMRI:
    """The demo subject's quantitative maps, derived once from the raw scans and cached.

    `volume(name)` serves one map as a phantom.Volume on its own grid (the MPM maps
    on the 0.8 mm oblique grid, the transmit map on the 4 mm B1 grid, the field map on
    the 3 mm B0 grid; every map keeps its own affine and answers box queries). Names
    and units follow qMRI-BIDS (Karakuzu et al. 2022): R1map and R2starmap in 1/s,
    T1map in s, PDmap in the arbitrary units of the FLASH signal (the A of Helms),
    MTsat in percent units, TB1map relative with 1.0 nominal, fieldmap in Hz.

    The derivations are the hMRI toolbox formulas (Tabelow et al. 2019, NeuroImage
    194:191) in numpy, one contrast echo where the toolbox extrapolates to TE 0
    (stated): R1 and A from the dual flip angle rational approximation (Helms et al.
    2008a, MRM 59:667) with the transmit map resampled onto the MPM grid, MT
    saturation from the MTw image (Helms et al. 2008b, MRM 60:1396) with the heuristic
    B1 correction (Helms et al. 2021, arXiv:2104.14878), R2* by a log-linear fit
    of the eight PDw echoes, the transmit map from the SE/STE ratio of eleven nominal
    flip angles with the mixing-time T1 correction, and the field map from the
    vendor's phase difference over the echo spacing. Not done (stated): the EPI
    distortion correction of the B1 series, the receive bias correction, and the
    imperfect-spoiling correction, so the maps are the toolbox's uncorrected variants.
    """

    MAPS = ("R1map", "T1map", "PDmap", "MTsat", "R2starmap", "TB1map", "fieldmap")

    #*************#
    #   volumes   #
    #*************#
    @classmethod
    def volume(cls, name, space=None):
        """One derived map as a Volume, deriving and caching the whole stack on first
        use; with `space` "bigbrain_mr" its affine carries the map into the
        BigBrain-MR world (`registration`), so it reads at the virtual brain's
        positions."""
        if name not in cls.MAPS:
            raise KeyError(f"unknown hMRI map {name!r}; known: {', '.join(cls.MAPS)}")
        from mrsandbox.resources import Resources

        cache = Resources.cache_dir() / "hmri/maps.npz"
        if not cache.exists():
            maps = cls.derive()
            np.savez_compressed(cache, **{key: value for key, value in maps.items()})
        stored = np.load(cache)
        data, affine = stored[name], stored[name + "_affine"]
        if name == "T1map":
            data = np.where(data > 0.0, 1.0 / np.where(data > 0.0, data, 1.0),
                            0.0).astype(np.float32)
        if space is not None:
            affine = cls.registration(space) @ affine
        return Volume(data, affine, name="hmri/" + name)

    @classmethod
    def registration(cls, space="bigbrain_mr"):
        """The world-to-world affine carrying the session's scanner frame into the
        BigBrain-MR world (phantom.registration.register), fitted once and cached:
        a shape fit of the brain tissue, the largest connected component of the
        voxels reading brain tissue in R1 (0.4 to 1.8 1/s) and MT saturation (0.3 to
        6 percent units) after a 2.4 mm erosion that cuts the brainstem's link to
        the neck, dilated back, against the BigBrain-MR grey and white matter labels,
        then a contrast fit on tissue classes from there: the session's grey and
        white matter (R1 below and above 0.85 1/s inside that tissue, median
        filtered) as a two-level image against the atlas's two-level labels. The R1
        values themselves are too speckled to hold a 12-parameter fit (stated); the
        classes are not. An affine between two subjects cannot follow the cortex, so
        the classes still overlap only in part (stated). Every map of the session
        shares the transform: the MPM, B1, and B0 grids are co-registered by
        acquisition (stated)."""
        import scipy.ndimage

        from mrsandbox.phantom.brain import Brain
        from mrsandbox.phantom.registration import register
        from mrsandbox.resources import Resources

        if space != "bigbrain_mr":
            raise KeyError(f"unknown space {space!r}; known: bigbrain_mr")
        cache = Resources.cache_dir() / "hmri/to_bigbrain_mr_classes.npy"
        if not cache.exists():
            r1, mt = cls.volume("R1map"), cls.volume("MTsat")
            tissue = (np.isfinite(r1.data) & (r1.data > 0.4) & (r1.data < 1.8)
                      & np.isfinite(mt.data) & (mt.data > 0.3) & (mt.data < 6.0))
            eroded = scipy.ndimage.binary_erosion(tissue, iterations=3)
            labels, count = scipy.ndimage.label(eroded)
            largest = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
            mask = scipy.ndimage.binary_dilation(largest, iterations=3) & tissue
            atlas = Volume.load("bigbrain_mr/labels_800um")
            brain = np.isin(atlas.data, Brain.BIGBRAIN_TISSUES["wm"] + Brain.BIGBRAIN_TISSUES["gm"])
            shape = register(Volume(mask.astype(np.float32), r1.affine),
                             Volume(brain.astype(np.float32), atlas.affine))
            classes = np.where(tissue, np.where(r1.data > 0.85, 2.0, 1.0), 0.0).astype(np.float32)
            classes = scipy.ndimage.median_filter(classes, size=3)
            levels = (1.0 * np.isin(atlas.data, Brain.BIGBRAIN_TISSUES["gm"])
                      + 2.0 * np.isin(atlas.data, Brain.BIGBRAIN_TISSUES["wm"])).astype(np.float32)
            refined = register(Volume(classes, shape @ r1.affine), Volume(levels, atlas.affine))
            np.save(cache, refined @ shape)
        return np.load(cache)

    @classmethod
    def cached(cls):
        """True when the derived stack (or every raw file) is already in the cache."""
        from mrsandbox.resources import Resources

        return (Resources.cache_dir() / "hmri/maps.npz").exists() or all(
            Resources.cached(name) for name in HMRI_DEMO)

    #**************#
    #   derive     #
    #**************#
    @classmethod
    def derive(cls):
        """Every map from the raw scans, as {name: array, name_affine: affine} with
        T1map served as 1 / R1map."""
        import nibabel as nib

        from mrsandbox.resources import Resources

        def image(key):
            im = nib.load(str(Resources.path("hmri/" + key)))
            return np.asarray(im.dataobj, dtype=np.float32), np.array(im.affine, dtype=np.float64)

        acq = HMRI_ACQUISITION
        support = lambda magnitude: magnitude > acq["support"] * np.percentile(magnitude, 99.0)
        # the transmit map on its own grid, then on the MPM grid for the corrections
        se = np.stack([image(f"b1_se_{k:02d}")[0] for k in range(1, 12)])
        ste = np.stack([image(f"b1_ste_{k:02d}")[0] for k in range(1, 12)])
        b1_affine = image("b1_se_01")[1]
        b1 = acq["b1"]
        transmit = cls.transmit(se, ste, b1["beta_deg"], b1["tm_s"], b1["t1_s"], pairs=b1["pairs"])
        transmit = cls.smoothed(np.where(support(se.max(axis=0)), transmit, 0.0), b1_affine,
                                b1["fwhm_mm"])
        pdw, mpm_affine = image("pdw_e1")
        head = support(pdw)
        t1w, mtw = image("t1w_e1")[0], image("mtw_e1")[0]
        flash = acq["flash"]
        b1_on_mpm = cls.resampled(transmit, b1_affine, mpm_affine, pdw.shape)
        b1_on_mpm = np.where(b1_on_mpm > 0.0, b1_on_mpm, 1.0)
        pdw_flip, t1w_flip = (np.radians(flash["flip_deg"]["pdw"]),
                              np.radians(flash["flip_deg"]["t1w"]))
        tr = flash["tr_s"]
        r1 = cls.r1(pdw, t1w, pdw_flip, t1w_flip, tr, tr, b1_on_mpm)
        pd = cls.proton_density(pdw, t1w, pdw_flip, t1w_flip, tr, tr, b1_on_mpm)
        # MT saturation in the Helms convention: the small-angle A and R1 at nominal
        # B1, then the heuristic B1 correction of the saturation itself
        mt = cls.mt_saturation(mtw, np.radians(flash["flip_deg"]["mtw"]), tr,
                               cls.proton_density(pdw, t1w, pdw_flip, t1w_flip, tr, tr,
                                                  small_angle=True),
                               cls.r1(pdw, t1w, pdw_flip, t1w_flip, tr, tr, small_angle=True))
        mt = cls.mt_corrected(mt, b1_on_mpm)
        echoes = np.stack([image(f"pdw_e{k}")[0] for k in range(1, 9)])
        r2star = cls.r2star(echoes, flash["te_s"])
        r1, pd, mt, r2star = (np.where(head, m, 0.0).astype(np.float32)
                              for m in (r1, pd, mt, r2star))
        phase, b0_affine = image("b0_phase")
        b0 = acq["b0"]
        radians = phase / b0["phase_range"] * np.pi
        field = cls.field_hz(radians, b0["te_s"][1] - b0["te_s"][0])
        field = np.where(support(image("b0_mag_e1")[0]), field, 0.0).astype(np.float32)
        return {"R1map": r1, "R1map_affine": mpm_affine, "PDmap": pd, "PDmap_affine": mpm_affine,
                "MTsat": mt, "MTsat_affine": mpm_affine, "R2starmap": r2star,
                "R2starmap_affine": mpm_affine, "TB1map": transmit, "TB1map_affine": b1_affine,
                "fieldmap": field, "fieldmap_affine": b0_affine,
                "T1map": r1, "T1map_affine": mpm_affine}

    #***************#
    #   r1 and pd   #
    #***************#
    @staticmethod
    def r1(pdw, t1w, pdw_flip_rad, t1w_flip_rad, pdw_tr_s, t1w_tr_s, b1=1.0, small_angle=False):
        """R1 in 1/s from a PDw and a T1w FLASH image: the dual flip angle rational
        approximation of the Ernst equation (Helms et al. 2008, MRM 59:667; the hMRI
        toolbox's hmri_calc_R1), the nominal flips scaled by the relative transmit
        field `b1`. The flip enters as 2 tan(alpha / 2) (Edwards et al. 2026, MRM,
        doi 10.1002/mrm.70174), accurate at the demo's 6 and 21 degrees where the
        small-angle form is not, or as alpha itself with `small_angle`, the form the
        MT saturation of Helms is defined on. Zero where either image is zero."""
        a, b = (_flip_term(b1 * pdw_flip_rad, small_angle),
                _flip_term(b1 * t1w_flip_rad, small_angle))
        valid = (pdw > 0.0) & (t1w > 0.0)
        pdw, t1w = np.where(valid, pdw, 1.0), np.where(valid, t1w, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = 0.5 * (pdw * a / pdw_tr_s - t1w * b / t1w_tr_s) / (t1w / b - pdw / a)
        return _finite(r1, valid)

    @staticmethod
    def proton_density(pdw, t1w, pdw_flip_rad, t1w_flip_rad, pdw_tr_s, t1w_tr_s, b1=1.0,
                       small_angle=False):
        """The signal amplitude A (an unnormalized proton density, arbitrary units) from
        the same pair (Helms et al. 2008, MRM 59:667; hmri_calc_A), the flip term as
        for `r1`."""
        a, b = (_flip_term(b1 * pdw_flip_rad, small_angle),
                _flip_term(b1 * t1w_flip_rad, small_angle))
        valid = (pdw > 0.0) & (t1w > 0.0)
        pdw, t1w = np.where(valid, pdw, 1.0), np.where(valid, t1w, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            amplitude = (t1w * pdw * (t1w_tr_s * a / b - pdw_tr_s * b / a)
                         / (pdw * t1w_tr_s * a - t1w * pdw_tr_s * b))
        return _finite(amplitude, valid)

    #*******************#
    #   mt saturation   #
    #*******************#
    @staticmethod
    def mt_saturation(mtw, flip_rad, tr_s, amplitude, r1):
        """MT saturation in percent units from the MTw image and the uncorrected A and
        R1 of the same session (Helms et al. 2008, MRM 60:1396; hmri_calc_MTsat):
        delta = (A alpha / S_MT - 1) R1 TR - alpha^2 / 2."""
        valid = mtw > 0.0
        signal = np.where(valid, mtw, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (amplitude * flip_rad - signal) / signal * r1 * tr_s - flip_rad ** 2 / 2.0
        return _finite(100.0 * delta, valid)

    @staticmethod
    def mt_corrected(mt_sat, b1, c=0.4):
        """The heuristic transmit correction of an MT saturation computed at nominal B1
        (Helms et al. 2021, arXiv:2104.14878; the hMRI 'helms' model): delta (1 - C) /
        (1 - C b1) with C = 0.4 at 3 T."""
        return (mt_sat * (1.0 - c) / (1.0 - c * b1)).astype(np.float32)

    #*************#
    #   r2 star   #
    #*************#
    @staticmethod
    def r2star(echoes, te_s):
        """R2* in 1/s by the log-linear least-squares fit of a multi-echo series
        (echoes, ...) at echo times `te_s` (the ESTATICS estimate on one contrast,
        Weiskopf et al. 2014, Front Neurosci 8:278); zero where any echo is zero."""
        te = np.asarray(te_s, dtype=np.float64)
        valid = np.all(echoes > 0.0, axis=0)
        logs = np.log(np.where(valid, echoes, 1.0))
        te_centred = te - te.mean()
        slope = (np.tensordot(te_centred, logs - logs.mean(axis=0), axes=(0, 0))
                 / np.sum(te_centred ** 2))
        return np.where(valid, -slope, 0.0).astype(np.float32)

    #*****************#
    #   transmit      #
    #*****************#
    @staticmethod
    def transmit(se, ste, beta_deg, tm_s, t1_s, pairs=5):
        """The relative transmit field (actual over nominal flip) from spin-echo and
        stimulated-echo images (series, ...) at nominal flips `beta_deg`: the ratio
        STE / SE exp(TM / T1) is cos(beta b1), so each pair yields acos(ratio) / beta
        and its mirror 180 / beta - that (the arccosine ambiguity of magnitudes); of
        the `pairs` strongest spin echoes per voxel, the candidate assignment with the
        smallest spread is taken and averaged (hmri_create_b1map, the toolbox's
        i3D_EPI method). Zero where the spin echo is silent."""
        beta = np.asarray(beta_deg, dtype=np.float64)
        valid = se.max(axis=0) > 0.0
        ratio = np.clip(np.exp(tm_s / t1_s) * ste / np.where(se > 0.0, se, 1.0), -1.0, 1.0)
        first = np.degrees(np.arccos(ratio)) / beta.reshape((-1,) + (1,) * (se.ndim - 1))
        second = 180.0 / beta.reshape((-1,) + (1,) * (se.ndim - 1)) - first
        strongest = np.argsort(-se, axis=0)[:pairs]
        take = lambda stack: np.take_along_axis(stack, strongest, axis=0)
        first, second = take(first), take(second)
        best, spread = np.zeros_like(first[0]), np.full(first[0].shape, np.inf)
        for choice in itertools.product((0, 1), repeat=pairs):
            picked = np.stack([second[k] if flag else first[k] for k, flag in enumerate(choice)])
            std = picked.std(axis=0)
            better = std < spread
            best, spread = (np.where(better, picked.mean(axis=0), best),
                            np.where(better, std, spread))
        return np.where(valid, best, 0.0).astype(np.float32)

    @staticmethod
    def smoothed(volume, affine, fwhm_mm):
        """A map smoothed by a Gaussian of `fwhm_mm` inside its support (normalized by
        the smoothed support, so edges do not fade), zero outside."""
        import scipy.ndimage

        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        sigma = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / spacing
        support = (volume > 0.0).astype(np.float64)
        blurred = scipy.ndimage.gaussian_filter(volume * support, sigma)
        weight = scipy.ndimage.gaussian_filter(support, sigma)
        return np.where(support > 0.0, blurred / np.where(weight > 0.0, weight, 1.0),
                        0.0).astype(np.float32)

    @staticmethod
    def resampled(volume, affine, target_affine, target_shape):
        """A map linearly interpolated onto another grid (world coordinates through the
        two affines), zero outside its extent."""
        import scipy.ndimage

        grid = np.indices(target_shape).reshape(3, -1).astype(np.float64)
        world = target_affine[:3, :3] @ grid + target_affine[:3, 3:4]
        source = np.linalg.solve(affine[:3, :3], world - affine[:3, 3:4])
        values = scipy.ndimage.map_coordinates(volume.astype(np.float64), source, order=1,
                                               mode="constant", cval=0.0)
        return values.reshape(target_shape).astype(np.float32)

    #*************#
    #   field     #
    #*************#
    @staticmethod
    def field_hz(phase_difference_rad, delta_te_s):
        """The field offset in Hz from a dual-echo phase difference over the echo
        spacing: phi / (2 pi delta TE), the vendor's sign (stated); the unambiguous
        range is +- 1 / (2 delta TE), no unwrapping applied (stated)."""
        return (phase_difference_rad / (2.0 * np.pi * delta_te_s)).astype(np.float32)

    #**************#
    #   cohort     #
    #**************#
    @classmethod
    def cohort(cls, folder=None):
        """The hmri_demo_prisma_3t cohort: the demo subject's own water T1 rows off its MPM
        T1 (1/r1 of the FLASH pair) at NOMINAL transmit, since its SE/STE transmit map
        reads a tissue contrast no transmit field has; grey and
        white matter only, the protocol's flip angles not resolving CSF's T1. The subject's
        own T1-weighted FLASH is segmented by FSL (scan.maps: conform, strip, segment) in
        `folder` (the cache unless given), and the files there are reused."""
        import dataclasses
        import pathlib

        import nibabel as nib

        from mrsandbox.hardware import Scanner
        from mrsandbox.resources import Resources, participants
        from mrsandbox.scan import maps
        from mrsandbox.scan.cohort import Cohort, Member
        from mrsandbox.scan.maps import MapReading
        from mrsandbox.scan.profile import ScannerProfile

        folder = (pathlib.Path(folder) if folder is not None
                  else Resources.cache_dir() / "hmri" / "cohort")
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "tissue_pve_2.nii.gz").exists():
            nib.save(maps.conform(nib.load(str(Resources.path("hmri/t1w_e1")))),
                     str(folder / "t1.nii.gz"))
            maps.tissues(folder / "t1.nii.gz", folder)
        # the T1 map at nominal transmit, inside the head (the derived stack's T1 carries
        # the measured transmit, so it is made here from the FLASH pair alone)
        pdw_image = nib.load(str(Resources.path("hmri/pdw_e1")))
        pdw, t1w = (np.asarray(nib.load(str(Resources.path(f"hmri/{k}"))).dataobj,
                               dtype=np.float32) for k in ("pdw_e1", "t1w_e1"))
        flash = HMRI_ACQUISITION["flash"]
        r1 = cls.r1(pdw, t1w, np.radians(flash["flip_deg"]["pdw"]),
                    np.radians(flash["flip_deg"]["t1w"]), flash["tr_s"], flash["tr_s"], 1.0)
        r1 = np.where(pdw > HMRI_ACQUISITION["support"] * np.percentile(pdw, 99.0), r1, 0.0)
        nib.save(nib.Nifti1Image(np.where(r1 > 0.0, 1.0 / np.where(r1 > 0.0, r1, 1.0),
                                          0.0).astype(np.float32), pdw_image.affine),
                 str(folder / "t1map.nii.gz"))
        reading = MapReading.from_files(
            "water.t1_s", folder / "t1map.nii.gz",
            *(folder / f"tissue_pve_{k}.nii.gz" for k in (1, 2, 0)),
            header={"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma",
                    "MagneticFieldStrength": 2.89, "ReceiveCoilName": "64Ch Head"})
        note = ("one subject's MPM T1 (variable flip angle, TR 25 ms, 6 and 21 degrees) at "
                "NOMINAL transmit: its SE/STE transmit map reads grey 1.03 and white 0.90, "
                "a tissue contrast no transmit field has; MPM T1s run longer than inversion "
                "recovery at 3 T; CSF not resolved by the protocol, left to the literature")
        records = tuple(dataclasses.replace(record, note=note) for record in reading.records(
            "hMRI toolbox demo subject, MPM (CC BY 4.0)",
            "Callaghan et al. 2019, Data in Brief 25:104132",
            "https://github.com/hMRI-group/hMRI-toolbox-demodata", method="variable flip angle")
            if record.leaf.split(".")[-1] in ("gm", "wm"))
        return participants.attach(Cohort("hmri_demo_prisma_3t", (Member(
            "the subject", ScannerProfile(Scanner.preset("siemens_prisma_3t"), records),
            (0.0, -35.0, 20.0), (20.0, 20.0, 20.0), "icbm152"),)))
