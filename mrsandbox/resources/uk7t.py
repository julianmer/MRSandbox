####################################################################################################
#                                             uk7t.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 17/09/26                                                                                #
#                                                                                                  #
# Purpose: The UK7T Network travelling-head deposit: one subject scanned with the harmonised       #
#          protocol on five Siemens 7 T scanners, shipped as one NIfTI archive. The archive is one #
#          checksummed resource, the members of a site are extracted on demand, and the 7 T maps   #
#          (DREAM transmit field, dual-echo B0, multi-echo R2*) are derived once and cached.        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import zipfile

import numpy as np

from mrsandbox.phantom.volume import Volume
from mrsandbox.resources.hmri import HMRI


#******************#
#   uk7t deposit   #
#******************#
# Source: Clarke, W. T., et al. Multi-site harmonization of 7 tesla MRI neuroimaging protocols.
#         NeuroImage 206 (2020) 116335. doi:10.1016/j.neuroimage.2019.116335.
# Data:   Oxford University Research Archive, uuid 55ca977f-62df-4cbf-b300-2dc893e36647,
#         ODC-By (attribution to W. Clarke and the University of Oxford); the NIfTI archive
#         (dcm2niix conversions, 7.1 GB, 200 files, five Siemens sites of one subject). The
#         deposit publishes no checksum; the md5 below was computed on 17/09/26.
# Units:  NIfTI int16 without scale factors; the DREAM flip map in tenths of a degree, the
#         field map as the vendor's phase difference on -4096..4095 for -pi..pi.
UK7T_DEPOSIT = {
    "uk7t/nifti_zip": {
        "url": "https://ora.ox.ac.uk/objects/uuid:55ca977f-62df-4cbf-b300-2dc893e36647/files/rz890rv14v",
        "md5": "eb5aa69b68a4cddbbec9dde7c2aba87a",
        "filename": "uk7t/NIfTI.zip",
    },
}

# Source: the Siemens Magnetom 7 T structural protocol of the same deposit (Protocols.zip,
#         Structural_Siemens_Magnetom7T.pdf): 3DREAM_B1_250V, gre_field_mapping_4mm_250V,
#         T2star_1.4_8echo, MP2RAGE_0.7.
# Units:  s, degrees.
UK7T_ACQUISITION = {
    "dream": {"prep_flip_deg": 60.0, "te_s": (0.00090, 0.00155), "tenths": 10.0},
    "b0": {"te_s": (0.00408, 0.00510), "phase_range": 4096.0},
    "t2star": {"te_s": tuple(0.004 + 0.005 * k for k in range(8)), "flip_deg": 15.0, "tr_s": 0.043},
    "mp2rage": {"ti_s": (0.725, 2.150), "tr_s": 3.5, "flip_deg": (5.0, 2.0), "echo_spacing_s": 0.0063,
                "slices": 224, "pat": 3},
    "support": 0.15,
}


#**************************************************************************************************#
#                                           Class UK7T                                             #
#**************************************************************************************************#
#                                                                                                  #
# One subject on five 7 T scanners: the per-site transmit, field, and R2* maps, derived once.      #
#                                                                                                  #
#**************************************************************************************************#
class UK7T:
    """One subject on five 7 T scanners: the per-site transmit, field, and R2* maps,
    derived once from the deposit's NIfTI archive and cached.

    `volume(name, site)` serves one map as a phantom.Volume on its own grid in the
    session's scanner coordinates (the maps of one site are co-registered by
    acquisition, stated): TB1map relative with 1.0 nominal from the 3DREAM flip map
    (Nehrke and Boernert 2012, MRM 68:1517; the sequence's own map in tenths of a
    degree over the 60 degree preparation flip, which puts the head centre near
    nominal), fieldmap in Hz from the vendor's dual-echo phase difference over the
    1.02 ms echo spacing (masked by the DREAM support), and R2starmap in 1/s by the
    log-linear fit of the eight-echo T2* series (HMRI.r2star). The MP2RAGE pair
    (`file(site, "MP2RAGE_MAG")`, `"MP2RAGE_PHS"`, two inversions each) is extracted
    on request with its parameters typed above; the T1 lookup is not derived yet.

    Args:
        site: "cam", "car", "gla", "not", or "oxf" (the deposit's author's site).
    """

    SITES = ("cam", "car", "gla", "not", "oxf")
    MAPS = ("TB1map", "fieldmap", "R2starmap")

    #*************#
    #   members   #
    #*************#
    @classmethod
    def file(cls, site, name):
        """The local path of one NIfTI member of a site, extracted from the archive on
        first use (the archive itself is fetched and verified on first use)."""
        from mrsandbox.resources import Resources

        if site not in cls.SITES:
            raise KeyError(f"unknown UK7T site {site!r}; known: {', '.join(cls.SITES)}")
        member = f"NIfTI/{site}/{name}.nii.gz"
        local = Resources.cache_dir() / "uk7t" / member
        if not local.exists():
            with zipfile.ZipFile(Resources.path("uk7t/nifti_zip")) as archive:
                archive.extract(member, Resources.cache_dir() / "uk7t")
        return local

    @classmethod
    def cached(cls, site="oxf"):
        """True when the site's derived stack, its members, or the archive is in the cache."""
        from mrsandbox.resources import Resources

        root = Resources.cache_dir() / "uk7t"
        return ((root / f"{site}_maps.npz").exists() or Resources.cached("uk7t/nifti_zip")
                or all((root / f"NIfTI/{site}/{name}.nii.gz").exists()
                       for name in ("DREAM_FA", "B0Map", "T2S_multi_MAG_e1")))

    #*************#
    #   volumes   #
    #*************#
    @classmethod
    def volume(cls, name, site="oxf"):
        """One derived map of a site as a Volume, deriving and caching the stack on first use."""
        if name not in cls.MAPS:
            raise KeyError(f"unknown UK7T map {name!r}; known: {', '.join(cls.MAPS)}")
        from mrsandbox.resources import Resources

        cache = Resources.cache_dir() / "uk7t" / f"{site}_maps.npz"
        if not cache.exists():
            np.savez_compressed(cache, **cls.derive(site))
        stored = np.load(cache)
        return Volume(stored[name], stored[name + "_affine"], name=f"uk7t/{site}/{name}")

    #**************#
    #   derive     #
    #**************#
    @classmethod
    def derive(cls, site="oxf"):
        """The three maps of a site from its members, as {name: array, name_affine: affine}."""
        import nibabel as nib

        def image(name):
            im = nib.load(str(cls.file(site, name)))
            return np.asarray(im.dataobj, dtype=np.float32), np.array(im.affine, dtype=np.float64)

        acq = UK7T_ACQUISITION
        flip, dream_affine = image("DREAM_FA")
        transmit = cls.transmit(flip, acq["dream"]["prep_flip_deg"], acq["dream"]["tenths"])
        phase, b0_affine = image("B0Map")
        b0 = acq["b0"]
        field = HMRI.field_hz(phase / b0["phase_range"] * np.pi, b0["te_s"][1] - b0["te_s"][0])
        support = HMRI.resampled((transmit > 0.0).astype(np.float32), dream_affine, b0_affine,
                                 phase.shape) > 0.5
        field = np.where(support, field, 0.0).astype(np.float32)
        echoes = np.stack([image(f"T2S_multi_MAG_e{k}")[0] for k in range(1, 9)])
        t2s_affine = image("T2S_multi_MAG_e1")[1]
        head = echoes[0] > acq["support"] * np.percentile(echoes[0], 99.0)
        r2star = np.where(head, HMRI.r2star(echoes, acq["t2star"]["te_s"]), 0.0).astype(np.float32)
        return {"TB1map": transmit, "TB1map_affine": dream_affine,
                "fieldmap": field, "fieldmap_affine": b0_affine,
                "R2starmap": r2star, "R2starmap_affine": t2s_affine}

    #*****************#
    #   transmit      #
    #*****************#
    @staticmethod
    def transmit(flip_map, prep_flip_deg, tenths=10.0):
        """The relative transmit field from a DREAM flip-angle map stored in tenths of a
        degree: the measured preparation flip over the nominal one; zero where the map is."""
        return np.where(flip_map > 0.0, flip_map / tenths / prep_flip_deg, 0.0).astype(np.float32)
