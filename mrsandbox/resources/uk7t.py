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
#          (DREAM transmit field, dual-echo B0, multi-echo R2*) are derived once and cached.       #
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
        "url": "https://ora.ox.ac.uk/objects/uuid:55ca977f-62df-4cbf-b300-2dc893e36647/"
               "files/rz890rv14v",
        "md5": "eb5aa69b68a4cddbbec9dde7c2aba87a",
        "filename": "uk7t/NIfTI.zip",
    },
}

# Source: the Siemens Magnetom 7 T structural protocol of the same deposit (Protocols.zip,
#         Structural_Siemens_Magnetom7T.pdf): 3DREAM_B1_250V, gre_field_mapping_4mm_250V,
#         T2star_1.4_8echo, MP2RAGE_0.7 (page 11: 224 partitions, no partial Fourier in either
#         direction, GRAPPA 3 in plane and none along the partitions, so every inversion is
#         followed by the full 224-echo train). The MP2RAGE phase is stored on 0..4095 for
#         0..2 pi; the inversion efficiency is the Marques 2010 value (stated).
# Units:  s, degrees.
UK7T_ACQUISITION = {
    "dream": {"prep_flip_deg": 60.0, "te_s": (0.00090, 0.00155), "tenths": 10.0},
    "b0": {"te_s": (0.00408, 0.00510), "phase_range": 4096.0},
    "t2star": {"te_s": tuple(0.004 + 0.005 * k for k in range(8)), "flip_deg": 15.0,
               "tr_s": 0.043},
    "mp2rage": {"ti_s": (0.725, 2.150), "tr_s": 3.5, "flip_deg": (5.0, 2.0),
                "echo_spacing_s": 0.0063, "train": 224, "te_s": 0.00264, "efficiency": 0.96,
                "phase_range": 4096.0},
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
    log-linear fit of the eight-echo T2* series (HMRI.r2star), and T1map in seconds
    from the MP2RAGE pair (`file(site, "MP2RAGE_MAG")`, `"MP2RAGE_PHS"`, two inversions
    each) through the Marques et al. 2010 lookup (NeuroImage 49:1271, Appendix): the
    unified image Re(S1* S2) / (|S1|^2 + |S2|^2) of the two inversion blocks is inverted
    on the monotone branch of the sequence's signal model at the typed timings, with
    the two flip angles scaled by the DREAM transmit map resampled onto the MP2RAGE
    grid (Marques and Gruetter 2013, PLoS ONE 8:e69294; the inversion efficiency
    stays the nominal 0.96, the pulse being the tailored adiabatic-like inversion of
    the protocol, stated), zero outside the head (INV2 below the support fraction of
    its 99th percentile).

    Args:
        site: "cam", "car", "gla", "not", or "oxf" (the deposit's author's site).
    """

    SITES = ("cam", "car", "gla", "not", "oxf")
    MAPS = ("TB1map", "fieldmap", "R2starmap", "T1map")

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
    def volume(cls, name, site="oxf", space=None):
        """One derived map of a site as a Volume, deriving and caching the stack on
        first use; with `space` "bigbrain_mr" its affine carries the map into the
        BigBrain-MR world (`registration`), so it reads at the virtual brain's
        positions."""
        if name not in cls.MAPS:
            raise KeyError(f"unknown UK7T map {name!r}; known: {', '.join(cls.MAPS)}")
        from mrsandbox.resources import Resources

        cache = Resources.cache_dir() / "uk7t" / f"{site}_maps.npz"
        if not cache.exists() or name not in np.load(cache):
            np.savez_compressed(cache, **cls.derive(site))
        stored = np.load(cache)
        affine = stored[name + "_affine"]
        if space is not None:
            affine = cls.registration(site, space) @ affine
        return Volume(stored[name], affine, name=f"uk7t/{site}/{name}")

    @classmethod
    def registration(cls, site="oxf", space="bigbrain_mr"):
        """The world-to-world affine carrying a site's scanner frame into the
        BigBrain-MR world (phantom.registration.register, the site's MP2RAGE T1 map
        against the BigBrain-MR T1 map, both 7 T T1 in seconds), fitted once and
        cached. Every map of the site shares it, the maps being co-registered by
        acquisition (stated)."""
        from mrsandbox.phantom.registration import register
        from mrsandbox.resources import Resources

        if space != "bigbrain_mr":
            raise KeyError(f"unknown space {space!r}; known: bigbrain_mr")
        cache = Resources.cache_dir() / "uk7t" / f"{site}_to_bigbrain_mr.npy"
        if not cache.exists():
            np.save(cache, register(cls.volume("T1map", site),
                                    Volume.load("bigbrain_mr/t1map_800um")))
        return np.load(cache)

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
        magnitude, mp2rage_affine = image("MP2RAGE_MAG")
        phase = image("MP2RAGE_PHS")[0] * (2.0 * np.pi / acq["mp2rage"]["phase_range"])
        inversions = magnitude * np.exp(1j * phase)
        b1 = HMRI.resampled(transmit, dream_affine, mp2rage_affine, magnitude.shape[:3])
        # outside the DREAM support (and where its noise floor sits) the nominal field
        t1 = cls.mp2rage_t1(inversions[..., 0], inversions[..., 1], acq["mp2rage"],
                            b1=np.where(b1 > 0.1, b1, 1.0))
        head = magnitude[..., 1] > acq["support"] * np.percentile(magnitude[..., 1], 99.0)
        t1 = np.where(head, t1, 0.0).astype(np.float32)
        return {"TB1map": transmit, "TB1map_affine": dream_affine,
                "fieldmap": field, "fieldmap_affine": b0_affine,
                "R2starmap": r2star, "R2starmap_affine": t2s_affine,
                "T1map": t1, "T1map_affine": mp2rage_affine}

    #*****************#
    #   mp2rage       #
    #*****************#
    @staticmethod
    def mp2rage_signals(t1_s, mp2rage, b1=1.0):
        """The two inversion-block signals of the MP2RAGE at T1 (seconds, any shape),
        Marques et al. 2010 Appendix, per unit M0: the longitudinal steady state over
        the inversion, the two `train`-echo FLASH blocks centred on the inversion
        times at flips `flip_deg` (scaled by `b1`) and echo spacing `echo_spacing_s`,
        and the recoveries between them; each block's signal is read at its centre
        (the k-space origin of a linearly ordered train). Returns (S1, S2)."""
        t1 = np.asarray(t1_s, dtype=np.float64)
        n, tr_flash = mp2rage["train"], mp2rage["echo_spacing_s"]
        ti1, ti2 = mp2rage["ti_s"]
        ta = ti1 - n * tr_flash / 2.0
        tb = ti2 - ti1 - n * tr_flash
        tc = mp2rage["tr_s"] - ti2 - n * tr_flash / 2.0
        a1, a2 = (np.radians(flip) * np.asarray(b1, dtype=np.float64)
                  for flip in mp2rage["flip_deg"])
        eff = mp2rage["efficiency"]
        e1, ea, eb, ec = (np.exp(-t / t1) for t in (tr_flash, ta, tb, tc))
        c1, c2 = np.cos(a1) * e1, np.cos(a2) * e1

        def block(m_start, c):
            # the longitudinal magnetization after a full train from m_start
            return m_start * c ** n + (1.0 - e1) * (1.0 - c ** n) / (1.0 - c)

        # the steady state: the magnetization before an inversion is what an inversion
        # of it (-eff times it) recovers to over the repetition, the recoveries as the
        # constant part and the surviving fraction as the coefficient
        recovered = block(block(1.0 - ea, c1) * eb + (1.0 - eb), c2) * ec + (1.0 - ec)
        mz = recovered / (1.0 + eff * ea * eb * ec * (c1 * c2) ** n)
        m_a = -eff * mz * ea + (1.0 - ea)
        s1 = np.sin(a1) * (m_a * c1 ** (n / 2.0 - 1.0)
                           + (1.0 - e1) * (1.0 - c1 ** (n / 2.0 - 1.0)) / (1.0 - c1))
        m_b = block(m_a, c1) * eb + (1.0 - eb)
        s2 = np.sin(a2) * (m_b * c2 ** (n / 2.0 - 1.0)
                           + (1.0 - e1) * (1.0 - c2 ** (n / 2.0 - 1.0)) / (1.0 - c2))
        return s1, s2

    @classmethod
    def mp2rage_unified(cls, t1_s, mp2rage, b1=1.0):
        """The unified MP2RAGE contrast Re(S1 S2) / (S1^2 + S2^2) at T1, in -0.5..0.5."""
        s1, s2 = cls.mp2rage_signals(t1_s, mp2rage, b1)
        return s1 * s2 / (s1 ** 2 + s2 ** 2)

    @classmethod
    def mp2rage_t1(cls, inv1, inv2, mp2rage, b1=1.0, t1_range_s=(0.1, 6.0)):
        """T1 in seconds from the complex inversion-block images: the unified image
        Re(inv1* inv2) / (|inv1|^2 + |inv2|^2) inverted on the monotone (decreasing)
        branch of `mp2rage_unified` over `t1_range_s`, per voxel with its own `b1`
        (the lookup is tabulated per distinct b1 value, rounded to 0.01)."""
        inv1, inv2 = np.asarray(inv1), np.asarray(inv2)
        unified = (inv1.conj() * inv2).real / (np.abs(inv1) ** 2 + np.abs(inv2) ** 2 + 1e-30)
        grid = np.linspace(t1_range_s[0], t1_range_s[1], 2000)
        b1 = np.round(np.broadcast_to(np.asarray(b1, dtype=np.float64), unified.shape), 2)
        t1 = np.zeros(unified.shape, dtype=np.float64)
        for value in np.unique(b1):
            curve = cls.mp2rage_unified(grid, mp2rage, value)
            top = int(np.argmax(curve))
            bottom = top + int(np.argmin(curve[top:]))
            # np.interp needs increasing x: the branch runs from the maximum down
            branch = slice(top, bottom + 1)
            where = b1 == value
            t1[where] = np.interp(-unified[where], -curve[branch], grid[branch])
        return t1

    #*****************#
    #   transmit      #
    #*****************#
    @staticmethod
    def transmit(flip_map, prep_flip_deg, tenths=10.0):
        """The relative transmit field from a DREAM flip-angle map stored in tenths of a
        degree: the measured preparation flip over the nominal one; zero where the map is."""
        return np.where(flip_map > 0.0, flip_map / tenths / prep_flip_deg, 0.0).astype(np.float32)

    #**************#
    #   cohort     #
    #**************#
    @classmethod
    def cohort(cls, folder=None):
        """The uk7t_oxford_7t cohort: the Oxford travelling head's own water T1 rows off
        its MP2RAGE T1 map, read per tissue on the subject's own segmentation (scan.maps)
        and shipped white matter only: the cortex reads near 2.0 s but cannot be told from
        CSF, whose T1 these timings do not resolve and the lookup clips onto the same
        value, so grey and CSF stay with the literature. The segmentation files
        land in `folder` (the cache unless given) and are reused when there."""
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
                  else Resources.cache_dir() / "uk7t" / "cohort")
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "tissue_pve_3.nii.gz").exists():
            # the segmentation image is INV2 times the regularized uniform image, inside
            # INV2's brain: bet finds the surface on INV2, a plain gradient echo with an
            # ordinary background, not on the uniform image, whose flat background and dark
            # CSF strip a third to half of the head; the product, the usual MP2RAGE input
            # to segmentation tools, adds INV2's proton density so the cortex (mid INV2,
            # dark uniform) parts from the CSF (dark on both)
            magnitude = nib.load(str(cls.file("oxf", "MP2RAGE_MAG")))
            inv2 = np.asarray(magnitude.dataobj, dtype=np.float32)[..., 1]
            nib.save(nib.Nifti1Image(inv2, magnitude.affine), str(folder / "inv2.nii.gz"))
            # the uniform image with the regularized combination of O'Brien et al. 2014
            # (PLoS ONE 9:e99676), (Re(INV1* INV2) - beta) / (|INV1|^2 + |INV2|^2 + 2 beta):
            # the plain Marques 2010 ratio leaves salt-and-pepper noise outside the head
            # that ruins bet and fast; beta well over the background power flattens the
            # background to the dark end (100 times the median power of the outermost
            # five-cell shell, the converter's exact-zero padding left out; the head sits
            # far above it)
            phase = (np.asarray(nib.load(str(cls.file("oxf", "MP2RAGE_PHS"))).dataobj,
                                dtype=np.float32)
                     * (2.0 * np.pi / UK7T_ACQUISITION["mp2rage"]["phase_range"]))
            inversions = np.asarray(magnitude.dataobj, dtype=np.float32) * np.exp(1j * phase)
            one, two = inversions[..., 0], inversions[..., 1]
            power = np.abs(one) ** 2 + np.abs(two) ** 2
            shell = np.ones(power.shape, dtype=bool)
            shell[5:-5, 5:-5, 5:-5] = False
            background = power[shell]
            beta = 100.0 * np.median(background[background > 0.0])
            uniform = (np.real(np.conj(one) * two) - beta) / (power + 2.0 * beta) + 0.5
            nib.save(nib.Nifti1Image((inv2 * uniform).astype(np.float32), magnitude.affine),
                     str(folder / "t1w_source.nii.gz"))
            # four classes: with three, the cortex (dark on the uniform image, T1 near 2 s)
            # and the CSF (dark too, its T1 clipped by the lookup onto the same 2 s) share
            # one class; four still cannot part them, so only the white row ships and grey
            # would be the deep-and-boundary mix
            maps.tissues(folder / "t1w_source.nii.gz", folder, strip_on=folder / "inv2.nii.gz",
                         classes=4, crop=False)
        t1 = cls.volume("T1map", "oxf")
        nib.save(nib.Nifti1Image(np.asarray(t1.data, dtype=np.float32), t1.affine),
                 str(folder / "t1map.nii.gz"))
        reading = MapReading.from_files(
            "water.t1_s", folder / "t1map.nii.gz",
            *(folder / f"tissue_pve_{k}.nii.gz" for k in (1, 3, 0)),
            header={"Manufacturer": "Siemens", "MagneticFieldStrength": 7.0})
        note = ("one subject's MP2RAGE T1 at 7 T (the DREAM transmit map in the flip "
                "angles): white matter only; the cortex reads near 2.0 s but cannot be "
                "told from CSF, whose T1 these timings do not resolve and the lookup clips "
                "onto the same value, so grey and CSF stay with the literature")
        records = tuple(dataclasses.replace(record, note=note) for record in reading.records(
            "UK7T Network travelling head, Oxford (ODC-By)",
            "Clarke et al. 2020, NeuroImage 206:116335",
            "https://ora.ox.ac.uk/objects/uuid:55ca977f-62df-4cbf-b300-2dc893e36647",
            processing="the map onto the tissues' grid by the affines, linear; the tissues "
                       "FSL fast, four classes, of INV2 times the regularized uniform "
                       "image inside INV2's brain (bet)", method="MP2RAGE")
            if record.leaf.endswith(".wm"))
        return participants.attach(Cohort("uk7t_oxford_7t", (Member(
            "the subject", ScannerProfile(Scanner.preset("generic_7t"), records),
            (0.0, -35.0, 20.0), (20.0, 20.0, 20.0), "icbm152"),)))
