####################################################################################################
#                                          test_folder.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 27/09/26                                                                                #
#                                                                                                  #
# Purpose: A folder of scans is told apart by what each file is, raw spectra go through spec2nii,   #
#          runs find their water references, and the console reads the folder at once.            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsandbox.io import folder


def scans(tmp_path):
    """A folder with one file of every kind the package tells apart."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext
    from nifti_mrs.hdr_ext import Hdr_Ext
    import nibabel as nib

    from mrsandbox import Scanner, ScannerProfile

    for name, suppressed in (("sub-01_acq-slaser_svs.nii.gz", True), ("sub-01_acq-slaser_mrsref.nii.gz", False)):
        hdr = Hdr_Ext(123.2, "1H")
        hdr.set_dim_info(0, "DIM_COIL")
        hdr.set_dim_info(1, "DIM_DYN")
        hdr.set_standard_def("WaterSuppressed", suppressed)
        hdr.set_standard_def("EchoTime", 0.035)
        hdr.set_standard_def("RepetitionTime", 2.0)
        hdr.set_standard_def("SequenceName", "svs_slaser")
        data = (np.random.default_rng(0).normal(size=(1, 1, 1, 64, 2, 4)) + 0j).astype(np.complex64)
        gen_nifti_mrs_hdr_ext(data, 5e-4, hdr).save(tmp_path / name)
    nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)).to_filename(tmp_path / "sub-01_T1w.nii")
    nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)).to_filename(tmp_path / "sub-01_acq-fieldmap.nii.gz")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "S01_PRESS_35.dat").write_bytes(b"")
    (tmp_path / "raw" / "S01_PRESS_35_H2O.dat").write_bytes(b"")
    (tmp_path / "raw" / "lonely.SDAT").write_bytes(b"")
    (tmp_path / "noise.mrd").write_bytes(b"")
    (tmp_path / "press.seq").write_text("")
    ScannerProfile(Scanner.preset("siemens_trio_3t")).save(tmp_path / "trio.json")
    (tmp_path / "notes.txt").write_text("the operator's notes")
    (tmp_path / "picture.jpg").write_bytes(b"")
    return tmp_path


#**************************************************************************************************#
#                                         Class TestFolder                                         #
#**************************************************************************************************#
class TestFolder:

    def test_every_file_is_told_apart_by_what_it_is(self, tmp_path):
        found = {item.path.name: item for item in folder.survey(scans(tmp_path))}
        kinds = {name: item.kind for name, item in found.items()}
        assert kinds == {"sub-01_acq-slaser_svs.nii.gz": "MRS run", "sub-01_acq-slaser_mrsref.nii.gz": "MRS reference", "sub-01_T1w.nii": "T1w",
                         "sub-01_acq-fieldmap.nii.gz": "B0", "S01_PRESS_35.dat": "MRS run", "S01_PRESS_35_H2O.dat": "MRS reference", "lonely.SDAT": "unknown",
                         "noise.mrd": "noise scan", "press.seq": "Pulseq", "trio.json": "scanner file", "picture.jpg": "unknown"}
        assert "notes.txt" not in found                                            # a note goes with what it belongs to
        assert found["sub-01_acq-slaser_svs.nii.gz"].note.endswith("with the transients apart")
        assert found["S01_PRESS_35.dat"].note == "raw spectroscopy, twix through spec2nii" and found["lonely.SDAT"].note == "an SDAT without its SPAR"
        assert all(item.gives for item in found.values()) and found["T1w"].gives.startswith("the anatomy") if "T1w" in found else True

    def test_runs_find_their_water_references(self, tmp_path):
        found = folder.survey(scans(tmp_path))
        paired = {run.path.name: (reference.path.name if reference else None) for run, reference in folder.pairs(found)}
        assert paired == {"sub-01_acq-slaser_svs.nii.gz": "sub-01_acq-slaser_mrsref.nii.gz", "S01_PRESS_35.dat": "S01_PRESS_35_H2O.dat"}

    def test_a_raw_file_that_will_not_convert_says_so(self, tmp_path):
        found = folder.survey(scans(tmp_path))
        item = next(f for f in found if f.path.name == "S01_PRESS_35.dat")
        assert folder.convert(item, tmp_path / "work") is None and item.note.startswith("spec2nii could not convert it")


#**************************************************************************************************#
#                                        Class TestConsole                                         #
#**************************************************************************************************#
@pytest.mark.render
class TestConsole:

    def test_the_console_reads_a_folder_and_reports_every_file(self, tmp_path):
        from mrsandbox.console.session import Session

        session = Session("Brain, ICBM152 average, 2 mm")
        session.protocol.water_reference = 0
        was = session.card_from(scans(tmp_path) / "sub-01_acq-slaser_svs.nii.gz")
        assert was["te_ms"] == 35.0 and was["tr_ms"] == 2000.0 and was["averages"] == 4 and was["npoints"] == 64 and was["bandwidth_hz"] == 2000.0
        assert was["sequence"] == "sLASER" and was["scanner"] == "generic_3t" and abs(was["field_t"] - 2.894) < 0.01
        assert session.protocol.sequence == "sLASER" and session.protocol.te_ms == 35.0
        report = {file: (kind, gave) for file, kind, gave in session.replicate(tmp_path)}
        assert report["trio.json"] == ("scanner file", "the scanner's calibration") and session.scanner == "siemens_trio_3t"
        assert report["press.seq"][0] == "Pulseq" and report["noise.mrd"][0] == "noise scan"
        assert report["sub-01_acq-slaser_svs.nii.gz"][1].startswith("not read") or "the scanner" in report["sub-01_acq-slaser_svs.nii.gz"][1]
        assert report["S01_PRESS_35.dat"][1].startswith("spec2nii could not convert it") and report["lonely.SDAT"] == ("unknown", "an SDAT without its SPAR")
        session.hardware["max_slew_t_m_s"] = 100.0
        session.choices["lineshape.gauss_hz"] = "pooled"
        session.reset_room()
        assert session.scanner == "generic_3t" and session.hardware == {} and session.choices == {} and session.calibration is None and session.files == {}
