####################################################################################################
#                                             dicom.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Images from DICOM, of every vendor: anatomical images (an MPRAGE, a 3D TFE) and maps    #
#          (B0, B1, T1) as NIfTI volumes in scanner coordinates. The conversion is dcm2niix's      #
#          (Li et al. 2016, J Neurosci Methods 264:47), the reference converter, which knows the   #
#          vendors' private tags, their intensity scaling (the Philips rescale slopes), enhanced   #
#          multi-frame files, mosaics, and the slices' order and orientation; writing that again   #
#          here would be writing it worse. It installs from PyPI (the console extra brings it).    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import pathlib
import shutil
import subprocess
import sys
import tempfile


#****************#
#   read dicom   #
#****************#
def read_dicom(source, folder=None):
    """Convert the DICOM images under `source` (a folder searched through its subfolders, or
    one file, which takes its series with it) and return {series: path of its NIfTI}, the
    series named by its number and protocol ("301_T1W_3D_TFE"). The volumes land in `folder`
    (a temporary one unless given) as .nii.gz with dcm2niix's JSON sidecars, anonymized. A
    spectroscopy series is no image and is left to spec2nii."""
    tool = shutil.which("dcm2niix") or shutil.which("dcm2niix", path=str(pathlib.Path(sys.executable).parent))
    if tool is None:
        raise ImportError("reading DICOM needs dcm2niix (pip install dcm2niix; the mrsandbox[console] extra brings it)")
    folder = pathlib.Path(folder) if folder is not None else pathlib.Path(tempfile.mkdtemp(prefix="mrsandbox_dicom_"))
    folder.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([tool, "-z", "y", "-b", "y", "-ba", "y", "-f", "%s_%p", "-o", str(folder), str(source)],
                          capture_output=True, text=True)
    found = {path.name[:-len(".nii.gz")]: path for path in sorted(folder.glob("*.nii.gz"))}
    if not found:
        raise ValueError(f"no DICOM image found under {source}: {done.stdout.strip()[-300:]}")
    return found
