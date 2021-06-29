import nibabel as nib
from nilearn import image, input_data, masking
import os
from thalpy import base
import numpy as np
import warnings

if os.path.exists("/data/backed_up/shared/ROIs/"):
    PATH_DIR = "/data/backed_up/shared/ROIs/"
elif os.path.exists("/Shared/lss_kahwang_hpc/data/ROIs/"):
    PATH_DIR = "/Shared/lss_kahwang_hpc/data/ROIs/"

# Masks -----------------------------------------------------------------------
MOREL_PATH = PATH_DIR + "Thalamus_Morel_consolidated_mask_v3.nii.gz"
MOREL_DICT = {
    1: "AN",
    2: "VM",
    3: "VL",
    4: "MGN",
    5: "MD",
    6: "PuA",
    7: "LP",
    8: "IL",
    9: "VA",
    10: "Po",
    11: "LGN",
    12: "PuM",
    13: "PuI",
    14: "PuL",
    17: "VP",
}

MOREL_LIST = [
    "AN",
    "VM",
    "VL",
    "MGN",
    "MD",
    "PuA",
    "LP",
    "IL",
    "VA",
    "Po",
    "LGN",
    "PuM",
    "PuI",
    "PuL",
    "VP",
]

SCHAEFER_YEO17_PATH = (
    PATH_DIR + "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
)
SCHAEFER_YEO7_PATH = (
    PATH_DIR + "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)
SCHAEFER_7CI = PATH_DIR + "Schaeffer400_7network_CI"
SCHAEFER_17CI = PATH_DIR + "Schaeffer400_17network_CI"


# Mask Functions ---------------------------------------------------------------
def get_roi_masker(roi_mask_path):
    roi_mask = nib.load(roi_mask_path)
    roi_masker = input_data.NiftiLabelsMasker(roi_mask)

    return roi_masker


def get_binary_masker(mask_path):
    binary_mask = nib.load(mask_path)
    binary_mask = image.math_img("img>0", img=binary_mask)
    binary_masker = input_data.NiftiMasker(binary_mask)
    return binary_masker


def union_brain_masks(subjects, brain_mask_WC):
    # get brain mask files from each run in subject fmriprep dir
    brain_masks = []
    for subject in subjects:
        brain_masks.extend(
            base.get_ses_files(subject, subject.fmriprep_dir, brain_mask_WC)
        )
    if not any(brain_masks):
        warnings.warn("No brain mask files")
        return
    print(f"Brain masks:\n{brain_masks}")

    union_mask = masking.intersect_masks(brain_masks, threshold=0)
    return union_mask


def get_brain_masker(subjects, brain_mask_WC):
    union_mask = union_brain_masks(subjects, brain_mask_WC)
    return input_data.NiftiMasker(union_mask)


def masker_count(masker):
    if masker.__class__.__name__ == "NiftiMasker":
        return np.count_nonzero(masker.mask_img.get_fdata())
    elif masker.__class__.__name__ == "NiftiLabelsMasker":
        return len(np.unique(masker.labels_img.get_fdata())) - 1
    else:
        raise TypeError(
            "Masker does not known type. Must be NiftiMasker or NifitLabelsMasker."
        )
