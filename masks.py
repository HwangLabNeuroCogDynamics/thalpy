import nibabel as nib
from nilearn import image, input_data
import os


if os.path.exists('/data/backed_up/shared/ROIs/'):
    PATH_DIR = '/data/backed_up/shared/ROIs/'
# TODO: either move all masks to argon lss or get which path to find masks from
elif os.path.exists('Shared/lss'):
    PATH_DIR = 'Shared/lss_kahwang'

# Masks -----------------------------------------------------------------------
MOREL_PATH = PATH_DIR + 'Thalamus_Morel_consolidated_mask_v3.nii.gz'
MOREL_DICT = {
    1: 'AN',
    2: 'VM',
    3: 'VL',
    4: 'MGN',
    5: 'MD',
    6: 'PuA',
    7: 'LP',
    8: 'IL',
    9: 'VA',
    10: 'Po',
    11: 'LGN',
    12: 'PuM',
    13: 'PuI',
    14: 'PuL',
    17: 'VP'
}

MOREL_LIST = ['AN', 'VM', 'VL', 'MGN', 'MD', 'PuA', 'LP',
              'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']

SCHAEFER_PATH = PATH_DIR + 'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz'



# Mask Functions ---------------------------------------------------------------
def get_binary_mask(mask_path):
    binary_mask = nib.load(mask_path)
    binary_mask = image.math_img("img>0", img=binary_mask)
    binary_masker = input_data.NiftiMasker(binary_mask)
    return binary_masker
