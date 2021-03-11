import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn import plotting, input_data, masking, image
import glob
import os
import common as cm
import masks
import multiprocessing
import functools as ft
import basic_settings as bs
import warnings
import masks
from concurrent.futures import ThreadPoolExecutor

print('hi')
ROI_TO_MASK = 'roi_to_mask'
ROI_TO_ROI = 'roi_to_roi'
ROI_TO_BRAIN = 'roi_to_brain'
FC_TYPES = [ROI_TO_MASK, ROI_TO_ROI, ROI_TO_BRAIN]


# Classes
# class FcArray():
#     def init(self, fc_type, subjects, sessions, n_masker, second_masker,
#              bold_WC, cores):
#         self.fc_type = fc_type
#         self.sessions = sessions
#         self.n_masker = first_masker
#         self.m_masker = second_masker
#         self.bold_WC = bold_WC
#         self.brain_mask_WC = brain_mask_WC
#         self.cores = cores

#         self.fcs = [FcSubject(sub) for sub in subjects]


# class FcSubject():
#     def __init__(self, subject):
#         self.subject = subject
#         self.matrix = np.zeros()


# Functions
# how to print what subjects were run??? make FC class and pickle!!!
def fc(dataset_dir, fc_type, first_mask, second_mask, output_name, subjects=[], sessions=None,
       bold_WC='*preproc_bold.nii.gz', brain_mask_WC='*', num=None,
       cores=(os.cpu_count() // 2), save=True):
    dir_tree = cm.DirectoryTree(dataset_dir, sessions=sessions)
    subjects = cm.get_subjects(dir_tree.fmriprep_dir, dir_tree, num=num)

    # instantiate first masker
    first_masker = get_roi_mask(first_mask)

    # instantiate second masker based on fc type
    if fc_type == ROI_TO_MASK:
        second_masker = masks.get_binary_mask(second_mask)
        second_masker_count = np.count_nonzero(second_masker.mask_img.get_fdata())
    elif fc_type == ROI_TO_ROI:
        second_masker = get_roi_mask(second_mask)
        second_masker_count = len(np.unique(second_masker.labels_img.get_fdata())) - 1
    elif fc_type == ROI_TO_BRAIN:
        second_masker, regions = get_brain_masker(subjects, brain_mask_WC)
    else:
        raise ValueError(f'Invalid FC type. Must be one of: {FC_TYPES}')

    # get number of rois in first mask
    first_masker_count = len(np.unique(first_masker.labels_img.get_fdata())) - 1

    # run fc on subjects in parallel
    pool = multiprocessing.Pool(cores)
    correlation_list = pool.map(ft.partial(fc_sub, first_masker, first_masker_count,
                                            second_masker, second_masker_count, bold_WC),
                                          subjects)

    # convert into numpy array with shape of (n x m x subjects)
    # where n and m are number of elements in first and scond region
    seed_to_voxel_correlations = np.zeros((first_masker_count, second_masker_count, len(subjects)))
    for index, corr in enumerate(correlation_list):
        seed_to_voxel_correlations[:, :, index] = corr
    print(seed_to_voxel_correlations.shape)

    # save fc correlation to numpy array file
    if save:
        os.makedirs(dir_tree.fc_dir, exist_ok=True)
        np.save(dir_tree.fc_dir + output_name, seed_to_voxel_correlations)

    return seed_to_voxel_correlations


def fc_sub(first_masker, first_masker_count, second_masker, second_masker_count, bold_WC, subject):
    print(f'\nFunctional connectivity on subject {subject.name}')

    # get subject's bold files based on wildcard
    bold_files = cm.get_ses_files(subject, subject.fmriprep_dir, bold_WC)
    if not any(bold_files):
        warnings.warn("No bold files found.")
        return

    # load bold files
    bold_imgs = load_bold_async(bold_files)
    TR = bold_imgs[0].shape[-1]
    if any(img.shape[-1] != TR for img in bold_imgs):
        raise ValueError('TRs must be equal for each bold file.')
    
    first_series = np.empty([first_masker_count, len(bold_imgs) * TR])
    second_series = np.empty([second_masker_count, len(bold_imgs) * TR])
    for i in range(len(bold_imgs)):
        first_series[:, TR * i: TR * (i +1)] = transform_bold_img(bold_imgs[i], first_masker)
        second_series[:, TR * i: TR * (i +1)] = transform_bold_img(bold_imgs[i], second_masker)
        
    # get FC correlation
    seed_to_voxel_correlations = generate_correlation_mat(first_series, second_series)

    print(seed_to_voxel_correlations.shape)

    return seed_to_voxel_correlations


def get_roi_mask(roi_mask_path):
    roi_mask = nib.load(roi_mask_path)
    roi_masker = input_data.NiftiLabelsMasker(roi_mask)
    return roi_masker




def get_brain_masker(subjects, brain_mask_WC):
    # get brain mask files from each run in subject fmriprep dir
    brain_masks = []
    for subject in subjects:
        brain_masks.extend(cm.get_ses_files(subject, subject.fmriprep_dir,
                                            brain_mask_WC))
    if not any(brain_masks):
        raise ValueError('No brain mask files')
    print(f'Brain masks:\n{brain_masks}')

    # union mask of all run brain masks
    union_mask = masking.intersect_masks(brain_masks, threshold=0)
    brain_masker = input_data.NiftiMasker(union_mask)

    return brain_masker


def generate_correlation_mat(x, y):
    """Correlate each n with each m.
    Parameters
    ----------
    x : np.array
      Shape N X T.
    y : np.array
      Shape M X T.
    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    tr = x.shape[1]
    if tr != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=tr - 1)
    s_y = y.std(1, ddof=tr - 1)
    cov = np.dot(x,
                 y.T) - tr * np.dot(mu_x[:, np.newaxis],
                                    mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def load_bold_async(bold_files):
    generator = ThreadPoolExecutor().map(nib.load, bold_files)
    return [img for img in generator]
    
def transform_bold_img(bold_img, masker):
    return np.swapaxes(masker.fit_transform(bold_img), 0, 1)


def convert_tot_series(bold_tuples, rois):
    # find number of regions (Voxels or rois) & time reps
    # then instantiate time series
    regions, tr = bold_tuples[0][1].shape
    first_series = np.zeros([rois, tr * len(bold_tuples)])
    second_series = np.zeros([regions, tr * len(bold_tuples)])

    # add run matrices to total matrices
    for index, bold_tuple in enumerate(bold_tuples):
        first_series[:, tr * index:tr * (index + 1)] = bold_tuple[0]
        second_series[:, tr * index:tr * (index + 1)] = bold_tuple[1]

    return first_series, second_series


fc('/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D', ROI_TO_MASK, masks.SCHAEFER_PATH,
   masks.MOREL_PATH, 'schaefer_thal', 
   bold_WC='*rest*preproc_bold.nii.gz', cores=10)
