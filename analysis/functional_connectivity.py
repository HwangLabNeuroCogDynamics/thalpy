from constants import wildcards
import masks
import motion
import base

import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn import plotting, input_data, masking, image
import glob
import os
import multiprocessing
import functools as ft
import warnings
from concurrent.futures import ThreadPoolExecutor
import tqdm


# Classes
class FcData:
    def init(
        self,
        fc_type,
        subjects,
        n_masker,
        m_masker,
        task,
        censor,
        filepath,
    ):
        self.fc_type = fc_type
        self.subjects = subjects
        self.n_masker = n_masker
        self.m_masker = m_masker
        if task:
            self.bold_WC = "*" + task + wildcards.BOLD_WC
            self.censor_WC = "*" + task + wildcards.REGRESSOR_WC
        else:
            self.bold_WC = wildcards.BOLD_WC
            self.censor_WC = wildcards.REGRESSOR_WC
        self.censor = censor
        self.filepath = filepath
        self.matrix = None


# Functions
# how to print what subjects were run??? make FC class and pickle!!!
def fc(
    dataset_dir,
    fc_type,
    n_mask,
    m_mask,
    output_name,
    subjects=[],
    sessions=None,
    task=None,
    num=None,
    cores=(os.cpu_count() // 2),
    censor=True,
    save=True,
):

    dir_tree = base.DirectoryTree(dataset_dir, sessions=sessions)
    subjects = base.get_subjects(dir_tree.fmriprep_dir, dir_tree, num=num)

    n_masker = masks.get_roi_mask(n_mask)
    n_masker_count = len(np.unique(n_masker.labels_img.get_fdata())) - 1

    # instantiate second masker based on fc type
    if fc_type == "roi_to_mask":
        m_masker = masks.get_binary_mask(m_mask)
        m_masker_count = np.count_nonzero(m_masker.mask_img.get_fdata())
    elif fc_type == "roi_to_roi":
        m_masker = masks.get_roi_mask(m_mask)
        m_masker_count = len(np.unique(m_masker.labels_img.get_fdata())) - 1
    elif fc_type == "roi_to_brain":
        brain_mask_WC = (
            wildcards.BRAIN_MASK_WC
            if task is None
            else "*" + task + wildcards.BRAIN_MASK_WC
        )
        m_masker = masks.get_brain_masker(subjects, brain_mask_WC)
        m_masker_count = np.count_nonzero(m_masker.mask_img.get_fdata())
    else:
        raise ValueError(
            f"Invalid FC type. Must be one of: ['roi_to_mask', 'roi_to_roi', 'roi_to_brain']"
        )

    fc_data = FcData(
        fc_type,
        subjects,
        n_masker,
        m_masker,
        task,
        censor,
        dir_tree.fc_dir + output_name,
    )

    print(
        f"Calculating functional connectivity for each subject in parallel with {cores} processes."
    )
    correlation_list = []
    pool = multiprocessing.Pool(cores)
    correlation_list = pool.imap(
        ft.partial(fc_sub, fc_data),
        fc_data.subjects,
    )

    # convert into numpy array with shape of (n x m x subjects)
    # where n and m are number of elements in first and scond region
    fc_data.matrix = np.zeros((n_masker_count, m_masker_count, len(subjects)))
    for index, corr in enumerate(correlation_list):
        fc_data.matrix[:, :, index] = corr

    # save fc correlation to numpy array file
    if save:
        os.makedirs(dir_tree.fc_dir, exist_ok=True)
        np.save(dir_tree.fc_dir + output_name, fc_data.matrix)

    return fc_data.matrix


def fc_sub(fc_data, subject):
    # get subject's bold files based on wildcard
    bold_files = base.get_ses_files(subject, subject.fmriprep_dir, fc_data.bold_WC)
    if not any(bold_files):
        warnings.warn("No bold files found.")
        return

    # load bold files
    bold_imgs = load_bold_async(bold_files)
    TR = bold_imgs[0].shape[-1]
    if any(img.shape[-1] != TR for img in bold_imgs):
        raise ValueError("TRs must be equal for each bold file.")

    # generate censor vector and remove censored points for each bold files
    if fc_data.censor:
        censor_files = base.get_ses_files(subject, subject.fmriprep_dir, "")
        censor_vectors = [
            base.censor_motion(censor_file) for censor_file in censor_files
        ]

    n_series = transform_bold_imgs(
        bold_imgs, fc_data.n_masker, fc_data.n_masker_count, censor_vectors
    )
    m_series = transform_bold_imgs(
        bold_imgs, fc_data.m_masker, fc_data.m_masker_count, censor_vectors
    )

    # get FC correlation
    seed_to_voxel_correlations = generate_correlation_mat(n_series, m_series)

    return seed_to_voxel_correlations


def load_bold_async(bold_files):
    generator = ThreadPoolExecutor().map(nib.load, bold_files)
    return [img for img in generator]


def transform_bold_imgs(bold_imgs, masker, masker_count, censor_vectors):
    series = np.empty([masker_count])

    for i, bold_img in enumerate(bold_imgs):
        transformed_img = np.swapaxes(masker.fit_transform(bold_img), 0, 1)
        if censor_vectors:
            transformed_img = np.delete(transformed_img, censor_vectors[i])
        series = np.append(series, transformed_img, axis=1)

    return series


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
        raise ValueError("x and y must have the same number of timepoints.")

    s_x = x.std(1, ddof=tr - 1)
    s_y = y.std(1, ddof=tr - 1)
    cov = np.dot(x, y.T) - tr * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])

    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


if __name__ == "__main__":
    fc(
        "/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D",
        "roi_to_mask",
        masks.SCHAEFER_PATH,
        masks.MOREL_PATH,
        "schaefer_thal",
        bold_WC="*rest*preproc_bold.nii.gz",
        cores=10,
    )
from constants import wildcards
import masks
import motion
import base

import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn import plotting, input_data, masking, image
import glob
import os
import multiprocessing
import functools as ft
import warnings
from concurrent.futures import ThreadPoolExecutor
import tqdm


# Classes
class FcData:
    def init(
        self,
        fc_type,
        subjects,
        n_masker,
        m_masker,
        task,
        censor,
        filepath,
    ):
        self.fc_type = fc_type
        self.subjects = subjects
        self.n_masker = n_masker
        self.m_masker = m_masker
        if task:
            self.bold_WC = "*" + task + wildcards.BOLD_WC
            self.censor_WC = "*" + task + wildcards.REGRESSOR_WC
        else:
            self.bold_WC = wildcards.BOLD_WC
            self.censor_WC = wildcards.REGRESSOR_WC
        self.censor = censor
        self.filepath = filepath
        self.matrix = None


# Functions
# how to print what subjects were run??? make FC class and pickle!!!
def fc(
    dataset_dir,
    fc_type,
    n_mask,
    m_mask,
    output_name,
    subjects=[],
    sessions=None,
    task=None,
    num=None,
    cores=(os.cpu_count() // 2),
    censor=True,
    save=True,
):

    dir_tree = base.DirectoryTree(dataset_dir, sessions=sessions)
    subjects = base.get_subjects(dir_tree.fmriprep_dir, dir_tree, num=num)

    n_masker = masks.get_roi_mask(n_mask)
    n_masker_count = len(np.unique(n_masker.labels_img.get_fdata())) - 1

    # instantiate second masker based on fc type
    if fc_type == "roi_to_mask":
        m_masker = masks.get_binary_mask(m_mask)
        m_masker_count = np.count_nonzero(m_masker.mask_img.get_fdata())
    elif fc_type == "roi_to_roi":
        m_masker = masks.get_roi_mask(m_mask)
        m_masker_count = len(np.unique(m_masker.labels_img.get_fdata())) - 1
    elif fc_type == "roi_to_brain":
        brain_mask_WC = (
            wildcards.BRAIN_MASK_WC
            if task is None
            else "*" + task + wildcards.BRAIN_MASK_WC
        )
        m_masker = masks.get_brain_masker(subjects, brain_mask_WC)
        m_masker_count = np.count_nonzero(m_masker.mask_img.get_fdata())
    else:
        raise ValueError(
            f"Invalid FC type. Must be one of: ['roi_to_mask', 'roi_to_roi', 'roi_to_brain']"
        )

    fc_data = FcData(
        fc_type,
        subjects,
        n_masker,
        m_masker,
        task,
        censor,
        dir_tree.fc_dir + output_name,
    )

    print(
        f"Calculating functional connectivity for each subject in parallel with {cores} processes."
    )
    correlation_list = []
    pool = multiprocessing.Pool(cores)
    correlation_list = pool.imap(
        ft.partial(fc_sub, fc_data),
        fc_data.subjects,
    )

    # convert into numpy array with shape of (n x m x subjects)
    # where n and m are number of elements in first and scond region
    fc_data.matrix = np.zeros((n_masker_count, m_masker_count, len(subjects)))
    for index, corr in enumerate(correlation_list):
        fc_data.matrix[:, :, index] = corr

    # save fc correlation to numpy array file
    if save:
        os.makedirs(dir_tree.fc_dir, exist_ok=True)
        np.save(dir_tree.fc_dir + output_name, fc_data.matrix)

    return fc_data.matrix


def fc_sub(fc_data, subject):
    # get subject's bold files based on wildcard
    bold_files = base.get_ses_files(subject, subject.fmriprep_dir, fc_data.bold_WC)
    if not any(bold_files):
        warnings.warn("No bold files found.")
        return

    # load bold files
    bold_imgs = load_bold_async(bold_files)
    TR = bold_imgs[0].shape[-1]
    if any(img.shape[-1] != TR for img in bold_imgs):
        raise ValueError("TRs must be equal for each bold file.")

    # generate censor vector and remove censored points for each bold files
    if fc_data.censor:
        censor_files = base.get_ses_files(subject, subject.fmriprep_dir, "")
        censor_vectors = [
            base.censor_motion(censor_file) for censor_file in censor_files
        ]

    n_series = transform_bold_imgs(
        bold_imgs, fc_data.n_masker, fc_data.n_masker_count, censor_vectors
    )
    m_series = transform_bold_imgs(
        bold_imgs, fc_data.m_masker, fc_data.m_masker_count, censor_vectors
    )

    # get FC correlation
    seed_to_voxel_correlations = generate_correlation_mat(n_series, m_series)

    return seed_to_voxel_correlations


def load_bold_async(bold_files):
    generator = ThreadPoolExecutor().map(nib.load, bold_files)
    return [img for img in generator]


def transform_bold_imgs(bold_imgs, masker, masker_count, censor_vectors):
    series = np.empty([masker_count])

    for i, bold_img in enumerate(bold_imgs):
        transformed_img = np.swapaxes(masker.fit_transform(bold_img), 0, 1)
        if censor_vectors:
            transformed_img = np.delete(transformed_img, censor_vectors[i])
        series = np.append(series, transformed_img, axis=1)

    return series


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
        raise ValueError("x and y must have the same number of timepoints.")

    s_x = x.std(1, ddof=tr - 1)
    s_y = y.std(1, ddof=tr - 1)
    cov = np.dot(x, y.T) - tr * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])

    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


if __name__ == "__main__":
    fc(
        "/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D",
        "roi_to_mask",
        masks.SCHAEFER_PATH,
        masks.MOREL_PATH,
        "schaefer_thal",
        bold_WC="*rest*preproc_bold.nii.gz",
        cores=10,
    )