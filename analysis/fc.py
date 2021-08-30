from thalpy.analysis import masks, motion, denoise
from thalpy import base
from thalpy.constants import wildcards


import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn import plotting, input_data, masking, image
import glob
import os
import multiprocessing
import functools as ft
import warnings
import pickle
from threadpoolctl import threadpool_limits
import math

# Classes
class FcData:
    def __init__(
        self,
        dataset_dir,
        n_masker,
        m_masker,
        output_file,
        subjects=None,
        sessions=None,
        num=None,
        censor=True,
        is_denoise=True,
        bold_dir=None,
        bold_WC=None,
        censor_WC=None,
        cores=(os.cpu_count() // 2),
    ):
        """[summary]

        Args:
            dataset_dir ([type]): [description]
            n_masker ([type]): [description]
            m_masker ([type]): [description]
            output_file ([type]): [description]
            subjects ([type], optional): [description]. Defaults to None.
            sessions ([type], optional): [description]. Defaults to None.
            num ([type], optional): [description]. Defaults to None.
            censor (bool, optional): [description]. Defaults to True.
            cores (tuple, optional): [description]. Defaults to (os.cpu_count() // 2).
        """
        self.output_file = output_file
        self.censor = censor
        self.is_denoise = is_denoise
        self.cores = cores
        self.dir_tree = base.DirectoryTree(dataset_dir, sessions=sessions)

        if bold_dir is None:
            print("yah")
            self.bold_dir = self.dir_tree.fmriprep_dir
            print
        else:
            self.bold_dir = bold_dir

        if not subjects:
            subjects = base.get_subjects(self.bold_dir, self.dir_tree, num=num)
        self.fc_subjects = [FcSubject(subject, self.dir_tree) for subject in subjects]

        self.n_masker = n_masker
        self.n = masks.masker_count(n_masker)
        self.m_masker = m_masker
        self.m = masks.masker_count(m_masker)

        if bold_WC is None:
            self.bold_WC = wildcards.BOLD_WC
        else:
            self.bold_WC = bold_WC

        if censor_WC is None:
            self.censor_WC = wildcards.REGRESSOR_WC
        else:
            self.censor_WC = censor_WC

        self.path = self.dir_tree.analysis_dir + output_file + ".p"

    @property
    def data(self):
        data_list = [
            fc_subject.seed_to_voxel_correlations
            for fc_subject in self.fc_subjects
            if fc_subject is not None
            and fc_subject.seed_to_voxel_correlations is not None
        ]
        return np.dstack(data_list)

    def calc_fc(self):
        print(
            f"Calculating functional connectivity for each subject in parallel with {self.cores} processes."
        )

        with threadpool_limits(limits=1, user_api="blas"):
            pool = multiprocessing.Pool(self.cores)
            fc_subjects_calculated = pool.map(
                ft.partial(
                    try_fc_sub,
                    self.n_masker,
                    self.m_masker,
                    self.n,
                    self.m,
                    self.bold_WC,
                    self.censor,
                    self.censor_WC,
                    self.is_denoise,
                    self.bold_dir,
                ),
                self.fc_subjects,
            )

        for index, subject in enumerate(fc_subjects_calculated):
            self.fc_subjects[index] = subject

        # save fc correlation to numpy array file
        print("saving")
        self.save()

    def save(self, path=None):
        if path is not None:
            self.path = path
        if ".p" not in self.path:
            self.path += ".p"

        pickle.dump(self, open(self.path, "wb"), protocol=4)


class FcSubject(base.Subject):
    def __init__(self, subject, dir_tree):
        self.n_series = None
        self.m_series = None
        self.TR = None
        self.seed_to_voxel_correlations = None
        super().__init__(
            subject.name, dir_tree, dir_tree.fmriprep_dir, sessions=subject.sessions
        )


def load(filepath):
    return pickle.load(open(filepath, "rb"))


def try_fc_sub(
    n_masker,
    m_masker,
    n,
    m,
    bold_WC,
    censor,
    censor_WC,
    is_denoise,
    bold_dir,
    fc_subject,
):

    try:
        fc_subject = fc_sub(
            n_masker,
            m_masker,
            n,
            m,
            bold_WC,
            censor,
            censor_WC,
            is_denoise,
            bold_dir,
            fc_subject,
        )

    except Exception as e:
        print(e)

    return fc_subject


def fc_sub(
    n_masker,
    m_masker,
    n,
    m,
    bold_WC,
    censor,
    censor_WC,
    is_denoise,
    bold_dir,
    fc_subject,
):
    print(f"Running FC on subject: {fc_subject.name}")
    # get subject's bold files based on wildcard
    print(bold_WC)
    bold_files = base.get_ses_files(fc_subject, bold_dir, bold_WC)
    if not any(bold_files):
        warnings.warn(f"Subject: {fc_subject.name} - No bold files found.")
        return

    # load bold files
    bold_imgs = [nib.load(bold) for bold in bold_files]

    # nuissance regressors denoising
    if is_denoise:
        regressor_files = base.get_ses_files(
            fc_subject, fc_subject.fmriprep_dir, censor_WC
        )
        for img_index in np.arange(len(bold_imgs)):
            bold_imgs[img_index] = denoise.denoise(
                bold_imgs[img_index], regressor_files[img_index], default_cols=True
            )

    # generate censor vector and remove censored points for each bold files
    if censor:
        fc_subject.censor_vectors = [
            motion.censor(regressor_file) for regressor_file in regressor_files
        ]
    else:
        fc_subject.censor_vectors = None

    fc_subject.n_series = transform_bold_imgs(
        bold_imgs, n_masker, n, fc_subject.censor_vectors
    )
    fc_subject.m_series = transform_bold_imgs(
        bold_imgs, m_masker, m, fc_subject.censor_vectors
    )
    fc_subject.TR = fc_subject.n_series.shape[-1]

    # get FC correlation
    fc_subject.seed_to_voxel_correlations = generate_correlation_mat(
        fc_subject.n_series, fc_subject.m_series
    )

    return fc_subject


def transform_bold_imgs(bold_imgs, masker, masker_count, censor_vectors):
    series = []

    for i, bold_img in enumerate(bold_imgs):
        transformed_img = np.swapaxes(masker.fit_transform(bold_img), 0, 1)
        if censor_vectors:
            transformed_img = np.delete(
                transformed_img, np.where(censor_vectors[i] == 0), axis=1
            )
        series.append(transformed_img)

    return np.hstack(series)


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