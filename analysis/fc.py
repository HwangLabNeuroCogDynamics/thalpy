from thalpy import regressors, denoise

from thalpy import base, masks
from thalpy.constants import wildcards

import numpy as np
from nilearn import plotting
import nibabel as nib
import os
import sys
import multiprocessing
import functools as ft
import warnings
import pickle
from threadpoolctl import threadpool_limits
import traceback


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
        bold_WC=wildcards.BOLD_WC,
        censor_WC=wildcards.REGRESSOR_WC,
    ):
        """Class to run functional connectivity analysis and hold data.

        Args:
            dataset_dir (str): filepath to root dataset directory
            n_masker (NiftiMasker): nilearn masking object - gives n dimension of fc matrix
            m_masker (NiftiMasker): nilearn masking object - gives m dimension of fc matrix
            output_file (str): name of output file
            subjects ([base.Subject], optional): List of Subjects to run fc on. Defaults to None.
            sessions (str, optional): [description]. List of sessions to include in fc calculation. Defaults to None.
            num (int, optional): Number of subjects to run. Defaults to None.
            censor (bool, optional): Whether to censor or not. Defaults to True.
            bold_dir (str, optional): Path to directory holding bold files. By default it will fmriprep directory of dataset.
            bold_WC (str, optional): Wilcard for pattern matching bold files. Defaults to "*preproc_bold.nii.gz".
            censor_WC (str, optional): Wilcard for pattern matching regressor files. Defaults to "*regressors.tsv".
        """
        self.output_file = output_file
        self.censor = censor
        self.is_denoise = is_denoise
        self.dir_tree = base.DirectoryTree(dataset_dir, sessions=sessions)
        self.bold_WC = bold_WC
        self.censor_WC = censor_WC

        if bold_dir is None:
            self.bold_dir = self.dir_tree.fmriprep_dir
        else:
            self.bold_dir = bold_dir

        if not subjects:
            subjects = base.get_subjects(self.bold_dir, self.dir_tree, num=num)
        self.fc_subjects = [FcSubject(subject, self.dir_tree)
                            for subject in subjects]

        self.n_masker = n_masker
        self.n = masks.masker_count(n_masker)
        self.m_masker = m_masker
        self.m = masks.masker_count(m_masker)

        os.makedirs(self.dir_tree.analysis_dir, exist_ok=True)
        self.path = self.dir_tree.analysis_dir + output_file

    @property
    def data(self):
        data_list = [
            fc_subject.seed_to_voxel_correlations
            for fc_subject in self.fc_subjects
            if fc_subject is not None
            and fc_subject.seed_to_voxel_correlations is not None
        ]
        return np.dstack(data_list)

    def plot(self):
        """Plots seed to voxel correlations for each subject.
        """
        for sub in self.fc_subjects:
            plot_correlations(sub.seed_to_voxel_correlations)

    def calc_fc(self, cores=8):
        """Calculates functional connectivity.

        Args:
            cores (int, optional): Number of cores to run in parallel. Defaults to 8.
        """

        print(
            f"Calculating functional connectivity for each subject in parallel with {cores} processes."
        )

        # with threadpool_limits(limits=1, user_api="blas"):
        pool = multiprocessing.Pool(cores)
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
        self.save()

    def save(self, path=None):
        if path is not None:
            self.path = path
        if ".p" not in self.path:
            self.path += ".p"

        print(f"Saving Fc Data at {self.path}")
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
        print(f"Error on Subject: {fc_subject.name}")
        traceback.print_exc()

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
    bold_files = base.get_ses_files(fc_subject, bold_dir, bold_WC)
    if not any(bold_files):
        warnings.warn(f"Subject: {fc_subject.name} - No bold files found.")
        return
    print(f"Subject Files: {bold_files}")
    # load bold files
    bold_imgs = [nib.load(bold) for bold in bold_files]

    if is_denoise or censor:
        regressor_files = base.get_ses_files(
            fc_subject, fc_subject.fmriprep_dir, censor_WC)
    # nuissance regressors denoising
    if is_denoise:
        for img_index in np.arange(len(bold_imgs)):
            bold_imgs[img_index] = denoise.denoise(
                bold_imgs[img_index], regressor_files[img_index], default_cols=True
            )

    # generate censor vector and remove censored points for each bold files
    if censor:
        fc_subject.censor_vectors = [
            regressors.censor(regressor_file) for regressor_file in regressor_files
        ]
    else:
        fc_subject.censor_vectors = None

    n_series = transform_bold_imgs(
        bold_imgs, n_masker, fc_subject.censor_vectors
    )
    m_series = transform_bold_imgs(
        bold_imgs, m_masker, fc_subject.censor_vectors
    )
    if n_series.shape[-1] != m_series.shape[-1]:  # check length
        raise Exception("Time series do not have same TRs.")
    fc_subject.TR = n_series.shape[-1]

    # get FC correlation
    fc_subject.seed_to_voxel_correlations = generate_correlation_mat(
        n_series, m_series
    )

    sys.stdout.flush()
    sys.stderr.flush()
    return fc_subject


def transform_bold_imgs(bold_imgs, masker, censor_vectors):
    series = []

    for i, bold_img in enumerate(bold_imgs):
        transformed_img = np.swapaxes(masker.fit_transform(bold_img), 0, 1)
        if censor_vectors:
            transformed_img = np.delete(
                transformed_img, np.where(censor_vectors[i] == 0), axis=1
            )
        # check for trs with 0 mean
        transformed_img = np.delete(transformed_img, np.where(
            transformed_img.mean(axis=0) == 0)[0], axis=1)

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
    cov = np.dot(x, y.T) - tr * \
        np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])

    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def plot_correlations(seed_to_voxel_correlations, vmax=0.5, vmin=-0.5):
    plotting.plot_matrix(seed_to_voxel_correlations, colorbar=True,
                         vmax=vmax, vmin=vmin)
    plotting.show()
