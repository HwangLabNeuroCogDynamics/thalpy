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
from concurrent.futures import ThreadPoolExecutor
import pickle
from threadpoolctl import threadpool_limits

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
        task=None,
        num=None,
        censor=True,
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
            task ([type], optional): [description]. Defaults to None.
            num ([type], optional): [description]. Defaults to None.
            censor (bool, optional): [description]. Defaults to True.
            cores (tuple, optional): [description]. Defaults to (os.cpu_count() // 2).
        """
        self.output_file = output_file
        self.censor = censor
        self.cores = cores
        self.dir_tree = base.DirectoryTree(dataset_dir, sessions=sessions)

        if not subjects:
            subjects = base.get_subjects(
                self.dir_tree.fmriprep_dir, self.dir_tree, num=num
            )
        self.fc_subjects = [FcSubject(subject, self.dir_tree) for subject in subjects]

        self.n_masker = n_masker
        self.n = masks.masker_count(n_masker)
        self.m_masker = m_masker
        self.m = masks.masker_count(m_masker)

        if task:
            self.bold_WC = "*" + task + wildcards.BOLD_WC
            self.censor_WC = "*" + task + wildcards.REGRESSOR_WC
        else:
            self.bold_WC = wildcards.BOLD_WC
            self.censor_WC = wildcards.REGRESSOR_WC

        self.path = self.dir_tree.analysis_dir + output_file + ".p"

    @property
    def data(self):
        data_list = [
            fc_subject.seed_to_voxel_correlations
            for fc_subject in self.fc_subjects
            if fc_subject.seed_to_voxel_correlations is not None
        ]
        return np.dstack(data_list)

    def calc_fc(self):
        print(
            f"Calculating functional connectivity for each subject in parallel with {self.cores} processes."
        )

        with threadpool_limits(limits=1, user_api="blas"):
            pool = multiprocessing.Pool(self.cores)
            fc_subjects_calculated = pool.imap(
                ft.partial(
                    try_fc_sub,
                    self.n_masker,
                    self.m_masker,
                    self.n,
                    self.m,
                    self.bold_WC,
                    self.censor,
                    self.censor_WC,
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


def load_fc(filepath):
    return pickle.load(open(filepath, "rb"))


def try_fc_sub(n_masker, m_masker, n, m, bold_WC, censor, censor_WC, subject):
    try:
        fc_subject = fc_sub(
            n_masker, m_masker, n, m, bold_WC, censor, censor_WC, subject
        )
        return fc_subject
    except Exception as e:
        print(e)
        return subject


def fc_sub(n_masker, m_masker, n, m, bold_WC, censor, censor_WC, fc_subject):
    # get subject's bold files based on wildcard
    bold_files = base.get_ses_files(fc_subject, fc_subject.fmriprep_dir, bold_WC)
    if not any(bold_files):
        warnings.warn(f"Subject: {fc_subject.name} - No bold files found.")
        return

    # load bold files
    bold_imgs = load_bold_async(bold_files)

    ## might not be true -- don't need every bold file to have equal TRs
    # TR = bold_imgs[0].shape[-1]
    # if any(img.shape[-1] != TR for img in bold_imgs):
    #     warnings.warn(
    #         f"Subject: {fc_subject.name} - TRs must be equal for each bold file."
    #     )
    #     return

    # nuissance regressors denoising
    regressor_files = base.get_ses_files(fc_subject, fc_subject.fmriprep_dir, censor_WC)
    for img_index in np.arange(len(bold_imgs)):
        bold_imgs[img_index] = denoise.denoise(
            bold_imgs[img_index], regressor_files[img_index], default_cols=True
        )

    # generate censor vector and remove censored points for each bold files
    if censor:
        fc_subject.censor_vectors = [
            motion.censor(regressor_file) for regressor_file in regressor_files
        ]

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


def load_bold_async(bold_files):
    generator = ThreadPoolExecutor().map(nib.load, bold_files)
    return [img for img in generator]


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


if __name__ == "__main__":
    dir_tree = base.DirectoryTree("/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D")
    subjects = base.get_subjects(dir_tree.fmriprep_dir, dir_tree, num=1)
    mask = masks.get_brain_masker(subjects, "*rest" + wildcards.BRAIN_MASK_WC)
    fc_data = FcData(
        "/mnt/nfs/lss/lss_kahwang_hpc/data/HCP_D",
        mask,
        mask,
        "full_fc",
        task="rest",
        cores=1,
        num=1,
    )
    fc_data.calc_fc()
    fc_data.save()