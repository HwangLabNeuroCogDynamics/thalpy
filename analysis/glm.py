import nibabel as nib
import os
import numpy as np
import nilearn
from thalpy import masks
import glob


def load_brik(subjects, masker, brik_file, task_list, zscore=True, kind="beta", start_index=None, stop_index=None):
    starting_dir = os.getcwd()
    if kind == "beta" and not start_index:
        start_index = 2
    elif kind == "tstat" and not start_index:
        start_index = 3

    num_tasks = len(task_list)
    if not stop_index:
        stop_index = num_tasks * 3 + start_index
    voxels = masks.masker_count(masker)

    final_subjects = []
    for sub_index, sub in enumerate(subjects):
        os.chdir(sub.deconvolve_dir)
        brik_files = glob.glob(brik_file)
        if len(brik_files) > 1:
            raise Exception("More than 1 brik files found. Check pattern.")

        if len(brik_files) == 0:
            print(
                f"Subject does not have brik file {brik_file} in {sub.deconvolve_dir}. Removing subject."
            )
            continue
        final_subjects.append(sub)

    num_subjects = len(final_subjects)
    stat_matrix = np.empty([voxels, num_tasks, num_subjects])

    if num_subjects == 0:
        raise Exception("No subjects to run. Check BRIK filepath.")

    for sub_index, sub in enumerate(final_subjects):
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        os.chdir(sub.deconvolve_dir)
        brik_files = glob.glob(brik_file)
        filepath = brik_files[0]

        brik_img = nib.load(filepath)

        if len(brik_img.shape) == 4:
            brik_img = nib.Nifti1Image(brik_img.get_fdata(), brik_img.affine)
        if len(brik_img.shape) == 5:
            brik_img = nib.Nifti1Image(np.squeeze(
                brik_img.get_fdata()), brik_img.affine)

        sub_brik_masked = masker.fit_transform(brik_img)

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, stat_index in enumerate(np.arange(start_index, stop_index, 3)):
            stat_matrix[:, task_index,
                        sub_index] = sub_brik_masked[stat_index, :]

        # zscore subject
        if zscore:
            stat_matrix[:, :, sub_index] = zscore_subject_2d(
                stat_matrix[:, :, sub_index]
            )

    os.chdir(starting_dir)
    return stat_matrix


def zscore_subject_2d(matrix):
    # 2D matrix shape [voxels, tasks]
    zscored_matrix = np.empty(matrix.shape)

    # zscore across 2d (voxel, task) within subject
    sample_mean = np.mean(matrix)
    std_dev = np.std(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            zscored_matrix[i, j] = (matrix[i, j] - sample_mean) / std_dev

    return zscored_matrix
