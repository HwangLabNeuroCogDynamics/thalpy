import nibabel as nib
import os
import numpy as np
import nilearn


def load_brik(subjects, masker, brik_file, voxels, task_list, zscore=True, kind="beta"):
    if kind == "beta":
        start_index = 2
    elif kind == "tstat":
        start_index = 3

    num_tasks = len(task_list)
    stop_index = num_tasks * 3 + start_index

    final_subjects = []
    for sub_index, sub in enumerate(subjects):
        filepath = sub.deconvolve_dir + brik_file
        if not os.path.exists(filepath):
            print(
                f"Subject does not have brik file {brik_file} in {sub.deconvolve_dir}. Removing subject."
            )
            continue
        final_subjects.append(sub)

    num_subjects = len(final_subjects)
    stat_matrix = np.empty([voxels, num_tasks, num_subjects])

    if num_subjects == 0:
        raise "No subjects to run. Check BRIK filepath."

    for sub_index, sub in enumerate(final_subjects):
        print(f"loading sub {sub.name}")

        # load 3dDeconvolve bucket
        filepath = sub.deconvolve_dir + brik_file
        sub_fullstats_4d = nib.load(filepath)
        sub_fullstats_4d_data = masker.fit_transform(sub_fullstats_4d)

        # convert to 4d array with only betas, start at 2 and get every 3
        for task_index, stat_index in enumerate(np.arange(start_index, stop_index, 3)):
            stat_matrix[:, task_index, sub_index] = sub_fullstats_4d_data[stat_index, :]

        # zscore subject
        if zscore:
            stat_matrix[:, :, sub_index] = zscore_subject_2d(
                stat_matrix[:, :, sub_index]
            )

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
