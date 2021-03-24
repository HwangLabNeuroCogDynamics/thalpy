from thalpy.analysis import fc
import glob
import numpy as np
from sklearn import preprocessing
import os


def pc_dataset(fc_matrix, module_file, thresholds=None, pc_axis=0, output_path=None):
    """Get participation coefficient of the functional connectivity matrix.

    Args:
        fc_file (path): filepath for .npy fc file in shape of [voxels, rois, subjects]
        module (file): filepath of mask assignment to modules
        thresholds (list(int), optional): List of integers reprenting percentile to threshold. Defaults to None.
        pc_axis (int, optional): the axis on which to perform the pc calculation. Defaults to 0.
        output_path (path, optional): output filepath for pc matrix .npy file. Defaults to None.

    Returns:
        nparray: participation coefficent matrix (values ranging from 0 to 1) in shape of
        [voxels, (thresholds, optional), subjects]
    """
    module_assignments = np.loadtxt(module_file)

    if thresholds:
        pc_matrix = np.empty(
            (fc_matrix.shape[pc_axis], len(thresholds), fc_matrix.shape[-1])
        )
        for sub_index in range(fc_matrix.shape[-1]):
            pc_matrix[:, :, sub_index] = pc_subject(
                fc_matrix[:, :, sub_index],
                module_assignments,
                thresholds,
                pc_axis=pc_axis,
            )
    else:
        pc_matrix = np.empty((fc_matrix.shape[pc_axis], fc_matrix.shape[-1]))
        for sub_index in range(fc_matrix.shape[-1]):
            pc_matrix[:, sub_index] = pc_subject(
                fc_matrix[:, :, sub_index], module_assignments, pc_axis=pc_axis
            )

    if output_path:
        np.save(output_path, pc_matrix)

    return pc_matrix


def pc_subject(matrix, module_assignments, thresholds=None, pc_axis=0):
    # threshold fc value to set any values less than percentile to 0
    if thresholds:
        sub_matrix = np.empty([matrix.shape[pc_axis], len(thresholds)])
        for thresh_index, threshold in enumerate(thresholds):
            temp_mat = matrix.copy()
            temp_mat[temp_mat < np.percentile(temp_mat, threshold)] = 0
            sub_matrix[:, thresh_index] = calc_pc(temp_mat, module_assignments)
    else:
        sub_matrix = calc_pc(matrix, module_assignments)

    # sub_matrix = preprocessing.minmax_scale(sub_matrix, feature_range=(0, 1), axis=0, copy=True)
    return sub_matrix


def calc_pc(matrix, module_assignments):
    """[summary]

    Args:
        matrix (2d nparray): 2darray containing decimal values
        pc_axis (int): axis for which pc value should be calculated

    Returns:
        pc (1d nparray): 1darray containing pc values (ranging between 0 and 1)
    """
    fc_sum = np.sum(matrix, axis=1)
    kis = np.zeros(np.shape(fc_sum))

    for module in np.unique(module_assignments):
        kis += np.square(
            np.sum(
                matrix[:, np.where(module_assignments == module)[0]],
                axis=1,
            )
            / fc_sum
        )

    pc = 1 - kis
    return pc