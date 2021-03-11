import glob
import common as cm
import numpy as np
import basic_settings as bs
from sklearn import preprocessing


def pc_dataset(fc_file, thresholds=None, pc_axis=0):
    """Get participation coefficient 

    Args:
        fc_file (path): filepath for .npy fc file in shape of [voxels, rois, subjects]
        thresholds (list(int), optional): List of integers reprenting percentile to threshold. Defaults to None.

    Returns:
        nparray: [description]
    """    
    fc_matrix = np.load(fc_file)
    if thresholds:
        pc_matrix = np.empty((fc_matrix.shape[pc_axis], len(thresholds), fc_matrix.shape[-1]))
        for sub_index in range(fc_matrix.shape[-1]):
            pc_matrix[:, :, sub_index] = pc_subject(fc_matrix[:, :, sub_index], thresholds, 
                                                    pc_axis=pc_axis)
    else:
        pc_matrix = np.empty((fc_matrix.shape[pc_axis], fc_matrix.shape[-1]))
        for sub_index in range(fc_matrix.shape[-1]):
            pc_matrix[:, sub_index] = pc_subject(fc_matrix[:, :, sub_index], pc_axis=pc_axis)

    return pc_matrix

    
def pc_subject(matrix, thresholds=None, pc_axis=0):
    if thresholds:
        sub_matrix = np.empty([matrix.shape[pc_axis], len(thresholds)])
    else:
        sub_matrix = np.empty([matrix.shape[pc_axis]])

    for voxel_index in np.arange(matrix.shape[pc_axis]):
        # threshold fc value to set any values less than percentile to 0
        if thresholds:
            for thresh_index, threshold in enumerate(thresholds):
                voxel_vector = get_voxel_vector(matrix, voxel_index, pc_axis).copy()
                voxel_vector[voxel_vector<np.percentile(voxel_vector, threshold)] = 0
                sub_matrix[voxel_index, thresh_index] = pc_voxel(voxel_vector)
        if True:
            voxel_vector = get_voxel_vector(matrix, voxel_index, pc_axis).copy()
            voxel_vector[voxel_vector < 0] = 0
            sub_matrix[voxel_index] = pc_voxel(voxel_vector)
        else:    
            voxel_vector = get_voxel_vector(matrix, voxel_index, pc_axis)
            sub_matrix[voxel_index] = pc_voxel(voxel_vector)
    
    # sub_matrix = preprocessing.minmax_scale(sub_matrix, feature_range=(0, 1), axis=0, copy=True)
    return sub_matrix

def get_voxel_vector(matrix, voxel_index, pc_axis):
    if pc_axis == 0:
        voxel_vector = matrix[voxel_index, :]
    elif pc_axis == 1:
        voxel_vector = matrix[:, voxel_index]
    
    return voxel_vector

def pc_voxel(fc_vector):
    # get sum of voxel
    sum_voxel = np.sum(fc_vector)
    pc = 1 - (np.sum(np.square(fc_vector / sum_voxel)))
    
    return pc