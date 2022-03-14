"""
The searchlight is a widely used approach for the study of the
fine-grained patterns of information in fMRI analysis, in which
multivariate statistical relationships are iteratively tested in the
neighborhood of each location of a domain.
"""
# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#
# License: simplified BSD

import time
import sys
import warnings

import numpy as np

from joblib import Parallel, delayed, cpu_count
from sklearn.exceptions import ConvergenceWarning

from nilearn import masking
from nilearn.image.resampling import coord_transform
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn._utils import check_niimg_4d
import sys


def search_light(X, func, args, A, n_jobs=-1, verbose=0):
    """Function for computing a search_light
    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.
    func : function
        to apply to X and y
    args : list
        additional arguments for func
    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.
    %(n_jobs_all)s
    %(verbose0)s
    Returns
    -------
    output : list of length (number of rows in A)
    """

    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        output = Parallel(n_jobs=n_jobs, verbose=verbose, backend="multiprocessing")(
            delayed(_group_iter_search_light)(
                A.rows[list_i],
                X, func, args, thread_id + 1, A.shape[0], verbose)
            for thread_id, list_i in enumerate(group_iter))
    return output


class GroupIterator(object):
    """Group iterator
    Provides group of features for search_light loop
    that may be used with Parallel.
    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s
    """

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i


def _group_iter_search_light(list_rows, X, func, args, thread_id, total, verbose=0):
    """Function for grouped iterations of search_light
    Parameters
    -----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).
    estimator : estimator object implementing 'fit'
        object to use to fit the data
    X : array-like of shape at least 2D
        data to fit.
    func : function
        to apply to X
    args : list
        additional arguments for func
    thread_id : int
        process id, used for display.
    total : int
        Total number of voxels, used for display
    verbose : int, optional
        The verbosity level. Default is 0
    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    output = []
    t0 = time.time()
    for i, row in enumerate(list_rows):
        print(list_rows.shape)
        print(X.shape)
        print(X[:, row].shape)
        output.append(func(X[:, row], row, *args))

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if (i % step == 0):
                # If there is only one job, progress information is fixed
                if total == len(list_rows):
                    crlf = "\r"
                else:
                    crlf = "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100. - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    "Job #%d, processed %d/%d voxels "
                    "(%0.2f%%, %i seconds remaining)%s"
                    % (thread_id, i, len(list_rows), percent, remaining, crlf))
                sys.stdout.flush()
                sys.stderr.flush()
    return output

##############################################################################
# Class for search_light #####################################################
##############################################################################


class SearchLight():
    """Implement search_light analysis using an arbitrary type of classifier.
    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.
    func : function 
        to apply to X and y
    args : list
        additional arguments for func
    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving voxels on which searchlight should be
        computed.
    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.
    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data
    %(n_jobs)s
    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(verbose0)s
    Notes
    ------
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.
    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(self, mask_img, func, args, process_mask_img=None, radius=2.,
                 n_jobs=1,
                 verbose=0):
        self.mask_img = mask_img
        self.func = func
        self.args = args
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self, imgs):
        """Fit the searchlight
        Parameters
        ----------
        imgs : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            4D image.
        """

        # check if image is 4D
        imgs = check_niimg_4d(imgs)

        # Get the seeds
        process_mask_img = self.process_mask_img
        if self.process_mask_img is None:
            process_mask_img = self.mask_img

        # Compute world coordinates of the seeds
        process_mask, process_mask_affine = masking._load_mask_img(
            process_mask_img)
        process_mask_coords = np.where(process_mask != 0)
        process_mask_coords = coord_transform(
            process_mask_coords[0], process_mask_coords[1],
            process_mask_coords[2], process_mask_affine)
        process_mask_coords = np.asarray(process_mask_coords).T

        
        X, A = _apply_mask_and_get_affinity(
            process_mask_coords, imgs, self.radius, True,
            mask_img=self.mask_img)

        print(process_mask.shape)
        print(X.shape)
        
        self.output = search_light(X, self.func, self.args, A,
                                   self.n_jobs, self.verbose)
        return self.output, self.output_3d
