from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from thalpy.analysis import plotting


def compute_PCA(
    matrix, masker=None, output_name="pca_", explained_variance=0.95, var_list=None, plot=True
):
    # set pca to explain 95% of variance
    pca = PCA(explained_variance)
    PCA_components = pca.fit_transform(matrix)

    # print variance explained by each component
    print("Explained variance:")
    print(pca.explained_variance_ratio_)

    # Plot the explained variances
    if plot:
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color="black")
        plt.xlabel("PCA features")
        plt.ylabel("variance %")
        plt.xticks(features)
        plt.show()

    if var_list is not None:
        # get and save loadings that represent variables contributions to components
        loadings = pca.components_.T
        loadings_df = pd.DataFrame(loadings, index=var_list)
        loadings_df.to_csv(output_name + "_loadings.csv")

        # get and save correlated loadings that represent each the correlations
        # between variables and components
        correlated_loadings = pca.components_.T * \
            np.sqrt(pca.explained_variance_)
        correlated_loadings = pd.DataFrame(correlated_loadings, index=var_list)
        correlated_loadings.to_csv(output_name + "_correlated_loadings.csv")

    # save each component into nifti form
    if masker:
        for index in range(pca.n_components_):
            if index == 10:
                break

            # save PC back into nifti image and visualize
            comp_array = PCA_components[:, index]
            img = masker.inverse_transform(comp_array)
            nib.save(img, f"{output_name}_component_{index}.nii")

            if plot:
                plotting.plot_thal(img)

    return PCA_components, loadings, correlated_loadings, pca.explained_variance_ratio_
