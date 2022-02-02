import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting


def plot_TR_distribution(fc_data):
    subjects = [subject for subject in fc_data.fc_subjects if subject is not None]
    data_left_arr = np.zeros(len(subjects))
    for index, subject in enumerate(subjects):
        if subject is not None:
            data_left_arr[index] = subject.TR

    ax = sns.distplot(
        data_left_arr,
        bins=20,
        kde=False,
    )
    ax.set(
        title="Distribution of Remaining TRs after Censoring",
        xlabel="TRs",
        ylabel="Count",
    )
    plt.show()


def plot_thal(img, vmin=None):
    z_slices = [-4, 0, 4, 8, 12, 16]
    plotting.plot_stat_map(img, display_mode="z", cut_coords=z_slices, colorbar=True)
