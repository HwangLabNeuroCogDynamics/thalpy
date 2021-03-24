from thalpy.analysis import fc

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_TR_distribution(fc_data):
    data_left_arr = np.zeros(len(fc_data.fc_subjects))
    for index, subject in enumerate(fc_data.fc_subjects):
        if subject.TR is not None:
            data_left_arr[index] = subject.TR

    sns.distplot(data_left_arr, kde=True)
    plt.show()