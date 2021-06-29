import pandas as pd
import numpy as np


def censor(df, threshold=0.2, verbose=False):
    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t")

    censor_vector = np.empty((len(df.index)))
    prev_motion = 0

    for index, row in enumerate(zip(df["framewise_displacement"])):
        # censor first three points
        if index < 3:
            censor_vector[index] = 0
            continue

        if row[0] > threshold:
            censor_vector[index] = 0
            prev_motion = index
        elif prev_motion + 1 == index or prev_motion + 2 == index:
            censor_vector[index] = 0
        else:
            censor_vector[index] = 1

    if verbose:
        percent_censored = round(
            np.count_nonzero(censor_vector == 0) / len(censor_vector) * 100
        )
        print(f"\tCensored {percent_censored}% of points")
    return censor_vector