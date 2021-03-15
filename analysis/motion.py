import pandas as pd


def censor_motion(filepath, df=None, threshold=0.2):
    if not df:
        df = pd.read_csv(filepath)

    censor_vector = []
    prev_motion = 0

    for index, row in enumerate(zip(df["framewise_displacement"])):
        # censor first three points
        if index < 3:
            censor_vector.append(0)
            continue

        if row[0] > threshold:
            censor_vector.append(0)
            prev_motion = index
        elif prev_motion + 1 == index or prev_motion + 2 == index:
            censor_vector.append(0)
        else:
            censor_vector.append(1)

    percent_censored = round(censor_vector.count(0) / len(censor_vector) * 100)
    print(f"\tCensored {percent_censored}% of points")
    return censor_vector