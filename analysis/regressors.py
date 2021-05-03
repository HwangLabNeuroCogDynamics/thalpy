import pandas as pd
import numpy as np
import logging

DEFAULT_COLUMNS = [
    "csf",
    "white_matter",
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
    "trans_x",
    "trans_x_power2",
    "trans_x_derivative1",
    "trans_x_derivative1_power2",
    "trans_y",
    "trans_y_power2",
    "trans_y_derivative1",
    "trans_y_derivative1_power2",
    "trans_z",
    "trans_z_power2",
    "trans_z_derivative1",
    "trans_z_derivative1_power2",
    "rot_x",
    "rot_x_power2",
    "rot_x_derivative1",
    "rot_x_derivative1_power2",
    "rot_y",
    "rot_y_power2",
    "rot_y_derivative1",
    "rot_y_derivative1_power2",
    "rot_z",
    "rot_z_power2",
    "rot_z_derivative1",
    "rot_z_derivative1_power2",
]


def load_regressors(regressor_file, cols=None, default_cols=True, verbose=False):
    """Loads regressor tsv into df with selected columns and fills in NaN values.

    Args:
        regressor_file (str): filepath to regressor tsv
        cols ([str], optional): List of column names. Defaults to None.
        default_cols (bool, optional): If true will use default columns: motion, csf, white matter, and compcor. Defaults to True.
        verbose (bool, optional): If true, will log more info. Defaults to False.

    Returns:
        regressor_df (Dataframe): Dataframe containing regressors from selected columns.
        regressor_names ([str]): List of regressor names.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    df_orig = pd.read_csv(regressor_file, sep="\t")

    regressor_df = pd.DataFrame()
    regressor_names = []
    if default_cols:
        regressor_df = df_orig[DEFAULT_COLUMNS]
        regressor_names = DEFAULT_COLUMNS
    if cols:
        regressor_df = regressor_df.append(df_orig[cols])
        regressor_names.append(cols)

    for col in regressor_df.columns:
        sum_nan = sum(regressor_df[col].isnull())
        if sum_nan > 0:
            logging.info("Filling in " + str(sum_nan) + " NaN value for " + col)
            regressor_df.loc[np.isnan(regressor_df[col]), col] = np.mean(
                regressor_df[col]
            )
    logging.info("# of Confound Regressors: " + str(len(regressor_df.columns)))

    return regressor_df, regressor_names
