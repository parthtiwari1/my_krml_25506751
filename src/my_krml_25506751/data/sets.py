import pandas as pd

def pop_target(df: pd.DataFrame, target_col: str):
    """
    Extract the target variable from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Features dataframe (all columns except target_col).
    pd.Series
        Target series (values of target_col).
    """
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    return df_copy, target
