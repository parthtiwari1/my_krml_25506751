import pandas as pd
import numpy as np
import os

def pop_target(df: pd.DataFrame, target_col: str):
    """Extract the target variable from dataframe."""
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    return df_copy, target


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, 
              X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally with numpy if they exist."""
    if X_train is not None:
        np.save(f'{path}X_train', X_train)
    if X_val is not None:
        np.save(f'{path}X_val', X_val)
    if X_test is not None:
        np.save(f'{path}X_test', X_test)
    if y_train is not None:
        np.save(f'{path}y_train', y_train)
    if y_val is not None:
        np.save(f'{path}y_val', y_val)
    if y_test is not None:
        np.save(f'{path}y_test', y_test)


def load_sets(path='../data/processed/'):
    """Load the different locally saved sets with numpy if they exist."""
    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test
    
def subset_x_y(target, features, start_index: int, end_index: int):
    """Keep only the rows for X and y (optional) sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    start_index : int
        Index of the starting observation
    end_index : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the features
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    """
    return features[start_index:end_index], target[start_index:end_index]

import pandas as pd

def split_sets_by_time(df: pd.DataFrame, target_col: str, test_ratio: float = 0.2):
    """
    Split an ordered dataframe into train/val/test blocks by index.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already sorted by time).
    target_col : str
        Name of the target column.
    test_ratio : float, default 0.2
        Fraction used for BOTH validation and test sets (train gets the rest).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : pd.DataFrame/pd.Series
    """
    if not (0 < test_ratio < 0.5):
        raise ValueError("test_ratio must be in (0, 0.5) because val and test both use this ratio.")

    df_copy = df.copy()
    y = df_copy.pop(target_col)

    n = len(df_copy)
    if n < 5:
        raise ValueError("Dataframe too small to split by time (need at least 5 rows).")

    # Number of rows for each of val and test
    cutoff = int(round(n * test_ratio))
    if cutoff == 0:
        cutoff = 1  # ensure at least 1 row for val/test when n is small

    # Train: [0 : n - 2*cutoff), Val: [n - 2*cutoff : n - cutoff), Test: [n - cutoff : n)
    X_train, y_train = subset_x_y(target=y, features=df_copy, start_index=0,          end_index=n - 2*cutoff)
    X_val,   y_val   = subset_x_y(target=y, features=df_copy, start_index=n - 2*cutoff, end_index=n - cutoff)
    X_test,  y_test  = subset_x_y(target=y, features=df_copy, start_index=n - cutoff,   end_index=n)

    return X_train, y_train, X_val, y_val, X_test, y_test

