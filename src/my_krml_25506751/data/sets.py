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
