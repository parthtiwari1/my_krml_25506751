import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data
    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    Returns
    -------
    None
    """
    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

def print_more_regressor_scores(y_preds: np.ndarray, y_actuals: np.ndarray, set_name: str = None):
    """
    Print additional regression metrics: R2, MSE, Explained Variance.
    
    Parameters
    ----------
    y_preds : np.ndarray
        Predicted target.
    y_actuals : np.ndarray
        Actual target.
    set_name : str, optional
        Name of the set to be printed.
    
    Returns
    -------
    None
    """
    print(f"R2 Score {set_name}: {r2_score(y_actuals, y_preds)}")
    print(f"MSE {set_name}: {mean_squared_error(y_actuals, y_preds)}")
    print(f"Explained Variance {set_name}: {explained_variance_score(y_actuals, y_preds)}")

def feature_importance(model: object, feature_names: list) -> pd.DataFrame:
    """
    Get feature importances from a tree-based model.
    
    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute (e.g., RandomForest).
    feature_names : list
        List of feature names.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with features and their importances, sorted descending.
    """
    import pandas as pd
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")

def cross_validate_model(model: object, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : object
        Model to cross-validate.
    X : np.ndarray
        Features.
    y : np.ndarray
        Target.
    cv : int, default 5
        Number of cross-validation folds.
    
    Returns
    -------
    dict
        Cross-validation scores (mean and std for RMSE).
    """
    scorer = make_scorer(rmse, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
    return {'mean_rmse': -scores.mean(), 'std_rmse': scores.std()}