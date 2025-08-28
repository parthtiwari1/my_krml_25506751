# read version from installed package
from importlib.metadata import version

from .features.dates import convert_to_date, extract_date_features
from .features.preprocessing import (
    handle_missing_values, scale_features, encode_categorical,
    drop_columns, detect_outliers, remove_outliers
)
from .data.sets import (
    pop_target, save_sets, load_sets, subset_x_y, split_sets_by_time,
    split_sets_random, split_sets_random_stratified, load_data, to_numpy_arrays
)
from .models.null import create_null_model, train_null_model
from .models.performance import (
    print_regressor_scores, print_more_regressor_scores,
    feature_importance, cross_validate_model
)
__all__ = [
    "convert_to_date", "extract_date_features",
    "handle_missing_values", "scale_features", "encode_categorical",
    "drop_columns", "detect_outliers", "remove_outliers",
    "pop_target", "save_sets", "load_sets", "subset_x_y",
    "split_sets_by_time", "split_sets_random", "split_sets_random_stratified",
    "load_data", "to_numpy_arrays",
    "create_null_model", "train_null_model",
    "feature_importance", "cross_validate_model",
    "train_linear_regression", "predict_with_model",
]
# read version from installed package
from importlib.metadata import version
from .my_krml_25506751 import load_data
__version__ = version("my_krml_25506751")