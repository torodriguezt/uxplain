"""Shared validation for the single-output conformal backends."""

from numbers import Real

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array


def check_confidence(confidence):
    if not isinstance(confidence, Real) or not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}. Pass the "
            "coverage level as a fraction, e.g. 0.9 for 90%."
        )


def check_fit_data(X_train, y_train, X_calib, y_calib):
    """Prevent broadcasting of column targets and mismatched calibration data."""
    arrays = []
    for X, y, name in (
        (X_train, y_train, "train"), (X_calib, y_calib, "calib")
    ):
        X_array, y = np.asarray(X), np.asarray(y)
        if X_array.ndim != 2 or min(X_array.shape) == 0:
            raise ValueError(f"X_{name} must be a non-empty 2D array.")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"y_{name} must be one-dimensional (single output).")
        if len(X) != len(y):
            raise ValueError(f"X_{name} and y_{name} must have the same length.")
        check_array(X, dtype=None, ensure_all_finite=True)
        arrays.extend((X if hasattr(X, "columns") else X_array, y))
    if arrays[0].shape[1] != arrays[2].shape[1]:
        raise ValueError("X_train and X_calib must have the same number of features.")
    return tuple(arrays)


def remember_features(estimator, X):
    """Record the raw feature schema before any model preprocessing."""
    estimator.n_features_in_ = X.shape[1]
    if hasattr(X, "columns"):
        estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
        estimator._feature_dtypes = X.dtypes
    elif hasattr(estimator, "feature_names_in_"):
        del estimator.feature_names_in_
        estimator._feature_dtypes = None


def prediction_frame(estimator, X):
    """Restore named columns when a masker supplies an anonymous array."""
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != estimator.n_features_in_:
        raise ValueError(f"X must have {estimator.n_features_in_} features.")
    if hasattr(estimator, "feature_names_in_"):
        names = estimator.feature_names_in_
        if hasattr(X, "columns"):
            if list(X.columns) != list(names):
                raise ValueError("DataFrame columns must match the training columns in order.")
            return X
        # Numeric perturbations must not be rounded back to integer features.
        frame = pd.DataFrame(values, columns=names)
        for name, dtype in getattr(estimator, "_feature_dtypes", {}).items():
            if isinstance(dtype, pd.CategoricalDtype):
                frame[name] = frame[name].astype(dtype)
        return frame
    return values
