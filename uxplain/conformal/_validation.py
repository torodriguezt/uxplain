"""Shared validation for the single-output conformal backends."""

from numbers import Real

import numpy as np


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
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError(f"X_{name} must be a non-empty 2D array.")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"y_{name} must be one-dimensional (single output).")
        if len(X) != len(y):
            raise ValueError(f"X_{name} and y_{name} must have the same length.")
        arrays.extend((X, y))
    if arrays[0].shape[1] != arrays[2].shape[1]:
        raise ValueError("X_train and X_calib must have the same number of features.")
    return tuple(arrays)
