"""
Uncertainty metrics for conformal prediction.

These functions define uncertainty measures
that can be explained using SHAP.
"""

from __future__ import annotations

import numpy as np


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute interval width.

    Parameters
    ----------
    lower : np.ndarray
        Lower prediction bounds.

    upper : np.ndarray
        Upper prediction bounds.

    Returns
    -------
    np.ndarray
        Interval widths.
    """

    return upper - lower


def make_interval_width_function(
    conformal_predictor,
    confidence: float = 0.9,
):
    """
    Create callable function for SHAP.

    This function maps:

        X → interval_width(X)

    Parameters
    ----------
    conformal_predictor
        Fitted conformal predictor.

    confidence : float

    Returns
    -------
    callable
        Function f(X) → interval width.
    """

    def interval_width_function(X):

        lower, upper = conformal_predictor.predict(
            X,
            confidence=confidence,
        )

        width = upper - lower

        return width

    return interval_width_function