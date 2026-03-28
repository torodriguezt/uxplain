"""
Uncertainty metrics for conformal prediction.

These functions define uncertainty measures
that can be explained using SHAP.
"""

from __future__ import annotations

from typing import Callable

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
) -> Callable[[np.ndarray], np.ndarray]:
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
    Callable[[np.ndarray], np.ndarray]
        Function f(X) → interval width.
    """

    def interval_width_function(X: np.ndarray) -> np.ndarray:

        lower, upper = conformal_predictor.predict(
            X,
            confidence=confidence,
        )

        return interval_width(lower, upper)

    return interval_width_function