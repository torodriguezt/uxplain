"""
Uncertainty metrics for conformal prediction.

These functions define scalar measures derived from a conformal predictor's
output ``(lower, upper)`` that can serve as the *target* of an explainer.
The default is the interval width — the most common notion of how
uncertain a prediction is — but explaining the lower bound, upper bound,
or midpoint is sometimes more informative:

- ``"width"``    : ``upper - lower``           (how wide the interval is)
- ``"lower"``    : ``lower``                   (the pessimistic prediction)
- ``"upper"``    : ``upper``                   (the optimistic prediction)
- ``"midpoint"`` : ``(lower + upper) / 2``     (the central prediction)
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


UncertaintyMetric = Literal["width", "lower", "upper", "midpoint"]


METRIC_LABELS: dict[str, str] = {
    "width": "Interval width",
    "lower": "Lower bound",
    "upper": "Upper bound",
    "midpoint": "Interval midpoint",
}


_METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "width": lambda lower, upper: upper - lower,
    "lower": lambda lower, _upper: lower,
    "upper": lambda _lower, upper: upper,
    "midpoint": lambda lower, upper: (lower + upper) / 2.0,
}


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


def metric_label(metric: UncertaintyMetric) -> str:
    """Return a human-readable label for ``metric`` (used by plots)."""

    if metric not in METRIC_LABELS:
        raise ValueError(
            f"Unknown uncertainty metric '{metric}'. "
            f"Choose from {list(METRIC_LABELS)}."
        )
    return METRIC_LABELS[metric]


def make_uncertainty_function(
    conformal_predictor,
    confidence: float = 0.9,
    metric: UncertaintyMetric = "width",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a callable ``X -> uncertainty_metric(X)`` for an explainer.

    The returned function calls the conformal predictor and reduces its
    ``(lower, upper)`` output to a single scalar per sample according to
    ``metric``.

    Parameters
    ----------
    conformal_predictor
        Fitted conformal predictor.
    confidence : float
        Nominal coverage level passed to ``conformal_predictor.predict``.
    metric : {"width", "lower", "upper", "midpoint"}
        Which scalar to return for each sample.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function ``f(X) -> np.ndarray`` of shape ``(n_samples,)``.
    """

    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown uncertainty metric '{metric}'. "
            f"Choose from {list(_METRIC_FUNCTIONS)}."
        )

    reducer = _METRIC_FUNCTIONS[metric]

    def uncertainty_function(X: np.ndarray) -> np.ndarray:
        lower, upper = conformal_predictor.predict(
            X,
            confidence=confidence,
        )
        return reducer(lower, upper)

    return uncertainty_function


def make_interval_width_function(
    conformal_predictor,
    confidence: float = 0.9,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Backwards-compatible alias for ``make_uncertainty_function(..., metric="width")``.
    """

    return make_uncertainty_function(
        conformal_predictor,
        confidence=confidence,
        metric="width",
    )
