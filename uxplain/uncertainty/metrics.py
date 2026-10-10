"""
Uncertainty metrics for conformal prediction.

These functions define scalar measures derived from a conformal predictor's
output that can serve as the *target* of an explainer.

Regression metrics reduce ``(lower, upper)`` to a single scalar:

- ``"width"``    : ``upper - lower``           (signed endpoint span)
- ``"lower"``    : ``lower``                   (the pessimistic prediction)
- ``"upper"``    : ``upper``                   (the optimistic prediction)
- ``"midpoint"`` : ``(lower + upper) / 2``     (the central prediction)

Classification metrics reduce ``(predict_set, predict_p, predict_proba)``
to a single scalar per sample:

- ``"set_size"``    : number of classes in the prediction set
                     (analogue of ``"width"``: ↑ = more uncertain)
- ``"credibility"`` : highest p-value across classes
                     (how compatible the most-likely class is with calibration)
- ``"confidence"``  : ``1 − second-highest p-value``
                     (how confidently we reject the runner-up class)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from ..conformal._validation import check_confidence
from ..conformal.crepes_classifier import CrepesConformalClassifier

RegressionMetric = Literal["width", "lower", "upper", "midpoint"]
ClassificationMetric = Literal["set_size", "credibility", "confidence"]
UncertaintyMetric = Literal[
    "width", "lower", "upper", "midpoint",
    "set_size", "credibility", "confidence",
]


METRIC_LABELS: dict[str, str] = {
    "width": "Interval width (signed span)",
    "lower": "Lower bound",
    "upper": "Upper bound",
    "midpoint": "Interval midpoint",
    "set_size": "Set size",
    "credibility": "Credibility (max p-value)",
    "confidence": "Confidence (1 - 2nd p-value)",
}


_REGRESSION_REDUCERS: dict[
    str, Callable[[np.ndarray, np.ndarray], np.ndarray]
] = {
    "width": lambda lower, upper: upper - lower,
    "lower": lambda lower, _upper: lower,
    "upper": lambda _lower, upper: upper,
    "midpoint": lambda lower, upper: (lower + upper) / 2.0,
}


def _set_size(cp, X: np.ndarray, confidence: float) -> np.ndarray:
    return cp.predict_set(X, confidence=confidence).sum(axis=1).astype(float)


def _credibility(cp, X: np.ndarray, _confidence: float) -> np.ndarray:
    return cp.predict_p(X).max(axis=1)


def _conformal_confidence(cp, X: np.ndarray, _confidence: float) -> np.ndarray:
    p = np.sort(cp.predict_p(X), axis=1)
    if p.shape[1] < 2:
        raise ValueError("The confidence metric requires at least two classes.")
    # 1 - second-highest p-value (well-defined for n_classes >= 2)
    return 1.0 - p[:, -2]


_CLASSIFICATION_REDUCERS: dict[
    str, Callable[[object, np.ndarray, float], np.ndarray]
] = {
    "set_size": _set_size,
    "credibility": _credibility,
    "confidence": _conformal_confidence,
}


def _finite_uncertainty(values):
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Cannot explain non-finite uncertainty values. Increase calibration "
            "data (including within Mondrian groups), lower confidence, or use "
            "a predictor and metric with finite outputs."
        )
    return values


def _check_deterministic_classifier(cp):
    if isinstance(cp, CrepesConformalClassifier) and cp.smoothing:
        raise ValueError(
            "Classification explanations require smoothing=False. Smoothed "
            "p-values depend on random draws and batch layout even with a fixed "
            "seed. Use smoothing=True only for prediction."
        )


def is_classifier_predictor(cp) -> bool:
    """Return True if ``cp`` exposes the conformal-classifier interface."""

    return hasattr(cp, "predict_set") and hasattr(cp, "predict_p")


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Compute the signed span; CQR empty sets may have a negative span."""

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

    Dispatches between regression and classification reducers based on
    the conformal predictor's interface (presence of ``predict_set``).
    Explanation targets must be deterministic functions of each row, independent
    of batch layout. Built-in classifiers with smoothing enabled are rejected;
    custom predictors must satisfy this contract themselves.

    Parameters
    ----------
    conformal_predictor
        Fitted conformal predictor (regressor or classifier).
    confidence : float
        Nominal coverage level passed to the conformal predictor.
    metric : str
        - Regression: ``"width"``, ``"lower"``, ``"upper"``, ``"midpoint"``
        - Classification: ``"set_size"``, ``"credibility"``,
          ``"confidence"``

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function ``f(X) -> np.ndarray`` of shape ``(n_samples,)``.
        Raises ``ValueError`` if the metric is non-finite, since the explainers
        require finite targets. Predictions themselves may remain unbounded.
    """

    check_confidence(confidence)
    if is_classifier_predictor(conformal_predictor):
        if metric not in _CLASSIFICATION_REDUCERS:
            raise ValueError(
                f"Metric '{metric}' is not valid for classification. "
                f"Choose from {list(_CLASSIFICATION_REDUCERS)}."
            )
        reducer = _CLASSIFICATION_REDUCERS[metric]
        _check_deterministic_classifier(conformal_predictor)

        def uncertainty_function(X: np.ndarray) -> np.ndarray:
            _check_deterministic_classifier(conformal_predictor)
            return _finite_uncertainty(reducer(conformal_predictor, X, confidence))

        return uncertainty_function

    if metric not in _REGRESSION_REDUCERS:
        raise ValueError(
            f"Metric '{metric}' is not valid for regression. "
            f"Choose from {list(_REGRESSION_REDUCERS)}."
        )

    reducer_reg = _REGRESSION_REDUCERS[metric]

    def uncertainty_function(X: np.ndarray) -> np.ndarray:
        lower, upper = conformal_predictor.predict(
            X,
            confidence=confidence,
        )
        with np.errstate(invalid="ignore", over="ignore"):
            values = reducer_reg(lower, upper)
        return _finite_uncertainty(values)

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
