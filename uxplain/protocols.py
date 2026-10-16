"""Public contracts for replaceable conformal predictors and explainers.

Components may inherit BaseEstimator for cloning and nested parameters. The
conformal explainer clones templates before fitting; it binds each explainer's
conformal_predictor parameter to the fitted conformal component. Prediction
of intervals/sets is separate from optional point prediction and probabilities.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ConformalPredictorProtocol(Protocol):
    """Regression backend, with explicit independent calibration observations.

    Implement predict(X) additionally if point prediction or default R-squared
    scoring is needed. An interval midpoint is not an implicit point estimator.
    """

    def fit(self, X, y, X_calib, y_calib) -> Any:
        """Fit on X/y, calibrate on X_calib/y_calib, and return self."""
        ...

    def predict_interval(self, X, confidence: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper arrays of shape (n_samples,)."""
        ...


class ConformalClassifierProtocol(Protocol):
    """Classification backend; columns follow the fitted classes_ order.

    Optional predict and predict_proba methods expose the underlying classifier.
    Conformal p-values are not class probabilities and need not sum to one.
    """
    classes_: np.ndarray

    def fit(self, X, y, X_calib, y_calib) -> Any:
        """Fit, calibrate on separate observations, populate classes_, return self."""
        ...

    def predict_set(self, X, confidence: float = 0.9) -> np.ndarray:
        """Return Boolean membership of shape (n_samples, n_classes)."""
        ...

    def predict_p_values(self, X) -> np.ndarray:
        """Return classwise conformal p-values of shape (n_samples, n_classes)."""
        ...


class UncertaintyExplainerProtocol(Protocol):
    """Configurable explainer; fitting prepares a fixed reference sample.

    Store configuration in constructor parameters, not in fit/explain arguments.
    Return an UncertaintyExplanation subclass to support result accessors.
    """
    conformal_predictor: Any

    def fit(self, X, y=None) -> Any:
        """Prepare reference observations, ignore y, and return self."""
        ...

    def explain(self, X) -> Any:
        """Return explanations without displaying figures; use result.plot()."""
        ...
