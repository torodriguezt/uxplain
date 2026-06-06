"""
Protocols defining the interfaces for conformal
prediction and uncertainty explanation.

Concrete implementations (e.g. crepes, shap, LIME)
must satisfy these interfaces to be used in the pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ConformalPredictorProtocol(Protocol):
    """Interface for regression conformal predictors."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> None: ...

    def predict(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class ConformalClassifierProtocol(Protocol):
    """Interface for classification conformal predictors.

    Implementations must expose ``predict_set``, ``predict_p``,
    and a ``classes_`` attribute aligned with the model's class order.
    """

    classes_: np.ndarray

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> None: ...

    def predict_set(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> np.ndarray:
        """Return bool array of shape (n_samples, n_classes)."""
        ...

    def predict_p(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return p-value array of shape (n_samples, n_classes)."""
        ...


class UncertaintyExplainerProtocol(Protocol):
    """Interface for uncertainty explainers."""

    def fit(
        self,
        X_background: np.ndarray,
        **kwargs,
    ) -> None: ...

    def explain(
        self,
        X: np.ndarray,
    ) -> Any: ...
