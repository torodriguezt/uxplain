"""
Protocols defining the interfaces for conformal
prediction and uncertainty explanation.

Concrete implementations (e.g. crepes, MAPIE, shap, LIME)
must satisfy these interfaces to be used in the pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ConformalPredictorProtocol(Protocol):
    """Interface for conformal predictors."""

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


class UncertaintyExplainerProtocol(Protocol):
    """Interface for uncertainty explainers."""

    def build_explainer(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None: ...

    def compute_shap_values(
        self,
        X: np.ndarray,
    ) -> Any: ...
