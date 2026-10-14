"""
Protocols defining the interfaces for conformal
prediction and uncertainty explanation.

Concrete implementations (e.g. crepes, shap, LIME)
must satisfy these interfaces to be used in the pipeline.
Any object with these methods works; inheritance is not required.
See the "Extending uxplain" page of the documentation for complete examples,
including a backend that wraps MAPIE.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ConformalPredictorProtocol(Protocol):
    """Interface for regression conformal predictors.

    ``fit`` trains the underlying model(s) on ``(X_train, y_train)`` and
    calibrates on ``(X_calib, y_calib)``. ``predict`` returns the lower and
    upper bounds of the intervals at nominal coverage ``confidence``.

    The explained summaries are computed from ``predict``, so it must be a
    deterministic function of each row (independent of the batch it arrives
    in) and accept arbitrary feature combinations: SHAP and partial dependence
    evaluate it on rows that mix observed and background values. The pipeline
    clones the predictor before fitting it, so give it ``get_params`` (e.g. by
    inheriting from :class:`sklearn.base.BaseEstimator`) or make it
    deep-copyable.
    """

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> Any: ...

    def predict(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class ConformalClassifierProtocol(Protocol):
    """Interface for classification conformal predictors.

    Implementations must expose ``predict_set``, ``predict_p``,
    and a ``classes_`` attribute aligned with the model's class order.
    The same determinism and cloning requirements as for
    :class:`ConformalPredictorProtocol` apply.
    """

    classes_: np.ndarray

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> Any: ...

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
    """Interface for uncertainty explainers.

    ``fit`` receives the background data (keyword arguments of
    :meth:`~uxplain.UncertaintyExplanationPipeline.explain` are forwarded to
    it); ``explain`` returns the explanation of the rows ``X``. Return a
    subclass of :class:`~uxplain.UncertaintyExplanation` so that the result's
    ``feature_importance()`` and ``plot()`` work. If the explainer has a ``cp``
    attribute, the pipeline points it to its fitted conformal predictor.
    """

    def fit(
        self,
        X_background: np.ndarray,
        **kwargs,
    ) -> None: ...

    def explain(
        self,
        X: np.ndarray,
    ) -> Any: ...
