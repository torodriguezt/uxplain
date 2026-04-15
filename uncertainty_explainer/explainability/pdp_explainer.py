"""
PDP explainer for uncertainty metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence
from typing import List, Optional, Tuple

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import make_interval_width_function


class _FunctionEstimator(RegressorMixin, BaseEstimator):
    """Wraps any callable as a sklearn-compatible regressor for PDP."""

    def __init__(self, func):
        self.func = func

    def fit(self, _X=None, _y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.func(X)


@dataclass
class PDPExplanation:
    """
    Result of ``PDPUncertaintyExplainer.explain()``.

    Attributes
    ----------
    values : np.ndarray, shape (n_features, grid_resolution)
        Averaged PDP values per feature (the marginal effect on
        interval width across the grid).

    grid_values : list of np.ndarray
        Grid points used for each feature.

    features : list of int
        Feature indices that were explained.

    feature_names : list of str or None
        Human-readable feature labels.

    individual : np.ndarray or None, shape (n_features, n_samples, grid_resolution)
        ICE lines — one curve per sample per feature.
        Only present when ``kind="individual"`` or ``kind="both"``.
    """

    values: np.ndarray
    grid_values: List[np.ndarray]
    features: List[int]
    feature_names: Optional[List[str]] = field(default=None)
    individual: Optional[np.ndarray] = field(default=None)


class PDPUncertaintyExplainer:
    """
    PDP-based explainer for uncertainty metrics.

    Mirrors the interface of ``ShapUncertaintyExplainer``:

    - Configuration is set at ``__init__``.
    - ``fit(X_background)`` builds the internal estimator.
    - ``explain(X)`` returns a ``PDPExplanation`` with ``.values``
      and ``.feature_names`` — analogous to ``shap.Explanation``.
    """

    def __init__(
        self,
        cp: ConformalPredictorProtocol,
        confidence: float = 0.9,
        method: str = "brute",
        features: Optional[List[int]] = None,
        grid_resolution: int = 100,
        percentiles: Tuple[float, float] = (0.05, 0.95),
        kind: str = "average",
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize PDP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        method : str
            PDP computation method. ``"brute"`` works for any estimator.

        features : list of int, optional
            Feature indices to explain. Defaults to all features.

        grid_resolution : int
            Number of grid points per feature.

        percentiles : tuple of float
            Lower and upper percentile bounds for the grid.

        kind : str
            ``"average"`` for PDP, ``"individual"`` for ICE,
            ``"both"`` for both.

        feature_names : list of str, optional
            Feature names for explanation output.
        """

        self.cp = cp
        self.confidence = confidence
        self.method = method
        self.features = features
        self.grid_resolution = grid_resolution
        self.percentiles = percentiles
        self.kind = kind
        self.feature_names = feature_names

        self._estimator: Optional[_FunctionEstimator] = None

    def fit(
        self,
        X_background: np.ndarray,
        **_,
    ) -> None:
        """
        Build the internal estimator from background data.

        Parameters
        ----------
        X_background : np.ndarray
            Data used to set the grid range for each feature.
        """

        width_function = make_interval_width_function(
            self.cp,
            confidence=self.confidence,
        )

        self._estimator = _FunctionEstimator(width_function).fit(X_background)

    def explain(
        self,
        X: np.ndarray,
    ) -> PDPExplanation:
        """
        Compute partial dependence for each feature.

        Parameters
        ----------
        X : np.ndarray
            Data to marginalize over (typically X_test).

        Returns
        -------
        PDPExplanation
            Result object with ``.values``, ``.grid_values``,
            ``.features``, ``.feature_names``, and ``.individual``.
        """

        if self._estimator is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        features = (
            self.features
            if self.features is not None
            else list(range(X.shape[1]))
        )

        values = []
        grid_values = []
        individual = [] if self.kind in ("individual", "both") else None

        for feature in features:
            pd_result = partial_dependence(
                self._estimator,
                X,
                features=[feature],
                method=self.method,
                grid_resolution=self.grid_resolution,
                percentiles=self.percentiles,
                kind=self.kind,
            )
            grid_values.append(pd_result["grid_values"][0])
            if self.kind in ("average", "both"):
                values.append(pd_result["average"][0])
            if self.kind in ("individual", "both"):
                individual.append(pd_result["individual"][0])

        explanation = PDPExplanation(
            values=np.array(values),
            grid_values=grid_values,
            features=features,
            individual=np.array(individual) if individual is not None else None,
        )

        if self.feature_names is not None:
            explanation.feature_names = self.feature_names

        return explanation
