"""
PDP explainer for uncertainty metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence
from sklearn.utils.validation import check_is_fitted

from ..conformal._validation import remember_features
from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import UncertaintyMetric, make_uncertainty_function
from .base import UncertaintyExplanation


class _FunctionEstimator(RegressorMixin, BaseEstimator):
    """Wraps any callable as a sklearn-compatible regressor for PDP."""

    def __init__(self, func):
        self.func = func

    def fit(self, _X=None, _y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.func(X)


@dataclass(repr=False)
class PDPExplanation(UncertaintyExplanation):
    """
    Result of ``PDPUncertaintyExplainer.explain()``.

    Attributes
    ----------
    values : list of np.ndarray
        Averaged values of the selected uncertainty metric across each
        feature's grid. One 1D array per feature;
        lengths may differ because sklearn caps the grid at the
        number of unique values (e.g. categorical features).

    grid_values : list of np.ndarray
        Grid points used for each feature.

    features : list of int
        Feature indices that were explained.

    feature_names : list of str or None
        Human-readable feature labels.

    individual : list of np.ndarray or None
        ICE lines — one array of shape ``(n_samples, n_grid_points)`` per
        feature. Grid lengths can differ across features.
        Only present when ``kind="individual"`` or ``kind="both"``.

    importance : list of float or None
        PDP-based feature importance, one value per entry in ``features``
        (same order). Computed as the standard deviation of each feature's
        averaged partial-dependence curve (Greenwell et al., 2018): a flat
        curve means the feature barely moves the uncertainty metric
        (importance ≈ 0), a strongly varying curve means high importance.
        Only covers 1D features; 2D interaction pairs are not included.
    kind : {"average", "individual", "both"}
        Requested partial-dependence output. Averaged curves are always stored.
    feature_pairs : list of tuple of int, optional
        Pairs of feature indices for two-dimensional partial dependence.
    values_2d : list of np.ndarray, optional
        Averaged surfaces, one array per pair of feature grids.
    grid_values_2d : list of tuple of np.ndarray, optional
        Two grid axes for each pair in ``feature_pairs``.
    metric : str
        The explained summary, such as ``"width"`` or ``"set_size"``.
    """

    values: list[np.ndarray]
    grid_values: list[np.ndarray]
    features: list[int]
    kind: str = field(default="average")
    feature_names: list[str] | None = field(default=None)
    individual: list[np.ndarray] | None = field(default=None)
    importance: list[float] | None = field(default=None)
    feature_pairs: list[tuple[int, int]] | None = field(default=None)
    values_2d: list[np.ndarray] | None = field(default=None)
    grid_values_2d: list[tuple[np.ndarray, np.ndarray]] | None = field(default=None)
    metric: str = field(default="width")

    method: ClassVar[str] = "pdp"

    def _name(self, feature) -> str:
        if self.feature_names is not None and not isinstance(feature, str):
            return str(self.feature_names[feature])
        return str(feature)

    def to_frame(self) -> pd.DataFrame:
        """Long table of the averaged curves: ``feature``, ``grid_value``, ``average``."""
        rows = [
            {"feature": self._name(feature), "grid_value": float(g), "average": float(v)}
            for feature, grid, curve in zip(self.features, self.grid_values, self.values)
            for g, v in zip(grid, curve)
        ]
        return pd.DataFrame(rows, columns=["feature", "grid_value", "average"])

    def plot(self, kind=None, show: bool = True, **kwargs) -> dict:
        """Partial-dependence figures: ``"pdp"``, ``"ice"``, ``"pdp_ice"``,
        ``"pdp_2d"`` and ``"importance"``.

        Keyword arguments are passed to `uxplain.plots.generate_pdp_plots`.
        """
        from ..plots import generate_pdp_plots

        kinds = [kind] if isinstance(kind, str) else kind
        return generate_pdp_plots(self, kinds=kinds, show=show, **kwargs)

    def _describe(self) -> str:
        names = [self._name(feature) for feature in self.features]
        return f"kind={self.kind!r}, features={names}"


class PDPUncertaintyExplainer(BaseEstimator):
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
        conformal_predictor: ConformalPredictorProtocol | None = None,
        *,
        confidence: float = 0.9,
        method: str = "brute",
        features: list[int] | None = None,
        grid_resolution: int = 100,
        grid_resolution_2d: int = 20,
        percentiles: tuple[float, float] = (0.05, 0.95),
        kind: str = "average",
        feature_names: list[str] | None = None,
        n_jobs: int | None = None,
        uncertainty_metric: UncertaintyMetric = "width",
    ):
        """
        Initialize PDP explainer.

        Parameters
        ----------
        conformal_predictor
            Fitted conformal predictor.

        confidence : float
            Nominal coverage in ``(0, 1)`` used to construct the explained
            intervals or prediction sets.

        method : str
            PDP computation method. ``"brute"`` works for any estimator.

        features : list, optional
            Features to explain, in the format accepted by ``fit()``. Used
            whenever ``fit()`` is called without ``features``. Defaults to all
            features.

        grid_resolution : int
            Number of grid points per 1D feature.

        grid_resolution_2d : int
            Number of grid points per axis for 2D interaction plots.
            Kept separate from ``grid_resolution`` because 2D requires
            ``grid_resolution_2d²`` evaluations per pair.

        percentiles : tuple of float
            Lower and upper percentile bounds for the grid.

        kind : str
            ``"average"`` for PDP, ``"individual"`` for ICE,
            ``"both"`` for both.

        feature_names : list of str, optional
            Feature names for explanation output.

        n_jobs : int, optional
            Reserved for estimator compatibility; currently unused. Features
            are evaluated sequentially.

        uncertainty_metric : str
            Scalar target: ``"width"``, ``"lower"``, ``"upper"`` or
            ``"midpoint"`` for regression; ``"set_size"``, ``"credibility"``
            or ``"confidence"`` for classification.
        """

        self.conformal_predictor = conformal_predictor
        self.confidence = confidence
        self.method = method
        self.features = features
        self.grid_resolution = grid_resolution
        self.grid_resolution_2d = grid_resolution_2d
        self.percentiles = percentiles
        self.kind = kind
        self.feature_names = feature_names
        self.n_jobs = n_jobs
        self.uncertainty_metric = uncertainty_metric


    def fit(self, X, y=None):
        """Prepare the explainer on reference observations and return self.

        Configure the explanation through the constructor or ``set_params``.
        ``y`` is accepted for compatibility and is ignored.
        """
        self.fitted_ = False
        if self.conformal_predictor is None:
            raise ValueError("A fitted conformal_predictor is required.")
        X_background = X
        self._feature_names = self.feature_names
        if self._feature_names is None and hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        remember_features(self, X if hasattr(X, "shape") else np.asarray(X))
        features = self.features
        if features is None:
            self.features_ = None
            self.feature_pairs_ = None
        else:
            features_1d = []
            features_2d = []
            for f in features:
                if isinstance(f, tuple):
                    features_2d.append(f)
                else:
                    features_1d.append(f)

            def _resolve(idx):
                if self._feature_names and isinstance(idx, str):
                    return self._feature_names.index(idx)
                return idx

            # Keep [] (explicitly empty) separate from None (not set = all features)
            self.features_ = [_resolve(f) for f in features_1d]
            self.feature_pairs_ = [
                (_resolve(a), _resolve(b)) for a, b in features_2d
            ] or None

        target_function = make_uncertainty_function(
            self.conformal_predictor,
            confidence=self.confidence,
            metric=self.uncertainty_metric,
        )

        self._estimator = _FunctionEstimator(target_function).fit(X_background)

        self.fitted_ = True
        return self

    def explain(
        self,
        X: np.ndarray,
    ) -> PDPExplanation:
        """
        Compute partial dependence for each feature.

        1D features and 2D interaction pairs are computed separately, the
        latter on the coarser ``grid_resolution_2d`` grid.

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

        check_is_fitted(self)

        feature_pairs = self.feature_pairs_ or []

        if self.features_ is None:
            features_1d = list(range(X.shape[1]))
        elif not self.features_ and feature_pairs:
            # Only tuples passed → compute 1D for each member of the pairs
            features_1d = sorted({f for pair in feature_pairs for f in pair})
        else:
            features_1d = self.features_

        # 1D: loop with partial_dependence
        values = []
        grid_values = []
        individual = [] if self.kind in ("individual", "both") else None

        # When kind="individual" we still need average values for pdp plots,
        # so internally request "both" and always populate values.
        internal_kind = "both" if self.kind == "individual" else self.kind

        for feature in features_1d:
            pd_result = partial_dependence(
                self._estimator,
                X,
                features=[feature],
                method=self.method,
                grid_resolution=self.grid_resolution,
                percentiles=self.percentiles,
                kind=internal_kind,
            )
            grid_values.append(pd_result["grid_values"][0])
            values.append(pd_result["average"][0])
            if self.kind in ("individual", "both"):
                individual.append(pd_result["individual"][0])

        # 2D interactions, on a coarser grid (cost grows as grid_resolution_2d^2)
        values_2d = []
        grid_values_2d = []

        for pair in feature_pairs:
            pd_result = partial_dependence(
                self._estimator,
                X,
                features=[pair],
                method=self.method,
                grid_resolution=self.grid_resolution_2d,
                percentiles=self.percentiles,
                kind="average",
            )
            values_2d.append(pd_result["average"][0])
            grid_values_2d.append(
                (pd_result["grid_values"][0], pd_result["grid_values"][1])
            )

        # PDP-based importance (Greenwell et al., 2018): flatness of the
        # averaged curve. ddof=0 keeps it well-defined for single-point grids.
        importance = [float(np.std(v)) for v in values]

        explanation = PDPExplanation(
            values=values,
            grid_values=grid_values,
            features=features_1d,
            kind=self.kind,
            individual=individual,
            importance=importance,
            feature_pairs=feature_pairs,
            values_2d=values_2d or None,
            grid_values_2d=grid_values_2d or None,
            metric=self.uncertainty_metric,
        )

        if self._feature_names is not None:
            explanation.feature_names = self._feature_names

        return explanation

    def __sklearn_is_fitted__(self):
        return getattr(self, "fitted_", False)
