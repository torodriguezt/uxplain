"""
PDP explainer for uncertainty metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import UncertaintyMetric, make_uncertainty_function


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
    values : list of np.ndarray
        Averaged PDP values per feature (the marginal effect on
        interval width across the grid). One 1D array per feature;
        lengths may differ because sklearn caps the grid at the
        number of unique values (e.g. categorical features).

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

    values: list[np.ndarray]
    grid_values: list[np.ndarray]
    features: list[int]
    kind: str = field(default="average")
    feature_names: list[str] | None = field(default=None)
    individual: list[np.ndarray] | None = field(default=None)
    feature_pairs: list[tuple[int, int]] | None = field(default=None)
    values_2d: list[np.ndarray] | None = field(default=None)
    grid_values_2d: list[tuple[np.ndarray, np.ndarray]] | None = field(default=None)
    metric: str = field(default="width")
    pd_results_raw: list | None = field(default=None)
    deciles_: dict | None = field(default=None)
    display_2d_: object | None = field(default=None)


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
        features: list[int] | None = None,
        grid_resolution: int = 100,
        grid_resolution_2d: int = 20,
        percentiles: tuple[float, float] = (0.05, 0.95),
        kind: str = "average",
        feature_names: list[str] | None = None,
        n_jobs: int | None = None,
        metric: UncertaintyMetric = "width",
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
            Number of parallel jobs for ``PartialDependenceDisplay``.
            ``-1`` uses all available cores.

        metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of ``(lower, upper)`` to explain.
        """

        self.cp = cp
        self.confidence = confidence
        self.method = method
        self.features = features
        self.grid_resolution = grid_resolution
        self.grid_resolution_2d = grid_resolution_2d
        self.percentiles = percentiles
        self.kind = kind
        self.feature_names = feature_names
        self.n_jobs = n_jobs
        self.metric = metric

        self._estimator: _FunctionEstimator | None = None

    def fit(
        self,
        X_background: np.ndarray,
        features: list | None = None,
        kind: str | None = None,
        grid_resolution: int | None = None,
        percentiles: tuple[float, float] | None = None,
        **_,
    ) -> None:
        """
        Build the internal estimator from background data.

        Parameters
        ----------
        X_background : np.ndarray
            Data used to set the grid range for each feature.
        features : list, optional
            Mixed list accepted by ``PartialDependenceDisplay.from_estimator``:

            - Integers or strings → 1D PDP (e.g. ``[0, 1, "age"]``)
            - Tuples of two → 2D interaction heatmap (e.g. ``[(0, 1)]``)
            - Both together → ``[0, 1, (0, 1)]``

            Unchanged when ``None``.
        kind : {"average", "individual", "both"}, optional
            Overrides the value set at ``__init__``. Unchanged when ``None``.
        grid_resolution : int, optional
            Overrides the value set at ``__init__``. Unchanged when ``None``.
        percentiles : tuple of float, optional
            Overrides the value set at ``__init__``. Unchanged when ``None``.
        """

        if kind is not None:
            self.kind = kind
        if grid_resolution is not None:
            self.grid_resolution = grid_resolution
        if percentiles is not None:
            self.percentiles = percentiles

        if features is None:
            self.features = None
            self.feature_pairs = None
        else:
            features_1d = []
            features_2d = []
            for f in features:
                if isinstance(f, tuple):
                    features_2d.append(f)
                else:
                    features_1d.append(f)

            def _resolve(idx):
                if self.feature_names and isinstance(idx, str):
                    return self.feature_names.index(idx)
                return idx

            # Keep [] (explicitly empty) separate from None (not set = all features)
            self.features = [_resolve(f) for f in features_1d]
            self.feature_pairs = [
                (_resolve(a), _resolve(b)) for a, b in features_2d
            ] or None

        target_function = make_uncertainty_function(
            self.cp,
            confidence=self.confidence,
            metric=self.metric,
        )

        self._estimator = _FunctionEstimator(target_function).fit(X_background)

    def explain(
        self,
        X: np.ndarray,
    ) -> PDPExplanation:
        """
        Compute partial dependence for each feature.

        Uses ``PartialDependenceDisplay.from_estimator`` which accepts a
        mixed feature list (ints for 1D, tuples for 2D) in a single call,
        enabling parallelism via ``n_jobs``.

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

        feature_pairs = self.feature_pairs or []

        if self.features is None:
            features_1d = list(range(X.shape[1]))
        elif not self.features and feature_pairs:
            # Only tuples passed → compute 1D for each member of the pairs
            features_1d = sorted({f for pair in feature_pairs for f in pair})
        else:
            features_1d = self.features

        # 1D: loop with partial_dependence
        values = []
        grid_values = []
        individual = [] if self.kind in ("individual", "both") else None
        pd_bunches = []
        deciles = {}
        X_arr = np.asarray(X)

        # When kind="individual" we still need average values for pdp plots,
        # so internally request "both" and always populate values.
        internal_kind = "both" if self.kind == "individual" else self.kind

        for i, feature in enumerate(features_1d):
            pd_result = partial_dependence(
                self._estimator,
                X,
                features=[feature],
                method=self.method,
                grid_resolution=self.grid_resolution,
                percentiles=self.percentiles,
                kind=internal_kind,
            )
            pd_bunches.append(pd_result)
            deciles[i] = np.percentile(X_arr[:, feature], np.arange(10, 100, 10))
            grid_values.append(pd_result["grid_values"][0])
            values.append(pd_result["average"][0])
            if self.kind in ("individual", "both"):
                individual.append(pd_result["individual"][0])

        # 2D: PartialDependenceDisplay.from_estimator with reduced grid
        values_2d = []
        grid_values_2d = []

        display_2d = None
        if feature_pairs:
            display_2d = PartialDependenceDisplay.from_estimator(
                self._estimator,
                X,
                features=feature_pairs,
                method=self.method,
                grid_resolution=self.grid_resolution_2d,
                percentiles=self.percentiles,
                kind="average",
                n_jobs=self.n_jobs,
                feature_names=self.feature_names,
            )
            for result in display_2d.pd_results:
                values_2d.append(result["average"][0])
                grid_values_2d.append((result["grid_values"][0], result["grid_values"][1]))

        explanation = PDPExplanation(
            values=values,
            grid_values=grid_values,
            features=features_1d,
            kind=self.kind,
            individual=individual,
            feature_pairs=feature_pairs,
            values_2d=values_2d or None,
            grid_values_2d=grid_values_2d or None,
            metric=self.metric,
            pd_results_raw=pd_bunches,
            deciles_=deciles,
            display_2d_=display_2d,
        )

        if self.feature_names is not None:
            explanation.feature_names = self.feature_names

        return explanation
