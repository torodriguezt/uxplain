"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral

import numpy as np
import shap
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..conformal._validation import remember_features
from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import (
    UncertaintyMetric,
    make_uncertainty_function,
)
from .base import ShapExplanation
from .fast_shap import tree_shap_values


@contextmanager
def _shap_random_state(rng):
    """Keep SHAP's legacy global NumPy RNG local to this explainer."""
    previous = np.random.get_state()
    np.random.set_state(rng.get_state())
    try:
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(previous)


class ShapUncertaintyExplainer(BaseEstimator):
    """
    SHAP-based explainer for uncertainty metrics.

    The Shapley values are those of the interventional game
    ``v_x(S) = mean_b u(x_S, b_{-S})`` over the background rows ``b``. Both
    computational paths, exact TreeSHAP on component models and the generic
    explainer on the composed function, receive the same background rows, so
    they compute the same game.
    """

    def __init__(
        self,
        conformal_predictor: ConformalPredictorProtocol | None = None,
        *,
        confidence: float = 0.9,
        algorithm: str = "auto",
        feature_names: list[str] | None = None,
        uncertainty_metric: UncertaintyMetric = "width",
        fast_path: bool = True,
        random_state: int | None = None,
        background_size: int | None = 100,
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        conformal_predictor
            Fitted conformal predictor.

        confidence : float
            Nominal coverage in ``(0, 1)`` used to construct the explained
            intervals or prediction sets.

        algorithm : str
            Passed to `shap.Explainer` for the generic path. ``"auto"``
            uses the exact algorithm up to ten features and the permutation
            approximation beyond.

        feature_names : list of str, optional
            Feature names for SHAP explanation output.

        uncertainty_metric : str
            Scalar target: ``"width"``, ``"lower"``, ``"upper"`` or
            ``"midpoint"`` for regression; ``"set_size"``, ``"credibility"``
            or ``"confidence"`` for classification.

        fast_path : bool
            Use exact TreeSHAP on the component models when the metric is an
            affine function of them and those models are tree ensembles (see
            `uxplain.explainability.fast_shap`). Falls back to the generic
            explainer whenever the shortcut does not apply. Set to ``False`` to
            always take the generic path, e.g. to compare the two.

        random_state : int, optional
            Seed for stochastic SHAP algorithms, such as permutation SHAP, and
            for drawing the background rows.

        background_size : int or None, default=100
            Number of background rows that define the game. Larger background
            data are subsampled once, without replacement, with ``random_state``
            (0 when ``None``, which reproduces SHAP's own default subsample).
            ``None`` keeps every row in both paths; the generic path then costs
            proportionally more.
        """

        self.conformal_predictor = conformal_predictor
        self.confidence = confidence
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.uncertainty_metric = uncertainty_metric
        self.fast_path = fast_path
        self.random_state = random_state
        self.background_size = background_size



    def fit(self, X, y=None):
        """Prepare the explainer on reference observations and return self.

        Configure the explanation through the constructor or ``set_params``.
        ``y`` is accepted for compatibility and is ignored.
        """
        self.fitted_ = False
        if self.conformal_predictor is None:
            raise ValueError("A fitted conformal_predictor is required.")
        if self.background_size is not None and (
            isinstance(self.background_size, bool)
            or not isinstance(self.background_size, Integral)
            or self.background_size < 1
        ):
            raise ValueError("self.background_size must be a positive integer or None.")

        X_background = X
        self._feature_names = self.feature_names
        if self._feature_names is None and hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        remember_features(self, X if hasattr(X, "shape") else np.asarray(X))
        self._target_function = make_uncertainty_function(
            self.conformal_predictor,
            confidence=self.confidence,
            metric=self.uncertainty_metric,
        )
        background = np.asarray(X_background)
        if self.background_size is not None and len(background) > self.background_size:
            # Same sampler and default seed that SHAP applies on its own, so the
            # default game matches plain ``shap.Explainer(f, X_background)``.
            seed = 0 if self.random_state is None else self.random_state
            background = shap.utils.sample(background, self.background_size,
                                           random_state=seed)
        self._background = background

        self._rng = np.random.RandomState(self.random_state)
        # An explicit masker keeps every background row: SHAP would otherwise
        # subsample to 100 rows on its own and define a different game.
        masker = shap.maskers.Independent(background, max_samples=len(background))
        with _shap_random_state(self._rng):
            self._shap_explainer = shap.Explainer(
                self._target_function,
                masker,
                algorithm=self.algorithm,
                seed=self.random_state,
            )
        self.used_fast_path_ = None
        self.algorithm_used_ = None

        self.fitted_ = True
        return self

    def explain(
        self,
        X: np.ndarray,
    ) -> ShapExplanation:
        """
        Compute SHAP values for X.

        Takes the exact TreeSHAP shortcut when it applies, otherwise runs the
        generic SHAP explainer over the composed uncertainty function.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Rows to explain using the fitted background reference.

        Returns
        -------
        ShapExplanation
            Attributions, base values and the explained rows. Use
            `ShapExplanation.to_shap` for SHAP's own plotting functions.
        """

        check_is_fitted(self)

        X = np.asarray(X)
        fast = None
        if self.fast_path:
            fast = tree_shap_values(
                self.conformal_predictor,
                self.uncertainty_metric,
                X,
                self._background,
                self._target_function,
            )
        self.used_fast_path_ = fast is not None

        if fast is not None:
            values, base_values = fast
            self.algorithm_used_ = "tree"
        else:
            with _shap_random_state(self._rng):
                generic = self._shap_explainer(X)
            values, base_values = generic.values, generic.base_values
            name = type(self._shap_explainer).__name__
            self.algorithm_used_ = name.removesuffix("Explainer").lower() or name

        return ShapExplanation(
            values=np.asarray(values, dtype=float),
            base_values=np.asarray(base_values, dtype=float),
            data=X,
            feature_names=None if self._feature_names is None else list(self._feature_names),
            metric=self.uncertainty_metric,
            algorithm=self.algorithm_used_,
        )

    def __sklearn_is_fitted__(self):
        return getattr(self, "fitted_", False)
