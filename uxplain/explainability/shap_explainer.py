"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral

import numpy as np
import shap
from sklearn.base import BaseEstimator

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
        cp: ConformalPredictorProtocol,
        confidence: float = 0.9,
        algorithm: str = "auto",
        feature_names: list[str] | None = None,
        metric: UncertaintyMetric = "width",
        fast_path: bool = True,
        random_state: int | None = None,
        background_size: int | None = 100,
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        algorithm : str
            Passed to :class:`shap.Explainer` for the generic path. ``"auto"``
            uses the exact algorithm up to ten features and the permutation
            approximation beyond.

        feature_names : list of str, optional
            Feature names for SHAP explanation output.

        metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of ``(lower, upper)`` to explain.

        fast_path : bool
            Use exact TreeSHAP on the component models when the metric is an
            affine function of them and those models are tree ensembles (see
            :mod:`uxplain.explainability.fast_shap`). Falls back to the generic
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

        if background_size is not None and (
            isinstance(background_size, bool)
            or not isinstance(background_size, Integral)
            or background_size < 1
        ):
            raise ValueError("background_size must be a positive integer or None.")

        self.cp = cp
        self.confidence = confidence
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.metric = metric
        self.fast_path = fast_path
        self.random_state = random_state
        self.background_size = background_size

        #: True when the last ``explain()`` call used exact TreeSHAP.
        self.used_fast_path: bool | None = None
        #: Algorithm of the last ``explain()`` call: ``"tree"`` for the exact
        #: TreeSHAP shortcut, otherwise the generic SHAP algorithm, e.g.
        #: ``"exact"`` or ``"permutation"``.
        self.algorithm_used: str | None = None

        self._shap_explainer: shap.Explainer | None = None
        self._background: np.ndarray | None = None
        self._target_function = None
        self._rng = np.random.RandomState(random_state)

    def fit(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None:
        """
        Build SHAP explainer from background data.
        """

        self._target_function = make_uncertainty_function(
            self.cp,
            confidence=self.confidence,
            metric=self.metric,
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
                algorithm=algorithm or self.algorithm,
                seed=self.random_state,
            )
        self.used_fast_path = None
        self.algorithm_used = None

    def explain(
        self,
        X: np.ndarray,
    ) -> ShapExplanation:
        """
        Compute SHAP values for X.

        Takes the exact TreeSHAP shortcut when it applies, otherwise runs the
        generic SHAP explainer over the composed uncertainty function.

        Returns
        -------
        ShapExplanation
            Attributions, base values and the explained rows. Use
            :meth:`ShapExplanation.to_shap` for SHAP's own plotting functions.
        """

        if self._shap_explainer is None:
            raise RuntimeError(
                "Explainer not fitted. Call fit() first."
            )

        X = np.asarray(X)
        fast = None
        if self.fast_path:
            fast = tree_shap_values(
                self.cp,
                self.metric,
                X,
                self._background,
                self._target_function,
            )
        self.used_fast_path = fast is not None

        if fast is not None:
            values, base_values = fast
            self.algorithm_used = "tree"
        else:
            with _shap_random_state(self._rng):
                generic = self._shap_explainer(X)
            values, base_values = generic.values, generic.base_values
            name = type(self._shap_explainer).__name__
            self.algorithm_used = name.removesuffix("Explainer").lower() or name

        return ShapExplanation(
            values=np.asarray(values, dtype=float),
            base_values=np.asarray(base_values, dtype=float),
            data=X,
            feature_names=None if self.feature_names is None else list(self.feature_names),
            metric=self.metric,
            algorithm=self.algorithm_used,
        )
