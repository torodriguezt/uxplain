"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from .conformal.crepes_predictor import (
    ConformalMethod,
    CrepesConformalPredictor,
)
from .conformal.cqr_predictor import CQRConformalPredictor
from .explainability.shap_explainer import ShapUncertaintyExplainer
from .explainability.pdp_explainer import PDPUncertaintyExplainer
from .explainability.lime_explainer import LimeUncertaintyExplainer
from .plots import generate_default_plots, generate_pdp_plots, generate_lime_plots
from .protocols import (
    ConformalPredictorProtocol,
    UncertaintyExplainerProtocol,
)
from .uncertainty.metrics import UncertaintyMetric

XAIMethod = Literal["shap", "pdp", "lime"]

VALID_PLOT_KINDS = ("beeswarm", "bar", "waterfall", "summary")
VALID_PDP_PLOT_KINDS = ("pdp", "ice", "pdp_ice", "importance", "pdp_2d")
VALID_LIME_PLOT_KINDS = ("local", "global")


@dataclass
class ExplanationResult:
    """Result of explain().

    Attributes
    ----------
    lower : np.ndarray
        Lower prediction bounds.
    upper : np.ndarray
        Upper prediction bounds.
    interval_width : np.ndarray
        Width of each prediction interval.
    explanation_values : Any
        Explanation object (e.g. shap.Explanation, lime output).
    """

    lower: np.ndarray
    upper: np.ndarray
    interval_width: np.ndarray
    explanation_values: Any


class UncertaintyExplanationPipeline:
    """
    Pipeline integrating conformal prediction and uncertainty explanation.

    Supports three XAI methods selectable via ``xai_method``:

    - ``"shap"`` — SHAP-based explanation (default)
    - ``"pdp"``  — Partial Dependence Plot explanation
    - ``"lime"`` — LIME-based explanation (local or global)
    """

    def __init__(
        self,
        model=None,
        confidence: float = 0.9,
        conformal_method: ConformalMethod | Literal["cqr"] = "normalized",
        xai_method: XAIMethod = "shap",
        uncertainty_metric: UncertaintyMetric = "width",
        lime_scope: Literal["local", "global"] = "local",
        n_lime_samples: int = 5000,
        random_state: int | None = None,
        lower_model=None,
        upper_model=None,
        conformal_predictor: ConformalPredictorProtocol | None = None,
        explainer: UncertaintyExplainerProtocol | None = None,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        model
            sklearn-compatible regressor. Required for crepes methods.
            Ignored when ``conformal_method="cqr"`` or when a custom
            ``conformal_predictor`` is provided.

        confidence : float

        conformal_method : str
            Conformal prediction method. One of ``"standard"``,
            ``"normalized"``, ``"mondrian"``, ``"normalized_mondrian"``
            (crepes-based), or ``"cqr"`` (Conformalized Quantile Regression).
            Ignored if ``conformal_predictor`` is provided.

        xai_method : {"shap", "pdp", "lime"}
            Explainability method to use. Ignored if ``explainer``
            is provided.

        uncertainty_metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of the conformal interval to explain:

            - ``"width"``    — interval width ``upper - lower`` (default;
              "how uncertain is the prediction").
            - ``"lower"``    — lower bound (drivers of the pessimistic
              prediction).
            - ``"upper"``    — upper bound (drivers of the optimistic
              prediction).
            - ``"midpoint"`` — ``(lower + upper) / 2`` (drivers of the
              central prediction).

            Ignored if ``explainer`` is provided.

        lime_scope : {"local", "global"}
            Scope for the LIME explainer when ``xai_method="lime"``.
            Ignored otherwise.

        n_lime_samples : int
            Number of perturbations per sample used by LIME.
            Higher values give more stable coefficients at the cost
            of speed. Ignored when ``xai_method != "lime"``.

        random_state : int, optional
            Seed used for stochastic steps: the auto calibration
            split in ``fit()`` and, when ``xai_method="lime"``, the
            LIME perturbation sampler.
        lower_model
            Quantile regressor for the lower bound, e.g.
            ``GradientBoostingRegressor(loss="quantile", alpha=0.05)``.
            Required when ``conformal_method="cqr"``.

        upper_model
            Quantile regressor for the upper bound, e.g.
            ``GradientBoostingRegressor(loss="quantile", alpha=0.95)``.
            Required when ``conformal_method="cqr"``.

        conformal_predictor : ConformalPredictorProtocol, optional
            Custom conformal predictor. Overrides ``conformal_method``
            when provided.

        explainer : UncertaintyExplainerProtocol, optional
            Custom explainer instance. Overrides ``xai_method``
            when provided.
        """

        self.confidence = confidence
        self.xai_method = xai_method
        self.uncertainty_metric = uncertainty_metric
        self.random_state = random_state

        if conformal_predictor is not None:
            self.cp = conformal_predictor
        elif conformal_method == "cqr":
            if lower_model is None or upper_model is None:
                raise ValueError(
                    "conformal_method='cqr' requires both 'lower_model' "
                    "and 'upper_model'."
                )
            self.cp = CQRConformalPredictor(lower_model, upper_model)
        elif model is not None:
            self.cp = CrepesConformalPredictor(
                model,
                method=conformal_method,
            )
        else:
            raise ValueError(
                "Either 'model' or 'conformal_predictor' must be provided, "
                "or set conformal_method='cqr' with 'lower_model' and 'upper_model'."
            )

        if explainer is not None:
            self.explainer = explainer
        elif xai_method == "pdp":
            self.explainer = PDPUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                metric=uncertainty_metric,
            )
        elif xai_method == "shap":
            self.explainer = ShapUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                metric=uncertainty_metric,
            )
        elif xai_method == "lime":
            self.explainer = LimeUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                scope=lime_scope,
                n_lime_samples=n_lime_samples,
                random_state=random_state,
                metric=uncertainty_metric,
            )
        else:
            raise ValueError(
                f"Unknown xai_method '{xai_method}'. "
                "Choose from 'shap', 'pdp', or 'lime'."
            )

        self._is_fitted = False
        self._explainer_kwargs = None
        self._feature_names = None

    def fit(
        self,
        X_train,
        y_train,
        X_calib=None,
        y_calib=None,
        calib_size: float = 0.2,
        X_background=None,
        random_state: int | None = None,
    ):
        """
        Fit conformal predictor.

        If X_calib and y_calib are not provided, they are
        split automatically from X_train using calib_size.

        Parameters
        ----------
        X_train : np.ndarray or DataFrame
        y_train : np.ndarray or Series
        X_calib : np.ndarray or DataFrame, optional
        y_calib : np.ndarray or Series, optional
        calib_size : float
            Fraction of X_train to use for calibration
            when X_calib is not provided.
        X_background : np.ndarray, optional
            Background data for the explainer (SHAP, PDP, or LIME).
            Defaults to X_calib.
        random_state : int, optional
            Seed for the auto calibration split. Overrides the
            pipeline-level ``random_state`` for this call only.
        """

        # Extract feature names from DataFrame before converting
        if hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
            if hasattr(self.explainer, "feature_names"):
                self.explainer.feature_names = self._feature_names

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_calib is not None:
            X_calib = np.asarray(X_calib)
        if y_calib is not None:
            y_calib = np.asarray(y_calib)

        # Auto-split if calibration set not provided
        if X_calib is None or y_calib is None:
            seed = random_state if random_state is not None else self.random_state
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, test_size=calib_size,
                random_state=seed,
            )

        # Strip feature names before fitting. SHAP/LIME later call predict()
        # with numpy arrays; sklearn emits a warning on every such call when
        # the estimator was fitted with a DataFrame.
        X_train = np.asarray(X_train)
        X_calib = np.asarray(X_calib)
        y_train = np.asarray(y_train)
        y_calib = np.asarray(y_calib)

        self.cp.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
        )

        self._X_background = (
            np.asarray(X_background) if X_background is not None
            else X_calib
        )
        self._is_fitted = True
        self._explainer_kwargs = None

    def predict(
        self,
        X,
        confidence: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        confidence : float, optional
            Overrides the pipeline default if provided.

        Returns
        -------
        lower : np.ndarray
        upper : np.ndarray
        """

        self._check_is_fitted()
        self._check_X(X)

        return self.cp.predict(
            np.asarray(X),
            confidence=self.confidence if confidence is None else confidence,
        )

    def explain(
        self,
        X,
        show_plots: bool = True,
        plot_kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
        **explainer_kwargs,
    ) -> ExplanationResult:
        """
        Explain interval width uncertainty.

        Parameters
        ----------
        X : np.ndarray
        show_plots : bool
            Whether to display plots.
        plot_kind : str or list of str, optional
            For SHAP: ``"beeswarm"``, ``"bar"``, ``"waterfall"``, ``"summary"``.
            For PDP:  ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"importance"``.
            For LIME: ``"local"``, ``"global"``.
            Defaults to method-specific defaults when ``None``.
        waterfall_index : int
            Sample index for SHAP waterfall plot.
        **explainer_kwargs
            Passed to explainer.fit().
            For SHAP: ``algorithm="auto"|"permutation"|...``
            For PDP:  ``kind="average"|"individual"|"both"``,
            ``grid_resolution``, ``percentiles``, ``features``.
        """

        self._check_is_fitted()
        self._check_X(X)
        if isinstance(self.explainer, PDPUncertaintyExplainer) and np.asarray(X).shape[0] == 1:
            raise ValueError(
                "PDP does not support single-sample (local) explanations. "
                "Use xai_method='shap' or 'lime' instead."
            )
        if plot_kind is not None:
            self._resolve_plot_kinds(plot_kind, X=X)
            kinds_list = [plot_kind] if isinstance(plot_kind, str) else plot_kind
            if "pdp_2d" in kinds_list:
                features_kw = explainer_kwargs.get("features", []) or []
                if not any(isinstance(f, tuple) for f in features_kw):
                    raise ValueError(
                        "plot_kind='pdp_2d' requires at least one feature pair as a tuple, "
                        "e.g. features=[(0, 1)] or features=[0, 1, (0, 1)]."
                    )

        # Rebuild explainer if kwargs changed
        if self._explainer_kwargs != explainer_kwargs:
            self.explainer.fit(
                self._X_background,
                **explainer_kwargs,
            )
            self._explainer_kwargs = explainer_kwargs

        explanation_values = self.explainer.explain(X)

        lower, upper = self.cp.predict(
            X, confidence=self.confidence
        )
        width = upper - lower

        if show_plots:
            self.plot(
                explanation_values,
                X=X,
                kind=plot_kind,
                waterfall_index=waterfall_index,
            )

        return ExplanationResult(
            lower=lower,
            upper=upper,
            interval_width=width,
            explanation_values=explanation_values,
        )

    explain_uncertainty = explain

    def plot(
        self,
        explanation_values,
        X=None,
        kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
    ):
        """
        Generate plot(s) from existing explanation results.

        Parameters
        ----------
        explanation_values
            Output of ``explain().explanation_values``.
        X : np.ndarray, optional
            Required for SHAP ``"summary"`` plot.
        kind : str or list of str, optional
            For SHAP: ``"beeswarm"``, ``"bar"``, ``"waterfall"``, ``"summary"``.
            For PDP:  ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"importance"``.
            For LIME: ``"local"``, ``"global"``.
            Defaults to method-specific defaults when ``None``.
        waterfall_index : int
            Sample index for SHAP waterfall plot.
        """

        kinds = self._resolve_plot_kinds(kind, X=X)

        if isinstance(self.explainer, PDPUncertaintyExplainer):
            generate_pdp_plots(
                explanation_values,
                kinds=kinds,
                feature_names=self._feature_names,
            )
        elif isinstance(self.explainer, LimeUncertaintyExplainer):
            generate_lime_plots(
                explanation_values,
                kinds=kinds,
                sample_index=waterfall_index,
            )
        else:
            generate_default_plots(
                explanation_values,
                X,
                kinds=kinds,
                feature_names=self._feature_names,
                waterfall_index=waterfall_index,
            )

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() first."
            )

    def _check_X(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X_arr.shape}"
            )

    def _resolve_plot_kinds(
        self,
        kind: str | list[str] | None,
        X=None,
    ) -> list[str]:
        if isinstance(self.explainer, PDPUncertaintyExplainer):
            valid = VALID_PDP_PLOT_KINDS
        elif isinstance(self.explainer, LimeUncertaintyExplainer):
            valid = VALID_LIME_PLOT_KINDS
        else:
            valid = VALID_PLOT_KINDS
        if kind is None:
            if (
                isinstance(self.explainer, ShapUncertaintyExplainer)
                and X is not None
                and np.asarray(X).shape[0] == 1
            ):
                return ["waterfall"]
            return None  # each plot function applies its own defaults
        if isinstance(kind, str):
            kind = [kind]
        for k in kind:
            if k not in valid:
                raise ValueError(
                    f"Unknown plot kind '{k}'. "
                    f"Choose from {valid}"
                )
        return kind