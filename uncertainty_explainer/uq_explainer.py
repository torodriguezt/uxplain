"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .conformal.crepes_predictor import (
    ConformalMethod,
    CrepesConformalPredictor,
)
from .explainability.shap_explainer import ShapUncertaintyExplainer
from .plots import generate_default_plots
from .protocols import (
    ConformalPredictorProtocol,
    UncertaintyExplainerProtocol,
)

VALID_PLOT_KINDS = ("beeswarm", "bar", "waterfall", "summary")


@dataclass
class ExplanationResult:
    """Result of explain_uncertainty().

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
    Pipeline integrating:

    - Conformal prediction
    - Uncertainty metrics
    - SHAP explanations
    """

    def __init__(
        self,
        model=None,
        confidence: float = 0.9,
        conformal_method: ConformalMethod = "normalized",
        conformal_predictor: ConformalPredictorProtocol | None = None,
        explainer: UncertaintyExplainerProtocol | None = None,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        model
            sklearn-compatible regressor. Required when using
            the default CrepesConformalPredictor. Can be omitted
            if a custom conformal_predictor is provided.

        confidence : float

        conformal_method : ConformalMethod
            Conformal prediction method to use.
            Ignored if conformal_predictor is provided.

        conformal_predictor : ConformalPredictorProtocol, optional
            Custom conformal predictor. If not provided,
            defaults to CrepesConformalPredictor(model).

        explainer : UncertaintyExplainerProtocol, optional
            Custom uncertainty explainer. If not provided,
            defaults to ShapUncertaintyExplainer.
        """

        self.confidence = confidence

        if conformal_predictor is not None:
            self.cp = conformal_predictor
        elif model is not None:
            self.cp = CrepesConformalPredictor(
                model,
                method=conformal_method,
            )
        else:
            raise ValueError(
                "Either 'model' or 'conformal_predictor' "
                "must be provided."
            )

        self.explainer = explainer or ShapUncertaintyExplainer(
            cp=self.cp,
            confidence=self.confidence,
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
            Background data for SHAP explainer.
            Defaults to X_calib.
        """

        # Extract feature names from DataFrame
        if hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
            if hasattr(self.explainer, "feature_names"):
                self.explainer.feature_names = self._feature_names

        # Auto-split if calibration set not provided
        if X_calib is None or y_calib is None:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, test_size=calib_size,
            )

        self.cp.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
        )

        self._X_background = (
            X_background if X_background is not None
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
            X,
            confidence=self.confidence if confidence is None else confidence,
        )

    def explain_uncertainty(
        self,
        X,
        show_plots: bool = True,
        plot_kind: str | list[str] | None = None,
        waterfall_index: int = 0,
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
            Which plots to show. Options: "beeswarm", "bar",
            "waterfall", "summary". Defaults to all.
        waterfall_index : int
            Sample index for waterfall plot.
        **explainer_kwargs
            Passed to explainer.fit(). For the default SHAP
            explainer: algorithm="auto", "permutation", etc.
        """

        self._check_is_fitted()
        self._check_X(X)

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
            kinds = self._resolve_plot_kinds(plot_kind)
            generate_default_plots(
                explanation_values,
                X,
                kinds=kinds,
                feature_names=self._feature_names,
                waterfall_index=waterfall_index,
            )

        return ExplanationResult(
            lower=lower,
            upper=upper,
            interval_width=width,
            explanation_values=explanation_values,
        )

    def plot(
        self,
        explanation_values,
        X=None,
        kind: str | list[str] = "beeswarm",
        waterfall_index: int = 0,
    ):
        """
        Generate specific plot(s) from existing results.

        Parameters
        ----------
        explanation_values
            Explanation values from explain_uncertainty().
        X : np.ndarray, optional
        kind : str or list of str
        waterfall_index : int
        """

        kinds = self._resolve_plot_kinds(kind)
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

    @staticmethod
    def _resolve_plot_kinds(
        kind: str | list[str] | None,
    ) -> list[str]:
        if kind is None:
            return list(VALID_PLOT_KINDS)
        if isinstance(kind, str):
            kind = [kind]
        for k in kind:
            if k not in VALID_PLOT_KINDS:
                raise ValueError(
                    f"Unknown plot kind '{k}'. "
                    f"Choose from {VALID_PLOT_KINDS}"
                )
        return kind