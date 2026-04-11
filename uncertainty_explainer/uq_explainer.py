"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .conformal.predictor import ConformalMethod, ConformalPredictor
from .explainability.explainer import (
    UncertaintyShapExplainer,
)
from .plots import generate_default_plots

VALID_PLOT_KINDS = ("beeswarm", "bar", "waterfall", "summary")


class UncertaintyExplanationPipeline:
    """
    Pipeline integrating:

    - Conformal prediction
    - Uncertainty metrics
    - SHAP explanations
    """

    def __init__(
        self,
        model,
        confidence: float = 0.9,
        conformal_method: ConformalMethod = "normalized",
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        model
            sklearn-compatible regressor.

        confidence : float

        conformal_method : ConformalMethod
            Conformal prediction method to use.
        """

        self.model = model
        self.confidence = confidence

        self.cp = ConformalPredictor(
            self.model,
            method=conformal_method,
        )

        self.shap_explainer = UncertaintyShapExplainer(
            cp=self.cp,
            confidence=self.confidence,
        )

        self._is_fitted = False
        self._explainer_algorithm = None
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

        # Extract feature names from DataFrame before converting
        if hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)

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
        self._explainer_algorithm = None

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
            confidence=confidence or self.confidence,
        )

    def explain_uncertainty(
        self,
        X,
        algorithm: str = "auto",
        show_plots: bool = True,
        plot_kind: str | list[str] | None = None,
        waterfall_index: int = 0,
    ):
        """
        Explain interval width uncertainty using SHAP.

        Parameters
        ----------
        X : np.ndarray
        algorithm : str
            SHAP algorithm ("auto", "permutation", etc.).
            The explainer is rebuilt only when algorithm changes.
        show_plots : bool
            Whether to display plots.
        plot_kind : str or list of str, optional
            Which plots to show. Options: "beeswarm", "bar",
            "waterfall", "summary". Defaults to all.
        waterfall_index : int
            Sample index for waterfall plot.
        """

        self._check_is_fitted()
        self._check_X(X)

        # Rebuild explainer only if algorithm changed
        if self._explainer_algorithm != algorithm:
            self.shap_explainer.build_explainer(
                self._X_background,
                algorithm,
            )
            self._explainer_algorithm = algorithm

        shap_values = (
            self.shap_explainer.compute_shap_values(X)
        )

        if self._feature_names is not None:
            shap_values.feature_names = self._feature_names

        lower, upper = self.cp.predict(
            X, confidence=self.confidence
        )
        width = upper - lower

        if show_plots:
            kinds = self._resolve_plot_kinds(plot_kind)
            generate_default_plots(
                shap_values,
                X,
                kinds=kinds,
                feature_names=self._feature_names,
                waterfall_index=waterfall_index,
            )

        return {
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "shap_values": shap_values,
        }

    def plot(
        self,
        shap_values,
        X=None,
        kind: str | list[str] = "beeswarm",
        waterfall_index: int = 0,
    ):
        """
        Generate specific SHAP plot(s) from existing results.

        Parameters
        ----------
        shap_values : shap.Explanation
            SHAP values from explain_uncertainty().
        X : np.ndarray, optional
        kind : str or list of str
        waterfall_index : int
        """

        kinds = self._resolve_plot_kinds(kind)
        generate_default_plots(
            shap_values,
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