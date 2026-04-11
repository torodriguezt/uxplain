"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

import numpy as np

from .conformal.predictor import ConformalMethod, ConformalPredictor
from .explainability.explainer import (
    UncertaintyShapExplainer,
)
from .plots import generate_default_plots


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

        # Conformal predictor
        self.cp = ConformalPredictor(
            self.model,
            method=conformal_method,
        )


        self.shap_explainer = UncertaintyShapExplainer(
            cp=self.cp,
            confidence=self.confidence,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_background=None,
        algorithm="auto",
    ):
        """
        Fit conformal predictor and build SHAP explainer.

        Parameters
        ----------
        X_background : np.ndarray, optional
            Background data for SHAP explainer. Defaults to X_calib.

        algorithm : str
            SHAP algorithm to use.
        """

        self.cp.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
        )

        if X_background is None:
            X_background = X_calib

        self.shap_explainer.build_explainer(
            X_background,
            algorithm,
        )

    def explain_uncertainty(
        self,
        X,
        show_plots=True,
    ):
        """
        Explain interval width uncertainty.
        """

        shap_values = (
            self.shap_explainer.compute_shap_values(
                X
            )
        )

        lower, upper = self.cp.predict(
            X, confidence=self.confidence
        )

        width = upper - lower

        if show_plots:
            generate_default_plots(shap_values, X)

        return {
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "shap_values": shap_values,
        }