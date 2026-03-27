"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

import numpy as np

from .conformal.predictor import ConformalPredictor
from .explainability.explainer import (
    UncertaintyShapExplainer,
)


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
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        model
            sklearn-compatible regressor.

        confidence : float
        """

        self.model = model
        self.confidence = confidence

        # Conformal predictor
        self.cp = ConformalPredictor(
            self.model
        )


        self.shap_explainer = (
        UncertaintyShapExplainer(
        cp=self.cp,
        confidence=self.confidence,
    )
)

    def fit(
        self,
        X_train,
        y_train,
        X_calib,
        y_calib,
    ):
        """
        Fit conformal predictor.
        """

        self.cp.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
        )

    def explain_uncertainty(
        self,
        X,
        X_background=None,
        algorithm = "auto",
        generate_plots=True,
    ):
        """
        Explain interval width uncertainty.
        """

        if X_background is None:
            X_background = X[:100]

        # Build SHAP explainer
        self.shap_explainer.build_explainer(
            X_background,
            algorithm
        )

        shap_values = (
            self.shap_explainer.compute_shap_values(
                X
            )
        )

        lower, upper = self.cp.predict(X)

        width = upper - lower

        if generate_plots:

            import shap

            shap.plots.beeswarm(
                shap_values
            )

            shap.plots.bar(
                shap_values
            )

            shap.plots.waterfall(
                shap_values[0]
            )

        return {
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "shap_values": shap_values,
        }