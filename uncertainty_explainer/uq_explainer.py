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

        # TODO: Move build_explainer to fit() to avoid rebuilding on every call
        self.shap_explainer.build_explainer(
            X_background,
            algorithm
        )

        shap_values = (
            self.shap_explainer.compute_shap_values(
                X
            )
        )

        lower, upper = self.cp.predict(
            X, confidence=self.confidence
        )

        width = upper - lower

        if generate_plots:

            import matplotlib.pyplot as plt
            import shap

            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.size": 12,
                "axes.labelsize": 13,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "figure.facecolor": "white",
                "axes.facecolor": "#fafafa",
                "axes.edgecolor": "#cccccc",
                "axes.linewidth": 0.8,
            })

            shap.plots.beeswarm(shap_values, show=False)
            plt.gcf().set_size_inches(10, 6)
            plt.tight_layout()
            plt.show()

            shap.plots.bar(shap_values, show=False)
            plt.gcf().set_size_inches(10, 5)
            plt.tight_layout()
            plt.show()

            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().set_size_inches(10, 5)
            plt.tight_layout()
            plt.show()

        return {
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "shap_values": shap_values,
        }