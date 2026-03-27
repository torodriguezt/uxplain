"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

import shap

from ..uncertainty.metrics import (
    make_interval_width_function,
)


class UncertaintyShapExplainer:
    """
    SHAP-based explainer for uncertainty metrics.
    """

    def __init__(
        self,
        cp,
        confidence: float = 0.9,
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float
        """

        self.cp = cp
        self.confidence = confidence

        self.explainer = None
        self.algorithm = "auto"

    def build_explainer(
        self,
        X_background,
        algorithm
    ):
        """
        Build SHAP explainer.
        """

        width_function = (
            make_interval_width_function(
                self.cp,
                confidence=self.confidence,
            )
        )


        self.explainer = shap.Explainer(
            width_function,
            X_background,
            algorithm=algorithm
        )

    def compute_shap_values(
        self,
        X,
    ):
        """
        Compute SHAP values.
        """

        if self.explainer is None:
            raise RuntimeError(
                "Explainer not built."
            )

        shap_values = self.explainer(X)

        return shap_values