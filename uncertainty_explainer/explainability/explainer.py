"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

import numpy as np
import shap

from ..conformal.predictor import ConformalPredictor
from ..uncertainty.metrics import (
    make_interval_width_function,
)


class UncertaintyShapExplainer:
    """
    SHAP-based explainer for uncertainty metrics.
    """

    def __init__(
        self,
        cp: ConformalPredictor,
        confidence: float = 0.9,
        algorithm: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        algorithm : str
        """

        self.cp = cp
        self.confidence = confidence
        self.algorithm = algorithm

        self.explainer: shap.Explainer | None = None

    def build_explainer(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None:
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
            algorithm=algorithm or self.algorithm,
        )

    def compute_shap_values(
        self,
        X: np.ndarray,
    ) -> shap.Explanation:
        """
        Compute SHAP values.
        """

        if self.explainer is None:
            raise RuntimeError(
                "Explainer not built."
            )

        shap_values = self.explainer(X)

        return shap_values