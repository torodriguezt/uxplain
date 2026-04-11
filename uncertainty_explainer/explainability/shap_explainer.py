"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

import numpy as np
import shap

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import (
    make_interval_width_function,
)


class ShapUncertaintyExplainer:
    """
    SHAP-based explainer for uncertainty metrics.
    """

    def __init__(
        self,
        cp: ConformalPredictorProtocol,
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

        self._shap_explainer: shap.Explainer | None = None

    def fit(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None:
        """
        Build SHAP explainer from background data.
        """

        width_function = (
            make_interval_width_function(
                self.cp,
                confidence=self.confidence,
            )
        )

        self._shap_explainer = shap.Explainer(
            width_function,
            X_background,
            algorithm=algorithm or self.algorithm,
        )

    def explain(
        self,
        X: np.ndarray,
    ) -> shap.Explanation:
        """
        Compute SHAP values for X.
        """

        if self._shap_explainer is None:
            raise RuntimeError(
                "Explainer not fitted. Call fit() first."
            )

        return self._shap_explainer(X)