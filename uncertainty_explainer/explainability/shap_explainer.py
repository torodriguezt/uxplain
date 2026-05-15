"""
SHAP explainer for uncertainty metrics.
"""

from __future__ import annotations

import numpy as np
import shap

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import (
    UncertaintyMetric,
    make_uncertainty_function,
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
        feature_names: list[str] | None = None,
        metric: UncertaintyMetric = "width",
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        algorithm : str

        feature_names : list of str, optional
            Feature names for SHAP explanation output.

        metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of ``(lower, upper)`` to explain.
        """

        self.cp = cp
        self.confidence = confidence
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.metric = metric

        self._shap_explainer: shap.Explainer | None = None

    def fit(
        self,
        X_background: np.ndarray,
        algorithm: str | None = None,
    ) -> None:
        """
        Build SHAP explainer from background data.
        """

        target_function = make_uncertainty_function(
            self.cp,
            confidence=self.confidence,
            metric=self.metric,
        )

        self._shap_explainer = shap.Explainer(
            target_function,
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

        explanation = self._shap_explainer(X)

        if self.feature_names is not None:
            explanation.feature_names = self.feature_names

        explanation.metric = self.metric

        return explanation