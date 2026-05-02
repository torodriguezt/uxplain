"""
LIME explainer for uncertainty metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from lime.lime_tabular import LimeTabularExplainer
import numpy as np

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import UncertaintyMetric, make_uncertainty_function


@dataclass
class LIMEExplanation:
    """
    Result of ``LimeUncertaintyExplainer.explain()``.

    Attributes
    ----------
    feature_names : list of str
        Names for each feature column.

    local_coefficients : np.ndarray, shape (n_samples, n_features)
        LIME linear coefficients for each explained sample.
        A positive coefficient means that feature increases the
        explained metric for that sample; negative means it decreases it.

    metric : str
        The uncertainty metric being explained ("width", "lower",
        "upper", or "midpoint"). Used by the plotting layer for axis
        labels.
    """

    feature_names: List[str]
    local_coefficients: np.ndarray
    metric: str = "width"


class LimeUncertaintyExplainer:
    """
    LIME-based explainer for uncertainty metrics.

    Produces one linear explanation per sample in X, collected into a
    ``LIMEExplanation`` with per-sample coefficients.
    """

    def __init__(
        self,
        cp: ConformalPredictorProtocol,
        confidence: float = 0.9,
        n_lime_samples: int = 5000,
        feature_names: Optional[List[str]] = None,
        random_state: int | None = None,
        metric: UncertaintyMetric = "width",
    ):
        """
        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        n_lime_samples : int
            Number of perturbed samples LIME generates per explained
            instance.  Higher values give more stable coefficients at
            the cost of speed.

        feature_names : list of str, optional
            Feature names for axis labels.

        random_state : int, optional
            Seed for LIME's perturbation sampling. Set for
            reproducible coefficients across runs.

        metric : {"width", "lower", "upper", "midpoint"}
            Which scalar function of ``(lower, upper)`` to explain.
        """

        self.cp = cp
        self.confidence = confidence
        self.n_lime_samples = n_lime_samples
        self.feature_names = feature_names
        self.random_state = random_state
        self.metric = metric

        self._lime_explainer: LimeTabularExplainer | None = None
        self._target_fn = None
        self._n_features: int | None = None

    def fit(
        self,
        X_background: np.ndarray,
        **_,
    ) -> None:
        """
        Build LIME explainer from background data.

        Parameters
        ----------
        X_background : np.ndarray
            Training / background data used to infer feature
            distributions for perturbation.
        """

        X_background = np.asarray(X_background)
        self._n_features = X_background.shape[1]

        fnames = self.feature_names or [str(i) for i in range(self._n_features)]

        self._lime_explainer = LimeTabularExplainer(
            X_background,
            feature_names=fnames,
            mode="regression",
            random_state=self.random_state,
        )
        self._target_fn = make_uncertainty_function(
            self.cp, self.confidence, metric=self.metric,
        )

    def explain(
        self,
        X: np.ndarray,
    ) -> LIMEExplanation:
        """
        Compute LIME explanations for X.

        Parameters
        ----------
        X : np.ndarray
            Samples to explain.

        Returns
        -------
        LIMEExplanation
            Contains per-sample LIME coefficients.
        """

        if self._lime_explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        fnames = self.feature_names or [str(i) for i in range(self._n_features)]

        coefficients = np.zeros((n_samples, self._n_features))

        for i, sample in enumerate(X):
            exp = self._lime_explainer.explain_instance(
                sample,
                self._target_fn,
                num_features=self._n_features,
                num_samples=self.n_lime_samples,
            )
            # as_map()[1] → {feature_index: coefficient}
            for feat_idx, coef in exp.as_map()[1]:
                coefficients[i, feat_idx] = coef

        return LIMEExplanation(
            feature_names=fnames,
            local_coefficients=coefficients,
            metric=self.metric,
        )
