"""
LIME explainer for uncertainty metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from ..protocols import ConformalPredictorProtocol
from ..uncertainty.metrics import make_interval_width_function


@dataclass
class LIMEExplanation:
    """
    Result of ``LimeUncertaintyExplainer.explain()``.

    Attributes
    ----------
    scope : {"local", "global"}
        Explanation scope used when this object was produced.

    feature_names : list of str
        Names for each feature column.

    local_coefficients : np.ndarray, shape (n_samples, n_features)
        LIME linear coefficients for each explained sample.
        A positive coefficient means that feature increases the
        predicted interval width for that sample; negative means
        it decreases it.

    global_importance : np.ndarray, shape (n_features,)
        Mean of absolute local coefficients across all samples —
        a global feature importance measure.
    """

    scope: str
    feature_names: List[str]
    local_coefficients: np.ndarray
    global_importance: np.ndarray = field(init=False)

    def __post_init__(self):
        self.global_importance = np.mean(np.abs(self.local_coefficients), axis=0)


class LimeUncertaintyExplainer:
    """
    LIME-based explainer for uncertainty metrics.

    Supports two scopes via the ``scope`` parameter:

    - ``"local"``  — one linear explanation per sample in X
    - ``"global"`` — aggregation of local explanations (mean of
      absolute coefficients), giving a dataset-level feature
      importance view

    In both cases the underlying computation is the same: LIME is
    run per sample and the results are collected into a
    ``LIMEExplanation``.  The ``scope`` value is stored in the result
    so that the plotting layer knows which view to use by default.
    """

    def __init__(
        self,
        cp: ConformalPredictorProtocol,
        confidence: float = 0.9,
        scope: Literal["local", "global"] = "local",
        n_lime_samples: int = 5000,
        feature_names: Optional[List[str]] = None,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        cp
            Fitted conformal predictor.

        confidence : float

        scope : {"local", "global"}
            Default explanation scope returned by ``explain()``.

        n_lime_samples : int
            Number of perturbed samples LIME generates per explained
            instance.  Higher values give more stable coefficients at
            the cost of speed.

        feature_names : list of str, optional
            Feature names for axis labels.

        random_state : int, optional
            Seed for LIME's perturbation sampling. Set for
            reproducible coefficients across runs.
        """

        self.cp = cp
        self.confidence = confidence
        self.scope = scope
        self.n_lime_samples = n_lime_samples
        self.feature_names = feature_names
        self.random_state = random_state

        self._lime_explainer: LimeTabularExplainer | None = None
        self._width_fn = None
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
        self._width_fn = make_interval_width_function(self.cp, self.confidence)

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
            Contains per-sample coefficients and aggregated global
            importance.  The ``scope`` attribute reflects which view
            was requested at init time.
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
                self._width_fn,
                num_features=self._n_features,
                num_samples=self.n_lime_samples,
            )
            # as_map()[1] → {feature_index: coefficient}
            for feat_idx, coef in exp.as_map()[1]:
                coefficients[i, feat_idx] = coef

        return LIMEExplanation(
            scope=self.scope,
            feature_names=fnames,
            local_coefficients=coefficients,
        )
