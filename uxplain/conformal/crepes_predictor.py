"""
Conformal prediction module.

Wrapper around crepes.WrapRegressor
for interval prediction.
"""

from __future__ import annotations

from numbers import Integral
from typing import Literal

from crepes import WrapRegressor
from crepes.extras import (
    DifficultyEstimator,
    MondrianCategorizer,
)
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._validation import check_confidence, check_fit_data, prediction_frame, remember_features

ConformalMethod = Literal[
    "standard",
    "normalized",
    "mondrian",
    "normalized_mondrian",
]


class _DeterministicMondrianCategorizer(MondrianCategorizer):
    """Fit and apply empirical bin thresholds without random tie jitter."""

    def fit(self, X=None, f=None, de=None, no_bins=10):
        self.f = f
        self.de = de
        scores = f(X) if f is not None else de.apply(X)
        boundaries = np.unique(np.quantile(scores, np.linspace(0, 1, no_bins + 1)))
        # Merge tied quantile boundaries. Constant scores form a single bin.
        self.bin_thresholds = np.concatenate(([-np.inf], boundaries[1:-1], [np.inf]))
        self.fitted = True
        self.fitted_ = True
        return self

    def apply(self, X):
        scores = self.f(X) if self.f is not None else self.de.apply(X)
        # Equivalent to right-closed bins; tied scores always share a category.
        return np.searchsorted(self.bin_thresholds[1:-1], scores, side="left")


class CrepesConformalPredictor(RegressorMixin, BaseEstimator):
    """
    Wrapper for conformal regression using crepes.

    Parameters
    ----------
    estimator : object
        Any sklearn-compatible regressor.
    method : ConformalMethod, default="normalized"
        Conformal prediction method to use:

        - "standard": basic conformal prediction
        - "normalized": uses a DifficultyEstimator for
          adaptive interval widths
        - "mondrian": uses a MondrianCategorizer for
          group-conditional coverage
        - "normalized_mondrian": combines both
    n_neighbors : int, default=25
        Neighbors used to estimate difficulty for normalized methods. Capped at
        the number of proper training observations when that sample is smaller.
    """

    def __init__(
        self,
        estimator,
        *,
        method: ConformalMethod = "normalized",
        n_neighbors: int = 25,
    ):
        self.estimator = estimator
        self.method = method
        self.n_neighbors = n_neighbors

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ):
        """
        Fit the regressor and calibrate conformal prediction intervals.

        The difficulty estimator and Mondrian categorizer, when selected,
        are fitted on the training data before calibration.

        Parameters
        ----------
        X : array-like of shape (n_train, n_features)
            Features used to fit the regressor and any auxiliary estimators.
        y : array-like of shape (n_train,)
            Training target values.
        X_calib : array-like of shape (n_calib, n_features)
            Held-out calibration features with the same columns as `X`.
        y_calib : array-like of shape (n_calib,)
            Calibration target values used to compute conformity scores.

        Returns
        -------
        self : CrepesConformalPredictor
            The fitted and calibrated predictor.
        """

        if self.method not in ("standard", "normalized", "mondrian", "normalized_mondrian"):
            raise ValueError(f"Unknown conformal regression method '{self.method}'.")
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, Integral) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        self._is_fitted = False
        self.wrapper_ = None
        self.difficulty_estimator_ = None
        self.mondrian_categorizer_ = None
        self._calibration_bins = None
        X, y, X_calib, y_calib = check_fit_data(
            X, y, X_calib, y_calib,
        )

        remember_features(self, X)
        self.estimator_ = clone(self.estimator, safe=False)
        # Train
        self.estimator_.fit(
            X,
            y,
        )

        # Build calibration kwargs based on method
        calibrate_kwargs = {}

        if self.method in ("normalized", "normalized_mondrian"):
            self.difficulty_estimator_ = DifficultyEstimator()
            self.n_neighbors_ = min(self.n_neighbors, len(X))
            self.difficulty_estimator_.fit(X, y=y, k=self.n_neighbors_)
            calibrate_kwargs["de"] = self.difficulty_estimator_

        if self.method in ("mondrian", "normalized_mondrian"):
            self.mondrian_categorizer_ = _DeterministicMondrianCategorizer()
            mc_kwargs = {"de": self.difficulty_estimator_} if self.method == "normalized_mondrian" else {"f": self.estimator_.predict}
            self.mondrian_categorizer_.fit(X, **mc_kwargs)
            calibrate_kwargs["mc"] = self.mondrian_categorizer_
            self._calibration_bins = np.unique(self.mondrian_categorizer_.apply(X_calib))

        self.wrapper_ = WrapRegressor(
            self.estimator_
        )

        # Calibrate
        self.wrapper_.calibrate(
            X_calib,
            y_calib,
            **calibrate_kwargs,
        )
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for which to generate intervals, in training column order.
        confidence : float, default=0.9
            Requested coverage level, strictly between 0 and 1. Mondrian
            methods calibrate this level within their selected groups.

        Returns
        -------
        lower : numpy.ndarray of shape (n_samples,)
            Lower interval endpoints.
        upper : numpy.ndarray of shape (n_samples,)
            Upper interval endpoints. A Mondrian group absent from calibration
            receives an unbounded interval with endpoints `(-inf, inf)`.
        """

        check_is_fitted(self)

        check_confidence(confidence)
        X = prediction_frame(self, X)

        intervals = self.wrapper_.predict_int(
            X,
            confidence=confidence,
        )

        if self.mondrian_categorizer_ is not None:
            bins = self.mondrian_categorizer_.apply(X)
            uncalibrated = ~np.isin(bins, self._calibration_bins)
            # No finite group quantile exists without calibration observations.
            # Some crepes versions otherwise leave these intervals at [0, 0].
            intervals[uncalibrated, 0] = -np.inf
            intervals[uncalibrated, 1] = np.inf

        lower = intervals[:, 0]
        upper = intervals[:, 1]

        return lower, upper

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def predict(self, X):
        """Return point predictions from the fitted base estimator."""
        check_is_fitted(self)
        return self.estimator_.predict(prediction_frame(self, X))
