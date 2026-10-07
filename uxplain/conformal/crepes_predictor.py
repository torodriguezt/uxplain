"""
Conformal prediction module.

Wrapper around crepes.WrapRegressor
for interval prediction.
"""

from __future__ import annotations

from typing import Literal

from crepes import WrapRegressor
from crepes.extras import (
    DifficultyEstimator,
    MondrianCategorizer,
)
import numpy as np

from ._validation import check_confidence, check_fit_data

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


class CrepesConformalPredictor:
    """
    Wrapper for conformal regression using crepes.

    Parameters
    ----------
    model : object
        Any sklearn-compatible regressor.
    method : ConformalMethod
        Conformal prediction method to use:
        - "standard": basic conformal prediction
        - "normalized": uses a DifficultyEstimator for
          adaptive interval widths
        - "mondrian": uses a MondrianCategorizer for
          group-conditional coverage
        - "normalized_mondrian": combines both
    """

    def __init__(
        self,
        model,
        method: ConformalMethod = "normalized",
    ):
        if method not in ("standard", "normalized", "mondrian", "normalized_mondrian"):
            raise ValueError(f"Unknown conformal regression method '{method}'.")
        self.model = model
        self.method = method
        self.wrapper = None
        self.difficulty_estimator = None
        self.mondrian_categorizer = None
        self._calibration_bins = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fit conformal predictor.

        Steps:
        1. Fit model
        2. Fit difficulty estimator and/or Mondrian categorizer
        3. Calibrate conformal predictor
        """

        self.wrapper = None
        self.difficulty_estimator = None
        self.mondrian_categorizer = None
        self._calibration_bins = None
        X_train, y_train, X_calib, y_calib = check_fit_data(
            X_train, y_train, X_calib, y_calib,
        )

        # Train
        self.model.fit(
            X_train,
            y_train,
        )

        # Build calibration kwargs based on method
        calibrate_kwargs = {}

        if self.method in ("normalized", "normalized_mondrian"):
            self.difficulty_estimator = DifficultyEstimator()
            self.difficulty_estimator.fit(X_train, y=y_train)
            calibrate_kwargs["de"] = self.difficulty_estimator

        if self.method in ("mondrian", "normalized_mondrian"):
            self.mondrian_categorizer = _DeterministicMondrianCategorizer()
            mc_kwargs = {"de": self.difficulty_estimator} if self.method == "normalized_mondrian" else {"f": self.model.predict}
            self.mondrian_categorizer.fit(X_train, **mc_kwargs)
            calibrate_kwargs["mc"] = self.mondrian_categorizer
            self._calibration_bins = np.unique(self.mondrian_categorizer.apply(X_calib))

        self.wrapper = WrapRegressor(
            self.model
        )

        # Calibrate
        self.wrapper.calibrate(
            X_calib,
            y_calib,
            **calibrate_kwargs,
        )

    def predict(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.

        Parameters
        ----------
        X : np.ndarray

        confidence : float

        Returns
        -------
        lower : np.ndarray
        upper : np.ndarray
        """

        if self.wrapper is None:
            raise RuntimeError("CrepesConformalPredictor not fitted.")

        check_confidence(confidence)
        X = np.asarray(X)

        intervals = self.wrapper.predict_int(
            X,
            confidence=confidence,
        )

        if self.mondrian_categorizer is not None:
            bins = self.mondrian_categorizer.apply(X)
            uncalibrated = ~np.isin(bins, self._calibration_bins)
            # No finite group quantile exists without calibration observations.
            # Some crepes versions otherwise leave these intervals at [0, 0].
            intervals[uncalibrated, 0] = -np.inf
            intervals[uncalibrated, 1] = np.inf

        lower = intervals[:, 0]
        upper = intervals[:, 1]

        return lower, upper
