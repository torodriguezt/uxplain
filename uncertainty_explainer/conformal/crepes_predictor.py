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

ConformalMethod = Literal[
    "standard",
    "normalized",
    "mondrian",
    "normalized_mondrian",
]


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
        self.model = model
        self.method = method
        self.wrapper = None
        self.difficulty_estimator = None
        self.mondrian_categorizer = None

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
            self.mondrian_categorizer = MondrianCategorizer()
            self.mondrian_categorizer.fit(X_train, y=y_train)
            calibrate_kwargs["mc"] = self.mondrian_categorizer

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

        intervals = self.wrapper.predict_int(
            X,
            confidence=confidence,
        )

        lower = intervals[:, 0]
        upper = intervals[:, 1]

        return lower, upper