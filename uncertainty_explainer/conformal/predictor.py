"""
Conformal prediction module.

Wrapper around crepes.WrapRegressor
for interval prediction.
"""

from __future__ import annotations

import numpy as np
from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator


class ConformalPredictor:
    """
    Wrapper for conformal regression using crepes.

    Parameters
    ----------
    model : object
        Any sklearn-compatible regressor.
    """

    def __init__(self, model):
        self.model = model
        self.wrapper = None
        self.difficulty_estimator = DifficultyEstimator()

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
        2. Calibrate conformal predictor
        """

        # Train
        self.model.fit(
            X_train,
            y_train,
        )

        self.difficulty_estimator.fit(X_train, y=y_train)

        self.wrapper = WrapRegressor(
            self.model
        )

        # Calibrate
        self.wrapper.calibrate(
            X_calib,
            y_calib,
            de=self.difficulty_estimator,
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
            raise RuntimeError("ConformalPredictor not fitted.")

        intervals = self.wrapper.predict_int(
            X,
            confidence=confidence,
        )

        lower = intervals[:, 0]
        upper = intervals[:, 1]

        return lower, upper