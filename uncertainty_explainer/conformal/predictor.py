"""
Conformal prediction module.

Wrapper around crepes.WrapRegressor
for interval prediction.
"""

from __future__ import annotations

import numpy as np

from crepes import WrapRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor


class ConformalPredictor:
    """
    Wrapper for conformal regression using crepes.

    Parameters
    ----------
    model : object
        Any sklearn-compatible regressor.
    """

    def __init__(self, model, difficulty_estimator=None):
        self.model = model
        self.wrapper = None

        if difficulty_estimator is None:
            self.difficulty_estimator = RandomForestRegressor(
                n_estimators=50, random_state=42
            )
        else:
            self.difficulty_estimator = clone(difficulty_estimator)

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

        train_preds = self.model.predict(X_train)
        abs_residuals = np.abs(train_preds - y_train)
        self.difficulty_estimator.fit(X_train, abs_residuals)

        sigmas_calib = self.difficulty_estimator.predict(X_calib)
        sigmas_calib = np.maximum(sigmas_calib, 1e-6)  # avoid zero or negative sigmas

        self.wrapper = WrapRegressor(
            self.model
         )

        # Calibrate
        self.wrapper.calibrate(
            X_calib,
            y_calib,
            sigmas = sigmas_calib,
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

        sigmas_test = self.difficulty_estimator.predict(X)
        sigmas_test = np.maximum(sigmas_test, 1e-6)

        intervals = self.wrapper.predict_int(
            X,
            sigmas = sigmas_test,
            confidence=confidence,
        )

        lower = intervals[:, 0]
        upper = intervals[:, 1]

        return lower, upper