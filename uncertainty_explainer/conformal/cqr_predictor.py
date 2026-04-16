"""
Conformalized Quantile Regression predictor.

Implements CQR (Romano, Sesia & Candès, 2019).
"""

from __future__ import annotations

import numpy as np


class CQRConformalPredictor:
    """
    Conformalized Quantile Regression (CQR) predictor.

    Fits two quantile regressors — one for the lower bound and one
    for the upper bound — then calibrates them using nonconformity
    scores on a held-out calibration set.

    Parameters
    ----------
    lower_model
        Quantile regressor trained at quantile level ``alpha / 2``.
        Must expose a sklearn-compatible ``fit(X, y)`` / ``predict(X)``
        interface.  Example::

            GradientBoostingRegressor(loss="quantile", alpha=0.05)

    upper_model
        Quantile regressor trained at quantile level ``1 - alpha / 2``.
        Example::

            GradientBoostingRegressor(loss="quantile", alpha=0.95)

    Notes
    -----
    The quantile levels baked into the models should match the
    ``confidence`` value used at predict time (e.g. models at 0.05 / 0.95
    pair with ``confidence=0.90``).  Changing ``confidence`` at predict
    time still gives valid coverage via the calibration correction, but
    intervals may be wider or narrower than optimal if the mismatch is
    large.
    """

    def __init__(self, lower_model, upper_model):
        self.lower_model = lower_model
        self.upper_model = upper_model

        self._scores: np.ndarray | None = None
        self._n_calib: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> None:
        """
        Fit quantile models and calibrate nonconformity scores.

        Parameters
        ----------
        X_train, y_train
            Training data.
        X_calib, y_calib
            Calibration data used to compute nonconformity scores.
        """

        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        q_low = self.lower_model.predict(X_calib)
        q_high = self.upper_model.predict(X_calib)

        # CQR nonconformity score: max(q_low - y, y - q_high)
        self._scores = np.maximum(q_low - y_calib, y_calib - q_high)
        self._n_calib = len(y_calib)

    def predict(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate calibrated prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        confidence : float
            Desired marginal coverage level.

        Returns
        -------
        lower : np.ndarray
        upper : np.ndarray
        """

        if self._scores is None:
            raise RuntimeError("CQRConformalPredictor not fitted. Call fit() first.")

        alpha = 1 - confidence
        # Finite-sample correction ensures valid marginal coverage
        level = min((1 + 1 / self._n_calib) * (1 - alpha), 1.0)
        adjustment = float(np.quantile(self._scores, level))

        lower = self.lower_model.predict(X) - adjustment
        upper = self.upper_model.predict(X) + adjustment

        return lower, upper
