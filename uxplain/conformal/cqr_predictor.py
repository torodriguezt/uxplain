"""
Conformalized Quantile Regression predictor.

Implements CQR (Romano, Sesia & Candès, 2019).
"""

from __future__ import annotations

import numpy as np

from ._validation import check_confidence, check_fit_data


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

    CQR may return an empty prediction set (``lower > upper``), for example
    after a negative calibration correction or quantile crossing. Endpoints
    are preserved, so ``upper - lower`` is a signed span in that case, not
    the nonnegative length of the empty set.
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

        self._scores = None
        self._n_calib = None
        X_train, y_train, X_calib, y_calib = check_fit_data(
            X_train, y_train, X_calib, y_calib,
        )
        y_calib = np.asarray(y_calib, dtype=float)
        if not np.all(np.isfinite(y_calib)):
            raise ValueError("y_calib must contain only finite values.")

        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        q_low = self._predict_quantile(self.lower_model, X_calib)
        q_high = self._predict_quantile(self.upper_model, X_calib)

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

        check_confidence(confidence)
        X = np.asarray(X)

        # Conformal quantile (Romano et al., 2019): the ceil((n + 1) * confidence)-th
        # smallest score. When that rank exceeds n, no finite adjustment
        # guarantees coverage, so the interval is unbounded. Do not subtract
        # a tolerance: that can lower the rank below the requested coverage.
        n = self._n_calib
        k = int(np.ceil((n + 1) * confidence))
        if k > n:
            adjustment = np.inf
        else:
            adjustment = float(np.sort(self._scores)[k - 1])

        lower = self._predict_quantile(self.lower_model, X) - adjustment
        upper = self._predict_quantile(self.upper_model, X) + adjustment

        return lower, upper

    @staticmethod
    def _predict_quantile(model, X):
        values = np.asarray(model.predict(X), dtype=float)
        if values.shape != (len(X),):
            raise ValueError("Quantile models must predict one value per sample.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Quantile model predictions must contain only finite values.")
        return values
