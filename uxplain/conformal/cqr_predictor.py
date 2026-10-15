"""
Conformalized Quantile Regression predictor.

Implements CQR (Romano, Patterson & Candès, 2019).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._validation import check_confidence, check_fit_data, prediction_frame, remember_features


class CQRConformalPredictor(RegressorMixin, BaseEstimator):
    """
    Conformalized Quantile Regression (CQR) predictor.

    Fits two quantile regressors — one for the lower bound and one
    for the upper bound — then calibrates them using nonconformity
    scores on a held-out calibration set.

    Parameters
    ----------
    lower_estimator : object
        Quantile regressor trained at quantile level ``alpha / 2``.
        Must expose a sklearn-compatible ``fit(X, y)`` / ``predict(X)``
        interface. Example:

        ```python
        GradientBoostingRegressor(loss="quantile", alpha=0.05)
        ```

    upper_estimator : object
        Quantile regressor trained at quantile level ``1 - alpha / 2``.
        Example:

        ```python
        GradientBoostingRegressor(loss="quantile", alpha=0.95)
        ```

    rearrange : bool, default=False
        Sort the two quantile predictions before calibrating and predicting
        (the rearrangement of Chernozhukov, Fernández-Val and Galichon, 2010),
        so that estimated quantiles never cross. The rearranged quantiles are
        still functions fitted on the training data only, so the conformal
        guarantee is unchanged. A negative calibration correction can still
        produce empty sets. Only the midpoint then stays affine in the two
        quantile models.

    estimator : object or None, default=None
        Optional point regressor, cloned and fitted independently on training
        observations. Required only for predict() and score(); the two quantile
        regressors alone suffice for predict_interval() and explanations.

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

    def __init__(self, lower_estimator, upper_estimator, *, rearrange: bool = False, estimator=None):
        self.lower_estimator = lower_estimator
        self.upper_estimator = upper_estimator
        self.rearrange = rearrange
        self.estimator = estimator


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ):
        """
        Fit quantile models and calibrate nonconformity scores.

        Parameters
        ----------
        X : array-like of shape (n_train, n_features)
            Features used to fit both quantile regressors.
        y : array-like of shape (n_train,)
            Training target values.
        X_calib : array-like of shape (n_calib, n_features)
            Held-out calibration features with the same columns as `X`.
        y_calib : array-like of shape (n_calib,)
            Finite calibration target values used to compute nonconformity
            scores from the two fitted quantile regressors.

        Returns
        -------
        self : CQRConformalPredictor
            The fitted and calibrated predictor.
        """

        self._is_fitted = False
        self._scores = None
        self._n_calib = None
        X, y, X_calib, y_calib = check_fit_data(
            X, y, X_calib, y_calib,
        )
        y_calib = np.asarray(y_calib, dtype=float)
        if not np.all(np.isfinite(y_calib)):
            raise ValueError("y_calib must contain only finite values.")

        remember_features(self, X)
        self.lower_estimator_ = clone(self.lower_estimator, safe=False)
        self.upper_estimator_ = clone(self.upper_estimator, safe=False)
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator, safe=False).fit(X, y)
        self.lower_estimator_.fit(X, y)
        self.upper_estimator_.fit(X, y)

        q_low, q_high = self._quantiles(X_calib)

        # CQR nonconformity score: max(q_low - y, y - q_high)
        self._scores = np.maximum(q_low - y_calib, y_calib - q_high)
        self._n_calib = len(y_calib)
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate calibrated prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for which to generate intervals, in training column order.
        confidence : float, default=0.9
            Desired marginal coverage level, strictly between 0 and 1.

        Returns
        -------
        lower : numpy.ndarray of shape (n_samples,)
            Lower interval endpoints after the calibration correction.
        upper : numpy.ndarray of shape (n_samples,)
            Upper interval endpoints after the calibration correction.

        Notes
        -----
        The correction uses the `ceil((n_calib + 1) * confidence)`-th smallest
        calibration score. If this rank exceeds `n_calib`, the intervals are
        unbounded with endpoints `(-inf, inf)`. A negative correction or
        crossing quantiles can produce an empty set with `lower > upper`;
        these endpoints are preserved and `upper - lower` is then a signed
        span, not the nonnegative length of the empty set.
        """

        check_is_fitted(self)

        check_confidence(confidence)
        X = prediction_frame(self, X)

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

        q_low, q_high = self._quantiles(X)
        return q_low - adjustment, q_high + adjustment

    def _quantiles(self, X):
        q_low = self._predict_quantile(self.lower_estimator_, X)
        q_high = self._predict_quantile(self.upper_estimator_, X)
        if self.rearrange:
            q_low, q_high = np.minimum(q_low, q_high), np.maximum(q_low, q_high)
        return q_low, q_high

    @staticmethod
    def _predict_quantile(estimator, X):
        values = np.asarray(estimator.predict(X), dtype=float)
        if values.shape != (len(X),):
            raise ValueError("Quantile models must predict one value per sample.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Quantile estimator predictions must contain only finite values.")
        return values

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def predict(self, X):
        """Return point predictions from the fitted base estimator."""
        check_is_fitted(self)
        if self.estimator is None:
            raise AttributeError("CQR point prediction requires estimator=; use predict_interval for intervals.")
        return self.estimator_.predict(prediction_frame(self, X))
