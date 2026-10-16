"""Custom conformal backends plugged in through the protocols."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from uxplain import ConformalExplainer


class ScaledResidualConformal(BaseEstimator):
    """Split conformal intervals ``m(x) +/- q * s(x)`` around any regressor.

    The point estimator ``m`` is trained on the first half of the training data,
    the scale estimator ``s`` on the absolute residuals of the second half, and
    ``q`` is the conformal quantile of the normalized calibration residuals.
    """

    def __init__(self, estimator, scale_estimator):
        self.estimator = estimator
        self.scale_estimator = scale_estimator

    def fit(self, X_train, y_train, X_calib, y_calib):
        half = len(X_train) // 2
        self.model_ = clone(self.estimator).fit(X_train[:half], y_train[:half])
        residuals = np.abs(y_train[half:] - self.model_.predict(X_train[half:]))
        self.scale_estimator_ = clone(self.scale_estimator).fit(X_train[half:], residuals)
        self.scores_ = np.sort(np.abs(y_calib - self.model_.predict(X_calib))
                               / self._scale(X_calib))
        return self

    def _scale(self, X):
        return np.maximum(self.scale_estimator_.predict(X), 1e-6)

    def predict_interval(self, X, confidence=0.9):
        n = len(self.scores_)
        rank = int(np.ceil((n + 1) * confidence))
        q = np.inf if rank > n else self.scores_[rank - 1]
        center, scale = self.model_.predict(X), self._scale(X)
        return center - q * scale, center + q * scale


@pytest.fixture(scope="module")
def heteroscedastic():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(1200, 3))
    y = X[:, 0] + np.exp(X[:, 1]) * rng.standard_normal(1200)
    return X[:900], y[:900], X[900:], y[900:]


def test_custom_backend_runs_through_the_pipeline(heteroscedastic):
    X_train, y_train, X_test, y_test = heteroscedastic
    pipeline = ConformalExplainer(
        conformal_predictor=ScaledResidualConformal(
            LinearRegression(), GradientBoostingRegressor(n_estimators=30, random_state=0),
        ),
        random_state=0,
    ).fit(X_train, y_train)

    assert pipeline.task_ == "regression"
    assert pipeline.conformal_method_ == "custom"
    coverage = pipeline.evaluate(X_test, y_test).loc["overall", "coverage"]
    assert 0.8 < coverage <= 1.0

    result = pipeline.explain(X_test[:20], show_plots=False)
    assert pipeline.algorithm_used_ == "exact"  # no shortcut for custom backends
    importance = result.feature_importance()
    # The spread depends on the second feature only.
    assert importance.index[0] == "Feature 1"
    assert pipeline.get_params()["conformal_predictor__scale_estimator__n_estimators"] == 30
