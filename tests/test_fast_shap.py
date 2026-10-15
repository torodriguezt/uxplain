"""Tests for the exact TreeSHAP shortcut used on affine conformal summaries."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    CrepesConformalPredictor,
    ShapUncertaintyExplainer,
)
from uxplain.explainability.fast_shap import tree_shap_values

REGRESSION_METRICS = ["width", "lower", "upper", "midpoint"]


def _fit_cqr(data):
    cp = CQRConformalPredictor(
        lower_estimator=GradientBoostingRegressor(
            loss="quantile", alpha=0.05, n_estimators=30, random_state=0
        ),
        upper_estimator=GradientBoostingRegressor(
            loss="quantile", alpha=0.95, n_estimators=30, random_state=0
        ),
    )
    cp.fit(data["X_train"], data["y_train"], X_calib=data["X_calib"], y_calib=data["y_calib"])
    return cp


def _fit_crepes(data, method):
    cp = CrepesConformalPredictor(
        RandomForestRegressor(n_estimators=20, random_state=0), method=method
    )
    cp.fit(data["X_train"], data["y_train"], X_calib=data["X_calib"], y_calib=data["y_calib"])
    return cp


def _explain(cp, data, metric, fast_path, n_rows=20):
    explainer = ShapUncertaintyExplainer(cp, uncertainty_metric=metric, fast_path=fast_path)
    explainer.fit(data["X_calib"])
    explanation = explainer.explain(data["X_test"][:n_rows])
    return explainer, explanation


class TestCQRFastPath:
    def test_negative_signed_width_agrees_with_generic_and_is_additive(self):
        # Wide training responses at x=(1, 1), narrow ones at x=(0, 0).
        # Calibrating on central responses at (1, 1) shrinks the endpoints
        # enough to produce empty sets elsewhere, without clipping their span.
        X_train = np.repeat([[0., 0.], [1., 1.]], 40, axis=0)
        y_train = np.r_[np.tile([-1., 1.], 20), np.tile([-10., 10.], 20)]
        cp = CQRConformalPredictor(*[
            GradientBoostingRegressor(
                loss="quantile", alpha=alpha, n_estimators=30,
                max_depth=1, random_state=0,
            )
            for alpha in (0.1, 0.9)
        ])
        cp.fit(X_train, y_train, X_calib=np.ones((20, 2)), y_calib=np.zeros(20))
        X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        lower, upper = cp.predict_interval(X)
        signed_width = upper - lower
        assert np.any(signed_width < 0)

        explanations = []
        for fast_path in (True, False):
            explainer = ShapUncertaintyExplainer(
                cp, uncertainty_metric="width", fast_path=fast_path, random_state=0,
            )
            explainer.fit(X)
            explanation = explainer.explain(X)
            assert explainer.used_fast_path_ is fast_path
            np.testing.assert_allclose(
                explanation.base_values + explanation.values.sum(axis=1),
                signed_width, atol=1e-5,
            )
            explanations.append(explanation)
        np.testing.assert_allclose(
            explanations[0].values, explanations[1].values, atol=1e-5,
        )

    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_fast_path_is_taken(self, data, metric):
        cp = _fit_cqr(data)
        explainer, _ = _explain(cp, data, metric, fast_path=True)
        assert explainer.used_fast_path_ is True

    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_agrees_with_generic_path(self, data, metric):
        cp = _fit_cqr(data)
        _, fast = _explain(cp, data, metric, fast_path=True)
        _, generic = _explain(cp, data, metric, fast_path=False)
        # TreeSHAP accumulates in float32, hence the loose-ish tolerance.
        assert np.abs(np.asarray(fast.values)
                      - np.asarray(generic.values)).max() < 1e-4

    @pytest.mark.parametrize("metric", REGRESSION_METRICS)
    def test_additivity(self, data, metric):
        """base_value + sum(phi) must reproduce the metric itself."""
        cp = _fit_cqr(data)
        explainer, explanation = _explain(cp, data, metric, fast_path=True)
        target = np.asarray(
            explainer._target_function(data["X_test"][:20])
        ).ravel()
        reconstructed = (np.asarray(explanation.base_values).ravel()
                         + np.asarray(explanation.values).sum(axis=1))
        assert np.abs(reconstructed - target).max() < 1e-4

    def test_disabled_by_flag(self, data):
        cp = _fit_cqr(data)
        explainer, _ = _explain(cp, data, "width", fast_path=False)
        assert explainer.used_fast_path_ is False


class TestRearrangedCQRFastPath:
    """Sorting the quantile models keeps only their average affine."""

    def _fit(self, data):
        cp = CQRConformalPredictor(
            lower_estimator=GradientBoostingRegressor(
                loss="quantile", alpha=0.05, n_estimators=30, random_state=0
            ),
            upper_estimator=GradientBoostingRegressor(
                loss="quantile", alpha=0.95, n_estimators=30, random_state=0
            ),
            rearrange=True,
        )
        cp.fit(data["X_train"], data["y_train"], X_calib=data["X_calib"], y_calib=data["y_calib"])
        return cp

    @pytest.mark.parametrize("metric", ["width", "lower", "upper"])
    def test_sorted_endpoints_use_generic_path(self, data, metric):
        explainer, _ = _explain(self._fit(data), data, metric, fast_path=True)
        assert explainer.used_fast_path_ is False

    def test_midpoint_keeps_fast_path(self, data):
        cp = self._fit(data)
        explainer, fast = _explain(cp, data, "midpoint", fast_path=True)
        _, generic = _explain(cp, data, "midpoint", fast_path=False)
        assert explainer.used_fast_path_ is True
        assert np.abs(np.asarray(fast.values) - np.asarray(generic.values)).max() < 1e-4


@pytest.fixture(scope="module")
def large():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((700, 3))
    y = X[:, 0] + np.abs(X[:, 1]) * rng.standard_normal(700)
    cp = CQRConformalPredictor(
        GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                  n_estimators=30, random_state=0),
        GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                  n_estimators=30, random_state=0),
    )
    cp.fit(X[:300], y[:300], X_calib=X[300:600], y_calib=y[300:600])
    return cp, X[300:600], X[600:620]


class TestSharedBackground:
    """Both computational paths must evaluate the same Shapley game."""

    def test_paths_agree_beyond_one_hundred_background_rows(self, large):
        cp, background, rows = large
        explanations = []
        for fast_path in (True, False):
            explainer = ShapUncertaintyExplainer(
                cp, uncertainty_metric="width", fast_path=fast_path, background_size=250,
            )
            explainer.fit(background)
            explanations.append(explainer.explain(rows))
            assert len(explainer._background) == 250
        assert np.abs(explanations[0].values - explanations[1].values).max() < 1e-4
        np.testing.assert_allclose(explanations[0].base_values,
                                   explanations[1].base_values, atol=1e-4)

    def test_background_is_subsampled_reproducibly(self, large):
        cp, background, _ = large
        first = ShapUncertaintyExplainer(cp, random_state=4)
        first.fit(background)
        second = ShapUncertaintyExplainer(cp, random_state=4)
        second.fit(background)
        assert len(first._background) == 100
        np.testing.assert_array_equal(first._background, second._background)

    def test_algorithm_used_is_reported(self, large):
        cp, background, rows = large
        explainer = ShapUncertaintyExplainer(cp, uncertainty_metric="width")
        explainer.fit(background)
        explainer.explain(rows)
        assert explainer.algorithm_used_ == "tree"
        explainer.fast_path = False
        explainer.explain(rows)
        assert explainer.algorithm_used_ == "exact"

    @pytest.mark.parametrize("size", [0, -3, 2.5, True])
    def test_invalid_background_size_is_rejected(self, large, size):
        with pytest.raises(ValueError, match="background_size"):
            ShapUncertaintyExplainer(large[0], background_size=size).fit(large[1])


class TestCrepesFastPath:
    """The shortcut must accept only the metrics that stay affine."""

    def test_standard_accepts_all_metrics(self, data):
        cp = _fit_crepes(data, "standard")
        for metric in REGRESSION_METRICS:
            explainer, _ = _explain(cp, data, metric, fast_path=True)
            assert explainer.used_fast_path_ is True, metric

    def test_standard_width_is_constant(self, data):
        """A standard interval has a fixed width, so nothing explains it."""
        cp = _fit_crepes(data, "standard")
        _, explanation = _explain(cp, data, "width", fast_path=True)
        assert np.allclose(explanation.values, 0.0)

    @pytest.mark.parametrize("metric", ["width", "lower", "upper"])
    def test_normalized_rejects_scaled_metrics(self, data, metric):
        """Normalized intervals scale with a difficulty estimator, so the
        endpoints and the width are no longer affine in the base estimator."""
        cp = _fit_crepes(data, "normalized")
        explainer, _ = _explain(cp, data, metric, fast_path=True)
        assert explainer.used_fast_path_ is False

    def test_normalized_accepts_midpoint(self, data):
        """The midpoint of a symmetric interval is the base prediction."""
        cp = _fit_crepes(data, "normalized")
        explainer, _ = _explain(cp, data, "midpoint", fast_path=True)
        assert explainer.used_fast_path_ is True


class TestFastPathSingleRow:
    """One row cannot show whether a summary varies, so the residual check must
    not rely on the explained rows alone."""

    @pytest.mark.parametrize("metric", ["width", "lower", "upper"])
    def test_normalized_rejects_scaled_metrics(self, data, metric):
        cp = _fit_crepes(data, "normalized")
        explainer, _ = _explain(cp, data, metric, fast_path=True, n_rows=1)
        assert explainer.used_fast_path_ is False

    def test_normalized_width_matches_generic_path(self, data):
        """The shortcut used to report all-zero attributions here."""
        cp = _fit_crepes(data, "normalized")
        _, fast = _explain(cp, data, "width", fast_path=True, n_rows=1)
        _, generic = _explain(cp, data, "width", fast_path=False, n_rows=1)
        np.testing.assert_allclose(fast.values, generic.values)
        assert np.abs(np.asarray(fast.values)).sum() > 0

    def test_cqr_still_takes_fast_path(self, data):
        """CQR summaries are affine everywhere, so one row is still exact."""
        cp = _fit_cqr(data)
        explainer, fast = _explain(cp, data, "width", fast_path=True, n_rows=1)
        _, generic = _explain(cp, data, "width", fast_path=False, n_rows=1)
        assert explainer.used_fast_path_ is True
        assert np.abs(np.asarray(fast.values)
                      - np.asarray(generic.values)).max() < 1e-4


class TestFastPathGuards:
    def test_custom_payoff_on_masked_rows_uses_generic_path(self):
        """Observed-row agreement does not prove affinity of the SHAP game."""
        class HybridPredictor:
            estimator = object()

            def predict_interval(self, X, confidence=0.9):
                X = np.asarray(X)
                width = 5 + (X[:, 0] - X[:, 1]) * (1 + 2 * X[:, 0])
                return np.zeros(len(X)), width

        explainer = ShapUncertaintyExplainer(HybridPredictor())
        explainer.fit(np.array([[0., 0.]]))
        explanation = explainer.explain(np.array([[1., 1.]]))
        # Width equals five on the explained and background rows but changes
        # on the two masked hybrids. The old constant shortcut returned zeros.
        assert explainer.used_fast_path_ is False
        np.testing.assert_allclose(explanation.values, [[2., -2.]])

    @pytest.mark.parametrize("metric", ["width", "lower", "upper"])
    def test_normalized_identical_rows_still_rejects_shortcut(self, data, metric):
        cp = _fit_crepes(data, "normalized")
        explainer = ShapUncertaintyExplainer(cp, uncertainty_metric=metric)
        row = data["X_test"][:1]
        explainer.fit(row)
        explainer.explain(row)
        assert explainer.used_fast_path_ is False

    def test_non_tree_model_falls_back(self, data):
        """Ridge is not a tree ensemble, so TreeSHAP cannot be used."""
        cp = CQRConformalPredictor(lower_estimator=Ridge(), upper_estimator=Ridge())
        cp.fit(data["X_train"], data["y_train"],
               X_calib=data["X_calib"], y_calib=data["y_calib"])
        explainer, _ = _explain(cp, data, "width", fast_path=True)
        assert explainer.used_fast_path_ is False

    def test_classification_metric_returns_none(self, data):
        """Classification summaries have no affine decomposition."""
        cp = _fit_cqr(data)
        result = tree_shap_values(
            cp, "set_size", data["X_test"][:5], data["X_calib"], lambda X: X[:, 0]
        )
        assert result is None

    def test_non_finite_summary_returns_none(self, data):
        """Unbounded intervals (too few calibration rows) cannot be decomposed."""
        cp = _fit_crepes(data, "standard")
        result = tree_shap_values(
            cp, "width", data["X_test"][:5], data["X_calib"],
            lambda X: np.full(len(X), np.inf),
        )
        assert result is None
