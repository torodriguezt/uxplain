"""Tests for coverage summaries, importance tables and their pipeline accessors."""

import matplotlib
import numpy as np
import pytest
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    UncertaintyExplanationPipeline,
    coverage_table,
    importance_bar,
    importance_table,
)

matplotlib.use("Agg")


class TestCoverageTable:
    def test_regression_counts_and_signed_size(self):
        lower = np.array([0., 0., 2., 1.])
        upper = np.array([1., 2., 1., 3.])
        y = np.array([0.5, 3., 1.5, 1.])
        table = coverage_table((lower, upper), y)
        overall = table.loc["overall"]
        assert overall["n"] == 4
        assert overall["coverage"] == pytest.approx(0.5)
        assert overall["mean_size"] == pytest.approx((1 + 2 - 1 + 2) / 4)
        # The crossed interval [2, 1] is an empty set.
        assert overall["share_empty"] == pytest.approx(0.25)

    def test_groups_add_rows(self):
        lower, upper = np.zeros(4), np.ones(4)
        y = np.array([0.5, 2., 0.5, 2.])
        table = coverage_table((lower, upper), y, groups=["a", "a", "b", "b"])
        assert table.index.tolist() == ["overall", "a", "b"]
        assert table.loc["a", "coverage"] == pytest.approx(0.5)

    def test_unknown_label_is_uncovered(self):
        sets = np.array([[True, False], [True, True], [False, False]])
        table = coverage_table(sets, ["x", "z", "y"], classes=["x", "y"])
        assert table.loc["overall", "coverage"] == pytest.approx(1 / 3)
        assert table.loc["overall", "share_empty"] == pytest.approx(1 / 3)
        assert table.loc["overall", "mean_size"] == pytest.approx(1.0)

    def test_classification_needs_one_column_per_class(self):
        with pytest.raises(ValueError, match="one column per class"):
            coverage_table(np.ones((2, 3), dtype=bool), [0, 1], classes=[0, 1])


class TestImportanceTable:
    def _explanation(self):
        values = np.array([[1., -2., 0.], [3., 0., 0.1], [-1., 2., 0.]])
        return shap.Explanation(values=values, feature_names=["a", "b", "c"])

    def test_mean_absolute_attribution_and_rank(self):
        table = importance_table(self._explanation())
        assert table.index.tolist() == ["a", "b", "c"]
        np.testing.assert_allclose(table["importance"], [5 / 3, 4 / 3, 0.1 / 3])
        assert table["rank"].tolist() == [1, 2, 3]

    def test_bootstrap_intervals_are_reproducible(self):
        first = importance_table(self._explanation(), n_boot=200, random_state=7)
        second = importance_table(self._explanation(), n_boot=200, random_state=7)
        assert (first == second).all().all()
        assert (first["ci_lower"] <= first["importance"]).all()
        assert (first["importance"] <= first["ci_upper"]).all()
        assert (first["rank_lower"] <= first["rank"]).all()
        assert (first["rank"] <= first["rank_upper"]).all()

    @pytest.mark.parametrize("n_boot", [-1, 1.5, True])
    def test_invalid_bootstrap_size(self, n_boot):
        with pytest.raises(ValueError, match="n_boot"):
            importance_table(self._explanation(), n_boot=n_boot)

    def test_feature_names_must_match(self):
        with pytest.raises(ValueError, match="feature_names"):
            importance_table(self._explanation(), feature_names=["a", "b"])


@pytest.fixture
def cqr_pipeline(data, quantile_lower, quantile_upper):
    pipeline = UncertaintyExplanationPipeline(
        conformal_predictor=CQRConformalPredictor(quantile_lower, quantile_upper),
    )
    return pipeline.fit(data["X_train"], data["y_train"], X_calib=data["X_calib"], y_calib=data["y_calib"])


class TestPipelineAccessors:
    def test_evaluate_regression(self, data, cqr_pipeline):
        table = cqr_pipeline.evaluate(data["X_test"], data["y_test"])
        lower, upper = cqr_pipeline.predict_interval(data["X_test"])
        expected = np.mean((lower <= data["y_test"]) & (data["y_test"] <= upper))
        assert table.loc["overall", "coverage"] == pytest.approx(expected)
        assert table.loc["overall", "n"] == len(data["y_test"])

    def test_evaluate_classification_by_class(self, classification_data):
        d = classification_data
        pipeline = UncertaintyExplanationPipeline(
            estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            task="classification",
        )
        pipeline.fit(d["X_train"], d["y_train"], X_calib=d["X_calib"], y_calib=d["y_calib"])
        table = pipeline.evaluate(d["X_test"], d["y_test"], groups=d["y_test"])
        assert table.index.tolist() == ["overall", 0, 1, 2]
        assert table.loc[[0, 1, 2], "n"].sum() == len(d["y_test"])
        np.testing.assert_array_equal(pipeline.classes_, [0, 1, 2])

    def test_classes_only_for_classification(self, cqr_pipeline):
        with pytest.raises(AttributeError, match="classification"):
            cqr_pipeline.classes_

    def test_fast_path_flags_are_public(self, data, cqr_pipeline):
        assert cqr_pipeline.used_fast_path_ is None
        cqr_pipeline.explain(data["X_test"][:5], show_plots=False)
        assert cqr_pipeline.used_fast_path_ is True
        assert cqr_pipeline.algorithm_used_ == "tree"

    def test_result_importance(self, data, cqr_pipeline):
        result = cqr_pipeline.explain(data["X_test"][:10], show_plots=False)
        table = result.feature_importance(n_boot=50, random_state=0)
        values = np.abs(result.explanation_values.values).mean(axis=0)
        np.testing.assert_allclose(np.sort(table["importance"])[::-1],
                                   np.sort(values)[::-1])
        assert {"ci_lower", "ci_upper", "rank_lower", "rank_upper"} <= set(table)

    def test_pdp_importance_bootstrap_needs_ice(self, data):
        pipeline = UncertaintyExplanationPipeline(estimator=Ridge(), xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        averaged = pipeline.explain(data["X_test"], show_plots=False, kind="average")
        with pytest.raises(ValueError, match="ICE"):
            averaged.feature_importance(n_boot=10)
        both = pipeline.explain(data["X_test"], show_plots=False, kind="both")
        table = both.feature_importance(n_boot=20, random_state=0)
        np.testing.assert_allclose(
            np.sort(table["importance"]),
            np.sort(both.explanation_values.importance),
        )


def test_importance_bar_draws_intervals():
    import matplotlib.pyplot as plt

    values = np.random.default_rng(0).standard_normal((30, 3))
    table = importance_table(shap.Explanation(values=values, feature_names=list("abc")),
                             n_boot=50, random_state=0)
    ax = importance_bar(table)
    assert len(ax.patches) == 3
    assert ax.containers  # bars plus error bars
    plt.close(ax.figure)


def test_gradient_boosting_cqr_reports_exact_algorithm_for_generic_path(data):
    cp = CQRConformalPredictor(
        GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=10),
        GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=10),
    )
    pipeline = UncertaintyExplanationPipeline(conformal_predictor=cp, fast_path=False)
    pipeline.fit(data["X_train"], data["y_train"], X_calib=data["X_calib"], y_calib=data["y_calib"])
    pipeline.explain(data["X_test"][:3], show_plots=False)
    assert pipeline.used_fast_path_ is False
    assert pipeline.algorithm_used_ == "exact"
