import numpy as np
import pytest
from sklearn.linear_model import Ridge

from uncertainty_explainer import (
    CrepesConformalPredictor,
    UncertaintyExplanationPipeline,
)
from uncertainty_explainer.uncertainty.metrics import (
    METRIC_LABELS,
    make_uncertainty_function,
    metric_label,
)


@pytest.fixture
def fitted_cp(data):
    cp = CrepesConformalPredictor(Ridge(), method="normalized")
    cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
    return cp


# ---------------------------------------------------------------------------
# Factory + labels
# ---------------------------------------------------------------------------

class TestMakeUncertaintyFunction:
    def test_width_matches_upper_minus_lower(self, fitted_cp, data):
        fn = make_uncertainty_function(fitted_cp, confidence=0.9, metric="width")
        lower, upper = fitted_cp.predict(data["X_test"], confidence=0.9)
        np.testing.assert_allclose(fn(data["X_test"]), upper - lower)

    def test_lower_matches_lower_bound(self, fitted_cp, data):
        fn = make_uncertainty_function(fitted_cp, confidence=0.9, metric="lower")
        lower, _ = fitted_cp.predict(data["X_test"], confidence=0.9)
        np.testing.assert_allclose(fn(data["X_test"]), lower)

    def test_upper_matches_upper_bound(self, fitted_cp, data):
        fn = make_uncertainty_function(fitted_cp, confidence=0.9, metric="upper")
        _, upper = fitted_cp.predict(data["X_test"], confidence=0.9)
        np.testing.assert_allclose(fn(data["X_test"]), upper)

    def test_midpoint_matches_average_of_bounds(self, fitted_cp, data):
        fn = make_uncertainty_function(fitted_cp, confidence=0.9, metric="midpoint")
        lower, upper = fitted_cp.predict(data["X_test"], confidence=0.9)
        np.testing.assert_allclose(fn(data["X_test"]), (lower + upper) / 2.0)

    def test_unknown_metric_raises(self, fitted_cp):
        with pytest.raises(ValueError, match="Unknown uncertainty metric"):
            make_uncertainty_function(fitted_cp, metric="bogus")


class TestMetricLabel:
    @pytest.mark.parametrize("metric", list(METRIC_LABELS))
    def test_returns_label_for_known_metric(self, metric):
        assert metric_label(metric) == METRIC_LABELS[metric]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown uncertainty metric"):
            metric_label("bogus")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineMetric:
    @pytest.mark.parametrize("metric", ["width", "lower", "upper", "midpoint"])
    def test_pipeline_propagates_metric_to_shap(self, data, metric):
        pipeline = UncertaintyExplanationPipeline(
            model=Ridge(), xai_method="shap", uncertainty_metric=metric,
        )
        pipeline.fit(data["X_train"], data["y_train"])
        assert pipeline.explainer.metric == metric
        # Smoke-test that explain() runs end-to-end with the chosen metric
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert result.explanation_values.values.shape[0] == 5

    @pytest.mark.parametrize("metric", ["width", "lower", "upper", "midpoint"])
    def test_pipeline_propagates_metric_to_pdp(self, data, metric):
        pipeline = UncertaintyExplanationPipeline(
            model=Ridge(), xai_method="pdp", uncertainty_metric=metric,
        )
        pipeline.fit(data["X_train"], data["y_train"])
        assert pipeline.explainer.metric == metric
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert result.explanation_values.metric == metric

    @pytest.mark.parametrize("metric", ["width", "lower", "upper", "midpoint"])
    def test_pipeline_propagates_metric_to_lime(self, data, metric):
        pipeline = UncertaintyExplanationPipeline(
            model=Ridge(),
            xai_method="lime",
            uncertainty_metric=metric,
            n_lime_samples=50,
        )
        pipeline.fit(data["X_train"], data["y_train"])
        assert pipeline.explainer.metric == metric
        result = pipeline.explain(data["X_test"][:3], show_plots=False)
        assert result.explanation_values.metric == metric

    def test_default_metric_is_width(self, data):
        pipeline = UncertaintyExplanationPipeline(model=Ridge())
        assert pipeline.uncertainty_metric == "width"
        assert pipeline.explainer.metric == "width"
