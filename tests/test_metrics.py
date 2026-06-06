import numpy as np
import pytest
from sklearn.linear_model import Ridge

from uxplain import (
    ClassificationExplanationResult,
    CrepesConformalClassifier,
    CrepesConformalPredictor,
    UncertaintyExplanationPipeline,
)
from uxplain.uncertainty.metrics import (
    METRIC_LABELS,
    make_uncertainty_function,
    metric_label,
)


@pytest.fixture
def fitted_cp(data):
    cp = CrepesConformalPredictor(Ridge(), method="normalized")
    cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
    return cp


@pytest.fixture
def fitted_classifier_cp(classification_data, classifier):
    cp = CrepesConformalClassifier(classifier, method="standard")
    cp.fit(
        classification_data["X_train"], classification_data["y_train"],
        classification_data["X_calib"], classification_data["y_calib"],
    )
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
        with pytest.raises(ValueError, match="not valid for regression"):
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


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    # crepes' WrapClassifier uses *smoothed* p-values that consume the global
    # NumPy RNG, so two consecutive calls give different outputs. We seed
    # immediately before each call so both paths see the same RNG state.

    def test_set_size_matches_predict_set_sum(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="set_size",
        )
        np.random.seed(0)
        actual = fn(classification_data["X_test"])
        np.random.seed(0)
        expected = fitted_classifier_cp.predict_set(
            classification_data["X_test"], confidence=0.9,
        ).sum(axis=1).astype(float)
        np.testing.assert_allclose(actual, expected)

    def test_set_size_is_integer_and_bounded(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="set_size",
        )
        sizes = fn(classification_data["X_test"])
        assert sizes.shape == (len(classification_data["X_test"]),)
        assert np.all(sizes == sizes.astype(int))
        assert np.all(sizes >= 0)
        assert np.all(sizes <= classification_data["n_classes"])

    def test_credibility_matches_max_pvalue(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="credibility",
        )
        np.random.seed(0)
        actual = fn(classification_data["X_test"])
        np.random.seed(0)
        expected = fitted_classifier_cp.predict_p(
            classification_data["X_test"],
        ).max(axis=1)
        np.testing.assert_allclose(actual, expected)

    def test_credibility_in_unit_interval(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="credibility",
        )
        values = fn(classification_data["X_test"])
        assert np.all((values >= 0.0) & (values <= 1.0 + 1e-9))

    def test_confidence_matches_one_minus_second_pvalue(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="confidence",
        )
        np.random.seed(0)
        actual = fn(classification_data["X_test"])
        np.random.seed(0)
        p_sorted = np.sort(
            fitted_classifier_cp.predict_p(classification_data["X_test"]), axis=1,
        )
        expected = 1.0 - p_sorted[:, -2]
        np.testing.assert_allclose(actual, expected)

    def test_confidence_in_unit_interval(
        self, fitted_classifier_cp, classification_data,
    ):
        fn = make_uncertainty_function(
            fitted_classifier_cp, confidence=0.9, metric="confidence",
        )
        values = fn(classification_data["X_test"])
        assert np.all((values >= 0.0) & (values <= 1.0 + 1e-9))

    def test_margin_metric_no_longer_supported(self, fitted_classifier_cp):
        with pytest.raises(ValueError, match="not valid for classification"):
            make_uncertainty_function(fitted_classifier_cp, metric="margin")

    def test_margin_not_in_labels(self):
        assert "margin" not in METRIC_LABELS

    def test_regression_metric_rejected_for_classifier(self, fitted_classifier_cp):
        with pytest.raises(ValueError, match="not valid for classification"):
            make_uncertainty_function(fitted_classifier_cp, metric="width")


class TestClassificationPipelineMetric:
    @pytest.mark.parametrize(
        "metric", ["set_size", "credibility", "confidence"],
    )
    def test_pipeline_propagates_metric(self, classification_data, classifier, metric):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier,
            task="classification",
            uncertainty_metric=metric,
            xai_method="shap",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        assert pipeline.uncertainty_metric == metric
        assert pipeline.explainer.metric == metric
        result = pipeline.explain(
            classification_data["X_test"][:5], show_plots=False,
        )
        assert isinstance(result, ClassificationExplanationResult)

    def test_default_metric_is_set_size_for_classification(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification",
        )
        assert pipeline.uncertainty_metric == "set_size"
        assert pipeline.explainer.metric == "set_size"

    def test_invalid_classification_metric_raises(self, classifier):
        with pytest.raises(ValueError, match="not valid for classification"):
            UncertaintyExplanationPipeline(
                model=classifier,
                task="classification",
                uncertainty_metric="width",
            )

    def test_margin_rejected_at_pipeline_level(self, classifier):
        with pytest.raises(ValueError, match="not valid for classification"):
            UncertaintyExplanationPipeline(
                model=classifier,
                task="classification",
                uncertainty_metric="margin",
            )
