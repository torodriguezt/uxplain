import matplotlib
import numpy as np
import pytest

from uncertainty_explainer import (
    ClassificationExplanationResult,
    UncertaintyExplanationPipeline,
)
from uncertainty_explainer.plots import (
    generate_lime_plots,
    generate_pdp_plots,
    generate_shap_plots,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures — fitted pipelines shared across test classes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_shap_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(model=Ridge(), xai_method="shap")
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:5], show_plots=False)


@pytest.fixture(scope="module")
def regression_pdp_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(model=Ridge(), xai_method="pdp")
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:5], show_plots=False)


@pytest.fixture(scope="module")
def regression_lime_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(
        model=Ridge(), xai_method="lime", n_lime_samples=50,
    )
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:3], show_plots=False)


@pytest.fixture
def classification_shap_result(classification_data, classifier):
    pipeline = UncertaintyExplanationPipeline(
        model=classifier, task="classification", xai_method="shap",
    )
    pipeline.fit(
        classification_data["X_train"], classification_data["y_train"],
    )
    result = pipeline.explain(classification_data["X_test"][:5], show_plots=False)
    return pipeline, result


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

class TestGenerateDefaultPlots:
    def test_beeswarm_returns_figure(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["beeswarm"],
            show=False,
        )
        assert "beeswarm" in figs

    def test_bar_returns_figure(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["bar"],
            show=False,
        )
        assert "bar" in figs

    def test_waterfall_returns_figure(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["waterfall"],
            show=False,
        )
        assert "waterfall" in figs

    def test_summary_returns_figure(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["summary"],
            show=False,
        )
        assert "summary" in figs

    def test_all_kinds_together(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["beeswarm", "bar", "waterfall", "summary"],
            show=False,
        )
        assert set(figs.keys()) == {"beeswarm", "bar", "waterfall", "summary"}

    def test_waterfall_index_respected(self, regression_shap_result, data):
        _, result = regression_shap_result
        figs = generate_shap_plots(
            result.explanation_values,
            X=data["X_test"][:5],
            kinds=["waterfall"],
            waterfall_index=2,
            show=False,
        )
        assert "waterfall" in figs


# ---------------------------------------------------------------------------
# PDP plots
# ---------------------------------------------------------------------------

class TestGeneratePDPPlots:
    def test_pdp_returns_figure(self, regression_pdp_result, data):
        pipeline, result = regression_pdp_result
        pipeline.explainer.fit(data["X_calib"], kind="both")
        exp = pipeline.explainer.explain(data["X_test"][:5])
        figs = generate_pdp_plots(exp, kinds=["pdp"], show=False)
        assert "pdp" in figs


# ---------------------------------------------------------------------------
# LIME plots
# ---------------------------------------------------------------------------

class TestGenerateLimePlots:
    def test_local_returns_figure(self, regression_lime_result):
        _, result = regression_lime_result
        figs = generate_lime_plots(
            result.explanation_values, kinds=["local"], show=False,
        )
        assert "local" in figs

    def test_local_plot_with_explicit_sample_index(self, regression_lime_result):
        _, result = regression_lime_result
        figs = generate_lime_plots(
            result.explanation_values, kinds=["local"], sample_index=1, show=False,
        )
        assert "local" in figs


# ---------------------------------------------------------------------------
# Classification result fields
# ---------------------------------------------------------------------------

class TestClassificationResultFields:
    def test_classification_result_fields(self, classification_shap_result, classification_data):
        _, result = classification_shap_result
        n = 5
        k = classification_data["n_classes"]
        assert isinstance(result, ClassificationExplanationResult)
        assert result.prediction_set.shape == (n, k)
        assert result.p_values.shape == (n, k)
        assert result.set_size.shape == (n,)
        assert result.classes.shape == (k,)
        assert np.all(result.p_values >= 0.0)
        assert np.all(result.p_values <= 1.0 + 1e-9)
        np.testing.assert_array_equal(
            result.set_size, result.prediction_set.sum(axis=1),
        )


# ---------------------------------------------------------------------------
# pipeline.plot() called directly
# ---------------------------------------------------------------------------

class TestPipelinePlotDirect:
    def test_plot_shap_direct(self, regression_shap_result, data):
        pipeline, result = regression_shap_result
        figs = pipeline.plot(
            result.explanation_values,
            X=data["X_test"][:5],
            kind=["bar"],
        )
        assert figs is None or isinstance(figs, dict)

    def test_plot_classification_direct(self, classification_shap_result, classification_data):
        pipeline, result = classification_shap_result
        pipeline.plot(
            result.explanation_values,
            X=classification_data["X_test"][:5],
            kind=["bar"],
            result=result,
        )
