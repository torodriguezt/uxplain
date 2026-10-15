import matplotlib
import numpy as np
import pytest

from uxplain import (
    ClassificationExplanationResult,
    UncertaintyExplanationPipeline,
)
from uxplain.plots import (
    generate_lime_plots,
    generate_pdp_plots,
    generate_shap_plots,
)

matplotlib.use("Agg")


@pytest.mark.parametrize("kind", ["bar", "beeswarm", "waterfall"])
def test_single_row_shap_slice_can_be_plotted(kind):
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=np.array([[0.3, -0.2]]), base_values=np.array([1.]),
        data=np.array([[2., 3.]]), feature_names=["a", "b"],
    )[0]
    figures = generate_shap_plots(explanation, kinds=[kind], show=False)
    assert kind in figures
    plt.close(figures[kind])


def test_waterfall_requires_real_baseline():
    import shap

    from uxplain.plotting import shap_waterfall

    with pytest.raises(ValueError, match="base_values"):
        shap_waterfall(shap.Explanation(values=np.array([[0.3, -0.2]])))


def test_waterfall_preserves_total_when_features_are_grouped():
    import matplotlib.pyplot as plt
    import shap
    from uxplain import shap_waterfall

    values = np.array([[0.8, -0.4, 0.15, -0.05]])
    explanation = shap.Explanation(values=values, base_values=np.array([2.0]))
    ax = shap_waterfall(explanation, max_display=2)
    # The displayed reference and result must include the grouped contributions.
    endpoints = ax.collections[0].get_offsets()[:, 0]
    np.testing.assert_allclose(endpoints, [2.0, 2.0 + values.sum()])
    assert "2 other features" in [tick.get_text() for tick in ax.get_yticklabels()]
    assert any(np.ptp(patch.get_xy()[:, 0]) > 0 for patch in ax.patches)
    plt.close(ax.figure)


def test_importance_panels_use_the_requested_common_order():
    import matplotlib.pyplot as plt
    import pandas as pd
    from uxplain import importance_plot

    table = pd.DataFrame({"importance": [0.8, 0.2], "ci_lower": [0.7, 0.1],
                          "ci_upper": [0.9, 0.3]}, index=["a", "b"])
    ax = importance_plot(table, order=["b", "a"])
    assert [tick.get_text() for tick in ax.get_yticklabels()][::-1] == ["b", "a"]
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 0], [0.8, 0.2])
    plt.close(ax.figure)


def test_pdp_interaction_with_constant_feature():
    import matplotlib.pyplot as plt

    from uxplain.plotting import pdp_interaction

    ax = pdp_interaction(np.array([1.]), np.array([0., 1.]), np.array([[2., 3.]]))
    assert len(ax.collections) == 1
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Fixtures — fitted pipelines shared across test classes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_shap_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(estimator=Ridge(), xai_method="shap")
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:5], show_plots=False)


@pytest.fixture(scope="module")
def regression_pdp_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(estimator=Ridge(), xai_method="pdp")
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:5], show_plots=False)


@pytest.fixture(scope="module")
def regression_lime_result(data):
    from sklearn.linear_model import Ridge
    pipeline = UncertaintyExplanationPipeline(
        estimator=Ridge(), xai_method="lime", n_lime_samples=50,
    )
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, pipeline.explain(data["X_test"][:3], show_plots=False)


@pytest.fixture
def classification_shap_result(classification_data, classifier):
    pipeline = UncertaintyExplanationPipeline(
        estimator=classifier, task="classification", xai_method="shap",
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
        pipeline.explainer_.set_params(kind="both").fit(data["X_calib"])
        exp = pipeline.explainer_.explain(data["X_test"][:5])
        figs = generate_pdp_plots(exp, kinds=["pdp"], show=False)
        assert "pdp" in figs

    def test_importance_populated(self, regression_pdp_result, data):
        pipeline, _ = regression_pdp_result
        pipeline.explainer_.fit(data["X_calib"])
        exp = pipeline.explainer_.explain(data["X_test"][:5])
        assert exp.importance is not None
        assert len(exp.importance) == len(exp.features)
        assert all(v >= 0 for v in exp.importance)

    def test_importance_returns_figure(self, regression_pdp_result, data):
        pipeline, _ = regression_pdp_result
        pipeline.explainer_.fit(data["X_calib"])
        exp = pipeline.explainer_.explain(data["X_test"][:5])
        figs = generate_pdp_plots(exp, kinds=["importance"], show=False)
        assert "importance" in figs


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
            show=False,
        )
        assert set(figs) == {"bar"}

    def test_plot_pdp_direct(self, regression_pdp_result):
        pipeline, result = regression_pdp_result
        figs = pipeline.plot(result.explanation_values, kind=["pdp"], show=False)
        assert set(figs) == {"pdp"}

    def test_plot_lime_direct(self, regression_lime_result):
        pipeline, result = regression_lime_result
        figs = pipeline.plot(result.explanation_values, kind=["local"], show=False)
        assert set(figs) == {"local"}

    def test_plot_classification_direct(self, classification_shap_result, classification_data):
        pipeline, result = classification_shap_result
        pipeline.plot(
            result.explanation_values,
            X=classification_data["X_test"][:5],
            kind=["bar"],
            result=result,
        )
