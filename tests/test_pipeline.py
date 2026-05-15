import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from uncertainty_explainer import (
    ClassificationExplanationResult,
    CQRConformalPredictor,
    ExplanationResult,
    LIMEExplanation,
    UncertaintyExplanationPipeline,
)
from uncertainty_explainer.explainability.pdp_explainer import PDPExplanation
from uncertainty_explainer.plots import generate_pdp_plots
import shap

# Avoid blocking plot windows during tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(xai_method="shap", **kwargs):
    return UncertaintyExplanationPipeline(
        model=Ridge(),
        xai_method=xai_method,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_requires_model_or_conformal_predictor(self):
        with pytest.raises(ValueError, match="model.*conformal_predictor"):
            UncertaintyExplanationPipeline()

    def test_invalid_xai_method_raises(self):
        with pytest.raises(ValueError, match="Unknown xai_method"):
            UncertaintyExplanationPipeline(model=Ridge(), xai_method="unknown")

    def test_custom_conformal_predictor_accepted(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        pipeline = UncertaintyExplanationPipeline(conformal_predictor=cp)
        pipeline.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = pipeline.predict(data["X_test"])
        assert lower.shape == (len(data["X_test"]),)


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_predict_before_fit_raises(self, data):
        pipeline = _make_pipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(data["X_test"])

    def test_predict_1d_raises(self, data):
        pipeline = _make_pipeline()
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="2D"):
            pipeline.predict(data["X_test"][:, 0])

    def test_predict_shapes(self, data):
        pipeline = _make_pipeline()
        pipeline.fit(data["X_train"], data["y_train"])
        lower, upper = pipeline.predict(data["X_test"])
        n = len(data["X_test"])
        assert lower.shape == (n,)
        assert upper.shape == (n,)
        assert np.all(lower <= upper)

    def test_auto_calib_split(self, data):
        # When X_calib / y_calib are not given, pipeline auto-splits
        pipeline = _make_pipeline()
        pipeline.fit(data["X_train"], data["y_train"], calib_size=0.2)
        lower, upper = pipeline.predict(data["X_test"])
        assert lower.shape == (len(data["X_test"]),)

    def test_explicit_calib_split(self, data):
        pipeline = _make_pipeline()
        pipeline.fit(
            data["X_train"], data["y_train"],
            data["X_calib"], data["y_calib"],
        )
        lower, upper = pipeline.predict(data["X_test"])
        assert lower.shape == (len(data["X_test"]),)

    def test_confidence_override(self, data):
        pipeline = _make_pipeline()
        pipeline.fit(data["X_train"], data["y_train"])
        lower_90, upper_90 = pipeline.predict(data["X_test"], confidence=0.90)
        lower_95, upper_95 = pipeline.predict(data["X_test"], confidence=0.95)
        assert (upper_95 - lower_95).mean() > (upper_90 - lower_90).mean()

    def test_feature_names_from_dataframe(self, data):
        cols = ["a", "b", "c", "d"]
        X_df = pd.DataFrame(data["X_train"], columns=cols)
        pipeline = _make_pipeline()
        pipeline.fit(X_df, data["y_train"])
        assert pipeline._feature_names == cols


# ---------------------------------------------------------------------------
# explain — SHAP
# ---------------------------------------------------------------------------

class TestExplainShap:
    def test_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_shap_result_shapes(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        X_small = data["X_test"][:5]
        result = pipeline.explain(X_small, show_plots=False)
        n = len(X_small)
        assert result.lower.shape == (n,)
        assert result.upper.shape == (n,)
        assert result.interval_width.shape == (n,)
        assert np.allclose(result.interval_width, result.upper - result.lower)

    def test_shap_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert isinstance(result.explanation_values, shap.Explanation)

    def test_shap_explainer_rebuilt_on_kwarg_change(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        X_small = data["X_test"][:5]
        pipeline.explain(X_small, show_plots=False)
        old_id = id(pipeline.explainer._shap_explainer)
        # Changing algorithm kwarg should trigger a rebuild
        pipeline.explain(X_small, show_plots=False, algorithm="permutation")
        new_id = id(pipeline.explainer._shap_explainer)
        assert old_id != new_id

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain(
                data["X_test"][:3], show_plots=False, plot_kind="invalid"
            )


# ---------------------------------------------------------------------------
# explain — PDP
# ---------------------------------------------------------------------------

class TestExplainPDP:
    def test_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_pdp_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        assert isinstance(result.explanation_values, PDPExplanation)

    def test_pdp_values_shape(self, data):
        n_features = data["X_test"].shape[1]
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        exp = result.explanation_values
        # values is a list of 1D arrays (one per feature); lengths may vary
        # for categorical features so we check the outer length, not .shape.
        assert len(exp.values) == n_features
        assert all(v.ndim == 1 for v in exp.values)

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain(
                data["X_test"][:3], show_plots=False, plot_kind="beeswarm"
            )


# ---------------------------------------------------------------------------
# explain — LIME
# ---------------------------------------------------------------------------

class TestExplainLIME:
    def test_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:3], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:3], show_plots=False)
        assert isinstance(result.explanation_values, LIMEExplanation)

    def test_local_coefficients_shape(self, data):
        n_features = data["X_test"].shape[1]
        X_small = data["X_test"][:3]
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain(X_small, show_plots=False)
        exp = result.explanation_values
        assert exp.local_coefficients.shape == (len(X_small), n_features)

    def test_feature_names_propagated(self, data):
        cols = ["a", "b", "c", "d"]
        X_df = pd.DataFrame(data["X_train"], columns=cols)
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(X_df, data["y_train"])
        result = pipeline.explain(data["X_test"][:2], show_plots=False)
        assert result.explanation_values.feature_names == cols

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain(
                data["X_test"][:2], show_plots=False, plot_kind="beeswarm"
            )


# ---------------------------------------------------------------------------
# PDP plot_kind / explainer.kind compatibility validation
# ---------------------------------------------------------------------------

def _fit_pdp_explanation(data, kind: str):
    """Helper: fit a PDP explainer with the given ``kind`` and return its explanation."""
    pipeline = _make_pipeline(xai_method="pdp")
    pipeline.fit(data["X_train"], data["y_train"])
    pipeline.explainer.fit(data["X_calib"], kind=kind)
    return pipeline.explainer.explain(data["X_test"][:10])


class TestPDPKindValidation:
    @pytest.mark.parametrize("bad_kind", ["average", "individual"])
    def test_pdp_ice_requires_both(self, data, bad_kind):
        explanation = _fit_pdp_explanation(data, kind=bad_kind)
        with pytest.raises(ValueError, match='plot_kind="pdp_ice"'):
            generate_pdp_plots(explanation, kinds=["pdp_ice"], show=False)

    def test_ice_requires_individual_or_both(self, data):
        explanation = _fit_pdp_explanation(data, kind="average")
        with pytest.raises(ValueError, match='plot_kind="ice"'):
            generate_pdp_plots(explanation, kinds=["ice"], show=False)

    def test_pdp_requires_average_or_both(self, data):
        explanation = _fit_pdp_explanation(data, kind="individual")
        with pytest.raises(ValueError, match='plot_kind="pdp"'):
            generate_pdp_plots(explanation, kinds=["pdp"], show=False)

    @pytest.mark.parametrize("plot_kind", ["pdp", "ice", "pdp_ice"])
    def test_kind_both_supports_all_plot_kinds(self, data, plot_kind):
        explanation = _fit_pdp_explanation(data, kind="both")
        figs = generate_pdp_plots(explanation, kinds=[plot_kind], show=False)
        assert plot_kind in figs

    @pytest.mark.parametrize("explainer_kind", ["average", "individual", "both"])
    def test_importance_works_under_any_kind(self, data, explainer_kind):
        # Importance uses .values, which is always populated (the explainer
        # internally requests "both" when kind="individual").
        explanation = _fit_pdp_explanation(data, kind=explainer_kind)
        figs = generate_pdp_plots(explanation, kinds=["importance"], show=False)
        assert "importance" in figs


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------

class TestClassificationPipeline:
    def test_auto_detects_classification_from_classifier(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(model=classifier)
        assert pipeline.task == "classification"

    def test_classification_requires_model(self):
        with pytest.raises(ValueError, match="classification.*requires a 'model'"):
            UncertaintyExplanationPipeline(task="classification")

    def test_invalid_conformal_method_for_classification(self, classifier):
        with pytest.raises(ValueError, match="not valid for"):
            UncertaintyExplanationPipeline(
                model=classifier,
                task="classification",
                conformal_method="normalized",
            )

    def test_predict_returns_prediction_set(self, classification_data, classifier):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        pred_set = pipeline.predict(classification_data["X_test"])
        assert pred_set.shape == (
            len(classification_data["X_test"]),
            classification_data["n_classes"],
        )
        assert pred_set.dtype == bool

    def test_explain_returns_classification_result(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification", xai_method="shap",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        X_small = classification_data["X_test"][:5]
        result = pipeline.explain(X_small, show_plots=False)

        assert isinstance(result, ClassificationExplanationResult)
        n = len(X_small)
        k = classification_data["n_classes"]
        assert result.prediction_set.shape == (n, k)
        assert result.prediction_set.dtype == bool
        assert result.p_values.shape == (n, k)
        assert result.set_size.shape == (n,)
        np.testing.assert_array_equal(
            result.set_size, result.prediction_set.sum(axis=1),
        )
        assert result.classes.shape == (k,)

    @pytest.mark.parametrize("xai_method", ["shap", "pdp", "lime"])
    def test_classification_works_with_each_xai_method(
        self, classification_data, classifier, xai_method,
    ):
        kwargs = {"n_lime_samples": 50} if xai_method == "lime" else {}
        pipeline = UncertaintyExplanationPipeline(
            model=classifier,
            task="classification",
            xai_method=xai_method,
            **kwargs,
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        result = pipeline.explain(
            classification_data["X_test"][:5], show_plots=False,
        )
        assert isinstance(result, ClassificationExplanationResult)
        assert result.explanation_values is not None

    def test_default_conformal_method_is_standard(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        assert pipeline.cp.method == "standard"

    @pytest.mark.parametrize("method", ["class_cond", "mondrian"])
    def test_conformal_method_class_cond_and_mondrian(
        self, classification_data, classifier, method,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification", conformal_method=method,
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        pred_set = pipeline.predict(classification_data["X_test"])
        assert pred_set.shape == (
            len(classification_data["X_test"]),
            classification_data["n_classes"],
        )

    def test_classification_auto_calib_stratified(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification", random_state=0,
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
            calib_size=0.2,
        )
        pred_set = pipeline.predict(classification_data["X_test"])
        assert pred_set.shape[0] == len(classification_data["X_test"])

    def test_classification_pdp_explanation_values_type(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification", xai_method="pdp",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        result = pipeline.explain(
            classification_data["X_test"][:5], show_plots=False,
        )
        assert isinstance(result, ClassificationExplanationResult)
        assert isinstance(result.explanation_values, PDPExplanation)

    def test_classification_lime_explanation_values_type(
        self, classification_data, classifier,
    ):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification",
            xai_method="lime", n_lime_samples=50,
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        result = pipeline.explain(
            classification_data["X_test"][:3], show_plots=False,
        )
        assert isinstance(result, ClassificationExplanationResult)
        assert isinstance(result.explanation_values, LIMEExplanation)
        assert result.explanation_values.local_coefficients.shape == (
            3, classification_data["X_test"].shape[1],
        )

    def test_explain_uncertainty_alias(self, classification_data, classifier):
        pipeline = UncertaintyExplanationPipeline(
            model=classifier, task="classification",
        )
        pipeline.fit(
            classification_data["X_train"], classification_data["y_train"],
        )
        result = pipeline.explain_uncertainty(
            classification_data["X_test"][:3], show_plots=False,
        )
        assert isinstance(result, ClassificationExplanationResult)

    def test_x_background_parameter(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        X_bg = data["X_train"][:20]
        pipeline.fit(
            data["X_train"], data["y_train"],
            data["X_calib"], data["y_calib"],
            X_background=X_bg,
        )
        assert pipeline._X_background.shape == X_bg.shape
        result = pipeline.explain(data["X_test"][:3], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_pdp_single_sample_raises(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="single-sample"):
            pipeline.explain(data["X_test"][:1], show_plots=False)

    def test_pdp_2d_requires_tuple_features(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="pdp_2d.*tuple"):
            pipeline.explain(
                data["X_test"][:5],
                show_plots=False,
                plot_kind="pdp_2d",
                features=[0, 1],
            )
