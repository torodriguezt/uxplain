import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from uncertainty_explainer import (
    CQRConformalPredictor,
    ExplanationResult,
    LIMEExplanation,
    UncertaintyExplanationPipeline,
)
from uncertainty_explainer.explainability.pdp_explainer import PDPExplanation
import shap


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
# explain_uncertainty — SHAP
# ---------------------------------------------------------------------------

class TestExplainShap:
    def test_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:5], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_shap_result_shapes(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        X_small = data["X_test"][:5]
        result = pipeline.explain_uncertainty(X_small, show_plots=False)
        n = len(X_small)
        assert result.lower.shape == (n,)
        assert result.upper.shape == (n,)
        assert result.interval_width.shape == (n,)
        assert np.allclose(result.interval_width, result.upper - result.lower)

    def test_shap_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:5], show_plots=False)
        assert isinstance(result.explanation_values, shap.Explanation)

    def test_shap_explainer_rebuilt_on_kwarg_change(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        X_small = data["X_test"][:5]
        pipeline.explain_uncertainty(X_small, show_plots=False)
        old_id = id(pipeline.explainer._shap_explainer)
        # Changing algorithm kwarg should trigger a rebuild
        pipeline.explain_uncertainty(X_small, show_plots=False, algorithm="permutation")
        new_id = id(pipeline.explainer._shap_explainer)
        assert old_id != new_id

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="shap")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain_uncertainty(
                data["X_test"][:3], show_plots=False, plot_kind="invalid"
            )


# ---------------------------------------------------------------------------
# explain_uncertainty — PDP
# ---------------------------------------------------------------------------

class TestExplainPDP:
    def test_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:5], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_pdp_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:5], show_plots=False)
        assert isinstance(result.explanation_values, PDPExplanation)

    def test_pdp_values_shape(self, data):
        n_features = data["X_test"].shape[1]
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:5], show_plots=False)
        exp = result.explanation_values
        assert exp.values.shape[0] == n_features

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="pdp")
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain_uncertainty(
                data["X_test"][:3], show_plots=False, plot_kind="beeswarm"
            )


# ---------------------------------------------------------------------------
# explain_uncertainty — LIME
# ---------------------------------------------------------------------------

class TestExplainLIME:
    def test_local_returns_explanation_result(self, data):
        pipeline = _make_pipeline(xai_method="lime", lime_scope="local", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:3], show_plots=False)
        assert isinstance(result, ExplanationResult)

    def test_local_explanation_values_type(self, data):
        pipeline = _make_pipeline(xai_method="lime", lime_scope="local", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:3], show_plots=False)
        assert isinstance(result.explanation_values, LIMEExplanation)
        assert result.explanation_values.scope == "local"

    def test_global_scope(self, data):
        pipeline = _make_pipeline(xai_method="lime", lime_scope="global", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:3], show_plots=False)
        assert result.explanation_values.scope == "global"

    def test_local_coefficients_shape(self, data):
        n_features = data["X_test"].shape[1]
        X_small = data["X_test"][:3]
        pipeline = _make_pipeline(xai_method="lime", lime_scope="local", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(X_small, show_plots=False)
        exp = result.explanation_values
        assert exp.local_coefficients.shape == (len(X_small), n_features)
        assert exp.global_importance.shape == (n_features,)

    def test_global_importance_is_mean_abs_local(self, data):
        pipeline = _make_pipeline(xai_method="lime", lime_scope="local", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:3], show_plots=False)
        exp = result.explanation_values
        expected = np.mean(np.abs(exp.local_coefficients), axis=0)
        np.testing.assert_allclose(exp.global_importance, expected)

    def test_feature_names_propagated(self, data):
        cols = ["a", "b", "c", "d"]
        X_df = pd.DataFrame(data["X_train"], columns=cols)
        pipeline = _make_pipeline(xai_method="lime", lime_scope="local", n_lime_samples=50)
        pipeline.fit(X_df, data["y_train"])
        result = pipeline.explain_uncertainty(data["X_test"][:2], show_plots=False)
        assert result.explanation_values.feature_names == cols

    def test_invalid_plot_kind_raises(self, data):
        pipeline = _make_pipeline(xai_method="lime", n_lime_samples=50)
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(ValueError, match="Unknown plot kind"):
            pipeline.explain_uncertainty(
                data["X_test"][:2], show_plots=False, plot_kind="beeswarm"
            )
