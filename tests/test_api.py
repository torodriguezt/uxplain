"""Fitted-object API: scikit-learn conventions, summaries, results and explanations."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    CrepesConformalClassifier,
    CrepesConformalPredictor,
    LIMEExplanation,
    LimeUncertaintyExplainer,
    PDPExplanation,
    PDPUncertaintyExplainer,
    ShapExplanation,
    ShapUncertaintyExplainer,
    UncertaintyExplanation,
    UncertaintyExplanationPipeline,
    importance_table,
)

matplotlib.use("Agg")


def _cqr_pipeline(**kwargs):
    return UncertaintyExplanationPipeline(
        conformal_predictor=CQRConformalPredictor(
            GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=20,
                                      random_state=0),
            GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=20,
                                      random_state=0),
        ),
        random_state=0,
        **kwargs,
    )


class TestScikitLearnConventions:
    def test_constructor_stores_parameters_unchanged(self):
        pipeline = UncertaintyExplanationPipeline(model=Ridge())
        params = pipeline.get_params(deep=False)
        assert params["task"] == "auto"
        assert params["uncertainty_metric"] is None
        assert params["conformal_method"] is None
        assert params["explainer"] is None

    def test_nested_parameters_reach_components(self):
        pipeline = _cqr_pipeline()
        params = pipeline.get_params()
        assert params["conformal_predictor__lower_model__alpha"] == 0.05
        pipeline.set_params(conformal_predictor__lower_model__alpha=0.1)
        assert pipeline.conformal_predictor.lower_model.alpha == 0.1

    def test_clone_returns_unfitted_copy_with_same_parameters(self, data):
        pipeline = _cqr_pipeline().fit(data["X_train"], data["y_train"])
        copy = clone(pipeline)
        assert not hasattr(copy, "conformal_predictor_")
        with pytest.raises(RuntimeError, match="not fitted"):
            copy.predict(data["X_test"])
        copy.fit(data["X_train"], data["y_train"])
        np.testing.assert_allclose(copy.predict(data["X_test"]),
                                   pipeline.predict(data["X_test"]))

    def test_fit_does_not_modify_supplied_components(self, data):
        model = Ridge()
        pipeline = UncertaintyExplanationPipeline(model=model, random_state=0)
        pipeline.fit(data["X_train"], data["y_train"])
        assert not hasattr(model, "coef_")
        assert pipeline.conformal_predictor_.model is not model
        assert hasattr(pipeline.conformal_predictor_.model, "coef_")

    def test_set_params_is_validated_at_fit(self, data):
        pipeline = UncertaintyExplanationPipeline(model=Ridge())
        pipeline.set_params(uncertainty_metric="set_size")
        with pytest.raises(ValueError, match="not valid for regression"):
            pipeline.fit(data["X_train"], data["y_train"])

    def test_set_params_changes_the_fitted_pipeline(self, data):
        pipeline = UncertaintyExplanationPipeline(model=Ridge(), random_state=0)
        pipeline.set_params(confidence=0.5).fit(data["X_train"], data["y_train"])
        narrow = pipeline.evaluate(data["X_test"], data["y_test"])
        pipeline.set_params(confidence=0.95).fit(data["X_train"], data["y_train"])
        wide = pipeline.evaluate(data["X_test"], data["y_test"])
        assert wide.loc["overall", "mean_size"] > narrow.loc["overall", "mean_size"]

    def test_repr_lists_parameters(self):
        text = repr(UncertaintyExplanationPipeline(model=Ridge(), xai_method="pdp"))
        assert text.startswith("UncertaintyExplanationPipeline(")
        assert "xai_method='pdp'" in text
        assert "object at 0x" not in text

    @pytest.mark.parametrize("component", [
        CQRConformalPredictor(Ridge(), Ridge(), rearrange=True),
        CrepesConformalPredictor(Ridge(), method="mondrian"),
        CrepesConformalClassifier(RandomForestClassifier(), method="class_cond"),
        ShapUncertaintyExplainer(None, metric="midpoint", background_size=50),
        LimeUncertaintyExplainer(None, n_lime_samples=10),
        PDPUncertaintyExplainer(None, grid_resolution=7),
    ])
    def test_components_are_estimators(self, component):
        copy = clone(component)
        assert type(copy) is type(component)
        assert copy.get_params(deep=False).keys() == component.get_params(deep=False).keys()
        assert "object at 0x" not in repr(component)

    def test_fitted_attributes(self, data):
        frame = pd.DataFrame(data["X_train"], columns=list("abcd"))
        pipeline = UncertaintyExplanationPipeline(model=Ridge()).fit(frame, data["y_train"])
        assert pipeline.task_ == "regression"
        assert pipeline.uncertainty_metric_ == "width"
        assert pipeline.conformal_method_ == "normalized"
        assert pipeline.n_features_in_ == 4
        assert pipeline.feature_names_in_.tolist() == list("abcd")
        assert pipeline.n_train_ + pipeline.n_calibration_ == len(frame)

    def test_cp_alias_is_deprecated(self, data):
        pipeline = UncertaintyExplanationPipeline(model=Ridge())
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.warns(FutureWarning, match="conformal_predictor_"):
            assert pipeline.cp is pipeline.conformal_predictor_


class TestSummaryAndPrediction:
    def test_summary_before_and_after_fit(self, data):
        pipeline = _cqr_pipeline()
        before = pipeline.summary()
        assert isinstance(before, pd.Series)
        assert before["fitted"] is False
        assert before["conformal predictor"] == "CQRConformalPredictor"
        pipeline.fit(data["X_train"], data["y_train"])
        pipeline.explain(data["X_test"][:3], show_plots=False)
        after = pipeline.summary()
        assert after["fitted"] is True
        assert after["conformal method"] == "cqr"
        assert after["features"] == 4
        assert after["last SHAP algorithm"] == "tree"
        assert after["explained summary"].startswith("width")

    def test_classification_summary_and_p_values(self, classification_data, classifier):
        d = classification_data
        pipeline = UncertaintyExplanationPipeline(model=classifier, random_state=0)
        pipeline.fit(d["X_train"], d["y_train"], d["X_calib"], d["y_calib"])
        p_values = pipeline.predict_p(d["X_test"])
        assert p_values.shape == (len(d["X_test"]), len(pipeline.classes_))
        sets = pipeline.predict(d["X_test"])
        np.testing.assert_array_equal(sets, p_values >= 1 - pipeline.confidence)
        assert pipeline.summary()["classes"] == 3

    def test_predict_p_is_classification_only(self, data):
        pipeline = UncertaintyExplanationPipeline(model=Ridge())
        pipeline.fit(data["X_train"], data["y_train"])
        with pytest.raises(AttributeError, match="classification"):
            pipeline.predict_p(data["X_test"])


class TestExplanationClasses:
    @pytest.mark.parametrize(("xai_method", "kind", "extra"), [
        ("shap", ShapExplanation, {}),
        ("lime", LIMEExplanation, {}),
        ("pdp", PDPExplanation, {"kind": "both"}),
    ])
    def test_common_interface(self, data, xai_method, kind, extra):
        pipeline = UncertaintyExplanationPipeline(
            model=Ridge(), xai_method=xai_method, n_lime_samples=50, random_state=0,
        ).fit(pd.DataFrame(data["X_train"], columns=list("abcd")), data["y_train"])
        X = pd.DataFrame(data["X_test"][:6], columns=list("abcd"))
        result = pipeline.explain(X, show_plots=False, **extra)
        explanation = result.explanation_values
        assert isinstance(explanation, kind)
        assert isinstance(explanation, UncertaintyExplanation)
        assert explanation.method == xai_method
        assert explanation.metric == "width"

        table = result.feature_importance()
        pd.testing.assert_frame_equal(table, importance_table(explanation))
        assert set(table.index) <= set("abcd")

        frame = explanation.to_frame()
        assert isinstance(frame, pd.DataFrame) and len(frame) > 0
        figures = result.plot(show=False)
        assert figures and all(hasattr(f, "savefig") for f in figures.values())
        matplotlib.pyplot.close("all")

        text = repr(explanation)
        assert text.startswith(type(explanation).__name__)
        assert "array(" not in text and len(text) < 200

    def test_shap_frame_reconstructs_the_summary(self, data):
        pipeline = _cqr_pipeline().fit(data["X_train"], data["y_train"])
        result = pipeline.explain(data["X_test"][:5], show_plots=False)
        frame = result.explanation_values.to_frame()
        reconstructed = frame.drop(columns="base_value").sum(axis=1) + frame["base_value"]
        np.testing.assert_allclose(reconstructed, result.interval_width, atol=1e-6)

    def test_result_repr_and_frames(self, data, classification_data, classifier):
        regression = _cqr_pipeline().fit(data["X_train"], data["y_train"])
        result = regression.explain(data["X_test"][:5], show_plots=False)
        assert repr(result).startswith("ExplanationResult(rows=5")
        assert result.to_frame().columns.tolist() == ["lower", "upper", "width"]

        d = classification_data
        classification = UncertaintyExplanationPipeline(model=classifier, random_state=0)
        classification.fit(d["X_train"], d["y_train"])
        outcome = classification.explain(d["X_test"][:4], show_plots=False)
        assert repr(outcome).startswith("ClassificationExplanationResult(rows=4, classes=3")
        frame = outcome.to_frame()
        assert frame.columns.tolist() == ["prediction_set", "set_size", "credibility",
                                          "confidence"]
        assert [len(s) for s in frame["prediction_set"]] == frame["set_size"].tolist()
        np.testing.assert_allclose(frame["credibility"], outcome.p_values.max(axis=1))
