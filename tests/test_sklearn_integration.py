"""Interoperability tests exercise sklearn tools and the complete fitted workflow."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

from uxplain import (
    CQRConformalPredictor, CrepesConformalPredictor,
    LimeUncertaintyExplainer, PDPUncertaintyExplainer,
    ShapUncertaintyExplainer, UncertaintyExplanationPipeline,
)


@pytest.mark.parametrize("estimator, method", [
    (Ridge(), "standard"), (Ridge(), "normalized"), (LogisticRegression(), "standard"),
])
def test_sklearn_common_estimator_checks(estimator, method):
    check_estimator(UncertaintyExplanationPipeline(
        estimator=estimator, conformal_method=method, random_state=0,
    ))


@pytest.fixture
def observations():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["age", "hours", "education"])
    y = X.age.to_numpy() + 0.4 * rng.normal(size=len(X))
    return X, y


@pytest.mark.parametrize("classification", [False, True])
def test_grid_search_and_point_prediction_semantics(observations, classification):
    X, y = observations
    estimator = LogisticRegression() if classification else Ridge()
    if classification:
        y = np.where(y > 0, "yes", "no")
    pipeline = UncertaintyExplanationPipeline(
        estimator=estimator, conformal_method="standard", random_state=5,
    )
    search = GridSearchCV(pipeline, {"calibration_size": [0.2, 0.3]}, cv=3)
    search.fit(X=X, y=y)
    fitted = search.best_estimator_
    assert np.isfinite(search.best_score_)
    assert is_classifier(fitted) == classification
    assert is_regressor(fitted) != classification
    assert search.predict(X).shape == (len(X),)
    if classification:
        assert set(search.predict(X)) <= {"yes", "no"}
        assert fitted.predict_set(X).shape == fitted.predict_p_values(X).shape == (len(X), 2)
        np.testing.assert_allclose(fitted.predict_proba(X).sum(axis=1), 1)
    else:
        lower, upper = fitted.predict_interval(X)
        assert lower.shape == upper.shape == (len(X),)
    assert not hasattr(estimator, "coef_")


@pytest.mark.parametrize("method", ["shap", "pdp", "lime"])
def test_named_preprocessing_works_through_explanation(observations, method):
    X, y = observations
    estimator = make_pipeline(
        ColumnTransformer([("scale", StandardScaler(), ["age", "hours", "education"])]),
        Ridge(),
    )
    # A varying summary exercises masked inputs and DataFrame reconstruction.
    pipeline = UncertaintyExplanationPipeline(
        estimator=estimator, conformal_method="standard", xai_method=method,
        uncertainty_metric="midpoint", random_state=9, n_lime_samples=50,
    ).fit(X=X, y=y)
    before = plt.get_fignums()
    result = pipeline.explain(X.iloc[:4])
    assert plt.get_fignums() == before
    assert result.explanation_values.feature_names == list(X.columns)
    assert np.isfinite(pipeline.predict(X)).all()
    # Preprocessing was fitted on proper training rows, excluding calibration.
    fitted_scale = pipeline.conformal_predictor_.estimator_.named_steps[
        "columntransformer"].named_transformers_["scale"]
    assert fitted_scale.n_samples_seen_ == pipeline.n_train_ < len(X)
    with pytest.raises(ValueError, match="feature names"):
        pipeline.predict(X.iloc[:, ::-1])


@pytest.mark.parametrize("cls", [ShapUncertaintyExplainer, PDPUncertaintyExplainer,
                                 LimeUncertaintyExplainer])
def test_standalone_fit_is_chainable_and_detectable(observations, cls):
    X, y = observations
    cp = CrepesConformalPredictor(Ridge(), method="standard")
    cp.fit(X.iloc[:80], y[:80], X.iloc[80:], y[80:])
    explainer = cls(cp, uncertainty_metric="midpoint")
    with pytest.raises(NotFittedError):
        check_is_fitted(explainer)
    assert explainer.fit(X) is explainer
    check_is_fitted(explainer)
    with pytest.raises(TypeError):
        explainer.fit(X, grid_resoluton=10)


def test_pdp_preserves_parameters_and_obeys_nested_set_params(observations):
    X, y = observations
    features = ["age"]
    explainer = PDPUncertaintyExplainer(features=features)
    pipeline = UncertaintyExplanationPipeline(
        estimator=Ridge(), conformal_method="standard", explainer=explainer,
        random_state=4,
    ).set_params(explainer__features=["hours"])
    pipeline.fit(X, y)
    result = pipeline.explain(X.iloc[:10])
    assert result.explanation_values.features == [1]
    assert pipeline.explainer_.get_params()["features"] == ["hours"]
    assert features == ["age"]
    assert clone(pipeline).get_params()["explainer__features"] == ["hours"]


def test_cqr_point_estimator_is_explicit(observations):
    X, y = observations
    cp = CQRConformalPredictor(Ridge(), Ridge()).fit(
        X.iloc[:80], y[:80], X.iloc[80:], y[80:],
    )
    assert len(cp.predict_interval(X)) == 2
    with pytest.raises(AttributeError, match="point prediction requires"):
        cp.predict(X)
    cp.set_params(estimator=Ridge()).fit(X.iloc[:80], y[:80], X.iloc[80:], y[80:])
    assert cp.predict(X).shape == (len(X),)
    assert not hasattr(cp.estimator, "coef_")
    assert not hasattr(cp.lower_estimator, "coef_")
