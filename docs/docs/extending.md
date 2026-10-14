# Extending uxplain

Pass a custom `conformal_predictor` to `UncertaintyExplanationPipeline` to explain
the intervals or sets produced by another conformal library. The pipeline clones
the backend, gives it separate training and calibration arrays, and connects the
fitted backend to the selected explainer. The custom predictor takes precedence
over `model` and `conformal_method`.

## Backend contracts

The protocols describe structural interfaces: a backend does not need to inherit
from them. Inputs are NumPy arrays with shape `(n_samples, n_features)` for `X`
and `(n_samples,)` for `y`.

| Backend | Required members | Output |
| --- | --- | --- |
| Regression | `fit(X_train, y_train, X_calib, y_calib)` | Fit using the separate training and calibration samples. |
| Regression | `predict(X, confidence=0.9)` | `(lower, upper)`, two arrays of shape `(n_samples,)`. |
| Classification | `fit(X_train, y_train, X_calib, y_calib)` | Fit and set `classes_`. |
| Classification | `predict_set(X, confidence=0.9)` | Boolean array of shape `(n_samples, n_classes)`. |
| Classification | `predict_p(X)` | Conformal p-values of shape `(n_samples, n_classes)`. |
| Classification | `classes_` | Class labels in the exact column order of both outputs. |

See [`ConformalPredictorProtocol`](api.md#uxplain.ConformalPredictorProtocol) and
[`ConformalClassifierProtocol`](api.md#uxplain.ConformalClassifierProtocol).
Classification is detected from the presence of both `predict_set` and
`predict_p`; use `task="classification"` or `task="regression"` to set it explicitly.

Inherit from scikit-learn's `BaseEstimator` and store constructor arguments
unchanged to support `get_params()`, `set_params()` and `clone()`. Otherwise the
object must support deep copying. Inspect the fitted clone through
`pipeline.conformal_predictor_`; the object supplied to the constructor is not
fitted in place. Nested estimator parameters remain available through names such
as `conformal_predictor__estimator__n_estimators`.

The scalar summary being explained must be finite and deterministic for each row,
independent of row order and batch size. SHAP and PDP evaluate artificial feature
combinations formed from observed and background rows; LIME evaluates perturbed
rows. A backend must accept these inputs. A fixed seed for random tie-breaking
does not by itself make predictions independent of batch layout. Use
non-smoothed p-values for classification explanations. Unbounded intervals may
still be valid predictions, but the chosen explanation target must be finite.

## A MAPIE regression adapter

This example uses MAPIE 1.5.0 and its `SplitConformalRegressor`. Install it as an
optional dependency alongside uxplain:

```bash
python -m pip install uxplain "mapie==1.5.0"
```

When working from this repository, install the local package with
`python -m pip install -e ".[dev]" "mapie==1.5.0"` instead. The same adapter is
exercised by `tests/test_extensibility.py`.

MAPIE configures its coverage levels before fitting. The adapter therefore
accepts a tuple of levels and selects the corresponding interval at prediction
time; requesting an unconfigured level raises an error. The normalized residual
score allows interval widths to vary with the features.

```python
import numpy as np
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from uxplain import UncertaintyExplanationPipeline


class MapieConformalRegressor(BaseEstimator):
    """uxplain backend for MAPIE's split conformal regressor."""

    def __init__(self, estimator, confidence_levels=(0.9,),
                 conformity_score="residual_normalized"):
        self.estimator = estimator
        self.confidence_levels = confidence_levels
        self.conformity_score = conformity_score

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.mapie_ = SplitConformalRegressor(
            clone(self.estimator),
            confidence_level=list(self.confidence_levels),
            conformity_score=self.conformity_score,
            prefit=False,
        )
        self.mapie_.fit(X_train, y_train)
        self.mapie_.conformalize(X_calib, y_calib)
        return self

    def predict(self, X, confidence=0.9):
        levels = list(self.confidence_levels)
        if confidence not in levels:
            raise ValueError(f"confidence must be one of {levels}.")
        _, intervals = self.mapie_.predict_interval(X)
        k = levels.index(confidence)
        return intervals[:, 0, k], intervals[:, 1, k]


# The second feature controls the noise scale in this synthetic example.
rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, size=(1200, 3))
y = X[:, 0] + np.exp(X[:, 1]) * rng.standard_normal(1200)
X_train, y_train = X[:900], y[:900]
X_test, y_test = X[900:], y[900:]

backend = MapieConformalRegressor(
    GradientBoostingRegressor(n_estimators=50, random_state=0),
)
pipeline = UncertaintyExplanationPipeline(
    conformal_predictor=backend,
    confidence=0.9,
    uncertainty_metric="width",
    random_state=0,
).fit(X_train, y_train)

print(pipeline.summary())
print(pipeline.evaluate(X_test, y_test))
result = pipeline.explain(X_test[:20], show_plots=False)
print(result.to_frame())
print(result.feature_importance())

assert pipeline.conformal_predictor_ is not backend
assert result.explanation_values.values.shape == (20, 3)
assert np.isfinite(result.explanation_values.values).all()
```

The pipeline reserves 20% of `X_train` for calibration by default. To supply your
own split, pass both `X_calib` and `y_calib` to `fit()` as shown in
[Getting started](getting-started.md#calibration). MAPIE's `fit()` sees only the
training sample, and `conformalize()` sees only the calibration sample. Keep the
test sample independent of both.

Custom backends use the generic SHAP path. With the three features in this
example, `pipeline.algorithm_used` is `"exact"`; this means exact SHAP evaluation
of the composed summary, rather than the TreeSHAP shortcut used by some built-in
backends. Evaluation reports empirical coverage on the selected test sample;
the backend's statistical guarantees still depend on its conformal method and
the data assumptions.

## Custom explainers

Pass an object implementing
[`UncertaintyExplainerProtocol`](api.md#uxplain.UncertaintyExplainerProtocol) as
`explainer=`. It needs `fit(X_background, **kwargs)` and `explain(X)`. The pipeline
clones the object; if it has a `cp` attribute, that attribute is bound to the
fitted conformal predictor. Additional keyword arguments to `pipeline.explain()`
are forwarded to the explainer's `fit()` method, which is run when the explainer
is first used and when those arguments change.

Return a subclass of
[`UncertaintyExplanation`](api.md#uxplain.UncertaintyExplanation) so the result
can expose `feature_importance()`, `to_frame()` and `plot()`. An explainer producing
SHAP attributions can return
[`ShapExplanation`](api.md#uxplain.ShapExplanation), supplying its values, base
values, feature names and metric. For a new explanation method, implement the
corresponding accessors in its result subclass, including its own importance
calculation. The built-in importance helper supports SHAP, LIME and PDP.

A custom explainer controls its own target and nominal coverage. Configure these
consistently with the pipeline: `xai_method` and `uncertainty_metric` do not
reconfigure a supplied explainer, and only its `cp` attribute is automatically
rebound. Keep explanatory summaries distinct from the coverage of the underlying
prediction intervals or sets.

## Check an adapter

Run the integration tests from the repository root after installing MAPIE:

```bash
python -c "import mapie; print(mapie.__version__)"
python -m pytest tests/test_extensibility.py -q
```

The MAPIE test is skipped if MAPIE is missing. For your own backend, also check
output shapes, class-column alignment, cloning, supported coverage levels and
row-wise determinism, including predictions on a row alone and within different
batches. Use held-out data for empirical evaluation and preserve the
[statistical assumptions](getting-started.md#statistical-scope-and-limitations)
when creating training and calibration samples.
