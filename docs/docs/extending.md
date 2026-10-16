# Extending uxplain

Pass a custom `conformal_predictor` to `ConformalExplainer` to explain
the intervals or sets produced by another conformal library. The main estimator clones
the backend, gives it separate training and calibration arrays, and connects the
fitted backend to the selected explainer. The custom predictor takes precedence
over `estimator` and `conformal_method`.

## Backend contracts

The protocols describe structural interfaces: a backend does not need to inherit
from them. Inputs have shape `(n_samples, n_features)` for `X`
and `(n_samples,)` for `y`. Training DataFrames retain their column names;
explainers may supply NumPy arrays containing masked numeric rows.

| Backend | Required members | Output |
| --- | --- | --- |
| Regression | `fit(X_train, y_train, X_calib, y_calib)` | Fit using the separate training and calibration samples. |
| Regression | `predict_interval(X, confidence=0.9)` | `(lower, upper)`, two arrays of shape `(n_samples,)`. |
| Classification | `fit(X_train, y_train, X_calib, y_calib)` | Fit and set `classes_`. |
| Classification | `predict_set(X, confidence=0.9)` | Boolean array of shape `(n_samples, n_classes)`. |
| Classification | `predict_p_values(X)` | Conformal p-values of shape `(n_samples, n_classes)`. |
| Classification | `classes_` | Class labels in the exact column order of both outputs. |

See [`ConformalPredictorProtocol`](api.md#uxplain.ConformalPredictorProtocol) and
[`ConformalClassifierProtocol`](api.md#uxplain.ConformalClassifierProtocol).
Classification is detected from the presence of both `predict_set` and
`predict_p_values`; use `task="classification"` or `task="regression"` to set it explicitly.

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

## Custom explainers

Pass an object implementing
[`UncertaintyExplainerProtocol`](api.md#uxplain.UncertaintyExplainerProtocol) as
`explainer=`. It needs `fit(X, y=None)` and `explain(X)`. `ConformalExplainer`
clones the object; if it has a `conformal_predictor` attribute, that attribute is bound to the
fitted conformal predictor. Configure the explainer through its constructor or nested `set_params`
parameters before fitting the main estimator. Its `fit` method returns itself,
and its `explain` method computes without plotting.

Return a subclass of
[`UncertaintyExplanation`](api.md#uxplain.UncertaintyExplanation) so the result
can expose `feature_importance()`, `to_frame()` and `plot()`. An explainer producing
SHAP attributions can return
[`ShapExplanation`](api.md#uxplain.ShapExplanation), supplying its values, base
values, feature names and metric. For a new explanation method, implement the
corresponding accessors in its result subclass, including its own importance
calculation. The built-in importance helper supports SHAP, LIME and PDP.

A custom explainer controls its own target and nominal coverage. Configure these
consistently with `ConformalExplainer`: `xai_method` and `uncertainty_metric` do not
reconfigure a supplied explainer, and only its `conformal_predictor` attribute is automatically
rebound. Keep explanatory summaries distinct from the coverage of the underlying
prediction intervals or sets.

## Check an adapter

Run the custom predictor integration test from the repository root:

```bash
python -m pytest tests/test_extensibility.py -q
```

For your own backend, also check
output shapes, class-column alignment, cloning, supported coverage levels and
row-wise determinism, including predictions on a row alone and within different
batches. Use held-out data for empirical evaluation and preserve the
[calibration assumptions](getting-started.md#calibration)
when creating training and calibration samples.
