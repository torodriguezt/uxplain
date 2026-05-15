# Uncertainty Explainer

A Python library that combines **conformal prediction** with **explainability methods (XAI)** to answer *why* a model is more or less uncertain for a given input.

Supports both **regression** (prediction intervals) and **classification** (prediction sets), with SHAP, PDP, and LIME as explainability backends.

---

## Installation

```bash
pip install uncertainty-explainer
```

**Requirements:** Python ≥ 3.10, scikit-learn ≥ 1.4, shap ≥ 0.45, crepes ≥ 0.8, lime ≥ 0.2, matplotlib ≥ 3.8

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

---

## Quick start — Regression

```python
from sklearn.ensemble import RandomForestRegressor
from uncertainty_explainer import UncertaintyExplanationPipeline

pipeline = UncertaintyExplanationPipeline(model=RandomForestRegressor())
pipeline.fit(X_train, y_train)
result = pipeline.explain(X_test)

print(result.lower)           # lower bound per sample
print(result.upper)           # upper bound per sample
print(result.interval_width)  # width = upper - lower
```

## Quick start — Classification

```python
from sklearn.ensemble import RandomForestClassifier
from uncertainty_explainer import UncertaintyExplanationPipeline

pipeline = UncertaintyExplanationPipeline(
    model=RandomForestClassifier(),
    task="classification",          # or omit — auto-detected from the model
    uncertainty_metric="set_size",  # default for classification
)
pipeline.fit(X_train, y_train)
result = pipeline.explain(X_test)

print(result.prediction_set)  # (n_samples, n_classes) bool — True = in set
print(result.p_values)        # (n_samples, n_classes) conformal p-values
print(result.set_size)        # (n_samples,) number of classes in each set
print(result.classes)         # class labels in model order
```

---

## Conformal predictors

### Regression

| Class | Description |
|---|---|
| `CrepesConformalPredictor` | Wraps [`crepes`](https://github.com/henrikbostrom/crepes). Methods: `"standard"`, `"normalized"` (default), `"mondrian"`, `"normalized_mondrian"`. |
| `CQRConformalPredictor` | Conformalized Quantile Regression (Romano et al., 2019). Requires two quantile regressors. |

```python
from uncertainty_explainer import CQRConformalPredictor
from sklearn.ensemble import GradientBoostingRegressor

pipeline = UncertaintyExplanationPipeline(
    conformal_predictor=CQRConformalPredictor(
        lower_model=GradientBoostingRegressor(loss="quantile", alpha=0.05),
        upper_model=GradientBoostingRegressor(loss="quantile", alpha=0.95),
    )
)
```

### Classification

| Class | Description |
|---|---|
| `CrepesConformalClassifier` | Wraps `crepes.WrapClassifier`. Methods: `"standard"` (default), `"class_cond"`, `"mondrian"`. |

```python
from uncertainty_explainer import CrepesConformalClassifier
from sklearn.ensemble import GradientBoostingClassifier

pipeline = UncertaintyExplanationPipeline(
    model=GradientBoostingClassifier(),
    task="classification",
    conformal_method="class_cond",  # per-class conditional coverage
)
```

---

## Explainability methods

All three XAI methods work for both regression and classification. They explain a **scalar uncertainty metric** (interval width for regression, set size for classification) as a function of input features.

| `xai_method` | Class | Output type |
|---|---|---|
| `"shap"` (default) | `ShapUncertaintyExplainer` | `shap.Explanation` |
| `"pdp"` | `PDPUncertaintyExplainer` | `PDPExplanation` |
| `"lime"` | `LimeUncertaintyExplainer` | `LIMEExplanation` |

```python
# SHAP (default)
pipeline = UncertaintyExplanationPipeline(model=model, xai_method="shap")

# PDP
pipeline = UncertaintyExplanationPipeline(model=model, xai_method="pdp")

# LIME
pipeline = UncertaintyExplanationPipeline(model=model, xai_method="lime", n_lime_samples=5000)
```

### Plot kinds

| Method | `plot_kind` options | Default |
|---|---|---|
| SHAP | `"beeswarm"`, `"bar"`, `"waterfall"`, `"summary"` | `["beeswarm", "bar", "waterfall"]` |
| PDP | `"pdp"`, `"ice"`, `"pdp_ice"`, `"importance"`, `"pdp_2d"` | `["pdp", "importance"]` |
| LIME | `"local"` | `["local"]` |

```python
result = pipeline.explain(X_test, plot_kind=["bar", "waterfall"], waterfall_index=3)
```

For classification, uncertainty-specific plots are generated automatically alongside the XAI plots:

| Classification plot | Description |
|---|---|
| `"set_size"` | Histogram of prediction set sizes across all test samples |
| `"p_values"` | Bar chart of conformal p-values per class for one sample |
| `"set_membership"` | Heatmap of set membership (samples × classes) |

---

## Uncertainty metrics

### Regression

| `uncertainty_metric` | Description |
|---|---|
| `"width"` (default) | `upper - lower` — how wide the interval is |
| `"lower"` | Lower prediction bound |
| `"upper"` | Upper prediction bound |
| `"midpoint"` | `(lower + upper) / 2` — central estimate |

### Classification

| `uncertainty_metric` | Description |
|---|---|
| `"set_size"` (default) | Number of classes in the prediction set — larger = more uncertain |
| `"credibility"` | Max p-value across classes — how compatible the top class is with calibration |
| `"confidence"` | `1 − second-highest p-value` — how decisively the runner-up is rejected |

```python
# Explain which features make the model include more classes in its prediction set
pipeline = UncertaintyExplanationPipeline(
    model=classifier,
    task="classification",
    uncertainty_metric="set_size",
)

# Explain which features drive the model's credibility score
pipeline = UncertaintyExplanationPipeline(
    model=classifier,
    task="classification",
    uncertainty_metric="credibility",
    xai_method="shap",
)
```

---

## Pipeline API

```python
UncertaintyExplanationPipeline(
    model=None,                     # sklearn-compatible regressor or classifier
    confidence=0.9,                 # nominal coverage level
    task="auto",                    # "auto" | "regression" | "classification"
    conformal_method=None,          # method string or None for task default
    xai_method="shap",              # "shap" | "pdp" | "lime"
    uncertainty_metric=None,        # None for task default ("width" / "set_size")
    n_lime_samples=5000,            # LIME perturbations per sample
    random_state=None,
    lower_model=None,               # quantile regressor for CQR (lower)
    upper_model=None,               # quantile regressor for CQR (upper)
    conformal_predictor=None,       # custom ConformalPredictorProtocol
    explainer=None,                 # custom UncertaintyExplainerProtocol
)
```

```python
# Fit — auto-splits calibration set if X_calib / y_calib not provided
pipeline.fit(X_train, y_train)
pipeline.fit(X_train, y_train, X_calib, y_calib)

# Predict
lower, upper = pipeline.predict(X_test)           # regression
prediction_set = pipeline.predict(X_test)          # classification — (n, n_classes) bool

# Explain (runs conformal prediction + XAI + plots)
result = pipeline.explain(X_test)
result = pipeline.explain(X_test, show_plots=False)
result = pipeline.explain(X_test, plot_kind="bar", waterfall_index=0)
```

---

## Custom predictors and explainers

The library exposes protocols for bringing your own components:

```python
from uncertainty_explainer import (
    ConformalPredictorProtocol,    # regression: fit + predict -> (lower, upper)
    ConformalClassifierProtocol,   # classification: fit + predict_set + predict_p
    UncertaintyExplainerProtocol,  # fit + explain -> Any
)
```

Type aliases for metrics are also importable:

```python
from uncertainty_explainer import RegressionMetric, ClassificationMetric, UncertaintyMetric
```

---

## Running tests

```bash
pytest
pytest --cov=uncertainty_explainer   # with coverage report
```

---

## Authors

- **Veronica Seguro Varela** — MSc student in Statistical Sciences, Universidad Nacional de Colombia, Medellín
- **Tomas Rodriguez Taborda** — student in Informatics and Computer Science Engineering & Statistics, Universidad Nacional de Colombia, Medellín
- **Rafael Izbicki** — PhD, Professor at Federal University of São Carlos, São Carlos
- **Johnatan Cardona** — PhD, Professor at Universidad Nacional de Colombia, Medellín
