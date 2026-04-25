# Uncertainty Explainer

A Python library that combines **conformal prediction** with **explainability methods (XAI)** to answer *why* a model is more or less uncertain for a given input.

Given any sklearn-compatible regressor, the pipeline:

1. Wraps it in a conformal predictor to produce calibrated prediction intervals
2. Explains the interval width using SHAP, PDP, or LIME — treating uncertainty itself as the target to be explained

---

## Installation

```bash
pip install -e ".[dev]"   # development install with test dependencies
pip install -e .           # install only
```

**Requirements:** Python ≥ 3.10, scikit-learn ≥ 1.4, shap ≥ 0.45, crepes ≥ 0.8, lime ≥ 0.2, matplotlib ≥ 3.8

---

## Quick start

```python
from sklearn.ensemble import RandomForestRegressor
from uncertainty_explainer import UncertaintyExplanationPipeline

pipeline = UncertaintyExplanationPipeline(model=RandomForestRegressor())
pipeline.fit(X_train, y_train)
result = pipeline.explain(X_test)

print(result.lower)           # lower prediction bound per sample
print(result.upper)           # upper prediction bound per sample
print(result.interval_width)  # width = upper - lower
```

---

## Conformal predictors

| Class | Description |
| --- | --- |
| `CrepesConformalPredictor` | Wraps [`crepes`](https://github.com/henrikbostrom/crepes). Supports `standard`, `normalized`, `mondrian`, and `normalized_mondrian` methods. Default predictor in the pipeline. |
| `CQRConformalPredictor` | Conformalized Quantile Regression (Romano et al., 2019). Requires two quantile regressors (lower and upper bound). |

```python
from uncertainty_explainer import CQRConformalPredictor
from sklearn.ensemble import GradientBoostingRegressor

cqr = CQRConformalPredictor(
    lower_model=GradientBoostingRegressor(loss="quantile", alpha=0.05),
    upper_model=GradientBoostingRegressor(loss="quantile", alpha=0.95),
)
pipeline = UncertaintyExplanationPipeline(conformal_predictor=cqr)
```

---

## Explainability methods

| `xai_method` | Class | Output |
| --- | --- | --- |
| `"shap"` (default) | `ShapUncertaintyExplainer` | `shap.Explanation` — global and local SHAP values |
| `"pdp"` | `PDPUncertaintyExplainer` | `PDPExplanation` — PDP curves and ICE lines per feature |
| `"lime"` | `LimeUncertaintyExplainer` | `LIMEExplanation` — linear coefficients per sample |

```python
# SHAP (default)
pipeline = UncertaintyExplanationPipeline(model=model, xai_method="shap")

# PDP
pipeline = UncertaintyExplanationPipeline(model=model, xai_method="pdp")

# LIME — local (one explanation per sample) or global (aggregated importance)
pipeline = UncertaintyExplanationPipeline(
    model=model, xai_method="lime", lime_scope="global"
)
```

### Plot kinds

Each method exposes different plot types via the `plot_kind` argument of `explain()`:

| Method | Available kinds | Default |
| --- | --- | --- |
| SHAP | `"beeswarm"`, `"bar"`, `"waterfall"`, `"summary"` | `["beeswarm", "bar", "waterfall"]` |
| PDP | `"pdp"`, `"ice"`, `"pdp_ice"`, `"importance"` | `["pdp", "importance"]` |
| LIME | `"local"`, `"global"` | `["local"]` / `["local", "global"]` |

```python
result = pipeline.explain(X_test, plot_kind=["bar", "waterfall"], waterfall_index=3)
```

---

## Pipeline API

```python
UncertaintyExplanationPipeline(
    model=None,                    # sklearn-compatible regressor
    confidence=0.9,                # nominal coverage level
    conformal_method="normalized", # ignored when conformal_predictor is given
    xai_method="shap",             # "shap" | "pdp" | "lime"
    lime_scope="local",            # "local" | "global" (LIME only)
    n_lime_samples=5000,           # perturbations per sample (LIME only)
    random_state=None,
    conformal_predictor=None,      # custom ConformalPredictorProtocol
    explainer=None,                # custom UncertaintyExplainerProtocol
)
```

```python
pipeline.fit(X_train, y_train)                        # auto-splits calibration set
pipeline.fit(X_train, y_train, X_calib, y_calib)      # explicit calibration set
lower, upper = pipeline.predict(X_test)
result = pipeline.explain(X_test, show_plots=False)
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
