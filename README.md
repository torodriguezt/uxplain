<h1 align="center">uxplain</h1>

<p align="center">
  <em>Why is your model uncertain about <strong>this</strong> prediction?</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/uxplain/"><img alt="PyPI" src="https://img.shields.io/pypi/v/uxplain.svg"></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/uxplain.svg">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

---

**uxplain** pairs **conformal prediction** with **explainability (XAI)** so uncertainty stops being a black box.

Conformal prediction tells you *how* uncertain a model is — a prediction interval for regression, a prediction set for classification. uxplain turns that uncertainty into a scalar target and hands it to SHAP, PDP, or LIME, so you also learn **which features drive it**.

```
model ──▶ conformal predictor ──▶ uncertainty metric ──▶ XAI ──▶ plots
          intervals / sets         width / set size       shap · pdp · lime
```

## Installation

```bash
pip install uxplain            # requires Python ≥ 3.10
pip install -e ".[dev]"        # development, with tests
```

## Quick start

<table>
<tr><th>Regression — prediction intervals</th><th>Classification — prediction sets</th></tr>
<tr valign="top">
<td>

```python
from sklearn.ensemble import RandomForestRegressor
from uxplain import UncertaintyExplanationPipeline

pipe = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(),
)
pipe.fit(X_train, y_train)
res = pipe.explain(X_test)

res.lower           # lower bound
res.upper           # upper bound
res.interval_width  # upper - lower
```

</td>
<td>

```python
from sklearn.ensemble import RandomForestClassifier
from uxplain import UncertaintyExplanationPipeline

pipe = UncertaintyExplanationPipeline(
    model=RandomForestClassifier(),
)
pipe.fit(X_train, y_train)
res = pipe.explain(X_test)

res.prediction_set  # (n, k) bool — True = in set
res.p_values        # (n, k) conformal p-values
res.set_size        # (n,) classes per set
```

</td>
</tr>
</table>

`explain()` runs conformal prediction, the XAI backend, and the plots in one call. The task is auto-detected from the model; `fit()` carves out a calibration split automatically unless you pass `X_calib` / `y_calib`.

Runnable walkthroughs live in [`notebooks/`](notebooks/).

## Configuration

Everything is set through the pipeline constructor:

```python
UncertaintyExplanationPipeline(
    model=None,               # sklearn-compatible regressor or classifier
    confidence=0.9,           # nominal coverage level
    task="auto",              # "auto" | "regression" | "classification"
    conformal_method=None,    # None → task default (see below)
    xai_method="shap",        # "shap" | "pdp" | "lime"
    uncertainty_metric=None,  # None → task default (see below)
    n_lime_samples=5000,
    random_state=None,
    lower_model=None,         # quantile regressor for CQR (lower)
    upper_model=None,         # quantile regressor for CQR (upper)
    conformal_predictor=None, # custom ConformalPredictorProtocol
    explainer=None,           # custom UncertaintyExplainerProtocol
)
```

### What gets explained

The XAI backend explains a **scalar uncertainty metric** as a function of the input features.

| Task | `uncertainty_metric` | Meaning |
|---|---|---|
| Regression | **`"width"`** *(default)* | `upper - lower` — how wide the interval is |
| | `"lower"` / `"upper"` / `"midpoint"` | Interval bounds, or their centre |
| Classification | **`"set_size"`** *(default)* | Classes in the prediction set — larger = more uncertain |
| | `"credibility"` | Max p-value — how compatible the top class is with calibration |
| | `"confidence"` | `1 − second-highest p-value` — how decisively the runner-up is rejected |

### Conformal predictors

| Class | Task | `conformal_method` |
|---|---|---|
| `CrepesConformalPredictor` | Regression | `"standard"`, **`"normalized"`**, `"mondrian"`, `"normalized_mondrian"` |
| `CQRConformalPredictor` | Regression | Conformalized Quantile Regression (Romano et al., 2019) — needs `lower_model` + `upper_model` |
| `CrepesConformalClassifier` | Classification | **`"standard"`**, `"class_cond"`, `"mondrian"` |

The `crepes`-backed predictors are the defaults. Use CQR by passing the quantile regressors:

```python
from sklearn.ensemble import GradientBoostingRegressor
from uxplain import CQRConformalPredictor

pipe = UncertaintyExplanationPipeline(
    conformal_predictor=CQRConformalPredictor(
        lower_model=GradientBoostingRegressor(loss="quantile", alpha=0.05),
        upper_model=GradientBoostingRegressor(loss="quantile", alpha=0.95),
    )
)
```

### Explainability backends

All three work for both tasks.

| `xai_method` | Explainer | `plot_kind` options *(default in bold)* |
|---|---|---|
| **`"shap"`** | `ShapUncertaintyExplainer` | **`"beeswarm"`**, **`"bar"`**, **`"waterfall"`**, `"summary"` |
| `"pdp"` | `PDPUncertaintyExplainer` | **`"pdp"`**, `"ice"`, `"pdp_ice"`, `"importance"`, `"pdp_2d"` |
| `"lime"` | `LimeUncertaintyExplainer` | **`"local"`** |

```python
res = pipe.explain(X_test, plot_kind=["bar", "waterfall"], waterfall_index=3)
res = pipe.explain(X_test, show_plots=False)   # results only, no figures
```

Classification runs additionally get uncertainty-specific plots for free: a histogram of set sizes (`"set_size"`), per-class p-values for one sample (`"p_values"`), and a samples × classes membership heatmap (`"set_membership"`).

## Bring your own components

Any object satisfying the exported protocols can be dropped into the pipeline:

```python
from uxplain import (
    ConformalPredictorProtocol,    # regression:     fit + predict -> (lower, upper)
    ConformalClassifierProtocol,   # classification: fit + predict_set + predict_p
    UncertaintyExplainerProtocol,  # any:            fit + explain
)
```

## Development

```bash
pytest                 # test suite
pytest --cov=uxplain   # with coverage
```

## Authors

| | |
|---|---|
| **Veronica Seguro Varela** | MSc student in Statistical Sciences, Universidad Nacional de Colombia, Medellín |
| **Tomas Rodriguez Taborda** | Student in Informatics & Computer Science Engineering and Statistics, Universidad Nacional de Colombia, Medellín |
| **Rafael Izbicki** | PhD, Professor at Federal University of São Carlos, São Carlos |
| **Johnatan Cardona Jimenez** | PhD, Professor at Universidad Nacional de Colombia, Medellín |

Released under the [MIT License](LICENSE).
