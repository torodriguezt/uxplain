# uxplain

**Why is your model uncertain about *this* prediction?**

Conformal prediction tells you *how* uncertain a model is — a prediction interval for
regression, a prediction set for classification. `uxplain` goes one step further: it turns that
uncertainty into a number and explains it with SHAP, PDP, or LIME, so you also learn **which
features drive it**.

- **Conformal backends:** [crepes](https://github.com/henrikbostrom/crepes) (standard,
  normalized, Mondrian) and conformalized quantile regression (CQR) for regression; crepes
  (standard, class-conditional, Mondrian) for classification.
- **Summaries *u(x)*:** interval `width`, `lower`, `upper` and `midpoint` for regression;
  `set_size`, `credibility` and `confidence` for classification.
- **Explainers:** SHAP, with an exact TreeSHAP shortcut when *u* is affine in tree-ensemble
  components (e.g. CQR width); PDP/ICE, including 2D interactions; LIME.

## Install

```bash
pip install uxplain     # Python >= 3.10
```

## Use

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

X, y = make_regression(n_samples=300, n_features=4, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipe = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(n_estimators=50, random_state=42),
    confidence=0.9,              # nominal coverage
    xai_method="shap",           # "shap" | "pdp" | "lime"
    uncertainty_metric="width",  # the summary u(x) to explain
    random_state=42,
)
pipe.fit(X_train, y_train)       # holds out a calibration split unless you pass one
result = pipe.explain(X_test[:5], show_plots=False)

result.lower, result.upper       # conformal intervals
result.explanation_values        # attributions of u(x)
```

Pass a classifier and the task switches to prediction sets (`result.prediction_set`,
`result.p_values`, `result.set_size`). Worked examples are in
[`notebooks/`](https://github.com/torodriguezt/uxplain/tree/main/notebooks),
and the [getting-started guide](https://github.com/torodriguezt/uxplain/blob/main/docs/docs/getting-started.md)
covers calibration, classification and development.

The explanations describe a scalar summary of the fitted conformal predictor.
They do not estimate causal effects or give a coverage guarantee for the
explanation itself. Coverage claims depend on the conformal method's assumptions,
including an appropriate held-out calibration sample.

## Development and release checks

```bash
python -m pip install -e ".[dev,release]"
python -m ruff check .
python -m pytest
python -m build
python -m twine check --strict dist/*
```

See the [release guide](https://github.com/torodriguezt/uxplain/blob/main/docs/docs/getting-started.md#releasing)
for installation checks and the publication checklist. Building and testing these
archives does not publish them to PyPI.

## Authors

- **Veronica Seguro Varela** — Universidad Nacional de Colombia, Medellín
- **Tomas Rodriguez Taborda** — Universidad Nacional de Colombia, Medellín
- **Rafael Izbicki** — Federal University of São Carlos
- **Johnatan Cardona Jimenez** — Universidad Nacional de Colombia, Medellín

MIT License.
