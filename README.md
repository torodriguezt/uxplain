# uxplain

Explainability for conformal prediction uncertainty.

`uxplain` fits a split-conformal predictor on any scikit-learn estimator, reduces its output to a
scalar uncertainty summary *u(x)*, and attributes *u(x)* to the input features with SHAP, PDP/ICE
or LIME. The result tells you which features make the model more or less uncertain, not just
which ones drive its predictions.

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
from uxplain import UncertaintyExplanationPipeline

pipe = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(),
    confidence=0.9,              # nominal coverage
    xai_method="shap",           # "shap" | "pdp" | "lime"
    uncertainty_metric="width",  # the summary u(x) to explain
)
pipe.fit(X_train, y_train)       # holds out a calibration split unless you pass one
result = pipe.explain(X_test)

result.lower, result.upper       # conformal intervals
result.explanation_values        # attributions of u(x)
```

Pass a classifier and the task switches to prediction sets (`result.prediction_set`,
`result.p_values`, `result.set_size`). Worked examples are in [`notebooks/`](notebooks/).

## Authors

- **Veronica Seguro Varela** — Universidad Nacional de Colombia, Medellín
- **Tomas Rodriguez Taborda** — Universidad Nacional de Colombia, Medellín
- **Rafael Izbicki** — Federal University of São Carlos
- **Johnatan Cardona Jimenez** — Universidad Nacional de Colombia, Medellín

MIT License.
