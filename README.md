# uxplain

**Why is your model uncertain about *this* prediction?**

Conformal prediction tells you *how* uncertain a model is — a prediction interval for
regression, a prediction set for classification. `uxplain` goes one step further: it turns that
uncertainty into a number and explains it with SHAP, PDP, or LIME, so you also learn **which
features drive it**.

```
model → conformal predictor → uncertainty metric → XAI → plots
        intervals / sets       width / set size     shap · pdp · lime
```

## Install

```bash
pip install uxplain     # Python ≥ 3.10
```

## Use

```python
from sklearn.ensemble import RandomForestRegressor
from uxplain import UncertaintyExplanationPipeline

pipe = UncertaintyExplanationPipeline(model=RandomForestRegressor())
pipe.fit(X_train, y_train)
result = pipe.explain(X_test)

result.lower, result.upper   # prediction interval
result.interval_width        # what the explanation is about
```

`explain()` runs the conformal predictor, the explainer, and the plots in one call. Pass a
classifier instead and you get prediction sets (`result.prediction_set`, `result.p_values`,
`result.set_size`) — the task is detected from the model, and `fit()` holds out its own
calibration split unless you give it one.

From there, everything is a constructor argument:

```python
UncertaintyExplanationPipeline(
    model=model,
    confidence=0.9,             # nominal coverage
    xai_method="shap",          # "shap" | "pdp" | "lime"
    uncertainty_metric="width", # regression: width | lower | upper | midpoint
                                # classification: set_size | credibility | confidence
    conformal_method=None,      # crepes methods, or pass your own conformal_predictor
)
```

Worked examples for both tasks are in [`notebooks/`](notebooks/).

## Authors

- **Veronica Seguro Varela** — MSc student in Statistical Sciences, Universidad Nacional de Colombia, Medellín
- **Tomas Rodriguez Taborda** — Student in Informatics and Computer Science Engineering & Statistics, Universidad Nacional de Colombia, Medellín
- **Rafael Izbicki** — PhD, Professor at Federal University of São Carlos, São Carlos
- **Johnatan Cardona Jimenez** — PhD, Professor at Universidad Nacional de Colombia, Medellín

MIT License.
