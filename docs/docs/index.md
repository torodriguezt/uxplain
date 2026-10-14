# uxplain

**Explainability for conformal prediction uncertainty** — regression and classification.

`uxplain` combines conformal prediction with explainability methods (SHAP, PDP, LIME) to
answer *why* a model is more or less uncertain for a given input. It supports both
regression (prediction intervals) and classification (prediction sets).

## Install

```bash
pip install uxplain
```

## Quick start

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

X, y = make_regression(n_samples=300, n_features=4, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipeline = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(n_estimators=50, random_state=42),
    random_state=42,
)
pipeline.fit(X_train, y_train)
result = pipeline.explain(X_test[:5], show_plots=False)
```

See [Getting started](getting-started.md) for runnable examples, calibration,
classification, development and release checks. The
[notebooks](https://github.com/torodriguezt/uxplain/tree/main/notebooks) cover the
explainability backends and plots.

## Links

- Repository: <https://github.com/torodriguezt/uxplain>
- Issues: <https://github.com/torodriguezt/uxplain/issues>
