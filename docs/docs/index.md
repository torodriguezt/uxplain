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
from uxplain import UncertaintyExplanationPipeline

pipeline = UncertaintyExplanationPipeline(model=RandomForestRegressor())
pipeline.fit(X_train, y_train)
result = pipeline.explain(X_test)
```

See the [README](https://github.com/vseguro/uxplain#readme) for the full API:
conformal predictors, classification pipelines, explainability backends, plot kinds,
and the custom-component protocols.

## Links

- Repository: <https://github.com/vseguro/uxplain>
- Issues: <https://github.com/vseguro/uxplain/issues>
