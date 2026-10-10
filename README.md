# uxplain

[![PyPI version](https://img.shields.io/pypi/v/uxplain.svg)](https://pypi.org/project/uxplain/)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://github.com/torodriguezt/uxplain/blob/main/pyproject.toml)
[![CI](https://github.com/torodriguezt/uxplain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/torodriguezt/uxplain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/torodriguezt/uxplain/blob/main/LICENSE)

`uxplain` explains which features make a model more or less uncertain. It wraps
scikit-learn compatible models with conformal prediction and uses SHAP, PDP/ICE
or LIME to explain summaries of their prediction intervals or sets.

## Installation

Requires Python 3.10 or later.

```bash
pip install uxplain
```

## Regression

Explain prediction interval width with SHAP. `fit()` automatically reserves a
separate calibration sample.

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

X, y = make_regression(n_samples=300, n_features=4, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipe = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(n_estimators=50, random_state=42),
    confidence=0.9,
    xai_method="shap",
    uncertainty_metric="width",
    random_state=42,
)
pipe.fit(X_train, y_train)
result = pipe.explain(X_test[:5], show_plots=False)

print(result.interval_width)
print(result.explanation_values)
```

## Classification

Use a classifier and `set_size` to explain how many labels enter the prediction set.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipe = UncertaintyExplanationPipeline(
    model=RandomForestClassifier(n_estimators=50, random_state=42),
    uncertainty_metric="set_size",
    random_state=42,
)
pipe.fit(X_train, y_train)
result = pipe.explain(X_test[:5], show_plots=False)

print(result.prediction_set)
print(result.explanation_values)
```

## Learn more

The [getting-started guide](https://github.com/torodriguezt/uxplain/blob/main/docs/docs/getting-started.md)
covers calibration, other uncertainty summaries and explainer options. See the
[notebooks](https://github.com/torodriguezt/uxplain/tree/main/notebooks) for more examples
and the [release guide](https://github.com/torodriguezt/uxplain/blob/main/docs/docs/getting-started.md#releasing)
for development and publishing.

Coverage guarantees depend on the conformal method's
[statistical assumptions](https://github.com/torodriguezt/uxplain/blob/main/docs/docs/getting-started.md#statistical-scope-and-limitations);
they do not extend to the explanations themselves.

## Authors

- **Tomas Rodriguez Taborda** — Universidad Nacional de Colombia, Medellín
- **Veronica Seguro Varela** — Universidad Nacional de Colombia, Medellín
- **Rafael Izbicki** — Federal University of São Carlos
- **Johnatan Cardona Jimenez** — Universidad Nacional de Colombia, Medellín

## License

[MIT](https://github.com/torodriguezt/uxplain/blob/main/LICENSE).
