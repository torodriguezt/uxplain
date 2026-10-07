# Getting started

`uxplain` explains scalar summaries of fitted conformal intervals and prediction
sets with SHAP, PDP/ICE or LIME. Python 3.10 or later is required.

## Installation

Install a published version with `python -m pip install uxplain`. To work with this
checkout instead, run `python -m pip install -e ".[dev,release]"` from the repository
root, preferably in a virtual environment created with `python -m venv .venv`.

## Regression

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

X, y = make_regression(n_samples=300, n_features=4, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipe = UncertaintyExplanationPipeline(
    model=RandomForestRegressor(n_estimators=50, random_state=42),
    conformal_method="normalized",
    confidence=0.9,
    uncertainty_metric="width",
    random_state=42,
)
pipe.fit(X_train, y_train)
result = pipe.explain(X_test[:5], show_plots=False)
print(result.interval_width)
```

The default regression method is `normalized`. A `standard` split-conformal
regressor has constant interval width at a fixed coverage level, so width
attributions are zero. CQR requires `conformal_method="cqr"` and two quantile
regressors supplied as `lower_model` and `upper_model`.

`confidence=0.9` is the requested nominal coverage, not a posterior probability
for an individual observation. Regression summaries are `width`, `lower`, `upper`
and `midpoint`. With very small calibration samples and high requested coverage,
bounds can be infinite; a finite-valued explanation then is not available. CQR
can also produce an empty interval (`lower > upper`). In that case `width`
retains the signed endpoint difference, which is negative, and `midpoint` is
only the arithmetic average of the endpoints. Clipping the difference at zero
would change the explained function and invalidate the affine TreeSHAP shortcut.

## Calibration

`fit(X_train, y_train)` reserves a calibration subset automatically. To use an
explicit, disjoint calibration sample:

```python
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42,
)
pipe.fit(X_fit, y_fit, X_calib=X_calib, y_calib=y_calib)
```

Keep final test data separate from training and calibration. Explanations describe
the fitted predictor's summary; they do not establish causal effects or add
conditional coverage guarantees. Exchangeability and the assumptions of the
selected conformal method still matter.

## Classification

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=300, n_features=4, n_informative=3,
    n_redundant=0, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = UncertaintyExplanationPipeline(
    model=RandomForestClassifier(n_estimators=50, random_state=42),
    uncertainty_metric="set_size",
    random_state=42,
)
pipe.fit(X_train, y_train)
result = pipe.explain(X_test[:5], show_plots=False)
print(result.classes)
print(result.prediction_set)
```

Each column of the Boolean prediction-set matrix corresponds to the label at the
same position in `result.classes`. Classification defaults to `standard`;
`class_cond` calibrates by class, and `mondrian` by a fitted partition. Available
summaries are `set_size`, `credibility` (maximum conformal p-value) and `confidence`
(one minus the second-largest p-value). The last summary is distinct from the
pipeline's nominal-coverage argument of the same name.

## Explainers and data

Choose `xai_method="shap"`, `"pdp"` or `"lime"` when constructing the pipeline.
`result.explanation_values` contains the corresponding explanation object. SHAP's
fast path applies to supported affine summaries of tree models; unsupported cases
use the generic explainer. Set `fast_shap=False` to request the generic path.

See the [example notebooks](https://github.com/torodriguezt/uxplain/tree/main/notebooks)
for plotting, PDP interactions, LIME and custom predictors. The
`uxplain.datasets` module provides `fetch_pnadc` and `load_geih`; microdata are not
bundled with the package. PNAD requires a download or an existing cache, while
GEIH requires a local file obtained from DANE.

## Releasing

From a clean checkout with the intended version and changelog:

```bash
python -m pip install -e ".[dev,release]"
python -m ruff check .
python -m pytest
python -m build
python -m twine check --strict dist/*
```

The build creates a source archive and builds its wheel from that archive. The
source archive includes the test suite. CI additionally installs the wheel and
runs those tests outside the checkout on Linux and Windows with Python 3.13.
The source tests cover every combination of Linux/Windows and Python 3.10–3.13.

Before publishing, confirm that CI passes, that the wheel contains the intended
version and license, and that the release version is available to your PyPI
project. Install the exact wheel into a fresh virtual environment, run
`python -m pip check`, and run the examples above from outside the source tree.
Keep only the intended release archives in the upload directory. Publishing is
a separate maintainer action; these checks do not upload anything.

For JSS, archive the manuscript, references, figures, replication scripts and a
fixed environment separately, tied to the exact package release. The local
manuscript and replication files are currently ignored by this repository, so a
package source archive alone is not the article's replication bundle.

Packaging references: [PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [Twine validation](https://twine.readthedocs.io/en/stable/).
