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
print(pipe.summary())
print(pipe.evaluate(X_test, y_test))

result = pipe.explain(X_test[:5], show_plots=False)
print(result.to_frame())
print(result.feature_importance())
```

`summary()` returns a pandas Series describing the configuration and fitted
state, including the training and calibration sample sizes. `evaluate()` returns
a DataFrame with empirical coverage, mean and median size, and the share of empty
predictions on held-out labelled data. Size means signed interval width for
regression and number of included labels for classification. Pass `groups=` to
add a row for each group; the overall row is always included.

`result.to_frame()` contains the predictions for the explained rows, while
`result.explanation_values.to_frame()` contains their explanation values.
`result.feature_importance()` ranks the features by their contribution to the
chosen uncertainty summary. With SHAP or LIME it uses mean absolute attribution
over the explained rows; with PDP it uses the standard deviation of the averaged
curve. These rankings describe the fitted summary and the selected observations.

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

`fit(X_train, y_train)` reserves a calibration subset using a random split
independent of features and labels, for both regression and classification.
The split is not stratified and is not retried to force class representation.
Set `random_state` in advance for reproducibility. To use an explicit, disjoint
calibration sample:

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
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from uxplain import UncertaintyExplanationPipeline

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
print(pipe.summary())
print(pipe.evaluate(X_test, y_test, groups=y_test))
print(pd.DataFrame(pipe.predict_p(X_test[:5]), columns=pipe.classes_))

result = pipe.explain(X_test[:5], show_plots=False)
print(result.classes)
print(result.prediction_set)
print(result.to_frame())
print(result.feature_importance())
```

Each column of the Boolean prediction-set matrix corresponds to the label at the
same position in `result.classes` and `pipe.classes_`. `pipe.predict(X)` returns
that matrix without computing explanations. `pipe.predict_p(X)` returns the
conformal p-value for every candidate class in the same column order. Conformal
p-values are not class probabilities and need not sum to one. For the built-in
classifier, a label belongs to the set when its p-value is at least
`1 - pipe.confidence`. `predict_p()` is only defined for classification.

Classification defaults to `standard`;
`class_cond` calibrates by class, and `mondrian` by a fitted partition. Available
summaries are `set_size`, `credibility` (maximum conformal p-value) and `confidence`
(one minus the second-largest p-value). The last summary is distinct from the
pipeline's nominal-coverage argument of the same name.

Classification explanations use non-smoothed p-values (`smoothing=False`, the
default of `CrepesConformalClassifier`). All three built-in explainers reject
`smoothing=True`. A fixed random seed can reproduce a batch of smoothed p-values,
but does not make the output a deterministic function of each observation:
changing its position or batch size can change its tie-breaking draw. Smoothed
prediction remains available directly through the conformal predictor.
[crepes API documentation](https://crepes.readthedocs.io/en/stable/crepes.html).
Custom predictors must also return finite, deterministic row-wise summaries.

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

See [Extending](extending.md) for the backend protocols and a runnable MAPIE
adapter. The [API reference](api.md) documents constructors, methods, results,
evaluation helpers, plotting functions and dataset loaders.

## Statistical scope and limitations

The standard split-conformal argument fixes the training data, fitted score
function and any model-fitting randomness. Calibration scores and the score of
a future observation must then be exchangeable. Reserving calibration indices
independently of observed features and labels preserves that argument for
exchangeable observations. Our previous automatic classification split used
label stratification. Because that selection depends on responses, the usual
argument cannot simply be assumed to apply to it; the default now uses an
unstratified split. This is an assessment of the theorem's applicability, not a
claim that every stratified design must undercover.

For standard classification, with calibration scores `s_1, ..., s_n`, the
non-smoothed p-value for candidate label `y` is
`p_y(x) = (1 + sum(s_i >= s(x, y))) / (n + 1)`.
Exchangeability makes the true-label p-value super-uniform, so
`P(p_Y(X) <= alpha) <= alpha`. crepes includes labels with
`p_y(x) >= alpha`, where `alpha = 1 - confidence`; its inclusive boundary is
also conservative. Thus coverage is at least the requested level under the
assumptions, averaged over calibration and test observations. It need not equal
that level, especially with ties, and is not a guarantee conditional on every
feature vector or every realized calibration sample. See
[Angelopoulos and Bates, Sections 1 and 4 and Appendix D](https://arxiv.org/html/2107.07511v6)
for the rank argument and conditional extensions.

`class_cond` applies calibration within each candidate class; `mondrian` groups
by the fitted model's predicted class. Their respective group guarantees require
exchangeability within those groups and a grouping rule fixed before calibration.
They are not the same as stratifying the train/calibration split. A represented
model class with no class-conditional calibration observations receives the
conservative non-smoothed p-value 1. All possible target labels must be supported
by the fitted classifier: a label absent from `classes_` cannot appear in a
prediction set. Unknown calibration labels raise an error. Do not repeatedly
split until desired class counts or coverage appear; use an adequate independent
training sample and a model supporting the intended label space.

Explicit calibration data remain the caller's responsibility. Keep them separate
from model selection and training, and do not tune the model, partition or seed
against calibration/test results. A random split alone cannot make clustered,
temporal or shifted observations exchangeable. Software tests check that the
implementation follows these choices; empirical coverage checks do not prove
the assumptions or a universal coverage guarantee.

For CQR, negative `width` means crossed endpoints and an empty prediction set;
it is not a negative set length or an estimate of negative noise. `midpoint` then
remains an arithmetic summary only. Explanations retain this signed target and
do not clip or reorder the bounds.

The PNAD Continua and GEIH loader unit tests use synthetic local extracts.
For empirical applications, record the source and survey period, verify the
variable dictionary, row counts, person-level joins, missing-value handling and
target coding, and archive those checks with the replication results. These examples
do not implement survey weights or cluster-aware calibration and cannot be
presented as design-based population coverage guarantees.

## Building the documentation

From the repository root, install the documentation extra and build with warnings
treated as errors:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict -f docs/mkdocs.yml
```

To preview edits, run `python -m mkdocs serve -f docs/mkdocs.yml` and open
<http://127.0.0.1:8000/uxplain/>. The strict build checks rendering and configuration;
Python examples must also be executed to verify that they work.

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
