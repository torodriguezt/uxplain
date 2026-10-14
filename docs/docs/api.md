# API reference

This reference is generated from the docstrings of the checked-out `uxplain`
package. Start with [Getting started](getting-started.md) for complete examples,
or [Extending](extending.md) to connect a custom backend.

Most objects below are available directly from `uxplain`. Dataset helpers live
in `uxplain.datasets`; plotting presets are available from `uxplain.plotting`.

## Pipeline

The pipeline fits and calibrates a predictor, produces predictions, evaluates
empirical coverage, and explains a scalar summary of uncertainty.

::: uxplain.UncertaintyExplanationPipeline

## Results and explanations

`explain()` returns an `ExplanationResult` for regression or a
`ClassificationExplanationResult` for classification. Both expose
`to_frame()`, `feature_importance()` and `plot()`. Their `explanation_values`
attribute holds a SHAP, LIME or partial-dependence explanation.

::: uxplain.ExplanationResult
    options:
      inherited_members: [feature_importance, plot]

::: uxplain.ClassificationExplanationResult
    options:
      inherited_members: [feature_importance, plot]

::: uxplain.UncertaintyExplanation

::: uxplain.ShapExplanation
    options:
      inherited_members: [feature_importance]

::: uxplain.LIMEExplanation
    options:
      inherited_members: [feature_importance]

::: uxplain.PDPExplanation
    options:
      inherited_members: [feature_importance]

## Conformal predictors

Each predictor separates model training from calibration. The regression
predictors return lower and upper bounds; the classifier returns prediction
sets and conformal p-values.

::: uxplain.CrepesConformalPredictor

::: uxplain.CQRConformalPredictor

::: uxplain.CrepesConformalClassifier

### Methods and uncertainty metrics

These type aliases enumerate the supported method and target names. A metric
selects the scalar explained by SHAP, PDP or LIME. Classification's
`"confidence"` metric is distinct from the nominal coverage argument.

::: uxplain.ConformalMethod

::: uxplain.ClassificationConformalMethod

::: uxplain.RegressionMetric

::: uxplain.ClassificationMetric

::: uxplain.UncertaintyMetric

## Explainers

The pipeline constructs these explainers automatically. Use them directly when
working with an already fitted conformal predictor.

::: uxplain.ShapUncertaintyExplainer

::: uxplain.PDPUncertaintyExplainer

::: uxplain.LimeUncertaintyExplainer

## Evaluation

`pipeline.evaluate()` wraps `coverage_table()`. The result and explanation
objects' `feature_importance()` methods wrap `importance_table()`.

::: uxplain.coverage_table

::: uxplain.importance_table

## Plots

These plotting functions accept an optional Matplotlib `ax` and return the
axes they draw on. Use an explanation's `plot()` method for a dictionary of
standard figures.

::: uxplain.shap_bar

::: uxplain.shap_beeswarm

::: uxplain.shap_waterfall

::: uxplain.pdp_curve

::: uxplain.ice_curves

::: uxplain.pdp_with_ice

::: uxplain.pdp_interaction

::: uxplain.pdp_importance

::: uxplain.lime_local

::: uxplain.importance_bar

::: uxplain.STYLE

::: uxplain.plotting.use_theme

## Datasets

The loaders obtain data from the survey producer or a local file; microdata are
not bundled with the package. Their returned objects include source provenance.
See each loader for input requirements, coding and sampling options.

::: uxplain.datasets.fetch_pnadc

::: uxplain.datasets.load_geih

::: uxplain.datasets.default_cache_dir

### Dataset metadata

::: uxplain.datasets.PNADC_CITATION

::: uxplain.datasets.GEIH_CITATION

::: uxplain.datasets.PNADC_FEATURES

::: uxplain.datasets.GEIH_FEATURES

::: uxplain.datasets.PNADC_VARIABLES

::: uxplain.datasets.GEIH_VARIABLES

::: uxplain.datasets.MAJOR_OCCUPATION_GROUPS

## Protocols

These structural interfaces describe custom predictors and explainers.
Inheritance from the protocols is optional; implementing their methods is
sufficient. See [Extending](extending.md) for a complete MAPIE adapter.

::: uxplain.ConformalPredictorProtocol

::: uxplain.ConformalClassifierProtocol

::: uxplain.UncertaintyExplainerProtocol
