# Changelog

All notable changes to **uxplain** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

### Added
- PNAD Continua and GEIH dataset loaders, without bundling survey microdata.
- Exact TreeSHAP for supported affine summaries of built-in conformal predictors,
  with generic SHAP as the fallback.
- Python 3.13 CI coverage and Linux/Windows distribution checks that install the
  wheel and run the tests included in the source archive outside the checkout.
- Runnable getting-started examples and release-validation instructions.

### Fixed
- Validate conformal methods, task names, coverage levels and calibration arrays;
  accept single-column targets without broadcasting CQR calibration scores.
- Return unbounded regression intervals for Mondrian groups absent from calibration.
- Keep Mondrian assignments deterministic when scores tie and seed permutation
  SHAP without changing the caller's NumPy random state.
- Restrict the TreeSHAP shortcut to supported algebraic decompositions instead of
  inferring global affinity from agreement on a finite sample.
- Prevent target leakage for alternative dataset targets and reject ambiguous
  joins of GEIH modules; preserve missing occupation and informality values.
- Preserve DataFrame feature order and reset pipeline state on refitting.
- Reject non-finite explanation targets explicitly, while allowing unbounded
  conformal prediction intervals.
- Plot single-observation SHAP slices and PDP interactions with constant features;
  require an explicit baseline for SHAP waterfall plots.

### Changed
- Use SPDX license metadata and include tests in the source distribution.
- The manuscript's replication results must be regenerated with this version
  before submission; this entry describes a local release candidate.

## [0.2.3]

### Added
- **PDP-based global feature importance.** `PDPExplanation` now carries an
  `importance` field — the standard deviation of each feature's averaged
  partial-dependence curve (Greenwell et al., 2018). Render it with the new
  `plot_kind="importance"` for the PDP backend.
- Runnable demo notebooks (`notebooks/01_regression.ipynb`,
  `notebooks/02_classification.ipynb`) covering conformal outputs, all SHAP
  plot kinds via `generate_shap_plots`, PDP / ICE / PDP+ICE / 2D PDP /
  importance, CQR with quantile regressors, and the classification metrics.

### Fixed
- README: the PDP importance plot kind is documented as `"importance"`
  (was inconsistently shown as `"bar"`, which collided with the SHAP bar plot),
  and the PDP default plot kinds now reflect the actual default (`["pdp"]`).

## [0.2.0]

### Added
- Classification support: prediction sets, conformal classifiers
  (`CrepesConformalClassifier`), and classification uncertainty metrics
  (`set_size`, `credibility`, `confidence`).
- Package renamed to `uxplain` and prepared for PyPI.

## [0.1.0]

### Added
- Initial release: regression conformal predictors (`CrepesConformalPredictor`,
  `CQRConformalPredictor`) with SHAP, PDP, and LIME explainability of
  interval-derived uncertainty metrics.
