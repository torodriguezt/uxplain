# Changelog

All notable changes to **uxplain** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/),
and the project adheres to [Semantic Versioning](https://semver.org/).

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
