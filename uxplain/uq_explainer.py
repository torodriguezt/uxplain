"""
Main estimator for conformal prediction and explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d, validate_data

from .conformal._validation import check_confidence
from .conformal.cqr_predictor import CQRConformalPredictor
from .conformal.crepes_classifier import (
    ClassificationConformalMethod,
    CrepesConformalClassifier,
)
from .conformal.crepes_predictor import (
    ConformalMethod,
    CrepesConformalPredictor,
)
from .evaluation import coverage_table
from .explainability.base import UncertaintyExplanation
from .explainability.lime_explainer import LimeUncertaintyExplainer
from .explainability.pdp_explainer import PDPUncertaintyExplainer
from .explainability.shap_explainer import ShapUncertaintyExplainer
from .plots import (
    generate_lime_plots,
    generate_pdp_plots,
    generate_shap_plots,
)
from .protocols import (
    ConformalClassifierProtocol,
    ConformalPredictorProtocol,
    UncertaintyExplainerProtocol,
)
from .uncertainty.metrics import (
    UncertaintyMetric,
    is_classifier_predictor,
    metric_label,
)

XAIMethod = Literal["shap", "pdp", "lime"]
TaskKind = Literal["auto", "regression", "classification"]

VALID_PLOT_KINDS = ("beeswarm", "bar", "waterfall", "summary")
VALID_PDP_PLOT_KINDS = ("pdp", "ice", "pdp_ice", "pdp_2d", "importance")
VALID_LIME_PLOT_KINDS = ("local",)

REGRESSION_METRICS = ("width", "lower", "upper", "midpoint")
CLASSIFICATION_METRICS = ("set_size", "credibility", "confidence")
REGRESSION_METHODS = ("standard", "normalized", "mondrian", "normalized_mondrian", "cqr")
CLASSIFICATION_METHODS = ("standard", "class_cond", "mondrian")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class _ResultMethods:
    """Accessors shared by the regression and classification results."""

    explanation_values: UncertaintyExplanation

    def feature_importance(self, n_boot: int = 0, level: float = 0.95,
                           random_state: int | None = None) -> pd.DataFrame:
        """Global feature importance of the explained summary.

        Parameters
        ----------
        n_boot : int, default=0
            Number of bootstrap resamples of explained rows. Partial dependence
            requires ICE curves from ``explain(..., kind="both")``. Intervals
            describe variability over these rows only, excluding estimator refitting
            or redrawing calibration data.
        level : float, default=0.95
            Coverage of bootstrap percentile intervals, between zero and one.
        random_state : int, optional
            Seed for bootstrap sampling.

        Returns
        -------
        pandas.DataFrame
            Features sorted by decreasing ``importance``, with ``importance``
            and ``rank`` columns. When ``n_boot > 0``, also includes
            ``ci_lower``, ``ci_upper``, ``rank_lower`` and ``rank_upper``.
        """
        return self.explanation_values.feature_importance(
            n_boot=n_boot, level=level, random_state=random_state,
        )

    def plot(self, kind=None, show: bool = True, **kwargs) -> dict:
        """Draw the explanation's standard figures.

        Parameters
        ----------
        kind : str or list of str, optional
            Plot kinds supported by ``explanation_values``. ``None`` selects
            the explanation method's default figures.
        show : bool, default=True
            Display the figures. Set to ``False`` to save or style them first.
        **kwargs : dict
            Options forwarded to the explanation's figure builder, such as
            ``figsize`` or the sample index of a local plot.

        Returns
        -------
        dict
            Mapping from plot kind to ``matplotlib.figure.Figure``.
        """
        return self.explanation_values.plot(kind=kind, show=show, **kwargs)


@dataclass(repr=False)
class ExplanationResult(_ResultMethods):
    """Result of `ConformalExplainer.explain` for regression.

    Attributes
    ----------
    lower : np.ndarray of shape (n_samples,)
        Lower prediction bounds.
    upper : np.ndarray of shape (n_samples,)
        Upper prediction bounds.
    interval_width : np.ndarray of shape (n_samples,)
        Signed endpoint span, ``upper - lower``. Negative for empty CQR sets.
    explanation_values : UncertaintyExplanation
        The explanation of the chosen summary: a `ShapExplanation`,
        `LIMEExplanation` or `PDPExplanation`.
    """

    lower: np.ndarray
    upper: np.ndarray
    interval_width: np.ndarray
    explanation_values: UncertaintyExplanation

    def to_frame(self) -> pd.DataFrame:
        """Return interval predictions for the explained observations.

        Returns
        -------
        pandas.DataFrame
            One row per observation, with ``lower``, ``upper`` and ``width``
            columns. Width is the signed difference ``upper - lower``.
        """
        return pd.DataFrame({"lower": self.lower, "upper": self.upper,
                             "width": self.interval_width})

    def __repr__(self) -> str:
        return (f"ExplanationResult(rows={len(self.lower)}, "
                f"mean_width={float(np.mean(self.interval_width)):.4g}, "
                f"explanation={self.explanation_values!r})")


@dataclass(repr=False)
class ClassificationExplanationResult(_ResultMethods):
    """Result of `ConformalExplainer.explain` for classification.

    Attributes
    ----------
    prediction_set : np.ndarray of shape (n_samples, n_classes), dtype=bool
        Conformal prediction sets — True at ``[i, k]`` means class ``k``
        is included in the set for sample ``i``.
    p_values : np.ndarray of shape (n_samples, n_classes)
        Conformal p-values per class.
    set_size : np.ndarray of shape (n_samples,)
        Size of the prediction set for each sample.
    classes : np.ndarray of shape (n_classes,)
        Class labels in the order used by ``prediction_set`` and ``p_values``.
    explanation_values : UncertaintyExplanation
        The explanation of the chosen summary: a `ShapExplanation`,
        `LIMEExplanation` or `PDPExplanation`.
    """

    prediction_set: np.ndarray
    p_values: np.ndarray
    set_size: np.ndarray
    classes: np.ndarray
    explanation_values: UncertaintyExplanation

    def to_frame(self) -> pd.DataFrame:
        """One row per explained observation: the prediction set and its summaries.

        Returns
        -------
        pandas.DataFrame
            Columns are ``prediction_set`` (a tuple of labels), ``set_size``,
            ``credibility`` (largest p-value) and ``confidence`` (one minus the
            second-largest p-value; not the nominal coverage level).
        """
        p = np.sort(np.asarray(self.p_values, dtype=float), axis=1)
        labels = np.asarray(self.classes)
        return pd.DataFrame({
            "prediction_set": [tuple(labels[row]) for row in np.asarray(self.prediction_set)],
            "set_size": np.asarray(self.set_size),
            "credibility": p[:, -1],
            "confidence": 1.0 - p[:, -2] if p.shape[1] > 1 else np.nan,
        })

    def __repr__(self) -> str:
        return (f"ClassificationExplanationResult(rows={len(self.set_size)}, "
                f"classes={len(self.classes)}, "
                f"mean_set_size={float(np.mean(self.set_size)):.4g}, "
                f"explanation={self.explanation_values!r})")


def _detect_task(
    estimator,
    conformal_predictor,
    task: TaskKind,
) -> Literal["regression", "classification"]:
    """Resolve ``task="auto"`` from the estimator / predictor."""

    if task not in ("auto", "regression", "classification"):
        raise ValueError(
            f"Unknown task '{task}'. Choose from 'auto', 'regression', or 'classification'."
        )
    if task != "auto":
        return task
    if conformal_predictor is not None:
        return (
            "classification" if is_classifier_predictor(conformal_predictor)
            else "regression"
        )
    if estimator is not None and is_classifier(estimator):
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# Conformal explainer
# ---------------------------------------------------------------------------

class ConformalExplainer(BaseEstimator):
    """
    Estimator integrating conformal prediction and summary explanation.

    The conformal explainer follows the scikit-learn estimator conventions: the
    constructor stores its parameters unchanged, ``fit`` builds and fits
    the conformal predictor, and ``explain`` prepares the explainer on its
    first call. Fitted state ends in an
    underscore (``conformal_predictor_``, ``explainer_``, ``task_``, ...).
    ``get_params``, ``set_params`` and `sklearn.base.clone` work as for
    any estimator; user-supplied components are cloned, never fitted in place.

    Supports both regression and classification (selected via ``task``).

    Three XAI methods are available, selectable via ``xai_method``:

    - ``"shap"`` — SHAP-based explanation (default)
    - ``"pdp"``  — Partial Dependence Plot explanation
    - ``"lime"`` — LIME-based explanation

    Examples
    --------
    Regression, with training and test arrays already prepared:

    ```python
    from sklearn.ensemble import RandomForestRegressor
    from uxplain import ConformalExplainer

    pipeline = ConformalExplainer(estimator=RandomForestRegressor())
    pipeline.fit(X_train, y_train)
    result = pipeline.explain(X_test)
    print(pipeline.summary())
    print(result.feature_importance())
    ```

    Classification, with training labels and test features already prepared:

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from uxplain import ConformalExplainer

    pipeline = ConformalExplainer(
        estimator=RandomForestClassifier(),
        task="classification",
        uncertainty_metric="set_size",
    )
    pipeline.fit(X_train, y_train)
    result = pipeline.explain(X_test)
    print(result.prediction_set)   # (n_samples, n_classes) bool
    print(result.set_size)         # (n_samples,)
    ```
    """

    def __init__(
        self,
        estimator=None,
        *,
        calibration_size: float = 0.2,
        confidence: float = 0.9,
        task: TaskKind = "auto",
        conformal_method: (
            ConformalMethod | ClassificationConformalMethod | Literal["cqr"]
            | None
        ) = None,
        xai_method: XAIMethod = "shap",
        uncertainty_metric: UncertaintyMetric | None = None,
        n_lime_samples: int = 5000,
        random_state: int | None = None,
        lower_estimator=None,
        upper_estimator=None,
        conformal_predictor: (
            ConformalPredictorProtocol | ConformalClassifierProtocol | None
        ) = None,
        explainer: UncertaintyExplainerProtocol | None = None,
        fast_path: bool = True,
    ):
        """
        Initialize the conformal explainer.

        Parameters
        ----------
        estimator : estimator, optional
            sklearn-compatible regressor or classifier. Required for crepes
            methods. Ignored when ``conformal_method="cqr"`` or when a custom
            ``conformal_predictor`` is provided.

        calibration_size : float, default=0.2
            Fraction reserved for calibration when no calibration data are supplied.

        confidence : float
            Nominal coverage level of the conformal predictions, e.g. ``0.9``.

        task : {"auto", "regression", "classification"}
            Task type. ``"auto"`` infers from the estimator / conformal_predictor.

        conformal_method : str, optional
            Conformal method.

            - Regression: ``"standard"``, ``"normalized"`` (default),
              ``"mondrian"``, ``"normalized_mondrian"`` (crepes), or ``"cqr"``.
            - Classification: ``"standard"`` (default), ``"class_cond"``,
              ``"mondrian"``.

            Ignored if ``conformal_predictor`` is provided.

        xai_method : {"shap", "pdp", "lime"}
            Explainability method to use. Ignored if ``explainer`` is provided.

        uncertainty_metric : str, optional
            Which scalar to explain. Defaults depend on ``task``:

            - Regression: ``"width"`` (interval width).
              Other options: ``"lower"``, ``"upper"``, ``"midpoint"``.
            - Classification: ``"set_size"`` (size of prediction set).
              Other options: ``"credibility"``, ``"confidence"``.

            Ignored if ``explainer`` is provided.

        n_lime_samples : int
            Number of LIME perturbations per sample. Ignored when
            ``xai_method != "lime"``.

        random_state : int, optional
            Seed for the auto calibration split, SHAP background rows and
            permutation sampling, and the LIME perturbation sampler.

        lower_estimator, upper_estimator : estimator, optional
            Quantile regressors for CQR. Required when
            ``conformal_method="cqr"`` (regression only).

        conformal_predictor : ConformalPredictorProtocol or ConformalClassifierProtocol, optional
            Custom conformal predictor (regressor or classifier) implementing
            `ConformalPredictorProtocol` or
            `ConformalClassifierProtocol`. Overrides
            ``conformal_method``. It is cloned at fit time.

        explainer : UncertaintyExplainerProtocol, optional
            Custom explainer instance. Overrides ``xai_method``. It is cloned
            at fit time; if it has a ``conformal_predictor`` attribute, the clone is bound to
            the conformal explainer's fitted conformal predictor.

        fast_path : bool
            When ``xai_method="shap"``, use exact TreeSHAP on the component
            models for metrics that are affine in them (e.g. CQR interval
            width, which is the difference of the two quantile models). Falls
            back to the generic explainer whenever that does not apply. Set to
            ``False`` to always use the generic path.
        """

        self.estimator = estimator
        self.calibration_size = calibration_size
        self.confidence = confidence
        self.task = task
        self.conformal_method = conformal_method
        self.xai_method = xai_method
        self.uncertainty_metric = uncertainty_metric
        self.n_lime_samples = n_lime_samples
        self.random_state = random_state
        self.lower_estimator = lower_estimator
        self.upper_estimator = upper_estimator
        self.conformal_predictor = conformal_predictor
        self.explainer = explainer
        self.fast_path = fast_path

    # -- configuration --------------------------------------------------------

    def _resolve(self) -> tuple[str, str, str]:
        """Validate the parameters and return ``(task, metric, method)``."""

        check_confidence(self.confidence)
        task = _detect_task(self.estimator, self.conformal_predictor, self.task)
        metric = self.uncertainty_metric
        if metric is None:
            metric = "set_size" if task == "classification" else "width"
        self._validate_metric(metric, task)
        method = self.conformal_method
        if method is None:
            method = "standard" if task == "classification" else "normalized"

        if self.conformal_predictor is None:
            if task == "classification":
                if self.estimator is None:
                    raise ValueError(
                        "task='classification' requires a 'estimator' (sklearn-"
                        "compatible classifier with predict_proba) or a custom "
                        "'conformal_predictor'."
                    )
                if method not in CLASSIFICATION_METHODS:
                    raise ValueError(
                        f"conformal_method='{method}' is not valid for "
                        "classification. Choose from 'standard', 'class_cond', "
                        "'mondrian'."
                    )
            elif method == "cqr":
                if self.lower_estimator is None or self.upper_estimator is None:
                    raise ValueError(
                        "conformal_method='cqr' requires both 'lower_estimator' "
                        "and 'upper_estimator'."
                    )
            elif self.estimator is None:
                raise ValueError(
                    "Either 'estimator' or 'conformal_predictor' must be provided, "
                    "or set conformal_method='cqr' with 'lower_estimator' and "
                    "'upper_estimator'."
                )
            elif method not in REGRESSION_METHODS:
                raise ValueError(f"Unknown conformal regression method '{method}'.")

        if self.explainer is None and self.xai_method not in ("shap", "pdp", "lime"):
            raise ValueError(
                f"Unknown xai_method '{self.xai_method}'. "
                "Choose from 'shap', 'pdp', or 'lime'."
            )
        return task, metric, method

    def _build(self, task: str, metric: str, method: str):
        """Fresh, unfitted conformal predictor and explainer for one fit."""

        if self.conformal_predictor is not None:
            cp = clone(self.conformal_predictor, safe=False)
        elif task == "classification":
            cp = CrepesConformalClassifier(
                clone(self.estimator, safe=False), method=method,
                random_state=self.random_state,
            )
        elif method == "cqr":
            cp = CQRConformalPredictor(
                clone(self.lower_estimator, safe=False), clone(self.upper_estimator, safe=False),
                estimator=clone(self.estimator, safe=False),
            )
        else:
            cp = CrepesConformalPredictor(clone(self.estimator, safe=False), method=method)

        if self.explainer is not None:
            explainer = clone(self.explainer, safe=False)
            if hasattr(explainer, "conformal_predictor"):
                explainer.conformal_predictor = cp
        elif self.xai_method == "pdp":
            explainer = PDPUncertaintyExplainer(
                conformal_predictor=cp, confidence=self.confidence, uncertainty_metric=metric,
            )
        elif self.xai_method == "shap":
            explainer = ShapUncertaintyExplainer(
                conformal_predictor=cp, confidence=self.confidence, uncertainty_metric=metric,
                fast_path=self.fast_path, random_state=self.random_state,
            )
        else:
            explainer = LimeUncertaintyExplainer(
                conformal_predictor=cp, confidence=self.confidence,
                n_lime_samples=self.n_lime_samples,
                random_state=self.random_state, uncertainty_metric=metric,
            )
        return cp, explainer

    @staticmethod
    def _method_label(cp, method: str) -> str:
        if isinstance(cp, CQRConformalPredictor):
            return "cqr"
        if isinstance(cp, (CrepesConformalPredictor, CrepesConformalClassifier)):
            return cp.method
        return "custom"

    # -- fitting and prediction ----------------------------------------------

    def fit(self, X, y, *, X_calib=None, y_calib=None, X_background=None):
        """Fit cloned components and return self.

        X and y contain the training observations. Supply X_calib and y_calib
        together for external calibration; otherwise calibration_size controls
        a label-independent split. X_background sets the explanation reference
        sample (calibration observations by default). Model preprocessing is
        fitted on the proper training subset only.
        """
        self._is_fitted = False
        task, metric, method = self._resolve()
        if not 0 < self.calibration_size < 1:
            raise ValueError("calibration_size must be strictly between 0 and 1.")
        checked_X = check_array(X, dtype=None)
        if not hasattr(X, "columns"):
            X = checked_X
        y = column_or_1d(y, warn=True)
        validate_data(self, X, y, reset=True, skip_check_array=True)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        cp, explainer = self._build(task, metric, method)
        self._feature_names = list(X.columns) if hasattr(X, "columns") else None
        if self._feature_names is not None and hasattr(explainer, "feature_names"):
            explainer.feature_names = self._feature_names
        if (X_calib is None) != (y_calib is None):
            raise ValueError("Pass both X_calib and y_calib, or neither.")
        if X_calib is None:
            X, X_calib, y, y_calib = train_test_split(
                X, y, test_size=self.calibration_size, random_state=self.random_state,
            )
        self._check_X(X_calib)
        if X_background is not None:
            self._check_X(X_background)
        cp.fit(X, y, X_calib, y_calib)
        self.conformal_predictor_ = cp
        self.explainer_ = explainer
        self.task_ = task
        self.uncertainty_metric_ = getattr(explainer, "uncertainty_metric", metric)
        self.conformal_method_ = self._method_label(cp, method)
        self.n_train_ = len(X)
        self.n_calibration_ = len(X_calib)
        self._X_background = X_calib if X_background is None else X_background
        self._is_fitted = True
        self._explainer_kwargs = None
        self._explainer_base_params = None
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        task = _detect_task(self.estimator, self.conformal_predictor, self.task)
        tags.estimator_type = "classifier" if task == "classification" else "regressor"
        tags.target_tags.required = True
        if task == "classification":
            tags.classifier_tags = ClassifierTags()
        else:
            tags.regressor_tags = RegressorTags()
        return tags

    def predict(self, X):
        """Return point values or class labels from the fitted base estimator.

        CQR needs an explicitly supplied point estimator; no midpoint is used
        as a substitute. Custom backends may provide intervals/sets only.
        """
        self._check_is_fitted()
        self._check_X(X)
        if not hasattr(self.conformal_predictor_, "predict"):
            raise AttributeError("This conformal predictor has no point estimator.")
        return self.conformal_predictor_.predict(X)

    def score(self, X, y, sample_weight=None):
        """Point-prediction R squared (regression) or accuracy (classification).

        Use evaluate for conformal coverage and size. Interval-only CQR/custom
        backends require an explicit conformal scorer in model selection.
        """
        prediction = self.predict(X)
        scorer = accuracy_score if self.task_ == "classification" else r2_score
        return scorer(y, prediction, sample_weight=sample_weight)

    @available_if(lambda self: _detect_task(self.estimator, self.conformal_predictor,
                                          self.task) == "classification")
    def predict_proba(self, X):
        """Return base-estimator class probabilities, distinct from p-values."""
        self._check_is_fitted()
        self._check_X(X)
        if self.task_ != "classification":
            raise AttributeError("predict_proba is only defined for classification.")
        return self.conformal_predictor_.predict_proba(X)

    @available_if(lambda self: _detect_task(self.estimator, self.conformal_predictor,
                                          self.task) == "classification")
    def predict_set(self, X, *, confidence=None):
        """Return Boolean class membership, with columns ordered as classes_."""
        self._check_is_fitted()
        self._check_X(X)
        if self.task_ != "classification":
            raise AttributeError("predict_set is only defined for classification.")
        conf = self.confidence if confidence is None else confidence
        self._check_confidence(conf)
        return self.conformal_predictor_.predict_set(X, confidence=conf)

    @available_if(lambda self: _detect_task(self.estimator, self.conformal_predictor,
                                          self.task) == "regression")
    def predict_interval(
        self,
        X,
        *,
        confidence: float | None = None,
    ):
        """
        Generate conformal predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features in the training column order. DataFrame column names must
            match the training columns when training used a DataFrame.
        confidence : float, optional
            Nominal coverage level, strictly between zero and one. Defaults to
            the conformal explainer's ``confidence``. Custom backends may restrict the
            supported coverage levels.

        Returns
        -------
        tuple of np.ndarray
            ``(lower, upper)``, each of shape ``(n_samples,)``.

        Raises
        ------
        NotFittedError
            If the conformal explainer has not been fitted.
        """

        self._check_is_fitted()
        self._check_X(X)

        conf = self.confidence if confidence is None else confidence
        self._check_confidence(conf)

        if self.task_ == "classification":
            raise AttributeError("predict_interval is only defined for regression.")
        return self.conformal_predictor_.predict_interval(X, confidence=conf)

    def predict_p_values(self, X) -> np.ndarray:
        """
        Conformal p-value of every candidate label (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features in the training column order.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Columns follow ``classes_``. A label enters the built-in predictor's set
            when its p-value is at least ``1 - confidence`` (the crepes
            backend includes equality as a conservative boundary convention).
            These p-values are conformity scores, not class probabilities.

        Raises
        ------
        AttributeError
            If this is a regression conformal explainer.
        NotFittedError
            If the conformal explainer has not been fitted.
        """

        self._check_is_fitted()
        if self.task_ != "classification":
            raise AttributeError("predict_p_values is only defined for classification.")
        self._check_X(X)
        return self.conformal_predictor_.predict_p_values(X)

    def evaluate(self, X, y, groups=None, confidence: float | None = None):
        """
        Empirical coverage and size of the conformal predictions on labelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Held-out features in the training column order.
        y : array-like of shape (n_samples,)
            Responses or class labels for the held-out observations.
        groups : array-like of shape (n_samples,), optional
            Group of each observation, e.g. ``y`` itself for per-class
            coverage. Adds one row per group.
        confidence : float, optional
            Nominal coverage level; defaults to the conformal explainer's ``confidence``.

        Returns
        -------
        pandas.DataFrame
            Rows ``"overall"`` and one per group; columns ``n``, ``coverage``,
            ``mean_size``, ``median_size`` and ``share_empty``. Size is the
            signed interval width (regression) or the prediction-set size
            (classification). Labels unknown to the fitted estimator count as
            uncovered. Floats display with four decimal places; calculations
            and exports retain full precision. Use ``to_string(float_format=...)``
            or ``to_html(float_format=...)`` to customize the display.
        """

        self._check_is_fitted()
        method = self.predict_set if self.task_ == "classification" else self.predict_interval
        prediction = method(X, confidence=confidence)
        return coverage_table(
            prediction, y,
            classes=self.classes_ if self.task_ == "classification" else None,
            groups=groups,
        )

    # -- fitted state -----------------------------------------------------------

    @property
    def classes_(self) -> np.ndarray:
        """Class labels, in the column order of prediction sets and p-values."""

        self._check_is_fitted()
        if self.task_ != "classification":
            raise AttributeError("classes_ is only defined for classification.")
        return np.asarray(self.conformal_predictor_.classes_)

    @property
    def cp(self):
        """Deprecated alias of ``conformal_predictor_``."""

        warnings.warn(
            "'cp' is deprecated and will be removed in a future release; "
            "use 'conformal_predictor_'.",
            FutureWarning,
            stacklevel=2,
        )
        self._check_is_fitted()
        return self.conformal_predictor_

    @property
    def used_fast_path_(self) -> bool | None:
        """Whether the last SHAP explanation took the exact TreeSHAP shortcut."""

        return getattr(getattr(self, "explainer_", None), "used_fast_path_", None)

    @property
    def algorithm_used_(self) -> str | None:
        """SHAP algorithm of the last explanation (``"tree"``, ``"exact"``, ...)."""

        return getattr(getattr(self, "explainer_", None), "algorithm_used_", None)

    def summary(self) -> pd.Series:
        """
        Configuration and fitted state of the conformal explainer, one entry per line.

        Returns
        -------
        pandas.Series
            Task, conformal predictor and method, nominal coverage, explained
            summary, explainer and, once fitted, the numbers of features,
            training, calibration and background rows (and classes).
        """

        if self.__sklearn_is_fitted__():
            task, metric = self.task_, self.uncertainty_metric_
            predictor = type(self.conformal_predictor_).__name__
            method = self.conformal_method_
            explainer = type(self.explainer_).__name__
        else:
            task, metric, method = self._resolve()
            predictor = (type(self.conformal_predictor).__name__
                         if self.conformal_predictor is not None else "built at fit")
            explainer = (type(self.explainer).__name__ if self.explainer is not None
                         else f"xai_method={self.xai_method!r}")

        entries = {
            "task": task,
            "conformal predictor": predictor,
            "conformal method": method,
            "nominal coverage": self.confidence,
            "explained summary": f"{metric} ({metric_label(metric)})",
            "explainer": explainer,
            "fitted": self.__sklearn_is_fitted__(),
        }
        if self.__sklearn_is_fitted__():
            entries["features"] = self.n_features_in_
            entries["training rows"] = self.n_train_
            entries["calibration rows"] = self.n_calibration_
            entries["background rows"] = len(self._X_background)
            if task == "classification":
                entries["classes"] = len(self.classes_)
            if self.algorithm_used_ is not None:
                entries["last SHAP algorithm"] = self.algorithm_used_
        return pd.Series(entries, dtype=object, name=type(self).__name__)

    # -- explanation -----------------------------------------------------------

    def explain(
        self,
        X,
        show_plots: bool = False,
        plot_kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
        figsize: tuple[float, float] | None = None,
        **explainer_kwargs,
    ):
        """
        Explain uncertainty for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations to explain, in the training column order.
            Partial dependence requires at least two observations.
        show_plots : bool, default=False
            Whether to display plots after computing the explanation.
        plot_kind : str or list of str, optional
            For SHAP: ``"beeswarm"``, ``"bar"``, ``"waterfall"``, ``"summary"``.
            For PDP:  ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"pdp_2d"``, ``"importance"``.
            For LIME: ``"local"``.
            Defaults to method-specific defaults when ``None``.
        waterfall_index : int, optional
            Sample index for SHAP waterfall plot or LIME local plot.
        figsize : tuple of (width, height), optional
            Figure size in inches forwarded to the plot function.
        **explainer_kwargs : dict
            Deprecated configuration overrides, validated by set_params.
            Configure a supplied explainer in its constructor instead.

        Returns
        -------
        ExplanationResult or ClassificationExplanationResult
            Conformal predictions at the conformal explainer's ``confidence`` and the
            explanation of ``uncertainty_metric_``. Use ``to_frame``,
            ``feature_importance`` and ``plot`` on the returned result.
        """

        self._check_is_fitted()
        self._check_X(X)
        if isinstance(self.explainer_, PDPUncertaintyExplainer) and np.asarray(X).shape[0] == 1:
            raise ValueError(
                "PDP does not support single-sample (local) explanations. "
                "Use xai_method='shap' or 'lime' instead."
            )
        features_kw = explainer_kwargs.get("features", getattr(self.explainer_, "features", None))
        if features_kw is None:
            features_kw = []
        pairs = [f for f in features_kw if isinstance(f, tuple)]
        is_pdp = isinstance(self.explainer_, PDPUncertaintyExplainer)

        if plot_kind is None and pairs and is_pdp:
            plot_kind = ["pdp_2d", "pdp"]
        if plot_kind is not None:
            self._resolve_plot_kinds(plot_kind, X=X)
            kinds = [plot_kind] if isinstance(plot_kind, str) else plot_kind
            if "pdp_2d" in kinds and not pairs:
                raise ValueError(
                    "plot_kind='pdp_2d' requires at least one feature pair as a tuple, "
                    "e.g. features=[(0, 1)] or features=[0, 1, (0, 1)]."
                )

        # Rebuild explainer if kwargs changed
        if self._explainer_kwargs != explainer_kwargs:
            if self._explainer_base_params is None and hasattr(self.explainer_, "get_params"):
                self._explainer_base_params = self.explainer_.get_params(deep=False)
            if self._explainer_base_params is not None:
                self.explainer_.set_params(**self._explainer_base_params)
            if explainer_kwargs:
                warnings.warn("Configure the explainer in its constructor or with set_params; "
                              "explain keyword configuration is deprecated.", FutureWarning,
                              stacklevel=2)
                self.explainer_.set_params(**explainer_kwargs)
            self.explainer_.fit(self._X_background)
            self._explainer_kwargs = explainer_kwargs

        explanation_values = self.explainer_.explain(X)

        X_arr = np.asarray(X)

        if self.task_ == "classification":
            prediction_set = self.conformal_predictor_.predict_set(
                X_arr, confidence=self.confidence,
            )
            p_values = self.conformal_predictor_.predict_p_values(X_arr)
            set_size = prediction_set.sum(axis=1)
            classes = np.asarray(self.conformal_predictor_.classes_)

            result = ClassificationExplanationResult(
                prediction_set=prediction_set,
                p_values=p_values,
                set_size=set_size,
                classes=classes,
                explanation_values=explanation_values,
            )
        else:
            lower, upper = self.conformal_predictor_.predict_interval(
                X_arr, confidence=self.confidence,
            )
            width = upper - lower
            result = ExplanationResult(
                lower=lower,
                upper=upper,
                interval_width=width,
                explanation_values=explanation_values,
            )

        if show_plots:
            self.plot(
                explanation_values,
                X=X,
                kind=plot_kind,
                waterfall_index=waterfall_index,
                figsize=figsize,
            )

        return result

    explain_uncertainty = explain

    def plot(
        self,
        explanation_values,
        X=None,
        kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
        figsize: tuple[float, float] | None = None,
        result=None,
        show: bool = True,
    ) -> dict:
        """
        Generate plot(s) from existing explanation results.

        Parameters
        ----------
        explanation_values : UncertaintyExplanation
            Method-specific explanation, typically ``result.explanation_values``
            from ``explain``.
        X : array-like of shape (n_samples, n_features), optional
            Explained feature values for SHAP beeswarm colors. Defaults to
            the data carried by the SHAP explanation. Also selects a waterfall
            as the default plot when a single SHAP observation is supplied.
        kind : str or list of str, optional
            SHAP: ``"bar"``, ``"beeswarm"``, ``"waterfall"``, ``"summary"``.
            PDP: ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"pdp_2d"``,
            ``"importance"``. LIME: ``"local"``. Defaults depend on the method.
        waterfall_index : int, optional
            Row for a SHAP waterfall or LIME local plot. When omitted, SHAP
            selects the largest total absolute contribution and LIME the first
            row.
        figsize : tuple of (width, height), optional
            Figure size in inches forwarded to the underlying plot
            function. When ``None``, each plot function applies its
            own default.
        result : object, optional
            Retained for compatibility; currently unused.
        show : bool, default=True
            Display the figures. Pass ``False`` to only build them, e.g. to
            save or restyle them first.

        Returns
        -------
        dict
            Maps each plot kind to its ``matplotlib`` figure.
        """

        self._check_is_fitted()
        kinds = self._resolve_plot_kinds(kind, X=X)

        if isinstance(self.explainer_, PDPUncertaintyExplainer):
            return generate_pdp_plots(
                explanation_values,
                kinds=kinds,
                feature_names=self._feature_names,
                figsize=figsize,
                show=show,
            )
        if isinstance(self.explainer_, LimeUncertaintyExplainer):
            return generate_lime_plots(
                explanation_values,
                kinds=kinds,
                sample_index=waterfall_index,
                figsize=figsize,
                show=show,
            )
        shap_kwargs = {"figsize": figsize} if figsize is not None else {}
        return generate_shap_plots(
            explanation_values,
            X,
            kinds=kinds,
            feature_names=self._feature_names,
            waterfall_index=waterfall_index,
            show=show,
            **shap_kwargs,
        )

    # -- checks ------------------------------------------------------------------

    def _validate_metric(self, metric, task):
        if task == "classification" and metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"uncertainty_metric='{metric}' is not valid for "
                f"classification. Choose from {CLASSIFICATION_METRICS}."
            )
        if task == "regression" and metric not in REGRESSION_METRICS:
            raise ValueError(
                f"uncertainty_metric='{metric}' is not valid for "
                f"regression. Choose from {REGRESSION_METRICS}."
            )

    def _check_confidence(self, confidence):
        check_confidence(confidence)

    def _check_is_fitted(self):
        check_is_fitted(self)

    def _check_X(self, X):
        check_array(X, dtype=None)
        validate_data(self, X, reset=False, skip_check_array=True)

    def _resolve_plot_kinds(
        self,
        kind: str | list[str] | None,
        X=None,
    ) -> list[str]:
        if isinstance(self.explainer_, PDPUncertaintyExplainer):
            valid = VALID_PDP_PLOT_KINDS
        elif isinstance(self.explainer_, LimeUncertaintyExplainer):
            valid = VALID_LIME_PLOT_KINDS
        else:
            valid = VALID_PLOT_KINDS
        if kind is None:
            if (
                isinstance(self.explainer_, ShapUncertaintyExplainer)
                and X is not None
                and np.asarray(X).shape[0] == 1
            ):
                return ["waterfall"]
            return None  # each plot function applies its own defaults
        if isinstance(kind, str):
            kind = [kind]
        for k in kind:
            if k not in valid:
                raise ValueError(
                    f"Unknown plot kind '{k}'. "
                    f"Choose from {valid}"
                )
        return kind


# Keep the original public name and module path available for existing imports
# and pickles created before the class was named ConformalExplainer.
UncertaintyExplanationPipeline = ConformalExplainer
