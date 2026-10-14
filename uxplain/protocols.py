"""
Protocols defining the interfaces for conformal
prediction and uncertainty explanation.

Concrete implementations (e.g. crepes, shap, LIME)
must satisfy these interfaces to be used in the pipeline.
Any object with these methods works; inheritance is not required.
See the "Extending uxplain" page of the documentation for complete examples,
including a backend that wraps MAPIE.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ConformalPredictorProtocol(Protocol):
    """Interface for regression conformal predictors.

    ``fit`` trains the underlying model(s) on ``(X_train, y_train)`` and
    calibrates on ``(X_calib, y_calib)``. ``predict`` returns the lower and
    upper bounds of the intervals at nominal coverage ``confidence``.

    The explained summaries are computed from ``predict``, so it must be a
    deterministic function of each row (independent of the batch it arrives
    in) and accept arbitrary feature combinations: SHAP and partial dependence
    evaluate it on rows that mix observed and background values. The pipeline
    clones the predictor before fitting it, so give it ``get_params`` (e.g. by
    inheriting from `sklearn.base.BaseEstimator`) or make it
    deep-copyable.
    """

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> Any:
        """Fit the model and calibrate prediction intervals on separate data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
            Features used to train the underlying model.
        y_train : np.ndarray of shape (n_train,)
            Training responses.
        X_calib : np.ndarray of shape (n_calib, n_features)
            Held-out calibration features in the same column order.
        y_calib : np.ndarray of shape (n_calib,)
            Calibration responses.

        Returns
        -------
        object
            Backend-defined return value, conventionally the fitted predictor.
            The pipeline uses the fitted instance and ignores this return value.
        """
        ...

    def predict(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal interval endpoints for each observation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features in the training column order.
        confidence : float, default=0.9
            Nominal coverage level, strictly between zero and one.

        Returns
        -------
        lower : np.ndarray of shape (n_samples,)
            Lower endpoints.
        upper : np.ndarray of shape (n_samples,)
            Upper endpoints. Preserve signed endpoints when the conformal
            method permits empty sets with ``upper < lower``.
        """
        ...


class ConformalClassifierProtocol(Protocol):
    """Interface for classification conformal predictors.

    Implementations must expose ``predict_set``, ``predict_p``,
    and a ``classes_`` attribute aligned with the model's class order.
    The same determinism and cloning requirements as for
    `ConformalPredictorProtocol` apply.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Labels in the column order returned by ``predict_set`` and ``predict_p``.
    """

    classes_: np.ndarray

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> Any:
        """Fit the classifier and calibrate prediction sets on separate data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
            Training features.
        y_train : np.ndarray of shape (n_train,)
            Training class labels.
        X_calib : np.ndarray of shape (n_calib, n_features)
            Held-out calibration features in the same column order.
        y_calib : np.ndarray of shape (n_calib,)
            Calibration class labels.

        Returns
        -------
        object
            Backend-defined return value, conventionally the fitted predictor.
            The pipeline ignores this value; fitting must populate ``classes_``.
        """
        ...

    def predict_set(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> np.ndarray:
        """Return membership of each candidate label in the prediction set.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features in the training column order.
        confidence : float, default=0.9
            Nominal coverage level, strictly between zero and one.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes), dtype=bool
            Whether each label is included, with columns ordered as ``classes_``.
        """
        ...

    def predict_p(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return conformal p-values for each candidate label.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features in the training column order.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Conformal p-values in the column order of ``classes_``. These
            measure conformity of candidate labels, not class probabilities.
        """
        ...


class UncertaintyExplainerProtocol(Protocol):
    """Interface for uncertainty explainers.

    ``fit`` receives the background data (keyword arguments of
    `UncertaintyExplanationPipeline.explain` are forwarded to
    it); ``explain`` returns the explanation of the rows ``X``. Return a
    subclass of `UncertaintyExplanation` so that the result's
    ``feature_importance()`` and ``plot()`` work. If the explainer has a ``cp``
    attribute, the pipeline points it to its fitted conformal predictor.
    """

    def fit(
        self,
        X_background: np.ndarray,
        **kwargs,
    ) -> None:
        """Prepare the explainer using background observations.

        Parameters
        ----------
        X_background : np.ndarray of shape (n_background, n_features)
            Background features supplied by the pipeline.
        **kwargs : dict
            Explainer-specific settings forwarded from ``pipeline.explain``.

        Returns
        -------
        None
            The explainer is prepared in place.
        """
        ...

    def explain(
        self,
        X: np.ndarray,
    ) -> Any:
        """Explain a scalar summary of conformal predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations to explain, supplied as an array or a DataFrame.

        Returns
        -------
        UncertaintyExplanation
            Method-specific explanation with ``feature_importance``,
            ``to_frame`` and ``plot`` accessors.
        """
        ...
