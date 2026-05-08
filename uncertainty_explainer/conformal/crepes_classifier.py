"""
Conformal classification module.

Wrapper around crepes.WrapClassifier for prediction sets.
"""

from __future__ import annotations

from typing import Literal

from crepes import WrapClassifier
from crepes.extras import MondrianCategorizer
import numpy as np

ClassificationConformalMethod = Literal[
    "standard",
    "class_cond",
    "mondrian",
]


class CrepesConformalClassifier:
    """
    Wrapper for conformal classification using crepes.

    Parameters
    ----------
    model : object
        Any sklearn-compatible classifier exposing
        ``fit``, ``predict``, and ``predict_proba``.
    method : ClassificationConformalMethod
        Conformal classification method:

        - ``"standard"``    : basic (marginal) conformal classifier
        - ``"class_cond"``  : class-conditional Mondrian (per-class coverage)
        - ``"mondrian"``    : Mondrian categorizer based on predicted class

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> cp = CrepesConformalClassifier(RandomForestClassifier(), method="class_cond")
    >>> cp.fit(X_train, y_train, X_calib, y_calib)
    >>> pred_set = cp.predict_set(X_test, confidence=0.9)   # (n, n_classes) bool
    >>> p_values = cp.predict_p(X_test)                     # (n, n_classes)
    """

    def __init__(
        self,
        model,
        method: ClassificationConformalMethod = "standard",
    ):
        self.model = model
        self.method = method
        self.wrapper = None
        self.mondrian_categorizer = None
        self.classes_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fit conformal classifier.

        Steps:
        1. Fit underlying model
        2. Optionally fit Mondrian categorizer
        3. Calibrate conformal classifier
        """

        # Train
        self.model.fit(X_train, y_train)
        self.classes_ = np.asarray(self.model.classes_)

        self.wrapper = WrapClassifier(self.model)

        # Build calibration kwargs based on method
        calibrate_kwargs = {}

        if self.method == "class_cond":
            calibrate_kwargs["class_cond"] = True
        elif self.method == "mondrian":
            calibrate_kwargs["mc"] = self.model.predict

        # Calibrate
        self.wrapper.calibrate(
            X_calib,
            y_calib,
            **calibrate_kwargs,
        )

    def predict_set(
        self,
        X: np.ndarray,
        confidence: float = 0.9,
    ) -> np.ndarray:
        """
        Generate prediction sets.

        Parameters
        ----------
        X : np.ndarray
        confidence : float

        Returns
        -------
        prediction_set : np.ndarray of shape (n_samples, n_classes), dtype=bool
            ``True`` at position ``[i, k]`` means class ``k`` is included in
            the prediction set for sample ``i``.
        """

        if self.wrapper is None:
            raise RuntimeError("CrepesConformalClassifier not fitted.")

        X = np.asarray(X)
        return self.wrapper.predict_set(X, confidence=confidence).astype(bool)

    def predict_p(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute conformal p-values for each class.

        Returns
        -------
        p_values : np.ndarray of shape (n_samples, n_classes)
        """

        if self.wrapper is None:
            raise RuntimeError("CrepesConformalClassifier not fitted.")

        X = np.asarray(X)
        return self.wrapper.predict_p(X)

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Underlying model probabilities."""

        return self.model.predict_proba(np.asarray(X))

