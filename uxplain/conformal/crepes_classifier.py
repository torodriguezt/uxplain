"""
Conformal classification module.

Wrapper around crepes.WrapClassifier for prediction sets.
"""

from __future__ import annotations

from typing import Literal

from crepes import WrapClassifier
import numpy as np

from ._validation import check_confidence, check_fit_data

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
    random_state : int, optional
        Seed for the tie-breaking draws of smoothed p-values. Ignored when
        ``smoothing=False``.
    smoothing : bool, default=False
        Use randomized tie-breaking for prediction. Even with a fixed seed,
        smoothed p-values can depend on row order and batch size, so built-in
        explainers require ``smoothing=False``. Non-smoothed p-values are
        deterministic for a fixed deterministic model and conservative under
        the conformal assumptions; smoothing does not repair invalid sampling.

    Notes
    -----
    Coverage requires calibration and future scores to be exchangeable after
    training, within the selected groups for conditional methods. The fitted
    model's classes must contain the full target label space. Unknown calibration
    labels raise an error; future labels absent from ``classes_`` cannot be
    covered. Do not select a split or tune the model on calibration outcomes.

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
        random_state: int | None = None,
        smoothing: bool = False,
    ):
        if method not in ("standard", "class_cond", "mondrian"):
            raise ValueError(f"Unknown conformal classification method '{method}'.")
        self.model = model
        self.method = method
        self.random_state = random_state
        self.smoothing = smoothing
        self.wrapper = None
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

        self.wrapper = None
        self.classes_ = None
        X_train, y_train, X_calib, y_calib = check_fit_data(
            X_train, y_train, X_calib, y_calib,
        )

        # Train
        self.model.fit(X_train, y_train)
        self.classes_ = np.asarray(self.model.classes_)
        if not np.all(np.isin(y_calib, self.classes_)):
            raise ValueError("y_calib contains classes absent from the fitted model.")

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

        check_confidence(confidence)
        X = np.asarray(X)
        # labels=False keeps the binary-array output; since crepes 0.9.1
        # the default (labels=True) returns a list of lists of labels.
        return self.wrapper.predict_set(
            X, confidence=confidence, smoothing=self.smoothing,
            seed=self.random_state, labels=False,
        ).astype(bool)

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
        return self.wrapper.predict_p(
            X, smoothing=self.smoothing, seed=self.random_state,
        )

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Underlying model probabilities."""

        return self.model.predict_proba(np.asarray(X))

