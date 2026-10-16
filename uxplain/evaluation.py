"""
Summaries of fitted conformal predictors and of their explanations.

- `coverage_table` reports empirical coverage and the size of the
  conformal outputs (signed interval width or prediction-set size) on labelled
  data, overall and within user-supplied groups.
- `importance_table` turns an explanation into a global feature ranking,
  optionally with bootstrap intervals over the explained rows.

Both return `pandas.DataFrame` objects, so they print, compare and
export like any other table. Coverage tables display four decimal places while
retaining the full numerical precision for calculations and exports.
`uxplain.ConformalExplainer`
exposes the first as ``evaluate()``; explanations and results expose the second
as ``feature_importance()``.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

__all__ = ["coverage_table", "importance_table"]


class EvaluationTable(pd.DataFrame):
    """A DataFrame with four-decimal display and unrounded numerical values.

    Pass ``float_format`` to ``to_string()`` or ``to_html()`` to override the
    display format. Numeric operations and exports such as ``to_csv()`` use
    the original values. DataFrame selections and copies retain this display.
    """

    @property
    def _constructor(self):
        return EvaluationTable

    def to_string(self, buf=None, *, float_format=None, **kwargs):
        """Format as text, displaying floats to four decimals by default."""
        if float_format is None:
            float_format = "{:.4f}".format
        return super().to_string(buf, float_format=float_format, **kwargs)

    def to_html(self, buf=None, *, float_format=None, **kwargs):
        """Format as HTML, displaying floats to four decimals by default."""
        if float_format is None:
            float_format = "{:.4f}".format
        return super().to_html(buf, float_format=float_format, **kwargs)

    def _repr_html_(self):
        """Use the same precision for notebook display without global options."""
        if not pd.get_option("display.notebook_repr_html"):
            return None
        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        truncated = (max_rows and len(self) > max_rows) or (
            max_cols and len(self.columns) > max_cols
        )
        if truncated and pd.get_option("display.large_repr") == "info":
            return super()._repr_html_()
        min_rows = pd.get_option("display.min_rows")
        if max_rows and len(self) > max_rows and min_rows:
            max_rows = min(max_rows, min_rows)
        return self.to_html(
            notebook=True,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=pd.get_option("display.show_dimensions"),
        )


def _covered_and_size(prediction, y, classes):
    if classes is None:
        lower, upper = (np.asarray(v, dtype=float) for v in prediction)
        y = np.asarray(y, dtype=float)
        if not (lower.shape == upper.shape == y.shape):
            raise ValueError("lower, upper and y must have the same shape.")
        return (lower <= y) & (y <= upper), upper - lower
    sets = np.asarray(prediction, dtype=bool)
    classes = list(classes)
    if sets.ndim != 2 or sets.shape[1] != len(classes):
        raise ValueError("prediction sets must have one column per class.")
    y = np.asarray(y)
    if len(y) != len(sets):
        raise ValueError("prediction sets and y must have the same length.")
    position = {label: k for k, label in enumerate(classes)}
    covered = np.zeros(len(y), dtype=bool)
    for i, label in enumerate(y):
        k = position.get(label)
        # A label the fitted estimator has never seen cannot be covered.
        covered[i] = k is not None and sets[i, k]
    return covered, sets.sum(axis=1).astype(float)


def coverage_table(prediction, y, classes=None, groups=None) -> pd.DataFrame:
    """Empirical coverage and size of conformal outputs on labelled data.

    Parameters
    ----------
    prediction
        ``(lower, upper)`` for regression, or a boolean prediction-set array
        of shape ``(n_samples, n_classes)`` for classification.
    y
        Observed responses or labels.
    classes : sequence, optional
        Class labels in the column order of the prediction sets. Required for
        classification; leave as ``None`` for regression.
    groups : array-like, optional
        Group label of each observation (e.g. the true class, or a survey
        stratum). Adds one row per group after the ``"overall"`` row.

    Returns
    -------
    pandas.DataFrame
        Columns ``n``, ``coverage``, ``mean_size``, ``median_size`` and
        ``share_empty``. Size is the signed interval width (negative for empty
        CQR sets) or the number of labels in the set. Floats display with four
        decimal places; calculations and exports retain full precision.
    """

    covered, size = _covered_and_size(prediction, y, classes)
    frame = pd.DataFrame({"covered": covered, "size": size})
    rows = {"overall": frame}
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(frame):
            raise ValueError("groups must have one entry per observation.")
        labels = list(pd.unique(groups))
        try:
            labels = sorted(labels)
        except TypeError:  # mixed label types keep their order of appearance
            pass
        for label in labels:
            rows[label] = frame[groups == label]

    table = EvaluationTable(
        {
            "n": [len(f) for f in rows.values()],
            "coverage": [f["covered"].mean() for f in rows.values()],
            "mean_size": [f["size"].mean() for f in rows.values()],
            "median_size": [f["size"].median() for f in rows.values()],
            "share_empty": [(f["size"] < (0 if classes is None else 1)).mean()
                            for f in rows.values()],
        },
        index=pd.Index(list(rows), name="group"),
    )
    return table


def _attribution_matrix(explanation):
    """``(rows x features)`` attributions and their feature names, if any."""
    if hasattr(explanation, "local_coefficients"):  # LIMEExplanation
        return (np.asarray(explanation.local_coefficients, dtype=float),
                explanation.feature_names)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("Attributions must form a (rows x features) matrix.")
    return values, getattr(explanation, "feature_names", None)


def _ranks(importance):
    """Rank 1 for the largest value; ties share the better rank."""
    importance = np.asarray(importance)
    return (importance[..., None, :] > importance[..., :, None]).sum(axis=-1) + 1


def importance_table(
    explanation,
    feature_names=None,
    n_boot: int = 0,
    level: float = 0.95,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Global importance of each feature for the explained summary.

    SHAP and LIME explanations are summarised by the mean absolute attribution
    over the explained rows. Partial-dependence explanations use the standard
    deviation of the averaged curve (Greenwell et al., 2018).

    Parameters
    ----------
    explanation
        A `shap.Explanation`, ``LIMEExplanation`` or ``PDPExplanation``.
    feature_names : sequence of str, optional
        Overrides the names carried by the explanation.
    n_boot : int, default=0
        Number of bootstrap resamples of the explained rows (of the ICE curves
        for partial dependence, which then needs ``kind="both"``). Adds
        percentile intervals for the importance and for the rank. The
        intervals describe variability over the explained sample only; they
        do not account for refitting the estimator or redrawing the calibration
        data.
    level : float, default=0.95
        Coverage of the bootstrap percentile intervals.
    random_state : int, optional
        Seed for the bootstrap.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature and sorted by decreasing importance, with columns
        ``importance`` and ``rank``; with ``n_boot > 0`` also ``ci_lower``,
        ``ci_upper``, ``rank_lower`` and ``rank_upper``.
    """

    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 0:
        raise ValueError("n_boot must be a non-negative integer.")
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1).")

    rng = np.random.default_rng(random_state)
    if hasattr(explanation, "grid_values") and hasattr(explanation, "importance"):
        names = feature_names or explanation.feature_names
        features = list(explanation.features)
        labels = ([names[f] if isinstance(f, Integral) else str(f) for f in features]
                  if names is not None else [str(f) for f in features])
        point = np.asarray(explanation.importance, dtype=float)
        boot = None
        if n_boot:
            if explanation.individual is None:
                raise ValueError(
                    "Bootstrap intervals for partial dependence need ICE curves; "
                    "explain with kind='both'."
                )
            curves = [np.asarray(c, dtype=float) for c in explanation.individual]
            n = len(curves[0])
            draws = rng.integers(0, n, size=(n_boot, n))
            boot = np.array([[c[idx].mean(axis=0).std() for c in curves]
                             for idx in draws])
    else:
        values, carried = _attribution_matrix(explanation)
        names = feature_names if feature_names is not None else carried
        labels = (list(names) if names is not None
                  else [f"Feature {j}" for j in range(values.shape[1])])
        if len(labels) != values.shape[1]:
            raise ValueError("feature_names must match the number of features.")
        magnitude = np.abs(values)
        point = magnitude.mean(axis=0)
        boot = None
        if n_boot:
            draws = rng.integers(0, len(values), size=(n_boot, len(values)))
            boot = np.stack([magnitude[idx].mean(axis=0) for idx in draws])

    table = pd.DataFrame(
        {"importance": point, "rank": _ranks(point)},
        index=pd.Index(labels, name="feature"),
    )
    if boot is not None:
        tail = 100 * (1 - level) / 2
        table["ci_lower"], table["ci_upper"] = np.percentile(boot, [tail, 100 - tail], axis=0)
        ranks = _ranks(boot)
        table["rank_lower"] = np.percentile(ranks, tail, axis=0, method="lower").astype(int)
        table["rank_upper"] = np.percentile(ranks, 100 - tail, axis=0,
                                            method="higher").astype(int)
    return table.sort_values("importance", ascending=False)
