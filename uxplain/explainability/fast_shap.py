"""
Exact TreeSHAP shortcut for conformal summaries that are linear in the
component models.

The generic path hands ``shap.Explainer`` a plain callable — the composition of
the base estimator(s) with the conformal calibration. Because SHAP cannot see that a
tree ensemble sits inside that callable, it falls back to Exact or permutation
explainers, whose cost grows with the number of features and the size of the
background set.

Several conformal summaries are, however, *affine* in the component models'
predictions. Writing the summary as

.. math::

    u(x) = \\sum_k w_k\\, m_k(x) + c,

with fixed weights :math:`w_k` and a constant :math:`c`, and using the linearity
of Shapley values in the payoff function, gives

.. math::

    \\phi_j\\big(u, x\\big) = \\sum_k w_k\\, \\phi_j\\big(m_k, x\\big).

So the attributions can be obtained by running exact TreeSHAP on each component
estimator and combining the results, which is orders of magnitude cheaper than
KernelSHAP over the composed function. The constant :math:`c` (the calibration
offset) shifts only the base value, never the attributions. The identity holds
because every component uses the same interventional game: the same background
rows that the generic explainer receives.

Eligible combinations:

===================================  ===================================
Predictor / metric                   Decomposition
===================================  ===================================
CQR, ``"width"``                     ``upper_estimator - lower_estimator``
CQR, ``"midpoint"``                  ``(lower_estimator + upper_estimator) / 2``
CQR, ``"lower"`` / ``"upper"``       the corresponding quantile estimator
rearranged CQR, ``"midpoint"``       ``(lower_estimator + upper_estimator) / 2``
crepes standard, ``"lower"``,        the base estimator (the calibration
``"upper"``, ``"midpoint"``          offset is constant)
crepes standard, ``"width"``         constant — all attributions are zero
===================================  ===================================

``"normalized"`` scales the interval by a difficulty estimator; only its
symmetric midpoint remains affine in the base estimator. Mondrian variants and
custom predictors use the generic explainer: an empirical residual check
cannot prove affinity on SHAP's masked inputs. In particular, a masked input
may fall in an uncalibrated Mondrian bin with an undefined midpoint.
"""

from __future__ import annotations

import numpy as np

from ..conformal.cqr_predictor import CQRConformalPredictor
from ..conformal.crepes_predictor import CrepesConformalPredictor

__all__ = ["describe_fast_path", "tree_shap_values"]

# Relative tolerance for the residual check that verifies the affine estimator.
_LINEARITY_RTOL = 1e-6

# Background rows added to the residual check. The explained rows alone are not
# enough: a single row, or a batch inside one Mondrian bin, always looks
# constant, so the check must also see the data that defines the baseline.
_CHECK_ROWS = 100


def _components(cp, metric: str):
    """Return ``[(estimator, weight), ...]`` for an affine summary, or ``None``.

    ``None`` means this predictor/metric pair has no affine decomposition and
    the caller must fall back to the generic explainer.
    """
    # --- CQR: interval endpoints are two independent quantile models ---
    # Require the built-in implementation, not attributes or subclasses whose
    # predict() may change the payoff away from the observed data. SHAP also
    # evaluates hybrid rows formed by mixing X and the background.
    if type(cp) is CQRConformalPredictor:
        lower_estimator, upper_estimator = cp.lower_estimator_, cp.upper_estimator_
        if getattr(cp, "rearrange", False):
            # min/max of the two models: only their average stays affine.
            if metric == "midpoint":
                return [(lower_estimator, 0.5), (upper_estimator, 0.5)]
            return None
        return {
            "width": [(upper_estimator, 1.0), (lower_estimator, -1.0)],
            "midpoint": [(lower_estimator, 0.5), (upper_estimator, 0.5)],
            "lower": [(lower_estimator, 1.0)],
            "upper": [(upper_estimator, 1.0)],
        }.get(metric)

    # --- crepes-style: a single base estimator with a calibration offset ---
    #
    # Only the standard method has a constant calibration offset. A normalized
    # or Mondrian width can look constant on all observed rows yet vary on the
    # hybrid rows used in the Shapley game, so residual checks are not enough.
    if type(cp) is not CrepesConformalPredictor:
        return None
    estimator = cp.estimator_
    if cp.method != "standard":
        if cp.method == "normalized":
            return [(estimator, 1.0)] if metric == "midpoint" else None
        return None
    return {
        # width drops the base estimator entirely; it qualifies only when the
        # calibration offset is the same for every sample.
        "width": [],
        "midpoint": [(estimator, 1.0)],
        "lower": [(estimator, 1.0)],
        "upper": [(estimator, 1.0)],
    }.get(metric)


def _tree_explainer(estimator, background):
    """Build a ``shap.TreeExplainer``, or return ``None`` if unsupported."""
    import shap

    try:
        # A plain array would be wrapped in a masker that keeps at most 100
        # rows; an explicit masker keeps exactly the rows the generic path uses.
        return shap.TreeExplainer(
            estimator,
            data=shap.maskers.Independent(background, max_samples=len(background)),
            feature_perturbation="interventional",
        )
    except Exception:  # noqa: BLE001 - any failure here just means "not a tree ensemble"
        # Not a tree ensemble SHAP recognises (linear estimator, SVM, pipeline, ...).
        return None


def _check_rows(X, background):
    """Rows on which the affine decomposition is verified: ``X`` plus a fixed
    sample of the background."""
    if len(background) > _CHECK_ROWS:
        idx = np.random.default_rng(0).choice(len(background), _CHECK_ROWS, replace=False)
        background = background[idx]
    return np.vstack([X, background])


def tree_shap_values(cp, metric, X, background, target_fn):
    """
    Compute SHAP values for ``metric`` via exact TreeSHAP on component models.

    Parameters
    ----------
    cp
        Fitted conformal predictor.
    metric : str
        Scalar summary being explained.
    X : np.ndarray
        Samples to explain.
    background : np.ndarray
        Reference data defining the baseline expectation.
    target_fn : callable
        The true ``X -> metric`` function, used to verify the decomposition (on
        ``X`` and a sample of ``background``) and to recover the calibration
        offset.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        ``(values, base_values)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples,)``. ``None`` when the shortcut does not apply, in which
        case the caller should use the generic explainer.
    """
    components = _components(cp, metric)
    if components is None:
        return None

    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)

    # An empty component list means the summary is claimed to be constant (e.g.
    # the width of a standard conformal interval). It flows through the same
    # path: the affine part is zero, every attribution is zero, and the residual
    # check below confirms the value really is constant.
    explainers = []
    for estimator, weight in components:
        expl = _tree_explainer(estimator, background)
        if expl is None:
            return None
        explainers.append((expl, weight))

    # The calibration offset is whatever the true summary adds on top of the
    # affine part. It must be constant across samples; if it is not, the
    # decomposition is wrong for this predictor and we refuse to use it. The
    # check runs before any SHAP value is computed, and on more than the
    # explained rows (see _CHECK_ROWS). Non-finite summaries, such as unbounded
    # intervals, cannot be decomposed either.
    check = _check_rows(X, background)
    actual = np.asarray(target_fn(check), dtype=float).ravel()
    if actual.shape != (len(check),) or not np.all(np.isfinite(actual)):
        return None
    linear_part = np.zeros(len(check))
    for expl, weight in explainers:
        linear_part += weight * np.asarray(
            expl.model.predict(check), dtype=float
        ).ravel()
    offset = actual - linear_part
    scale = max(np.abs(actual).max(), 1.0)
    if not np.all(np.isfinite(offset)) or np.ptp(offset) > _LINEARITY_RTOL * scale:
        return None

    values = np.zeros_like(X)
    base = 0.0
    for expl, weight in explainers:
        phi = np.asarray(expl.shap_values(X, check_additivity=False), dtype=float)
        if phi.ndim != 2 or phi.shape != X.shape:
            return None
        values += weight * phi
        base += weight * float(np.mean(np.atleast_1d(expl.expected_value)))

    base_values = np.full(len(X), base + float(offset.mean()))
    return values, base_values


def describe_fast_path(cp, metric: str) -> str:
    """One-line description of whether the shortcut applies, for logs and docs."""
    components = _components(cp, metric)
    if components is None:
        return f"metric '{metric}': no affine decomposition, using generic SHAP"
    if not components:
        return (
            f"metric '{metric}': candidate constant summary — attributions are "
            "zero if the calibration offset does not vary across samples"
        )
    return (
        f"metric '{metric}': candidate affine decomposition over "
        f"{len(components)} component estimator(s), eligible for exact TreeSHAP"
    )
