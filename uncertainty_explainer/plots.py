"""
Visualization utilities for SHAP and PDP explanations.
"""

from __future__ import annotations

from typing import List

import numpy as np
import shap
import matplotlib.pyplot as plt


def generate_default_plots(
    shap_values,
    X=None,
    kinds: list[str] | None = None,
    feature_names: list[str] | None = None,
    waterfall_index: int = 0,
    show: bool = True,
):
    """
    Generate SHAP plots.

    Parameters
    ----------
    shap_values
        SHAP explanation object.

    X : np.ndarray, optional

    kinds : list of str, optional
        Which plots to generate. Options: "beeswarm",
        "bar", "waterfall", "summary".
        Defaults to ["beeswarm", "bar", "waterfall"].

    feature_names : list of str, optional
        Feature names for axis labels.

    waterfall_index : int
        Sample index for waterfall plot.

    show : bool
        Whether to display plots.

    Returns
    -------
    dict
        Dictionary with matplotlib figures.
    """

    if kinds is None:
        kinds = ["beeswarm", "bar", "waterfall"]

    if feature_names is not None:
        shap_values.feature_names = feature_names

    figures = {}

    if "summary" in kinds:
        plt.figure()
        shap.summary_plot(
            shap_values,
            X,
            show=False,
        )
        figures["summary"] = plt.gcf()

    if "bar" in kinds:
        plt.figure()
        shap.plots.bar(
            shap_values,
            show=False,
        )
        figures["bar"] = plt.gcf()

    if "beeswarm" in kinds:
        plt.figure()
        shap.plots.beeswarm(
            shap_values,
            show=False,
        )
        figures["beeswarm"] = plt.gcf()

    if "waterfall" in kinds:
        plt.figure()
        shap.plots.waterfall(
            shap_values[waterfall_index],
            show=False,
        )
        figures["waterfall"] = plt.gcf()

    if show:
        plt.show()

    return figures


def generate_pdp_plots(
    explanation,
    kinds: List[str] | None = None,
    feature_names: List[str] | None = None,
    show: bool = True,
) -> dict:
    """
    Generate PDP plots from a ``PDPExplanation`` object.

    Parameters
    ----------
    explanation : PDPExplanation
        Output of ``PDPUncertaintyExplainer.explain()``.

    kinds : list of str, optional
        Which plots to generate. Options:

        - ``"pdp"``        — average partial dependence line per feature
        - ``"ice"``        — individual conditional expectation lines
        - ``"pdp_ice"``    — PDP overlaid on ICE lines
        - ``"importance"`` — bar chart of feature importance (PDP std)

        Defaults to ``["pdp", "importance"]``.

    feature_names : list of str, optional
        Overrides ``explanation.feature_names`` for axis labels.

    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    dict
        Dictionary mapping plot kind to matplotlib figure.
    """

    if kinds is None:
        kinds = ["pdp", "importance"]

    if feature_names is not None:
        explanation.feature_names = feature_names

    def _fname(i: int) -> str:
        if explanation.feature_names is not None and i < len(explanation.feature_names):
            return explanation.feature_names[i]
        return f"Feature {explanation.features[i]}"

    n = len(explanation.features)
    figures = {}

    # --- PDP ---
    if "pdp" in kinds:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n):
            ax = axes[i]
            ax.plot(explanation.grid_values[i], explanation.values[i], lw=2)
            ax.set_xlabel(_fname(i))
            ax.set_ylabel("Interval width")
            ax.set_title(f"PDP — {_fname(i)}")
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("Partial Dependence — Uncertainty (interval width)", y=1.01)
        fig.tight_layout()
        figures["pdp"] = fig

    # --- ICE ---
    if "ice" in kinds:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n):
            ax = axes[i]
            if explanation.individual is None:
                ax.set_title(f"ICE — {_fname(i)}\n(no individual data)")
                continue
            for line in explanation.individual[i]:
                ax.plot(explanation.grid_values[i], line, lw=0.5, alpha=0.3, color="steelblue")
            ax.set_xlabel(_fname(i))
            ax.set_ylabel("Interval width")
            ax.set_title(f"ICE — {_fname(i)}")
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("ICE — Uncertainty (interval width)", y=1.01)
        fig.tight_layout()
        figures["ice"] = fig

    # --- PDP + ICE overlay ---
    if "pdp_ice" in kinds:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n):
            ax = axes[i]
            if explanation.individual is not None:
                for line in explanation.individual[i]:
                    ax.plot(explanation.grid_values[i], line, lw=0.5, alpha=0.2, color="steelblue")
            ax.plot(explanation.grid_values[i], explanation.values[i], lw=2.5, color="tomato", label="PDP")
            ax.set_xlabel(_fname(i))
            ax.set_ylabel("Interval width")
            ax.set_title(f"PDP + ICE — {_fname(i)}")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("PDP + ICE — Uncertainty (interval width)", y=1.01)
        fig.tight_layout()
        figures["pdp_ice"] = fig

    # --- Feature importance (PDP std — Greenwell et al. 2018) ---
    if "importance" in kinds:
        importance = np.array([
            explanation.values[i].std(ddof=1)
            for i in range(n)
        ])
        order = np.argsort(importance)
        labels = [_fname(i) for i in order]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        bars = ax.barh(labels, importance[order], color="steelblue")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlabel(r"$I(\mathbf{x}_S)$ — std of PDP values")
        ax.set_title("Feature importance — Uncertainty PDP")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["importance"] = fig

    if show:
        plt.show()

    return figures