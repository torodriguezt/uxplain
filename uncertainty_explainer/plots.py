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
    waterfall_index: int | None = None,
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

    waterfall_index : int or None
        Sample index for waterfall plot. When ``None``, auto-selects
        the instance with the highest total absolute SHAP contribution.

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

    auto_selected = waterfall_index is None
    if auto_selected:
        waterfall_index = int(np.abs(shap_values.values).sum(axis=1).argmax())

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
        fig = plt.gcf()
        title = (
            f"Highest SHAP contribution instance (sample {waterfall_index})"
            if auto_selected
            else f"SHAP waterfall — sample {waterfall_index}"
        )
        fig.suptitle(title, y=1.01, fontsize=11)
        figures["waterfall"] = fig

    if show:
        plt.show()

    return figures


def generate_lime_plots(
    explanation,
    kinds: List[str] | None = None,
    sample_index: int = 0,
    show: bool = True,
) -> dict:
    """
    Generate LIME plots from a ``LIMEExplanation`` object.

    Parameters
    ----------
    explanation : LIMEExplanation
        Output of ``LimeUncertaintyExplainer.explain()``.

    kinds : list of str, optional
        Which plots to generate. Options:

        - ``"local"``  — LIME coefficients for a single sample
        - ``"global"`` — mean absolute importance across all samples

        Defaults to ``["local"]`` when ``explanation.scope == "local"``
        and ``["local", "global"]`` when ``scope == "global"``.

    sample_index : int
        Sample row to highlight in the ``"local"`` plot.

    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    dict
        Dictionary mapping plot kind to matplotlib figure.
    """

    if kinds is None:
        kinds = ["local", "global"] if explanation.scope == "global" else ["local"]

    if sample_index is None:
        sample_index = 0

    fnames = explanation.feature_names
    n = len(fnames)
    figures = {}

    # --- Local ---
    if "local" in kinds:
        coefs = explanation.local_coefficients[sample_index]
        order = np.argsort(np.abs(coefs))
        labels = [fnames[i] for i in order]
        colors = ["steelblue" if c >= 0 else "tomato" for c in coefs[order]]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        bars = ax.barh(labels, coefs[order], color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME coefficient (effect on interval width)")
        ax.set_title(f"LIME — Local explanation (sample {sample_index})")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["local"] = fig

    # --- Global ---
    if "global" in kinds:
        importance = explanation.global_importance
        order = np.argsort(importance)
        labels = [fnames[i] for i in order]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        bars = ax.barh(labels, importance[order], color="steelblue")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlabel(r"Mean $|\text{LIME coefficient}|$ (effect on interval width)")
        ax.set_title("LIME — Global feature importance")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["global"] = fig

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
        has_1d = len(explanation.features) > 0
        if explanation.kind == "individual":
            kinds = ["ice"] if has_1d else []
        elif explanation.kind == "both":
            kinds = ["pdp_ice"] if has_1d else []
        else:
            kinds = ["pdp"] if has_1d else []

    if feature_names is not None:
        explanation.feature_names = feature_names

    def _fname(i: int) -> str:
        if explanation.feature_names is not None and i < len(explanation.feature_names):
            return explanation.feature_names[i]
        return f"Feature {explanation.features[i]}"

    n = len(explanation.features)
    figures = {}

    # --- PDP ---
    if "pdp" in kinds and n > 0:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n):
            ax = axes[i]
            ax.plot(explanation.grid_values[i], explanation.values[i], lw=2, color="tomato")
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
    if "ice" in kinds and n > 0:
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
    if "pdp_ice" in kinds and n > 0:
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

    # --- 2D PDP heatmaps ---
    if "pdp_2d" in kinds and explanation.values_2d:
        for idx, (pair, z, (gx, gy)) in enumerate(
            zip(explanation.feature_pairs, explanation.values_2d, explanation.grid_values_2d)
        ):
            fname_a = (
                explanation.feature_names[pair[0]]
                if explanation.feature_names else f"Feature {pair[0]}"
            )
            fname_b = (
                explanation.feature_names[pair[1]]
                if explanation.feature_names else f"Feature {pair[1]}"
            )
            GX, GY = np.meshgrid(gx, gy)
            fig, ax = plt.subplots(figsize=(6, 5))
            filled = ax.contourf(GX, GY, z.T, levels=12, cmap="RdYlBu_r")
            lines = ax.contour(GX, GY, z.T, levels=filled.levels, colors="white", linewidths=0.6, alpha=0.5)
            ax.clabel(lines, inline=True, fontsize=7, fmt="%.2f")
            fig.colorbar(filled, ax=ax, label="Interval width")
            ax.set_xlabel(fname_a)
            ax.set_ylabel(fname_b)
            ax.set_title(f"2D PDP — {fname_a} × {fname_b}")
            fig.tight_layout()
            figures[f"pdp_2d_{idx}"] = fig

    # --- Feature importance (PDP std — Greenwell et al. 2018) ---
    if "importance" in kinds and n > 0:
        importance = np.array([
            explanation.values[i].std(ddof=1)
            for i in range(n)
        ])
        order = np.argsort(importance)
        labels = [_fname(i) for i in order]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        bars = ax.barh(labels, importance[order], color="gray")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlabel(r"$I(\mathbf{x}_S)$ — std of PDP values")
        ax.set_title("Feature importance")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["importance"] = fig

    if show:
        plt.show()

    return figures