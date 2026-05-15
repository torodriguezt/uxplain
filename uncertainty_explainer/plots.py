"""
Visualization utilities for SHAP, PDP, LIME, and conformal-classification
explanations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap

from .uncertainty.metrics import metric_label


def generate_shap_plots(
    explanation,
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
    explanation
        SHAP explanation object (``shap.Explanation``).

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
        explanation.feature_names = feature_names

    auto_selected = waterfall_index is None
    if auto_selected:
        waterfall_index = int(np.abs(explanation.values).sum(axis=1).argmax())

    metric_name = metric_label(getattr(explanation, "metric", "width"))
    metric_text = metric_name.lower()

    figures = {}

    if "summary" in kinds:
        plt.figure()
        shap.summary_plot(
            explanation,
            X,
            show=False,
            rng=np.random.default_rng(0),
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP summary — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["summary"] = fig

    if "bar" in kinds:
        plt.figure()
        shap.plots.bar(
            explanation,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP bar — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["bar"] = fig

    if "beeswarm" in kinds:
        plt.figure()
        shap.plots.beeswarm(
            explanation,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP beeswarm — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["beeswarm"] = fig

    if "waterfall" in kinds:
        plt.figure()
        shap.plots.waterfall(
            explanation[waterfall_index],
            show=False,
        )
        fig = plt.gcf()
        sample_label = (
            f"Highest SHAP contribution instance (sample {waterfall_index})"
            if auto_selected
            else f"SHAP waterfall — sample {waterfall_index}"
        )
        fig.suptitle(
            f"{sample_label} — Uncertainty metric ({metric_text})",
            y=1.01, fontsize=11,
        )
        figures["waterfall"] = fig

    if show:
        plt.show()

    return figures


generate_default_plots = generate_shap_plots


def generate_lime_plots(
    explanation,
    kinds: list[str] | None = None,
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

        Defaults to ``["local"]``.

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
        kinds = ["local"]

    if sample_index is None:
        sample_index = 0

    fnames = explanation.feature_names
    n = len(fnames)
    metric_text = metric_label(getattr(explanation, "metric", "width")).lower()
    figures = {}

    # --- Local ---
    if "local" in kinds:
        coefs = explanation.local_coefficients[sample_index]
        order = np.argsort(np.abs(coefs))
        labels = [fnames[i] for i in order]
        colors = ["lightgray" if c >= 0 else "tomato" for c in coefs[order]]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * n)))
        bars = ax.barh(labels, coefs[order], color=colors, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"LIME coefficient (effect on {metric_text})")
        ax.set_title(
            f"LIME — Local explanation - Uncertainty metric {metric_text}"
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["local"] = fig

    if show:
        plt.show()

    return figures


_PDP_KIND_REQUIREMENTS = {
    "pdp": ("average", "both"),
    "ice": ("individual", "both"),
    "pdp_ice": ("both",),
}


def _validate_pdp_kinds(kinds: list[str], explanation_kind: str) -> None:
    for plot_kind in kinds:
        allowed = _PDP_KIND_REQUIREMENTS.get(plot_kind)
        if allowed is None:
            continue
        if explanation_kind not in allowed:
            allowed_str = " or ".join(f'"{k}"' for k in allowed)
            raise ValueError(
                f'plot_kind="{plot_kind}" requires the explainer to be fitted '
                f'with kind={allowed_str}, but got kind="{explanation_kind}".'
            )


def generate_pdp_plots(
    explanation,
    kinds: list[str] | None = None,
    feature_names: list[str] | None = None,
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

    _validate_pdp_kinds(kinds, explanation.kind)

    if feature_names is not None:
        explanation.feature_names = feature_names

    def _fname(i: int) -> str:
        if explanation.feature_names is not None and i < len(explanation.feature_names):
            return explanation.feature_names[i]
        return f"Feature {explanation.features[i]}"

    metric_name = metric_label(getattr(explanation, "metric", "width"))
    metric_text = metric_name.lower()

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
            ax.set_ylabel(metric_name)
            ax.set_title(f"PDP — {_fname(i)}")
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"Partial Dependence — Uncertainty metric ({metric_text})", y=1.01)
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
            ax.set_ylabel(metric_name)
            ax.set_title(f"ICE — {_fname(i)}")
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"ICE — Uncertainty metric ({metric_text})", y=1.01)
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
            ax.set_ylabel(metric_name)
            ax.set_title(f"PDP + ICE — {_fname(i)}")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"PDP + ICE — Uncertainty metric ({metric_text})", y=1.01)
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
            fig.colorbar(filled, ax=ax, label=metric_name)
            ax.set_xlabel(fname_a)
            ax.set_ylabel(fname_b)
            ax.set_title(
                f"2D PDP — {fname_a} × {fname_b} — Uncertainty metric ({metric_text})"
            )
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
        ax.set_title(f"Feature importance — uncertainty metric ({metric_text})")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["importance"] = fig

    if show:
        plt.show()

    return figures


def generate_classification_plots(
    result,
    kinds: list[str] | None = None,
    sample_index: int = 0,
    show: bool = True,
) -> dict:
    """
    Generate plots specific to conformal classification output.

    Operates on the conformal output (prediction sets, p-values), not
    on the explainer output. The XAI explainer plots
    (``generate_default_plots``, ``generate_pdp_plots``,
    ``generate_lime_plots``) work unchanged for classification because
    they explain a scalar uncertainty metric.

    Parameters
    ----------
    result : ClassificationExplanationResult
        Output of ``UncertaintyExplanationPipeline.explain()`` when
        ``task="classification"``.

    kinds : list of str, optional
        Which plots to generate. Options:

        - ``"set_size"``       — histogram of prediction set sizes
        - ``"p_values"``       — bar chart of p-values per class for one sample
        - ``"set_membership"`` — heatmap of set membership across samples x classes

        Defaults to ``["set_size", "p_values"]``.

    sample_index : int
        Sample row to highlight in the ``"p_values"`` plot.

    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    dict
        Dictionary mapping plot kind to matplotlib figure.
    """

    if kinds is None:
        kinds = ["set_size", "p_values"]

    classes = np.asarray(result.classes)
    class_labels = [str(c) for c in classes]
    figures = {}

    # --- Set-size distribution ---
    if "set_size" in kinds:
        sizes = np.asarray(result.set_size).astype(int)
        max_size = int(sizes.max()) if len(sizes) else 0
        bins = np.arange(0, max_size + 2) - 0.5

        fig, ax = plt.subplots(figsize=(6, 4))
        counts, _, patches = ax.hist(sizes, bins=bins, color="salmon",
                                     edgecolor="black", linewidth=0.5)
        ax.set_xticks(np.arange(0, max_size + 1))
        ax.set_xlabel("Prediction set size")
        ax.set_ylabel("Number of samples")
        mean_size = float(sizes.mean()) if len(sizes) else 0.0
        ax.set_title(
            f"Prediction set size distribution (mean = {mean_size:.2f})"
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    count,
                    f"{int(count)}",
                    ha="center", va="bottom", fontsize=8,
                )
        fig.tight_layout()
        figures["set_size"] = fig

    # --- P-values per class for a single sample ---
    if "p_values" in kinds:
        p_row = np.asarray(result.p_values)[sample_index]
        in_set = np.asarray(result.prediction_set)[sample_index]
        colors = ["tomato" if inc else "lightgray" for inc in in_set]

        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(classes))))
        bars = ax.barh(class_labels, p_row, color=colors,
                       edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel("Conformal p-value")
        ax.set_ylabel("Class")
        ax.set_title("P-values per class — red = in prediction set")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        figures["p_values"] = fig

    # --- Set membership heatmap ---
    if "set_membership" in kinds:
        membership = np.asarray(result.prediction_set).astype(int)
        fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(classes)),
                                         max(3, 0.15 * membership.shape[0])))
        im = ax.imshow(membership, aspect="auto", cmap="RdYlBu_r",
                       interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_xlabel("Class")
        ax.set_ylabel("Sample index")
        ax.set_title("Prediction set membership (1 = in set)")
        fig.colorbar(im, ax=ax, ticks=[0, 1])
        fig.tight_layout()
        figures["set_membership"] = fig

    if show:
        plt.show()

    return figures