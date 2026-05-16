"""
Visualization utilities for SHAP, PDP, LIME, and conformal-classification
explanations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import PartialDependenceDisplay

from .uncertainty.metrics import metric_label


def _fix_waterfall_text_overlap(fig):
    """
    Reposition SHAP waterfall contribution labels that overlap with the
    y-tick labels (feature names). For small negative bars, SHAP places
    the value text at the arrow tip with right-alignment, which extends
    further left and lands on top of the y-tick label. We detect overlaps
    after rendering and shift the offending text well inside the chart
    area with left alignment so it remains readable.
    """
    fig.canvas.draw()
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()

    ytick_bboxes = [
        lbl.get_window_extent(renderer=renderer)
        for lbl in ax.get_yticklabels()
        if lbl.get_text()
    ]
    if not ytick_bboxes:
        return

    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    safe_x = xlim[0] + 0.08 * x_range

    for txt in ax.texts:
        if not txt.get_text() or txt.get_horizontalalignment() != "right":
            continue
        txt_bbox = txt.get_window_extent(renderer=renderer)
        if any(txt_bbox.overlaps(yb) for yb in ytick_bboxes):
            _, y = txt.get_position()
            txt.set_position((safe_x, y))
            txt.set_horizontalalignment("left")

    fig.canvas.draw()


def generate_shap_plots(
    explanation,
    X=None,
    kinds: list[str] | None = None,
    feature_names: list[str] | None = None,
    waterfall_index: int | None = None,
    figsize: tuple[float, float] = (10, 6),
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

    figsize : tuple of (width, height), optional
        Figure size in inches for each plot. Default ``(10, 6)``.
        The waterfall plot enforces a minimum width of 12 inches so that
        contribution labels do not overlap with feature names.

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
        plt.figure(figsize=figsize)
        shap.summary_plot(
            explanation,
            X,
            show=False,
            rng=np.random.default_rng(0),
            plot_size=figsize,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP summary — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["summary"] = fig

    if "bar" in kinds:
        plt.figure(figsize=figsize)
        shap.plots.bar(
            explanation,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(*figsize)
        fig.suptitle(f"SHAP bar — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["bar"] = fig

    if "beeswarm" in kinds:
        plt.figure(figsize=figsize)
        shap.plots.beeswarm(
            explanation,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(*figsize)
        fig.suptitle(f"SHAP beeswarm — Uncertainty metric ({metric_text})", y=1.01, fontsize=11)
        figures["beeswarm"] = fig

    if "waterfall" in kinds:
        plt.figure(figsize=figsize)
        shap.plots.waterfall(
            explanation[waterfall_index],
            show=False,
        )
        fig = plt.gcf()
        w, h = figsize
        fig.set_size_inches(max(w, 12), h)
        _fix_waterfall_text_overlap(fig)
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
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> dict:
    """
    Generate LIME plots from a ``LIMEExplanation`` object.

    Uses LIME's native ``as_pyplot_figure()`` when raw explanations are
    available (the default for explanations produced by this library).

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

    figsize : tuple of (width, height), optional
        Figure size in inches. When ``None``, scales height with the
        number of features (``(8, max(4, 0.5 * n_features))``).

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

    metric_text = metric_label(getattr(explanation, "metric", "width")).lower()
    figures = {}

    n_features = len(explanation.feature_names) if explanation.feature_names is not None else 0
    if figsize is None:
        figsize = (8, max(4, 0.5 * n_features))

    if "local" in kinds:
        raw = getattr(explanation, "raw_explanations", None)
        if raw is not None:
            exp = raw[sample_index]
            fig = exp.as_pyplot_figure(label=1)
            fig.set_size_inches(*figsize)
            ax = fig.axes[0]
            ax.set_title(
                f"LIME — Local explanation (sample {sample_index})\n"
                f"Uncertainty metric: {metric_text}",
                fontsize=10,
            )
            ax.set_xlabel(f"LIME coefficient (effect on {metric_text})", fontsize=9)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
        else:
            fnames = explanation.feature_names
            n = len(fnames)
            coefs = explanation.local_coefficients[sample_index]
            order = np.argsort(np.abs(coefs))
            labels = [fnames[i] for i in order]
            colors = ["tomato" if c >= 0 else "steelblue" for c in coefs[order]]
            fig, ax = plt.subplots(figsize=figsize)
            bars = ax.barh(labels, coefs[order], color=colors, edgecolor="black", linewidth=0.5)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"LIME coefficient (effect on {metric_text})")
            ax.set_title(f"LIME — Local explanation — Uncertainty metric ({metric_text})")
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
    max_ice_lines: int = 80,
    figsize: tuple[float, float] | None = None,
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

        Defaults to ``["pdp"]``.

    feature_names : list of str, optional
        Overrides ``explanation.feature_names`` for axis labels.

    max_ice_lines : int
        Maximum number of ICE lines to draw per feature. When there are
        more samples than this limit, a random subset is drawn to avoid
        overplotting. Default is 80.

    figsize : tuple of (width, height), optional
        Figure size in inches. When ``None``, scales with the number of
        features (~5 inches per column, 3.5 per row, 3-col grid).

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

    def _auto_figsize(n_panels: int) -> tuple[float, float]:
        if figsize is not None:
            return figsize
        ncols = min(3, max(n_panels, 1))
        nrows = int(np.ceil(n_panels / ncols))
        return (5 * ncols, 3.5 * nrows)

    use_native = (
        n > 0
        and getattr(explanation, "pd_results_raw", None) is not None
        and getattr(explanation, "deciles_", None) is not None
    )

    if use_native:
        fnames_display = [_fname(i) for i in range(n)]
        features_for_display = [(i,) for i in range(n)]

        def _make_display(kind="average", subsample=None):
            kwargs = dict(
                pd_results=explanation.pd_results_raw,
                features=features_for_display,
                feature_names=fnames_display,
                target_idx=0,
                deciles=explanation.deciles_,
                kind=kind,
            )
            if subsample is not None:
                kwargs["subsample"] = subsample
            return PartialDependenceDisplay(**kwargs)

        def _style_pdp_fig(fig, title):
            for ax in np.array(fig.axes).ravel():
                if ax is not None and ax.get_visible():
                    ax.set_ylabel(metric_name, fontsize=9)
                    ax.grid(True, linestyle="--", alpha=0.35)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
            fig.suptitle(title, y=1.01, fontsize=11)
            fig.tight_layout()

        # --- PDP ---
        if "pdp" in kinds:
            display = _make_display(kind="average")
            display.plot(
                n_cols=min(3, n),
                line_kw={"color": "tomato", "lw": 2},
            )
            display.figure_.set_size_inches(*_auto_figsize(n))
            _style_pdp_fig(
                display.figure_,
                f"Partial Dependence — Uncertainty metric ({metric_text})",
            )
            figures["pdp"] = display.figure_

        # --- ICE ---
        if "ice" in kinds and explanation.individual is not None:
            display = _make_display(kind="individual", subsample=max_ice_lines)
            display.plot(
                n_cols=min(3, n),
                ice_lines_kw={"color": "steelblue", "lw": 0.7, "alpha": 0.3},
            )
            display.figure_.set_size_inches(*_auto_figsize(n))
            _style_pdp_fig(
                display.figure_,
                f"ICE — Uncertainty metric ({metric_text})",
            )
            figures["ice"] = display.figure_

        # --- PDP + ICE overlay ---
        if "pdp_ice" in kinds and explanation.individual is not None:
            display = _make_display(kind="both", subsample=max_ice_lines)
            display.plot(
                n_cols=min(3, n),
                ice_lines_kw={"color": "steelblue", "lw": 0.7, "alpha": 0.25},
                pd_line_kw={"color": "tomato", "lw": 2.5, "label": "PDP"},
            )
            display.figure_.set_size_inches(*_auto_figsize(n))
            _style_pdp_fig(
                display.figure_,
                f"PDP + ICE — Uncertainty metric ({metric_text})",
            )
            figures["pdp_ice"] = display.figure_

    else:
        # Fallback: manual plots from stored values
        if "pdp" in kinds and n > 0:
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=_auto_figsize(n))
            axes = np.array(axes).flatten()
            for i in range(n):
                ax = axes[i]
                ax.plot(explanation.grid_values[i], explanation.values[i], lw=2, color="tomato")
                ax.set_xlabel(_fname(i))
                ax.set_ylabel(metric_name)
                ax.set_title(f"PDP — {_fname(i)}")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes[n:]:
                ax.set_visible(False)
            fig.suptitle(f"Partial Dependence — Uncertainty metric ({metric_text})", y=1.01)
            fig.tight_layout()
            figures["pdp"] = fig

        if "ice" in kinds and n > 0:
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=_auto_figsize(n))
            axes = np.array(axes).flatten()
            for i in range(n):
                ax = axes[i]
                if explanation.individual is None:
                    ax.set_title(f"ICE — {_fname(i)}\n(no individual data)")
                    continue
                lines = explanation.individual[i]
                if len(lines) > max_ice_lines:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(lines), max_ice_lines, replace=False)
                    lines = lines[idx]
                alpha = max(0.15, min(0.5, 30 / len(lines)))
                for line in lines:
                    ax.plot(explanation.grid_values[i], line, lw=0.7, alpha=alpha, color="steelblue")
                ax.set_xlabel(_fname(i))
                ax.set_ylabel(metric_name)
                ax.set_title(f"ICE — {_fname(i)}")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes[n:]:
                ax.set_visible(False)
            fig.suptitle(f"ICE — Uncertainty metric ({metric_text})", y=1.01)
            fig.tight_layout()
            figures["ice"] = fig

        if "pdp_ice" in kinds and n > 0:
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=_auto_figsize(n))
            axes = np.array(axes).flatten()
            for i in range(n):
                ax = axes[i]
                if explanation.individual is not None:
                    lines = explanation.individual[i]
                    if len(lines) > max_ice_lines:
                        rng = np.random.default_rng(0)
                        idx = rng.choice(len(lines), max_ice_lines, replace=False)
                        lines = lines[idx]
                    alpha = max(0.12, min(0.4, 25 / len(lines)))
                    for line in lines:
                        ax.plot(explanation.grid_values[i], line, lw=0.7, alpha=alpha, color="steelblue")
                ax.plot(
                    explanation.grid_values[i], explanation.values[i],
                    lw=3, color="tomato", label="PDP", zorder=5,
                )
                ax.set_xlabel(_fname(i))
                ax.set_ylabel(metric_name)
                ax.set_title(f"PDP + ICE — {_fname(i)}")
                ax.legend(fontsize=9, framealpha=0.85)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes[n:]:
                ax.set_visible(False)
            fig.suptitle(f"PDP + ICE — Uncertainty metric ({metric_text})", y=1.01)
            fig.tight_layout()
            figures["pdp_ice"] = fig

    # --- 2D PDP heatmaps — native sklearn PartialDependenceDisplay ---
    if "pdp_2d" in kinds:
        display_2d = getattr(explanation, "display_2d_", None)
        if display_2d is None:
            raise ValueError(
                'plot_kind="pdp_2d" requires the explainer to be fit with '
                "at least one feature pair as a tuple, e.g. "
                "features=[(0, 1)]. No 2D partial dependence was computed."
            )
        display_2d.plot(contour_kw={"cmap": "RdYlBu_r"})
        fig = display_2d.figure_
        n_pairs = len(getattr(explanation, "feature_pairs", None) or [])
        fig.set_size_inches(*_auto_figsize(n_pairs))
        fig.suptitle(
            f"2D Partial Dependence — Uncertainty metric ({metric_text})",
            y=1.01, fontsize=11,
        )
        fig.tight_layout()
        figures["pdp_2d"] = fig

    if show:
        plt.show()

    return figures


