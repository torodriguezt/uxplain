"""
Plotting primitives for uncertainty explanations.

Every figure in uxplain is drawn here with plain matplotlib from the
raw arrays produced by the explainers — SHAP values, partial-dependence grids,
LIME coefficients. Nothing is delegated to the plotting layers of ``shap``,
``sklearn.inspection`` or ``lime``, so the caller owns every artist.

Each primitive follows the same contract:

- it accepts an existing ``ax`` (and draws into it) or creates its own figure,
- it returns the ``Axes`` it drew on,
- it takes styling from `STYLE`, which can be mutated globally or
  overridden per call.

That makes the primitives composable:

```python
import matplotlib.pyplot as plt
from uxplain import shap_bar, shap_beeswarm

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
shap_bar(explanation, ax=axes[0])
shap_beeswarm(explanation, ax=axes[1])
```
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty.metrics import metric_label

__all__ = [
    "STYLE",
    "ice_curves",
    "importance_bar",
    "lime_local",
    "pdp_curve",
    "pdp_importance",
    "pdp_interaction",
    "pdp_with_ice",
    "shap_bar",
    "shap_beeswarm",
    "shap_waterfall",
]


#: Global styling. Mutate to restyle every figure at once, or pass the same
#: keys per call to override a single plot. See `use_theme` for the
#: bundled presets.
STYLE: dict[str, Any] = {
    # --- data ink -----------------------------------------------------------
    # One base tonality carries everything. The contrasting hue is spent only
    # where it distinguishes something the reader could not otherwise see:
    # marks that pull the metric in the opposite direction. When every mark in
    # a plot shares a sign, nothing is being distinguished, so the plot stays
    # entirely in the base tone (see `adaptive_sign_color`).
    "magnitude_color": "#3d5a73",  # the base tone
    "positive_color": "#3d5a73",   # same as base: the common direction
    "negative_color": "#b2493f",   # the contrasting tone
    "adaptive_sign_color": True,   # drop the contrast when signs are uniform
    "pdp_color": "#3d5a73",        # a lone line needs no second hue
    "ice_color": "#9aa5ae",        # individual conditional expectation lines
    # Sequential ramp for a continuous feature value: one hue, light -> dark.
    "colormap": ("#dce6ef", "#7ba0c0", "#1f4e79"),
    # --- chrome -------------------------------------------------------------
    "title_color": "#1a1a1a",
    "label_color": "#3f3f3f",
    "tick_color": "#5c5c5c",
    "grid_color": "#dcdcd8",
    "grid_linestyle": "-",
    "grid_linewidth": 0.8,
    "grid_alpha": 1.0,
    "baseline_color": "#b0b0ac",
    "reference_color": "#8c8c88",  # zero line, baseline marker
    # --- typography ---------------------------------------------------------
    "title_size": 17,
    "label_size": 15,
    "tick_size": 13,
    "legend_size": 13,
    "annotation_size": 13,
    "title_weight": "normal",
    "title_loc": "left",
    # --- marks --------------------------------------------------------------
    "point_size": 18,
    "point_alpha": 0.8,
    "bar_edge_color": "none",
    "bar_height": 0.62,
    "line_width": 2.2,
}


#: Bundled presets, applied with `use_theme`.
THEMES: dict[str, dict[str, Any]] = {
    # Austere and print-first: no hue at all, so the figure survives grayscale
    # reproduction. Opposing marks take a lighter step of the same gray.
    "mono": {
        "magnitude_color": "#4a4a4a",
        "positive_color": "#4a4a4a",
        "negative_color": "#a5a5a5",
        "pdp_color": "#2b2b2b",
        "ice_color": "#c4c4c4",
        "colormap": ("#e8e8e8", "#9a9a9a", "#2b2b2b"),
    },
    # One muted slate carries every figure; a terracotta appears only on marks
    # that pull the metric the other way. Colorblind-safe.
    "diverging": {
        "magnitude_color": "#3d5a73",
        "positive_color": "#3d5a73",
        "negative_color": "#b2493f",
        "pdp_color": "#3d5a73",
        "ice_color": "#9aa5ae",
        "colormap": ("#dce6ef", "#7ba0c0", "#1f4e79"),
    },
    # A single editorial hue throughout; opposing marks take a lighter step of
    # that same hue rather than a second one.
    "editorial": {
        "magnitude_color": "#1f4e79",
        "positive_color": "#1f4e79",
        "negative_color": "#9db8cd",
        "pdp_color": "#1f4e79",
        "ice_color": "#c2ccd4",
        "colormap": ("#e4ebf1", "#89a8c2", "#173d5e"),
    },
}


def use_theme(name: str, **overrides) -> dict:
    """Apply a bundled theme to `STYLE` in place.

    Parameters
    ----------
    name : {"mono", "diverging", "editorial"}
        Preset to apply. ``"mono"`` is grayscale-safe, ``"diverging"`` reserves
        hue for the sign of a contribution, ``"editorial"`` keeps a single hue.
    **overrides : object
        Extra style keys applied on top of the preset.

    Returns
    -------
    dict
        The updated global `STYLE` dictionary, returned by reference.
    """
    if name not in THEMES:
        raise ValueError(
            f"Unknown theme '{name}'. Choose from {sorted(THEMES)}."
        )
    STYLE.update(THEMES[name])
    STYLE.update(overrides)
    return STYLE


def _style(overrides: dict | None = None) -> dict:
    """Merge per-call overrides over the global style."""
    if not overrides:
        return dict(STYLE)
    merged = dict(STYLE)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _feature_cmap(style: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("uxplain", list(style["colormap"]))


def _new_ax(ax, figsize):
    """Return ``(fig, ax, created)``; create a figure only when ax is None."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _finish(ax, style: dict, *, title=None, xlabel=None, ylabel=None,
            grid_axis="x") -> None:
    """Apply the shared look: recessive chrome, muted ink, hairline grid."""
    if title:
        ax.set_title(title, fontsize=style["title_size"],
                     color=style["title_color"], loc=style["title_loc"],
                     fontweight=style["title_weight"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style["label_size"],
                      color=style["label_color"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style["label_size"],
                      color=style["label_color"])

    ax.tick_params(labelsize=style["tick_size"], colors=style["tick_color"],
                   length=0 if grid_axis else 3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(style["tick_color"])

    if grid_axis:
        ax.grid(True, axis=grid_axis, color=style["grid_color"],
                linestyle=style["grid_linestyle"],
                linewidth=style["grid_linewidth"],
                alpha=style["grid_alpha"], zorder=0)
        ax.set_axisbelow(True)

    # Keep only the axis the categories sit on; the grid carries the rest.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style["baseline_color"])
        ax.spines[side].set_linewidth(0.9)


def _resolve_names(explanation, feature_names, n_features) -> list[str]:
    names = (feature_names if feature_names is not None
             else getattr(explanation, "feature_names", None))
    if names is None:
        return [f"Feature {i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names must match the number of feature columns.")
    return list(names)


def _metric_text(explanation, metric=None) -> str:
    return metric_label(metric or getattr(explanation, "metric", "width"))


def _unpack_shap(explanation):
    """Extract ``(values, data, base_values)`` from a SHAP explanation."""
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("SHAP values must have nonempty sample and feature dimensions.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite to plot contributions.")
    data = getattr(explanation, "data", None)
    if data is not None:
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != values.shape:
            raise ValueError("Feature data must have the same shape as SHAP values.")
    base = getattr(explanation, "base_values", None)
    if base is not None:
        base = np.asarray(base, dtype=float)
    return values, data, base


def _top_features(importance: np.ndarray, max_display: int | None):
    """Return indices of the ``max_display`` most important features, ascending."""
    order = np.argsort(importance)
    _validate_max_display(max_display)
    if max_display is not None and max_display < len(order):
        order = order[-max_display:]
    return order


def _validate_max_display(max_display):
    if max_display is not None and (
        isinstance(max_display, bool)
        or not isinstance(max_display, Integral)
        or max_display < 1
    ):
        raise ValueError("max_display must be a positive integer or None.")


def _decimals(values) -> int:
    """Decimals that show the largest magnitude with three significant digits."""
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 2
    return int(np.clip(2 - np.floor(np.log10(values.max())), 0, 6))


def _format_number(value: float) -> str:
    """Compact label for a feature value: no scientific notation below 1e9."""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _sign_colors(values, style: dict) -> list[str]:
    """Colour signed marks, spending the contrasting hue only when it earns it.

    If every value points the same way there is nothing for a second hue to
    distinguish, so the whole plot stays in the base tone and matches the
    unsigned figures beside it.
    """
    values = np.asarray(values, dtype=float)
    mixed = (values > 0).any() and (values < 0).any()
    if style["adaptive_sign_color"] and not mixed:
        return [style["magnitude_color"]] * len(values)
    return [style["positive_color"] if v >= 0 else style["negative_color"]
            for v in values]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_bar(
    explanation,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    show_values: bool = True,
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Mean absolute SHAP contribution per feature, as a horizontal bar chart.

    Parameters
    ----------
    explanation : ShapExplanation or shap.Explanation
        Object exposing finite `values` of shape `(n_samples, n_features)`
        or `(n_features,)` for a single row.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_names : sequence of str, optional
        One label per feature. Defaults to the explanation's feature names,
        or `Feature 0`, `Feature 1`, and so on when none are available.
    max_display : int, optional
        Positive number of highest-ranked features to display. `None` shows
        all features. Omitted features are not aggregated into another bar.
    figsize : tuple of float, default=(9, 5)
        Figure width and height in inches, used only when creating a figure.
    show_values : bool, default=True
        Annotate each bar with its numeric value.
    metric : str, optional
        Uncertainty metric used for the title. Defaults to the explanation's
        metric, or `"width"` if it has no metric attribute.
    title : str, optional
        Title override. `None` generates a title from the metric; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, such as `magnitude_color` or
        `annotation_size`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the chart, with the largest importance at the top.
    """
    style = _style(style_overrides)
    values, _, _ = _unpack_shap(explanation)
    if values.ndim == 1:
        values = values[None, :]
    names = _resolve_names(explanation, feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = _top_features(importance, max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    bars = ax.barh(y, importance[order], color=style["magnitude_color"],
                   edgecolor=style["bar_edge_color"],
                   height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])

    if show_values:
        ax.bar_label(bars, fmt=f"%.{_decimals(importance)}f", padding=5,
                     fontsize=style["annotation_size"],
                     color=style["label_color"])
        ax.set_xlim(0, max(importance[order].max() * 1.18, 1e-9))

    _finish(ax, style,
            title=title if title is not None
            else f"Mean absolute contribution to {_metric_text(explanation, metric).lower()}",
            xlabel="mean(|SHAP value|)")
    fig.tight_layout()
    return ax


def importance_bar(
    table,
    ax=None,
    max_display: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    show_values: bool = True,
    xlabel: str = "Importance",
    title: str | None = None,
    **style_overrides,
):
    """Global importance table as a horizontal bar chart.

    Parameters
    ----------
    table : pandas.DataFrame
        Nonempty table from `uxplain.evaluation.importance_table` or a result's
        `feature_importance()`, indexed by feature with an `importance` column.
        If both `ci_lower` and `ci_upper` are present, they are drawn as error
        bars and must bracket the corresponding importance estimate.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    max_display : int, optional
        Positive number of features with the largest importance to display.
        `None` shows all features; omitted features are not aggregated.
    figsize : tuple of float, default=(9, 5)
        Figure width and height in inches, used only when creating a figure.
    show_values : bool, default=True
        Annotate importance estimates at the bar ends, or beyond the upper
        error-bar endpoints when interval columns are present.
    xlabel : str, default="Importance"
        Horizontal-axis label.
    title : str, optional
        Plot title. No title is added when omitted or empty.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the chart, with the largest importance at the top.
    """
    style = _style(style_overrides)
    importance = np.asarray(table["importance"], dtype=float)
    names = [str(name) for name in table.index]
    order = _top_features(importance, max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    ax.barh(y, importance[order], color=style["magnitude_color"],
            edgecolor=style["bar_edge_color"],
            height=style["bar_height"], zorder=3)
    upper_end = importance[order]
    if "ci_lower" in table and "ci_upper" in table:
        lower = np.asarray(table["ci_lower"], dtype=float)[order]
        upper = np.asarray(table["ci_upper"], dtype=float)[order]
        ax.errorbar(importance[order], y,
                    xerr=[importance[order] - lower, upper - importance[order]],
                    fmt="none", ecolor=style["title_color"], elinewidth=1.1,
                    capsize=3, zorder=4)
        upper_end = upper
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])

    if show_values:
        decimals = _decimals(importance)
        for yi, (end, value) in enumerate(zip(upper_end, importance[order])):
            ax.text(end, yi, f"  {value:.{decimals}f}", va="center", ha="left",
                    fontsize=style["annotation_size"], color=style["label_color"])
    ax.set_xlim(0, max(np.max(upper_end) * 1.22, 1e-9))

    _finish(ax, style, title=title or "", xlabel=xlabel)
    fig.tight_layout()
    return ax


def _swarm_offsets(vals: np.ndarray, n_bins: int = 100,
                   row_height: float = 0.4) -> np.ndarray:
    """Vertical offsets that spread overlapping points into a beeswarm.

    Points are bucketed along the x axis; within a bucket they are stacked
    outwards, alternating above and below the row centre.
    """
    n = len(vals)
    if n == 0:
        return np.zeros(0)
    span = vals.max() - vals.min()
    if span <= 0:
        return np.zeros(n)

    quant = np.round(n_bins * (vals - vals.min()) / span)
    order = np.argsort(quant + np.random.RandomState(0).randn(n) * 1e-6)

    offsets = np.zeros(n)
    layer, last_bin = 0, -1
    for idx in order:
        if quant[idx] != last_bin:
            layer = 0
        # 0, +1, -1, +2, -2, ... away from the row centre
        offsets[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[idx]

    peak = np.abs(offsets).max()
    if peak > 0:
        offsets *= 0.9 * (row_height / peak)
    return offsets


def shap_beeswarm(
    explanation,
    X=None,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int | None = None,
    figsize: tuple[float, float] = (9, 5),
    colorbar: bool = True,
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Per-instance SHAP contributions, one swarm per feature.

    Point colour encodes the feature's own value, so the plot shows both how
    much a feature moves the metric and in which direction high/low values push.

    Parameters
    ----------
    explanation : ShapExplanation or shap.Explanation
        Object exposing finite `values` of shape `(n_samples, n_features)`
        or `(n_features,)` for a single row. Its optional `data` supplies
        feature values for colouring when `X` is not provided.
    X : array-like, optional
        Feature values with the same shape as the SHAP values. Overrides
        `explanation.data`. Colours are scaled separately for each feature
        using its 5th and 95th percentiles. Without feature data, all points
        use the same colour.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_names : sequence of str, optional
        One label per feature. Defaults to the explanation's names or numbered
        labels starting at `Feature 0`.
    max_display : int, optional
        Positive number of features with the largest mean absolute SHAP value
        to display. `None` shows all features; omitted features are not pooled.
    figsize : tuple of float, default=(9, 5)
        Figure width and height in inches, used only when creating a figure.
    colorbar : bool, default=True
        Add a low-to-high feature-value colour bar when feature data exist.
    metric : str, optional
        Uncertainty metric used for axis and title text. Defaults to the
        explanation's metric, or `"width"` if it has no metric attribute.
    title : str, optional
        Title override. `None` generates a title from the metric; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `colormap`,
        `point_size` and `point_alpha`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Main plot axes, excluding the optional colour-bar axes.
    """
    style = _style(style_overrides)
    values, data, _ = _unpack_shap(explanation)
    if X is not None:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != values.shape:
            raise ValueError("X must have the same shape as SHAP values.")
    names = _resolve_names(explanation, feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = _top_features(importance, max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    cmap = _feature_cmap(style)

    for row, feat in enumerate(order):
        v = values[:, feat]
        ys = row + _swarm_offsets(v)
        if data is not None:
            col = data[:, feat]
            # Robust scaling so a few outliers don't flatten the colour range.
            lo, hi = np.nanpercentile(col, [5, 95])
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
            ax.scatter(v, ys, c=col, cmap=cmap, norm=norm,
                       s=style["point_size"], linewidths=0,
                       alpha=style["point_alpha"], zorder=3)
        else:
            ax.scatter(v, ys, color=style["magnitude_color"],
                       s=style["point_size"], linewidths=0,
                       alpha=style["point_alpha"], zorder=3)

    ax.axvline(0, color=style["reference_color"], linewidth=0.9, zorder=2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_ylim(-0.6, len(order) - 0.4)

    if colorbar and data is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=32, fraction=0.035)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["low", "high"])
        cbar.ax.tick_params(labelsize=style["tick_size"], length=0,
                            colors=style["tick_color"])
        cbar.set_label("Feature value", fontsize=style["label_size"],
                       color=style["label_color"])
        cbar.outline.set_visible(False)

    metric_txt = _metric_text(explanation, metric).lower()
    _finish(ax, style,
            title=title if title is not None
            else f"Per-instance contribution to {metric_txt}",
            xlabel=f"SHAP value (effect on {metric_txt})",
            grid_axis="x")
    fig.tight_layout()
    return ax


def shap_waterfall(
    explanation,
    index: int = 0,
    ax=None,
    feature_names: Sequence[str] | None = None,
    max_display: int = 10,
    figsize: tuple[float, float] = (10, 6),
    metric: str | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Additive decomposition of a single instance's metric value.

    Bars run cumulatively from the baseline ``E[f(X)]`` to the instance's
    ``f(x)``; features are ordered by absolute contribution and any remainder is
    collapsed into a single "other features" bar.

    Parameters
    ----------
    explanation : ShapExplanation or shap.Explanation
        Object exposing finite SHAP `values` of shape
        `(n_samples, n_features)` and finite `base_values` as a scalar or one
        value per row. A one-dimensional values array represents a single row.
        Optional `data` with the same shape adds feature values to the labels.
    index : int, default=0
        Row of the explanation to display.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_names : sequence of str, optional
        One label per feature. Defaults to the explanation's names or numbered
        labels starting at `Feature 0`.
    max_display : int, default=10
        Positive number of individual features to display, selected by absolute
        contribution. If more features exist, one additional bar contains the
        signed sum of their contributions, preserving the additive total.
    figsize : tuple of float, default=(10, 6)
        Figure width and height in inches, used only when creating a figure.
    metric : str, optional
        Uncertainty metric used for axis and title text. Defaults to the
        explanation's metric, or `"width"` if it has no metric attribute.
    title : str, optional
        Title override. `None` generates a title from the metric and index;
        `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `positive_color`,
        `negative_color` and `adaptive_sign_color`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the contributions and annotated baseline and total.
    """
    style = _style(style_overrides)
    values, data, base = _unpack_shap(explanation)
    if values.ndim == 1:
        values = values[None, :]
    names = _resolve_names(explanation, feature_names, values.shape[1])

    phi = values[index]
    if base is None:
        raise ValueError("A SHAP waterfall requires base_values for its additive baseline.")
    base_value = float(base[index]) if base.ndim else float(base)
    if not np.isfinite(base_value):
        raise ValueError("A SHAP waterfall requires a finite baseline.")
    row = data[index] if data is not None else None

    _validate_max_display(max_display)
    order = np.argsort(np.abs(phi))[::-1]
    shown, hidden = order[:max_display], order[max_display:]

    labels, contribs = [], []
    for feat in shown:
        label = names[feat]
        if row is not None:
            label = f"{_format_number(row[feat])} = {label}"
        labels.append(label)
        contribs.append(phi[feat])
    if len(hidden):
        labels.append(f"{len(hidden)} other features")
        contribs.append(float(phi[hidden].sum()))

    contribs = np.asarray(contribs)
    # Draw smallest contribution at the top of the axis, largest at the bottom.
    labels, contribs = labels[::-1], contribs[::-1]

    starts = base_value + np.concatenate([[0.0], np.cumsum(contribs)[:-1]])
    total = base_value + contribs.sum()

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(contribs))
    colors = _sign_colors(contribs, style)
    ax.barh(y, contribs, left=starts, color=colors,
            edgecolor=style["bar_edge_color"],
            height=style["bar_height"], zorder=3)

    span = max(np.abs(contribs).max(), abs(total - base_value), 1e-9)
    pad = 0.02 * span
    decimals = _decimals([*contribs, base_value, total])
    for yi, (start, c) in enumerate(zip(starts, contribs)):
        tip = start + c
        ax.text(tip + (pad if c >= 0 else -pad), yi, f"{c:+.{decimals}f}",
                va="center", ha="left" if c >= 0 else "right",
                fontsize=style["annotation_size"], color=style["label_color"])

    ax.axvline(base_value, color=style["reference_color"],
               linestyle="--", linewidth=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Reserve a header band inside the axes for the baseline/outcome labels, so
    # they never collide with the title above or the tick labels below.
    header = len(contribs) - 0.3 + 0.7
    ax.set_ylim(-0.7, header)

    metric_txt = _metric_text(explanation, metric)
    _finish(ax, style,
            title=title if title is not None
            else f"How {metric_txt.lower()} is built up for sample {index}",
            xlabel=metric_txt)

    label_y = header - 0.25
    ax.text(base_value, label_y, f"$E[f(X)]$ = {base_value:.{decimals}f}",
            ha="center", va="center", fontsize=style["annotation_size"],
            color=style["tick_color"])
    ax.text(total, label_y, f"$f(x)$ = {total:.{decimals}f}",
            ha="center", va="center", fontsize=style["annotation_size"],
            color=style["title_color"])

    lo = min(starts.min(), base_value, total)
    hi = max((starts + contribs).max(), base_value, total)
    ax.set_xlim(lo - 0.22 * span, hi + 0.22 * span)
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

def pdp_curve(
    grid: np.ndarray,
    values: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    title: str | None = None,
    label: str | None = "PDP",
    **style_overrides,
):
    """Plot the averaged partial-dependence curve for one feature.

    Parameters
    ----------
    grid : numpy.ndarray of shape (n_grid,)
        Feature values at which partial dependence was evaluated.
    values : numpy.ndarray of shape (n_grid,)
        Average uncertainty summary at each grid value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_name : str, default=""
        Feature label for the horizontal axis and default title.
    figsize : tuple of float, default=(8, 6)
        Figure width and height in inches, used only when creating a figure.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the vertical axis and title.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    label : str, optional
        Label attached to the plotted line, default `"PDP"`. Call `ax.legend()`
        to display a legend; this function does not add one automatically.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, such as `pdp_color` and
        `line_width`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the averaged curve.
    """
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)
    ax.plot(grid, values, color=style["pdp_color"],
            lw=style["line_width"], label=label, zorder=5)
    _finish(ax, style,
            title=title if title is not None
            else f"Average {metric_name.lower()} across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def ice_curves(
    grid: np.ndarray,
    individual: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    max_lines: int = 80,
    title: str | None = None,
    **style_overrides,
):
    """Plot individual conditional expectation curves, one per observation.

    Parameters
    ----------
    grid : numpy.ndarray of shape (n_grid,)
        Feature values at which the individual curves were evaluated.
    individual : numpy.ndarray of shape (n_samples, n_grid)
        Uncertainty summary for each observation at every grid value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_name : str, default=""
        Feature label for the horizontal axis and default title.
    figsize : tuple of float, default=(8, 6)
        Figure width and height in inches, used only when creating a figure.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the vertical axis and title.
    max_lines : int, default=80
        Maximum number of curves to draw. If there are more observations, a
        reproducible subset is sampled without replacement using a fixed seed.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `ice_color`.
        Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the selected individual curves.
    """
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)

    lines = np.asarray(individual)
    if len(lines) > max_lines:
        idx = np.random.RandomState(0).choice(len(lines), max_lines, replace=False)
        lines = lines[idx]
    alpha = float(np.clip(25 / max(len(lines), 1), 0.10, 0.45))
    for line in lines:
        ax.plot(grid, line, color=style["ice_color"], lw=0.7,
                alpha=alpha, zorder=3)

    _finish(ax, style,
            title=title if title is not None
            else f"{metric_name} per observation across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def pdp_with_ice(
    grid: np.ndarray,
    values: np.ndarray,
    individual: np.ndarray,
    ax=None,
    feature_name: str = "",
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    max_lines: int = 80,
    title: str | None = None,
    **style_overrides,
):
    """Plot ICE curves with the averaged partial-dependence curve on top.

    Parameters
    ----------
    grid : numpy.ndarray of shape (n_grid,)
        Feature values shared by the average and individual curves.
    values : numpy.ndarray of shape (n_grid,)
        Average uncertainty summary at each grid value.
    individual : numpy.ndarray of shape (n_samples, n_grid)
        Uncertainty summary for each observation at every grid value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_name : str, default=""
        Feature label for the horizontal axis and default title.
    figsize : tuple of float, default=(8, 6)
        Figure width and height in inches, used only when creating a figure.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the vertical axis and title.
    max_lines : int, default=80
        Maximum number of individual curves, reproducibly sampled without
        replacement if needed. The supplied average curve is not recomputed.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `ice_color`,
        `pdp_color` and `line_width`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing both kinds of curves and a legend for the average.
    """
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)
    ice_curves(grid, individual, ax=ax, feature_name=feature_name,
               metric_name=metric_name, max_lines=max_lines,
               title="", **style_overrides)
    ax.plot(grid, values, color=style["pdp_color"],
            lw=style["line_width"] + 0.8, label="Average", zorder=6)
    legend = ax.legend(fontsize=style["legend_size"], frameon=False,
                       loc="best")
    for text in legend.get_texts():
        text.set_color(style["label_color"])
    _finish(ax, style,
            title=title if title is not None
            else f"Average {metric_name.lower()} across {feature_name}",
            xlabel=feature_name, ylabel=metric_name, grid_axis="both")
    fig.tight_layout()
    return ax


def pdp_importance(
    importance: Sequence[float],
    feature_names: Sequence[str],
    ax=None,
    figsize: tuple[float, float] = (8, 5),
    metric_name: str = "Interval width",
    title: str | None = None,
    **style_overrides,
):
    """Plot global PDP importance: the spread of each feature's PDP curve.

    Parameters
    ----------
    importance : sequence of float
        Nonempty sequence with one importance value per feature, computed as
        the standard deviation of its averaged partial-dependence curve.
        This function plots the supplied values without recomputing them.
    feature_names : sequence of str
        One label per feature, in the same order as `importance`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    figsize : tuple of float, default=(8, 5)
        Figure width and height in inches, used only when creating a figure.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the axis and default title.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, such as `magnitude_color` and
        `annotation_size`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Horizontal bar-chart axes, ordered with the largest importance at the
        top and annotated with values to three decimal places.
    """
    style = _style(style_overrides)
    imp = np.asarray(importance, dtype=float)
    order = np.argsort(imp)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    bars = ax.barh(y, imp[order], color=style["magnitude_color"],
                   edgecolor=style["bar_edge_color"],
                   height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=style["annotation_size"],
                 color=style["label_color"])
    ax.set_xlim(0, max(imp.max() * 1.18, 1e-9))

    _finish(ax, style,
            title=title if title is not None
            else f"How strongly each feature moves {metric_name.lower()}",
            xlabel=f"PDP importance (std of the {metric_name.lower()} curve)")
    fig.tight_layout()
    return ax


def pdp_interaction(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    ax=None,
    feature_names: tuple[str, str] = ("", ""),
    figsize: tuple[float, float] = (8, 6),
    metric_name: str = "Interval width",
    levels: int = 14,
    title: str | None = None,
    **style_overrides,
):
    """Plot two-way partial dependence as a filled contour map.

    Parameters
    ----------
    grid_x : numpy.ndarray of shape (n_grid_x,)
        Grid for the first feature, shown on the horizontal axis.
    grid_y : numpy.ndarray of shape (n_grid_y,)
        Grid for the second feature, shown on the vertical axis.
    values : numpy.ndarray of shape (n_grid_x, n_grid_y)
        Averaged uncertainty summary at each feature combination. The first
        dimension corresponds to `grid_x`; the array is transposed for plotting.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    feature_names : tuple of str, default=("", "")
        Labels for the horizontal and vertical axes, respectively.
    figsize : tuple of float, default=(8, 6)
        Figure width and height in inches, used only when creating a figure.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the colour bar and title.
    levels : int, default=14
        Level count passed to Matplotlib's `contourf`. Ignored when a feature
        has a one-point grid: that case uses a colour mesh instead of contours.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `colormap`.
        Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Main plot axes, excluding the colour-bar axes.
    """
    style = _style(style_overrides)
    fig, ax, _ = _new_ax(ax, figsize)

    # partial_dependence returns (len(grid_x), len(grid_y)); transpose for
    # contourf, which expects (n_rows=y, n_cols=x).
    Z = np.asarray(values).T
    if len(grid_x) < 2 or len(grid_y) < 2:
        # A constant feature yields a one-point grid, which contourf rejects.
        cs = ax.pcolormesh(grid_x, grid_y, Z, shading="nearest", cmap=_feature_cmap(style))
    else:
        cs = ax.contourf(grid_x, grid_y, Z, levels=levels, cmap=_feature_cmap(style))
    cbar = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(metric_name, fontsize=style["label_size"],
                   color=style["label_color"])
    cbar.ax.tick_params(labelsize=style["tick_size"], length=0,
                        colors=style["tick_color"])
    cbar.outline.set_visible(False)

    _finish(ax, style,
            title=title if title is not None
            else f"{metric_name} across {feature_names[0]} and {feature_names[1]}",
            xlabel=feature_names[0], ylabel=feature_names[1], grid_axis=None)
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def lime_local(
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    ax=None,
    figsize: tuple[float, float] = (9, 5),
    max_display: int | None = None,
    metric_name: str = "Interval width",
    sample_index: int | None = None,
    title: str | None = None,
    **style_overrides,
):
    """Plot local LIME surrogate coefficients for a single instance.

    Parameters
    ----------
    coefficients : numpy.ndarray of shape (n_features,)
        Signed local surrogate coefficients for one observation.
    feature_names : sequence of str
        One label per coefficient, in the same order as `coefficients`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    figsize : tuple of float, default=(9, 5)
        Figure width and height in inches, used only when creating a figure.
    max_display : int, optional
        Positive number of features with the largest absolute coefficients to
        display. `None` shows all features; omitted coefficients are not pooled.
    metric_name : str, default="Interval width"
        Human-readable uncertainty summary label for the axis and default title.
    sample_index : int, optional
        Observation identifier appended to the default title. It does not
        select data: `coefficients` must already contain the desired row.
    title : str, optional
        Title override. `None` generates a title; `""` omits it.
    **style_overrides : object
        Per-call overrides for keys in `STYLE`, including `positive_color`,
        `negative_color` and `adaptive_sign_color`. Values of `None` are ignored.

    Returns
    -------
    matplotlib.axes.Axes
        Horizontal bar-chart axes with the largest absolute coefficient at
        the top and bar directions preserving the coefficient signs.
    """
    style = _style(style_overrides)
    coefs = np.asarray(coefficients, dtype=float)
    order = _top_features(np.abs(coefs), max_display)

    fig, ax, _ = _new_ax(ax, figsize)
    y = np.arange(len(order))
    colors = _sign_colors(coefs[order], style)
    ax.barh(y, coefs[order], color=colors,
            edgecolor=style["bar_edge_color"],
            height=style["bar_height"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.axvline(0, color=style["reference_color"], linewidth=0.9, zorder=2)

    if title is None:
        title = f"Local drivers of {metric_name.lower()}"
        if sample_index is not None:
            title += f" for sample {sample_index}"
    _finish(ax, style, title=title,
            xlabel=f"LIME coefficient (effect on {metric_name.lower()})")
    fig.tight_layout()
    return ax
