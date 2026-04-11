"""
Visualization utilities for SHAP explanations.
"""

from __future__ import annotations

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