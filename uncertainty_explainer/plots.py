"""
Visualization utilities for SHAP explanations.
"""

from __future__ import annotations

import shap
import matplotlib.pyplot as plt


def generate_default_plots(
    shap_values,
    X,
    show: bool = True,
):
    """
    Generate standard SHAP plots.

    Includes:
    - summary plot
    - bar plot
    - beeswarm plot

    Parameters
    ----------
    shap_values
        SHAP explanation object.

    X : np.ndarray

    show : bool
        Whether to display plots.

    Returns
    -------
    dict
        Dictionary with matplotlib figures.
    """

    figures = {}

    # Summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
    )
    figures["summary"] = plt.gcf()

    # Bar plot
    plt.figure()
    shap.plots.bar(
        shap_values,
        show=False,
    )
    figures["bar"] = plt.gcf()

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        show=False,
    )
    figures["beeswarm"] = plt.gcf()

    if show:
        plt.show()

    return figures