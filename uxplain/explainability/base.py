"""
Common interface of the explanations returned by uxplain's explainers.

Every explainer returns a subclass of `UncertaintyExplanation`:
`ShapExplanation`, `LIMEExplanation` or `PDPExplanation`. They share

- ``method``: ``"shap"``, ``"lime"`` or ``"pdp"``;
- ``metric``: the explained conformal summary, e.g. ``"width"``;
- ``feature_names``;
- `feature_importance`, a global ranking;
- `to_frame`, the attributions as a table;
- `plot`, the method's standard figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


class UncertaintyExplanation:
    """Base class of SHAP, LIME and partial-dependence explanations.

    Subclasses hold the method-specific arrays; this class provides the shared
    accessors. Instances are created by the explainers, not by users.
    """

    #: Explanation method: ``"shap"``, ``"lime"`` or ``"pdp"``.
    method: ClassVar[str] = ""

    def feature_importance(
        self,
        n_boot: int = 0,
        level: float = 0.95,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Global importance of each feature for the explained summary.

        SHAP and LIME use the mean absolute attribution over the explained
        rows; partial dependence uses the standard deviation of the averaged
        curve.

        Parameters
        ----------
        n_boot : int, default=0
            Number of bootstrap resamples of explained rows. For partial
            dependence, requires ICE curves from ``explain(..., kind="both")``.
            Intervals describe variability over these rows only, excluding
            estimator refitting or redrawing calibration data.
        level : float, default=0.95
            Coverage of bootstrap percentile intervals, between zero and one.
        random_state : int, optional
            Seed for bootstrap sampling.

        Returns
        -------
        pandas.DataFrame
            Features sorted by decreasing ``importance``, with columns
            ``importance`` and ``rank``. When ``n_boot > 0``, also includes
            ``ci_lower``, ``ci_upper``, ``rank_lower`` and ``rank_upper``.
        """
        from ..evaluation import importance_table

        return importance_table(self, n_boot=n_boot, level=level,
                                random_state=random_state)

    def to_frame(self) -> pd.DataFrame:
        """Return the explanation as a table.

        Returns
        -------
        pandas.DataFrame
            Method-specific values: attributions for SHAP, local coefficients
            for LIME, or grid values and averaged curves for partial dependence.
            See each subclass for its column layout.
        """
        raise NotImplementedError

    def plot(self, kind=None, show: bool = True, **kwargs) -> dict:
        """Draw the method's standard figures.

        Parameters
        ----------
        kind : str or list of str, optional
            Plot kinds supported by the concrete explanation. ``None`` selects
            the method's default figures.
        show : bool, default=True
            Display the figures. Set to ``False`` to save or style them first.
        **kwargs : dict
            Options forwarded to the method's figure builder, such as
            ``figsize`` or a sample index for a local explanation.

        Returns
        -------
        dict
            Mapping from plot kind to ``matplotlib.figure.Figure``.
        """
        raise NotImplementedError

    def _describe(self) -> str:
        return ""

    def __repr__(self) -> str:
        details = self._describe()
        return f"{type(self).__name__}(metric={self.metric!r}{', ' + details if details else ''})"


def _feature_labels(names, n_features: int) -> list[str]:
    if names is None:
        return [f"Feature {j}" for j in range(n_features)]
    return [str(name) for name in names]


@dataclass(repr=False)
class ShapExplanation(UncertaintyExplanation):
    """Shapley values of a conformal summary for the explained rows.

    Attributes
    ----------
    values : np.ndarray of shape (n_samples, n_features)
        Attribution of each feature to the summary of each row.
    base_values : np.ndarray of shape (n_samples,)
        Mean summary over the background rows; ``base_values + values.sum(1)``
        reproduces the explained summary.
    data : np.ndarray of shape (n_samples, n_features), optional
        Feature values of the explained rows.
    feature_names : list of str, optional
        Names of the columns in ``values`` and ``data``.
    metric : str
        The explained summary, e.g. ``"width"`` or ``"set_size"``.
    algorithm : str, optional
        ``"tree"`` for the exact TreeSHAP shortcut, otherwise the SHAP
        algorithm used on the composed function (``"exact"``, ``"permutation"``).
    """

    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray | None = None
    feature_names: list[str] | None = None
    metric: str = "width"
    algorithm: str | None = None

    method: ClassVar[str] = "shap"

    def __len__(self) -> int:
        return len(np.atleast_2d(self.values))

    def to_frame(self) -> pd.DataFrame:
        """Return one row per explained observation.

        Returns
        -------
        pandas.DataFrame
            One attribution column per feature, followed by ``base_value``.
            Unnamed features are labelled ``Feature 0``, ``Feature 1``, etc.
        """
        values = np.atleast_2d(np.asarray(self.values, dtype=float))
        frame = pd.DataFrame(values, columns=_feature_labels(self.feature_names,
                                                             values.shape[1]))
        frame["base_value"] = np.broadcast_to(np.asarray(self.base_values, dtype=float),
                                              len(frame))
        return frame

    def to_shap(self):
        """Convert to the object accepted by SHAP's own plotting functions.

        Returns
        -------
        shap.Explanation
            The same attributions, base values, data and feature names.
        """
        import shap

        return shap.Explanation(values=np.asarray(self.values),
                                base_values=np.asarray(self.base_values),
                                data=None if self.data is None else np.asarray(self.data),
                                feature_names=self.feature_names)

    def plot(self, kind=None, show: bool = True, **kwargs) -> dict:
        """SHAP figures: ``"bar"``, ``"beeswarm"`` and ``"waterfall"``.

        Parameters
        ----------
        kind : str or list of str, optional
            Any of ``"bar"``, ``"beeswarm"``, ``"waterfall"`` or ``"summary"``
            (an alias of ``"beeswarm"``). Defaults to the first three kinds.
        show : bool, default=True
            Display the figures. Set to ``False`` to save or style them first.
        **kwargs : dict
            Options for ``uxplain.plots.generate_shap_plots``, including
            ``waterfall_index``, ``max_display`` and ``figsize``.

        Returns
        -------
        dict
            Mapping from plot kind to ``matplotlib.figure.Figure``.
        """
        from ..plots import generate_shap_plots

        kinds = [kind] if isinstance(kind, str) else kind
        return generate_shap_plots(self, kinds=kinds, show=show, **kwargs)

    def _describe(self) -> str:
        values = np.atleast_2d(self.values)
        text = f"rows={values.shape[0]}, features={values.shape[1]}"
        return text + (f", algorithm={self.algorithm!r}" if self.algorithm else "")
