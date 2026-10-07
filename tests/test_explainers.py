"""Reproducibility checks for stochastic explanations."""

import numpy as np

from uxplain import ShapUncertaintyExplainer


class InteractionPredictor:
    def predict(self, X, confidence=0.9):
        X = np.asarray(X)
        return np.zeros(len(X)), np.exp(X.sum(axis=1))


def test_permutation_seed_is_independent_of_other_explainers_and_numpy():
    data = np.random.default_rng(11).normal(scale=0.2, size=(6, 4))
    first = ShapUncertaintyExplainer(
        InteractionPredictor(), algorithm="permutation", random_state=4,
    )
    second = ShapUncertaintyExplainer(
        InteractionPredictor(), algorithm="permutation", random_state=4,
    )
    first.fit(data[:4])
    second.fit(data[:4])
    values = first.explain(data[4:]).values
    np.random.normal(size=100)
    np.testing.assert_array_equal(values, second.explain(data[4:]).values)


def test_shap_does_not_change_numpy_global_random_state():
    previous = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        explainer = ShapUncertaintyExplainer(
            InteractionPredictor(), algorithm="permutation", random_state=4,
        )
        explainer.fit(np.array([[0., 0.], [1., 1.]]))
        explainer.explain(np.array([[0.5, 0.5]]))
        np.testing.assert_array_equal(np.random.random(4), expected)
    finally:
        np.random.set_state(previous)
