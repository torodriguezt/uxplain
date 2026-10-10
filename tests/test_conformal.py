import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    CrepesConformalClassifier,
    CrepesConformalPredictor,
)
from uxplain.conformal.crepes_predictor import _DeterministicMondrianCategorizer


class TestCrepesConformalPredictor:
    @pytest.mark.parametrize(
        "method", ["standard", "normalized", "mondrian", "normalized_mondrian"]
    )
    def test_fit_predict_shapes(self, method, data):
        cp = CrepesConformalPredictor(Ridge(), method=method)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"])

        n = len(data["X_test"])
        assert lower.shape == (n,)
        assert upper.shape == (n,)

    @pytest.mark.parametrize(
        "method", ["standard", "normalized", "mondrian", "normalized_mondrian"]
    )
    def test_lower_leq_upper(self, method, data):
        cp = CrepesConformalPredictor(Ridge(), method=method)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"])

        assert np.all(lower <= upper)

    def test_not_fitted_raises(self, data):
        cp = CrepesConformalPredictor(Ridge())
        with pytest.raises(RuntimeError, match="not fitted"):
            cp.predict(data["X_test"])

    def test_coverage_approximately_nominal(self, data):
        cp = CrepesConformalPredictor(Ridge(), method="normalized")
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"], confidence=0.9)

        coverage = np.mean(
            (data["y_test"] >= lower) & (data["y_test"] <= upper)
        )
        # Loose bound: should be well above 70% for valid conformal predictor
        assert coverage >= 0.7

    def test_higher_confidence_gives_wider_intervals(self, data):
        cp = CrepesConformalPredictor(Ridge())
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])

        lower_90, upper_90 = cp.predict(data["X_test"], confidence=0.90)
        lower_95, upper_95 = cp.predict(data["X_test"], confidence=0.95)

        width_90 = (upper_90 - lower_90).mean()
        width_95 = (upper_95 - lower_95).mean()
        assert width_95 > width_90

    def test_uncalibrated_mondrian_bin_is_unbounded(self):
        X_train = np.linspace(0, 1, 100).reshape(-1, 1)
        X_calib = np.linspace(0.01, 0.04, 20).reshape(-1, 1)
        cp = CrepesConformalPredictor(Ridge(alpha=0), method="mondrian")
        cp.fit(X_train, X_train[:, 0], X_calib, X_calib[:, 0])
        lower, upper = cp.predict(np.array([[0.02], [0.95]]), confidence=0.5)
        assert np.isfinite(lower[0]) and np.isfinite(upper[0])
        assert np.isneginf(lower[1]) and np.isposinf(upper[1])

    def test_mondrian_ties_have_deterministic_predictions(self):
        X_train = np.arange(100).reshape(-1, 1)
        X_calib = np.arange(100, 300).reshape(-1, 1)
        cp = CrepesConformalPredictor(DummyRegressor(), method="mondrian")
        cp.fit(X_train, np.zeros(100), X_calib, np.arange(200))
        X_test = np.arange(50).reshape(-1, 1)
        first = cp.predict(X_test)
        np.testing.assert_array_equal(cp.predict(X_test), first)
        reversed_prediction = cp.predict(X_test[::-1])
        np.testing.assert_array_equal(reversed_prediction[0][::-1], first[0])
        np.testing.assert_array_equal(reversed_prediction[1][::-1], first[1])

    @pytest.mark.parametrize("use_difficulty", [False, True])
    def test_mondrian_partition_does_not_depend_on_global_seed(self, use_difficulty):
        class TinyScores:
            def apply(self, X):
                return X[:, 0]

        X = np.arange(100).reshape(-1, 1) * 1e-11
        kwargs = {"de": TinyScores()} if use_difficulty else {"f": lambda X: X[:, 0]}
        rng_state = np.random.get_state()
        try:
            np.random.seed(0)
            first = _DeterministicMondrianCategorizer().fit(X, **kwargs)
            np.random.seed(1)
            second = _DeterministicMondrianCategorizer().fit(X, **kwargs)
        finally:
            np.random.set_state(rng_state)
        np.testing.assert_array_equal(first.bin_thresholds, second.bin_thresholds)
        np.testing.assert_array_equal(first.apply(X), second.apply(X))

    def test_constant_mondrian_scores_form_one_bin(self):
        X = np.ones((100, 1))
        mc = _DeterministicMondrianCategorizer().fit(X, f=lambda X: X[:, 0])
        np.testing.assert_array_equal(mc.bin_thresholds, [-np.inf, np.inf])
        np.testing.assert_array_equal(mc.apply(X), np.zeros(100, dtype=int))


class TestCQRConformalPredictor:
    def test_fit_predict_shapes(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"])

        n = len(data["X_test"])
        assert lower.shape == (n,)
        assert upper.shape == (n,)

    def test_lower_leq_upper(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"])

        assert np.all(lower <= upper)

    def test_not_fitted_raises(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        with pytest.raises(RuntimeError, match="not fitted"):
            cp.predict(data["X_test"])

    def test_coverage_approximately_nominal(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        lower, upper = cp.predict(data["X_test"], confidence=0.9)

        coverage = np.mean(
            (data["y_test"] >= lower) & (data["y_test"] <= upper)
        )
        assert coverage >= 0.7

    def test_higher_confidence_gives_wider_intervals(self, data, quantile_lower, quantile_upper):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])

        lower_90, upper_90 = cp.predict(data["X_test"], confidence=0.90)
        lower_95, upper_95 = cp.predict(data["X_test"], confidence=0.95)

        width_90 = (upper_90 - lower_90).mean()
        width_95 = (upper_95 - lower_95).mean()
        assert width_95 > width_90

    def test_adjustment_is_exact_conformal_quantile(
        self, data, quantile_lower, quantile_upper,
    ):
        """The offset is the ceil((n + 1) * confidence)-th smallest score."""
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        k = int(np.ceil((len(data["y_calib"]) + 1) * 0.9))
        lower, _ = cp.predict(data["X_test"], confidence=0.9)
        np.testing.assert_allclose(
            quantile_lower.predict(data["X_test"]) - lower,
            np.sort(cp._scores)[k - 1],
        )

    def test_unbounded_when_calibration_too_small(
        self, data, quantile_lower, quantile_upper,
    ):
        """With 9 calibration rows, 95% coverage needs an infinite interval."""
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"],
               data["X_calib"][:9], data["y_calib"][:9])
        lower, upper = cp.predict(data["X_test"], confidence=0.95)
        assert np.all(np.isneginf(lower)) and np.all(np.isposinf(upper))

    def test_rank_does_not_round_down_above_boundary(
        self, data, quantile_lower, quantile_upper,
    ):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        cp.fit(data["X_train"], data["y_train"],
               data["X_calib"][:9], data["y_calib"][:9])
        lower, upper = cp.predict(data["X_test"], confidence=0.9 + 1e-12)
        assert np.all(np.isneginf(lower)) and np.all(np.isposinf(upper))

    def test_nonfinite_calibration_target_is_rejected(
        self, data, quantile_lower, quantile_upper,
    ):
        cp = CQRConformalPredictor(quantile_lower, quantile_upper)
        y_calib = data["y_calib"].copy()
        y_calib[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            cp.fit(data["X_train"], data["y_train"], data["X_calib"], y_calib)

    def test_empty_cqr_set_preserves_signed_endpoints(self):
        class FixedQuantile:
            def __init__(self, sign):
                self.sign = sign

            def fit(self, X, y):
                pass

            def predict(self, X):
                return self.sign * X[:, 0]

        cp = CQRConformalPredictor(FixedQuantile(-1), FixedQuantile(1))
        cp.fit(np.ones((10, 1)), np.zeros(10), np.full((20, 1), 10), np.zeros(20))
        lower, upper = cp.predict(np.ones((1, 1)))
        np.testing.assert_array_equal(lower, [9.0])
        np.testing.assert_array_equal(upper, [-9.0])
        from uxplain.uncertainty.metrics import make_uncertainty_function

        for metric, expected in (("width", -18), ("midpoint", 0),
                                 ("lower", 9), ("upper", -9)):
            target = make_uncertainty_function(cp, metric=metric)
            np.testing.assert_array_equal(target(np.ones((1, 1))), [expected])


class TestCrepesConformalClassifier:
    @pytest.mark.parametrize("method", ["standard", "class_cond", "mondrian"])
    def test_predict_set_shape_and_dtype(self, method, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier, method=method)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        pred_set = cp.predict_set(classification_data["X_test"], confidence=0.9)

        n = len(classification_data["X_test"])
        assert pred_set.shape == (n, classification_data["n_classes"])
        assert pred_set.dtype == bool

    @pytest.mark.parametrize("method", ["standard", "class_cond", "mondrian"])
    def test_predict_p_shape_and_range(self, method, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier, method=method)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        p = cp.predict_p(classification_data["X_test"])

        n = len(classification_data["X_test"])
        assert p.shape == (n, classification_data["n_classes"])
        assert np.all((p >= 0.0) & (p <= 1.0 + 1e-9))

    def test_classes_matches_underlying_model(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        np.testing.assert_array_equal(cp.classes_, classifier.classes_)

    def test_predict_set_before_fit_raises(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier)
        with pytest.raises(RuntimeError, match="not fitted"):
            cp.predict_set(classification_data["X_test"])

    def test_predict_p_before_fit_raises(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier)
        with pytest.raises(RuntimeError, match="not fitted"):
            cp.predict_p(classification_data["X_test"])

    def test_coverage_approximately_nominal(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier, method="standard")
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        pred_set = cp.predict_set(classification_data["X_test"], confidence=0.9)
        y_test = classification_data["y_test"]
        # Marginal coverage = fraction of samples whose true class is in the set
        covered = pred_set[np.arange(len(y_test)), y_test]
        assert covered.mean() >= 0.75

    def test_higher_confidence_gives_larger_or_equal_sets(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier, method="standard")
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        sizes_80 = cp.predict_set(classification_data["X_test"], confidence=0.80).sum(axis=1)
        sizes_99 = cp.predict_set(classification_data["X_test"], confidence=0.99).sum(axis=1)
        assert sizes_99.mean() >= sizes_80.mean()

    def test_predict_proba_delegates_to_model(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        proba = cp.predict_proba(classification_data["X_test"])
        np.testing.assert_allclose(
            proba, classifier.predict_proba(classification_data["X_test"]),
        )

    def test_p_values_deterministic_by_default(self, classification_data, classifier):
        """Each row's p-values depend on that row only: no draw, no batch order."""
        cp = CrepesConformalClassifier(classifier)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        X = classification_data["X_test"]
        p = cp.predict_p(X)
        np.testing.assert_array_equal(cp.predict_p(X), p)
        np.testing.assert_array_equal(cp.predict_p(X[::-1])[::-1], p)
        np.testing.assert_array_equal(
            np.vstack([cp.predict_p(row[None, :]) for row in X[:3]]), p[:3],
        )

    def test_seeded_smoothing_is_not_a_rowwise_function(
        self, classification_data, classifier,
    ):
        cp = CrepesConformalClassifier(classifier, smoothing=True, random_state=42)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        repeated = np.repeat(classification_data["X_test"][:1], 2, axis=0)
        p = cp.predict_p(repeated)
        np.testing.assert_array_equal(p, cp.predict_p(repeated))
        assert np.any(p[0] != p[1])  # Same x, different batch positions.

    def test_smoothing_is_opt_in(self, classification_data, classifier):
        """Smoothed p-values never exceed the conservative non-smoothed ones."""
        cp = CrepesConformalClassifier(classifier)
        cp.fit(
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_calib"], classification_data["y_calib"],
        )
        X = classification_data["X_test"]
        p_plain = cp.predict_p(X)
        cp.smoothing = True
        p_smooth = cp.predict_p(X)
        assert np.all(p_smooth <= p_plain)
        assert np.any(p_smooth < p_plain)

    def test_unseen_calibration_label_is_rejected(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier)
        y_calib = classification_data["y_calib"].copy()
        y_calib[0] = 99
        with pytest.raises(ValueError, match="classes absent"):
            cp.fit(classification_data["X_train"], classification_data["y_train"],
                   classification_data["X_calib"], y_calib)

    def test_absent_calibration_class_has_p_value_one(self, classification_data, classifier):
        cp = CrepesConformalClassifier(classifier, method="class_cond")
        keep = classification_data["y_calib"] != 2
        cp.fit(classification_data["X_train"], classification_data["y_train"],
               classification_data["X_calib"][keep], classification_data["y_calib"][keep])
        p = cp.predict_p(classification_data["X_test"])
        np.testing.assert_array_equal(p[:, np.flatnonzero(cp.classes_ == 2)[0]], 1)


class TestBackendValidation:
    @pytest.mark.parametrize("backend", ["crepes", "cqr"])
    def test_column_targets_do_not_broadcast(self, backend, data):
        def make_cp():
            if backend == "cqr":
                return CQRConformalPredictor(Ridge(), Ridge())
            return CrepesConformalPredictor(Ridge(), method="standard")

        flat = make_cp()
        flat.fit(data["X_train"], data["y_train"], data["X_calib"], data["y_calib"])
        column = make_cp()
        column.fit(data["X_train"], data["y_train"][:, None],
                   data["X_calib"], data["y_calib"][:, None])
        np.testing.assert_allclose(column.predict(data["X_test"]),
                                   flat.predict(data["X_test"]))

    @pytest.mark.parametrize("backend", ["crepes", "cqr"])
    def test_multioutput_targets_are_rejected(self, backend, data):
        cp = (CQRConformalPredictor(Ridge(), Ridge()) if backend == "cqr"
              else CrepesConformalPredictor(Ridge()))
        with pytest.raises(ValueError, match="single output"):
            cp.fit(data["X_train"], np.column_stack([data["y_train"]] * 2),
                   data["X_calib"], data["y_calib"])

    @pytest.mark.parametrize("backend", ["crepes", "cqr", "classification"])
    @pytest.mark.parametrize("confidence", [0, 1, -0.1, 90, np.nan, np.inf, "0.9"])
    def test_invalid_confidence_is_rejected(
        self, backend, confidence, data, classification_data, classifier,
    ):
        if backend == "classification":
            cp = CrepesConformalClassifier(classifier)
            fit_data = classification_data
        else:
            cp = (CQRConformalPredictor(Ridge(), Ridge()) if backend == "cqr"
                  else CrepesConformalPredictor(Ridge(), method="standard"))
            fit_data = data
        cp.fit(fit_data["X_train"], fit_data["y_train"],
               fit_data["X_calib"], fit_data["y_calib"])
        predict = cp.predict_set if backend == "classification" else cp.predict
        with pytest.raises(ValueError, match="confidence must be in"):
            predict(fit_data["X_test"], confidence=confidence)

    @pytest.mark.parametrize("cp_class", [CrepesConformalPredictor, CrepesConformalClassifier])
    def test_unknown_method_is_rejected(self, cp_class):
        with pytest.raises(ValueError, match="Unknown conformal"):
            cp_class(Ridge(), method="typo")

    @pytest.mark.parametrize("backend", ["crepes", "cqr"])
    def test_empty_calibration_is_rejected(self, backend, data):
        cp = (CQRConformalPredictor(Ridge(), Ridge()) if backend == "cqr"
              else CrepesConformalPredictor(Ridge()))
        with pytest.raises(ValueError, match="non-empty"):
            cp.fit(data["X_train"], data["y_train"],
                   data["X_calib"][:0], data["y_calib"][:0])
