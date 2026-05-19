import numpy as np
import pytest
from sklearn.linear_model import Ridge

from uxplain import (
    CQRConformalPredictor,
    CrepesConformalClassifier,
    CrepesConformalPredictor,
)


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
