import numpy as np
import pytest
from sklearn.linear_model import Ridge

from uncertainty_explainer import CQRConformalPredictor, CrepesConformalPredictor


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
