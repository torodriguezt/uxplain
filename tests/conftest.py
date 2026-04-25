import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 4))
    y = X @ np.array([1.5, -2.0, 0.5, 1.0]) + rng.standard_normal(n) * 0.5
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
    return {
        "X_train": X_tr,
        "X_calib": X_cal,
        "X_test": X_te,
        "y_train": y_tr,
        "y_calib": y_cal,
        "y_test": y_te,
    }


@pytest.fixture
def ridge():
    return Ridge()


@pytest.fixture
def quantile_lower():
    return GradientBoostingRegressor(
        loss="quantile", alpha=0.05, n_estimators=30, random_state=0
    )


@pytest.fixture
def quantile_upper():
    return GradientBoostingRegressor(
        loss="quantile", alpha=0.95, n_estimators=30, random_state=0
    )
