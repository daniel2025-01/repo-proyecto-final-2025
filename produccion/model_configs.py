"""Configuración de modelos y sus hiperparámetros para el tuning."""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

MODEL_CONFIGS = {
    "linear_regression": {
        "estimator": LinearRegression,
        "params": [
            {"fit_intercept": True, "positive": False},
            {"fit_intercept": True, "positive": True},
        ],
    },
    "random_forest": {
        "estimator": RandomForestRegressor,
        "params": [
            {"n_estimators": 100, "max_depth": None, "random_state": 42},
            {"n_estimators": 200, "max_depth": 10, "random_state": 42},
            {"n_estimators": 300, "max_depth": 15, "random_state": 42},
        ],
    },
    "gboost": {
        "estimator": GradientBoostingRegressor,
        "params": [
            {"n_estimators": 100, "learning_rate": 0.1},
            {"n_estimators": 200, "learning_rate": 0.05},
            {"n_estimators": 300, "learning_rate": 0.03},
        ],
    },
    "svr": {
        "estimator": SVR,
        "params": [
            {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            {"kernel": "rbf", "C": 10.0, "epsilon": 0.1},
            {"kernel": "rbf", "C": 100.0, "epsilon": 0.01},
        ],
    },
}