"""
Script principal para ejecutar el pipeline end-to-end del proyecto de ventas.

Pasos:
1. Cargar datos crudos.
2. Separar en train (80%) y validación (20%) de forma secuencial.
3. Construir o cargar el pipeline de features.
4. Entrenar varios modelos, registrar RMSE y tiempo en MLflow.
5. Seleccionar el mejor modelo y guardar el pipeline completo en disco.
6. Generar predicciones sobre el set de validación y guardarlas en CSV.

El código está pensado para seguir PEP8 y ser evaluado con pylint.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "notebooks", "models")

RAW_FILE = os.path.join(RAW_DIR, "sales_data_sample.csv")
FEATURE_PIPELINE_FILE = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
FULL_PIPELINE_FILE = os.path.join(MODELS_DIR, "full_sales_forecast_pipeline.pkl")
PREDICTIONS_FILE = os.path.join(PROCESSED_DIR, "test_predictions.csv")

TARGET_COL = "SALES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el dataset crudo desde un archivo CSV."""
    logging.info("Cargando datos crudos desde %s", path)
    df = pd.read_csv(path, encoding="latin1")
    logging.info("Dataset cargado con shape %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. SPLIT SECUENCIAL 80/20
# ---------------------------------------------------------------------------

def train_val_split_sequential(
    df: pd.DataFrame, target_col: str, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza un split secuencial del dataset en train y validación.

    No se barajan los datos para respetar el orden temporal.
    """
    n_rows = len(df)
    split_index = int(n_rows * train_ratio)

    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    x_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    logging.info("Split 80/20 realizado: train=%s, val=%s", x_train.shape, x_val.shape)
    return x_train, x_val, y_train, y_val


# ---------------------------------------------------------------------------
# 3. CARGA DEL PIPELINE DE FEATURES
# ---------------------------------------------------------------------------

def load_feature_pipeline(path: str) -> Pipeline:
    """
    Carga desde disco el pipeline de features creado en el notebook 03.

    Si no existe, lanza un error para obligar a crearlo antes.
    """
    if not os.path.exists(path):
        msg = (
            f"No se encontró el pipeline de features en {path}. "
            "Asegúrate de ejecutar el notebook 03_feature_creation "
            "y de que guarde feature_pipeline.pkl en notebooks/models."
        )
        raise FileNotFoundError(msg)

    logging.info("Cargando feature_pipeline desde %s", path)
    feature_pipeline: Pipeline = joblib.load(path)
    return feature_pipeline


# ---------------------------------------------------------------------------
# 4. CONFIGURACIÓN DE MODELOS
# ---------------------------------------------------------------------------

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Devuelve el diccionario de modelos y sus configuraciones de hiperparámetros."""
    return {
        "linear_regression": {
            "estimator": LinearRegression,
            "params": [
                {"fit_intercept": True, "positive": False},
                {"fit_intercept": True, "positive": True},
                {"fit_intercept": False, "positive": False},
            ],
        },
        "random_forest": {
            "estimator": RandomForestRegressor,
            "params": [
                {"n_estimators": 100, "max_depth": None, "random_state": 42},
                {"n_estimators": 200, "max_depth": 10, "random_state": 42},
                {"n_estimators": 300, "max_depth": 5, "random_state": 42},
            ],
        },
        "gradient_boosting": {
            "estimator": GradientBoostingRegressor,
            "params": [
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "random_state": 42,
                },
                {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "random_state": 42,
                },
            ],
        },
        "svr": {
            "estimator": SVR,
            "params": [
                {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
                {"kernel": "rbf", "C": 10.0, "epsilon": 0.01},
                {"kernel": "linear", "C": 1.0, "epsilon": 0.1},
            ],
        },
    }


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO CON MLflow Y SELECCIÓN DE MEJOR MODELO
# ---------------------------------------------------------------------------

def train_and_select_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    feature_pipeline: Pipeline,
) -> Pipeline:
    """
    Entrena varios modelos, registra sus métricas en MLflow y devuelve el pipeline ganador.
    """
    model_configs = get_model_configs()

    best_rmse = np.inf
    best_model_name = None
    best_config: Dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

    mlflow.set_experiment("Proyecto Final")

    start_time = time.time()

    with mlflow.start_run(run_name="tuning_modelos_ventas"):
        for model_name, cfg in model_configs.items():
            estimator_class = cfg["estimator"]

            for idx, param_dict in enumerate(cfg["params"], start=1):
                logging.info(
                    "Entrenando %s - config %s con params: %s",
                    model_name,
                    idx,
                    param_dict,
                )

                model = estimator_class(**param_dict)

                full_pipeline = Pipeline(
                    steps=[
                        ("features", feature_pipeline),
                        ("model", model),
                    ],
                )

                full_pipeline.fit(x_train, y_train)
                y_pred = full_pipeline.predict(x_val)

                rmse = mean_squared_error(y_val, y_pred, squared=False)
                logging.info("RMSE validación (%s, cfg %s): %.4f", model_name, idx, rmse)

                metric_name = f"rmse_{model_name}_cfg_{idx}"
                mlflow.log_metric(metric_name, rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = model_name
                    best_config = param_dict
                    best_pipeline = full_pipeline

        elapsed_time = round(time.time() - start_time, 2)
        mlflow.log_metric("tiempo_total_entrenamiento_s", elapsed_time)

    logging.info("=== Mejor modelo encontrado ===")
    logging.info("Modelo: %s", best_model_name)
    logging.info("Hiperparámetros: %s", best_config)
    logging.info("RMSE validación: %.4f", best_rmse)

    if best_pipeline is None:
        raise RuntimeError("No se pudo determinar un modelo ganador.")

    return best_pipeline


# ---------------------------------------------------------------------------
# 6. GUARDAR PIPELINE Y GENERAR PREDICCIONES
# ---------------------------------------------------------------------------

def save_full_pipeline(pipeline: Pipeline, path: str) -> None:
    """Guarda en disco el pipeline completo (features + modelo ganador)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logging.info("Pipeline completo guardado en %s", path)


def generate_and_save_predictions(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    path: str,
) -> None:
    """Genera predicciones y las guarda en un CSV."""
    logging.info("Generando predicciones sobre el conjunto de validación.")
    y_pred = pipeline.predict(x_test)

    preds_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
        },
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    preds_df.to_csv(path, index=False)
    logging.info("Predicciones guardadas en %s", path)


# ---------------------------------------------------------------------------
# 7. FUNCIÓN MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta el pipeline end-to-end del proyecto."""
    logging.info("Iniciando ejecución del pipeline end-to-end.")

    df_raw = load_raw_data(RAW_FILE)

    x_train, x_val, y_train, y_val = train_val_split_sequential(
        df_raw,
        target_col=TARGET_COL,
        train_ratio=0.8,
    )

    feature_pipeline = load_feature_pipeline(FEATURE_PIPELINE_FILE)

    best_pipeline = train_and_select_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_pipeline=feature_pipeline,
    )

    save_full_pipeline(best_pipeline, FULL_PIPELINE_FILE)

    generate_and_save_predictions(
        pipeline=best_pipeline,
        x_test=x_val,
        y_test=y_val,
        path=PREDICTIONS_FILE,
    )

    logging.info("Pipeline end-to-end finalizado correctamente.")


if __name__ == "__main__":
    main()

