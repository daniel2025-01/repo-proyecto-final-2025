"""Sales Forecast API - Proyecto final MLOps."""

from datetime import datetime, timezone
from flask import Flask, request, jsonify
import joblib
import pandas as pd

from produccion import config

# ---------------------------------------------------------------------
# CONFIGURACIÓN Y CARGA DEL MODELO
# ---------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = config.MODELS_DIR / "full_sales_forecast_pipeline.pkl"
MODEL_NAME = "full_sales_forecast_pipeline"

# Carga del modelo entrenado
model = joblib.load(MODEL_PATH)

# Rellena estos diccionarios con lo que reportaste en tus notebooks
MODEL_HYPERPARAMETERS = {
    "estimator": "LinearRegression",
    "fit_intercept": True,
    "positive": False,
}

MODEL_METRICS = {
    "rmse_val": 1234.56,   # pon aquí tu RMSE real
    "r2_val": 0.89,        # ejemplo
}

# ---------------------------------------------------------------------
# ENDPOINT: RAÍZ
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Página principal de la API."""
    return jsonify(
        {
            "message": "Sales Forecast API is running",
            "available_endpoints": {
                "health": "/health",
                "predict_one": "/predict_one",
                "predict_batch": "/predict_batch",
            },
        }
    )


# ---------------------------------------------------------------------
# ENDPOINT: HEALTHCHECK
# ---------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Chequeo simple de salud."""
    return jsonify(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------
# ENDPOINT: PREDICCIÓN INDIVIDUAL
#   - GET  → muestra ejemplo
#   - POST → realiza predicción
# ---------------------------------------------------------------------
@app.route("/predict_one", methods=["GET", "POST"])
def predict_one():
    # 1) Si entras desde el navegador con GET → ejemplo
    if request.method == "GET":
        return jsonify(
            {
                "message": "Usa POST con JSON en el body para obtener predicciones.",
                "example_payload": {
                    "QUANTITYORDERED": 30,
                    "PRICEEACH": 95.7,
                    "PRODUCTLINE": "Classic Cars",
                    "CITY_TOP": "NYC",
                    "STATUS": "Shipped",
                    "ORDERDATE": "2004-02-01",
                },
                "how_to_use": {
                    "method": "POST",
                    "url": "http://127.0.0.1:5000/predict_one",
                    "content_type": "application/json",
                },
            }
        )

    # 2) Si es POST → hacemos la predicción
    payload = request.get_json(force=True) or {}
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]

    response = {
        "prediction": float(pred),
        "model": {
            "name": MODEL_NAME,
            "hyperparameters": MODEL_HYPERPARAMETERS,
            "metrics": MODEL_METRICS,
        },
        "request_metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        },
    }
    return jsonify(response)


# ---------------------------------------------------------------------
# ENDPOINT: PREDICCIÓN POR LOTE
# ---------------------------------------------------------------------
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predicción por lote (batch)."""

    payload = request.get_json(silent=True)

    # Si no viene JSON o viene vacío
    if payload is None:
        return (
            jsonify(
                {
                    "error": "Body vacío o no es JSON válido.",
                    "hint": "En Postman: Body -> raw -> JSON y pega un objeto con la llave 'records'.",
                }
            ),
            400,
        )

    records = payload.get("records")

    if not isinstance(records, list) or len(records) == 0:
        return (
            jsonify(
                {
                    "error": "La clave 'records' debe ser una lista con al menos un registro.",
                    "example": {
                        "records": [
                            {
                                "QUANTITYORDERED": 30,
                                "PRICEEACH": 95.7,
                                "PRODUCTLINE": "Classic Cars",
                                "CITY_TOP": "NYC",
                                "STATUS": "Shipped",
                                "ORDERDATE": "2004-02-01",
                            }
                        ]
                    },
                }
            ),
            400,
        )

    df = pd.DataFrame(records)
    preds = model.predict(df)

    result = {
        "predictions": [float(p) for p in preds],
        "model": {
            "name": MODEL_NAME,
            "hyperparameters": MODEL_HYPERPARAMETERS,
            "metrics": MODEL_METRICS,
        },
        "request_metadata": {
            "batch_size": len(records),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return jsonify(result)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Imprimir el mapa de rutas para depurar
    print("=== URL MAP ===")
    print(app.url_map)
    print("===============\n")

    app.run(host="0.0.0.0", port=5000, debug=True)