"""API Flask para servir el modelo de ventas."""

from datetime import datetime, timezone
from flask import Flask, request, jsonify
import joblib
import pandas as pd

from produccion import config

app = Flask(__name__)

# -------------------------------------------------
# Página principal
# -------------------------------------------------
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

# -------------------------------------------------
# Endpoint de salud (SOLO UNO)
# -------------------------------------------------
@app.get("/health")
def health():
    """Chequeo simple de salud de la API."""
    return jsonify(
        {
            "status": "ok",
            "detail": "API healthy",
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )

# -------------------------------------------------
# Carga del modelo
# -------------------------------------------------
MODEL_PATH = config.MODELS_DIR / "full_sales_forecast_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# Predicción individual
# -------------------------------------------------
@app.post("/predict_one")
def predict_one():
    """Recibe un solo registro en JSON y devuelve una predicción."""
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]

    return jsonify(
        {
            "prediction": float(pred),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

# -------------------------------------------------
# Predicción por lote
# -------------------------------------------------
@app.post("/predict_batch")
def predict_batch():
    """Recibe varios registros en JSON (clave 'records') y devuelve predicciones."""
    data = request.get_json()
    df = pd.DataFrame(data["records"])
    preds = model.predict(df)

    return jsonify(
        {
            "predictions": preds.tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

# -------------------------------------------------
# Entrada principal
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)