"""
api/predictor.py — Carga el modelo UNA vez y ejecuta predicciones (patrón Unidad 05).

Diferencia clave con la U5: allá la API re-implementaba el preprocesamiento a mano
(preprocesamiento.json). Aquí modelo.pkl ES el Pipeline completo (imputación +
one-hot + Random Forest), así que basta con pasarle los datos crudos del cliente.
"""
import json
import pickle
import logging
from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent.parent       # raíz del proyecto (o /app en Docker)
MODEL_PATH   = BASE_DIR / "artifacts" / "modelo.pkl"        # exportado por pipeline/07_model_registry.py
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"      # métricas + cortes de prioridad (etapa 06)

# Mismo orden de columnas que en el entrenamiento (config.FEATURES)
FEATURES = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES",
            "Nombre_territorio", "REGION", "SEGMENTO", "MARCA"]

ACCIONES = {
    "ALTA":  "Contacto prioritario: llamada del asesor + SMS",
    "MEDIA": "Contacto de bajo costo: SMS o email",
    "BAJA":  "No priorizar en esta campaña",
}

log = logging.getLogger(__name__)


class Predictor:
    """Singleton: el modelo se lee de disco una sola vez, al arrancar la API."""

    def __init__(self):
        self.modelo = None
        self.metricas = {}

    def cargar(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe {MODEL_PATH}. Ejecuta primero: python run_pipeline.py")
        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)
        with open(METRICS_PATH) as f:
            self.metricas = json.load(f)
        log.info("Modelo cargado: run %s | AUC test %.3f",
                 self.metricas.get("run_name"), self.metricas.get("auc", 0))

    def prioridad(self, score: float) -> str:
        """Cortes calculados con datos en la etapa 06: ALTA = top 10%, MEDIA = siguiente 20%."""
        if score >= self.metricas["corte_prioridad_alta"]:
            return "ALTA"
        if score >= self.metricas["corte_prioridad_media"]:
            return "MEDIA"
        return "BAJA"

    def predecir(self, datos: dict) -> dict:
        if self.modelo is None:
            raise RuntimeError("Modelo no cargado. Llama a cargar() primero.")
        X = pd.DataFrame([{col: datos.get(col) for col in FEATURES}])   # datos crudos, 1 fila
        score = round(float(self.modelo.predict_proba(X)[0, 1]), 4)
        nivel = self.prioridad(score)
        return {
            "score_propension": score,
            "prioridad": nivel,
            "accion_sugerida": ACCIONES[nivel],
            "modelo": "RandomForest (Pipeline sklearn)",
            "version_modelo": self.metricas.get("run_name", "desconocida"),
        }


predictor = Predictor()   # instancia global compartida por los endpoints
