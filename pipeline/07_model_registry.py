"""
Etapa 7 — MLflow Model Registry: registrar → Staging → exportar para la API.

Entrada : artifacts/mejor_run.json (de la etapa 6)
Salida  : modelo registrado en MLflow (Staging + alias "champion") · artifacts/modelo.pkl
Ejecutar: python pipeline/07_model_registry.py

Por qué exportar modelo.pkl: la API (Docker) carga un solo archivo, igual que en la U5,
sin necesitar MLflow ni la base mlflow.db dentro del contenedor.
"""
import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

import utils as U
import config as C

log = U.get_logger("REGISTRY")


def run():
    log.info("=== ETAPA 7: MODEL REGISTRY ===")
    ruta = C.ARTIFACTS_DIR / "mejor_run.json"
    if not ruta.exists():
        raise FileNotFoundError("No existe mejor_run.json. Ejecuta primero: python pipeline/06_mejor_modelo.py")
    mejor = json.loads(ruta.read_text())

    U.configurar_mlflow()
    version = mlflow.register_model(f"runs:/{mejor['run_id']}/modelo", C.MODEL_REGISTRY_NAME).version
    log.info("  Registrado: %s v%s (run %s)", C.MODEL_REGISTRY_NAME, version, mejor["run_name"])

    client = MlflowClient()
    # Stage "Staging" como en la clase (U6). Es una API deprecada en MLflow 3 pero funciona;
    # por eso además se asigna el alias "champion", que es el reemplazo moderno.
    client.transition_model_version_stage(C.MODEL_REGISTRY_NAME, version, "Staging")
    client.set_registered_model_alias(C.MODEL_REGISTRY_NAME, "champion", version)
    log.info("  ✓ Promovido a Staging + alias 'champion'")

    # Verificación: cargar DESDE el Registry (no desde el run) y predecir datos crudos
    modelo = mlflow.sklearn.load_model(f"models:/{C.MODEL_REGISTRY_NAME}@champion")
    _, _, _, X_test, _ = U.cargar_split()
    log.info("  Verificación: scores de 3 clientes de test = %s",
             modelo.predict_proba(X_test.head(3))[:, 1].round(3).tolist())

    # Exportar para la API
    with open(C.ARTIFACTS_DIR / "modelo.pkl", "wb") as f:
        pickle.dump(modelo, f)
    log.info("  Exportado para la API: artifacts/modelo.pkl")
    log.info("=== ETAPA 7 COMPLETADA ===")


if __name__ == "__main__":
    run()
