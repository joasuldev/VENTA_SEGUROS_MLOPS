"""
Etapa 2 — Línea base: el Random Forest del notebook del docente, SIN tuning.

Entrada : artifacts/train.csv · artifacts/test.csv
Salida  : Run "Base-RF-docente" en MLflow
Ejecutar: python pipeline/02_base_pipeline.py

Por qué existe: todo tuning se compara contra una línea base (Unidad 06).
Usa los mismos hiperparámetros del notebook (200 árboles, sin balanceo),
pero dentro del Pipeline y con el split por cliente → resultado honesto.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.ensemble import RandomForestClassifier

import utils as U
import config as C

log = U.get_logger("BASE")


def run():
    log.info("=== ETAPA 2: LÍNEA BASE (RF del docente, sin tuning) ===")
    X_train, y_train, _, X_test, y_test = U.cargar_split()

    params = {"n_estimators": 200, "max_depth": None, "class_weight": None}
    modelo = U.construir_pipeline(
        RandomForestClassifier(**params, random_state=C.RANDOM_STATE, n_jobs=-1))
    modelo.fit(X_train, y_train)

    metricas = U.evaluar(modelo, X_test, y_test)
    log.info("  AUC=%.4f | lift top10=%.2fx | captura top20=%.0f%% | quality gate=%s",
             metricas["auc"], metricas["lift_top10"], 100 * metricas["captura_top20"],
             "APROBADO" if metricas["quality_gate"] else "RECHAZADO")

    U.configurar_mlflow()
    run_id = U.registrar_run("Base-RF-docente", modelo, metricas,
                             {**params, "metodo_busqueda": "ninguno (línea base)"}, X_test)
    log.info("  Run registrado en MLflow: %s", run_id)
    log.info("=== ETAPA 2 COMPLETADA ===")


if __name__ == "__main__":
    run()
