"""
Etapa 6 — Selección del mejor modelo + QUALITY GATE.

Entrada : runs de MLflow (los más recientes de cada tipo)
Salida  : artifacts/mejor_run.json · artifacts/metrics.json
Ejecutar: python pipeline/06_mejor_modelo.py

Si el mejor modelo NO supera el quality gate, el script termina con error
(sys.exit 1) y el pipeline se detiene: un modelo malo nunca llega al Registry.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import numpy as np

import utils as U
import config as C

log = U.get_logger("MEJOR")


def run():
    log.info("=== ETAPA 6: MEJOR MODELO + QUALITY GATE ===")
    U.configurar_mlflow()
    runs = mlflow.search_runs(experiment_names=[C.MLFLOW_EXPERIMENT], order_by=["start_time DESC"])
    runs = runs.drop_duplicates(subset="tags.mlflow.runName", keep="first")
    mejor = runs.sort_values("metrics.auc", ascending=False).iloc[0]   # criterio: mayor AUC
    nombre, run_id = mejor["tags.mlflow.runName"], mejor["run_id"]
    auc, lift = mejor["metrics.auc"], mejor["metrics.lift_top10"]
    log.info("  Mejor run: %s (%s) | AUC=%.4f | lift top10=%.2fx", nombre, run_id[:8], auc, lift)

    # ── Quality gate: el modelo debe superar AMBOS umbrales ─────────────────
    if auc < C.AUC_MINIMO or lift < C.LIFT_TOP10_MINIMO:
        log.error("XXX QUALITY GATE RECHAZADO: AUC=%.4f (mín %.2f) | lift=%.2f (mín %.2f)",
                  auc, C.AUC_MINIMO, lift, C.LIFT_TOP10_MINIMO)
        sys.exit(1)
    log.info("  ✓ QUALITY GATE APROBADO: AUC %.4f ≥ %.2f y lift %.2f ≥ %.2f",
             auc, C.AUC_MINIMO, lift, C.LIFT_TOP10_MINIMO)

    # ── Cortes de prioridad para la API, calculados con datos (no a ojo) ────
    # ALTA = top 10% de scores del test ; MEDIA = siguiente 20% ; BAJA = el resto
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    _, _, _, X_test, y_test = U.cargar_split()
    scores = modelo.predict_proba(X_test)[:, 1]
    corte_alta, corte_media = np.quantile(scores, 0.90), np.quantile(scores, 0.70)

    metricas = {k.replace("metrics.", ""): round(float(v), 4)
                for k, v in mejor.items() if k.startswith("metrics.") and v == v}
    metricas.update({
        "modelo": "RandomForestClassifier (Pipeline sklearn)",
        "run_name": nombre, "run_id": run_id,
        "corte_prioridad_alta": round(float(corte_alta), 4),
        "corte_prioridad_media": round(float(corte_media), 4),
        "tasa_compra_promedio": round(float(y_test.mean()), 4),
        "auc_minimo": C.AUC_MINIMO, "lift_top10_minimo": C.LIFT_TOP10_MINIMO,
    })
    with open(C.ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metricas, f, indent=2)
    with open(C.ARTIFACTS_DIR / "mejor_run.json", "w") as f:
        json.dump({"run_id": run_id, "run_name": nombre}, f, indent=2)
    log.info("  Cortes de prioridad: ALTA ≥ %.3f | MEDIA ≥ %.3f", corte_alta, corte_media)
    log.info("=== ETAPA 6 COMPLETADA ===")


if __name__ == "__main__":
    run()
