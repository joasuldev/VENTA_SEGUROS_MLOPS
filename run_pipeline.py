"""
run_pipeline.py — Orquestador: ejecuta las 7 etapas en orden con un solo comando.

Uso (desde la raíz del proyecto):
    python run_pipeline.py             # pipeline completo
    python run_pipeline.py --desde 5   # retomar desde la etapa 5 (útil para depurar)

Si una etapa falla, el pipeline se detiene con sys.exit(1): compatible con CI/CD.
Patrón de run_pipeline.py de las Unidades 04 y 06.
"""
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path

import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(C.LOG_PATH, encoding="utf-8")],
)
log = logging.getLogger("orquestador")

ETAPAS = [
    ("1 Ingesta + split por cliente", "pipeline/01_ingest.py"),
    ("2 Línea base (RF docente)",     "pipeline/02_base_pipeline.py"),
    ("3 GridSearch + MLflow",         "pipeline/03_gridsearch_mlflow.py"),
    ("4 RandomSearch + MLflow",       "pipeline/04_randomsearch_mlflow.py"),
    ("5 Comparar runs",               "pipeline/05_comparar_runs.py"),
    ("6 Mejor modelo + quality gate", "pipeline/06_mejor_modelo.py"),
    ("7 Model Registry + export",     "pipeline/07_model_registry.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", type=int, default=1, help="número de etapa desde la que iniciar")
    desde = parser.parse_args().desde

    log.info("=" * 60)
    log.info("  PIPELINE MLOps — SEGURO ANTIRROBOS")
    log.info("=" * 60)
    resumen, inicio_total = [], time.time()
    for i, (nombre, script) in enumerate(ETAPAS, start=1):
        if i < desde:
            continue
        log.info(">>> %s", nombre)
        t0 = time.time()
        resultado = subprocess.run([sys.executable, str(C.ROOT_DIR / script)], cwd=C.ROOT_DIR)
        dur = time.time() - t0
        ok = resultado.returncode == 0
        resumen.append((nombre, ok, dur))
        if not ok:
            log.error("XXX FALLÓ: %s (%.1f s). Pipeline detenido.", nombre, dur)
            log.error("    Depura con: python %s", script)
            sys.exit(1)
        log.info("<<< OK: %s (%.1f s)", nombre, dur)

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETADO EN %.1f s", time.time() - inicio_total)
    for nombre, ok, dur in resumen:
        log.info("   [%s] %-32s %6.1f s", "OK" if ok else "XX", nombre, dur)
    log.info("  MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
