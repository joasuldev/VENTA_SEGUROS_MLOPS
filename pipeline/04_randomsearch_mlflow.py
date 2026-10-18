"""
Etapa 4 — RandomizedSearchCV + MLflow.

Entrada : artifacts/train.csv · artifacts/test.csv
Salida  : Run "RandomSearch-RF" en MLflow
Ejecutar: python pipeline/04_randomsearch_mlflow.py

Muestrea N_ITER combinaciones de un espacio continuo (scipy.stats).
Misma semilla, mismo split, misma métrica que el GridSearch → runs comparables.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

import utils as U
import config as C

log = U.get_logger("RANDOM")

N_ITER = 12   # mismo presupuesto que el grid (12 combinaciones) → comparación justa
PARAM_DIST = {
    "modelo__max_depth": randint(3, 12),
    "modelo__min_samples_leaf": randint(5, 100),
    "modelo__max_features": ["sqrt", "log2", None],
}


def run():
    log.info("=== ETAPA 4: RANDOMSEARCH + MLFLOW ===")
    X_train, y_train, grupos, X_test, y_test = U.cargar_split()

    base = U.construir_pipeline(RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=C.RANDOM_STATE, n_jobs=-1))
    rs = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=N_ITER, scoring=C.SCORING, random_state=C.RANDOM_STATE,
        cv=StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE),
        n_jobs=1, verbose=1,
    )
    log.info("Muestreando %d combinaciones x %d folds...", N_ITER, C.CV_FOLDS)
    rs.fit(X_train, y_train, groups=grupos)
    log.info("  Mejores parámetros: %s | AUC CV=%.4f", rs.best_params_, rs.best_score_)

    metricas = U.evaluar(rs.best_estimator_, X_test, y_test)
    metricas["cv_auc"] = round(rs.best_score_, 4)
    log.info("  Test: AUC=%.4f | lift top10=%.2fx | quality gate=%s", metricas["auc"],
             metricas["lift_top10"], "APROBADO" if metricas["quality_gate"] else "RECHAZADO")

    U.configurar_mlflow()
    params = {k.replace("modelo__", ""): v for k, v in rs.best_params_.items()}
    params.update({"n_estimators": 200, "class_weight": "balanced_subsample",
                   "metodo_busqueda": "RandomizedSearchCV", "n_iter": N_ITER})
    run_id = U.registrar_run("RandomSearch-RF", rs.best_estimator_, metricas, params, X_test)
    log.info("  Run registrado en MLflow: %s", run_id)
    log.info("=== ETAPA 4 COMPLETADA ===")


if __name__ == "__main__":
    run()
