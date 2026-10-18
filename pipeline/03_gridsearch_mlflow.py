"""
Etapa 3 — GridSearchCV + MLflow.

Entrada : artifacts/train.csv · artifacts/test.csv
Salida  : Run "GridSearch-RF" en MLflow
Ejecutar: python pipeline/03_gridsearch_mlflow.py

Diferencias con el laboratorio de la U6 (y por qué):
  - Se busca sobre el Pipeline completo → el preprocesamiento se re-aprende en cada fold.
  - class_weight en vez de SMOTE → no se cuelan puntos sintéticos en la validación.
  - StratifiedGroupKFold por cliente → la validación cruzada tampoco tiene fuga.
  - scoring = roc_auc → con 2.6% de compradores, el F1 a umbral 0.5 engaña.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

import utils as U
import config as C

log = U.get_logger("GRID")

# Espacio de búsqueda: 3 x 2 x 2 = 12 combinaciones (x 3 folds = 36 entrenamientos)
# El prefijo "modelo__" apunta al paso "modelo" dentro del Pipeline.
PARAM_GRID = {
    "modelo__max_depth": [4, 6, 8],
    "modelo__min_samples_leaf": [10, 30],
    "modelo__max_features": ["sqrt", None],
}


def run():
    log.info("=== ETAPA 3: GRIDSEARCH + MLFLOW ===")
    X_train, y_train, grupos, X_test, y_test = U.cargar_split()

    base = U.construir_pipeline(RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=C.RANDOM_STATE, n_jobs=-1))
    gs = GridSearchCV(
        base, PARAM_GRID, scoring=C.SCORING,
        cv=StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE),
        n_jobs=1, verbose=1,
    )
    log.info("Buscando en %d combinaciones x %d folds...", len(ParameterGrid(PARAM_GRID)), C.CV_FOLDS)
    gs.fit(X_train, y_train, groups=grupos)   # groups = cliente_id → folds sin fuga
    log.info("  Mejores parámetros: %s | AUC CV=%.4f", gs.best_params_, gs.best_score_)

    metricas = U.evaluar(gs.best_estimator_, X_test, y_test)
    metricas["cv_auc"] = round(gs.best_score_, 4)
    log.info("  Test: AUC=%.4f | lift top10=%.2fx | quality gate=%s", metricas["auc"],
             metricas["lift_top10"], "APROBADO" if metricas["quality_gate"] else "RECHAZADO")

    U.configurar_mlflow()
    params = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}
    params.update({"n_estimators": 200, "class_weight": "balanced_subsample",
                   "metodo_busqueda": "GridSearchCV", "n_combinaciones": len(gs.cv_results_["params"])})
    run_id = U.registrar_run("GridSearch-RF", gs.best_estimator_, metricas, params, X_test)
    log.info("  Run registrado en MLflow: %s", run_id)
    log.info("=== ETAPA 3 COMPLETADA ===")


if __name__ == "__main__":
    run()
