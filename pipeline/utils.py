"""
pipeline/utils.py — Funciones compartidas.

Centralizar aquí evita copiar y pegar el mismo código en cada script.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

import config as C


def get_logger(etapa: str) -> logging.Logger:
    """Logger con el mismo formato del curso: HH:MM:SS | ETAPA | NIVEL | mensaje."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | {etapa:<8} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger(etapa)


# ── Datos ────────────────────────────────────────────────────────────────────
def cargar_split():
    """Lee train/test generados por 01_ingest. Mismo split en TODOS los runs (regla U6)."""
    for ruta in (C.TRAIN_PATH, C.TEST_PATH):
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero: python pipeline/01_ingest.py")
    train, test = pd.read_csv(C.TRAIN_PATH), pd.read_csv(C.TEST_PATH)
    return (train[C.FEATURES], train[C.TARGET], train[C.ID_COL],
            test[C.FEATURES], test[C.TARGET])


# ── Modelo ───────────────────────────────────────────────────────────────────
def construir_pipeline(clasificador) -> Pipeline:
    """Pipeline ÚNICO = preprocesamiento + modelo.

    Es la pieza clave del proyecto: se entrena, se registra en MLflow y se sirve
    en la API como UN solo objeto. Así la API recibe datos crudos (con nulos o
    categorías nuevas) y aplica exactamente la misma transformación del entrenamiento.
    """
    preprocesamiento = ColumnTransformer([
        # Numéricas: la mediana es robusta a los outliers de Mto_TC y SUELDO_ESTIMADO
        ("num", SimpleImputer(strategy="median"), C.COLS_NUMERICAS),
        # Categóricas: moda para nulos + one-hot; 'ignore' tolera categorías no vistas
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), C.COLS_CATEGORICAS),
    ])
    return Pipeline([("preprocesamiento", preprocesamiento), ("modelo", clasificador)])


# ── Métricas ─────────────────────────────────────────────────────────────────
def lift_y_captura(y_true, scores, fraccion: float):
    """Lift: cuántas veces más compra el top X% vs el promedio.
    Captura: qué % de todos los compradores está dentro de ese top X%."""
    y = np.asarray(y_true)
    orden = np.argsort(-np.asarray(scores))      # de mayor a menor score
    n = max(1, int(len(y) * fraccion))
    top = y[orden][:n]
    return float(top.mean() / y.mean()), float(top.sum() / y.sum())


def evaluar(modelo, X_test, y_test) -> dict:
    """Métricas de negocio + técnicas, y si el modelo pasa el quality gate."""
    scores = modelo.predict_proba(X_test)[:, 1]
    pred = modelo.predict(X_test)
    lift10, cap10 = lift_y_captura(y_test, scores, 0.10)
    _, cap20 = lift_y_captura(y_test, scores, 0.20)
    m = {
        "auc": roc_auc_score(y_test, scores),             # métrica principal
        "pr_auc": average_precision_score(y_test, scores),
        "lift_top10": lift10,
        "captura_top10": cap10,
        "captura_top20": cap20,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    m = {k: round(v, 4) for k, v in m.items()}
    m["quality_gate"] = int(m["auc"] >= C.AUC_MINIMO and m["lift_top10"] >= C.LIFT_TOP10_MINIMO)
    return m


# ── MLflow ───────────────────────────────────────────────────────────────────
def configurar_mlflow():
    """Tracking en sqlite (mlflow.db en la raíz) + experimento del proyecto."""
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)


def registrar_run(nombre_run: str, modelo, metricas: dict, params: dict, X_ejemplo) -> str:
    """Registra UN run comparable: parámetros + métricas + el Pipeline completo."""
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.log_params(params)
        mlflow.log_params({"random_state": C.RANDOM_STATE, "cv_folds": C.CV_FOLDS,
                           "scoring": C.SCORING, "split": "StratifiedGroupKFold por cliente",
                           "n_features": len(C.FEATURES)})
        mlflow.log_metrics(metricas)
        mlflow.set_tag("quality_gate", "APROBADO" if metricas["quality_gate"] else "RECHAZADO")
        # MLflow 3.x: name= (antes artifact_path=) + tipos confiables para skops
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            skops_trusted_types=C.SKOPS_TRUSTED_TYPES,
            input_example=X_ejemplo.head(3),
        )
        return run.info.run_id
