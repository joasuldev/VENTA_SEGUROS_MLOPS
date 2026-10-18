"""
Etapa 1 — Ingesta, limpieza mínima, validación y split por cliente.

Entrada : data/raw/DS_Seguro_Antirobos.csv
Salida  : artifacts/train.csv · artifacts/test.csv · artifacts/ingest_resumen.json
Ejecutar: python pipeline/01_ingest.py

Qué NO hace esta etapa (a propósito): imputar nulos ni codificar categorías.
Eso vive dentro del Pipeline de scikit-learn (etapa 2 en adelante), para que:
  1) las medianas/modas se aprendan SOLO con train (sin fuga de datos), y
  2) la API aplique exactamente la misma transformación que el entrenamiento.
"""
import sys
import json
import logging
from pathlib import Path

# Permite importar config.py desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | INGEST   | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def cargar_datos(ruta: Path = C.RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el CSV crudo. Falla con un mensaje claro si el archivo no existe."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {ruta}. ¿Ejecutaste 'dvc pull' o copiaste el CSV a data/raw/?"
        )
    df = pd.read_csv(ruta, sep=C.CSV_SEP)
    log.info("Cargado: %d filas x %d columnas", *df.shape)
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y convierte el target vacío en 0."""
    df = df.copy()
    n_antes = len(df)
    # Duplicados exactos: la misma fila repetida no aporta información y sesga el modelo
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("  Duplicados exactos eliminados: %d", n_antes - len(df))
    # Target: vacío = "no compró" → 0 ; 1 = "compró"
    df[C.TARGET] = df[C.TARGET].fillna(0).astype(int)
    log.info("  Target %s: %d compradores (%.2f%%)",
             C.TARGET, df[C.TARGET].sum(), 100 * df[C.TARGET].mean())
    return df


def crear_cliente_id(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna el mismo ID a las filas del mismo cliente (fotos mensuales)."""
    df = df.copy()
    # dropna=False: un nulo en alguna columna clave también forma parte de la "firma"
    df[C.ID_COL] = df.groupby(C.COLS_ID_CLIENTE, dropna=False).ngroup()
    log.info("  Clientes únicos: %d (de %d filas)", df[C.ID_COL].nunique(), len(df))
    return df


def separar_por_cliente(df: pd.DataFrame):
    """Split train/test estratificado y AGRUPADO por cliente.

    Estratificado: train y test mantienen ~la misma tasa de compradores.
    Agrupado: todas las filas de un cliente caen del mismo lado (sin fuga).
    """
    sgkf = StratifiedGroupKFold(n_splits=C.N_PARTES, shuffle=True, random_state=C.RANDOM_STATE)
    idx_train, idx_test = next(sgkf.split(df, df[C.TARGET], groups=df[C.ID_COL]))
    train, test = df.iloc[idx_train].copy(), df.iloc[idx_test].copy()
    log.info("  Train: %d filas (%.2f%% compradores) | Test: %d filas (%.2f%% compradores)",
             len(train), 100 * train[C.TARGET].mean(), len(test), 100 * test[C.TARGET].mean())
    return train, test


def validar_salida(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Quality check de datos. Lanza ValueError si falla: el pipeline NO continúa.

    Patrón validar_salida() de la Unidad 04: evitar errores silenciosos.
    """
    errores = []
    for nombre, parte in [("train", train), ("test", test)]:
        faltantes = set(C.COLUMNAS_REQUERIDAS + [C.ID_COL]) - set(parte.columns)
        if faltantes:
            errores.append(f"{nombre}: COLUMNAS FALTANTES {faltantes}")
            continue
        valores = set(parte[C.TARGET].unique())
        if not valores.issubset({0, 1}):
            errores.append(f"{nombre}: TARGET no binario {valores}")
        if parte[C.TARGET].sum() == 0:
            errores.append(f"{nombre}: sin compradores (target todo 0)")
    if errores:  # si faltan columnas no tiene sentido seguir validando
        raise ValueError("VALIDACIÓN ETAPA 1 FALLIDA:\n" + "\n".join(errores))

    # La regla clave anti-fuga: ningún cliente en ambos lados
    compartidos = set(train[C.ID_COL]) & set(test[C.ID_COL])
    if compartidos:
        errores.append(f"FUGA: {len(compartidos)} clientes en train y test a la vez")
    if len(train) + len(test) < C.MIN_FILAS:
        errores.append(f"FILAS INSUFICIENTES: {len(train) + len(test)} (mínimo {C.MIN_FILAS})")
    if errores:
        raise ValueError("VALIDACIÓN ETAPA 1 FALLIDA:\n" + "\n".join(errores))
    log.info("  ✓ Validación OK: columnas completas | target binario | 0 clientes compartidos")


def run() -> tuple:
    """Ejecuta la etapa completa y guarda los artefactos."""
    log.info("=== ETAPA 1: INGESTA ===")
    df = cargar_datos()
    faltantes = set(C.COLUMNAS_REQUERIDAS) - set(df.columns)
    if faltantes:  # validar el CSV crudo ANTES de procesarlo
        raise ValueError(f"El CSV no trae las columnas requeridas: {faltantes}")
    df = limpiar(df)
    df = crear_cliente_id(df)
    train, test = separar_por_cliente(df)
    validar_salida(train, test)

    train.to_csv(C.TRAIN_PATH, index=False)
    test.to_csv(C.TEST_PATH, index=False)
    resumen = {
        "filas_train": len(train), "filas_test": len(test),
        "clientes_train": int(train[C.ID_COL].nunique()),
        "clientes_test": int(test[C.ID_COL].nunique()),
        "tasa_compra_train": round(float(train[C.TARGET].mean()), 4),
        "tasa_compra_test": round(float(test[C.TARGET].mean()), 4),
        "clientes_compartidos": 0,
    }
    with open(C.INGEST_RESUMEN, "w") as f:
        json.dump(resumen, f, indent=2)
    log.info("Artefactos guardados: %s | %s", C.TRAIN_PATH.name, C.TEST_PATH.name)
    log.info("=== ETAPA 1 COMPLETADA ===")
    return train, test


if __name__ == "__main__":
    run()
