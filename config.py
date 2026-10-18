"""
config.py — Configuración centralizada del proyecto Venta de Seguros Antirrobos.

ÚNICO lugar donde se definen rutas, columnas y parámetros.
Todas las etapas del pipeline y la API importan desde aquí:

    import config as C
    df = pd.read_csv(C.RAW_DATA_PATH, sep=C.CSV_SEP)

Patrón tomado del config.py de la Unidad 04 del curso.
"""
from pathlib import Path

# ── Directorios ──────────────────────────────────────────────────────────────
# Rutas absolutas a partir de este archivo: funcionan desde cualquier carpeta
# (terminal, pytest, uvicorn o dentro del contenedor Docker).
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"      # salidas de cada etapa del pipeline

# ── Archivos ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = RAW_DIR / "DS_Seguro_Antirobos.csv"   # dataset (versionado con DVC)
CSV_SEP         = ";"                                    # el CSV original usa punto y coma
TRAIN_PATH      = ARTIFACTS_DIR / "train.csv"            # salida de 01_ingest
TEST_PATH       = ARTIFACTS_DIR / "test.csv"             # salida de 01_ingest
INGEST_RESUMEN  = ARTIFACTS_DIR / "ingest_resumen.json"  # métricas de calidad de datos
LOG_PATH        = ROOT_DIR / "pipeline_run.log"          # log del orquestador

# ── Variable objetivo ────────────────────────────────────────────────────────
# FLAG_SS = 1 si el cliente compró el seguro. En el CSV los "no compró" vienen
# vacíos, por eso se convierten a 0 (supuesto documentado en el README).
TARGET = "FLAG_SS"

# ── Identificación de cliente ────────────────────────────────────────────────
# El CSV no trae ID. Un mismo cliente aparece en varias filas (fotos mensuales:
# solo cambia ANTIGUEDAD_MES en 1). Estas columnas, juntas, identifican al cliente.
# Se usa para que un cliente NUNCA quede a la vez en train y en test (evita fuga).
ID_COL          = "cliente_id"
COLS_ID_CLIENTE = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "SEXO",
                   "Nombre_territorio", "REGION", "MARCA"]

# ── Variables del modelo (decisiones validadas en el análisis previo) ────────
COLS_NUMERICAS   = ["Mto_TC", "SUELDO_ESTIMADO", "EDAD", "ANTIGUEDAD_MES"]
COLS_CATEGORICAS = ["Nombre_territorio", "REGION", "SEGMENTO", "MARCA"]
FEATURES         = COLS_NUMERICAS + COLS_CATEGORICAS

# Variables EXCLUIDAS y por qué (para la defensa):
#   FLAG_UNICEF         → no figura en el diccionario.
#   FLAG_LIMA_PROVINCIA → 100% redundante con REGION (verificado fila por fila).
#   SEXO                → equidad: excluirla cuesta solo -0.004 de AUC.

# Columnas mínimas que debe traer el CSV crudo (validación de la etapa 1)
COLUMNAS_REQUERIDAS = sorted(set(FEATURES + COLS_ID_CLIENTE + [TARGET]))

# ── Split train / test ───────────────────────────────────────────────────────
# Split estratificado POR CLIENTE: 1 de cada N_PARTES grupos va a test (≈ 33%).
RANDOM_STATE = 42          # misma semilla en TODOS los runs (regla de la Unidad 06)
N_PARTES     = 3           # 3 partes → 1/3 test, 2/3 train
MIN_FILAS    = 1_000       # validación: menos filas indica un CSV incompleto

# ── MLflow (Unidades 03 y 06) ────────────────────────────────────────────────
# Backend sqlite (como U3/U4/U10): no requiere MLFLOW_ALLOW_FILE_STORE.
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT   = "seguro-antirrobos"
MODEL_REGISTRY_NAME = "SeguroAntirrobos-Clasificador"
# MLflow 3.x guarda el modelo con skops; el Pipeline (OneHotEncoder + RandomForest)
# usa estos tipos, que hay que declarar como confiables (validado con MLflow 3.17).
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]

# ── Validación cruzada y tuning (Unidad 06) ──────────────────────────────────
CV_FOLDS = 5
SCORING  = "roc_auc"       # AUC: no depende de un umbral fijo (con 2.6% positivos el F1 engaña)

# ── Quality gate (Unidades 04 y 10) ──────────────────────────────────────────
# Un modelo solo se registra si supera AMBOS umbrales en el test.
AUC_MINIMO       = 0.65    # capacidad de ordenar compradores vs no compradores
LIFT_TOP10_MINIMO = 2.0    # el top 10% del ranking debe comprar ≥ 2x la tasa promedio

# ── Crear directorios necesarios ─────────────────────────────────────────────
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
