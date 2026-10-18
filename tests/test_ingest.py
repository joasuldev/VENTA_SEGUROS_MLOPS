"""
Tests de la Etapa 1 (pipeline/01_ingest.py).

Ejecutar desde la raíz del proyecto:  pytest -v
Usan datos pequeños inventados → corren en segundos y no dependen del CSV real
(salvo el último test, que se salta si el CSV no está).
"""
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

# El archivo se llama "01_ingest.py": un nombre que empieza con número no se puede
# importar con "import", así que se carga por ruta (mismo estilo numerado de la Unidad 06).
_spec = importlib.util.spec_from_file_location("ingest", ROOT / "pipeline" / "01_ingest.py")
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)


def _cliente(**cambios):
    """Fila base de un cliente; 'cambios' permite variar columnas puntuales."""
    fila = {"Mto_TC": 3000, "MARCA": "Visa", "Nombre_territorio": "T.CENTRO",
            "FLAG_LIMA_PROVINCIA": 0.0, "REGION": "CENTRO", "SUELDO_ESTIMADO": 1500.0,
            "EDAD": 30.0, "SEXO": "M", "ANTIGUEDAD_MES": 40.0, "SEGMENTO": "CLASICO",
            "FLAG_SS": None, "FLAG_UNICEF": None}
    fila.update(cambios)
    return fila


@pytest.fixture
def df_mini():
    """60 clientes distintos (12 compradores), cada uno con 2 fotos mensuales."""
    filas = []
    for i in range(60):
        compra = 1.0 if i % 5 == 0 else None
        for mes in (40.0, 41.0):  # misma persona, antigüedad +1 mes
            filas.append(_cliente(Mto_TC=1000 + i, ANTIGUEDAD_MES=mes, FLAG_SS=compra))
    return pd.DataFrame(filas)


def test_target_vacio_se_convierte_en_cero(df_mini):
    out = ingest.limpiar(df_mini)
    assert set(out[C.TARGET].unique()) == {0, 1}
    assert out[C.TARGET].sum() == 24          # 12 compradores x 2 fotos


def test_elimina_duplicados_exactos():
    df = pd.DataFrame([_cliente(), _cliente(), _cliente(Mto_TC=9999)])
    assert len(ingest.limpiar(df)) == 2


def test_fotos_mensuales_del_mismo_cliente_comparten_id(df_mini):
    out = ingest.crear_cliente_id(ingest.limpiar(df_mini))
    assert out[C.ID_COL].nunique() == 60      # 120 filas, 60 personas


def test_split_sin_clientes_compartidos(df_mini):
    df = ingest.crear_cliente_id(ingest.limpiar(df_mini))
    train, test = ingest.separar_por_cliente(df)
    assert set(train[C.ID_COL]).isdisjoint(set(test[C.ID_COL]))
    assert len(train) + len(test) == len(df)  # no se pierde ninguna fila


def test_validacion_detecta_fuga(df_mini):
    df = ingest.crear_cliente_id(ingest.limpiar(df_mini))
    train, test = ingest.separar_por_cliente(df)
    test_con_fuga = pd.concat([test, train.head(2)])   # metemos clientes de train en test
    monkey_min = C.MIN_FILAS
    C.MIN_FILAS = 0                                     # el mini dataset tiene pocas filas
    try:
        with pytest.raises(ValueError, match="FUGA"):
            ingest.validar_salida(train, test_con_fuga)
    finally:
        C.MIN_FILAS = monkey_min


def test_validacion_detecta_columna_faltante(df_mini):
    df = ingest.crear_cliente_id(ingest.limpiar(df_mini))
    train, test = ingest.separar_por_cliente(df)
    with pytest.raises(ValueError, match="COLUMNAS FALTANTES"):
        ingest.validar_salida(train.drop(columns=["REGION"]), test)


@pytest.mark.skipif(not C.RAW_DATA_PATH.exists(), reason="CSV real no disponible")
def test_ingesta_con_datos_reales():
    """Prueba de integración: la etapa completa corre con el CSV del docente."""
    train, test = ingest.run()
    assert set(train[C.ID_COL]).isdisjoint(set(test[C.ID_COL]))
    # La tasa de compra debe parecerse en ambos lados gracias a la estratificación
    assert abs(train[C.TARGET].mean() - test[C.TARGET].mean()) < 0.01
