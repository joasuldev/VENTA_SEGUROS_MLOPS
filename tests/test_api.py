"""
Tests de la API (patrón tests/test_api.py de la Unidad 05).

Requieren el modelo entrenado (artifacts/modelo.pkl): si no existe, se saltan.
Ejecutar desde la raíz:  pytest -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "modelo.pkl").exists(),
                                reason="Modelo no entrenado: ejecuta python run_pipeline.py")

# Perfil de ALTA propensión según el análisis: Oriente, bajo valor, joven, poca antigüedad
ALTO = {"Mto_TC": 3000, "SUELDO_ESTIMADO": 1500, "EDAD": 32, "ANTIGUEDAD_MES": 35,
        "Nombre_territorio": "T.NOR ORIENTE", "REGION": "ORIENTE", "SEGMENTO": "BAJO VALOR", "MARCA": "Visa"}
# Perfil de BAJA propensión: VIP, banca premium, línea alta y mucha antigüedad
BAJO = {"Mto_TC": 20000, "SUELDO_ESTIMADO": 9000, "EDAD": 50, "ANTIGUEDAD_MES": 200,
        "Nombre_territorio": "T.BANCA PREMIUM", "REGION": "LIMA MODERNA", "SEGMENTO": "VIP", "MARCA": "Visa"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.app import app
    with TestClient(app) as c:      # el "with" dispara el lifespan → carga el modelo
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["auc_test"] >= 0.65            # el modelo servido pasó el quality gate


def test_root_info(client):
    assert client.get("/").json()["prediccion"] == "POST /predecir"


def test_predecir_estructura(client):
    r = client.post("/predecir", json=ALTO)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score_propension"] <= 1.0
    assert body["prioridad"] in {"ALTA", "MEDIA", "BAJA"}


def test_perfil_alto_supera_perfil_bajo(client):
    """El ranking debe tener sentido de negocio."""
    alto = client.post("/predecir", json=ALTO).json()["score_propension"]
    bajo = client.post("/predecir", json=BAJO).json()["score_propension"]
    assert alto > bajo


def test_prioridad_coherente_con_score(client):
    from api.predictor import predictor
    body = client.post("/predecir", json=ALTO).json()
    assert body["prioridad"] == predictor.prioridad(body["score_propension"])


def test_acepta_nulos_y_categoria_nueva(client):
    """Ventaja del Pipeline único: imputa nulos y tolera territorios no vistos."""
    cliente = {"Mto_TC": 3000, "Nombre_territorio": "TERRITORIO_NUEVO", "MARCA": "Visa"}
    assert client.post("/predecir", json=cliente).status_code == 200


def test_score_determinista(client):
    s1 = client.post("/predecir", json=ALTO).json()["score_propension"]
    s2 = client.post("/predecir", json=ALTO).json()["score_propension"]
    assert s1 == s2


def test_valida_edad_menor_18(client):
    assert client.post("/predecir", json={**ALTO, "EDAD": 15}).status_code == 422


def test_valida_monto_negativo(client):
    assert client.post("/predecir", json={**ALTO, "Mto_TC": -100}).status_code == 422


def test_falta_campo_obligatorio(client):
    sin_territorio = {k: v for k, v in ALTO.items() if k != "Nombre_territorio"}
    assert client.post("/predecir", json=sin_territorio).status_code == 422


def test_swagger_disponible(client):
    assert client.get("/docs").status_code == 200
