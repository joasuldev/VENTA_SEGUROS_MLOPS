"""
api/app.py — API REST del modelo de propensión al Seguro Antirrobos.

Endpoints:
    GET  /          → información básica
    GET  /health    → estado de la API y métricas del modelo en servicio
    POST /predecir  → score de propensión + prioridad + acción comercial
    GET  /docs      → Swagger UI (autogenerado por FastAPI)

Ejecutar en Codespace:  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
Con Docker:             docker compose up --build
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import ClienteInput, PrediccionOutput, HealthResponse
from api.predictor import predictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | API | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar (no en cada request): las predicciones son rápidas."""
    predictor.cargar()
    yield


app = FastAPI(
    title="API Seguro Antirrobos — Propensión de compra",
    description="Prioriza a qué clientes ofrecer el seguro antirrobos "
                "(billetera S/50, celular S/350, laptop S/680).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Info"])
def root():
    return {"api": "Seguro Antirrobos — Propensión de compra", "docs": "/docs",
            "health": "/health", "prediccion": "POST /predecir"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    m = predictor.metricas
    return HealthResponse(status="ok", modelo="RandomForest (Pipeline sklearn)",
                          version_modelo=m.get("run_name", "desconocida"),
                          auc_test=m.get("auc", 0.0), lift_top10=m.get("lift_top10", 0.0))


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(cliente: ClienteInput):
    """Recibe los datos crudos de un cliente (se aceptan nulos) y devuelve su prioridad."""
    try:
        resultado = predictor.predecir(cliente.model_dump())
        log.info("score=%.4f prioridad=%s", resultado["score_propension"], resultado["prioridad"])
        return PrediccionOutput(**resultado)
    except Exception as e:
        log.error("Error en predicción: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
