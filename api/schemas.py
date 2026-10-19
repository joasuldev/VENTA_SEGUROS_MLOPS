"""
api/schemas.py — Contratos de entrada y salida (Pydantic), patrón de la Unidad 05.

Pydantic valida tipos y rangos automáticamente: si un dato es inválido la API
responde HTTP 422 con un mensaje claro y el modelo nunca recibe datos corruptos.

Los campos que en el dataset real pueden venir vacíos se declaran opcionales:
el Pipeline del modelo los imputa igual que en el entrenamiento.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    """Datos de un cliente de la financiera (mismas columnas que el entrenamiento)."""
    Mto_TC: float = Field(..., ge=0, description="Monto de la oferta de tarjeta de crédito (S/)")
    SUELDO_ESTIMADO: Optional[float] = Field(None, gt=0, description="Sueldo estimado (S/)")
    EDAD: Optional[int] = Field(None, ge=18, le=100, description="Edad del cliente")
    ANTIGUEDAD_MES: Optional[float] = Field(None, ge=0, description="Meses de antigüedad como cliente")
    Nombre_territorio: str = Field(..., description="Territorio comercial, ej. T.NOR ORIENTE")
    REGION: Optional[str] = Field(None, description="Región de residencia, ej. ORIENTE")
    SEGMENTO: Optional[str] = Field(None, description="CLASICO | BAJO VALOR | VIP | Micro empresa | ...")
    MARCA: str = Field("Visa", description="Operador de la tarjeta: Visa | MasterCard")

    # Ejemplo que aparece precargado en /docs (Swagger): perfil de alta propensión
    model_config = {"json_schema_extra": {"example": {
        "Mto_TC": 3000, "SUELDO_ESTIMADO": 1500, "EDAD": 32, "ANTIGUEDAD_MES": 35,
        "Nombre_territorio": "T.NOR ORIENTE", "REGION": "ORIENTE",
        "SEGMENTO": "BAJO VALOR", "MARCA": "Visa",
    }}}


class PrediccionOutput(BaseModel):
    """Respuesta orientada a la acción comercial."""
    score_propension: float = Field(..., description="Score 0-1 para ORDENAR clientes (no es una probabilidad calibrada)")
    prioridad: Literal["ALTA", "MEDIA", "BAJA"]
    accion_sugerida: str
    modelo: str
    version_modelo: str


class HealthResponse(BaseModel):
    """Estado de la API y del modelo cargado."""
    status: str
    modelo: str
    version_modelo: str
    auc_test: float
    lift_top10: float
