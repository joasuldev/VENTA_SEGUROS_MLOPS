# Dockerfile — API Seguro Antirrobos (patrón Unidad 05)
# Build: docker build -t seguro-antirrobos-api .
# Run  : docker run -p 8000:8000 seguro-antirrobos-api

# Misma versión de Python con la que se entrenó el modelo en el Codespace (3.14):
# el pickle se carga sin sorpresas de compatibilidad.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencias primero: Docker cachea esta capa y no reinstala si solo cambia el código
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Código de la API
COPY api/ ./api/

# 3) Artefactos del pipeline: el modelo (Pipeline completo) y sus métricas/cortes
COPY artifacts/modelo.pkl artifacts/metrics.json ./artifacts/

EXPOSE 8000

# Docker revisa periódicamente que la API responda (usa urllib: sin dependencias extra)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
