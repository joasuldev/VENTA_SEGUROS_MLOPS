"""
Etapa 5 — Comparar runs con mlflow.search_runs().

Entrada : experimento "seguro-antirrobos" en MLflow
Salida  : artifacts/comparacion_runs.csv · artifacts/comparacion_runs.png
Ejecutar: python pipeline/05_comparar_runs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import matplotlib
matplotlib.use("Agg")            # sin pantalla (Codespaces / CI)
import matplotlib.pyplot as plt

import utils as U
import config as C

log = U.get_logger("COMPARAR")

COLUMNAS = ["tags.mlflow.runName", "params.metodo_busqueda", "metrics.auc",
            "metrics.lift_top10", "metrics.captura_top20", "metrics.cv_auc",
            "tags.quality_gate", "start_time"]


def obtener_ultimos_runs():
    """Último run de cada tipo (base, grid, random): si re-ejecutas, compara lo más reciente."""
    U.configurar_mlflow()
    runs = mlflow.search_runs(experiment_names=[C.MLFLOW_EXPERIMENT],
                              order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError("No hay runs. Ejecuta primero las etapas 02, 03 y 04.")
    ultimos = runs.drop_duplicates(subset="tags.mlflow.runName", keep="first")
    return ultimos.sort_values("metrics.auc", ascending=False)


def run():
    log.info("=== ETAPA 5: COMPARAR RUNS ===")
    runs = obtener_ultimos_runs()
    tabla = runs[[c for c in COLUMNAS if c in runs.columns]]
    log.info("\n%s", tabla.drop(columns=["start_time"], errors="ignore").to_string(index=False))
    tabla.to_csv(C.ARTIFACTS_DIR / "comparacion_runs.csv", index=False)

    # Gráfico: 3 métricas lado a lado; la línea roja es el umbral del quality gate
    nombres = runs["tags.mlflow.runName"].tolist()
    paneles = [("metrics.auc", "AUC (test)", C.AUC_MINIMO),
               ("metrics.lift_top10", "Lift top 10%", C.LIFT_TOP10_MINIMO),
               ("metrics.captura_top20", "Captura de compradores con top 20%", None)]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Comparación de runs — Seguro Antirrobos (test por cliente)", fontweight="bold")
    for ax, (col, titulo, umbral) in zip(ejes, paneles):
        valores = runs[col].tolist()
        barras = ax.bar(nombres, valores, color=["#2E75B6", "#C55A11", "#1E6E38"][:len(nombres)])
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontweight="bold")
        if umbral is not None:
            ax.axhline(umbral, color="red", linestyle="--", label=f"Quality gate ({umbral})")
            ax.legend(fontsize=8)
        ax.set_title(titulo, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    salida = C.ARTIFACTS_DIR / "comparacion_runs.png"
    plt.savefig(salida, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Gráfico guardado: %s", salida)
    log.info("=== ETAPA 5 COMPLETADA ===")


if __name__ == "__main__":
    run()
