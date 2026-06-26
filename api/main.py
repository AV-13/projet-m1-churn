"""Application FastAPI exposant le service de prédiction du churn."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import CustomerFeatures, HealthResponse, PredictionResponse
from .service import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charge le modèle au démarrage (et non à chaque requête).
    try:
        service.load()
    except FileNotFoundError:
        print("[api] Artefacts introuvables — lancez `make train` d'abord.")
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Service d'inférence pour la prédiction du churn client.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if service.is_loaded else "model_not_loaded",
        model_loaded=service.is_loaded,
        model_name=service.metadata.get("model_name"),
        model_version=service.metadata.get("model_version"),
    )


@app.get("/model-info")
def model_info():
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    meta = service.metadata
    return {
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "balancing": meta.get("balancing"),
        "threshold": meta.get("threshold"),
        "n_features": len(meta.get("features", [])),
        "metrics_test": meta.get("metrics_test_tuned"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Lancez l'entraînement.")
    try:
        result = service.predict(customer.model_dump())
    except Exception as exc:  # erreurs d'inférence inattendues
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {exc}")
    return PredictionResponse(**result)
