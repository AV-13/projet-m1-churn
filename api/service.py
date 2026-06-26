"""Service d'inférence : chargement des artefacts et prédiction."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Rend le package `churn` importable (src/ ajouté au path).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn.persistence import load_artifacts  # noqa: E402


def _risk_level(proba: float) -> str:
    if proba >= 0.66:
        return "élevé"
    if proba >= 0.33:
        return "modéré"
    return "faible"


class ChurnService:
    """Encapsule le pipeline entraîné et la logique de prédiction."""

    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}
        self._loaded = False

    def load(self) -> None:
        self.pipeline, self.metadata = load_artifacts()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def features(self) -> list[str]:
        return self.metadata.get("features", [])

    def predict(self, payload: dict) -> dict:
        if not self._loaded:
            raise RuntimeError("Modèle non chargé.")
        # Construit un DataFrame dans l'ordre exact des features attendues.
        row = {f: payload.get(f) for f in self.features}
        X = pd.DataFrame([row], columns=self.features)
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        prediction = int(proba >= self.threshold)
        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction,
            "threshold": round(self.threshold, 4),
            "risk_level": _risk_level(proba),
            "model_version": self.metadata.get("model_version", "unknown"),
        }


# Instance unique partagée par l'application.
service = ChurnService()
