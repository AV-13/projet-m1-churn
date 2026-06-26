"""Sérialisation et chargement des artefacts du modèle."""
from __future__ import annotations

import json

import joblib

from .config import resolve

MODEL_FILE = "final_model.joblib"
META_FILE = "metadata.json"


def save_artifacts(pipeline, metadata: dict, models_dir: str = "models") -> None:
    """Sauvegarde le pipeline complet (préprocessing + modèle) et ses métadonnées."""
    out = resolve(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / MODEL_FILE)
    with open(out / META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_artifacts(models_dir: str = "models"):
    """Charge le pipeline et les métadonnées. Lève FileNotFoundError si absent."""
    out = resolve(models_dir)
    pipeline = joblib.load(out / MODEL_FILE)
    with open(out / META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata
