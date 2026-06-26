"""Chargement de la configuration et résolution des chemins.

Tous les chemins de `config.yaml` sont relatifs à la racine du projet ; ce module
les résout en chemins absolus afin que le code fonctionne quel que soit le
répertoire d'exécution.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# src/churn/config.py -> parents[2] == racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Charge config.yaml et renvoie un dictionnaire."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(*parts: str | Path) -> Path:
    """Résout un chemin relatif à la racine du projet."""
    return PROJECT_ROOT.joinpath(*[str(p) for p in parts])
