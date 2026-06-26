"""Chargement des données et découpage train/test stratifié."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve


def load_data(cfg: dict) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(resolve(cfg["data"]["path"]))


def get_feature_columns(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Détermine les colonnes features et les sépare en numériques / catégorielles.

    L'identifiant et la cible sont exclus. Le typage est inféré depuis le DataFrame,
    ce qui rend le pipeline robuste au schéma exact du dataset.
    """
    target = cfg["data"]["target"]
    id_col = cfg["data"]["id_column"]
    drop = {target, id_col}
    features = [c for c in df.columns if c not in drop]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return features, numeric, categorical


def split_data(df: pd.DataFrame, cfg: dict):
    """Découpe en train/test de manière stratifiée sur la cible."""
    target = cfg["data"]["target"]
    id_col = cfg["data"]["id_column"]
    X = df.drop(columns=[target, id_col])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=cfg["seed"],
    )
