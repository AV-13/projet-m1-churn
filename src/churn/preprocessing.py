"""Pipeline de préprocessing (anti-fuite de données).

Le préprocesseur est intégré dans le pipeline modèle et `fit` uniquement sur le
jeu d'entraînement (y compris à l'intérieur de la validation croisée), garantissant
l'absence de fuite de données vers le test.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Construit le ColumnTransformer numérique + catégoriel."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            # Les NaN catégoriels (ex. complaint_type) deviennent une modalité "None".
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )
