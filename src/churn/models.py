"""Définition des modèles, grilles d'hyperparamètres et construction des pipelines."""
from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_models(seed: int) -> dict:
    """Retourne les modèles comparés.

    - logreg        : baseline simple et interprétable
    - random_forest : ensemble d'arbres, non-linéaire
    - hist_gb       : gradient boosting (souvent le plus performant)
    - mlp           : réseau de neurones (Deep Learning, exigence du sujet)
    """
    return {
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }


def param_grids() -> dict:
    """Petites grilles d'hyperparamètres (recherche maîtrisée, non exhaustive)."""
    return {
        "logreg": {"model__C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 5],
        },
        "hist_gb": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [31, 63],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(64, 32), (128, 64)],
            "model__alpha": [1e-4, 1e-3],
        },
    }


def build_pipeline(preprocessor, model, sampler=None) -> ImbPipeline:
    """Assemble préprocesseur + (sampler) + modèle dans un pipeline imblearn.

    Le sampler n'est appliqué qu'à l'étape `fit` (jamais au moment de prédire),
    ce qui le rend sûr vis-à-vis de la validation croisée et de l'inférence.
    """
    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)
