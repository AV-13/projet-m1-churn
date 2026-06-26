"""Stratégies de gestion du déséquilibre des classes."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def get_samplers(seed: int) -> dict:
    """Retourne les samplers comparés dans l'étude du déséquilibre.

    `None` correspond à l'absence de rééchantillonnage (baseline).
    """
    return {
        "none": None,
        "smote": SMOTE(random_state=seed),
        "oversample": RandomOverSampler(random_state=seed),
        "undersample": RandomUnderSampler(random_state=seed),
    }
