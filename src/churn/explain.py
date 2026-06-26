"""Interprétabilité : importance par permutation et SHAP (best-effort)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_report(pipeline, X_test, y_test, path, top_n: int = 15):
    """Calcule l'importance par permutation (agnostique au modèle) et trace le top N."""
    import matplotlib.pyplot as plt

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=0,
        scoring="average_precision", n_jobs=-1,
    )
    importances = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
    )
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#2c7fb8")
    ax.set_xlabel("Baisse de PR-AUC (importance)")
    ax.set_title("Importance des variables (permutation)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return importances


def shap_summary(pipeline, X_sample, path, max_display: int = 15) -> bool:
    """Trace un résumé SHAP sur un échantillon. Renvoie False si SHAP échoue."""
    try:
        import matplotlib.pyplot as plt
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        X_trans = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.Explainer(model, X_trans, feature_names=feature_names)
        shap_values = explainer(X_trans)
        # Pour la classification binaire, certains explainers renvoient 2 sorties.
        plt.figure()
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                          max_display=max_display, show=False)
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return True
    except Exception as exc:  # pragma: no cover - SHAP peut échouer selon le modèle
        print(f"[explain] SHAP indisponible ({type(exc).__name__}: {exc}) — "
              f"on s'appuie sur l'importance par permutation.")
        return False
