"""Pipeline d'entraînement de bout en bout.

Étapes :
  1. Étude du déséquilibre (baseline LogReg × stratégies de rééquilibrage)
  2. Comparaison multi-modèles (CV stratifiée, métrique de sélection = PR-AUC)
  3. Recherche d'hyperparamètres maîtrisée sur le meilleur modèle
  4. Optimisation du seuil de décision (probabilités out-of-fold)
  5. Évaluation sur le test + figures
  6. Interprétabilité (permutation + SHAP)
  7. Sérialisation des artefacts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from . import evaluate as ev
from . import explain as ex
from .balancing import get_samplers
from .config import load_config, resolve
from .data import get_feature_columns, load_data, split_data
from .models import build_pipeline, get_models, param_grids
from .persistence import save_artifacts
from .preprocessing import build_preprocessor

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
}


def _ensure_dirs(cfg):
    (resolve(cfg["paths"]["reports_dir"], "metrics")).mkdir(parents=True, exist_ok=True)
    (resolve(cfg["paths"]["reports_dir"], "figures")).mkdir(parents=True, exist_ok=True)
    (resolve(cfg["paths"]["models_dir"])).mkdir(parents=True, exist_ok=True)


def imbalance_study(preprocessor, X_train, y_train, cv, seed) -> pd.DataFrame:
    """Compare les stratégies de rééquilibrage sur une régression logistique."""
    from sklearn.linear_model import LogisticRegression

    rows = []
    for name, sampler in get_samplers(seed).items():
        # class_weight uniquement quand on ne rééchantillonne pas (sinon redondant)
        cw = "balanced" if name == "none" else None
        model = LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)
        pipe = build_pipeline(preprocessor, model, sampler)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({
            "strategie": name,
            **{m: scores[f"test_{m}"].mean() for m in SCORING},
        })
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df


def model_study(preprocessor, X_train, y_train, cv, seed, balancing) -> pd.DataFrame:
    """Compare les modèles avec la stratégie de rééquilibrage retenue."""
    sampler = get_samplers(seed)[balancing]
    rows = []
    for name, model in get_models(seed).items():
        pipe = build_pipeline(preprocessor, model, sampler)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({
            "modele": name,
            **{m: scores[f"test_{m}"].mean() for m in SCORING},
            **{f"{m}_std": scores[f"test_{m}"].std() for m in SCORING},
        })
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df


def main():
    cfg = load_config()
    seed = cfg["seed"]
    _ensure_dirs(cfg)
    rng = np.random.RandomState(seed)

    print("=== Chargement des données ===")
    df = load_data(cfg)
    features, numeric, categorical = get_feature_columns(df, cfg)
    print(f"{df.shape[0]} lignes, {len(features)} features "
          f"({len(numeric)} num / {len(categorical)} cat)")
    print(f"Taux de churn : {df[cfg['data']['target']].mean():.1%}")

    X_train, X_test, y_train, y_test = split_data(df, cfg)
    preprocessor = build_preprocessor(numeric, categorical)
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

    # 1. Étude du déséquilibre -------------------------------------------------
    print("\n=== 1. Étude du déséquilibre (baseline LogReg) ===")
    imb = imbalance_study(preprocessor, X_train, y_train, cv, seed)
    print(imb.round(4).to_string(index=False))
    imb.to_csv(resolve(cfg["paths"]["reports_dir"], "metrics", "imbalance_study.csv"),
               index=False)

    # 2. Comparaison multi-modèles --------------------------------------------
    balancing = cfg["training"]["balancing"]
    print(f"\n=== 2. Comparaison des modèles (rééquilibrage = {balancing}) ===")
    comp = model_study(preprocessor, X_train, y_train, cv, seed, balancing)
    print(comp.round(4).to_string(index=False))
    comp.to_csv(resolve(cfg["paths"]["reports_dir"], "metrics", "model_comparison.csv"),
                index=False)

    best_name = comp.iloc[0]["modele"]
    print(f"\nMeilleur modèle (PR-AUC) : {best_name}")

    # 3. Recherche d'hyperparamètres ------------------------------------------
    print(f"\n=== 3. Recherche d'hyperparamètres sur {best_name} ===")
    sampler = get_samplers(seed)[balancing]
    base_pipe = build_pipeline(preprocessor, get_models(seed)[best_name], sampler)
    grid = param_grids().get(best_name, {})
    search = GridSearchCV(base_pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"Meilleurs paramètres : {search.best_params_}")
    print(f"PR-AUC (CV) : {search.best_score_:.4f}")

    # 4. Optimisation du seuil (probabilités out-of-fold sur le train) --------
    print("\n=== 4. Optimisation du seuil de décision ===")
    oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    threshold = ev.find_best_threshold(y_train, oof_proba,
                                       metric=cfg["training"]["threshold_metric"])
    print(f"Seuil optimal ({cfg['training']['threshold_metric']}) : {threshold:.3f}")

    # 5. Évaluation finale sur le test ----------------------------------------
    print("\n=== 5. Évaluation sur le jeu de test ===")
    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    metrics_default = ev.compute_metrics(y_test, test_proba, threshold=0.5)
    metrics_tuned = ev.compute_metrics(y_test, test_proba, threshold=threshold)
    print("Seuil 0.50 :", {k: round(v, 4) for k, v in metrics_default.items()})
    print("Seuil opt. :", {k: round(v, 4) for k, v in metrics_tuned.items()})

    fig_dir = resolve(cfg["paths"]["reports_dir"], "figures")
    ev.plot_confusion(y_test, test_proba, threshold, fig_dir / "confusion_matrix.png")
    ev.plot_roc_pr(y_test, test_proba, fig_dir / "roc_curve.png", fig_dir / "pr_curve.png")

    # 6. Interprétabilité ------------------------------------------------------
    print("\n=== 6. Interprétabilité ===")
    ex.permutation_importance_report(best_pipe, X_test, y_test,
                                     fig_dir / "permutation_importance.png")
    sample = X_train.sample(min(500, len(X_train)), random_state=seed)
    ex.shap_summary(best_pipe, sample, fig_dir / "shap_summary.png")

    # 7. Sérialisation ---------------------------------------------------------
    print("\n=== 7. Sérialisation des artefacts ===")
    metadata = {
        "model_version": cfg["model_version"],
        "model_name": best_name,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "balancing": balancing,
        "threshold": threshold,
        "features": list(X_train.columns),
        "numeric": numeric,
        "categorical": categorical,
        "metrics_test_tuned": metrics_tuned,
        "metrics_test_default": metrics_default,
        "churn_rate": float(df[cfg["data"]["target"]].mean()),
    }
    save_artifacts(best_pipe, metadata, cfg["paths"]["models_dir"])
    print(f"Artefacts sauvegardés dans {resolve(cfg['paths']['models_dir'])}/")
    print("\n✅ Entraînement terminé.")


if __name__ == "__main__":
    main()
