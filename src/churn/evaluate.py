"""Métriques, optimisation du seuil et figures d'évaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Calcule les métriques adaptées à un problème déséquilibré.

    Les métriques indépendantes du seuil (ROC-AUC, PR-AUC) utilisent la probabilité ;
    les métriques de classification (F1, recall, precision, accuracy) utilisent le seuil.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Recherche le seuil de décision optimisant la métrique choisie.

    Par défaut maximise le F1 ; 'recall' privilégie la détection des churners.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        if metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


# --------------------------------------------------------------------------- #
# Figures (sauvegardées pour le rapport)
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_proba, threshold, path) -> None:
    import matplotlib.pyplot as plt

    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Reste (0)", "Churn (1)"])
    ax.set_yticklabels(["Reste (0)", "Churn (1)"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion (seuil={threshold:.2f})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_roc_pr(y_true, y_proba, roc_path, pr_path) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Faux positifs"); ax.set_ylabel("Vrais positifs")
    ax.set_title("Courbe ROC"); ax.legend()
    fig.tight_layout(); fig.savefig(roc_path, dpi=120); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    base = float(np.mean(y_true))
    ax.axhline(base, ls="--", color="grey", label=f"Aléatoire = {base:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Courbe Precision-Recall"); ax.legend()
    fig.tight_layout(); fig.savefig(pr_path, dpi=120); plt.close(fig)
