"""
Metricas para mantenimiento predictivo - nunca accuracy sola.

El desbalanceo (3.4% falla) hace que accuracy sea enganosa,
asi que reportamos precision, recall, F1, ROC-AUC y matriz de confusion.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Calcula todo lo que importa cuando la clase positiva es rara.
    Copias defensivas para no mutar entradas.
    """
    y_true = np.array(y_true, copy=True).ravel()
    y_pred = np.array(y_pred, copy=True).ravel()

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),  # [[tn, fp],[fn, tp]]
    }

    if y_proba is not None:
        y_proba = np.array(y_proba, copy=True).ravel()
        # roc_auc necesita ambas clases en y_true
        if len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            out["roc_auc"] = float("nan")

    return out


def plot_confusion_matrix(y_true, y_pred, save_path: str = "results/matriz_confusion.png", labels=("No falla", "Falla")):
    """Guarda matriz de confusion con numeros."""
    y_true = np.array(y_true, copy=True).ravel()
    y_pred = np.array(y_pred, copy=True).ravel()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_roc_curve(y_true, y_proba, save_path: str = "results/curva_roc.png", label: str = "modelo"):
    """Guarda curva ROC."""
    y_true = np.array(y_true, copy=True).ravel()
    y_proba = np.array(y_proba, copy=True).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="azar")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
