"""Metricas de evaluacion: AUC-ROC, AP, Accuracy, ROC curve, confusion matrix."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict:
    """AUC-ROC, Average Precision y Accuracy al threshold dado."""
    preds = (scores >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(targets, scores)),
        "average_precision": float(average_precision_score(targets, scores)),
        "accuracy": float(accuracy_score(targets, preds)),
        "threshold": threshold,
        "n_samples": int(len(targets)),
        "n_positive": int(targets.sum()),
    }


def plot_roc_curve(
    targets: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve",
) -> None:
    fpr, tpr, _ = roc_curve(targets, scores)
    auc = roc_auc_score(targets, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    targets: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
) -> None:
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict, save_path: Path) -> None:
    with save_path.open("w") as f:
        json.dump(metrics, f, indent=2)
