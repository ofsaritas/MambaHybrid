"""Classification metrics used by train_hybrid / evaluate_hybrid."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def _to_pred(y_pred_probs: np.ndarray) -> np.ndarray:
    if getattr(y_pred_probs, "ndim", 1) > 1:
        return np.argmax(y_pred_probs, axis=1)
    return np.asarray(y_pred_probs)


def compute_metrics(y_true, y_pred_probs) -> dict:
    y_true = np.asarray(y_true)
    y_pred = _to_pred(y_pred_probs)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def cm_report(y_true, y_pred_probs, classes):
    """Return (confusion_matrix_list, classification_report_dict)."""
    y_true = np.asarray(y_true)
    y_pred = _to_pred(y_pred_probs)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[str(c) for c in classes],
        output_dict=True,
        zero_division=0,
    )
    return cm, rep
