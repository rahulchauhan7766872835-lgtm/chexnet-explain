from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_clinical_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[float]
) -> Dict[str, float]:
    """
    Computes diagnostic metrics for medical radiograph triage:
      - Accuracy
      - Sensitivity (Recall)
      - Specificity
      - Precision
      - F1-Score
      - Area Under ROC Curve (AUROC)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix extraction
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    # ROC AUC (only if both classes present in y_true)
    if len(np.unique(y_true_arr)) > 1:
        try:
            auroc = float(roc_auc_score(y_true_arr, y_probs_arr))
        except ValueError:
            auroc = 0.5
    else:
        auroc = 1.0

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(rec, 4),      # Recall
        "specificity": round(specificity, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "tp": int(tp) if cm.shape == (2, 2) else 0,
        "fp": int(fp) if cm.shape == (2, 2) else 0,
        "tn": int(tn) if cm.shape == (2, 2) else 0,
        "fn": int(fn) if cm.shape == (2, 2) else 0,
    }
