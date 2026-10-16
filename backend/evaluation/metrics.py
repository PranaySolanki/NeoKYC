from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


def calculate_confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    """
    Computes confusion matrix entries: True Positives (TP), False Positives (FP),
    True Negatives (TN), and False Negatives (FN).
    
    Positive class = 1 (Manipulated / Fake)
    Negative class = 0 (Real)
    """
    preds = (scores >= threshold).astype(np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calculate_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Area Under the ROC Curve via trapezoidal integration.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 0.5

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = np.concatenate([[0.0], tps / pos_count])
    fpr = np.concatenate([[0.0], fps / neg_count])

    auc = np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]))
    return float(np.clip(auc, 0.0, 1.0))


def calculate_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, Dict[str, int]]]:
    """
    Calculates full set of classification metrics:
      - Accuracy
      - Precision
      - Recall (Sensitivity / TPR)
      - Specificity (TNR)
      - F1-Score
      - ROC-AUC
      - Confusion Matrix (TP, FP, TN, FN)
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    cm = calculate_confusion_matrix(scores, labels, threshold=threshold)

    total = len(labels)
    acc = float((cm["tp"] + cm["tn"]) / total) if total > 0 else 0.0
    prec = float(cm["tp"] / (cm["tp"] + cm["fp"])) if (cm["tp"] + cm["fp"]) > 0 else 0.0
    rec = float(cm["tp"] / (cm["tp"] + cm["fn"])) if (cm["tp"] + cm["fn"]) > 0 else 0.0
    spec = float(cm["tn"] / (cm["tn"] + cm["fp"])) if (cm["tn"] + cm["fp"]) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    auc = calculate_roc_auc(scores, labels)

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


def calibrate_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    criterion: str = "f1",
    num_steps: int = 200,
) -> Tuple[float, float]:
    """
    Calibrates the decision threshold using VALIDATION data only.
    
    Supported criteria:
      - 'f1': Maximizes validation F1-score (harmonic mean of precision and recall)
      - 'youden': Maximizes Youden's J statistic (Sensitivity + Specificity - 1)
      - 'eer': Equal Error Rate (minimizes |FPR - (1 - TPR)|)
      
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    val_scores = np.asarray(val_scores, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int32)

    best_threshold = 0.5
    best_value = -1.0

    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        metrics = calculate_classification_metrics(val_scores, val_labels, threshold=thresh)
        if criterion == "f1":
            val = metrics["f1"]
        elif criterion == "youden":
            val = metrics["recall"] + metrics["specificity"] - 1.0
        elif criterion == "eer":
            # Distance from EER point
            val = -abs(metrics["specificity"] - metrics["recall"])
        elif criterion == "accuracy":
            val = metrics["accuracy"]
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if val > best_value:
            best_value = val
            best_threshold = float(thresh)

    return best_threshold, best_value


def aggregate_video_scores(frame_scores: List[float], method: str = "mean") -> float:
    """
    Aggregates frame-level manipulation probabilities into a video-level score.
    
    Methods:
      - 'mean': Standard arithmetic mean (standard in FaceForensics++ literature)
      - 'median': Median frame score (resilient to isolated transient face detection errors)
      - 'max': Maximum score (strict zero-trust heuristic: any high-confidence fake triggers alert)
    """
    if not frame_scores:
        return 0.5
    arr = np.asarray(frame_scores, dtype=np.float64)
    if method == "mean":
        return float(np.mean(arr))
    elif method == "median":
        return float(np.median(arr))
    elif method == "max":
        return float(np.max(arr))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
