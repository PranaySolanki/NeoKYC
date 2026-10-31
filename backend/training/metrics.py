from typing import Dict, List, Tuple
import torch
import numpy as np


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using trapezoidal numerical integration on sorted prediction scores.
    
    Args:
        scores: Predicted probabilities or logits for positive class (Manipulated).
        labels: Binary ground truth (0 = Real, 1 = Manipulated).
        
    Returns:
        AUC float in [0.0, 1.0]. Returns 0.5 if all labels are identical.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 0.5

    # Sort descending by score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Cumulative True Positives and False Positives
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    # TPR and FPR rates
    tpr = np.concatenate([[0.0], tps / pos_count])
    fpr = np.concatenate([[0.0], fps / neg_count])

    # Trapezoidal integration along FPR axis: sum( 0.5 * (y_k + y_{k-1}) * (x_k - x_{k-1}) )
    auc = np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]))
    return float(np.clip(auc, 0.0, 1.0))


def compute_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, F1, and AUC for binary deepfake detection.
    
    Args:
        logits: Output logits of shape (N, 2) or (N, 1)
        targets: Ground truth binary labels of shape (N,)
        threshold: Classification threshold for positive class
    """
    if logits.ndim == 2 and logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)
    else:
        probs = logits.detach().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)

    labels = targets.detach().cpu().numpy().astype(np.int32)

    correct = np.sum(preds == labels)
    total = len(labels)
    acc = float(correct / total) if total > 0 else 0.0

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    auc = compute_roc_auc(probs, labels)

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
