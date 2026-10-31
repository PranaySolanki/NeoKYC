import os
import json
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader

from ..models.f3net import F3Net
from .metrics import (
    calculate_classification_metrics,
    calibrate_threshold,
    aggregate_video_scores,
)


class F3NetEvaluator:
    """
    Evaluation engine for F3-Net deepfake detection.
    
    Supports:
      - Frame-level metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
      - Video-level aggregation matching FaceForensics++ benchmark protocol
      - Validation-only threshold calibration (fixed when evaluating test set)
      - Comprehensive Markdown and JSON reporting
    """
    def __init__(
        self,
        model: F3Net,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def collect_predictions(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
        """
        Runs dataloader inference and extracts manipulation probabilities and ground truths.
        
        Returns:
            Tuple of (frame_scores, frame_labels, metadata_list)
        """
        frame_scores: List[float] = []
        frame_labels: List[int] = []
        metadata_list: List[Dict[str, str]] = []

        with torch.no_grad():
            for images, labels, meta in dataloader:
                images = images.to(self.device)
                probs = self.model.predict_proba(images)

                # Positive class probability (index 1 = Manipulated)
                prob_fake = probs[:, 1].cpu().numpy()
                lbls = labels.numpy()

                frame_scores.extend(prob_fake.tolist())
                frame_labels.extend(lbls.tolist())

                # Reconstruct per-sample metadata
                batch_size = len(labels)
                for b in range(batch_size):
                    metadata_list.append({
                        k: (v[b] if isinstance(v, (list, tuple)) else str(v[b]))
                        for k, v in meta.items()
                    })

        return np.array(frame_scores), np.array(frame_labels), metadata_list

    def aggregate_by_video(
        self,
        frame_scores: np.ndarray,
        frame_labels: np.ndarray,
        metadata_list: List[Dict[str, str]],
        method: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregates frame predictions per video using the specified aggregation method.
        """
        video_groups = defaultdict(lambda: {"scores": [], "label": None})

        for score, label, meta in zip(frame_scores, frame_labels, metadata_list):
            v_id = meta.get("video_id", "unknown")
            video_groups[v_id]["scores"].append(score)
            if video_groups[v_id]["label"] is None:
                video_groups[v_id]["label"] = int(label)

        video_scores = []
        video_labels = []

        for v_id, entry in sorted(video_groups.items()):
            agg_score = aggregate_video_scores(entry["scores"], method=method)
            video_scores.append(agg_score)
            video_labels.append(entry["label"])

        return np.array(video_scores), np.array(video_labels)

    def evaluate_split(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
        video_agg_method: str = "mean",
    ) -> Dict[str, Union[float, Dict]]:
        """
        Evaluates a dataset split and returns both frame-level and video-level metrics.
        """
        f_scores, f_labels, meta = self.collect_predictions(dataloader)

        if len(f_labels) == 0:
            return {"error": "Dataloader returned 0 samples."}

        # Frame-level metrics
        frame_metrics = calculate_classification_metrics(f_scores, f_labels, threshold=threshold)

        # Video-level metrics
        v_scores, v_labels = self.aggregate_by_video(f_scores, f_labels, meta, method=video_agg_method)
        video_metrics = calculate_classification_metrics(v_scores, v_labels, threshold=threshold)

        return {
            "threshold_used": float(threshold),
            "frame_level": frame_metrics,
            "video_level": {
                "aggregation_method": video_agg_method,
                "total_videos": len(v_labels),
                **video_metrics,
            },
        }

    def calibrate_on_validation(
        self,
        val_dataloader: DataLoader,
        criterion: str = "f1",
    ) -> Tuple[float, Dict[str, Union[float, Dict]]]:
        """
        Calibrates the decision threshold using the validation split.
        Returns the calibrated threshold and the validation evaluation metrics.
        """
        f_scores, f_labels, _ = self.collect_predictions(val_dataloader)
        if len(f_labels) == 0:
            raise ValueError("Validation set is empty, cannot calibrate threshold.")

        optimal_threshold, best_val = calibrate_threshold(f_scores, f_labels, criterion=criterion)
        val_results = self.evaluate_split(val_dataloader, threshold=optimal_threshold)

        return optimal_threshold, val_results

    def generate_markdown_report(
        self,
        results: Dict[str, Union[float, Dict]],
        split_name: str = "Test",
        is_calibrated: bool = False,
    ) -> str:
        """
        Formats evaluation metrics into a clean GitHub-Flavored Markdown report.
        """
        f = results.get("frame_level", {})
        v = results.get("video_level", {})
        cm_f = f.get("confusion_matrix", {})
        cm_v = v.get("confusion_matrix", {})

        thresh = results.get("threshold_used", 0.5)
        cal_status = "Calibrated on Validation Data" if is_calibrated else "Default (0.50)"

        md = f"""# F³-Net Deepfake Detection Evaluation Report ({split_name} Set)

**Threshold Applied:** `{thresh:.4f}` ({cal_status})

---

### Frame-Level Performance
| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | **{f.get('auc', 0.0) * 100:.2f}%** |
| **Accuracy** | **{f.get('accuracy', 0.0) * 100:.2f}%** |
| **Precision** | {f.get('precision', 0.0) * 100:.2f}% |
| **Recall (Sensitivity)** | {f.get('recall', 0.0) * 100:.2f}% |
| **Specificity (TNR)** | {f.get('specificity', 0.0) * 100:.2f}% |
| **F1-Score** | {f.get('f1', 0.0) * 100:.2f}% |

**Frame Confusion Matrix:**
* True Positives (Manipulated detected as Manipulated): **{cm_f.get('tp', 0)}**
* True Negatives (Real detected as Real): **{cm_f.get('tn', 0)}**
* False Positives (Real misclassified as Manipulated): **{cm_f.get('fp', 0)}**
* False Negatives (Manipulated missed): **{cm_f.get('fn', 0)}**

---

### Video-Level Performance (Aggregation: `{v.get('aggregation_method', 'mean')}`)
| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | **{v.get('auc', 0.0) * 100:.2f}%** |
| **Accuracy** | **{v.get('accuracy', 0.0) * 100:.2f}%** |
| **Precision** | {v.get('precision', 0.0) * 100:.2f}% |
| **Recall (Sensitivity)** | {v.get('recall', 0.0) * 100:.2f}% |
| **F1-Score** | {v.get('f1', 0.0) * 100:.2f}% |
| **Total Videos Analyzed** | **{v.get('total_videos', 0)}** |

**Video Confusion Matrix:**
* True Positives: **{cm_v.get('tp', 0)}** | True Negatives: **{cm_v.get('tn', 0)}**
* False Positives: **{cm_v.get('fp', 0)}** | False Negatives: **{cm_v.get('fn', 0)}**
"""

def main():
    import argparse
    from ..data.faceforensics import create_ff_dataloaders

    parser = argparse.ArgumentParser(description="Evaluate F3-Net model on FaceForensics++ benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default="data/FaceForensics++", help="Path to FF++ root")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--compression", type=str, default="c40", choices=["c40", "c23", "raw"], help="Compression level")
    parser.add_argument("--calibrate_on", type=str, default="val", choices=["val", "none"], help="Calibrate threshold on val split")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to output JSON")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config_dict = checkpoint.get("config", {})
    mode = config_dict.get("mode", "Mix")
    num_classes = config_dict.get("num_classes", 2)
    img_size = config_dict.get("img_size", 299)
    use_learnable_filters = config_dict.get("use_learnable_filters", True)

    model = F3Net(
        num_classes=num_classes,
        mode=mode,
        img_size=img_size,
        use_learnable_filters=use_learnable_filters,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    evaluator = F3NetEvaluator(model=model, device=device)

    train_loader, val_loader, test_loader = create_ff_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        compression=args.compression,
    )

    threshold = 0.5
    calibrated = False
    if args.calibrate_on == "val":
        print("Calibrating decision threshold on validation set...")
        threshold, _ = evaluator.calibrate_on_validation(val_loader)
        calibrated = True
        print(f"Optimal decision threshold calibrated to: {threshold:.4f}")

    target_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    print(f"Evaluating {args.split} split with threshold={threshold:.4f}...")
    results = evaluator.evaluate_split(target_loader, threshold=threshold)

    report_md = evaluator.generate_markdown_report(results, split_name=args.split.capitalize(), is_calibrated=calibrated)
    print("\n" + report_md)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()

