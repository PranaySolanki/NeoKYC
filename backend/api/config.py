"""
API service configuration for the F³-Net deepfake detection endpoints.

Follows the project's existing pattern of using dataclasses for typed configs
(see backend.training.config.TrainingConfig).

All thresholds are configurable and MUST be validated against a proper evaluation
dataset before being used for real-world eKYC decisions. Default values are
conservative placeholders, not empirically calibrated cutoffs.
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DeepfakeServiceConfig:
    """
    Configuration for the F³-Net inference service.

    Attributes:
        checkpoint_path: Path to a trained F³-Net checkpoint (.pth).
                         If the file does not exist, the service starts in
                         'model_unavailable' mode and returns no scores.
        model_mode: F3Net operational mode ('Mix', 'Both', 'FAD', 'LFS').
        img_size: Input image resolution the model was trained on.
        num_classes: Number of output classes (2 for binary Real/Fake).
        decision_threshold: Probability threshold above which a face crop
                            is classified as manipulated. This value MUST be
                            calibrated on a held-out validation set before
                            deployment — do NOT treat the default as validated.
        max_video_frames: Maximum number of frames to sample from a video.
        video_aggregation_method: How to combine per-frame scores into a
                                  video-level score ('mean', 'median', 'max').
        max_upload_bytes: Maximum allowed upload size in bytes (default 50 MB).
        allowed_image_extensions: File extensions accepted for image uploads.
        allowed_video_extensions: File extensions accepted for video uploads.
        face_detection_confidence: Minimum confidence for OpenCV DNN face detector.
        face_expansion_scale: Bounding-box expansion ratio for face crops
                              (1.3× per F3-Net and FaceForensics++ papers).
        device: Compute device ('cpu' or 'cuda'). Auto-detected if left as 'auto'.
    """
    # Model checkpoint — default is checkpoints/f3net/best_model.pth
    checkpoint_path: str = os.path.join("checkpoints", "f3net", "best_model.pth")

    # Model architecture (must match the checkpoint)
    model_mode: str = "Mix"
    img_size: int = 299
    num_classes: int = 2
    use_learnable_filters: bool = True

    # Decision threshold — NOT validated by default.
    # Must be calibrated using backend.evaluation.metrics.calibrate_threshold()
    # on a proper validation split before production use.
    decision_threshold: float = 0.5

    # Video inference
    max_video_frames: int = 32
    video_aggregation_method: str = "mean"  # 'mean', 'median', 'max'

    # Upload limits
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB

    allowed_image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    allowed_video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    # Face detection
    face_detection_confidence: float = 0.5
    face_expansion_scale: float = 1.3

    # Device
    device: str = "auto"

    @staticmethod
    def get_repo_root() -> str:
        """Returns the absolute path to the repository root."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def resolve_checkpoint_path(self) -> str:
        """
        Resolves the checkpoint path.
        Priority:
          1. F3NET_CHECKPOINT_PATH environment variable (if set and non-empty)
          2. self.checkpoint_path
        Relative paths are consistently resolved relative to the repository root
        rather than the current working directory.
        """
        env_override = os.environ.get("F3NET_CHECKPOINT_PATH", "").strip()
        raw_path = env_override if env_override else self.checkpoint_path
        if os.path.isabs(raw_path):
            return os.path.normpath(raw_path)
        return os.path.normpath(os.path.join(self.get_repo_root(), raw_path))

    def resolve_device(self) -> str:
        """Resolves 'auto' to the best available device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
