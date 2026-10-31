import os
from dataclasses import dataclass, field
from typing import List, Optional

# Repository root: calculated relative to this config file (backend/training/config.py -> ../..)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_repo_path(path: Optional[str]) -> Optional[str]:
    """Resolves relative paths from the repository root, preserving absolute paths."""
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


@dataclass
class TrainingConfig:
    """
    Configuration parameters for F3-Net training.
    
    Default values match the reference paper:
      - Optimizer: SGD
      - Momentum: 0.9
      - Initial LR: 0.002
      - Cosine Annealing scheduler
      - Primary benchmark compression: c40 (FaceForensics++ LQ)
    """
    # Dataset and Data Loading
    dataset_dir: str = "data/FaceForensics++"
    compression: str = "c40"
    manipulation_types: List[str] = field(default_factory=lambda: ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    frames_per_video: int = 10
    batch_size: int = 16
    num_workers: int = 2

    # Model Configuration
    mode: str = "Mix"  # 'Mix', 'Both', 'FAD', 'LFS'
    num_classes: int = 2
    img_size: int = 299
    use_learnable_filters: bool = True
    pretrained_xception_path: Optional[str] = None

    # Optimizer & Scheduler Settings (Paper defaults)
    optimizer: str = "sgd"
    lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    max_epochs: int = 10
    grad_clip_norm: float = 5.0

    # Logging and Checkpointing — default output_dir is checkpoints/f3net
    output_dir: str = "checkpoints/f3net"
    save_freq: int = 1
    log_freq: int = 20
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    device: str = "auto"  # Automatically uses CUDA if available, or 'cpu' / 'cuda'

    def resolve_output_dir(self) -> str:
        """Resolves output_dir relative to the repository root."""
        return _resolve_repo_path(self.output_dir)

    def resolve_dataset_dir(self) -> str:
        """Resolves dataset_dir relative to the repository root."""
        return _resolve_repo_path(self.dataset_dir)

    def resolve_resume_checkpoint(self) -> Optional[str]:
        """Resolves resume_checkpoint relative to the repository root if set."""
        return _resolve_repo_path(self.resume_checkpoint)

    def resolve_pretrained_xception_path(self) -> Optional[str]:
        """Resolves pretrained_xception_path relative to the repository root if set."""
        return _resolve_repo_path(self.pretrained_xception_path)


