import os
import json
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ..models.f3net import F3Net
from .config import TrainingConfig
from .metrics import compute_binary_metrics


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class Trainer:
    """
    Trainer for F3-Net deepfake detection.
    """
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[F3Net] = None,
    ):
        self.config = config
        set_seed(config.seed)

        # Device determination
        if config.device == "cpu":
            self.device = torch.device("cpu")
        elif config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # "auto" or default
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize or receive model
        if model is not None:
            self.model = model.to(self.device)
        else:
            pretrained_path = config.resolve_pretrained_xception_path()
            if pretrained_path is not None:
                if not os.path.isfile(pretrained_path):
                    raise FileNotFoundError(
                        f"Configured pretrained Xception weights file not found at: {pretrained_path}. "
                        "Please verify the file path or omit the pretrained weights argument to use random initialization."
                    )
            self.model = F3Net(
                num_classes=config.num_classes,
                img_size=config.img_size,
                mode=config.mode,
                use_learnable_filters=config.use_learnable_filters,
                pretrained_xception=pretrained_path,
            ).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Reference SGD with momentum 0.9
        if config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Learning rate scheduler: Cosine Annealing per paper
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=1e-6,
        )

        self.start_epoch = 0
        self.best_val_auc = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
            "val_f1": [],
        }

        # Ensure output directory exists (resolved relative to repo root)
        output_dir = self.config.resolve_output_dir() if hasattr(self.config, "resolve_output_dir") else self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for i, (images, labels, _) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for training stability
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

            if i % self.config.log_freq == 0:
                print(f"Epoch [{epoch}/{self.config.max_epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        return total_loss / max(batches, 1)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / max(len(dataloader), 1)
        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            metrics = compute_binary_metrics(cat_logits, cat_labels)
        else:
            metrics = {"accuracy": 0.0, "auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        metrics["loss"] = avg_loss
        return metrics

    def save_checkpoint(self, epoch: int, val_auc: float, is_best: bool = False) -> str:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_auc": val_auc,
            "config": self.config.__dict__,
        }
        output_dir = self.config.resolve_output_dir() if hasattr(self.config, "resolve_output_dir") else self.config.output_dir
        latest_path = os.path.join(output_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved new best model to {best_path} (Val AUC: {val_auc:.4f})")

        return latest_path

    def load_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_auc = checkpoint.get("val_auc", 0.0)
        print(f"[Checkpoint] Resumed from {path} at epoch {self.start_epoch} with Val AUC: {self.best_val_auc:.4f}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, list]:
        """Main training loop across epochs."""
        print(f"Starting F3-Net training on device: {self.device} (Mode: {self.config.mode})")

        for epoch in range(self.start_epoch, self.config.max_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["accuracy"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_acc)
            self.history["val_auc"].append(val_auc)
            self.history["val_f1"].append(val_metrics["f1"])

            print(
                f"Epoch [{epoch + 1}/{self.config.max_epochs}] ({elapsed:.1f}s) "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
            )

            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc

            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_auc, is_best=is_best)

            # Save metrics history to disk
            output_dir = self.config.resolve_output_dir() if hasattr(self.config, "resolve_output_dir") else self.config.output_dir
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history
