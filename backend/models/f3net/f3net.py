import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fad import FAD_Head
from .lfs import LFS_Head
from .xception import XceptionBackbone
from .mixblock import MixBlock


class F3Net(nn.Module):
    """
    Complete Frequency in Face Forgery Network (F3-Net).
    
    Paper:
      "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues"
      (ECCV 2020 - Qian et al.)
      
    Architecture:
      Stream 1: Frequency-Aware Decomposition (FAD) -> 12-channel spatial decomposition -> Xception Stream 1
      Stream 2: Local Frequency Statistics (LFS)   -> 6-channel frequency energy map    -> Xception Stream 2
      Fusion:   Two-stream cross-attention via MixBlock at Middle Flow (after Block 7, 728-ch, 19x19)
      Head:     Concatenation of pooled 2048-dim features (4096-dim) -> Linear classifier
      
    Modes supported:
      - 'Mix':  Full dual-stream collaborative model with MixBlock cross-attention (Default).
      - 'Both': Dual-stream model with direct concatenation of pooled embeddings (no MixBlock).
      - 'FAD':  Single-stream model using FAD branch only.
      - 'LFS':  Single-stream model using LFS branch only.
      
    Args:
        num_classes: Number of output classes (default 2 for [Real, Manipulated] logits).
        img_size: Expected input square image resolution (default 299).
        mode: Operational mode ('Mix', 'Both', 'FAD', 'LFS').
        dropout_rate: Classification head dropout probability (default 0.5).
        use_learnable_filters: Whether FAD/LFS filters contain trainable residuals (default True).
        pretrained_xception: Optional path to .pth checkpoint or state dict containing ImageNet Xception weights.
    """
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 299,
        mode: str = "Mix",
        dropout_rate: float = 0.5,
        use_learnable_filters: bool = True,
        pretrained_xception: Optional[Union[str, os.PathLike, Dict[str, torch.Tensor]]] = None,
    ):
        super().__init__()
        valid_modes = {"Mix", "Both", "FAD", "LFS"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}")

        self.num_classes = num_classes
        self.img_size = img_size
        self.mode = mode

        # 1. Initialize Frequency Preprocessing Heads
        if mode in {"Mix", "Both", "FAD"}:
            self.FAD_head = FAD_Head(size=img_size, use_learnable=use_learnable_filters)
            self.FAD_xcep = XceptionBackbone(in_channels=12, num_classes=num_classes, is_lfs_branch=False)

        if mode in {"Mix", "Both", "LFS"}:
            self.LFS_head = LFS_Head(size=img_size, window_size=10, stride=2, padding=4, M=6, use_learnable=use_learnable_filters)
            self.LFS_xcep = XceptionBackbone(in_channels=6, num_classes=num_classes, is_lfs_branch=True)

        # 2. Collaborative Cross-Attention Fusion Block (Stage after Block 7: 728 channels, 19x19)
        if mode == "Mix":
            self.mix_block = MixBlock(c_in=728, width=19, height=19)

        # 3. Classification Head
        in_features = 4096 if mode in {"Mix", "Both"} else 2048
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

        # 4. Optional Pretrained Xception Initialization
        if pretrained_xception is not None:
            self.load_pretrained_xception(pretrained_xception)

    def load_pretrained_xception(
        self,
        weights_path_or_dict: Union[str, os.PathLike, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, int]]:
        """
        Loads ImageNet-pretrained Xception weights into active stream backbones.
        Adapts the first convolution layer for multi-channel frequency inputs (12-ch FAD, 6-ch LFS).
        
        Args:
            weights_path_or_dict: Path to .pth checkpoint file or state dictionary.
            
        Returns:
            Dictionary mapping branch names to load summaries.
        """
        results: Dict[str, Dict[str, int]] = {}
        if hasattr(self, "FAD_xcep"):
            results["FAD_xcep"] = self.FAD_xcep.load_imagenet_state_dict(weights_path_or_dict)
        if hasattr(self, "LFS_xcep"):
            results["LFS_xcep"] = self.LFS_xcep.load_imagenet_state_dict(weights_path_or_dict)
        return results

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of F3-Net.
        
        Args:
            x: Input RGB batch of shape (N, 3, img_size, img_size), e.g., (N, 3, 299, 299).
            
        Returns:
            Tuple of (logits, features):
              - logits: Classification tensor of shape (N, num_classes)
              - features: Joint feature embedding of shape (N, 4096) or (N, 2048)
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input tensor of shape (N, 3, H, W), got {x.shape}")
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            raise ValueError(f"Expected input dimensions ({self.img_size}, {self.img_size}), got ({x.shape[2]}, {x.shape[3]})")

        if self.mode == "Mix":
            # 1. Frequency preprocessing
            x_fad = self.FAD_head(x)   # (N, 12, 299, 299)
            x_lfs = self.LFS_head(x)   # (N, 6, 149, 149)

            # 2. Backbone Entry Flow & Middle Flow through Block 7
            mid_fad = self.FAD_xcep.forward_stage_0_7(x_fad)  # (N, 728, 19, 19)
            mid_lfs = self.LFS_xcep.forward_stage_0_7(x_lfs)  # (N, 728, 19, 19)

            # 3. MixBlock Cross-Attention Collaborative Recalibration
            fused_fad, fused_lfs = self.mix_block(mid_fad, mid_lfs)  # (N, 728, 19, 19) each

            # 4. Backbone Exit Flow & Pooling
            exit_fad = self.FAD_xcep.forward_stage_8_12(fused_fad)   # (N, 2048, 10, 10)
            exit_lfs = self.LFS_xcep.forward_stage_8_12(fused_lfs)   # (N, 2048, 10, 10)

            fea_fad = self.FAD_xcep.pool_features(exit_fad)          # (N, 2048)
            fea_lfs = self.LFS_xcep.pool_features(exit_lfs)          # (N, 2048)

            # 5. Dual-stream Concatenation & Classification
            features = torch.cat([fea_fad, fea_lfs], dim=1)          # (N, 4096)

        elif self.mode == "Both":
            x_fad = self.FAD_head(x)
            x_lfs = self.LFS_head(x)

            fea_fad = self.FAD_xcep(x_fad)  # (N, 2048)
            fea_lfs = self.LFS_xcep(x_lfs)  # (N, 2048)
            features = torch.cat([fea_fad, fea_lfs], dim=1)  # (N, 4096)

        elif self.mode == "FAD":
            x_fad = self.FAD_head(x)
            features = self.FAD_xcep(x_fad)  # (N, 2048)

        elif self.mode == "LFS":
            x_lfs = self.LFS_head(x)
            features = self.LFS_xcep(x_lfs)  # (N, 2048)

        logits = self.fc(self.dropout(features))
        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper returning normalized class probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            if self.num_classes == 1:
                prob_fake = torch.sigmoid(logits)
                return torch.cat([1.0 - prob_fake, prob_fake], dim=1)
            else:
                return F.softmax(logits, dim=1)
