from typing import Tuple
import torch
import torch.nn as nn


class MixBlock(nn.Module):
    """
    Two-stream Collaborative Learning Block (MixBlock) for F3-Net.
    
    Reproduces the exact cross-attention mechanism from the ECCV 2020 paper
    and official reference implementation:
      - Projects FAD and LFS feature maps into query and key spaces via 1x1 convolutions.
      - Constructs concatenated cross-stream query [q_FAD, q_LFS] and key [k_FAD; k_LFS].
      - Computes batched channel-wise affinity energy via torch.bmm:
          energy = M_query @ M_key  -> (B*C, W, W)
      - Computes softmax attention map: A in (B, C, W, W).
      - Recalibrates cross-stream features via bounded gating:
          g(gamma) = 2 * sigmoid(gamma) - 1
      - Adds gated depthwise residual transformations to both streams.
      
    Args:
        c_in: Number of channels at fusion stage (default 728 after Block 7)
        width: Spatial width (default 19)
        height: Spatial height (default 19)
    """
    def __init__(self, c_in: int = 728, width: int = 19, height: int = 19):
        super().__init__()
        self.c_in = c_in
        self.width = width
        self.height = height

        # 1x1 Convolutions for Query and Key projections
        self.FAD_query = nn.Conv2d(c_in, c_in, kernel_size=1, bias=False)
        self.LFS_query = nn.Conv2d(c_in, c_in, kernel_size=1, bias=False)

        self.FAD_key = nn.Conv2d(c_in, c_in, kernel_size=1, bias=False)
        self.LFS_key = nn.Conv2d(c_in, c_in, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        # Learnable gating scalars initialized to 0 (identity residual start)
        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        # Depthwise residual convolutions with batch normalization
        self.FAD_conv = nn.Conv2d(c_in, c_in, kernel_size=1, groups=c_in, bias=False)
        self.FAD_bn = nn.BatchNorm2d(c_in)

        self.LFS_conv = nn.Conv2d(c_in, c_in, kernel_size=1, groups=c_in, bias=False)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD: torch.Tensor, x_LFS: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass between FAD and LFS streams.
        
        Args:
            x_FAD: Tensor of shape (B, C, W, H)
            x_LFS: Tensor of shape (B, C, W, H)
            
        Returns:
            Tuple of (y_FAD, y_LFS), each with shape (B, C, W, H)
        """
        b, c, w, h = x_FAD.size()
        assert w == h, f"MixBlock requires square spatial features, got {w}x{h}"
        assert x_FAD.shape == x_LFS.shape, f"FAD and LFS shapes must match, got {x_FAD.shape} vs {x_LFS.shape}"

        # 1. Compute queries and reshape to (B*C, W, H)
        q_FAD = self.FAD_query(x_FAD).view(-1, w, h)
        q_LFS = self.LFS_query(x_LFS).view(-1, w, h)
        # Concatenate along height dimension -> (B*C, W, 2H)
        m_query = torch.cat([q_FAD, q_LFS], dim=2)

        # 2. Compute keys and transpose to (B*C, H, W)
        k_FAD = self.FAD_key(x_FAD).view(-1, w, h).transpose(1, 2)
        k_LFS = self.LFS_key(x_LFS).view(-1, w, h).transpose(1, 2)
        # Concatenate along height dimension -> (B*C, 2H, W)
        m_key = torch.cat([k_FAD, k_LFS], dim=1)

        # 3. Channel-wise cross-stream affinity energy: (W, 2H) @ (2H, W) = (W, W)
        energy = torch.bmm(m_query, m_key)  # (B*C, W, W)
        attention = self.softmax(energy).view(b, c, w, w)

        # 4. Feature modulation and residual cross-fusion
        # Gating factors: 2 * sigmoid(gamma) - 1 in (-1, 1), initialized to 0
        gate_lfs = torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0
        att_lfs = x_LFS * attention * gate_lfs
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_lfs))

        gate_fad = torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0
        att_fad = x_FAD * attention * gate_fad
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_fad))

        return y_FAD, y_LFS
