import torch
import torch.nn as nn
from typing import List
from .dct import create_dct_matrix
from .fad import FrequencyFilter


class LFS_Head(nn.Module):
    """
    Local Frequency Statistics (LFS) Module for F3-Net.
    
    Operates on local patches using a sliding-window DCT:
      - Converts RGB to Grayscale
      - Extracts patches via sliding window (window_size=10, stride=2, padding=4)
      - Computes 10x10 2D-DCT on each patch
      - Computes log spectral energy: log10(|DCT| + 1e-15)
      - Decomposes into M frequency bands (default M=6) using normalized filters
      - Aggregates energy over each patch and reshapes into a spatial statistics map
      
    For a 299x299 input, the output shape is precisely:
      N x M x 149 x 149 (149 x 149 x 6 for M=6)
    """
    def __init__(
        self,
        size: int = 299,
        window_size: int = 10,
        stride: int = 2,
        padding: int = 4,
        M: int = 6,
        use_learnable: bool = True
    ):
        super().__init__()
        self.size = size
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.M = M

        # Orthonormal DCT matrix for patches (window_size x window_size)
        dct_patch = create_dct_matrix(window_size)
        self.register_buffer("dct_patch", dct_patch)
        self.register_buffer("dct_patch_t", dct_patch.transpose(-1, -2))

        # Unfold layer for sliding window patch extraction
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size),
            stride=stride,
            padding=padding
        )

        # Filters for M frequency bands with normalization enabled
        self.filters = nn.ModuleList([
            FrequencyFilter(
                size=window_size,
                band_start=window_size * 2.0 / M * i,
                band_end=window_size * 2.0 / M * (i + 1),
                use_learnable=use_learnable,
                norm=True
            )
            for i in range(M)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input RGB image tensor of shape (N, 3, H, W), typically (N, 3, 299, 299)
            
        Returns:
            Local Frequency Statistics tensor of shape (N, M, H_out, W_out),
            e.g., (N, 6, 149, 149) for 299x299 input.
        """
        n, c, h, w = x.shape

        # 1. RGB to Grayscale
        # Uses standard luminance weighting: 0.299 R + 0.587 G + 0.114 B
        if c == 3:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        elif c == 1:
            x_gray = x
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")

        # 2. Rescale input to [0, 245-255] range for numerical consistency with reference
        # If input is in [-1, 1], (x + 1) * 122.5 maps it to [0, 245]
        # If input is in [0, 1], x * 255 maps it to [0, 255]
        if x_gray.min() < 0:
            x_scaled = (x_gray + 1.0) * 122.5
        elif x_gray.max() <= 1.0:
            x_scaled = x_gray * 255.0
        else:
            x_scaled = x_gray

        # Compute output spatial dimension using reference formula
        h_out = int((h - self.window_size + 2 * self.padding) / self.stride) + 1
        w_out = int((w - self.window_size + 2 * self.padding) / self.stride) + 1

        # 3. Sliding window unfold: (N, C * S * S, L) where L = H_out * W_out
        x_unfolded = self.unfold(x_scaled)
        l_blocks = x_unfolded.shape[2]
        
        # Reshape to (N, L, 1, S, S) for batch patch DCT
        s = self.window_size
        x_patches = x_unfolded.transpose(1, 2).reshape(n, l_blocks, 1, s, s)

        # 4. Patch 2D-DCT: D_patch @ patches @ D_patch.T
        x_dct = torch.matmul(torch.matmul(self.dct_patch, x_patches), self.dct_patch_t)

        # 5. Logarithmic spectral amplitude
        log_spectral = torch.log10(torch.abs(x_dct) + 1e-15)

        # 6. Apply M bandpass filters and pool patch energy
        band_maps = []
        for flt in self.filters:
            filtered_band = flt(log_spectral)  # (N, L, 1, S, S)
            # Sum spectral energy over spatial patch dimensions (dims 2, 3, 4)
            pooled_energy = torch.sum(filtered_band, dim=[2, 3, 4])  # (N, L)
            # Reshape back to (N, 1, H_out, W_out)
            spatial_map = pooled_energy.reshape(n, h_out, w_out).unsqueeze(1)
            band_maps.append(spatial_map)

        # Concatenate along channel dimension -> (N, M, H_out, W_out)
        return torch.cat(band_maps, dim=1)
