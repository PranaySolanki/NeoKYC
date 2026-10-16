import torch
import torch.nn as nn
from typing import List, Tuple
from .dct import create_dct_matrix


def generate_filter_mask(size: int, band_start: float, band_end: float) -> torch.Tensor:
    """
    Generates a triangular frequency-band mask M(u, v) on a discrete grid.
    
    A point (u, v) is in band [band_start, band_end] if:
        band_start <= u + v < band_end (or <= band_end at maximum extent)
    """
    u = torch.arange(size).view(-1, 1).repeat(1, size)
    v = torch.arange(size).view(1, -1).repeat(size, 1)
    dist = (u + v).float()
    mask = (dist >= band_start) & (dist <= band_end)
    return mask.float()


def norm_sigma(x: torch.Tensor) -> torch.Tensor:
    """
    Constrains learnable weights to (-1, 1) to preserve band structure
    as defined in F3-Net paper (Section 3.1): sigma_norm(w) = 2 * sigmoid(w) - 1.
    """
    return 2.0 * torch.sigmoid(x) - 1.0


class FrequencyFilter(nn.Module):
    """
    Frequency-domain filter component.
    Combines a non-trainable geometric base mask with a learnable bounded residual.
    """
    def __init__(self, size: int, band_start: float, band_end: float, use_learnable: bool = True, norm: bool = False):
        super().__init__()
        self.size = size
        self.use_learnable = use_learnable
        self.norm = norm

        base_mask = generate_filter_mask(size, band_start, band_end)
        self.register_buffer("base", base_mask)

        if self.use_learnable:
            # Initialized with zero-mean, 0.1 std per reference implementation
            self.learnable = nn.Parameter(torch.randn(size, size) * 0.1)
        else:
            self.register_parameter("learnable", None)

        if self.norm:
            # Sum of active elements for normalization
            count = torch.sum(base_mask).clamp(min=1.0)
            self.register_buffer("ft_num", count)
        else:
            self.register_buffer("ft_num", None)

    def get_filter(self) -> torch.Tensor:
        if self.use_learnable and self.learnable is not None:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm and self.ft_num is not None:
            return filt / self.ft_num
        return filt

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Filters frequency-domain representations.
        
        Args:
            x_freq: Frequency tensor with shape (..., size, size)
        """
        filt = self.get_filter()
        return x_freq * filt


class FAD_Head(nn.Module):
    """
    Frequency-Aware Decomposition (FAD) Module for F3-Net.
    
    Decomposes an input image into 3 frequency bands (Low, Mid, High) plus an All-pass band:
      - Low:   [0, size / 2.82]
      - Mid:   [size / 2.82, size / 2.0]
      - High:  [size / 2.0, 2 * size]
      - All:   [0, 2 * size]
      
    For each band, 2D-DCT is applied, filtered, and then transformed back via 2D-IDCT.
    Concatenating the 4 spatial outputs gives 4 * 3 = 12 channels for RGB input.
    """
    def __init__(self, size: int = 299, use_learnable: bool = True):
        super().__init__()
        self.size = size

        # Register orthogonal DCT matrices
        dct_mat = create_dct_matrix(size)
        self.register_buffer("dct_mat", dct_mat)
        self.register_buffer("dct_mat_t", dct_mat.transpose(-1, -2))

        # Band boundaries matching paper and reference implementation
        low_bound = size / 2.82
        mid_bound = size / 2.0
        high_bound = float(size * 2)

        low_filter = FrequencyFilter(size, 0.0, low_bound, use_learnable=use_learnable)
        mid_filter = FrequencyFilter(size, low_bound, mid_bound, use_learnable=use_learnable)
        high_filter = FrequencyFilter(size, mid_bound, high_bound, use_learnable=use_learnable)
        all_filter = FrequencyFilter(size, 0.0, high_bound, use_learnable=use_learnable)

        self.filters = nn.ModuleList([low_filter, mid_filter, high_filter, all_filter])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial RGB image tensor of shape (N, 3, size, size)
            
        Returns:
            Decomposed spatial feature tensor of shape (N, 12, size, size)
        """
        # Forward 2D-DCT: D @ x @ D.T  -> (N, 3, size, size)
        x_freq = torch.matmul(torch.matmul(self.dct_mat, x), self.dct_mat_t)

        decomposed_bands = []
        for flt in self.filters:
            # Apply frequency filtering in frequency domain
            filtered_freq = flt(x_freq)
            # Inverse 2D-DCT: D.T @ filtered_freq @ D -> (N, 3, size, size)
            spatial_band = torch.matmul(torch.matmul(self.dct_mat_t, filtered_freq), self.dct_mat)
            decomposed_bands.append(spatial_band)

        # Concatenate 4 bands * 3 channels = 12 channels
        return torch.cat(decomposed_bands, dim=1)
