import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


def create_dct_matrix(size: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Constructs an orthonormal DCT-II transformation matrix of shape (size, size).
    
    Formula:
        D[0, j] = sqrt(1 / N)
        D[i, j] = sqrt(2 / N) * cos( (j + 0.5) * pi * i / N )  for i > 0
    
    Property:
        D is orthogonal: D @ D.T = D.T @ D = I
        Thus, the inverse DCT is simply matrix multiplication with D.T.
    """
    matrix = torch.zeros((size, size), dtype=dtype, device=device)
    factor_0 = math.sqrt(1.0 / size)
    factor_k = math.sqrt(2.0 / size)

    # i = 0 (DC component)
    matrix[0, :] = factor_0

    # i > 0 (AC components)
    for i in range(1, size):
        for j in range(size):
            matrix[i, j] = factor_k * math.cos((j + 0.5) * math.pi * i / size)

    return matrix


def dct2(x: torch.Tensor, dct_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes batched 2D Discrete Cosine Transform (DCT-II) along the last two dimensions.
    
    Args:
        x: Tensor of shape (..., H, W)
        dct_mat: Optional square DCT matrix (if H == W)
        
    Returns:
        Frequency-domain tensor of shape (..., H, W)
    """
    *_, h, w = x.shape
    if dct_mat is not None and h == w and dct_mat.shape == (h, w):
        d_h = dct_mat
        d_w = dct_mat
    else:
        d_h = create_dct_matrix(h, dtype=x.dtype, device=x.device)
        d_w = d_h if h == w else create_dct_matrix(w, dtype=x.dtype, device=x.device)

    # Forward 2D DCT: D_h @ x @ D_w.T
    return torch.matmul(torch.matmul(d_h, x), d_w.transpose(-1, -2))


def idct2(x_freq: torch.Tensor, dct_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes batched 2D Inverse Discrete Cosine Transform along the last two dimensions.
    
    Args:
        x_freq: Tensor of shape (..., H, W) in frequency domain
        dct_mat: Optional square DCT matrix (if H == W)
        
    Returns:
        Spatial-domain reconstructed tensor of shape (..., H, W)
    """
    *_, h, w = x_freq.shape
    if dct_mat is not None and h == w and dct_mat.shape == (h, w):
        d_h = dct_mat
        d_w = dct_mat
    else:
        d_h = create_dct_matrix(h, dtype=x_freq.dtype, device=x_freq.device)
        d_w = d_h if h == w else create_dct_matrix(w, dtype=x_freq.dtype, device=x_freq.device)

    # Inverse 2D DCT: D_h.T @ x_freq @ D_w
    return torch.matmul(torch.matmul(d_h.transpose(-1, -2), x_freq), d_w)


class DCT2DLayer(nn.Module):
    """
    Differentiable 2D DCT layer with pre-allocated transformation matrices.
    """
    def __init__(self, height: int, width: Optional[int] = None):
        super().__init__()
        self.height = height
        self.width = width if width is not None else height

        d_h = create_dct_matrix(self.height)
        d_w = d_h if self.width == self.height else create_dct_matrix(self.width)

        self.register_buffer("d_h", d_h)
        self.register_buffer("d_h_t", d_h.transpose(-1, -2))
        self.register_buffer("d_w", d_w)
        self.register_buffer("d_w_t", d_w.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward 2D DCT: D_h @ x @ D_w.T"""
        return torch.matmul(torch.matmul(self.d_h, x), self.d_w_t)

    def inverse(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Inverse 2D DCT: D_h.T @ x_freq @ D_w"""
        return torch.matmul(torch.matmul(self.d_h_t, x_freq), self.d_w)
