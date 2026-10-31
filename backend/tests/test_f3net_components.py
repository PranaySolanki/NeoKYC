import unittest
import torch
import sys
import os

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.f3net.dct import create_dct_matrix, dct2, idct2, DCT2DLayer
from backend.models.f3net.fad import FAD_Head, FrequencyFilter, generate_filter_mask, norm_sigma
from backend.models.f3net.lfs import LFS_Head


class TestDCTUtilities(unittest.TestCase):
    def test_orthogonality(self):
        """Verify D @ D.T == I for various dimensions."""
        for size in [8, 10, 16, 299]:
            d = create_dct_matrix(size)
            eye = torch.eye(size, dtype=d.dtype)
            prod = torch.matmul(d, d.transpose(-1, -2))
            diff = torch.max(torch.abs(prod - eye)).item()
            self.assertLess(diff, 1e-5, f"DCT matrix of size {size} is not orthogonal (max diff: {diff})")

    def test_reconstruction_accuracy(self):
        """Verify idct2(dct2(x)) == x with numerical precision."""
        torch.manual_seed(42)
        # Test small synthetic tensor and standard 299x299 image dimensions
        shapes = [(2, 3, 16, 16), (1, 1, 32, 32), (2, 3, 299, 299)]
        for shape in shapes:
            x = torch.randn(*shape, dtype=torch.float32)
            x_freq = dct2(x)
            x_rec = idct2(x_freq)
            max_error = torch.max(torch.abs(x - x_rec)).item()
            self.assertLess(max_error, 1e-4, f"Reconstruction error too high for shape {shape}: {max_error}")

    def test_dct_gradient_propagation(self):
        """Verify autograd propagates gradients through DCT and IDCT."""
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        freq = dct2(x)
        rec = idct2(freq)
        loss = rec.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradient is None")
        self.assertTrue(torch.all(torch.isfinite(x.grad)), "Gradients are not finite")
        # Gradient of identity reconstruction should be all ones
        grad_diff = torch.max(torch.abs(x.grad - torch.ones_like(x))).item()
        self.assertLess(grad_diff, 1e-4, "Gradient mismatch through DCT-IDCT identity")

    def test_dct_layer(self):
        """Test DCT2DLayer forward and inverse with registered buffers."""
        layer = DCT2DLayer(16)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        freq = layer(x)
        rec = layer.inverse(freq)
        self.assertEqual(freq.shape, x.shape)
        self.assertLess(torch.max(torch.abs(x - rec)).item(), 1e-4)


class TestFADComponent(unittest.TestCase):
    def test_filter_mask_generation(self):
        """Verify geometric frequency mask partitioning."""
        size = 16
        mask_low = generate_filter_mask(size, 0, 5)
        mask_high = generate_filter_mask(size, 6, 32)
        # Masks should be binary
        self.assertTrue(torch.all((mask_low == 0) | (mask_low == 1)))
        self.assertTrue(torch.all((mask_high == 0) | (mask_high == 1)))
        # Low and high should not overlap
        overlap = torch.sum(mask_low * mask_high).item()
        self.assertEqual(overlap, 0.0)

    def test_norm_sigma_bounds(self):
        """Verify norm_sigma maps to [-1, 1] with floating point limits."""
        x = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0])
        bounded = norm_sigma(x)
        self.assertTrue(torch.all(bounded >= -1.0))
        self.assertTrue(torch.all(bounded <= 1.0))
        # For non-saturated inputs, strictly between -1 and 1
        moderate = norm_sigma(torch.tensor([-2.0, 0.0, 2.0]))
        self.assertTrue(torch.all(moderate > -1.0))
        self.assertTrue(torch.all(moderate < 1.0))

    def test_fad_output_dimensions_and_finiteness(self):
        """Test FAD_Head output shape (N, 12, H, W) and finite values."""
        fad = FAD_Head(size=32, use_learnable=True)
        x = torch.randn(2, 3, 32, 32)
        out = fad(x)

        # 4 filters * 3 channels = 12 channels
        self.assertEqual(out.shape, (2, 12, 32, 32))
        self.assertTrue(torch.all(torch.isfinite(out)), "FAD output contains NaN or Inf")

    def test_fad_gradient_flow(self):
        """Verify gradients flow to both input image and learnable filter parameters."""
        fad = FAD_Head(size=16, use_learnable=True)
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        out = fad(x)
        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

        for i, flt in enumerate(fad.filters):
            self.assertIsNotNone(flt.learnable.grad, f"Filter {i} learnable param has no gradient")
            self.assertTrue(torch.all(torch.isfinite(flt.learnable.grad)))

    def test_fad_standard_299_dimensions(self):
        """Verify full 299x299 input yields expected (N, 12, 299, 299) output."""
        fad = FAD_Head(size=299, use_learnable=False)
        x = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            out = fad(x)
        self.assertEqual(out.shape, (1, 12, 299, 299))


class TestLFSComponent(unittest.TestCase):
    def test_lfs_synthetic_dimensions_and_finiteness(self):
        """Test LFS_Head with custom size and M=6 bands."""
        lfs = LFS_Head(size=32, window_size=6, stride=2, padding=2, M=6)
        x = torch.randn(2, 3, 32, 32)
        out = lfs(x)

        # Expected H_out = (32 - 6 + 4)/2 + 1 = 16
        self.assertEqual(out.shape, (2, 6, 16, 16))
        self.assertTrue(torch.all(torch.isfinite(out)), "LFS output contains NaN or Inf")

    def test_lfs_standard_299_dimension_149x149x6(self):
        """Verify the exact 149x149x6 output reported in paper for 299x299 input."""
        lfs = LFS_Head(size=299, window_size=10, stride=2, padding=4, M=6, use_learnable=False)
        x = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            out = lfs(x)

        # Exact dimension match from reference: N x 6 x 149 x 149
        self.assertEqual(out.shape, (1, 6, 149, 149), f"Expected (1, 6, 149, 149), got {out.shape}")
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_lfs_gradient_flow(self):
        """Verify autograd propagates gradients through unfold, patch DCT, log-energy, and filters."""
        lfs = LFS_Head(size=32, window_size=6, stride=2, padding=2, M=3, use_learnable=True)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = lfs(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "LFS input gradient is None")
        self.assertTrue(torch.all(torch.isfinite(x.grad)), "LFS input gradients not finite")
        for i, flt in enumerate(lfs.filters):
            self.assertIsNotNone(flt.learnable.grad, f"LFS filter {i} has no gradient")
            self.assertTrue(torch.all(torch.isfinite(flt.learnable.grad)))


if __name__ == "__main__":
    unittest.main()
