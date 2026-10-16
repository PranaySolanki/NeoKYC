import unittest
import torch
import sys
import os

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.f3net.xception import XceptionBackbone, XceptionBlock, SeparableConv2d
from backend.models.f3net.mixblock import MixBlock
from backend.models.f3net.f3net import F3Net


class TestXceptionBackbone(unittest.TestCase):
    def test_fad_branch_forward_and_stages(self):
        """Test Xception backbone with 12-channel FAD input."""
        backbone = XceptionBackbone(in_channels=12, is_lfs_branch=False)
        x_fad = torch.randn(2, 12, 299, 299)

        # Stage 0-7: Middle flow output at Block 7
        stage_0_7 = backbone.forward_stage_0_7(x_fad)
        self.assertEqual(stage_0_7.shape, (2, 728, 19, 19), "FAD Stage 0-7 output must be [N, 728, 19, 19]")

        # Stage 8-12: Exit flow output
        stage_8_12 = backbone.forward_stage_8_12(stage_0_7)
        self.assertEqual(stage_8_12.shape, (2, 2048, 10, 10), "FAD Stage 8-12 output must be [N, 2048, 10, 10]")

        # Final global pooling
        pooled = backbone.pool_features(stage_8_12)
        self.assertEqual(pooled.shape, (2, 2048), "Pooled features must be [N, 2048]")

    def test_lfs_branch_forward_and_stages(self):
        """Test Xception backbone with 6-channel 149x149 LFS input."""
        backbone = XceptionBackbone(in_channels=6, is_lfs_branch=True)
        x_lfs = torch.randn(2, 6, 149, 149)

        # Stage 0-7 must align to exactly 19x19 with 728 channels
        stage_0_7 = backbone.forward_stage_0_7(x_lfs)
        self.assertEqual(stage_0_7.shape, (2, 728, 19, 19), "LFS Stage 0-7 output must match FAD at [N, 728, 19, 19]")

        stage_8_12 = backbone.forward_stage_8_12(stage_0_7)
        self.assertEqual(stage_8_12.shape, (2, 2048, 10, 10))

        pooled = backbone.pool_features(stage_8_12)
        self.assertEqual(pooled.shape, (2, 2048))

    def test_imagenet_conv1_expansion_adaptation(self):
        """Test adaptation of 3-channel weights to 12 channels for FAD and 6 channels for LFS."""
        backbone_fad = XceptionBackbone(in_channels=12, is_lfs_branch=False)
        mock_state = {
            "conv1.weight": torch.randn(32, 3, 3, 3),
            "bn1.weight": torch.ones(32),
            "bn1.bias": torch.zeros(32),
        }
        res = backbone_fad.load_imagenet_state_dict(mock_state)
        self.assertEqual(backbone_fad.conv1.weight.shape, (32, 12, 3, 3))
        self.assertGreater(res["matched_keys"], 0)

    def test_pretrained_weights_file_and_wrapper_loading(self):
        """Test loading pretrained weights from a .pth file with 'module.' and dict wrapper."""
        import tempfile
        backbone_fad = XceptionBackbone(in_channels=12, is_lfs_branch=False)
        mock_ckpt = {
            "state_dict": {
                "module.conv1.weight": torch.randn(32, 3, 3, 3),
                "module.bn1.weight": torch.ones(32),
                "module.bn1.bias": torch.zeros(32),
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        try:
            torch.save(mock_ckpt, temp_path)
            res = backbone_fad.load_imagenet_state_dict(temp_path)
            self.assertEqual(backbone_fad.conv1.weight.shape, (32, 12, 3, 3))
            self.assertGreater(res["matched_keys"], 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_missing_pretrained_file_raises_filenotfound(self):
        """Test that attempting to load a non-existent file raises FileNotFoundError."""
        backbone = XceptionBackbone(in_channels=12)
        non_existent_path = "checkpoints/pretrained/does_not_exist_xyz123.pth"
        with self.assertRaises(FileNotFoundError) as ctx:
            backbone.load_imagenet_state_dict(non_existent_path)
        self.assertIn("does_not_exist_xyz123.pth", str(ctx.exception))

    def test_incompatible_checkpoint_raises_value_error(self):
        """Test that incompatible checkpoints fail clearly with ValueError."""
        backbone = XceptionBackbone(in_channels=12)

        # 1. Non-dict input
        with self.assertRaises(ValueError):
            backbone.load_imagenet_state_dict([1, 2, 3])  # type: ignore

        # 2. Missing conv1.weight
        with self.assertRaises(ValueError) as ctx_missing:
            backbone.load_imagenet_state_dict({"fc.weight": torch.randn(10, 10)})
        self.assertIn("conv1.weight", str(ctx_missing.exception))

        # 3. Incompatible conv1 shape (e.g. ResNet conv1 with 64 channels, 7x7)
        with self.assertRaises(ValueError) as ctx_shape:
            backbone.load_imagenet_state_dict({"conv1.weight": torch.randn(64, 3, 7, 7)})
        self.assertIn("32 output channels", str(ctx_shape.exception))



class TestMixBlock(unittest.TestCase):
    def test_mixblock_dimensions(self):
        """Test that MixBlock preserves spatial and channel dimensions."""
        mix = MixBlock(c_in=728, width=19, height=19)
        x_fad = torch.randn(2, 728, 19, 19)
        x_lfs = torch.randn(2, 728, 19, 19)

        y_fad, y_lfs = mix(x_fad, x_lfs)
        self.assertEqual(y_fad.shape, (2, 728, 19, 19))
        self.assertEqual(y_lfs.shape, (2, 728, 19, 19))
        self.assertTrue(torch.all(torch.isfinite(y_fad)))
        self.assertTrue(torch.all(torch.isfinite(y_lfs)))

    def test_mixblock_gradients(self):
        """Verify backpropagation through cross-attention to both input streams and parameters."""
        mix = MixBlock(c_in=32, width=8, height=8)
        x_fad = torch.randn(2, 32, 8, 8, requires_grad=True)
        x_lfs = torch.randn(2, 32, 8, 8, requires_grad=True)

        y_fad, y_lfs = mix(x_fad, x_lfs)
        loss = (y_fad.sum() + y_lfs.sum())
        loss.backward()

        self.assertIsNotNone(x_fad.grad)
        self.assertIsNotNone(x_lfs.grad)
        self.assertTrue(torch.all(torch.isfinite(x_fad.grad)))
        self.assertTrue(torch.all(torch.isfinite(x_lfs.grad)))
        self.assertIsNotNone(mix.FAD_gamma.grad)
        self.assertIsNotNone(mix.LFS_gamma.grad)


class TestF3NetIntegration(unittest.TestCase):
    def test_full_model_mix_mode(self):
        """Test full end-to-end forward and backward pass in 'Mix' collaborative mode."""
        model = F3Net(num_classes=2, img_size=299, mode="Mix", use_learnable_filters=True)
        # Batch of 2 synthetic RGB images
        x = torch.randn(2, 3, 299, 299, requires_grad=True)

        logits, features = model(x)
        self.assertEqual(logits.shape, (2, 2), "Logits should be [N, 2]")
        self.assertEqual(features.shape, (2, 4096), "Fused features should be [N, 4096]")
        self.assertTrue(torch.all(torch.isfinite(logits)))
        self.assertTrue(torch.all(torch.isfinite(features)))

        # Backward gradient flow test
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Input gradient must not be None")
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

    def test_predict_proba(self):
        """Test inference probability prediction."""
        model = F3Net(num_classes=2, img_size=299, mode="Mix")
        x = torch.randn(2, 3, 299, 299)
        probs = model.predict_proba(x)

        self.assertEqual(probs.shape, (2, 2))
        # Probabilities must sum to 1.0 along class dimension
        prob_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

    def test_all_operational_modes(self):
        """Test F3Net operational modes: 'Both', 'FAD', 'LFS'."""
        x = torch.randn(1, 3, 299, 299)

        # Mode 'Both' (Direct concatenation without MixBlock)
        model_both = F3Net(num_classes=2, img_size=299, mode="Both", use_learnable_filters=False)
        with torch.no_grad():
            l_both, f_both = model_both(x)
        self.assertEqual(l_both.shape, (1, 2))
        self.assertEqual(f_both.shape, (1, 4096))

        # Mode 'FAD' (Single stream)
        model_fad = F3Net(num_classes=2, img_size=299, mode="FAD", use_learnable_filters=False)
        with torch.no_grad():
            l_fad, f_fad = model_fad(x)
        self.assertEqual(l_fad.shape, (1, 2))
        self.assertEqual(f_fad.shape, (1, 2048))

        # Mode 'LFS' (Single stream)
        model_lfs = F3Net(num_classes=2, img_size=299, mode="LFS", use_learnable_filters=False)
        with torch.no_grad():
            l_lfs, f_lfs = model_lfs(x)
        self.assertEqual(l_lfs.shape, (1, 2))
        self.assertEqual(f_lfs.shape, (1, 2048))

    def test_device_compatibility(self):
        """Test CPU execution and CUDA execution if available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = F3Net(num_classes=2, img_size=299, mode="Mix", use_learnable_filters=False).to(device)
        x = torch.randn(1, 3, 299, 299, device=device)
        with torch.no_grad():
            logits, features = model(x)
        self.assertEqual(logits.device.type, device.type)
        self.assertEqual(features.device.type, device.type)

    def test_f3net_pretrained_initialization_and_channel_adaptation(self):
        """Test full F3Net initialization with mock ImageNet weights and mathematical scaling."""
        mock_weights = {
            "conv1.weight": torch.ones(32, 3, 3, 3),
            "bn1.weight": torch.ones(32),
            "bn1.bias": torch.zeros(32),
            "conv2.weight": torch.randn(64, 32, 3, 3),
            "bn2.weight": torch.ones(64),
            "bn2.bias": torch.zeros(64),
        }
        model = F3Net(num_classes=2, img_size=299, mode="Mix", pretrained_xception=mock_weights)

        # FAD branch: 3 -> 12 channels (repeated 4x, scaled by 3/12 = 0.25)
        self.assertEqual(model.FAD_xcep.conv1.weight.shape, (32, 12, 3, 3))
        self.assertTrue(torch.allclose(model.FAD_xcep.conv1.weight, torch.full((32, 12, 3, 3), 0.25)))

        # LFS branch: 3 -> 6 channels (repeated 2x, scaled by 3/6 = 0.5)
        self.assertEqual(model.LFS_xcep.conv1.weight.shape, (32, 6, 3, 3))
        self.assertTrue(torch.allclose(model.LFS_xcep.conv1.weight, torch.full((32, 6, 3, 3), 0.5)))

    def test_f3net_missing_pretrained_file_raises(self):
        """Test F3Net constructor raises FileNotFoundError on non-existent checkpoint path."""
        with self.assertRaises(FileNotFoundError):
            F3Net(pretrained_xception="checkpoints/pretrained/missing_xception.pth")

    def test_f3net_incompatible_checkpoint_raises(self):
        """Test F3Net constructor raises ValueError on incompatible weights format."""
        with self.assertRaises(ValueError):
            F3Net(pretrained_xception={"conv1.weight": torch.randn(64, 3, 7, 7)})



if __name__ == "__main__":
    unittest.main()
