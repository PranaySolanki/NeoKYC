import unittest
import torch
import numpy as np
import os
import sys
import tempfile
import shutil
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.data.transforms import expand_bounding_box, F3NetPreprocessing
from backend.data.faceforensics import FaceForensicsDataset
from backend.training.metrics import compute_roc_auc, compute_binary_metrics
from backend.training.config import TrainingConfig
from backend.training.trainer import Trainer


class TestDataTransforms(unittest.TestCase):
    def test_expand_bounding_box(self):
        """Verify 1.3x expansion around center and boundary clamping."""
        w, h = 500, 500
        bbox = (100, 100, 200, 200)  # width=100, height=100, center=(150, 150)
        # Expected new width: 130, new height: 130 -> [85, 85, 215, 215]
        new_bbox = expand_bounding_box(bbox, w, h, scale=1.3)
        self.assertEqual(new_bbox, (85, 85, 215, 215))

        # Test boundary clamping at origin
        edge_bbox = (5, 5, 50, 50)
        clamped = expand_bounding_box(edge_bbox, w, h, scale=1.3)
        self.assertGreaterEqual(clamped[0], 0)
        self.assertGreaterEqual(clamped[1], 0)

    def test_preprocessing_pipeline(self):
        """Verify resize to (299, 299) and [-1, 1] normalization range."""
        prep = F3NetPreprocessing(target_size=299, bbox_scale=1.3, is_training=False)
        dummy_img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        tensor = prep(dummy_img, bbox=(50, 50, 250, 250))

        self.assertEqual(tensor.shape, (3, 299, 299))
        self.assertTrue(torch.all(tensor >= -1.05) and torch.all(tensor <= 1.05))


class TestMetrics(unittest.TestCase):
    def test_roc_auc_perfect_prediction(self):
        """Test AUC is 1.0 for perfect ranking."""
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0])
        auc = compute_roc_auc(scores, labels)
        self.assertAlmostEqual(auc, 1.0, places=4)

    def test_binary_metrics_accuracy_and_auc(self):
        """Test compute_binary_metrics returns expected dictionary values."""
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])  # Sample 0: class 0, Sample 1: class 1
        targets = torch.tensor([0, 1])
        metrics = compute_binary_metrics(logits, targets)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


class TestSplitAntiLeakage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create dummy directory structure simulating FaceForensics++
        # Real video '001', manipulated video '001_892' (derived from '001')
        real_dir = os.path.join(self.temp_dir, "original_sequences", "youtube", "c40", "frames", "001")
        fake_dir = os.path.join(self.temp_dir, "manipulated_sequences", "Deepfakes", "c40", "frames", "001_892")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(os.path.join(real_dir, "frame_000.png"))
        img.save(os.path.join(fake_dir, "frame_000.png"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_anti_leakage_guarantee(self):
        """Verify that source '001' and manipulated '001_892' both map strictly to 'train' split."""
        train_ds = FaceForensicsDataset(self.temp_dir, split="train", compression="c40")
        val_ds = FaceForensicsDataset(self.temp_dir, split="val", compression="c40")

        # '001' is in train set (0..719)
        self.assertEqual(len(train_ds), 2, "Both real '001' and fake '001_892' should be in train split")
        self.assertEqual(len(val_ds), 0, "Neither sample should leak into val split")


class TestTrainerPretrainedWeights(unittest.TestCase):
    def test_trainer_missing_pretrained_weights_raises_filenotfound(self):
        """Verify Trainer fails fast when configured pretrained Xception weights file does not exist."""
        cfg = TrainingConfig(
            pretrained_xception_path="checkpoints/pretrained/nonexistent_xception_weights.pth"
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            Trainer(cfg)
        self.assertIn("nonexistent_xception_weights.pth", str(ctx.exception))

    def test_trainer_with_valid_pretrained_weights_file(self):
        """Verify Trainer properly initializes model using valid pretrained weights file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        try:
            mock_weights = {
                "conv1.weight": torch.randn(32, 3, 3, 3),
                "bn1.weight": torch.ones(32),
                "bn1.bias": torch.zeros(32),
            }
            torch.save(mock_weights, temp_path)
            cfg = TrainingConfig(pretrained_xception_path=temp_path)
            trainer = Trainer(cfg)
            self.assertEqual(trainer.model.FAD_xcep.conv1.weight.shape, (32, 12, 3, 3))
            self.assertEqual(trainer.model.LFS_xcep.conv1.weight.shape, (32, 6, 3, 3))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

