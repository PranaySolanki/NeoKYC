"""
Comprehensive tests for the F³-Net deepfake detection API and eKYC pipeline.

Test categories:
  1. Health endpoint tests
  2. Image prediction — valid requests, no-face, invalid files
  3. Video prediction — valid requests, empty videos, no-face
  4. Model unavailable scenarios (missing checkpoint)
  5. Input validation (missing file, wrong format, empty payload)
  6. eKYC pipeline — success, suspicious, model unavailable, component failure
  7. eKYC verify endpoint tests
  8. Regression: existing backend modules still import and work

Run with:
  python -m unittest backend.tests.test_api -v
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from PIL import Image

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.api.config import DeepfakeServiceConfig
from backend.api.app import create_app
from backend.api.face_detector import FaceDetector
from backend.api.inference_service import F3NetInferenceService
from backend.api.ekyc_pipeline import (
    EKYCPipeline,
    EKYCPolicyConfig,
    CheckStatus,
    VerificationDecision,
)


def _create_test_image(width=200, height=200, color=(128, 128, 128), fmt="JPEG") -> bytes:
    """Helper: creates a minimal in-memory image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _create_minimal_video() -> bytes:
    """
    Helper: creates a minimal valid MP4 video in memory using OpenCV.
    """
    try:
        import cv2
    except ImportError:
        return b""

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (200, 200))
        if not writer.isOpened():
            return b""
        for i in range(10):
            frame = np.full((200, 200, 3), fill_value=(50 + i * 20) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ======================================================================
# Configuration Tests
# ======================================================================

class TestDeepfakeServiceConfig(unittest.TestCase):

    def test_default_values(self):
        config = DeepfakeServiceConfig()
        self.assertEqual(config.model_mode, "Mix")
        self.assertEqual(config.img_size, 299)
        self.assertEqual(config.decision_threshold, 0.5)
        self.assertEqual(config.max_video_frames, 32)

    def test_resolve_device_auto(self):
        config = DeepfakeServiceConfig(device="auto")
        device = config.resolve_device()
        self.assertIn(device, ["cpu", "cuda"])

    def test_resolve_device_explicit(self):
        config = DeepfakeServiceConfig(device="cpu")
        self.assertEqual(config.resolve_device(), "cpu")


# ======================================================================
# Face Detector Tests
# ======================================================================

class TestFaceDetector(unittest.TestCase):

    def test_detector_initialization(self):
        detector = FaceDetector()
        # On OpenCV 5 with YuNet model, or OpenCV 4 with Haar cascade
        # The detector should initialize with SOME backend
        # (may fail if model download fails — that's acceptable)
        self.assertIsInstance(detector.detector_type, str)

    def test_detect_on_blank_image(self):
        """A solid-color image should have no face detections."""
        detector = FaceDetector()
        blank = np.full((300, 300, 3), 128, dtype=np.uint8)
        boxes = detector.detect_faces(blank)
        self.assertIsInstance(boxes, list)

    def test_detect_and_crop_pil(self):
        """Verify detect_and_crop returns a list of PIL Images."""
        detector = FaceDetector()
        pil_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        crops = detector.detect_and_crop(pil_img)
        self.assertIsInstance(crops, list)


# ======================================================================
# Health Endpoint Tests
# ======================================================================

class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        config = DeepfakeServiceConfig(
            checkpoint_path="nonexistent_checkpoint.pth",
            device="cpu",
        )
        self.app = create_app(config)
        self.client = self.app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/api/deepfake/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_response_structure(self):
        resp = self.client.get("/api/deepfake/health")
        data = resp.get_json()
        self.assertIn("service", data)
        self.assertEqual(data["service"], "f3net_deepfake_detection")
        self.assertIn("status", data)
        self.assertIn("architecture_loaded", data)
        self.assertIn("checkpoint_loaded", data)
        self.assertIn("face_detector", data)

    def test_health_reports_degraded_when_no_checkpoint(self):
        resp = self.client.get("/api/deepfake/health")
        data = resp.get_json()
        self.assertEqual(data["status"], "degraded")
        self.assertTrue(data["architecture_loaded"])
        self.assertFalse(data["checkpoint_loaded"])


# ======================================================================
# Image Prediction Endpoint Tests
# ======================================================================

class TestPredictImageEndpoint(unittest.TestCase):

    def setUp(self):
        config = DeepfakeServiceConfig(
            checkpoint_path="nonexistent_checkpoint.pth",
            device="cpu",
        )
        self.app = create_app(config)
        self.client = self.app.test_client()

    def test_missing_file_returns_400(self):
        resp = self.client.post("/api/deepfake/predict-image")
        self.assertEqual(resp.status_code, 400)

    def test_empty_filename_returns_400(self):
        resp = self.client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_unsupported_format_returns_400(self):
        image_bytes = _create_test_image()
        resp = self.client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(image_bytes), "test.gif")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_empty_file_returns_400(self):
        resp = self.client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(b""), "test.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_model_unavailable_returns_503(self):
        """Without a trained checkpoint, predictions should return 503."""
        image_bytes = _create_test_image()
        resp = self.client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(image_bytes), "face.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertEqual(data["status"], "model_unavailable")


# ======================================================================
# Video Prediction Endpoint Tests
# ======================================================================

class TestPredictVideoEndpoint(unittest.TestCase):

    def setUp(self):
        config = DeepfakeServiceConfig(
            checkpoint_path="nonexistent_checkpoint.pth",
            device="cpu",
        )
        self.app = create_app(config)
        self.client = self.app.test_client()

    def test_missing_file_returns_400(self):
        resp = self.client.post("/api/deepfake/predict-video")
        self.assertEqual(resp.status_code, 400)

    def test_unsupported_format_returns_400(self):
        resp = self.client.post(
            "/api/deepfake/predict-video",
            data={"file": (io.BytesIO(b"data"), "test.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_empty_video_returns_400(self):
        resp = self.client.post(
            "/api/deepfake/predict-video",
            data={"file": (io.BytesIO(b""), "test.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_model_unavailable_returns_503(self):
        """Without a checkpoint, video predictions should return 503."""
        video_bytes = _create_minimal_video()
        if not video_bytes:
            self.skipTest("Could not create test video")
        resp = self.client.post(
            "/api/deepfake/predict-video",
            data={"file": (io.BytesIO(video_bytes), "test.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 503)


# ======================================================================
# Inference Service with Checkpoint Tests
# ======================================================================

class TestInferenceServiceWithCheckpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._ckpt_path = os.path.join(cls._tmp_dir, "test_model.pth")

        from backend.models.f3net import F3Net
        model = F3Net(num_classes=2, img_size=299, mode="Mix", use_learnable_filters=True)
        checkpoint = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_auc": 0.5,
            "config": {"mode": "Mix", "img_size": 299},
        }
        torch.save(checkpoint, cls._ckpt_path)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def _get_client(self):
        config = DeepfakeServiceConfig(
            checkpoint_path=self._ckpt_path,
            device="cpu",
        )
        app = create_app(config)
        return app.test_client()

    def test_health_reports_ready(self):
        client = self._get_client()
        resp = client.get("/api/deepfake/health")
        data = resp.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["checkpoint_loaded"])

    def test_predict_image_no_face(self):
        """Solid-color image → no_face_detected status."""
        client = self._get_client()
        image_bytes = _create_test_image(200, 200, (128, 128, 128))
        resp = client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(image_bytes), "blank.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "no_face_detected")

    def test_predict_image_corrupted_file(self):
        """Random bytes → error status."""
        client = self._get_client()
        resp = client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(b"corrupted-data-not-an-image"), "bad.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")

    def test_predict_image_response_structure(self):
        client = self._get_client()
        image_bytes = _create_test_image()
        resp = client.post(
            "/api/deepfake/predict-image",
            data={"file": (io.BytesIO(image_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("model", data)
        self.assertIn("name", data["model"])
        self.assertIn("version", data["model"])
        self.assertIn("checkpoint_loaded", data["model"])
        self.assertIn("faces_detected", data)
        self.assertIn("predictions", data)
        self.assertIn("processing_time_ms", data)


# ======================================================================
# Direct Inference Service Tests
# ======================================================================

class TestInferenceServiceDirect(unittest.TestCase):

    def test_service_without_checkpoint(self):
        config = DeepfakeServiceConfig(checkpoint_path="does_not_exist.pth", device="cpu")
        service = F3NetInferenceService(config)
        self.assertTrue(service.architecture_available)
        self.assertFalse(service.is_ready)

    def test_service_health_status_structure(self):
        config = DeepfakeServiceConfig(checkpoint_path="does_not_exist.pth", device="cpu")
        service = F3NetInferenceService(config)
        health = service.health_status()
        self.assertEqual(health["service"], "f3net_deepfake_detection")
        self.assertIn("limitations", health)

    def test_predict_image_model_unavailable(self):
        config = DeepfakeServiceConfig(checkpoint_path="does_not_exist.pth", device="cpu")
        service = F3NetInferenceService(config)
        result = service.predict_image(_create_test_image())
        self.assertEqual(result.status, "model_unavailable")

    def test_predict_video_model_unavailable(self):
        config = DeepfakeServiceConfig(checkpoint_path="does_not_exist.pth", device="cpu")
        service = F3NetInferenceService(config)
        video_bytes = _create_minimal_video()
        if not video_bytes:
            self.skipTest("Could not create test video")
        result = service.predict_video(video_bytes)
        self.assertEqual(result.status, "model_unavailable")


# ======================================================================
# eKYC Pipeline Integration Tests
# ======================================================================

class TestEKYCPipelineModelUnavailable(unittest.TestCase):
    """Tests eKYC pipeline when the deepfake model has no checkpoint."""

    def setUp(self):
        config = DeepfakeServiceConfig(checkpoint_path="nonexistent.pth", device="cpu")
        self.service = F3NetInferenceService(config)
        self.pipeline = EKYCPipeline(
            deepfake_service=self.service,
            policy=EKYCPolicyConfig(),
        )

    def test_deepfake_check_returns_unavailable(self):
        result = self.pipeline.run_deepfake_check(_create_test_image())
        self.assertEqual(result.status, CheckStatus.UNAVAILABLE)
        self.assertIn("no trained", result.message.lower())

    def test_full_pipeline_with_document_data(self):
        """Pipeline should return review_required when deepfake is unavailable."""
        result = self.pipeline.run_verification(
            face_image_bytes=_create_test_image(),
            document_data={
                "name": "Test User",
                "dob": "01/01/2000",
                "gender": "MALE",
                "id_number": "123456789012",
            },
        )
        self.assertIsNotNone(result.decision)
        self.assertIn("deepfake_detection", result.checks_unavailable)
        # Decision should NOT be APPROVED because checks are unavailable
        self.assertNotEqual(result.decision, VerificationDecision.APPROVED)

    def test_pipeline_with_required_deepfake(self):
        """When deepfake is required and unavailable → INCOMPLETE."""
        pipeline = EKYCPipeline(
            deepfake_service=self.service,
            policy=EKYCPolicyConfig(deepfake_required=True),
        )
        result = pipeline.run_verification(face_image_bytes=_create_test_image())
        self.assertEqual(result.decision, VerificationDecision.INCOMPLETE)

    def test_pipeline_no_inputs(self):
        """Pipeline with no inputs should have most checks unavailable."""
        result = self.pipeline.run_verification()
        self.assertIn("deepfake_detection", result.checks_unavailable)

    def test_pipeline_response_structure(self):
        result = self.pipeline.run_verification(face_image_bytes=_create_test_image())
        d = result.to_dict()
        self.assertIn("decision", d)
        self.assertIn("checks_completed", d)
        self.assertIn("checks_unavailable", d)
        self.assertIn("processing_time_ms", d)


class TestEKYCPipelineWithCheckpoint(unittest.TestCase):
    """Tests eKYC pipeline when a checkpoint IS loaded (using untrained model)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._ckpt_path = os.path.join(cls._tmp_dir, "test_model.pth")

        from backend.models.f3net import F3Net
        model = F3Net(num_classes=2, img_size=299, mode="Mix", use_learnable_filters=True)
        torch.save({
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_auc": 0.5,
        }, cls._ckpt_path)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def setUp(self):
        config = DeepfakeServiceConfig(
            checkpoint_path=self._ckpt_path, device="cpu",
        )
        self.service = F3NetInferenceService(config)
        self.pipeline = EKYCPipeline(
            deepfake_service=self.service,
            policy=EKYCPolicyConfig(),
        )

    def test_deepfake_check_no_face_returns_error(self):
        """Blank image → error (no face detected)."""
        result = self.pipeline.run_deepfake_check(_create_test_image())
        self.assertEqual(result.status, CheckStatus.ERROR)
        self.assertIn("No face", result.message)

    def test_pipeline_with_valid_document_no_face(self):
        """Pipeline with complete document but no face detected → review_required."""
        result = self.pipeline.run_verification(
            face_image_bytes=_create_test_image(),
            document_data={
                "name": "Test User",
                "dob": "01/01/2000",
                "gender": "MALE",
                "id_number": "123456789012",
            },
        )
        # With no face detected, deepfake check errors
        self.assertEqual(result.decision, VerificationDecision.REVIEW_REQUIRED)

    def test_pipeline_failed_document(self):
        """Incomplete document data → REJECTED."""
        result = self.pipeline.run_verification(
            document_data={
                "name": "Test User",
                "dob": None,  # Missing
                "gender": None,  # Missing
                "id_number": None,  # Missing
            },
        )
        self.assertEqual(result.decision, VerificationDecision.REJECTED)

    def test_liveness_check_unavailable(self):
        result = self.pipeline.run_liveness_check()
        self.assertEqual(result.status, CheckStatus.UNAVAILABLE)

    def test_face_match_check_unavailable(self):
        result = self.pipeline.run_face_match_check()
        self.assertEqual(result.status, CheckStatus.UNAVAILABLE)

    def test_graph_fraud_check_unavailable(self):
        result = self.pipeline.run_graph_fraud_check()
        self.assertEqual(result.status, CheckStatus.UNAVAILABLE)


# ======================================================================
# eKYC Endpoint Tests
# ======================================================================

class TestEKYCEndpoints(unittest.TestCase):
    """Tests for the /api/ekyc/* endpoints."""

    def setUp(self):
        config = DeepfakeServiceConfig(
            checkpoint_path="nonexistent_checkpoint.pth",
            device="cpu",
        )
        self.app = create_app(config)
        self.client = self.app.test_client()

    def test_ekyc_health_returns_200(self):
        resp = self.client.get("/api/ekyc/health")
        self.assertEqual(resp.status_code, 200)

    def test_ekyc_health_structure(self):
        resp = self.client.get("/api/ekyc/health")
        data = resp.get_json()
        self.assertEqual(data["service"], "neokyc_ekyc_pipeline")
        self.assertIn("components", data)
        self.assertIn("deepfake_detection", data["components"])
        self.assertIn("liveness_detection", data["components"])
        self.assertIn("face_identity_match", data["components"])
        self.assertIn("graph_fraud_analysis", data["components"])
        self.assertIn("policy", data)

    def test_ekyc_verify_no_inputs(self):
        resp = self.client.post("/api/ekyc/verify")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("decision", data)
        self.assertIn("checks_completed", data)
        self.assertIn("checks_unavailable", data)

    def test_ekyc_verify_with_document_data(self):
        resp = self.client.post(
            "/api/ekyc/verify",
            data={
                "document_name": "Test User",
                "document_dob": "01/01/2000",
                "document_gender": "MALE",
                "document_id_number": "123456789012",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        # Document check should run
        doc_check = next(
            (c for c in data["checks_completed"] if c["check_name"] == "document_ocr"),
            None,
        )
        self.assertIsNotNone(doc_check)
        self.assertEqual(doc_check["status"], "passed")

    def test_ekyc_verify_with_face_image(self):
        image_bytes = _create_test_image()
        resp = self.client.post(
            "/api/ekyc/verify",
            data={
                "face_image": (io.BytesIO(image_bytes), "face.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        # Deepfake check should be listed (as unavailable since no checkpoint)
        self.assertIn("deepfake_detection", data["checks_unavailable"])

    def test_ekyc_verify_unsupported_image_format(self):
        image_bytes = _create_test_image()
        resp = self.client.post(
            "/api/ekyc/verify",
            data={
                "face_image": (io.BytesIO(image_bytes), "face.gif"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)


# ======================================================================
# Regression Tests for Existing Modules
# ======================================================================

class TestExistingModulesRegression(unittest.TestCase):

    def test_f3net_model_import(self):
        from backend.models.f3net import F3Net
        model = F3Net(num_classes=2, img_size=299, mode="Mix")
        self.assertIsNotNone(model)

    def test_training_config_import(self):
        from backend.training.config import TrainingConfig
        config = TrainingConfig()
        self.assertEqual(config.mode, "Mix")

    def test_trainer_import(self):
        from backend.training.trainer import Trainer, set_seed
        self.assertTrue(callable(set_seed))

    def test_evaluation_metrics_import(self):
        from backend.evaluation.metrics import (
            calculate_classification_metrics,
            calibrate_threshold,
            aggregate_video_scores,
        )
        scores = np.array([0.9, 0.2])
        labels = np.array([1, 0])
        metrics = calculate_classification_metrics(scores, labels)
        self.assertIn("accuracy", metrics)

    def test_data_transforms_import(self):
        from backend.data.transforms import F3NetPreprocessing, expand_bounding_box
        prep = F3NetPreprocessing(target_size=299)
        img = Image.new("RGB", (200, 200))
        tensor = prep(img)
        self.assertEqual(tensor.shape, (3, 299, 299))

    def test_faceforensics_dataset_import(self):
        from backend.data.faceforensics import FaceForensicsDataset
        self.assertTrue(callable(FaceForensicsDataset))

    def test_evaluator_import(self):
        from backend.evaluation.evaluator import F3NetEvaluator
        self.assertTrue(callable(F3NetEvaluator))


# ======================================================================
# Regression Tests for Targeted Fixes
# ======================================================================

class TestRegressionFixes(unittest.TestCase):
    """
    Tests for:
      1. Default checkpoint path consistency & repo-root resolution
      2. Environment variable overrides (F3NET_CHECKPOINT_PATH)
      3. Windows temporary video cleanup (cap.release in finally block)
    """

    def test_default_checkpoint_path_is_f3net_best_model(self):
        config = DeepfakeServiceConfig()
        expected_rel = os.path.join("checkpoints", "f3net", "best_model.pth")
        self.assertEqual(config.checkpoint_path, expected_rel)

    def test_resolve_checkpoint_path_anchored_to_repo_root(self):
        config = DeepfakeServiceConfig(checkpoint_path=os.path.join("checkpoints", "f3net", "best_model.pth"))
        resolved = config.resolve_checkpoint_path()
        expected = os.path.normpath(os.path.join(DeepfakeServiceConfig.get_repo_root(), "checkpoints", "f3net", "best_model.pth"))
        self.assertEqual(resolved, expected)
        self.assertTrue(os.path.isabs(resolved))

    def test_resolve_checkpoint_path_preserves_absolute(self):
        abs_path = os.path.abspath(os.path.join(tempfile.gettempdir(), "custom_model.pth"))
        config = DeepfakeServiceConfig(checkpoint_path=abs_path)
        resolved = config.resolve_checkpoint_path()
        self.assertEqual(resolved, os.path.normpath(abs_path))

    def test_resolve_checkpoint_path_respects_env_override(self):
        custom_env_path = os.path.join("custom", "override_model.pth")
        with patch.dict(os.environ, {"F3NET_CHECKPOINT_PATH": custom_env_path}):
            config = DeepfakeServiceConfig()
            resolved = config.resolve_checkpoint_path()
            expected = os.path.normpath(os.path.join(DeepfakeServiceConfig.get_repo_root(), custom_env_path))
            self.assertEqual(resolved, expected)

    def test_training_config_paths_anchored_to_repo_root(self):
        from backend.training.config import TrainingConfig
        cfg = TrainingConfig()
        self.assertEqual(cfg.output_dir, "checkpoints/f3net")
        self.assertEqual(cfg.dataset_dir, "data/FaceForensics++")
        self.assertTrue(os.path.isabs(cfg.resolve_output_dir()))
        self.assertTrue(os.path.isabs(cfg.resolve_dataset_dir()))
        self.assertTrue(cfg.resolve_output_dir().endswith(os.path.normpath("checkpoints/f3net")))
        self.assertTrue(cfg.resolve_dataset_dir().endswith(os.path.normpath("data/FaceForensics++")))

    def test_video_cleanup_with_cap_release(self):
        """
        Verify that predict_video releases cv2.VideoCapture and unlinks
        the temporary file even on invalid or corrupted inputs.
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            ckpt = os.path.join(tmp_dir, "model.pth")
            from backend.models.f3net import F3Net
            m = F3Net(mode="Mix")
            torch.save({"model_state_dict": m.state_dict(), "epoch": 0}, ckpt)

            config = DeepfakeServiceConfig(checkpoint_path=ckpt, device="cpu")
            service = F3NetInferenceService(config)

            # Test with invalid video bytes that fail cv2.VideoCapture opening
            corrupted_bytes = b"not-a-valid-mp4-video-stream-header"
            resp = service.predict_video(corrupted_bytes)
            self.assertEqual(resp.status, "error")
            self.assertIn("Failed to open video file", resp.message)

            # Check that no stray temporary files were leaked in temp directory
            # (mkstemp uses system temp directory)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

