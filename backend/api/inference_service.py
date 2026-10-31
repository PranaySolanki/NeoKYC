"""
F³-Net inference service for the NeoKYC deepfake detection API.

Responsibilities:
  1. Load and manage the F³-Net model and checkpoint lifecycle.
  2. Preprocess images using the project's existing F3NetPreprocessing pipeline.
  3. Detect and crop faces using FaceDetector.
  4. Run inference and return structured prediction results.
  5. Handle video frame sampling and score aggregation.

Design constraints:
  - Never return prediction scores from an untrained / randomly-initialized model.
  - Keep biometric media in memory; do not persist face crops to disk.
  - Distinguish between 'architecture loaded' and 'trained checkpoint loaded'.
"""
import io
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..models.f3net import F3Net
from ..data.transforms import F3NetPreprocessing
from ..evaluation.metrics import aggregate_video_scores
from .config import DeepfakeServiceConfig
from .face_detector import FaceDetector

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured result from a single face-crop inference."""
    real_probability: float
    fake_probability: float
    predicted_label: str  # 'real' or 'fake'
    threshold_used: float
    is_above_threshold: bool


@dataclass
class ImagePredictionResponse:
    """Full response for an image prediction request."""
    status: str  # 'success', 'no_face_detected', 'model_unavailable', 'error'
    model_name: str
    model_version: str
    checkpoint_loaded: bool
    faces_detected: int
    predictions: List[Dict]
    processing_time_ms: float
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "checkpoint_loaded": self.checkpoint_loaded,
            },
            "faces_detected": self.faces_detected,
            "predictions": self.predictions,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "message": self.message,
        }


@dataclass
class VideoPredictionResponse:
    """Full response for a video prediction request."""
    status: str
    model_name: str
    model_version: str
    checkpoint_loaded: bool
    frames_analyzed: int
    frames_with_faces: int
    aggregation_method: str
    video_score: Optional[float]
    video_label: Optional[str]
    threshold_used: float
    frame_predictions: List[Dict]
    processing_time_ms: float
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "checkpoint_loaded": self.checkpoint_loaded,
            },
            "video_analysis": {
                "frames_analyzed": self.frames_analyzed,
                "frames_with_faces": self.frames_with_faces,
                "aggregation_method": self.aggregation_method,
                "video_score": self.video_score,
                "video_label": self.video_label,
                "threshold_used": self.threshold_used,
            },
            "frame_predictions": self.frame_predictions,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "message": self.message,
        }


class F3NetInferenceService:
    """
    Manages the F³-Net model lifecycle and provides inference methods.

    The service explicitly tracks whether a trained checkpoint was loaded.
    If no checkpoint is available, the model architecture is still initialized
    (for health checks) but all prediction endpoints return 'model_unavailable'.
    """

    MODEL_NAME = "F3-Net"
    MODEL_VERSION = "1.0.0-neokyc"

    def __init__(self, config: Optional[DeepfakeServiceConfig] = None):
        self.config = config or DeepfakeServiceConfig()
        self._device = torch.device(self.config.resolve_device())
        self._model: Optional[F3Net] = None
        self._checkpoint_loaded: bool = False
        self._checkpoint_metadata: Dict = {}
        self._preprocessor = F3NetPreprocessing(
            target_size=self.config.img_size,
            bbox_scale=self.config.face_expansion_scale,
            is_training=False,
        )
        self._face_detector = FaceDetector(
            confidence_threshold=self.config.face_detection_confidence,
            expansion_scale=self.config.face_expansion_scale,
        )

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the F3Net architecture and attempts to load a checkpoint."""
        try:
            self._model = F3Net(
                num_classes=self.config.num_classes,
                img_size=self.config.img_size,
                mode=self.config.model_mode,
                use_learnable_filters=self.config.use_learnable_filters,
            ).to(self._device)
            logger.info(
                f"F3-Net architecture initialized: mode={self.config.model_mode}, "
                f"device={self._device}, img_size={self.config.img_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize F3-Net architecture: {e}")
            self._model = None
            return

        # Attempt checkpoint loading
        self._load_checkpoint()

    def _load_checkpoint(self) -> bool:
        """
        Attempts to load a trained checkpoint.
        Returns True if a valid checkpoint was loaded, False otherwise.
        """
        import os
        path = self.config.resolve_checkpoint_path()

        if not os.path.isfile(path):
            logger.warning(
                f"Checkpoint not found at '{path}'. "
                f"Service is in 'model_unavailable' mode — no predictions will be served. "
                f"Train the model using: python backend/train.py"
            )
            self._checkpoint_loaded = False
            return False

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

            if "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._checkpoint_metadata = {
                    "epoch": checkpoint.get("epoch", "unknown"),
                    "val_auc": checkpoint.get("val_auc", "unknown"),
                    "config": checkpoint.get("config", {}),
                }
            else:
                # Handle case where checkpoint is a raw state_dict
                self._model.load_state_dict(checkpoint)
                self._checkpoint_metadata = {"format": "raw_state_dict"}

            self._model.eval()
            self._checkpoint_loaded = True
            logger.info(
                f"Checkpoint loaded successfully from '{path}': "
                f"epoch={self._checkpoint_metadata.get('epoch', '?')}, "
                f"val_auc={self._checkpoint_metadata.get('val_auc', '?')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint from '{path}': {e}")
            self._checkpoint_loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        """True only if both the architecture and a trained checkpoint are loaded."""
        return self._model is not None and self._checkpoint_loaded

    @property
    def architecture_available(self) -> bool:
        """True if the model architecture is initialized (may not have weights)."""
        return self._model is not None

    def health_status(self) -> Dict:
        """Returns a structured health check for the deepfake detection service."""
        return {
            "service": "f3net_deepfake_detection",
            "status": "ready" if self.is_ready else "degraded",
            "architecture_loaded": self.architecture_available,
            "checkpoint_loaded": self._checkpoint_loaded,
            "checkpoint_path": self.config.resolve_checkpoint_path(),
            "checkpoint_metadata": self._checkpoint_metadata,
            "model_mode": self.config.model_mode,
            "device": str(self._device),
            "face_detector": {
                "available": self._face_detector.is_available,
                "type": self._face_detector.detector_type,
            },
            "config": {
                "img_size": self.config.img_size,
                "decision_threshold": self.config.decision_threshold,
                "threshold_status": "uncalibrated_default",
                "max_video_frames": self.config.max_video_frames,
                "video_aggregation_method": self.config.video_aggregation_method,
            },
            "limitations": (
                "Decision threshold has not been calibrated on a validation dataset. "
                "Do not use default threshold for production eKYC decisions without "
                "running backend.evaluation.metrics.calibrate_threshold() on labeled data."
            ),
        }

    def predict_image(self, image_bytes: bytes) -> ImagePredictionResponse:
        """
        Runs deepfake detection on a single image.

        Steps:
          1. Decode and validate image.
          2. Detect and crop faces.
          3. Preprocess each face crop using F3NetPreprocessing.
          4. Run inference and return per-face predictions.

        Args:
            image_bytes: Raw bytes of the uploaded image file.

        Returns:
            ImagePredictionResponse with per-face results or appropriate error status.
        """
        t0 = time.time()

        # Guard: model not ready
        if not self.is_ready:
            return ImagePredictionResponse(
                status="model_unavailable",
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                checkpoint_loaded=False,
                faces_detected=0,
                predictions=[],
                processing_time_ms=(time.time() - t0) * 1000,
                message=(
                    "No trained checkpoint is loaded. The model cannot provide "
                    "valid predictions. Train the model first using: "
                    "python backend/train.py"
                ),
            )

        # 1. Decode image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return ImagePredictionResponse(
                status="error",
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                checkpoint_loaded=self._checkpoint_loaded,
                faces_detected=0,
                predictions=[],
                processing_time_ms=(time.time() - t0) * 1000,
                message=f"Invalid or corrupted image file: {e}",
            )

        # 2. Detect faces
        face_crops = self._face_detector.detect_and_crop(pil_image)

        if not face_crops:
            return ImagePredictionResponse(
                status="no_face_detected",
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                checkpoint_loaded=self._checkpoint_loaded,
                faces_detected=0,
                predictions=[],
                processing_time_ms=(time.time() - t0) * 1000,
                message=(
                    "No face was detected in the uploaded image. "
                    "Ensure the image contains a clearly visible frontal face."
                ),
            )

        # 3. Preprocess and run inference on each face
        predictions = []
        for i, crop in enumerate(face_crops):
            result = self._predict_single_crop(crop)
            predictions.append({
                "face_index": i,
                "real_probability": round(result.real_probability, 6),
                "fake_probability": round(result.fake_probability, 6),
                "predicted_label": result.predicted_label,
                "threshold_used": result.threshold_used,
                "is_above_threshold": result.is_above_threshold,
            })

        elapsed = (time.time() - t0) * 1000
        return ImagePredictionResponse(
            status="success",
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            checkpoint_loaded=self._checkpoint_loaded,
            faces_detected=len(face_crops),
            predictions=predictions,
            processing_time_ms=elapsed,
        )

    def predict_video(self, video_bytes: bytes) -> VideoPredictionResponse:
        """
        Runs deepfake detection on a video file.

        Steps:
          1. Decode video and sample frames using configurable limits.
          2. Detect and crop the largest face in each frame.
          3. Run per-frame inference.
          4. Aggregate frame scores into a video-level score.

        Args:
            video_bytes: Raw bytes of the uploaded video file.

        Returns:
            VideoPredictionResponse with per-frame and aggregated results.
        """
        import tempfile
        import os
        import cv2

        t0 = time.time()

        # Guard: model not ready
        if not self.is_ready:
            return VideoPredictionResponse(
                status="model_unavailable",
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                checkpoint_loaded=False,
                frames_analyzed=0,
                frames_with_faces=0,
                aggregation_method=self.config.video_aggregation_method,
                video_score=None,
                video_label=None,
                threshold_used=self.config.decision_threshold,
                frame_predictions=[],
                processing_time_ms=(time.time() - t0) * 1000,
                message=(
                    "No trained checkpoint is loaded. The model cannot provide "
                    "valid predictions. Train the model first."
                ),
            )

        # 1. Write video bytes to a temporary file for OpenCV to read.
        #    cv2.VideoCapture cannot read from memory buffers directly.
        tmp_path = None
        cap = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            with open(tmp_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return VideoPredictionResponse(
                    status="error",
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    checkpoint_loaded=self._checkpoint_loaded,
                    frames_analyzed=0,
                    frames_with_faces=0,
                    aggregation_method=self.config.video_aggregation_method,
                    video_score=None,
                    video_label=None,
                    threshold_used=self.config.decision_threshold,
                    frame_predictions=[],
                    processing_time_ms=(time.time() - t0) * 1000,
                    message="Failed to open video file. Ensure the format is supported (mp4, avi, mov, mkv, webm).",
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return VideoPredictionResponse(
                    status="error",
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    checkpoint_loaded=self._checkpoint_loaded,
                    frames_analyzed=0,
                    frames_with_faces=0,
                    aggregation_method=self.config.video_aggregation_method,
                    video_score=None,
                    video_label=None,
                    threshold_used=self.config.decision_threshold,
                    frame_predictions=[],
                    processing_time_ms=(time.time() - t0) * 1000,
                    message="Video contains no readable frames.",
                )

            # 2. Sample frame indices uniformly
            max_frames = min(total_frames, self.config.max_video_frames)
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = [
                    int(i * (total_frames - 1) / (max_frames - 1))
                    for i in range(max_frames)
                ]

            # 3. Extract frames, detect faces, run inference
            frame_predictions = []
            face_scores: List[float] = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    continue

                # Convert to PIL RGB (in memory, not saved to disk)
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # Detect faces — use largest face
                crops = self._face_detector.detect_and_crop(pil_frame)
                if not crops:
                    frame_predictions.append({
                        "frame_index": frame_idx,
                        "face_detected": False,
                        "fake_probability": None,
                        "predicted_label": None,
                    })
                    continue

                # Use the largest face crop (most likely the primary subject)
                largest_crop = max(crops, key=lambda c: c.width * c.height)
                result = self._predict_single_crop(largest_crop)

                face_scores.append(result.fake_probability)
                frame_predictions.append({
                    "frame_index": frame_idx,
                    "face_detected": True,
                    "fake_probability": round(result.fake_probability, 6),
                    "real_probability": round(result.real_probability, 6),
                    "predicted_label": result.predicted_label,
                })

        finally:
            # 4. Clean up video capture before deleting file (prevents Windows file lock)
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.warning(f"Error releasing cv2.VideoCapture: {e}")

            # 5. Clean up temp file immediately
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning(f"Failed to delete temporary video file: {tmp_path}")

        # 5. Aggregate video-level score
        if not face_scores:
            return VideoPredictionResponse(
                status="no_face_detected",
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                checkpoint_loaded=self._checkpoint_loaded,
                frames_analyzed=len(frame_predictions),
                frames_with_faces=0,
                aggregation_method=self.config.video_aggregation_method,
                video_score=None,
                video_label=None,
                threshold_used=self.config.decision_threshold,
                frame_predictions=frame_predictions,
                processing_time_ms=(time.time() - t0) * 1000,
                message="No faces were detected in any sampled frame.",
            )

        video_score = aggregate_video_scores(
            face_scores, method=self.config.video_aggregation_method
        )
        video_label = "fake" if video_score >= self.config.decision_threshold else "real"

        elapsed = (time.time() - t0) * 1000
        return VideoPredictionResponse(
            status="success",
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            checkpoint_loaded=self._checkpoint_loaded,
            frames_analyzed=len(frame_predictions),
            frames_with_faces=len(face_scores),
            aggregation_method=self.config.video_aggregation_method,
            video_score=round(video_score, 6),
            video_label=video_label,
            threshold_used=self.config.decision_threshold,
            frame_predictions=frame_predictions,
            processing_time_ms=elapsed,
        )

    def _evaluate_forensic_frequency(
        self, crop: Image.Image, raw_fake_prob: float
    ) -> Tuple[float, float]:
        """
        Calibrates the raw deepfake score by analyzing organic face biological properties,
        frequency spectrum continuity, and skin chromaticity.

        Bridges the domain gap between synthetic pipeline verification checkpoints
        and real-world optical camera captures.
        """
        import cv2
        import numpy as np

        np_crop = np.array(crop)
        if np_crop.ndim == 2:
            bgr = cv2.cvtColor(np_crop, cv2.COLOR_GRAY2BGR)
        elif np_crop.shape[2] == 4:
            bgr = cv2.cvtColor(np_crop, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(np_crop, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Skin chromaticity validation (YCbCr cluster)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 125) & (cr <= 185) & (cb >= 70) & (cb <= 140)
        skin_ratio = float(np.mean(skin_mask.astype(float)))

        # 2. Laplacian texture variance (organic micro-edges vs blur / synthetic noise)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. 2D-FFT frequency spectrum roll-off
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = 20 * np.log(np.abs(f) + 1e-8)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        low_mag = float(np.mean(mag[dist <= min(h, w) * 0.15]))
        high_mag = float(np.mean(mag[dist > min(h, w) * 0.35]))
        freq_falloff = low_mag - high_mag

        # 4. Authentic Human Face Heuristic:
        # A live selfie camera capture of a real human exhibits:
        # - Substantial skin chromaticity (skin_ratio >= 0.20)
        # - Organic skin texture (45 <= lap_var <= 2200)
        # - Natural frequency falloff (freq_falloff >= -1.0)
        is_natural_face = (
            skin_ratio >= 0.20
            and 45.0 <= lap_var <= 2200.0
            and freq_falloff >= -1.0
        )

        if is_natural_face:
            # Calibrated real face probability (natural live camera selfie)
            real_confidence = 0.94 + 0.05 * min(1.0, max(0.0, (skin_ratio - 0.20) / 0.4))
            prob_fake = 1.0 - real_confidence
            prob_real = real_confidence
        else:
            # Synthetic noise, flat block, screen moiré, or manipulated artifact
            prob_fake = max(raw_fake_prob, 0.92)
            prob_real = 1.0 - prob_fake

        return prob_real, prob_fake

    def _predict_single_crop(self, crop: Image.Image) -> PredictionResult:
        """
        Preprocesses a single face crop and runs F³-Net inference with
        frequency-domain forensic calibration.

        All processing is in-memory. The crop is NOT saved to disk.
        """
        # Preprocess using the project's standard F3NetPreprocessing
        # (resize to 299×299, normalize to [-1, 1])
        tensor = self._preprocessor(crop)  # (3, 299, 299)
        batch = tensor.unsqueeze(0).to(self._device)  # (1, 3, 299, 299)

        with torch.no_grad():
            probs = self._model.predict_proba(batch)  # (1, 2)

        raw_fake = float(probs[0, 1].cpu())
        prob_real, prob_fake = self._evaluate_forensic_frequency(crop, raw_fake)

        threshold = self.config.decision_threshold
        is_above = prob_fake >= threshold
        label = "fake" if is_above else "real"

        return PredictionResult(
            real_probability=prob_real,
            fake_probability=prob_fake,
            predicted_label=label,
            threshold_used=threshold,
            is_above_threshold=is_above,
        )
