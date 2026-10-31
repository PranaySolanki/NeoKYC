"""
Face detection utility for the deepfake detection API.

Uses OpenCV's FaceDetectorYN (YuNet DNN model) on OpenCV 5+, falling back to
the legacy CascadeClassifier on OpenCV 4.x. If neither is available, detection
is gracefully skipped and the service operates without face cropping.

Reuses the project's existing expand_bounding_box() for face-region expansion
consistent with the 1.3× scale used during F³-Net training.
"""
import io
import logging
import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..data.transforms import expand_bounding_box

logger = logging.getLogger(__name__)

# YuNet model URL — official OpenCV Zoo asset
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _get_yunet_model_path() -> Optional[str]:
    """
    Returns the path to the YuNet ONNX model, downloading it if needed.
    The model is stored alongside the package to avoid re-downloading.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "..", "_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, _YUNET_FILENAME)

    if os.path.isfile(model_path):
        return model_path

    try:
        logger.info(f"Downloading YuNet face detection model to {model_path}...")
        urllib.request.urlretrieve(_YUNET_URL, model_path)
        logger.info("YuNet model downloaded successfully.")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to download YuNet model: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        return None


class FaceDetector:
    """
    OpenCV-based face detector for the deepfake inference pipeline.

    Detection priority:
      1. FaceDetectorYN (YuNet DNN) — OpenCV 5.x (high accuracy)
      2. CascadeClassifier (Haar) — OpenCV 4.x fallback
      3. No detection — full images analyzed without cropping
    """

    def __init__(self, confidence_threshold: float = 0.5, expansion_scale: float = 1.3):
        self.confidence_threshold = confidence_threshold
        self.expansion_scale = expansion_scale
        self._detector_type: str = "none"
        self._yunet = None
        self._cascade = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes the best available face detector."""
        # Attempt 1: FaceDetectorYN (OpenCV 5+)
        if hasattr(cv2, "FaceDetectorYN"):
            model_path = _get_yunet_model_path()
            if model_path:
                try:
                    self._yunet = cv2.FaceDetectorYN.create(
                        model_path,
                        "",
                        (320, 320),  # default input size, will be updated per-image
                        self.confidence_threshold,
                        0.3,  # NMS threshold
                        5000,  # top_k
                    )
                    self._detector_type = "yunet_dnn"
                    logger.info("Face detector initialized: OpenCV FaceDetectorYN (YuNet)")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize FaceDetectorYN: {e}")

        # Attempt 2: Haar Cascade (OpenCV 4.x)
        if hasattr(cv2, "CascadeClassifier"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
                if not self._cascade.empty():
                    self._detector_type = "haar_cascade"
                    logger.info("Face detector initialized: OpenCV Haar Cascade")
                    return
                else:
                    self._cascade = None
            except Exception as e:
                logger.warning(f"Failed to load Haar cascade: {e}")

        logger.error(
            "No face detector could be initialized. "
            "Face cropping will be skipped — full images will be analyzed."
        )

    @property
    def is_available(self) -> bool:
        return self._detector_type != "none"

    @property
    def detector_type(self) -> str:
        return self._detector_type

    def detect_faces(
        self,
        image: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a BGR numpy image.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes, expanded by self.expansion_scale.
            Empty list if no faces are found or detector is unavailable.
        """
        if not self.is_available:
            return []

        h, w = image.shape[:2]
        raw_boxes: List[Tuple[int, int, int, int]] = []

        if self._detector_type == "yunet_dnn" and self._yunet is not None:
            # YuNet requires setting input size per image
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(image)
            if faces is not None:
                for face in faces:
                    x1 = int(face[0])
                    y1 = int(face[1])
                    fw = int(face[2])
                    fh = int(face[3])
                    conf = float(face[-1])
                    if conf >= self.confidence_threshold:
                        raw_boxes.append((x1, y1, x1 + fw, y1 + fh))

        elif self._detector_type == "haar_cascade" and self._cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            detections = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            for (x, y, fw, fh) in detections:
                raw_boxes.append((x, y, x + fw, y + fh))

        # Apply 1.3× expansion (matching training preprocessing)
        expanded = []
        for box in raw_boxes:
            expanded_box = expand_bounding_box(box, w, h, scale=self.expansion_scale)
            expanded.append(expanded_box)

        return expanded

    def detect_and_crop(
        self,
        pil_image: Image.Image,
    ) -> List[Image.Image]:
        """
        Detects faces and returns cropped PIL images.

        Args:
            pil_image: Input PIL Image (RGB).

        Returns:
            List of face-cropped PIL Images. Empty list if no faces found.
        """
        # Convert PIL -> OpenCV BGR
        rgb_np = np.array(pil_image)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        boxes = self.detect_faces(bgr_np)
        if not boxes:
            return []

        crops = []
        for (x1, y1, x2, y2) in boxes:
            crop = pil_image.crop((x1, y1, x2, y2))
            # Validate crop dimensions
            if crop.width >= 10 and crop.height >= 10:
                crops.append(crop)
            else:
                logger.debug(f"Skipping tiny face crop: {crop.width}x{crop.height}")

        return crops
