"""
Flask REST API for F³-Net deepfake detection in the NeoKYC eKYC platform.

Endpoints:
  GET  /api/deepfake/health          — Service health and model status
  POST /api/deepfake/predict-image   — Single image deepfake detection
  POST /api/deepfake/predict-video   — Video deepfake detection
  GET  /api/ekyc/health              — Full pipeline health and component status
  POST /api/ekyc/verify              — Multi-stage eKYC verification

This module provides a deepfake-detection *signal* within the eKYC pipeline.
It does NOT replace document verification, face matching, liveness checks,
or graph-based fraud detection. Its score must be consumed as one input
among many in the final eKYC decision.

Usage:
  python -m backend.api.app                          # Production (default port 5000)
  python -m backend.api.app --debug --port 5001      # Development mode

Configuration:
  Set the F3NET_CHECKPOINT_PATH environment variable or pass the checkpoint
  path via command-line arguments. See DeepfakeServiceConfig for all options.
"""
import argparse
import logging
import os
import sys

from flask import Flask, request, jsonify

# Add project root to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.api.config import DeepfakeServiceConfig
from backend.api.inference_service import F3NetInferenceService
from backend.api.ekyc_pipeline import EKYCPipeline, EKYCPolicyConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app(config: DeepfakeServiceConfig = None) -> Flask:
    """
    Flask application factory.

    Creates and configures the Flask app with the deepfake detection endpoints.
    Uses the application factory pattern for testability — test code can call
    create_app() with a custom config without starting a server.
    """
    if config is None:
        config = DeepfakeServiceConfig(
            checkpoint_path=os.environ.get(
                "F3NET_CHECKPOINT_PATH",
                DeepfakeServiceConfig.checkpoint_path,
            ),
            device=os.environ.get("F3NET_DEVICE", "auto"),
        )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes

    # Initialize the inference service (loads model and checkpoint)
    service = F3NetInferenceService(config)

    # Initialize the eKYC pipeline orchestrator
    ekyc_policy = EKYCPolicyConfig(
        deepfake_threshold=config.decision_threshold,
    )
    ekyc_pipeline = EKYCPipeline(deepfake_service=service, policy=ekyc_policy)

    # ------------------------------------------------------------------
    # GET /api/deepfake/health
    # ------------------------------------------------------------------
    @app.route("/api/deepfake/health", methods=["GET"])
    def deepfake_health():
        """
        Health check for the deepfake detection service.

        Distinguishes between:
          - 'ready':    Architecture loaded AND trained checkpoint loaded.
          - 'degraded': Architecture loaded but NO trained checkpoint.
          - 'error':    Architecture failed to initialize.

        Returns 200 with status details in all cases (health checks
        should not fail with HTTP errors).
        """
        health = service.health_status()
        return jsonify(health), 200

    # ------------------------------------------------------------------
    # POST /api/deepfake/predict-image
    # ------------------------------------------------------------------
    @app.route("/api/deepfake/predict-image", methods=["POST"])
    def predict_image():
        """
        Deepfake detection on a single uploaded image.

        Request:
          Content-Type: multipart/form-data
          Field: 'file' (required) — Image file (jpg, png, bmp, webp)

        Response: JSON with per-face predictions.

        Status codes:
          200: Prediction completed (check response 'status' for details)
          400: Missing file or invalid format
          413: File too large
          503: Model not ready
        """
        # Validate file presence
        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded. Send an image file as 'file' in multipart/form-data.",
            }), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty filename. Provide a valid image file.",
            }), 400

        # Validate extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in config.allowed_image_extensions:
            return jsonify({
                "status": "error",
                "message": (
                    f"Unsupported image format '{ext}'. "
                    f"Allowed: {', '.join(config.allowed_image_extensions)}"
                ),
            }), 400

        # Read file bytes into memory (no disk persistence)
        try:
            image_bytes = file.read()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to read uploaded file: {e}",
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "message": "Uploaded file is empty.",
            }), 400

        # Run inference
        result = service.predict_image(image_bytes)

        if result.status == "model_unavailable":
            return jsonify(result.to_dict()), 503

        return jsonify(result.to_dict()), 200

    # ------------------------------------------------------------------
    # POST /api/deepfake/predict-video
    # ------------------------------------------------------------------
    @app.route("/api/deepfake/predict-video", methods=["POST"])
    def predict_video():
        """
        Deepfake detection on an uploaded video.

        Request:
          Content-Type: multipart/form-data
          Field: 'file' (required) — Video file (mp4, avi, mov, mkv, webm)

        Response: JSON with per-frame and aggregated video-level predictions.

        Status codes:
          200: Prediction completed (check response 'status' for details)
          400: Missing file or invalid format
          413: File too large
          503: Model not ready
        """
        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded. Send a video file as 'file' in multipart/form-data.",
            }), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty filename. Provide a valid video file.",
            }), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in config.allowed_video_extensions:
            return jsonify({
                "status": "error",
                "message": (
                    f"Unsupported video format '{ext}'. "
                    f"Allowed: {', '.join(config.allowed_video_extensions)}"
                ),
            }), 400

        try:
            video_bytes = file.read()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to read uploaded file: {e}",
            }), 400

        if len(video_bytes) == 0:
            return jsonify({
                "status": "error",
                "message": "Uploaded video file is empty.",
            }), 400

        result = service.predict_video(video_bytes)

        if result.status == "model_unavailable":
            return jsonify(result.to_dict()), 503

        return jsonify(result.to_dict()), 200

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = config.max_upload_bytes / (1024 * 1024)
        return jsonify({
            "status": "error",
            "message": f"File too large. Maximum upload size is {max_mb:.0f} MB.",
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.exception("Unhandled server error")
        return jsonify({
            "status": "error",
            "message": "Internal server error. Check server logs for details.",
        }), 500

    # ------------------------------------------------------------------
    # GET /api/ekyc/health
    # ------------------------------------------------------------------
    @app.route("/api/ekyc/health", methods=["GET"])
    def ekyc_health():
        """
        Full eKYC pipeline health check.

        Reports the status of every verification component:
          - deepfake_detection: F³-Net model and checkpoint
          - liveness_detection: placeholder (not yet implemented)
          - face_identity_match: placeholder (not yet implemented)
          - graph_fraud_analysis: placeholder (not yet implemented)

        Returns 200 with component statuses.
        """
        deepfake_health = service.health_status()
        pipeline_health = {
            "service": "neokyc_ekyc_pipeline",
            "components": {
                "deepfake_detection": {
                    "status": "ready" if service.is_ready else "unavailable",
                    **deepfake_health,
                },
                "liveness_detection": {
                    "status": "unavailable",
                    "implemented": False,
                    "message": "Pending implementation with mobile FaceLivenessScreen.",
                },
                "face_identity_match": {
                    "status": "unavailable",
                    "implemented": False,
                    "message": "ArcFace face matching not yet implemented.",
                },
                "graph_fraud_analysis": {
                    "status": "unavailable",
                    "implemented": False,
                    "message": "Neo4j graph fraud analysis not yet implemented.",
                },
            },
            "policy": {
                "deepfake_required": ekyc_policy.deepfake_required,
                "liveness_required": ekyc_policy.liveness_required,
                "face_match_required": ekyc_policy.face_match_required,
                "graph_fraud_required": ekyc_policy.graph_fraud_required,
                "auto_reject_on_deepfake": ekyc_policy.auto_reject_on_deepfake,
            },
        }
        return jsonify(pipeline_health), 200

    # ------------------------------------------------------------------
    # POST /api/ekyc/verify
    # ------------------------------------------------------------------
    @app.route("/api/ekyc/verify", methods=["POST"])
    def ekyc_verify():
        """
        Multi-stage eKYC verification endpoint.

        Request:
          Content-Type: multipart/form-data
          Field: 'face_image' (optional) — Face image for deepfake detection
          Field: 'document_name' (optional) — Extracted name from document OCR
          Field: 'document_dob' (optional) — Extracted DOB
          Field: 'document_gender' (optional) — Extracted gender
          Field: 'document_id_number' (optional) — Extracted ID number

        Response: JSON with verification decision and all check results.

        The response explicitly lists which checks ran, which were
        unavailable, and the aggregated decision. Biometric data
        (face images, embeddings) is NOT included in the response.
        """
        # Extract document data from form fields
        document_data = None
        if any(request.form.get(f) for f in [
            "document_name", "document_dob",
            "document_gender", "document_id_number"
        ]):
            document_data = {
                "name": request.form.get("document_name"),
                "dob": request.form.get("document_dob"),
                "gender": request.form.get("document_gender"),
                "id_number": request.form.get("document_id_number"),
            }

        # Extract face image
        face_image_bytes = None
        if "face_image" in request.files:
            file = request.files["face_image"]
            if file.filename:
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in config.allowed_image_extensions:
                    return jsonify({
                        "status": "error",
                        "message": f"Unsupported image format '{ext}'.",
                    }), 400
                face_image_bytes = file.read()
                if len(face_image_bytes) == 0:
                    face_image_bytes = None

        # Run pipeline
        result = ekyc_pipeline.run_verification(
            face_image_bytes=face_image_bytes,
            document_data=document_data,
        )
        return jsonify(result.to_dict()), 200

    return app


def main():
    """Entry point for running the deepfake API server."""
    parser = argparse.ArgumentParser(description="F³-Net Deepfake Detection API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to F3-Net checkpoint (.pth). Overrides env var F3NET_CHECKPOINT_PATH.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold (MUST be calibrated before production use)",
    )
    args = parser.parse_args()

    config = DeepfakeServiceConfig(
        checkpoint_path=args.checkpoint or os.environ.get(
            "F3NET_CHECKPOINT_PATH", DeepfakeServiceConfig.checkpoint_path
        ),
        device=args.device,
        decision_threshold=args.threshold,
    )

    app = create_app(config)

    logger.info(f"Starting F³-Net Deepfake Detection API on {args.host}:{args.port}")
    logger.info(f"Checkpoint: {config.checkpoint_path}")
    logger.info(f"Device: {config.resolve_device()}")
    logger.info(f"Threshold: {config.decision_threshold} (uncalibrated default)")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
