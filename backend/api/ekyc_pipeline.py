"""
eKYC verification pipeline orchestrator.

Coordinates the multi-stage identity verification workflow:

  1. Document capture & OCR         (existing — mobile app + MLKit)
  2. Live face capture               (existing — mobile app CameraView)
  3. Face detection & alignment      (FaceDetector / F3NetPreprocessing)
  4. Liveness check                  (placeholder — not yet implemented)
  5. F³-Net deepfake analysis        (F3NetInferenceService)
  6. Face identity matching          (placeholder — ArcFace, not yet implemented)
  7. Neo4j graph fraud analysis      (placeholder — not yet implemented)
  8. Final eKYC decision             (EKYCDecisionEngine)

Design principles:
  - Each check produces its own result; the final decision aggregates them.
  - No single check can unilaterally approve or reject a user.
  - Missing or unavailable checks are recorded, not silently skipped.
  - Thresholds are configurable and documented as uncalibrated.
  - Biometric data stays in memory; nothing is persisted to disk.
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .config import DeepfakeServiceConfig
from .inference_service import F3NetInferenceService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Result Types
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    """Status of an individual verification check."""
    PASSED = "passed"
    FAILED = "failed"
    SUSPICIOUS = "suspicious"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    SKIPPED = "skipped"


class VerificationDecision(str, Enum):
    """Final eKYC verification decision."""
    APPROVED = "approved"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"
    INCOMPLETE = "incomplete"


@dataclass
class CheckResult:
    """Result from a single verification check."""
    check_name: str
    status: CheckStatus
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "details": self.details,
            "message": self.message,
        }


@dataclass
class EKYCVerificationResult:
    """Aggregated result from the full eKYC verification pipeline."""
    decision: VerificationDecision
    checks_completed: List[CheckResult]
    checks_unavailable: List[str]
    processing_time_ms: float
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "checks_completed": [c.to_dict() for c in self.checks_completed],
            "checks_unavailable": self.checks_unavailable,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Decision Configuration
# ---------------------------------------------------------------------------

@dataclass
class EKYCPolicyConfig:
    """
    Policy configuration for the eKYC decision engine.

    All thresholds are UNCALIBRATED DEFAULTS. They must be validated
    against a labeled evaluation dataset before production use.

    The policy determines how individual check results are combined
    into a final verification decision.
    """
    # Deepfake detection
    deepfake_threshold: float = 0.5  # Uncalibrated — see DeepfakeServiceConfig
    deepfake_required: bool = False  # If True, model_unavailable → INCOMPLETE

    # Liveness check (placeholder for future implementation)
    liveness_required: bool = False

    # Face matching (placeholder for future implementation)
    face_match_threshold: float = 0.6  # Uncalibrated
    face_match_required: bool = False

    # Graph fraud (placeholder for future implementation)
    graph_fraud_required: bool = False

    # Decision policy
    auto_reject_on_deepfake: bool = False  # Never auto-reject on uncalibrated threshold
    require_all_checks: bool = False       # If True, any unavailable → INCOMPLETE


# ---------------------------------------------------------------------------
# eKYC Pipeline Orchestrator
# ---------------------------------------------------------------------------

class EKYCPipeline:
    """
    Orchestrates the full eKYC verification workflow.

    The pipeline coordinates all verification checks and produces
    a structured decision. It does NOT make the final approval/rejection
    decision by itself when critical checks are unavailable or when
    thresholds have not been calibrated.
    """

    def __init__(
        self,
        deepfake_service: F3NetInferenceService,
        policy: Optional[EKYCPolicyConfig] = None,
    ):
        self.deepfake_service = deepfake_service
        self.policy = policy or EKYCPolicyConfig()

    def run_deepfake_check(self, image_bytes: bytes) -> CheckResult:
        """
        Runs F³-Net deepfake detection on a captured face image.

        This is a SIGNAL, not proof. The result must be combined with
        other checks (liveness, face matching, fraud analysis) in the
        final eKYC decision.
        """
        if not self.deepfake_service.is_ready:
            return CheckResult(
                check_name="deepfake_detection",
                status=CheckStatus.UNAVAILABLE,
                details={
                    "model_loaded": False,
                    "reason": "No trained F³-Net checkpoint is loaded.",
                },
                message=(
                    "Deepfake detection is unavailable — no trained model checkpoint. "
                    "This check was skipped, not passed."
                ),
            )

        try:
            result = self.deepfake_service.predict_image(image_bytes)
            response_dict = result.to_dict()

            if result.status == "error":
                return CheckResult(
                    check_name="deepfake_detection",
                    status=CheckStatus.ERROR,
                    details=response_dict,
                    message=result.message,
                )

            if result.status == "no_face_detected":
                return CheckResult(
                    check_name="deepfake_detection",
                    status=CheckStatus.ERROR,
                    details=response_dict,
                    message="No face detected in the image for deepfake analysis.",
                )

            # Extract the primary face prediction (first/largest face)
            predictions = result.predictions
            if not predictions:
                return CheckResult(
                    check_name="deepfake_detection",
                    status=CheckStatus.ERROR,
                    details=response_dict,
                    message="Inference completed but no face predictions were returned.",
                )

            primary = predictions[0]
            fake_prob = primary.get("fake_probability", 0.0)
            threshold = primary.get("threshold_used", self.policy.deepfake_threshold)

            if fake_prob >= threshold:
                status = CheckStatus.SUSPICIOUS
                message = (
                    f"Potential manipulation detected (score: {fake_prob:.4f}, "
                    f"threshold: {threshold:.4f}). This flag requires human review — "
                    f"the threshold has not been empirically calibrated."
                )
            else:
                status = CheckStatus.PASSED
                message = (
                    f"No manipulation signal detected (score: {fake_prob:.4f}, "
                    f"threshold: {threshold:.4f}). Note: this does not guarantee "
                    f"authenticity — threshold is uncalibrated."
                )

            return CheckResult(
                check_name="deepfake_detection",
                status=status,
                details={
                    "fake_probability": fake_prob,
                    "real_probability": primary.get("real_probability", 1.0 - fake_prob),
                    "threshold_used": threshold,
                    "threshold_calibrated": False,
                    "faces_analyzed": result.faces_detected,
                    "model_checkpoint_loaded": result.checkpoint_loaded,
                    "processing_time_ms": result.processing_time_ms,
                },
                message=message,
            )

        except Exception as e:
            logger.exception("Deepfake check failed with unexpected error")
            return CheckResult(
                check_name="deepfake_detection",
                status=CheckStatus.ERROR,
                details={"error": str(e)},
                message=f"Deepfake detection failed: {e}",
            )

    def run_liveness_check(self) -> CheckResult:
        """
        Placeholder for liveness detection.
        Currently returns UNAVAILABLE — to be implemented with the
        FaceLivenessScreen mobile module.
        """
        return CheckResult(
            check_name="liveness_detection",
            status=CheckStatus.UNAVAILABLE,
            details={"implemented": False},
            message=(
                "Liveness detection is not yet implemented in the backend. "
                "The mobile FaceLivenessScreen captures data but server-side "
                "verification is pending."
            ),
        )

    def run_face_match_check(self) -> CheckResult:
        """
        Placeholder for face identity matching (e.g., ArcFace).
        Currently returns UNAVAILABLE.
        """
        return CheckResult(
            check_name="face_identity_match",
            status=CheckStatus.UNAVAILABLE,
            details={"implemented": False},
            message=(
                "Face identity matching (ArcFace) is not yet implemented. "
                "The ResultScreen references this capability but the backend "
                "module is pending."
            ),
        )

    def run_graph_fraud_check(self) -> CheckResult:
        """
        Placeholder for Neo4j graph-based fraud analysis.
        Currently returns UNAVAILABLE.
        """
        return CheckResult(
            check_name="graph_fraud_analysis",
            status=CheckStatus.UNAVAILABLE,
            details={"implemented": False},
            message=(
                "Neo4j graph fraud analysis is not yet implemented. "
                "The ResultScreen references this capability but the backend "
                "module is pending."
            ),
        )

    def run_verification(
        self,
        face_image_bytes: Optional[bytes] = None,
        document_data: Optional[Dict[str, Any]] = None,
    ) -> EKYCVerificationResult:
        """
        Runs the full eKYC verification pipeline.

        Actual flow (matching repository state):
          1. Document OCR — handled by mobile app (DocumentScanScreen)
          2. Face capture — handled by mobile app (FaceLivenessScreen)
          3. Deepfake detection — F³-Net (implemented)
          4. Liveness check — placeholder
          5. Face identity match — placeholder
          6. Graph fraud analysis — placeholder
          7. Final decision — aggregation of all checks

        Args:
            face_image_bytes: Raw bytes of the captured face image.
            document_data: Pre-extracted document fields from OCR.

        Returns:
            EKYCVerificationResult with all check outcomes and final decision.
        """
        t0 = time.time()
        checks: List[CheckResult] = []
        unavailable: List[str] = []

        # Step 1: Document verification (data comes from mobile OCR)
        if document_data:
            doc_valid = all(
                document_data.get(f) for f in ["name", "dob", "gender", "id_number"]
            )
            checks.append(CheckResult(
                check_name="document_ocr",
                status=CheckStatus.PASSED if doc_valid else CheckStatus.FAILED,
                details={
                    "fields_present": {
                        k: bool(document_data.get(k))
                        for k in ["name", "dob", "gender", "id_number"]
                    },
                },
                message="Document fields validated." if doc_valid else "Document has missing fields.",
            ))
        else:
            unavailable.append("document_ocr")

        # Step 2: Deepfake detection (F³-Net)
        if face_image_bytes:
            deepfake_result = self.run_deepfake_check(face_image_bytes)
            checks.append(deepfake_result)
            if deepfake_result.status == CheckStatus.UNAVAILABLE:
                unavailable.append("deepfake_detection")
        else:
            unavailable.append("deepfake_detection")

        # Step 3: Liveness check (placeholder)
        liveness_result = self.run_liveness_check()
        checks.append(liveness_result)
        if liveness_result.status == CheckStatus.UNAVAILABLE:
            unavailable.append("liveness_detection")

        # Step 4: Face identity matching (placeholder)
        face_match_result = self.run_face_match_check()
        checks.append(face_match_result)
        if face_match_result.status == CheckStatus.UNAVAILABLE:
            unavailable.append("face_identity_match")

        # Step 5: Graph fraud analysis (placeholder)
        graph_result = self.run_graph_fraud_check()
        checks.append(graph_result)
        if graph_result.status == CheckStatus.UNAVAILABLE:
            unavailable.append("graph_fraud_analysis")

        # Step 6: Final decision
        decision, message = self._make_decision(checks, unavailable)

        elapsed = (time.time() - t0) * 1000
        return EKYCVerificationResult(
            decision=decision,
            checks_completed=checks,
            checks_unavailable=unavailable,
            processing_time_ms=elapsed,
            message=message,
        )

    def _make_decision(
        self,
        checks: List[CheckResult],
        unavailable: List[str],
    ) -> tuple:
        """
        Aggregates individual check results into a final verification decision.

        Rules:
          - Any FAILED check → REJECTED (hard failures)
          - Any SUSPICIOUS check → REVIEW_REQUIRED (needs human review)
          - Any ERROR in required checks → INCOMPLETE
          - Any UNAVAILABLE in required checks → INCOMPLETE
          - If policy.require_all_checks and any unavailable → INCOMPLETE
          - All checks PASSED → APPROVED (with caveats if some unavailable)
        """
        statuses = {c.check_name: c.status for c in checks}

        # Hard failures
        failed = [name for name, s in statuses.items() if s == CheckStatus.FAILED]
        if failed:
            return (
                VerificationDecision.REJECTED,
                f"Verification failed due to: {', '.join(failed)}.",
            )

        # Suspicious signals → always require human review
        suspicious = [name for name, s in statuses.items() if s == CheckStatus.SUSPICIOUS]
        if suspicious:
            return (
                VerificationDecision.REVIEW_REQUIRED,
                f"Suspicious signals detected in: {', '.join(suspicious)}. "
                f"Human review required — thresholds are uncalibrated.",
            )

        # Errors in checks that ran
        errors = [name for name, s in statuses.items() if s == CheckStatus.ERROR]
        if errors:
            return (
                VerificationDecision.REVIEW_REQUIRED,
                f"Errors occurred in: {', '.join(errors)}. Manual review needed.",
            )

        # Required checks that are unavailable
        required_missing = []
        if self.policy.deepfake_required and "deepfake_detection" in unavailable:
            required_missing.append("deepfake_detection")
        if self.policy.liveness_required and "liveness_detection" in unavailable:
            required_missing.append("liveness_detection")
        if self.policy.face_match_required and "face_identity_match" in unavailable:
            required_missing.append("face_identity_match")
        if self.policy.graph_fraud_required and "graph_fraud_analysis" in unavailable:
            required_missing.append("graph_fraud_analysis")

        if required_missing:
            return (
                VerificationDecision.INCOMPLETE,
                f"Required checks are unavailable: {', '.join(required_missing)}. "
                f"Cannot make a verification decision.",
            )

        if self.policy.require_all_checks and unavailable:
            return (
                VerificationDecision.INCOMPLETE,
                f"Policy requires all checks. Unavailable: {', '.join(unavailable)}.",
            )

        # All available checks passed
        if unavailable:
            return (
                VerificationDecision.REVIEW_REQUIRED,
                f"Available checks passed but some checks were unavailable: "
                f"{', '.join(unavailable)}. Consider manual review.",
            )

        return (
            VerificationDecision.APPROVED,
            "All verification checks passed.",
        )
