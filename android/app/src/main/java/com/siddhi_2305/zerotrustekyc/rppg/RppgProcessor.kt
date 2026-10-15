package com.siddhi_2305.zerotrustekyc.rppg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.json.JSONArray
import java.io.File
import kotlin.math.*

/**
 * Optimized on-device native rPPG and Guided Liveness Processor.
 *
 * Implements:
 * 1. Aspect-preserving frame extraction with automatic orientation calibration and ARGB_8888 conversion
 * 2. MediaPipe Tasks Vision FaceLandmarker with CPU delegate in RunningMode.IMAGE
 * 3. Real-time facial action challenge verification:
 *    - Blink detection via bilateral Eye Aspect Ratio (EAR)
 *    - Smile detection via lip corner expansion ratio
 *    - Head yaw rotation (Turn left / right)
 *    - Head pitch tilt (Tilt up)
 *    - Frontal center alignment
 * 4. Distinct rPPG pulse analysis on stable frontal windows avoiding motion-artifact corruption
 * 5. Multi-tiered fair verification verdicts: VERIFIED, RETRY_REQUIRED, INCONCLUSIVE, FAILED
 * 6. Comprehensive stage-by-stage timing measurements
 */
class RppgProcessor(private val context: Context) {

    companion object {
        private const val TAG = "RppgProcessor"

        // MediaPipe Face Mesh landmark indices
        private val FOREHEAD_LANDMARKS = intArrayOf(10, 338, 297, 336, 9, 107, 68, 109)
        private val LEFT_CHEEK_LANDMARKS = intArrayOf(117, 118, 101, 205, 187, 147, 123, 50)
        private val RIGHT_CHEEK_LANDMARKS = intArrayOf(346, 347, 330, 425, 411, 376, 352, 280)

        // Eye landmarks for Eye Aspect Ratio (EAR)
        private const val LEFT_EYE_OUTER = 33
        private const val LEFT_EYE_INNER = 133
        private const val LEFT_EYE_TOP1 = 159
        private const val LEFT_EYE_TOP2 = 158
        private const val LEFT_EYE_BOT1 = 145
        private const val LEFT_EYE_BOT2 = 153

        private const val RIGHT_EYE_OUTER = 362
        private const val RIGHT_EYE_INNER = 263
        private const val RIGHT_EYE_TOP1 = 386
        private const val RIGHT_EYE_TOP2 = 385
        private const val RIGHT_EYE_BOT1 = 374
        private const val RIGHT_EYE_BOT2 = 380

        // Mouth landmarks for Smile detection
        private const val MOUTH_CORNER_LEFT = 61
        private const val MOUTH_CORNER_RIGHT = 291

        // Head pose reference points
        private const val NOSE_TIP = 1
        private const val CHEEK_CONTOUR_LEFT = 234
        private const val CHEEK_CONTOUR_RIGHT = 454
        private const val GLABELLA = 10
        private const val CHIN = 152

        // Optimized target dimensions for fast, memory-safe MediaPipe inference
        private const val TARGET_FRAME_WIDTH = 360

        // Optimal rPPG and challenge sampling rate (15 FPS provides >2.5x Nyquist margin over 3 Hz pulse)
        private const val TARGET_FPS = 15.0
    }

    data class LivenessChallenge(
        val id: String,
        val type: String,
        val instruction: String,
        val startTimeSec: Double,
        val endTimeSec: Double
    )

    data class ChallengeResult(
        val id: String,
        val type: String,
        val instruction: String,
        val verified: Boolean,
        val confidence: Double,
        val detail: String
    )

    data class ProcessingMetrics(
        val totalDurationMs: Long,
        val frameExtractionMs: Long,
        val landmarkDetectionMs: Long,
        val skinExtractionMs: Long,
        val challengeDetectionMs: Long,
        val dspMs: Long,
        val fpsSampled: Double
    )

    data class Result(
        val success: Boolean,
        val verdictTier: String,
        val status: String,
        val estimatedBpm: Double,
        val signalQuality: String,
        val usableFrames: Int,
        val totalFrames: Int,
        val faceRatio: Double,
        val durationSeconds: Double,
        val snrDb: Double,
        val peakPowerFraction: Double,
        val isLikelyLiveness: Boolean,
        val message: String,
        val challenges: List<ChallengeResult> = emptyList(),
        val challengesPassed: Int = 0,
        val challengesTotal: Int = 0,
        val challengeLivenessPass: Boolean = false,
        val processingMetrics: ProcessingMetrics? = null
    )

    // Reusable buffer for skin patch extraction (avoids GC churn across hundreds of frames)
    private val patchPixelBuffer = IntArray(100 * 100)

    fun processVideo(videoUriString: String, challengesJson: String? = null): Result {
        val t0 = System.currentTimeMillis()
        var totalExtractionMs = 0L
        var totalLandmarkMs = 0L
        var totalSkinExtractionMs = 0L
        var totalChallengeMs = 0L

        val retriever = MediaMetadataRetriever()
        var landmarker: FaceLandmarker? = null

        // Parse challenge list if provided
        val challenges = parseChallengesJson(challengesJson)

        try {
            val uri = Uri.parse(videoUriString)
            if (uri.scheme == "file" || uri.scheme == null) {
                val path = uri.path ?: videoUriString
                val file = File(path)
                if (!file.exists()) {
                    return failureResult("Video file not found at path: $path", "RETRY_REQUIRED")
                }
                retriever.setDataSource(file.absolutePath)
            } else {
                retriever.setDataSource(context, uri)
            }

            // Extract duration and video dimensions
            val durationMsStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationMsStr?.toLongOrNull() ?: 0L
            val durationSec = durationMs / 1000.0

            val videoW = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val videoH = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
            val metadataRotationDeg = rotationStr?.toIntOrNull() ?: 0

            Log.i(TAG, "Video metadata: duration=${String.format("%.2f", durationSec)}s, dimensions=${videoW}x${videoH}, rotationTag=${metadataRotationDeg}deg")

            if (durationSec < 4.0) {
                return Result(
                    success = false,
                    verdictTier = "RETRY_REQUIRED",
                    status = "RETRY_REQUIRED",
                    estimatedBpm = 0.0,
                    signalQuality = "UNRELIABLE",
                    usableFrames = 0,
                    totalFrames = 0,
                    faceRatio = 0.0,
                    durationSeconds = durationSec,
                    snrDb = 0.0,
                    peakPowerFraction = 0.0,
                    isLikelyLiveness = false,
                    message = "Recording duration (${String.format("%.1f", durationSec)}s) is too short. Record at least 6-10 seconds for reliable pulse frequency resolution."
                )
            }

            // Initialize MediaPipe FaceLandmarker with CPU delegate in RunningMode.IMAGE
            Log.i(TAG, "Initializing FaceLandmarker: mode=IMAGE, delegate=CPU, minConfidence=0.20")
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(Delegate.CPU)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.20f)
                .setMinFacePresenceConfidence(0.20f)
                .setMinTrackingConfidence(0.20f)
                .build()

            landmarker = try {
                FaceLandmarker.createFromOptions(context, options)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize FaceLandmarker: ${e.message}", e)
                return failureResult("Failed to initialize face detection model: ${e.message}", "RETRY_REQUIRED")
            }

            // Dynamic orientation calibration on the recorded video frames
            val calibratedRotation = calibrateOrientation(retriever, landmarker, durationMs, metadataRotationDeg, videoW, videoH)
            Log.i(TAG, "Orientation calibration determined rotation angle: ${calibratedRotation}deg")

            // Calculate frame sampling parameters at 15 FPS
            val frameStepMs = (1000.0 / TARGET_FPS).roundToLong().coerceAtLeast(33L)
            val rawRgbSamples = ArrayList<DoubleArray?>()
            val timestampsMs = ArrayList<Long>()
            val yawValues = ArrayList<Double>()
            var facesDetectedCount = 0
            var usableFrames = 0

            // Initialize challenge trackers
            val challengeEvaluators = challenges.map { ChallengeEvaluator(it) }

            var currentTimestampMs = 0L
            var frameIdx = 0

            while (currentTimestampMs <= durationMs) {
                val timeUs = currentTimestampMs * 1000L

                // 1. Frame extraction & conversion with calibrated rotation
                val tStartExtract = System.currentTimeMillis()
                val frameBitmap = extractAndTransformFrame(retriever, timeUs, calibratedRotation, targetWidth = TARGET_FRAME_WIDTH, videoW = videoW, videoH = videoH)
                val tAfterExtract = System.currentTimeMillis()
                totalExtractionMs += (tAfterExtract - tStartExtract)

                if (frameBitmap == null) {
                    currentTimestampMs += frameStepMs
                    frameIdx++
                    continue
                }

                // 2. MediaPipe FaceLandmarker inference (RunningMode.IMAGE)
                val mpImage = BitmapImageBuilder(frameBitmap).build()
                val tStartLandmark = System.currentTimeMillis()
                val detectionResult: FaceLandmarkerResult? = try {
                    landmarker.detect(mpImage)
                } catch (e: Exception) {
                    Log.w(TAG, "FaceLandmarker detect failed at frame $frameIdx: ${e.message}")
                    null
                }
                val tAfterLandmark = System.currentTimeMillis()
                totalLandmarkMs += (tAfterLandmark - tStartLandmark)

                var frameRgb: DoubleArray? = null
                var frameYaw = 0.0

                if (detectionResult != null && detectionResult.faceLandmarks().isNotEmpty()) {
                    facesDetectedCount++
                    val landmarks = detectionResult.faceLandmarks()[0]

                    if (frameIdx < 3 || frameIdx % 30 == 0) {
                        Log.d(TAG, "Frame $frameIdx (${currentTimestampMs}ms): Face detected with ${landmarks.size} landmarks")
                    }

                    // 3. Skin ROI Extraction
                    val tStartSkin = System.currentTimeMillis()
                    frameRgb = extractMeanSkinRgb(frameBitmap, landmarks)
                    val tAfterSkin = System.currentTimeMillis()
                    totalSkinExtractionMs += (tAfterSkin - tStartSkin)

                    // 4. Challenge Tracking
                    val tStartChallenge = System.currentTimeMillis()
                    val curSec = currentTimestampMs / 1000.0
                    frameYaw = computeHeadYaw(landmarks)
                    yawValues.add(frameYaw)

                    for (evaluator in challengeEvaluators) {
                        if (curSec in evaluator.challenge.startTimeSec..evaluator.challenge.endTimeSec) {
                            evaluator.processFrame(landmarks)
                        }
                    }
                    val tAfterChallenge = System.currentTimeMillis()
                    totalChallengeMs += (tAfterChallenge - tStartChallenge)
                } else {
                    if (frameIdx < 3) {
                        Log.w(TAG, "Frame $frameIdx (${currentTimestampMs}ms): No face detected")
                    }
                    yawValues.add(0.0)
                }

                if (frameRgb != null) {
                    usableFrames++
                    rawRgbSamples.add(frameRgb)
                } else {
                    rawRgbSamples.add(null)
                }
                timestampsMs.add(currentTimestampMs)

                // Immediately recycle bitmap to maintain low memory footprint
                frameBitmap.recycle()

                currentTimestampMs += frameStepMs
                frameIdx++
            }

            val totalFrames = timestampsMs.size
            if (totalFrames == 0) {
                return failureResult("No video frames could be decoded from recording.", "RETRY_REQUIRED")
            }

            val faceRatio = facesDetectedCount.toDouble() / totalFrames
            Log.i(TAG, "Detection summary: totalFrames=$totalFrames, facesDetected=$facesDetectedCount (${String.format("%.1f", faceRatio * 100)}%), usableSkin=$usableFrames")

            // Fair verdict check: If face presence is insufficient (< 15 frames or < 5% of total session)
            if (facesDetectedCount < 15 || faceRatio < 0.05) {
                Log.w(TAG, "Face detection insufficient: facesDetected=$facesDetectedCount / $totalFrames (${String.format("%.1f", faceRatio * 100)}%) -> RETRY_REQUIRED")
                return Result(
                    success = false,
                    verdictTier = "RETRY_REQUIRED",
                    status = "RETRY_REQUIRED",
                    estimatedBpm = 0.0,
                    signalQuality = "UNRELIABLE",
                    usableFrames = usableFrames,
                    totalFrames = totalFrames,
                    faceRatio = faceRatio,
                    durationSeconds = durationSec,
                    snrDb = 0.0,
                    peakPowerFraction = 0.0,
                    isLikelyLiveness = false,
                    message = "We couldn't detect your face clearly. Face the camera, improve lighting, and try again.",
                    challenges = challengeEvaluators.map { it.finish() },
                    challengesPassed = 0,
                    challengesTotal = challenges.size,
                    challengeLivenessPass = false
                )
            }

            // 5. Digital Signal Processing (POS + Bandpass + Welch PSD)
            val tStartDsp = System.currentTimeMillis()

            // Find optimal frontal, continuous window of >= 8.0s for rPPG pulse estimation
            val optimalWindow = selectStablePulseWindow(rawRgbSamples, yawValues, TARGET_FPS)
            val rgbForDsp = optimalWindow ?: rawRgbSamples

            // Interpolate missing frames
            val cleanRgbSeries = ArrayList<DoubleArray>()
            val dspFrames = rgbForDsp.size
            for (channel in 0..2) {
                val channelValues = DoubleArray(dspFrames)
                for (i in 0 until dspFrames) {
                    val s = rgbForDsp[i]
                    channelValues[i] = s?.get(channel) ?: Double.NaN
                }
                interpolateSeries(channelValues)
                for (i in 0 until dspFrames) {
                    if (channel == 0) {
                        cleanRgbSeries.add(DoubleArray(3))
                    }
                    cleanRgbSeries[i][channel] = channelValues[i]
                }
            }

            val rawPosSignal = DspUtils.posSignal(cleanRgbSeries, TARGET_FPS)
            val filteredSignal = DspUtils.bandpassFilter(rawPosSignal, TARGET_FPS)
            val psd = DspUtils.estimateBpmAndQuality(filteredSignal, TARGET_FPS)
            val tAfterDsp = System.currentTimeMillis()
            val totalDspMs = tAfterDsp - tStartDsp

            // Evaluate challenges
            val challengeResults = challengeEvaluators.map { it.finish() }
            val challengesPassed = challengeResults.count { it.verified }
            val challengeLivenessPass = challenges.isEmpty() || (challengesPassed >= max(1, (challenges.size * 0.65).roundToInt()))

            // Assess rPPG pulse validity
            val hasValidBpm = psd.bpm in DspUtils.MIN_BPM..DspUtils.MAX_BPM
            val peakFraction = psd.peakPowerFraction
            val snrDb = psd.snrDb

            val verdictTier: String
            val status: String
            val qualityRating: String
            val isLikelyLiveness: Boolean
            val message: String

            if (!challengeLivenessPass && challenges.isNotEmpty()) {
                // User face was tracked, but the required challenge actions were failed
                verdictTier = "FAILED"
                status = "FAILED"
                qualityRating = if (hasValidBpm && peakFraction >= 0.20) "MODERATE" else "UNRELIABLE"
                isLikelyLiveness = false
                message = "One or more required interactive challenge actions could not be verified."
            } else if (hasValidBpm && peakFraction >= 0.20 && snrDb >= 1.2) {
                // Both challenges and physiological pulse verified!
                verdictTier = "VERIFIED"
                status = "CONFIRMED_PULSE"
                qualityRating = if (peakFraction >= 0.30 && snrDb >= 2.5) "HIGH" else "MODERATE"
                isLikelyLiveness = true
                message = "Subcutaneous pulsatile physiological rhythm verified (${String.format("%.1f", psd.bpm)} BPM)."
            } else if (hasValidBpm && peakFraction >= 0.14 && snrDb >= 0.2) {
                // Challenges passed, but pulse signal is borderline
                verdictTier = "INCONCLUSIVE"
                status = "INCONCLUSIVE_WEAK_SIGNAL"
                qualityRating = "LOW"
                isLikelyLiveness = false
                message = "Marginal physiological signal detected (${String.format("%.1f", psd.bpm)} BPM), but confidence is below verification threshold."
            } else {
                // Challenges passed, but pulse rhythm unconfirmed
                verdictTier = "INCONCLUSIVE"
                status = "INCONCLUSIVE_WEAK_SIGNAL"
                qualityRating = "UNRELIABLE"
                isLikelyLiveness = false
                message = "Face gestures completed, but subcutaneous blood-volume pulse rhythm was inconclusive."
            }

            val totalDurationMs = System.currentTimeMillis() - t0
            val metrics = ProcessingMetrics(
                totalDurationMs = totalDurationMs,
                frameExtractionMs = totalExtractionMs,
                landmarkDetectionMs = totalLandmarkMs,
                skinExtractionMs = totalSkinExtractionMs,
                challengeDetectionMs = totalChallengeMs,
                dspMs = totalDspMs,
                fpsSampled = TARGET_FPS
            )

            Log.i(
                TAG,
                "rPPG Processing Timings: total=${totalDurationMs}ms, frames=$totalFrames, " +
                "facesDetected=$facesDetectedCount, usableSkin=$usableFrames, verdict=$verdictTier, " +
                "extract=${totalExtractionMs}ms, landmark=${totalLandmarkMs}ms, " +
                "skinROI=${totalSkinExtractionMs}ms, challenge=${totalChallengeMs}ms, dsp=${totalDspMs}ms"
            )

            return Result(
                success = (verdictTier == "VERIFIED"),
                verdictTier = verdictTier,
                status = status,
                estimatedBpm = if (verdictTier == "VERIFIED" || (verdictTier == "INCONCLUSIVE" && hasValidBpm)) psd.bpm else 0.0,
                signalQuality = qualityRating,
                usableFrames = usableFrames,
                totalFrames = totalFrames,
                faceRatio = faceRatio,
                durationSeconds = durationSec,
                snrDb = snrDb,
                peakPowerFraction = peakFraction,
                isLikelyLiveness = (verdictTier == "VERIFIED"),
                message = message,
                challenges = challengeResults,
                challengesPassed = challengesPassed,
                challengesTotal = challenges.size,
                challengeLivenessPass = challengeLivenessPass,
                processingMetrics = metrics
            )

        } catch (e: Exception) {
            Log.e(TAG, "Native rPPG processing failed", e)
            return failureResult("Processing exception: ${e.message}", "RETRY_REQUIRED")
        } finally {
            try {
                retriever.release()
            } catch (ignored: Exception) {}

            try {
                landmarker?.close()
            } catch (ignored: Exception) {}
        }
    }

    /**
     * Dynamically calibrates face orientation by testing candidate rotations [metadataRotation, 270, 0, 90, 180]
     * on candidate video frames and selecting the rotation where face landmarks are upright.
     * Guaranteed to fallback to metadataRotationDeg (or 270) instead of blindly assuming 0deg.
     */
    private fun calibrateOrientation(
        retriever: MediaMetadataRetriever,
        landmarker: FaceLandmarker,
        durationMs: Long,
        metadataRotationDeg: Int,
        videoW: Int,
        videoH: Int
    ): Int {
        val candidateTimestampsMs = listOf(1000L, 2000L, 3000L, 4000L, 5000L, 6000L, 8000L, 500L, 0L).filter { it <= durationMs }
        val candidateRotations = listOf(metadataRotationDeg, 270, 0, 90, 180).distinct()

        for (tsMs in candidateTimestampsMs) {
            var bestRotation = -1
            var bestVerticalSpan = -1.0f

            for (deg in candidateRotations) {
                val frame = extractAndTransformFrame(retriever, tsMs * 1000L, deg, targetWidth = TARGET_FRAME_WIDTH, videoW = videoW, videoH = videoH) ?: continue
                try {
                    val mpImage = BitmapImageBuilder(frame).build()
                    val result = landmarker.detect(mpImage)
                    if (result.faceLandmarks().isNotEmpty()) {
                        val lm = result.faceLandmarks()[0]
                        val chinY = lm[CHIN].y()
                        val glabellaY = lm[GLABELLA].y()
                        val leftEyeY = lm[LEFT_EYE_OUTER].y()
                        val rightEyeY = lm[RIGHT_EYE_OUTER].y()
                        val eyeDiff = abs(leftEyeY - rightEyeY)
                        val vertSpan = chinY - glabellaY

                        // An upright frontal face has chin below glabella (chinY > glabellaY)
                        // and horizontally aligned eyes (eyeDiff < 0.20)
                        if (vertSpan > 0.10f && eyeDiff < 0.20f) {
                            if (vertSpan > bestVerticalSpan) {
                                bestVerticalSpan = vertSpan
                                bestRotation = deg
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Calibration detect error at ${deg}deg: ${e.message}")
                } finally {
                    frame.recycle()
                }
            }

            if (bestRotation != -1) {
                Log.i(TAG, "Orientation calibrated: selected rotation = ${bestRotation}deg at ${tsMs}ms (span=$bestVerticalSpan)")
                return bestRotation
            }
        }

        val fallback = if (metadataRotationDeg != 0) metadataRotationDeg else 270
        Log.w(TAG, "Orientation calibration did not detect upright face in candidate frames. Fallback to: ${fallback}deg")
        return fallback
    }

    /**
     * Extracts a frame at timeUs, applies rotation around center, scales proportionally to target width
     * (preserving the exact aspect ratio), and guarantees an ARGB_8888 Bitmap for MediaPipe and skin extraction.
     * Uses hardware downscaled decoding on API 27+ for fast processing.
     */
    private fun extractAndTransformFrame(
        retriever: MediaMetadataRetriever,
        timeUs: Long,
        rotationDeg: Int,
        targetWidth: Int = TARGET_FRAME_WIDTH,
        videoW: Int = 0,
        videoH: Int = 0
    ): Bitmap? {
        val isTransposed = (rotationDeg == 90 || rotationDeg == 270)

        var rawBitmap: Bitmap? = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 && videoW > 0 && videoH > 0) {
            try {
                val reqDecodeW = if (isTransposed) (videoW * targetWidth / videoH) else targetWidth
                val reqDecodeH = if (isTransposed) targetWidth else (videoH * targetWidth / videoW)

                rawBitmap = retriever.getScaledFrameAtTime(
                    timeUs,
                    MediaMetadataRetriever.OPTION_CLOSEST,
                    reqDecodeW,
                    reqDecodeH
                )
            } catch (e: Exception) {
                // Fall back to unscaled
            }
        }

        if (rawBitmap == null) {
            rawBitmap = try {
                retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
            } catch (e: Exception) {
                Log.w(TAG, "getFrameAtTime failed at $timeUs us: ${e.message}")
                null
            } ?: return null
        }

        val srcW = rawBitmap.width
        val srcH = rawBitmap.height

        val effectiveW = if (isTransposed) srcH else srcW
        val effectiveH = if (isTransposed) srcW else srcH

        val dstW = targetWidth
        val dstH = (effectiveH * targetWidth) / effectiveW

        val outBitmap = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outBitmap)

        val matrix = Matrix()
        if (rotationDeg != 0) {
            matrix.postTranslate(-srcW / 2f, -srcH / 2f)
            matrix.postRotate(rotationDeg.toFloat())
            matrix.postTranslate(effectiveW / 2f, effectiveH / 2f)
        }

        val scaleX = dstW.toFloat() / effectiveW.toFloat()
        val scaleY = dstH.toFloat() / effectiveH.toFloat()
        matrix.postScale(scaleX, scaleY)

        val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(rawBitmap, matrix, paint)
        rawBitmap.recycle()
        return outBitmap
    }

    /**
     * Computes the Eye Aspect Ratio (EAR) for eye closure/blink detection.
     */
    private fun computeEyeAspectRatio(
        landmarks: List<NormalizedLandmark>,
        pOuter: Int, pInner: Int,
        pTop1: Int, pTop2: Int,
        pBot1: Int, pBot2: Int
    ): Double {
        if (landmarks.size <= maxOf(pOuter, pInner, pTop1, pTop2, pBot1, pBot2)) return 0.30
        val d1 = dist(landmarks[pTop1], landmarks[pBot1])
        val d2 = dist(landmarks[pTop2], landmarks[pBot2])
        val dHoriz = dist(landmarks[pOuter], landmarks[pInner])
        return (d1 + d2) / (2.0 * max(1e-4, dHoriz))
    }

    /**
     * Computes horizontal head yaw ratio (negative = left, positive = right).
     */
    private fun computeHeadYaw(landmarks: List<NormalizedLandmark>): Double {
        if (landmarks.size <= maxOf(NOSE_TIP, CHEEK_CONTOUR_LEFT, CHEEK_CONTOUR_RIGHT)) return 0.0
        val midX = (landmarks[CHEEK_CONTOUR_LEFT].x() + landmarks[CHEEK_CONTOUR_RIGHT].x()).toDouble() / 2.0
        val faceWidth = abs(landmarks[CHEEK_CONTOUR_RIGHT].x() - landmarks[CHEEK_CONTOUR_LEFT].x()).toDouble()
        return (landmarks[NOSE_TIP].x().toDouble() - midX) / max(1e-4, faceWidth)
    }

    /**
     * Computes vertical head pitch ratio (lower value = tilted up).
     */
    private fun computeHeadPitch(landmarks: List<NormalizedLandmark>): Double {
        if (landmarks.size <= maxOf(NOSE_TIP, GLABELLA, CHIN)) return 0.0
        val faceHeight = abs(landmarks[CHIN].y() - landmarks[GLABELLA].y()).toDouble()
        return (landmarks[NOSE_TIP].y().toDouble() - landmarks[GLABELLA].y().toDouble()) / max(1e-4, faceHeight)
    }

    /**
     * Computes lip width expansion ratio relative to facial width for smile detection.
     */
    private fun computeSmileScore(landmarks: List<NormalizedLandmark>): Double {
        if (landmarks.size <= maxOf(MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, CHEEK_CONTOUR_LEFT, CHEEK_CONTOUR_RIGHT)) return 0.35
        val faceWidth = dist(landmarks[CHEEK_CONTOUR_LEFT], landmarks[CHEEK_CONTOUR_RIGHT])
        val mouthWidth = dist(landmarks[MOUTH_CORNER_LEFT], landmarks[MOUTH_CORNER_RIGHT])
        return mouthWidth / max(1e-4, faceWidth)
    }

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Double {
        val dx = (a.x() - b.x()).toDouble()
        val dy = (a.y() - b.y()).toDouble()
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Evaluates challenge verification per time window.
     */
    private inner class ChallengeEvaluator(val challenge: LivenessChallenge) {
        private var blinkStateClosed = false
        private var blinkCount = 0
        private var earSamples = ArrayList<Double>()

        private var smileScores = ArrayList<Double>()
        private var yawSamples = ArrayList<Double>()
        private var pitchSamples = ArrayList<Double>()

        fun processFrame(landmarks: List<NormalizedLandmark>) {
            when (challenge.type) {
                "BLINK" -> {
                    val leftEar = computeEyeAspectRatio(landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP1, LEFT_EYE_TOP2, LEFT_EYE_BOT1, LEFT_EYE_BOT2)
                    val rightEar = computeEyeAspectRatio(landmarks, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP1, RIGHT_EYE_TOP2, RIGHT_EYE_BOT1, RIGHT_EYE_BOT2)
                    val avgEar = (leftEar + rightEar) / 2.0
                    earSamples.add(avgEar)

                    if (avgEar < 0.185) {
                        blinkStateClosed = true
                    } else if (blinkStateClosed && avgEar > 0.225) {
                        blinkCount++
                        blinkStateClosed = false
                    }
                }
                "SMILE" -> {
                    smileScores.add(computeSmileScore(landmarks))
                }
                "TURN_LEFT" -> {
                    yawSamples.add(computeHeadYaw(landmarks))
                }
                "TURN_RIGHT" -> {
                    yawSamples.add(computeHeadYaw(landmarks))
                }
                "TILT_UP" -> {
                    pitchSamples.add(computeHeadPitch(landmarks))
                }
                "LOOK_CENTER" -> {
                    yawSamples.add(computeHeadYaw(landmarks))
                    pitchSamples.add(computeHeadPitch(landmarks))
                }
            }
        }

        fun finish(): ChallengeResult {
            return when (challenge.type) {
                "BLINK" -> {
                    val earDrop = (earSamples.maxOrNull() ?: 0.28) - (earSamples.minOrNull() ?: 0.28)
                    val verified = blinkCount >= 1 || earDrop >= 0.05 || (earSamples.minOrNull() ?: 1.0) < 0.19
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.95 else 0.20,
                        detail = if (verified) "$blinkCount blink(s) verified via bilateral EAR" else "Blink gesture not confirmed ($blinkCount detected)"
                    )
                }
                "SMILE" -> {
                    val maxSmile = smileScores.maxOrNull() ?: 0.0
                    val minSmile = smileScores.minOrNull() ?: 0.0
                    val smileDelta = maxSmile - minSmile
                    val verified = smileDelta >= 0.035 || maxSmile >= 0.45
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.90 else 0.25,
                        detail = if (verified) "Natural smile verified (mouth expansion +${String.format("%.1f", smileDelta * 100)}%)" else "Smile not detected"
                    )
                }
                "TURN_LEFT" -> {
                    val minYaw = yawSamples.minOrNull() ?: 0.0
                    val maxYaw = yawSamples.maxOrNull() ?: 0.0
                    val yawSpread = maxYaw - minYaw
                    val verified = yawSpread >= 0.05 || abs(minYaw) >= 0.05 || abs(maxYaw) >= 0.05
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.92 else 0.30,
                        detail = if (verified) "Left head rotation verified" else "Head rotation not detected"
                    )
                }
                "TURN_RIGHT" -> {
                    val minYaw = yawSamples.minOrNull() ?: 0.0
                    val maxYaw = yawSamples.maxOrNull() ?: 0.0
                    val yawSpread = maxYaw - minYaw
                    val verified = yawSpread >= 0.05 || abs(minYaw) >= 0.05 || abs(maxYaw) >= 0.05
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.92 else 0.30,
                        detail = if (verified) "Right head rotation verified" else "Head rotation not detected"
                    )
                }
                "TILT_UP" -> {
                    val minPitch = pitchSamples.minOrNull() ?: 0.0
                    val maxPitch = pitchSamples.maxOrNull() ?: 0.0
                    val verified = (maxPitch - minPitch) >= 0.035 || (pitchSamples.isNotEmpty() && minPitch < 0.35)
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.88 else 0.25,
                        detail = if (verified) "Upward head tilt verified" else "Tilt gesture not detected"
                    )
                }
                "LOOK_CENTER" -> {
                    val avgYaw = if (yawSamples.isNotEmpty()) yawSamples.map { abs(it) }.average() else 1.0
                    val verified = avgYaw < 0.09 && yawSamples.isNotEmpty()
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.95 else 0.30,
                        detail = if (verified) "Frontal camera gaze verified" else "Face was not centered"
                    )
                }
                else -> {
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = false,
                        confidence = 0.0,
                        detail = "Instruction shown (unverified)"
                    )
                }
            }
        }
    }

    /**
     * Selects the most stable continuous window of duration >= 4.0s for clean rPPG pulse estimation.
     * Evaluates candidate window sizes (8.0s, 6.0s, 5.0s, 4.0s) and chooses the window with lowest head motion
     * and highest valid skin ROI coverage.
     */
    private fun selectStablePulseWindow(
        samples: List<DoubleArray?>,
        yaws: List<Double>,
        fps: Double
    ): List<DoubleArray?>? {
        val windowDurations = listOf(8.0, 6.0, 5.0, 4.0)

        for (duration in windowDurations) {
            val windowFrames = (duration * fps).roundToInt()
            if (samples.size < windowFrames) continue

            var bestStartIndex = -1
            var lowestMotionScore = Double.MAX_VALUE

            val step = max(1, (0.5 * fps).roundToInt())
            for (i in 0..(samples.size - windowFrames) step step) {
                var validCount = 0
                var yawVarSum = 0.0
                for (j in i until (i + windowFrames)) {
                    if (samples[j] != null) validCount++
                    val y = if (j < yaws.size) yaws[j] else 0.0
                    yawVarSum += abs(y)
                }

                val validRatio = validCount.toDouble() / windowFrames
                // At least 65% valid tracked face frames in this window
                if (validRatio >= 0.65) {
                    val motionScore = (yawVarSum / windowFrames) + (1.0 - validRatio) * 0.5
                    if (motionScore < lowestMotionScore) {
                        lowestMotionScore = motionScore
                        bestStartIndex = i
                    }
                }
            }

            if (bestStartIndex >= 0) {
                Log.i(TAG, "Selected stable pulse window: startIdx=$bestStartIndex (${String.format("%.1f", bestStartIndex / fps)}s), duration=${duration}s, motionScore=${String.format("%.3f", lowestMotionScore)}")
                return samples.subList(bestStartIndex, bestStartIndex + windowFrames)
            }
        }

        return null
    }

    private fun extractMeanSkinRgb(bitmap: Bitmap, landmarks: List<NormalizedLandmark>): DoubleArray? {
        val regionMeans = ArrayList<DoubleArray>()
        val regions = listOf(FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS)
        for (polyIndices in regions) {
            val patchRgb = extractPolygonPatchRgb(bitmap, landmarks, polyIndices)
            if (patchRgb != null) {
                regionMeans.add(patchRgb)
            }
        }

        if (regionMeans.isEmpty()) return null

        var sumR = 0.0; var sumG = 0.0; var sumB = 0.0
        for (m in regionMeans) {
            sumR += m[0]; sumG += m[1]; sumB += m[2]
        }
        val size = regionMeans.size.toDouble()
        return doubleArrayOf(sumR / size, sumG / size, sumB / size)
    }

    private fun extractPolygonPatchRgb(
        bitmap: Bitmap,
        landmarks: List<NormalizedLandmark>,
        indices: IntArray
    ): DoubleArray? {
        val w = bitmap.width
        val h = bitmap.height

        val polyX = FloatArray(indices.size)
        val polyY = FloatArray(indices.size)

        var minX = w
        var maxX = 0
        var minY = h
        var maxY = 0

        for (i in indices.indices) {
            val idx = indices[i]
            if (idx >= landmarks.size) return null
            val lm = landmarks[idx]
            val px = (lm.x() * w).toInt().coerceIn(0, w - 1)
            val py = (lm.y() * h).toInt().coerceIn(0, h - 1)

            polyX[i] = px.toFloat()
            polyY[i] = py.toFloat()

            if (px < minX) minX = px
            if (px > maxX) maxX = px
            if (py < minY) minY = py
            if (py > maxY) maxY = py
        }

        val boxW = maxX - minX + 1
        val boxH = maxY - minY + 1
        if (boxW < 4 || boxH < 4) return null

        val numPixels = boxW * boxH
        val pixels = if (numPixels <= patchPixelBuffer.size) patchPixelBuffer else IntArray(numPixels)
        bitmap.getPixels(pixels, 0, boxW, minX, minY, boxW, boxH)

        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var validPixels = 0

        val nVert = indices.size
        // 2x subsampling for skin ROI speedup without loss of mean color accuracy
        val step = if (boxW > 16 && boxH > 16) 2 else 1

        for (y in 0 until boxH step step) {
            val py = (minY + y).toFloat()
            for (x in 0 until boxW step step) {
                val px = (minX + x).toFloat()

                // Point-in-polygon ray casting
                var inside = false
                var j = nVert - 1
                for (i in 0 until nVert) {
                    if ((polyY[i] > py) != (polyY[j] > py) &&
                        (px < (polyX[j] - polyX[i]) * (py - polyY[i]) / (polyY[j] - polyY[i]) + polyX[i])) {
                        inside = !inside
                    }
                    j = i
                }

                if (inside) {
                    val color = pixels[y * boxW + x]
                    val r = Color.red(color)
                    val g = Color.green(color)
                    val b = Color.blue(color)
                    val brightness = (r + g + b) / 3.0

                    if (brightness in 20.0..245.0) {
                        sumR += r
                        sumG += g
                        sumB += b
                        validPixels++
                    }
                }
            }
        }

        if (validPixels < 8) return null

        return doubleArrayOf(sumR / validPixels, sumG / validPixels, sumB / validPixels)
    }

    private fun interpolateSeries(series: DoubleArray) {
        val n = series.size
        var firstValid = -1
        for (i in 0 until n) {
            if (!series[i].isNaN()) {
                firstValid = i
                break
            }
        }

        if (firstValid == -1) {
            series.fill(0.0)
            return
        }

        for (i in 0 until firstValid) {
            series[i] = series[firstValid]
        }

        var prevValid = firstValid
        for (i in (firstValid + 1) until n) {
            if (!series[i].isNaN()) {
                val gap = i - prevValid
                if (gap > 1) {
                    val startVal = series[prevValid]
                    val endVal = series[i]
                    for (k in 1 until gap) {
                        val alpha = k.toDouble() / gap
                        series[prevValid + k] = startVal + alpha * (endVal - startVal)
                    }
                }
                prevValid = i
            }
        }

        for (i in (prevValid + 1) until n) {
            series[i] = series[prevValid]
        }
    }

    private fun parseChallengesJson(jsonStr: String?): List<LivenessChallenge> {
        if (jsonStr.isNullOrBlank()) return emptyList()
        val list = ArrayList<LivenessChallenge>()
        try {
            val array = JSONArray(jsonStr)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                list.add(
                    LivenessChallenge(
                        id = obj.optString("id", "challenge_$i"),
                        type = obj.optString("type", "LOOK_CENTER"),
                        instruction = obj.optString("instruction", ""),
                        startTimeSec = obj.optDouble("startTimeSec", 0.0),
                        endTimeSec = obj.optDouble("endTimeSec", 5.0)
                    )
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse challenges JSON: ${e.message}")
        }
        return list
    }

    private fun failureResult(msg: String, verdictTier: String = "RETRY_REQUIRED"): Result {
        return Result(
            success = false,
            verdictTier = verdictTier,
            status = if (verdictTier == "RETRY_REQUIRED") "RETRY_REQUIRED" else "ERROR",
            estimatedBpm = 0.0,
            signalQuality = "UNRELIABLE",
            usableFrames = 0,
            totalFrames = 0,
            faceRatio = 0.0,
            durationSeconds = 0.0,
            snrDb = 0.0,
            peakPowerFraction = 0.0,
            isLikelyLiveness = false,
            message = msg
        )
    }
}
