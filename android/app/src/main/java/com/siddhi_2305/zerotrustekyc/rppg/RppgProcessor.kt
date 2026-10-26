package com.siddhi_2305.zerotrustekyc.rppg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
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
 * 1. Sequential video decoding with sampled, aspect-preserving frame conversion
 * 2. MediaPipe Tasks Vision FaceLandmarker in RunningMode.VIDEO
 * 3. Real-time facial action challenge verification:
 *    - Blink detection via bilateral Eye Aspect Ratio (EAR)
 *    - Smile detection via lip corner expansion ratio
 *    - Head yaw rotation (Turn left / right)
 *    - Head pitch tilt (Tilt up)
 *    - Frontal center alignment
 * 4. Distinct rPPG pulse analysis on stable frontal windows avoiding motion-artifact corruption
 * 5. Comprehensive stage-by-stage timing measurements
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
        private const val TARGET_FRAME_WIDTH = 256
        private const val TARGET_FRAME_HEIGHT = 340

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
        val status: String,
        val verdict: String = "INCONCLUSIVE",
        val verdictTitle: String = "INCONCLUSIVE — TRY AGAIN",
        val verdictDescription: String = "",
        val spoofReason: String? = null,
        val estimatedBpm: Double = 0.0,
        val signalQuality: String = "UNRELIABLE",
        val usableFrames: Int = 0,
        val totalFrames: Int = 0,
        val faceRatio: Double = 0.0,
        val durationSeconds: Double = 0.0,
        val snrDb: Double = 0.0,
        val peakPowerFraction: Double = 0.0,
        val isLikelyLiveness: Boolean = false,
        val message: String = "",
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
                    return failureResult("Video file not found at path: $path")
                }
                retriever.setDataSource(file.absolutePath)
            } else {
                retriever.setDataSource(context, uri)
            }

            // Extract duration and video dimensions
            val durationMsStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationMsStr?.toLongOrNull() ?: 0L
            val durationSec = durationMs / 1000.0

            if (durationSec < 4.0) {
                return Result(
                    success = false,
                    status = "INCONCLUSIVE_TOO_SHORT",
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

            val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
            val rotationDeg = rotationStr?.toIntOrNull() ?: 0

            // Initialize MediaPipe FaceLandmarker in VIDEO running mode
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.20f)
                .setMinFacePresenceConfidence(0.20f)
                .setMinTrackingConfidence(0.20f)
                .build()

            val faceLandmarker = FaceLandmarker.createFromOptions(context, options)
            landmarker = faceLandmarker

            val rawRgbSamples = ArrayList<DoubleArray?>()
            val yawValues = ArrayList<Double>()
            val allEarSamples = ArrayList<Double>()
            val allSmileSamples = ArrayList<Double>()
            val allPitchSamples = ArrayList<Double>()
            val allDepthSamples = ArrayList<Double>()
            val allSkinRgb = ArrayList<DoubleArray>()
            var usableFrames = 0

            // Initialize challenge trackers
            val challengeEvaluators = challenges.map { ChallengeEvaluator(it) }

            var frameIdx = 0
            // Sequential decoding avoids repeatedly seeking and decoding from keyframes.
            val totalFrames = VideoFrameDecoder(context).decode(
                videoUriString,
                TARGET_FPS,
                rotationDeg,
            ) { currentTimestampMs, frameBitmap ->
                try {
                // 2. MediaPipe FaceLandmarker inference
                val mpImage = BitmapImageBuilder(frameBitmap).build()
                val tStartLandmark = System.currentTimeMillis()
                val detectionResult: FaceLandmarkerResult? = try {
                    faceLandmarker.detectForVideo(mpImage, currentTimestampMs)
                } catch (e: Exception) {
                    Log.w(TAG, "FaceLandmarker detect failed at frame $frameIdx: ${e.message}")
                    null
                }
                val tAfterLandmark = System.currentTimeMillis()
                totalLandmarkMs += (tAfterLandmark - tStartLandmark)

                var frameRgb: DoubleArray? = null
                var frameYaw = 0.0

                if (detectionResult != null && detectionResult.faceLandmarks().isNotEmpty()) {
                    val landmarks = detectionResult.faceLandmarks()[0]

                    val leftEar = computeEyeAspectRatio(landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP1, LEFT_EYE_TOP2, LEFT_EYE_BOT1, LEFT_EYE_BOT2)
                    val rightEar = computeEyeAspectRatio(landmarks, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP1, RIGHT_EYE_TOP2, RIGHT_EYE_BOT1, RIGHT_EYE_BOT2)
                    val avgEar = (leftEar + rightEar) / 2.0
                    allEarSamples.add(avgEar)

                    val smileScore = computeSmileScore(landmarks)
                    allSmileSamples.add(smileScore)

                    frameYaw = computeHeadYaw(landmarks)
                    yawValues.add(frameYaw)

                    val pitchVal = computeHeadPitch(landmarks)
                    allPitchSamples.add(pitchVal)

                    val depthVal = computeFaceDepth(landmarks)
                    allDepthSamples.add(depthVal)

                    // 3. Skin ROI Extraction
                    val tStartSkin = System.currentTimeMillis()
                    frameRgb = extractMeanSkinRgb(frameBitmap, landmarks)
                    val tAfterSkin = System.currentTimeMillis()
                    totalSkinExtractionMs += (tAfterSkin - tStartSkin)
                    if (frameRgb != null) {
                        allSkinRgb.add(frameRgb)
                    }

                    // 4. Challenge Tracking
                    val tStartChallenge = System.currentTimeMillis()
                    val curSec = currentTimestampMs / 1000.0

                    for (evaluator in challengeEvaluators) {
                        if (curSec in evaluator.challenge.startTimeSec..evaluator.challenge.endTimeSec) {
                            evaluator.processFrame(landmarks)
                        }
                    }
                    val tAfterChallenge = System.currentTimeMillis()
                    totalChallengeMs += (tAfterChallenge - tStartChallenge)
                } else {
                    yawValues.add(0.0)
                }

                if (frameRgb != null) {
                    usableFrames++
                    rawRgbSamples.add(frameRgb)
                } else {
                    rawRgbSamples.add(null)
                }
                frameIdx++
                } finally {
                    // Process one sampled bitmap at a time to keep peak memory bounded.
                    frameBitmap.recycle()
                }
            }

            if (totalFrames == 0) {
                return failureResult("No video frames could be decoded from recording.")
            }

            val faceRatio = usableFrames.toDouble() / totalFrames

            // Quality check 1: Usable face presence ratio
            if (faceRatio < 0.40) {
                return Result(
                    success = false,
                    status = "INCONCLUSIVE_NO_FACE",
                    verdict = "VERIFICATION_FAILED",
                    verdictTitle = "VERIFICATION FAILED — TRY AGAIN",
                    verdictDescription = "Face tracking was lost in ${String.format("%.0f", (1.0 - faceRatio) * 100)}% of frames. Please center your face in good lighting and try again.",
                    estimatedBpm = 0.0,
                    signalQuality = "UNRELIABLE",
                    usableFrames = usableFrames,
                    totalFrames = totalFrames,
                    faceRatio = faceRatio,
                    durationSeconds = durationSec,
                    snrDb = 0.0,
                    peakPowerFraction = 0.0,
                    isLikelyLiveness = false,
                    message = "Face tracking was lost in ${String.format("%.1f", (1.0 - faceRatio) * 100)}% of frames. Keep your face centered in good lighting.",
                    challenges = challengeEvaluators.map { it.finish() },
                    challengesPassed = 0,
                    challengesTotal = challenges.size,
                    challengeLivenessPass = false
                )
            }

            // 5. Evaluate guided challenges
            val challengeResults = challengeEvaluators.map { it.finish() }
            val challengesPassed = challengeResults.count { it.verified }
            val challengeLivenessPass = passesRandomizedActions(challengeResults)

            // 6. Digital Signal Processing (POS + Bandpass + Welch PSD)
            val tStartDsp = System.currentTimeMillis()
            val optimalWindow = selectStablePulseWindow(rawRgbSamples, yawValues, TARGET_FPS)

            var status = "INCONCLUSIVE_MOTION"
            var qualityRating = "UNRELIABLE"
            var estimatedBpm = 0.0
            var peakFraction = 0.0
            var snrDb = 0.0
            var pulseMessage = "There was not enough steady face footage to measure pulse."

            if (optimalWindow != null) {
                val cleanRgbSeries = ArrayList<DoubleArray>()
                val dspFrames = optimalWindow.size
                for (channel in 0..2) {
                    val channelValues = DoubleArray(dspFrames)
                    for (i in 0 until dspFrames) {
                        val s = optimalWindow[i]
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

                val hasValidBpm = psd.bpm in DspUtils.MIN_BPM..DspUtils.MAX_BPM
                peakFraction = psd.peakPowerFraction
                snrDb = psd.snrDb

                if (hasValidBpm && peakFraction >= 0.38 && snrDb >= -1.0) {
                    status = "CONFIRMED_PULSE"
                    qualityRating = if (peakFraction >= 0.55 && snrDb >= 1.5) "HIGH" else "MODERATE"
                    estimatedBpm = psd.bpm
                    pulseMessage = "A consistent pulse-like rhythm was detected (${String.format("%.1f", psd.bpm)} BPM)."
                } else if (hasValidBpm && peakFraction >= 0.25 && snrDb >= -3.5) {
                    status = "INCONCLUSIVE_WEAK_SIGNAL"
                    qualityRating = "LOW"
                    estimatedBpm = psd.bpm
                    pulseMessage = "A weak pulse candidate was detected."
                } else {
                    status = "INCONCLUSIVE_WEAK_SIGNAL"
                    qualityRating = "UNRELIABLE"
                    estimatedBpm = 0.0
                    pulseMessage = "The pulse signal was too weak to assess."
                }
            } else {
                status = "INCONCLUSIVE_MOTION"
                qualityRating = "UNRELIABLE"
                estimatedBpm = 0.0
                pulseMessage = "There was not enough continuous, steady face footage to estimate a pulse reliably."
            }
            val totalDspMs = System.currentTimeMillis() - tStartDsp

            // 7. Multi-Signal Anti-Spoof Analysis
            val earMean = if (allEarSamples.isNotEmpty()) allEarSamples.average() else 0.0
            val earStdDev = if (allEarSamples.size > 1) {
                sqrt(allEarSamples.map { (it - earMean) * (it - earMean) }.average())
            } else 0.0
            val earRange = if (allEarSamples.isNotEmpty()) (allEarSamples.maxOrNull() ?: 0.0) - (allEarSamples.minOrNull() ?: 0.0) else 0.0

            val smileMean = if (allSmileSamples.isNotEmpty()) allSmileSamples.average() else 0.0
            val smileStdDev = if (allSmileSamples.size > 1) {
                sqrt(allSmileSamples.map { (it - smileMean) * (it - smileMean) }.average())
            } else 0.0
            val smileRange = if (allSmileSamples.isNotEmpty()) (allSmileSamples.maxOrNull() ?: 0.0) - (allSmileSamples.minOrNull() ?: 0.0) else 0.0

            // Check A: Static printed photograph / still image spoof
            val isStaticPhoto = (faceRatio >= 0.55 && usableFrames >= 35) &&
                (earStdDev < 0.0035 && earRange < 0.015) &&
                (smileStdDev < 0.0025 && smileRange < 0.012) &&
                (challengesPassed == 0) &&
                (peakFraction < 0.22)

            // Check B: Physical mask / non-skin chrominance check
            var isMaskDetected = false
            if (allSkinRgb.size >= 25 && challengesPassed == 0 && peakFraction < 0.20) {
                val avgR = allSkinRgb.map { it[0] }.average()
                val avgG = allSkinRgb.map { it[1] }.average()
                val avgB = allSkinRgb.map { it[2] }.average()
                val rgRatio = if (avgG > 1e-4) avgR / avgG else 1.0
                val bgRatio = if (avgG > 1e-4) avgB / avgG else 1.0
                if (avgR < avgG * 0.92 || rgRatio > 2.6 || bgRatio > 1.35 || bgRatio < 0.30) {
                    isMaskDetected = true
                }
            }

            // 8. Final 4-Tier Verdict Decision Engine
            val verdict: String
            val verdictTitle: String
            val verdictDescription: String
            var spoofReason: String? = null
            val isLikelyLiveness: Boolean

            if (isStaticPhoto) {
                verdict = "SPOOF_DETECTED"
                verdictTitle = "SPOOF / FAKE FACE DETECTED"
                verdictDescription = "A static photograph or printed image was detected. Live physical face required."
                spoofReason = "STATIC_PHOTO"
                isLikelyLiveness = false
            } else if (isMaskDetected) {
                verdict = "SPOOF_DETECTED"
                verdictTitle = "SPOOF / FAKE FACE DETECTED"
                verdictDescription = "Non-human skin texture or physical mask detected. Live physical face required."
                spoofReason = "MASK_DETECTED"
                isLikelyLiveness = false
            } else if (faceRatio < 0.50 || usableFrames < 30) {
                verdict = "VERIFICATION_FAILED"
                verdictTitle = "VERIFICATION FAILED — TRY AGAIN"
                verdictDescription = "Face tracking was lost or unclear. Please center your face in good lighting and try again."
                isLikelyLiveness = false
            } else if (!challengeLivenessPass) {
                verdict = "VERIFICATION_FAILED"
                verdictTitle = "VERIFICATION FAILED — TRY AGAIN"
                verdictDescription = "One or more required gestures were not completed. Please follow each on-screen prompt and try again."
                isLikelyLiveness = false
            } else {
                // All required gesture challenges passed! Evaluate biometric & rPPG signals together:
                val hasLivingDynamics = (earRange >= 0.035 || smileRange >= 0.03)
                if (status == "CONFIRMED_PULSE" && faceRatio >= 0.65) {
                    verdict = "REAL_FACE_VERIFIED"
                    verdictTitle = "REAL FACE VERIFIED"
                    verdictDescription = "Live human face verified through confirmed facial gestures and physiological consistency."
                    isLikelyLiveness = true
                } else if (faceRatio >= 0.70 && status != "UNRELIABLE" && hasLivingDynamics) {
                    verdict = "REAL_FACE_VERIFIED"
                    verdictTitle = "REAL FACE VERIFIED"
                    verdictDescription = "Live human face verified through confirmed randomized gestures and facial dynamics."
                    isLikelyLiveness = true
                } else if (status == "INCONCLUSIVE_MOTION" || status == "INCONCLUSIVE_WEAK_SIGNAL") {
                    verdict = "INCONCLUSIVE"
                    verdictTitle = "INCONCLUSIVE — TRY AGAIN"
                    verdictDescription = "Facial gestures were detected, but camera signals were inconclusive. Please hold steady in good lighting and try again."
                    isLikelyLiveness = false
                } else {
                    verdict = "INCONCLUSIVE"
                    verdictTitle = "INCONCLUSIVE — TRY AGAIN"
                    verdictDescription = "Available evidence is insufficient to make a reliable decision. Please try again."
                    isLikelyLiveness = false
                }
            }

            val totalDurationMs = System.currentTimeMillis() - t0
            val measuredStageMs = totalLandmarkMs + totalSkinExtractionMs + totalChallengeMs + totalDspMs
            totalExtractionMs = (totalDurationMs - measuredStageMs).coerceAtLeast(0L)
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
                "Liveness Assessment: verdict=$verdict, gestures=$challengesPassed/${challenges.size}, pulse=$status, faceRatio=${String.format("%.2f", faceRatio)}"
            )

            return Result(
                success = true,
                status = status,
                verdict = verdict,
                verdictTitle = verdictTitle,
                verdictDescription = verdictDescription,
                spoofReason = spoofReason,
                estimatedBpm = if (qualityRating != "UNRELIABLE") estimatedBpm else 0.0,
                signalQuality = qualityRating,
                usableFrames = usableFrames,
                totalFrames = totalFrames,
                faceRatio = faceRatio,
                durationSeconds = durationSec,
                snrDb = snrDb,
                peakPowerFraction = peakFraction,
                isLikelyLiveness = isLikelyLiveness,
                message = pulseMessage,
                challenges = challengeResults,
                challengesPassed = challengesPassed,
                challengesTotal = challenges.size,
                challengeLivenessPass = challengeLivenessPass,
                processingMetrics = metrics
            )

        } catch (e: Exception) {
            Log.e(TAG, "Native rPPG processing failed", e)
            return failureResult("Processing exception: ${e.message}")
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

    /**
     * Computes normalized 3D relative depth (nose tip vs mid-cheek contour).
     * For a real 3D face, the nose tip is in front of cheeks (positive depth).
     */
    private fun computeFaceDepth(landmarks: List<NormalizedLandmark>): Double {
        if (landmarks.size <= maxOf(NOSE_TIP, CHEEK_CONTOUR_LEFT, CHEEK_CONTOUR_RIGHT)) return 0.10
        val faceWidth = abs(landmarks[CHEEK_CONTOUR_RIGHT].x() - landmarks[CHEEK_CONTOUR_LEFT].x()).toDouble()
        val midCheekZ = (landmarks[CHEEK_CONTOUR_LEFT].z() + landmarks[CHEEK_CONTOUR_RIGHT].z()).toDouble() / 2.0
        val noseZ = landmarks[NOSE_TIP].z().toDouble()
        return (midCheekZ - noseZ) / max(1e-4, faceWidth)
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

                    if (avgEar < 0.19) {
                        blinkStateClosed = true
                    } else if (blinkStateClosed && avgEar > 0.22) {
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
                    val earDrop = (earSamples.maxOrNull() ?: 0.3) - (earSamples.minOrNull() ?: 0.3)
                    val minEar = earSamples.minOrNull() ?: 0.3
                    val verified = blinkCount >= 1 || (earDrop >= 0.05 && minEar < 0.20)
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.95 else 0.20,
                        detail = if (verified) "Blink completed" else "Blink not completed"
                    )
                }
                "SMILE" -> {
                    val maxSmile = smileScores.maxOrNull() ?: 0.0
                    val minSmile = smileScores.minOrNull() ?: 0.0
                    val smileDelta = maxSmile - minSmile
                    val verified = smileDelta >= 0.04
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.90 else 0.25,
                        detail = if (verified) "Smile completed" else "Smile not completed"
                    )
                }
                "TURN_LEFT" -> {
                    val minYaw = yawSamples.minOrNull() ?: 0.0
                    val maxYaw = yawSamples.maxOrNull() ?: 0.0
                    val verified = (maxYaw - minYaw) >= 0.055
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.92 else 0.30,
                        detail = if (verified) "Left head turn completed" else "Left head turn not completed"
                    )
                }
                "TURN_RIGHT" -> {
                    val minYaw = yawSamples.minOrNull() ?: 0.0
                    val maxYaw = yawSamples.maxOrNull() ?: 0.0
                    val verified = (maxYaw - minYaw) >= 0.055
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.92 else 0.30,
                        detail = if (verified) "Right head turn completed" else "Right head turn not completed"
                    )
                }
                "TILT_UP" -> {
                    val minPitch = pitchSamples.minOrNull() ?: 0.0
                    val maxPitch = pitchSamples.maxOrNull() ?: 0.0
                    val verified = (maxPitch - minPitch) >= 0.04
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.88 else 0.25,
                        detail = if (verified) "Upward head tilt completed" else "Upward head tilt not completed"
                    )
                }
                "LOOK_CENTER" -> {
                    val avgYaw = yawSamples.map { abs(it) }.average()
                    val verified = avgYaw < 0.075 && yawSamples.isNotEmpty()
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = verified,
                        confidence = if (verified) 0.95 else 0.30,
                        detail = if (verified) "Face centered" else "Face not centered"
                    )
                }
                else -> {
                    ChallengeResult(
                        id = challenge.id,
                        type = challenge.type,
                        instruction = challenge.instruction,
                        verified = false,
                        confidence = 0.0,
                        detail = "Gesture not completed"
                    )
                }
            }
        }
    }

    /**
     * Selects the most stable continuous window of duration 6.0s - 8.0s for clean rPPG pulse estimation.
     * Prefers the scheduled still period in the latter half of the recording.
     */
    private fun selectStablePulseWindow(
        samples: List<DoubleArray?>,
        yaws: List<Double>,
        fps: Double
    ): List<DoubleArray?>? {
        val targetDurations = listOf(8.0, 7.0, 6.0)
        for (dur in targetDurations) {
            val windowFrames = (dur * fps).roundToInt()
            if (samples.size < windowFrames) continue

            var bestStartIndex = -1
            var lowestMotionScore = Double.MAX_VALUE
            val step = max(1, (0.5 * fps).roundToInt())

            // Search starting from the later part of the clip (where LOOK_CENTER still window is scheduled)
            for (i in (samples.size - windowFrames) downTo 0 step step) {
                var validCount = 0
                var yawVarSum = 0.0
                var currentGap = 0
                var longestGap = 0
                for (j in i until (i + windowFrames)) {
                    if (samples[j] != null) {
                        validCount++
                        currentGap = 0
                    } else {
                        currentGap++
                        longestGap = max(longestGap, currentGap)
                    }
                    val y = if (j < yaws.size) yaws[j] else 0.0
                    yawVarSum += abs(y)
                }

                val validRatio = validCount.toDouble() / windowFrames
                if (validRatio >= 0.90 && longestGap <= 3) {
                    val motionScore = yawVarSum / windowFrames
                    if (motionScore < 0.085 && motionScore < lowestMotionScore) {
                        lowestMotionScore = motionScore
                        bestStartIndex = i
                    }
                }
            }

            if (bestStartIndex >= 0) {
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

    /** Centering checks are setup, not proof of life; every randomized action must pass. */
    private fun passesRandomizedActions(results: List<ChallengeResult>): Boolean {
        val randomizedActions = results.filter { it.type != "LOOK_CENTER" }
        return randomizedActions.size >= 2 && randomizedActions.all { it.verified }
    }

    private fun failureResult(msg: String): Result {
        return Result(
            success = false,
            status = "ERROR",
            verdict = "VERIFICATION_FAILED",
            verdictTitle = "VERIFICATION FAILED — TRY AGAIN",
            verdictDescription = msg,
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
