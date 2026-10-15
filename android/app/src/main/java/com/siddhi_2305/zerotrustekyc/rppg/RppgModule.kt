package com.siddhi_2305.zerotrustekyc.rppg

import android.net.Uri
import com.facebook.react.bridge.*
import java.io.File
import java.util.concurrent.Executors

/**
 * React Native bridge module exposing offline rPPG and challenge analysis to JavaScript.
 */
class RppgModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    private val executor = Executors.newSingleThreadExecutor()
    private val processor = RppgProcessor(reactContext)

    override fun getName(): String = "RppgModule"

    @ReactMethod
    fun processVideo(videoUri: String, promise: Promise) {
        processVideoInternal(videoUri, null, promise)
    }

    @ReactMethod
    fun processVideoWithChallenges(videoUri: String, challengesJson: String?, promise: Promise) {
        processVideoInternal(videoUri, challengesJson, promise)
    }

    private fun processVideoInternal(videoUri: String, challengesJson: String?, promise: Promise) {
        executor.execute {
            try {
                val result = processor.processVideo(videoUri, challengesJson)

                val map = Arguments.createMap().apply {
                    putBoolean("success", result.success)
                    putString("verdictTier", result.verdictTier)
                    putString("status", result.status)
                    putDouble("estimatedBpm", result.estimatedBpm)
                    putString("signalQuality", result.signalQuality)
                    putInt("usableFrames", result.usableFrames)
                    putInt("totalFrames", result.totalFrames)
                    putDouble("faceRatio", result.faceRatio)
                    putDouble("durationSeconds", result.durationSeconds)
                    putDouble("snrDb", result.snrDb)
                    putDouble("peakPowerFraction", result.peakPowerFraction)
                    putBoolean("isLikelyLiveness", result.isLikelyLiveness)
                    putString("message", result.message)

                    // Guided Challenge Results
                    putInt("challengesPassed", result.challengesPassed)
                    putInt("challengesTotal", result.challengesTotal)
                    putBoolean("challengeLivenessPass", result.challengeLivenessPass)

                    val challengesArray = Arguments.createArray()
                    for (ch in result.challenges) {
                        val chMap = Arguments.createMap().apply {
                            putString("id", ch.id)
                            putString("type", ch.type)
                            putString("instruction", ch.instruction)
                            putBoolean("verified", ch.verified)
                            putDouble("confidence", ch.confidence)
                            putString("detail", ch.detail)
                        }
                        challengesArray.pushMap(chMap)
                    }
                    putArray("challenges", challengesArray)

                    // Processing Timing Metrics
                    val m = result.processingMetrics
                    if (m != null) {
                        val metricsMap = Arguments.createMap().apply {
                            putDouble("totalDurationMs", m.totalDurationMs.toDouble())
                            putDouble("frameExtractionMs", m.frameExtractionMs.toDouble())
                            putDouble("landmarkDetectionMs", m.landmarkDetectionMs.toDouble())
                            putDouble("skinExtractionMs", m.skinExtractionMs.toDouble())
                            putDouble("challengeDetectionMs", m.challengeDetectionMs.toDouble())
                            putDouble("dspMs", m.dspMs.toDouble())
                            putDouble("fpsSampled", m.fpsSampled)
                        }
                        putMap("processingMetrics", metricsMap)
                    }
                }

                promise.resolve(map)
            } catch (e: Exception) {
                promise.reject("RPPG_ERROR", "Error executing rPPG analysis: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun deleteVideoFile(videoUri: String, promise: Promise) {
        executor.execute {
            try {
                val uri = Uri.parse(videoUri)
                val path = if (uri.scheme == "file") uri.path ?: videoUri else videoUri
                val file = File(path)
                val deleted = if (file.exists()) file.delete() else true
                promise.resolve(deleted)
            } catch (e: Exception) {
                promise.resolve(false)
            }
        }
    }
}
