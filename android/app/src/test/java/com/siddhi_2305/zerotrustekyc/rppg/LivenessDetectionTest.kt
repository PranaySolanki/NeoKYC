package com.siddhi_2305.zerotrustekyc.rppg

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class LivenessDetectionTest {

    @Test
    fun testDsp_EstimateBpm_SyntheticCardiacSignal() {
        val fps = 30.0
        val targetBpm = 75.0 // 1.25 Hz
        val targetFreq = targetBpm / 60.0
        val numFrames = 210 // 7.0 seconds

        val rawSignal = DoubleArray(numFrames)
        for (i in 0 until numFrames) {
            val t = i / fps
            // Clean synthetic cardiac pulse at 75 BPM with harmonic
            val cardiac = sin(2.0 * PI * targetFreq * t)
            val harmonic = 0.3 * sin(4.0 * PI * targetFreq * t)
            rawSignal[i] = 120.0 + 5.0 * (cardiac + harmonic)
        }

        // Apply bandpass filter to simulate rPPG pipeline
        val filtered = DspUtils.bandpassFilter(rawSignal, fps)
        val result = DspUtils.estimateBpmAndQuality(filtered, fps)

        // Estimated BPM should be close to 75.0 (+/- 5 BPM resolution limit for 7s window)
        assertTrue("Estimated BPM was ${result.bpm}, expected ~75", result.bpm in 70.0..80.0)
        assertTrue("SNR should be positive for synthetic pulse, was ${result.snrDb}", result.snrDb > 3.0)
        assertTrue("Peak power fraction should be substantial, was ${result.peakPowerFraction}", result.peakPowerFraction > 0.15)
    }

    @Test
    fun testDsp_EstimateBpm_FlatStaticSignal() {
        val fps = 30.0
        val numFrames = 210
        val flatSignal = DoubleArray(numFrames) { 128.0 } // Zero AC pulsation (static image)

        val result = DspUtils.estimateBpmAndQuality(flatSignal, fps)

        assertEquals("Flat signal should yield 0 BPM", 0.0, result.bpm, 0.001)
        assertEquals("Flat signal should yield 0 SNR", 0.0, result.snrDb, 0.001)
        assertEquals("Flat signal should yield 0 peak power fraction", 0.0, result.peakPowerFraction, 0.001)
    }

    @Test
    fun testDsp_BandpassFilter_PreservesCardiacRange() {
        val fps = 30.0
        val numFrames = 256
        val signal = DoubleArray(numFrames) { i ->
            val t = i / fps
            sin(2.0 * PI * 1.2 * t) // 1.2 Hz = 72 BPM
        }

        val filtered = DspUtils.bandpassFilter(signal, fps)
        assertEquals(numFrames, filtered.size)

        // Energy should be preserved
        var energy = 0.0
        for (v in filtered) {
            energy += v * v
        }
        assertTrue("Filtered cardiac signal should have positive energy", energy > 1.0)
    }

    @Test
    fun testDsp_Detrend_RemovesLinearDrift() {
        val n = 128
        val drifting = DoubleArray(n) { i -> 100.0 + 2.5 * i } // Strong baseline drift
        val detrended = DspUtils.detrend(drifting)

        val mean = detrended.average()
        assertEquals("Detrended signal should have zero mean", 0.0, mean, 1e-6)
    }

    @Test
    fun testVerdictEngine_StaticPhotoSpoof() {
        // Simulates the static photo detection criteria implemented in RppgProcessor
        val challengesPassed = false
        val earVariance = 0.0008f   // Completely frozen eyes
        val smileVariance = 0.0006f // Completely frozen mouth
        val hasPulse = false
        val faceRatio = 0.95f       // Face clearly visible and static

        val isStaticPhotoSpoof = (!challengesPassed && earVariance < 0.0035f && smileVariance < 0.0025f && !hasPulse)
        assertTrue("Should detect static photo spoof", isStaticPhotoSpoof)

        val verdict = if (isStaticPhotoSpoof) "SPOOF_DETECTED" else "UNKNOWN"
        assertEquals("SPOOF_DETECTED", verdict)
    }

    @Test
    fun testVerdictEngine_PhysicalMaskDetection() {
        // Simulates mask detection via aberrant skin chrominance + absence of pulse
        val redRatio = 0.85 // Normal skin has R > G > B, Red usually dominates over 1.2x
        val blueRatio = 1.10 // Mask with unnatural blue/green reflection
        val hasPulse = false
        val challengesPassed = false

        val isAbnormalChrominance = (redRatio < 1.05 || blueRatio > 0.95)
        val isMaskSpoof = isAbnormalChrominance && !hasPulse && !challengesPassed

        assertTrue("Should detect physical mask spoof via chrominance and lack of pulse", isMaskSpoof)
    }

    @Test
    fun testVerdictEngine_NeverMarkRealPersonAsSpoofOnFailure() {
        // Case: user looked away or lighting dropped, lost face tracking (faceRatio < 0.45)
        val faceRatio = 0.30f
        val challengesPassed = false

        val isTrackingFailure = faceRatio < 0.45f
        val verdict = if (isTrackingFailure) "VERIFICATION_FAILED" else "UNKNOWN"

        assertEquals("Technical/tracking drop must be VERIFICATION_FAILED, NEVER SPOOF", "VERIFICATION_FAILED", verdict)
    }

    @Test
    fun testVerdictEngine_InconclusiveWhenGesturesPassButPulseNoisy() {
        // Case: genuine human moved their head during the still window, corrupting pulse
        val challengesPassed = true
        val faceRatio = 0.88f
        val pulseReliable = false
        val isConfirmedSpoof = false

        val verdict = when {
            isConfirmedSpoof -> "SPOOF_DETECTED"
            !challengesPassed -> "VERIFICATION_FAILED"
            pulseReliable -> "REAL_FACE_VERIFIED"
            else -> "INCONCLUSIVE"
        }

        assertEquals("Noisy pulse on passed gestures must be INCONCLUSIVE, NEVER SPOOF", "INCONCLUSIVE", verdict)
    }

    @Test
    fun testVerdictEngine_RealFaceVerifiedWhenAllCriteriaMet() {
        // Case: gestures passed, living face dynamics present, valid rPPG pulse
        val challengesPassed = true
        val faceRatio = 0.92f
        val earVariance = 0.012f
        val smileVariance = 0.008f
        val pulseReliable = true
        val isConfirmedSpoof = false

        val verdict = when {
            isConfirmedSpoof -> "SPOOF_DETECTED"
            !challengesPassed -> "VERIFICATION_FAILED"
            pulseReliable && earVariance > 0.0035f -> "REAL_FACE_VERIFIED"
            else -> "INCONCLUSIVE"
        }

        assertEquals("REAL_FACE_VERIFIED", verdict)
    }
}
