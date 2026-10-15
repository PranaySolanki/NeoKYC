package com.siddhi_2305.zerotrustekyc.rppg

import kotlin.math.*

/**
 * Digital Signal Processing utilities for on-device rPPG pulse estimation:
 * - Cooley-Tukey Radix-2 FFT and Inverse FFT
 * - Least-squares polynomial baseline detrending
 * - Zero-phase frequency-domain Bandpass Filtering (42 - 180 BPM)
 * - POS (Plane-Orthogonal-to-Skin) chrominance projection algorithm
 * - Welch Power Spectral Density (PSD) estimation & Signal-to-Noise Ratio (SNR)
 */
object DspUtils {

    const val MIN_BPM = 42.0
    const val MAX_BPM = 180.0
    const val WINDOW_SECONDS = 1.6
    const val CARDIAC_HALF_BAND_HZ = 0.10 // +/- 6 BPM around detected peak

    data class PsdResult(
        val frequencies: DoubleArray,
        val power: DoubleArray,
        val peakHz: Double,
        val bpm: Double,
        val peakPowerFraction: Double,
        val snrDb: Double
    )

    /**
     * POS (Plane-Orthogonal-to-Skin) rPPG extraction algorithm
     * Based on Wang et al., IEEE TBME 2017: "Algorithmic Principles of Remote PPG"
     *
     * @param rgb Array of [R, G, B] values for each temporal frame
     * @param fps Effective sampling rate in frames per second
     * @return Raw extracted pulsatile signal of length rgb.size
     */
    fun posSignal(rgb: List<DoubleArray>, fps: Double, windowSec: Double = WINDOW_SECONDS): DoubleArray {
        val count = rgb.size
        val window = max(8, (windowSec * fps).roundToInt())
        val output = DoubleArray(count)
        val weights = DoubleArray(count)

        if (count < window) {
            // Signal shorter than sliding window: process as single segment
            return singleSegmentPos(rgb)
        }

        for (start in 0..(count - window)) {
            val end = start + window

            // 1. Temporal mean of current segment
            var meanR = 0.0
            var meanG = 0.0
            var meanB = 0.0
            for (i in start until end) {
                meanR += rgb[i][0]
                meanG += rgb[i][1]
                meanB += rgb[i][2]
            }
            meanR /= window
            meanG /= window
            meanB /= window

            if (meanR < 1e-6 || meanG < 1e-6 || meanB < 1e-6) {
                continue
            }

            // 2. Temporal normalization & projection to S1, S2
            val s1 = DoubleArray(window)
            val s2 = DoubleArray(window)
            var sumS1 = 0.0
            var sumS2 = 0.0

            for (i in 0 until window) {
                val idx = start + i
                val rNorm = rgb[idx][0] / meanR
                val gNorm = rgb[idx][1] / meanG
                val bNorm = rgb[idx][2] / meanB

                s1[i] = gNorm - bNorm
                s2[i] = gNorm + bNorm - 2.0 * rNorm
                sumS1 += s1[i]
                sumS2 += s2[i]
            }

            // 3. Compute standard deviations
            val avgS1 = sumS1 / window
            val avgS2 = sumS2 / window
            var varS1 = 0.0
            var varS2 = 0.0
            for (i in 0 until window) {
                val diff1 = s1[i] - avgS1
                val diff2 = s2[i] - avgS2
                varS1 += diff1 * diff1
                varS2 += diff2 * diff2
            }
            val std1 = sqrt(varS1 / window)
            val std2 = sqrt(varS2 / window)

            if (std2 < 1e-8) {
                continue
            }

            // 4. Alpha tuning factor
            val alpha = std1 / std2

            // 5. Combine and remove segment mean
            var sumH = 0.0
            val h = DoubleArray(window)
            for (i in 0 until window) {
                h[i] = s1[i] + alpha * s2[i]
                sumH += h[i]
            }
            val avgH = sumH / window

            // 6. Overlap-add
            for (i in 0 until window) {
                val target = start + i
                output[target] += (h[i] - avgH)
                weights[target] += 1.0
            }
        }

        // Normalize by overlap weights
        for (i in 0 until count) {
            if (weights[i] > 0.0) {
                output[i] /= weights[i]
            }
        }

        return output
    }

    private fun singleSegmentPos(rgb: List<DoubleArray>): DoubleArray {
        val count = rgb.size
        var meanR = 0.0; var meanG = 0.0; var meanB = 0.0
        for (item in rgb) {
            meanR += item[0]; meanG += item[1]; meanB += item[2]
        }
        meanR = max(1e-6, meanR / count)
        meanG = max(1e-6, meanG / count)
        meanB = max(1e-6, meanB / count)

        val s1 = DoubleArray(count)
        val s2 = DoubleArray(count)
        var sumS1 = 0.0; var sumS2 = 0.0
        for (i in 0 until count) {
            val rNorm = rgb[i][0] / meanR
            val gNorm = rgb[i][1] / meanG
            val bNorm = rgb[i][2] / meanB
            s1[i] = gNorm - bNorm
            s2[i] = gNorm + bNorm - 2.0 * rNorm
            sumS1 += s1[i]; sumS2 += s2[i]
        }
        val avgS1 = sumS1 / count
        val avgS2 = sumS2 / count
        var varS1 = 0.0; var varS2 = 0.0
        for (i in 0 until count) {
            varS1 += (s1[i] - avgS1) * (s1[i] - avgS1)
            varS2 += (s2[i] - avgS2) * (s2[i] - avgS2)
        }
        val alpha = if (varS2 > 1e-12) sqrt(varS1 / varS2) else 1.0
        val out = DoubleArray(count)
        var sumOut = 0.0
        for (i in 0 until count) {
            out[i] = s1[i] + alpha * s2[i]
            sumOut += out[i]
        }
        val avgOut = sumOut / count
        for (i in 0 until count) {
            out[i] -= avgOut
        }
        return out
    }

    /**
     * Least-squares linear detrending: y[n] - (beta0 + beta1 * n)
     */
    fun detrend(signal: DoubleArray): DoubleArray {
        val n = signal.size
        if (n < 2) return signal.clone()

        var sumX = 0.0
        var sumY = 0.0
        var sumXX = 0.0
        var sumXY = 0.0

        for (i in 0 until n) {
            val x = i.toDouble()
            val y = signal[i]
            sumX += x
            sumY += y
            sumXX += x * x
            sumXY += x * y
        }

        val denominator = n * sumXX - sumX * sumX
        if (abs(denominator) < 1e-12) return signal.clone()

        val slope = (n * sumXY - sumX * sumY) / denominator
        val intercept = (sumY - slope * sumX) / n

        val detrended = DoubleArray(n)
        for (i in 0 until n) {
            detrended[i] = signal[i] - (intercept + slope * i)
        }
        return detrended
    }

    /**
     * Zero-phase frequency-domain bandpass filter in cardiac pulse band [minHz, maxHz]
     * Includes smooth cosine taper on transition edges to prevent Gibbs ringing.
     */
    fun bandpassFilter(signal: DoubleArray, fps: Double, minBpm: Double = MIN_BPM, maxBpm: Double = MAX_BPM): DoubleArray {
        val n = signal.size
        if (n < 8) return signal.clone()

        val detrended = detrend(signal)
        val fftSize = nextPowerOf2(n)

        val real = DoubleArray(fftSize)
        val imag = DoubleArray(fftSize)
        System.arraycopy(detrended, 0, real, 0, n)

        // Apply 5% Tukey taper to signal edges to avoid FFT edge leakage
        val taperLen = max(2, (n * 0.05).toInt())
        for (i in 0 until taperLen) {
            val w = 0.5 * (1.0 - cos(PI * i / taperLen))
            real[i] *= w
            real[n - 1 - i] *= w
        }

        fft(real, imag)

        val minHz = minBpm / 60.0
        val maxHz = min(maxBpm / 60.0, fps * 0.48)
        val transHz = 0.10 // 0.10 Hz transition band

        for (k in 0..(fftSize / 2)) {
            val freq = k * fps / fftSize
            var gain = 0.0

            if (freq in minHz..maxHz) {
                gain = 1.0
            } else if (freq in (minHz - transHz)..minHz) {
                // Low transition taper
                gain = 0.5 * (1.0 + cos(PI * (minHz - freq) / transHz))
            } else if (freq in maxHz..(maxHz + transHz)) {
                // High transition taper
                gain = 0.5 * (1.0 + cos(PI * (freq - maxHz) / transHz))
            }

            real[k] *= gain
            imag[k] *= gain

            if (k > 0 && k < fftSize / 2) {
                val mirror = fftSize - k
                real[mirror] *= gain
                imag[mirror] *= gain
            }
        }

        ifft(real, imag)

        val filtered = DoubleArray(n)
        System.arraycopy(real, 0, filtered, 0, n)
        return filtered
    }

    /**
     * Welch Power Spectral Density (PSD) estimation & Signal-to-Noise Ratio (SNR)
     */
    fun estimateBpmAndQuality(signal: DoubleArray, fps: Double): PsdResult {
        val n = signal.size
        if (n < 16) {
            return PsdResult(DoubleArray(0), DoubleArray(0), 0.0, 0.0, 0.0, 0.0)
        }

        // Segment size: min(n, 8 * fps), at least 32 points
        val segLen = min(n, max(32, (8.0 * fps).roundToInt()))
        val step = max(1, segLen / 2) // 50% overlap
        val nFft = max(1024, nextPowerOf2(segLen * 4)) // High frequency resolution

        val numFreqBins = nFft / 2 + 1
        val avgPower = DoubleArray(numFreqBins)
        val freqs = DoubleArray(numFreqBins) { k -> k * fps / nFft }

        // Precompute Hann window
        val hann = DoubleArray(segLen)
        var hannSumSq = 0.0
        for (i in 0 until segLen) {
            hann[i] = 0.5 * (1.0 - cos(2.0 * PI * i / (segLen - 1)))
            hannSumSq += hann[i] * hann[i]
        }
        if (hannSumSq < 1e-12) hannSumSq = 1.0

        var segmentsCount = 0
        var segStart = 0

        while (segStart + segLen <= n) {
            val real = DoubleArray(nFft)
            val imag = DoubleArray(nFft)

            var segMean = 0.0
            for (i in 0 until segLen) {
                segMean += signal[segStart + i]
            }
            segMean /= segLen

            for (i in 0 until segLen) {
                real[i] = (signal[segStart + i] - segMean) * hann[i]
            }

            fft(real, imag)

            for (k in 0 until numFreqBins) {
                val p = (real[k] * real[k] + imag[k] * imag[k]) / (fps * hannSumSq)
                avgPower[k] += p
            }

            segmentsCount++
            segStart += step
        }

        // If no full segments fit, do a single window on full length
        if (segmentsCount == 0) {
            val real = DoubleArray(nFft)
            val imag = DoubleArray(nFft)
            val mean = signal.average()
            for (i in 0 until n) {
                val w = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)))
                real[i] = (signal[i] - mean) * w
            }
            fft(real, imag)
            for (k in 0 until numFreqBins) {
                avgPower[k] = (real[k] * real[k] + imag[k] * imag[k])
            }
            segmentsCount = 1
        } else {
            for (k in 0 until numFreqBins) {
                avgPower[k] /= segmentsCount
            }
        }

        // Evaluate pulse frequency in cardiac range [MIN_BPM, MAX_BPM]
        val minHz = MIN_BPM / 60.0
        val maxHz = min(MAX_BPM / 60.0, fps * 0.48)

        var peakIdx = -1
        var maxP = -1.0
        var totalCardiacPower = 0.0

        for (k in 0 until numFreqBins) {
            val f = freqs[k]
            if (f in minHz..maxHz) {
                val p = avgPower[k]
                totalCardiacPower += p
                if (p > maxP) {
                    maxP = p
                    peakIdx = k
                }
            }
        }

        if (peakIdx == -1 || totalCardiacPower <= 1e-12) {
            return PsdResult(freqs, avgPower, 0.0, 0.0, 0.0, 0.0)
        }

        val peakHz = freqs[peakIdx]
        val bpm = peakHz * 60.0

        // In-band peak power within +/- CARDIAC_HALF_BAND_HZ (+/- 6 BPM)
        var peakBandPower = 0.0
        for (k in 0 until numFreqBins) {
            val f = freqs[k]
            if (f in minHz..maxHz && abs(f - peakHz) <= CARDIAC_HALF_BAND_HZ) {
                peakBandPower += avgPower[k]
            }
        }

        val peakPowerFraction = (peakBandPower / totalCardiacPower).coerceIn(0.0, 1.0)
        val noisePower = max(0.0, totalCardiacPower - peakBandPower)
        val snrDb = if (noisePower > 1e-12 && peakBandPower > 1e-12) {
            10.0 * log10(peakBandPower / noisePower)
        } else {
            0.0
        }

        return PsdResult(
            frequencies = freqs,
            power = avgPower,
            peakHz = peakHz,
            bpm = bpm,
            peakPowerFraction = peakPowerFraction,
            snrDb = snrDb
        )
    }

    // ---------------- FAST FOURIER TRANSFORM (RADIX-2) ----------------

    fun nextPowerOf2(value: Int): Int {
        var p = 1
        while (p < value) {
            p = p shl 1
        }
        return p
    }

    fun fft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        if (n <= 1) return

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
            var m = n shr 1
            while (m >= 1 && j >= m) {
                j -= m
                m = m shr 1
            }
            j += m
        }

        // Cooley-Tukey butterfly computations
        var len = 2
        while (len <= n) {
            val half = len shr 1
            val theta = -2.0 * PI / len
            val wStepR = cos(theta)
            val wStepI = sin(theta)

            var i = 0
            while (i < n) {
                var wR = 1.0
                var wI = 0.0
                for (k in 0 until half) {
                    val uR = real[i + k]
                    val uI = imag[i + k]
                    val vR = real[i + k + half] * wR - imag[i + k + half] * wI
                    val vI = real[i + k + half] * wI + imag[i + k + half] * wR

                    real[i + k] = uR + vR
                    imag[i + k] = uI + vI
                    real[i + k + half] = uR - vR
                    imag[i + k + half] = uI - vI

                    val nextWR = wR * wStepR - wI * wStepI
                    val nextWI = wR * wStepI + wI * wStepR
                    wR = nextWR
                    wI = nextWI
                }
                i += len
            }
            len = len shl 1
        }
    }

    fun ifft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        // Complex conjugate
        for (i in 0 until n) {
            imag[i] = -imag[i]
        }
        fft(real, imag)
        for (i in 0 until n) {
            real[i] /= n
            imag[i] = -imag[i] / n
        }
    }
}
