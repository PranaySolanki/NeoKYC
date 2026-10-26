package com.siddhi_2305.zerotrustekyc.rppg

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.Image
import android.net.Uri
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Decodes video sequentially and only converts frames needed by the analysis.
 * Random timestamp seeks through MediaMetadataRetriever repeatedly decode keyframes and
 * can take far longer than the recording itself on high-resolution phone videos.
 */
internal class VideoFrameDecoder(private val context: Context) {

    fun decode(
        videoUri: String,
        targetFps: Double,
        rotationDegrees: Int,
        onFrame: (timestampMs: Long, bitmap: Bitmap) -> Unit,
    ): Int {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null

        try {
            val uri = Uri.parse(videoUri)
            if (uri.scheme == "file" || uri.scheme == null) {
                extractor.setDataSource(File(uri.path ?: videoUri).absolutePath)
            } else {
                extractor.setDataSource(context, uri, null)
            }

            var videoTrack = -1
            var videoFormat: MediaFormat? = null
            for (track in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(track)
                if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    videoTrack = track
                    videoFormat = format
                    break
                }
            }
            require(videoTrack >= 0 && videoFormat != null) { "No video track found in recording." }

            extractor.selectTrack(videoTrack)
            val mime = videoFormat.getString(MediaFormat.KEY_MIME)
                ?: error("Video track has no MIME type.")
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(videoFormat, null, null, 0)
            codec.start()

            val frameStepUs = (1_000_000.0 / targetFps).roundToInt().coerceAtLeast(1)
            var nextFrameUs = 0L
            var lastTimestampMs = -1L
            var frameCount = 0
            var inputEnded = false
            var outputEnded = false
            val bufferInfo = MediaCodec.BufferInfo()

            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = codec.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex)
                            ?: error("Video decoder returned an empty input buffer.")
                        inputBuffer.clear()
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(
                                inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputEnded = true
                        } else {
                            val sampleTimeUs = extractor.sampleTime.coerceAtLeast(0L)
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                sampleTimeUs,
                                extractor.sampleFlags,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER,
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED,
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    else -> if (outputIndex >= 0) {
                        val endOfStream = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        if (bufferInfo.size > 0) {
                            val frameUs = bufferInfo.presentationTimeUs.coerceAtLeast(0L)
                            if (frameUs >= nextFrameUs) {
                                val image = codec.getOutputImage(outputIndex)
                                    ?: error("Video decoder did not provide a readable frame.")
                                try {
                                    val bitmap = image.toSmallBitmap(rotationDegrees)
                                    // MediaPipe VIDEO mode requires strictly increasing timestamps.
                                    val timestampMs = max(frameUs / 1_000L, lastTimestampMs + 1L)
                                    onFrame(timestampMs, bitmap)
                                    lastTimestampMs = timestampMs
                                    frameCount++
                                    nextFrameUs += frameStepUs
                                    while (nextFrameUs <= frameUs) nextFrameUs += frameStepUs
                                } finally {
                                    image.close()
                                }
                            }
                        }
                        codec.releaseOutputBuffer(outputIndex, false)
                        outputEnded = endOfStream
                    }
                }
            }

            return frameCount
        } finally {
            try { codec?.stop() } catch (_: Exception) { }
            try { codec?.release() } catch (_: Exception) { }
            try { extractor.release() } catch (_: Exception) { }
        }
    }

    /** Convert a sampled flexible YUV frame directly to a small RGB bitmap. */
    private fun Image.toSmallBitmap(rotationDegrees: Int): Bitmap {
        val srcWidth = width
        val srcHeight = height
        val rotation = ((rotationDegrees % 360) + 360) % 360
        val rotatedWidth = if (rotation == 90 || rotation == 270) srcHeight else srcWidth
        val rotatedHeight = if (rotation == 90 || rotation == 270) srcWidth else srcHeight
        val scale = min(256.0 / rotatedWidth, 340.0 / rotatedHeight)
        val outWidth = max(1, (rotatedWidth * scale).roundToInt())
        val outHeight = max(1, (rotatedHeight * scale).roundToInt())

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBytes = yPlane.buffer.duplicate()
        val uBytes = uPlane.buffer.duplicate()
        val vBytes = vPlane.buffer.duplicate()
        val pixels = IntArray(outWidth * outHeight)

        for (outY in 0 until outHeight) {
            for (outX in 0 until outWidth) {
                val rotatedX = min(rotatedWidth - 1, outX * rotatedWidth / outWidth)
                val rotatedY = min(rotatedHeight - 1, outY * rotatedHeight / outHeight)
                val (x, y) = when (rotation) {
                    90 -> rotatedY to (srcHeight - 1 - rotatedX)
                    180 -> (srcWidth - 1 - rotatedX) to (srcHeight - 1 - rotatedY)
                    270 -> (srcWidth - 1 - rotatedY) to rotatedX
                    else -> rotatedX to rotatedY
                }

                val yValue = yBytes.get(yPlane.buffer.position() + y * yPlane.rowStride + x * yPlane.pixelStride).toInt() and 0xff
                val uvX = x / 2
                val uvY = y / 2
                val uValue = (uBytes.get(uPlane.buffer.position() + uvY * uPlane.rowStride + uvX * uPlane.pixelStride).toInt() and 0xff) - 128
                val vValue = (vBytes.get(vPlane.buffer.position() + uvY * vPlane.rowStride + uvX * vPlane.pixelStride).toInt() and 0xff) - 128
                val luminance = max(0, yValue - 16) * 298
                val red = ((luminance + 409 * vValue + 128) shr 8).coerceIn(0, 255)
                val green = ((luminance - 100 * uValue - 208 * vValue + 128) shr 8).coerceIn(0, 255)
                val blue = ((luminance + 516 * uValue + 128) shr 8).coerceIn(0, 255)
                pixels[outY * outWidth + outX] = (0xff shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }

        return Bitmap.createBitmap(pixels, outWidth, outHeight, Bitmap.Config.ARGB_8888)
    }
}
