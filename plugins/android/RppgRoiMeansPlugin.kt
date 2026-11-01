import android.graphics.ImageFormat
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry
import kotlin.math.max
import kotlin.math.min

/** Samples mean RGB values from worklet-supplied skin ROIs in YUV frames. */
class RppgRoiMeansPlugin : FrameProcessorPlugin() {
  override fun callback(frame: Frame, params: Map<String, Any>?): Any? {
    val image = frame.getImage()
    if (image.format != ImageFormat.YUV_420_888 || image.planes.size < 3) return null
    if (params == null) return null
    val planes = image.planes
    val yPlane = planes[0]
    val uPlane = planes[1]
    val vPlane = planes[2]
    val yBytes = yPlane.buffer.duplicate()
    val uBytes = uPlane.buffer.duplicate()
    val vBytes = vPlane.buffer.duplicate()
    val width = image.width
    val height = image.height
    // Frame.orientation describes the image orientation; reverse it to get the
    // rotation needed to map upright ROI coordinates back into the raw YUV image.
    val rotation = frame.getOrientation().reversed().toSurfaceRotation() * 90
    val uprightWidth = if (rotation == 90 || rotation == 270) height else width
    val uprightHeight = if (rotation == 90 || rotation == 270) width else height
    var red = 0.0
    var green = 0.0
    var blue = 0.0
    var count = 0

    for (index in 0..2) {
      val left = ((params["roi${index}Left"] as? Number)?.toDouble() ?: continue) * uprightWidth
      val top = ((params["roi${index}Top"] as? Number)?.toDouble() ?: continue) * uprightHeight
      val right = ((params["roi${index}Right"] as? Number)?.toDouble() ?: continue) * uprightWidth
      val bottom = ((params["roi${index}Bottom"] as? Number)?.toDouble() ?: continue) * uprightHeight
      var y = max(0, top.toInt())
      val yEnd = min(uprightHeight, bottom.toInt())
      while (y < yEnd) {
        var x = max(0, left.toInt())
        val xEnd = min(uprightWidth, right.toInt())
        while (x < xEnd) {
          val raw = mapUprightToRaw(x, y, width, height, rotation)
          if (raw.first in 0 until width && raw.second in 0 until height) {
            val yy = yBytes.get(raw.second * yPlane.rowStride + raw.first * yPlane.pixelStride).toInt() and 0xff
            val uu = uBytes.get((raw.second / 2) * uPlane.rowStride + (raw.first / 2) * uPlane.pixelStride).toInt() and 0xff
            val vv = vBytes.get((raw.second / 2) * vPlane.rowStride + (raw.first / 2) * vPlane.pixelStride).toInt() and 0xff
            val c = max(0, yy - 16)
            red += clamp((298 * c + 409 * (vv - 128) + 128) shr 8)
            green += clamp((298 * c - 100 * (uu - 128) - 208 * (vv - 128) + 128) shr 8)
            blue += clamp((298 * c + 516 * (uu - 128) + 128) shr 8)
            count++
          }
          x += 6
        }
        y += 6
      }
    }
    if (count == 0) return null
    return mapOf("r" to red / count, "g" to green / count, "b" to blue / count)
  }

  private fun clamp(value: Int) = min(255, max(0, value))

  private fun mapUprightToRaw(x: Int, y: Int, width: Int, height: Int, rotation: Int): Pair<Int, Int> = when (rotation) {
    90 -> y to (height - 1 - x)
    180 -> (width - 1 - x) to (height - 1 - y)
    270 -> (width - 1 - y) to x
    else -> x to y
  }

  companion object {
    fun register() {
      FrameProcessorPluginRegistry.addFrameProcessorPlugin("rppgRoiMeans") { _, _ -> RppgRoiMeansPlugin() }
    }
  }
}
