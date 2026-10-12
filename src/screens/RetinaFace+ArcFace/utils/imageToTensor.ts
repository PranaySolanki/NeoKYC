import * as ImageManipulator from 'expo-image-manipulator';
import { decode } from 'base64-arraybuffer';
import { decode as decodeJpeg } from 'jpeg-js'; // or use canvas/bitmap extraction

/**
 * Resizes the captured image to 320x320 and extracts normalized Float32 RGB tensor
 */
export async function photoPathToTensor(photoUri: string): Promise<Float32Array> {
  // 1. Resize photo to exact RetinaFace input shape (320x320)
  const manipResult = await ImageManipulator.manipulateAsync(
    photoUri,
    [{ resize: { width: 320, height: 320 } }],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );

  if (!manipResult.base64) {
    throw new Error('Failed to retrieve base64 image data');
  }

  // 2. Decode JPEG base64 to raw pixel bytes
  const rawBytes = decode(manipResult.base64);
  const jpegData = decodeJpeg(rawBytes, { useTArray: true }); // returns RGBA Uint8Array

  const width = 320;
  const height = 320;
  const float32Array = new Float32Array(1 * width * height * 3);

  // 3. Normalize pixels: RetinaFace expects BGR or normalized RGB (mean subtraction: [104, 117, 123] or [0, 1])
  let srcIdx = 0;
  let dstIdx = 0;

  for (let i = 0; i < width * height; i++) {
    const r = jpegData.data[srcIdx];
    const g = jpegData.data[srcIdx + 1];
    const b = jpegData.data[srcIdx + 2];
    // Alpha is at srcIdx + 3 (ignored)

    // RetinaFace standard normalization (RGB -> [0, 1])
    float32Array[dstIdx] = r / 255.0;
    float32Array[dstIdx + 1] = g / 255.0;
    float32Array[dstIdx + 2] = b / 255.0;

    srcIdx += 4;
    dstIdx += 3;
  }

  return float32Array;
}