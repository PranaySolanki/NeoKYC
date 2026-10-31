import * as ImageManipulator from 'expo-image-manipulator';
import { decode as decodeBase64 } from 'base64-arraybuffer';
import jpeg from 'jpeg-js';

export interface PreprocessOutput {
  tensor: Float32Array;
  croppedUri: string;
}

export async function preprocessPhotoForRetinaFace(
  photoUri: string,
  imgWidth?: number,
  imgHeight?: number
): Promise<PreprocessOutput> {
  const w = imgWidth || 1080;
  const h = imgHeight || 1920;

  // 1. Calculate square center-crop based on shorter dimension
  const squareDim = Math.min(w, h);
  const originX = Math.round((w - squareDim) / 2);
  const originY = Math.round((h - squareDim) / 2);

  // Crop to square, then scale to exactly 320x320
  const actions: ImageManipulator.Action[] = [
    {
      crop: {
        originX,
        originY,
        width: squareDim,
        height: squareDim,
      },
    },
    {
      resize: {
        width: 320,
        height: 320,
      },
    },
  ];

  const manipResult = await ImageManipulator.manipulateAsync(
    photoUri,
    actions,
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );

  if (!manipResult.base64) {
    throw new Error('ImageManipulator failed to generate base64 data.');
  }

  // 2. Decode raw uncompressed RGB pixels
  const jpegBuffer = decodeBase64(manipResult.base64) as ArrayBuffer;
  const rawPixels = jpeg.decode(jpegBuffer, { useTArray: true });

  const numPixels = 320 * 320;
  const inputTensor = new Float32Array(numPixels * 3);

  // 3. RetinaFace MobileNet0.25 mean subtraction: RGB [123, 117, 104]
  const meanR = 123.0;
  const meanG = 117.0;
  const meanB = 104.0;

  for (let i = 0; i < numPixels; i++) {
    const r = rawPixels.data[i * 4];
    const g = rawPixels.data[i * 4 + 1];
    const b = rawPixels.data[i * 4 + 2];

    inputTensor[i * 3] = r - meanR;
    inputTensor[i * 3 + 1] = g - meanG;
    inputTensor[i * 3 + 2] = b - meanB;
  }

  return {
    tensor: inputTensor,
    croppedUri: manipResult.uri,
  };
}