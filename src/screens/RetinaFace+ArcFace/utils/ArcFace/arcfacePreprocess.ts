import * as ImageManipulator from 'expo-image-manipulator';
import UPNG from 'upng-js';
import { decode as atob } from 'base-64';
import { FaceDetection } from '../RetinaFace/retinafaceParser';

export interface ArcFaceInputResult {
  tensor: Float32Array;
  croppedFaceUri: string;
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

export async function preprocessFaceForArcFace(
  squareImageUri: string,
  imageDimension: number,
  detection: FaceDetection
): Promise<ArcFaceInputResult> {
  const { box } = detection;

  // Add 15% context margin
  const boxW = (box.x2 - box.x1) * imageDimension;
  const boxH = (box.y2 - box.y1) * imageDimension;
  const marginX = boxW * 0.15;
  const marginY = boxH * 0.15;

  const originX = Math.max(0, Math.floor(box.x1 * imageDimension - marginX));
  const originY = Math.max(0, Math.floor(box.y1 * imageDimension - marginY));
  const cropW = Math.min(imageDimension - originX, Math.floor(boxW + marginX * 2));
  const cropH = Math.min(imageDimension - originY, Math.floor(boxH + marginY * 2));

  // 1. Crop and resize to 112x112 as PNG
  const manipulated = await ImageManipulator.manipulateAsync(
    squareImageUri,
    [
      { crop: { originX, originY, width: cropW, height: cropH } },
      { resize: { width: 112, height: 112 } },
    ],
    { format: ImageManipulator.SaveFormat.PNG, base64: true }
  );

  if (!manipulated.base64) {
    throw new Error('Failed to retrieve base64 data for ArcFace.');
  }

  // 2. Decode the compressed PNG into uncompressed raw RGBA pixels
  const arrayBuffer = base64ToArrayBuffer(manipulated.base64);
  const imgData = UPNG.decode(arrayBuffer);
  const rgbaFrames = UPNG.toRGBA8(imgData);
  const rawPixels = new Uint8Array(rgbaFrames[0]); // Uncompressed 112x112x4 (RGBA)

  // 3. Build NHWC tensor: [1, 112, 112, 3] matching your model log
  const tensor = new Float32Array(1 * 112 * 112 * 3);
  const totalPixels = 112 * 112;

  for (let i = 0; i < totalPixels; i++) {
    const srcIdx = i * 4; // R, G, B, A
    const dstIdx = i * 3; // R, G, B

    // ArcFace normalization: (pixel - 127.5) / 128.0
    tensor[dstIdx] = (rawPixels[srcIdx] - 127.5) / 128.0;         // R
    tensor[dstIdx + 1] = (rawPixels[srcIdx + 1] - 127.5) / 128.0; // G
    tensor[dstIdx + 2] = (rawPixels[srcIdx + 2] - 127.5) / 128.0; // B
  }

  return {
    tensor,
    croppedFaceUri: manipulated.uri,
  };
}