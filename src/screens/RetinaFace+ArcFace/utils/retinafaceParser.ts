export interface FaceDetection {
  box: { x1: number; y1: number; x2: number; y2: number };
  score: number;
  landmarks: {
    leftEye: [number, number];
    rightEye: [number, number];
    nose: [number, number];
    mouthLeft: [number, number];
    mouthRight: [number, number];
  };
}

export function parseRetinaFaceOutputs(
  boxesBuffer: Float32Array,
  scoresBuffer: Float32Array,
  landmarksBuffer: Float32Array,
  scoreThreshold: number = 0.3
): FaceDetection | null {
  const numPriors = 4200;
  let bestScore = -1;
  let bestIdx = -1;

  for (let i = 0; i < numPriors; i++) {
    const score = scoresBuffer[i];
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  console.log(`[RetinaFace Debug] Highest Prior Score: ${(bestScore * 100).toFixed(2)}% at index ${bestIdx}`);

  if (bestScore < scoreThreshold || bestIdx === -1) {
    return null;
  }

  const bOffset = bestIdx * 4;
  const x1 = Math.max(0, Math.min(1, boxesBuffer[bOffset]));
  const y1 = Math.max(0, Math.min(1, boxesBuffer[bOffset + 1]));
  const x2 = Math.max(0, Math.min(1, boxesBuffer[bOffset + 2]));
  const y2 = Math.max(0, Math.min(1, boxesBuffer[bOffset + 3]));

  const lOffset = bestIdx * 10;
  const landmarks = {
    leftEye: [landmarksBuffer[lOffset], landmarksBuffer[lOffset + 1]] as [number, number],
    rightEye: [landmarksBuffer[lOffset + 2], landmarksBuffer[lOffset + 3]] as [number, number],
    nose: [landmarksBuffer[lOffset + 4], landmarksBuffer[lOffset + 5]] as [number, number],
    mouthLeft: [landmarksBuffer[lOffset + 6], landmarksBuffer[lOffset + 7]] as [number, number],
    mouthRight: [landmarksBuffer[lOffset + 8], landmarksBuffer[lOffset + 9]] as [number, number],
  };

  return {
    box: { x1, y1, x2, y2 },
    score: bestScore,
    landmarks,
  };
}