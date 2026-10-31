/**
 * Scales and maps normalized RetinaFace bounding box and landmarks
 * to native UI overlay dimensions.
 */
export function scaleCoordinatesToPreview(
  coordX: number,
  coordY: number,
  previewWidth: number,
  previewHeight: number,
  isFrontCamera: boolean = true
) {
  'worklet';
  // Mirror X-axis for natural front camera preview
  const adjustedX = isFrontCamera ? 1.0 - coordX : coordX;
  return {
    x: adjustedX * previewWidth,
    y: coordY * previewHeight,
  };
}