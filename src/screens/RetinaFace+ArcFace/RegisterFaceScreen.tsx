import React, { useRef, useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { parseRetinaFaceOutputs, FaceDetection } from './utils/retinafaceParser';
import { preprocessPhotoForRetinaFace } from './utils/imagePreprocess';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const SQUARE_SIZE = SCREEN_WIDTH * 0.9; // 90% of screen width for viewfinder & display

interface RegisterFaceScreenProps {
  onNavigate?: (screenName: string) => void;
}

export default function RegisterFaceScreen({ onNavigate }: RegisterFaceScreenProps = {}) {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();

  const retinaPlugin = useTensorflowModel(
    require('../../assets/models/RetinaFace/retinaface_mb025_320.tflite'),
    []
  );

  const [status, setStatus] = useState<string>('Center your face inside the square');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [detection, setDetection] = useState<FaceDetection | null>(null);
  const [previewUri, setPreviewUri] = useState<string | null>(null);

  const handleCaptureAndDetect = async () => {
    if (isProcessing) return;

    if (!cameraRef.current) {
      Alert.alert('Camera Error', 'Camera instance is not ready.');
      return;
    }

    if (retinaPlugin.state !== 'loaded' || !retinaPlugin.model) {
      Alert.alert('Model Loading', 'RetinaFace model is still initializing.');
      return;
    }

    try {
      setIsProcessing(true);
      setStatus('Capturing photo...');

      // Capture without skipping processing to ensure correct EXIF rotation
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        skipProcessing: false,
      });

      if (!photo || !photo.uri) {
        throw new Error('Failed to capture frame URI.');
      }

      setStatus('Processing image...');

      // Center-crop to 1:1 and build tensor
      const { tensor: inputTensor, croppedUri } = await preprocessPhotoForRetinaFace(
        photo.uri,
        photo.width,
        photo.height
      );

      // Freeze on the exact square that the model evaluated
      setPreviewUri(croppedUri);
      setStatus('Running face detection...');

      const outputs = await retinaPlugin.model.run([inputTensor.buffer as ArrayBuffer]);

      if (outputs && outputs.length >= 3) {
        const boxes = new Float32Array(outputs[0]);
        const scores = new Float32Array(outputs[1]);
        const landmarks = new Float32Array(outputs[2]);

        const result = parseRetinaFaceOutputs(boxes, scores, landmarks, 0.45);

        if (result) {
          setDetection(result);
          setStatus(`Face Detected! (${(result.score * 100).toFixed(1)}%)`);
        } else {
          setDetection(null);
          setStatus('No clear face detected. Please try again.');
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setStatus(`Error: ${message}`);
      Alert.alert('Processing Failed', message);
      setPreviewUri(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRetake = () => {
    setPreviewUri(null);
    setDetection(null);
    setStatus('Center your face inside the square');
  };

  const handleProceed = () => {
    if (!detection || !previewUri) return;
    Alert.alert(
      'Ready for ArcFace',
      'The 5 landmark points and canonical crop will now be sent to ArcFace.',
      [
        {
          text: 'View Result',
          onPress: () => onNavigate?.('Result'),
        },
        { text: 'OK' },
      ]
    );
  };

  if (!permission) {
    return <View style={styles.centerContainer}><Text style={styles.infoText}>Loading permissions...</Text></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.infoText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Direct 1:1 overlay mapping onto the square view
  const renderDetectionOverlay = () => {
    if (!detection) return null;

    const boxW = (detection.box.x2 - detection.box.x1) * SQUARE_SIZE;
    const boxH = (detection.box.y2 - detection.box.y1) * SQUARE_SIZE;
    const boxLeft = detection.box.x1 * SQUARE_SIZE;
    const boxTop = detection.box.y1 * SQUARE_SIZE;

    const mapLandmark = (pt: [number, number]) => ({
      left: pt[0] * SQUARE_SIZE - 5,
      top: pt[1] * SQUARE_SIZE - 5,
    });

    const leftEye = mapLandmark(detection.landmarks.leftEye);
    const rightEye = mapLandmark(detection.landmarks.rightEye);
    const nose = mapLandmark(detection.landmarks.nose);
    const mouthLeft = mapLandmark(detection.landmarks.mouthLeft);
    const mouthRight = mapLandmark(detection.landmarks.mouthRight);

    return (
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
        {/* Full Proportional Bounding Box */}
        <View
          style={[
            styles.detectionBox,
            {
              left: boxLeft,
              top: boxTop,
              width: boxW,
              height: boxH,
            },
          ]}
        />

        {/* 5 Facial Landmarks */}
        <View style={[styles.landmarkPoint, { left: leftEye.left, top: leftEye.top, backgroundColor: '#00FF00' }]} />
        <View style={[styles.landmarkPoint, { left: rightEye.left, top: rightEye.top, backgroundColor: '#00FF00' }]} />
        <View style={[styles.landmarkPoint, { left: nose.left, top: nose.top, backgroundColor: '#00E5FF' }]} />
        <View style={[styles.landmarkPoint, { left: mouthLeft.left, top: mouthLeft.top, backgroundColor: '#FF0055' }]} />
        <View style={[styles.landmarkPoint, { left: mouthRight.left, top: mouthRight.top, backgroundColor: '#FF0055' }]} />
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Centered Square Viewport */}
      <View style={styles.viewportContainer}>
        <View style={styles.squareFrame}>
          {previewUri ? (
            <Image source={{ uri: previewUri }} style={styles.previewImage} resizeMode="cover" />
          ) : (
            <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />
          )}

          {/* Landmarks & Bounding Box directly over the square */}
          {renderDetectionOverlay()}
        </View>

        {!previewUri && (
          <Text style={styles.guideText}>Keep your face well-lit and centered</Text>
        )}
      </View>

      {/* Bottom Controls */}
      <View style={styles.controlsContainer}>
        <Text style={styles.statusText}>{status}</Text>

        {!previewUri ? (
          <TouchableOpacity
            style={[styles.captureButton, isProcessing && styles.captureButtonDisabled]}
            onPress={handleCaptureAndDetect}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <ActivityIndicator color="#000" />
            ) : (
              <View style={styles.captureInnerCircle} />
            )}
          </TouchableOpacity>
        ) : (
          <View style={styles.actionRow}>
            <TouchableOpacity style={styles.retakeButton} onPress={handleRetake}>
              <Text style={styles.retakeButtonText}>Retake</Text>
            </TouchableOpacity>

            {detection && (
              <TouchableOpacity style={styles.confirmButton} onPress={handleProceed}>
                <Text style={styles.confirmButtonText}>Confirm Face</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
    justifyContent: 'space-between',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#121212',
    padding: 20,
  },
  infoText: {
    color: '#FFF',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 12,
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: '#FFF',
    fontWeight: 'bold',
  },
  viewportContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 20,
  },
  squareFrame: {
    width: SQUARE_SIZE,
    height: SQUARE_SIZE,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#333',
    backgroundColor: '#000',
    position: 'relative',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  guideText: {
    color: '#AAA',
    fontSize: 13,
    marginTop: 14,
  },
  detectionBox: {
    position: 'absolute',
    borderWidth: 2.5,
    borderColor: '#00FF00',
    borderRadius: 8,
    backgroundColor: 'rgba(0, 255, 0, 0.12)',
  },
  landmarkPoint: {
    position: 'absolute',
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 1.5,
    borderColor: '#FFF',
  },
  controlsContainer: {
    paddingBottom: 40,
    alignItems: 'center',
  },
  statusText: {
    color: '#FFF',
    fontSize: 14,
    marginBottom: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 16,
    overflow: 'hidden',
  },
  captureButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: '#FFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonDisabled: {
    opacity: 0.5,
  },
  captureInnerCircle: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFF',
  },
  actionRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    paddingHorizontal: 30,
    gap: 16,
  },
  retakeButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  retakeButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  confirmButton: {
    flex: 1,
    backgroundColor: '#00E676',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  confirmButtonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '700',
  },
});