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
  ScrollView,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Crypto from 'expo-crypto';
import { useModels } from '../../assets/models/ModelContext';
import { parseRetinaFaceOutputs, FaceDetection } from '../RetinaFace+ArcFace/utils/RetinaFace/retinafaceParser';
import { preprocessPhotoForRetinaFace } from '../RetinaFace+ArcFace/utils/RetinaFace/imagePreprocess';
import { preprocessFaceForArcFace } from '../RetinaFace+ArcFace/utils/ArcFace/arcfacePreprocess';
import { supabase } from '../../services/supabase';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SQUARE_SIZE = SCREEN_WIDTH * 0.85;

export interface RegisterFaceScreenProps {
  onNavigate?: (screenName: string, params?: Record<string, unknown>) => void;
  applicantData?: {
    name: string | null;
    dob: string | null;
    gender: string | null;
    id_number: string | null;
  };
}

export function RegisterFaceScreen({ onNavigate, applicantData }: RegisterFaceScreenProps) {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();

  const { retinaModel, arcfaceModel } = useModels();

  const [status, setStatus] = useState<string>('Center your face in the square');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [detection, setDetection] = useState<FaceDetection | null>(null);
  const [previewUri, setPreviewUri] = useState<string | null>(null);

  const handleCaptureAndDetect = async () => {
    if (isProcessing || !cameraRef.current) return;

    if (!retinaModel) {
      Alert.alert('Initializing', 'RetinaFace model is not ready.');
      return;
    }

    try {
      setIsProcessing(true);
      setStatus('Capturing photo...');

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        skipProcessing: false,
      });

      if (!photo?.uri) throw new Error('Failed to capture frame from camera.');

      setStatus('Detecting face with RetinaFace...');
      const { tensor: inputTensor, croppedUri } = await preprocessPhotoForRetinaFace(
        photo.uri,
        photo.width,
        photo.height
      );

      setPreviewUri(croppedUri);

      const outputs = await retinaModel.run([inputTensor.buffer as ArrayBuffer]);
      const result = parseRetinaFaceOutputs(
        new Float32Array(outputs[0]),
        new Float32Array(outputs[1]),
        new Float32Array(outputs[2]),
        0.45
      );

      if (result) {
        setDetection(result);
        setStatus(`Face detected (${(result.score * 100).toFixed(1)}%)! Ready to enroll.`);
      } else {
        setDetection(null);
        setPreviewUri(null);
        setStatus('No face detected. Please ensure good lighting and try again.');
        Alert.alert('Detection Failed', 'No clear face detected in the frame. Please try again.');
      }
    } catch (err: unknown) {
      Alert.alert('Detection Error', err instanceof Error ? err.message : String(err));
      setPreviewUri(null);
      setDetection(null);
      setStatus('Error occurred. Try capturing again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatToISODate = (dateStr: string | null | undefined): string | null => {
  if (!dateStr) return null;

  const cleaned = dateStr.trim();

  // Matches DD/MM/YYYY or DD-MM-YYYY
  const dmyMatch = cleaned.match(/^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$/);
  if (dmyMatch) {
    const day = dmyMatch[1].padStart(2, '0');
    const month = dmyMatch[2].padStart(2, '0');
    const year = dmyMatch[3];
    return `${year}-${month}-${day}`; // Converts 15/02/2003 -> 2003-02-15
  }

  // Matches single 4-digit Year of Birth (e.g., 2003 -> 2003-01-01)
  const yobMatch = cleaned.match(/^(\d{4})$/);
  if (yobMatch) {
    return `${yobMatch[1]}-01-01`;
  }

  // If already YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(cleaned)) {
    return cleaned;
  }

  return null;
};

  const handleRegister = async () => {
    if (!detection || !previewUri) return;

    if (!arcfaceModel) {
      Alert.alert('Initializing', 'ArcFace model is not ready.');
      return;
    }

    try {
      setIsProcessing(true);
      setStatus('Extracting 512-D ArcFace signature...');

      const { tensor: arcTensor } = await preprocessFaceForArcFace(
        previewUri,
        SQUARE_SIZE,
        detection
      );

      const outputs = await arcfaceModel.run([arcTensor.buffer as ArrayBuffer]);
      const embedding = new Float32Array(outputs[0]);

      if (embedding.length !== 512) {
        throw new Error(`Invalid embedding size: ${embedding.length} (expected 512)`);
      }

      setStatus('Computing secure hash and saving...');

      const rawId = applicantData?.id_number || 'UNKNOWN_ID';
      const idSha256 = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        rawId
      );

      const timestamp = new Date().toISOString();
      const { error } = await supabase.from('person_entries').insert([
        {
          //date_of_birth: applicantData?.dob || null,
          full_name:applicantData?.name||null,
          date_of_birth: formatToISODate(applicantData?.dob),
          gender: applicantData?.gender || null,
          aadhaar_number: idSha256,
          face_coordinates: Array.from(embedding),
          created_at: timestamp,
          last_referenced_at: timestamp,
        },
      ]);

      if (error) {
        throw new Error(error.message);
      }

      Alert.alert('Success!', 'User registered successfully in database.', [
        {
          text: 'OK',
          onPress: () => onNavigate?.('HomeDash'),
        },
      ]);
    } catch (err: unknown) {
      Alert.alert('Registration Error', err instanceof Error ? err.message : String(err));
      setStatus('Failed to complete enrollment.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRetake = () => {
    setPreviewUri(null);
    setDetection(null);
    setStatus('Center your face in the square');
  };

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
        <View
          style={[
            styles.detectionBox,
            { left: boxLeft, top: boxTop, width: boxW, height: boxH },
          ]}
        />
        <View style={[styles.landmarkPoint, { left: leftEye.left, top: leftEye.top, backgroundColor: '#00FF00' }]} />
        <View style={[styles.landmarkPoint, { left: rightEye.left, top: rightEye.top, backgroundColor: '#00FF00' }]} />
        <View style={[styles.landmarkPoint, { left: nose.left, top: nose.top, backgroundColor: '#00E5FF' }]} />
        <View style={[styles.landmarkPoint, { left: mouthLeft.left, top: mouthLeft.top, backgroundColor: '#FF0055' }]} />
        <View style={[styles.landmarkPoint, { left: mouthRight.left, top: mouthRight.top, backgroundColor: '#FF0055' }]} />
      </View>
    );
  };

  if (!permission?.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.infoText}>Camera permission is required.</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnPrimaryText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        {/* Top Header with Back Button */}
        <View style={styles.headerBar}>
          <TouchableOpacity
            style={styles.headerBackButton}
            onPress={() => onNavigate?.('DocumentScan')}
            disabled={isProcessing}
          >
            <Text style={styles.headerBackText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Register Face</Text>
          <View style={styles.headerSpacer} />
        </View>

        {/* Viewfinder / Square Frame */}
        <View style={styles.viewport}>
          <View style={styles.squareFrame}>
            {previewUri ? (
              <Image source={{ uri: previewUri }} style={styles.previewImage} resizeMode="cover" />
            ) : (
              <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />
            )}
            {renderDetectionOverlay()}
          </View>
          <Text style={styles.statusBadge}>{status}</Text>
        </View>

        {/* Controls */}
        {previewUri ? (
          <View style={styles.inputSection}>
            <View style={styles.actionRow}>
              <TouchableOpacity
                style={styles.btnSecondary}
                onPress={handleRetake}
                disabled={isProcessing}
              >
                <Text style={styles.btnSecondaryText}>Retake</Text>
              </TouchableOpacity>

              {detection && (
                <TouchableOpacity
                  style={[styles.btnConfirm, isProcessing && styles.btnDisabled]}
                  onPress={handleRegister}
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <ActivityIndicator color="#000" />
                  ) : (
                    <Text style={styles.btnConfirmText}>Enroll User</Text>
                  )}
                </TouchableOpacity>
              )}
            </View>
          </View>
        ) : (
          <View style={styles.controls}>
            <TouchableOpacity
              style={[styles.captureBtn, isProcessing && styles.btnDisabled]}
              onPress={handleCaptureAndDetect}
              disabled={isProcessing}
            >
              {isProcessing ? <ActivityIndicator color="#000" /> : <View style={styles.innerCircle} />}
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212' },
  scrollContent: { flexGrow: 1, justifyContent: 'space-between', paddingVertical: 20, alignItems: 'center' },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#121212' },
  infoText: { color: '#FFF', fontSize: 16, marginBottom: 12 },
  headerBar: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 10,
    marginBottom: 16,
  },
  headerBackButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  headerBackText: { color: '#0A84FF', fontSize: 15, fontWeight: '600' },
  headerSpacer: { width: 60 },
  title: { color: '#FFF', fontSize: 20, fontWeight: '700' },
  viewport: { alignItems: 'center' },
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
  previewImage: { width: '100%', height: '100%' },
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
  statusBadge: {
    color: '#DDD',
    fontSize: 13,
    marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 12,
    textAlign: 'center',
    maxWidth: SQUARE_SIZE,
  },
  inputSection: { width: SQUARE_SIZE, marginTop: 16, alignItems: 'center' },
  actionRow: { flexDirection: 'row', width: '100%', gap: 12 },
  btnSecondary: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  btnSecondaryText: { color: '#FFF', fontSize: 16, fontWeight: '600' },
  btnConfirm: {
    flex: 1,
    backgroundColor: '#00E676',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  btnConfirmText: { color: '#000', fontSize: 16, fontWeight: '700' },
  controls: { alignItems: 'center', marginTop: 20 },
  captureBtn: {
    width: 74,
    height: 74,
    borderRadius: 37,
    borderWidth: 4,
    borderColor: '#FFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  innerCircle: { width: 58, height: 58, borderRadius: 29, backgroundColor: '#FFF' },
  btnDisabled: { opacity: 0.5 },
  btnPrimary: { backgroundColor: '#0A84FF', paddingHorizontal: 20, paddingVertical: 10, borderRadius: 8 },
  btnPrimaryText: { color: '#FFF', fontWeight: 'bold' },
});

export default RegisterFaceScreen;