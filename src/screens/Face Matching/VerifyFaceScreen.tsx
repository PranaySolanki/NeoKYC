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
import { useModels } from '../../assets/models/ModelContext';
import { parseRetinaFaceOutputs, FaceDetection } from '../RetinaFace+ArcFace/utils/RetinaFace/retinafaceParser';
import { preprocessPhotoForRetinaFace } from '../RetinaFace+ArcFace/utils/RetinaFace/imagePreprocess';
import { preprocessFaceForArcFace } from '../RetinaFace+ArcFace/utils/ArcFace/arcfacePreprocess';
import { matchLiveEmbedding } from '../../utils/storage';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SQUARE_SIZE = SCREEN_WIDTH * 0.85;

interface VerifyFaceScreenProps {
  onNavigate?: (screenName: string) => void;
}

export function VerifyFaceScreen({ onNavigate }: VerifyFaceScreenProps) {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();

  // Instantly accessible from root ModelProvider
  const { retinaModel, arcfaceModel } = useModels();

  const [status, setStatus] = useState<string>('Look into camera to verify identity');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [detection, setDetection] = useState<FaceDetection | null>(null);
  const [matchedName, setMatchedName] = useState<string | null>(null);
  const [similarityScore, setSimilarityScore] = useState<number>(0);

  const handleVerify = async () => {
    if (isProcessing || !cameraRef.current) return;

    try {
      setIsProcessing(true);
      setStatus('Capturing frame...');

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        skipProcessing: false,
      });

      if (!photo?.uri) throw new Error('Failed to capture frame from camera.');

      setStatus('Running RetinaFace detection...');
      const { tensor: retinaTensor, croppedUri } = await preprocessPhotoForRetinaFace(
        photo.uri,
        photo.width,
        photo.height
      );
      setPreviewUri(croppedUri);

      const retinaOutputs = await retinaModel!.run([retinaTensor.buffer as ArrayBuffer]);
      const result = parseRetinaFaceOutputs(
        new Float32Array(retinaOutputs[0]),
        new Float32Array(retinaOutputs[1]),
        new Float32Array(retinaOutputs[2]),
        0.45
      );

      if (!result) {
        setDetection(null);
        setPreviewUri(null);
        setStatus('No face detected. Please center your face.');
        Alert.alert('Detection Failed', 'No clear face found. Please try again.');
        return;
      }

      setDetection(result);

      setStatus('Extracting face signature with ArcFace...');
      const { tensor: arcTensor } = await preprocessFaceForArcFace(
        croppedUri,
        SQUARE_SIZE,
        result
      );

      

      const arcOutputs = await arcfaceModel!.run([arcTensor.buffer as ArrayBuffer]);
      const liveEmbedding = new Float32Array(arcOutputs[0]);

      setStatus('Comparing with registered database...');
      // 0.60 (60%) threshold prevents false positives between family members
      const match = await matchLiveEmbedding(liveEmbedding, 0.60);

      setSimilarityScore(match.similarity);
      if (match.matchedUser) {
        setMatchedName(match.matchedUser);
        setStatus(`Verified: ${match.matchedUser} (${(match.similarity * 100).toFixed(1)}%)`);
      } else {
        setMatchedName(null);
        setStatus(`Unrecognized Face (Best match: ${(match.similarity * 100).toFixed(1)}%)`);
      }
    } catch (err: unknown) {
      Alert.alert('Verification Error', err instanceof Error ? err.message : String(err));
      setPreviewUri(null);
      setDetection(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setPreviewUri(null);
    setDetection(null);
    setMatchedName(null);
    setSimilarityScore(0);
    setStatus('Look into camera to verify identity');
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

    const isMatch = Boolean(matchedName);
    const boxColor = isMatch ? '#00FF00' : '#FF453A';

    return (
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
        <View
          style={[
            styles.detectionBox,
            {
              left: boxLeft,
              top: boxTop,
              width: boxW,
              height: boxH,
              borderColor: boxColor,
              backgroundColor: isMatch ? 'rgba(0, 255, 0, 0.12)' : 'rgba(255, 69, 58, 0.12)',
            },
          ]}
        />
        <View style={[styles.landmarkPoint, { left: mapLandmark(detection.landmarks.leftEye).left, top: mapLandmark(detection.landmarks.leftEye).top, backgroundColor: boxColor }]} />
        <View style={[styles.landmarkPoint, { left: mapLandmark(detection.landmarks.rightEye).left, top: mapLandmark(detection.landmarks.rightEye).top, backgroundColor: boxColor }]} />
        <View style={[styles.landmarkPoint, { left: mapLandmark(detection.landmarks.nose).left, top: mapLandmark(detection.landmarks.nose).top, backgroundColor: '#00E5FF' }]} />
        <View style={[styles.landmarkPoint, { left: mapLandmark(detection.landmarks.mouthLeft).left, top: mapLandmark(detection.landmarks.mouthLeft).top, backgroundColor: '#FF0055' }]} />
        <View style={[styles.landmarkPoint, { left: mapLandmark(detection.landmarks.mouthRight).left, top: mapLandmark(detection.landmarks.mouthRight).top, backgroundColor: '#FF0055' }]} />
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
      {/* Header with Back Button */}
      <View style={styles.headerBar}>
        <TouchableOpacity
          style={styles.headerBackButton}
          onPress={() => onNavigate?.('FaceMatchingHub')}
          disabled={isProcessing}
        >
          <Text style={styles.headerBackText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Verify Face</Text>
        <View style={styles.headerSpacer} />
      </View>

      <View style={styles.viewport}>
        <View style={styles.squareFrame}>
          {previewUri ? (
            <Image source={{ uri: previewUri }} style={styles.previewImage} resizeMode="cover" />
          ) : (
            <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />
          )}
          {renderDetectionOverlay()}
        </View>

        {previewUri && (
          <View style={[styles.resultCard, matchedName ? styles.matchBorder : styles.failBorder]}>
            <Text style={styles.resultTitle}>
              {matchedName ? `Welcome, ${matchedName}!` : 'Unrecognized Person'}
            </Text>
            <Text style={styles.resultSubtitle}>
              Similarity: {(similarityScore * 100).toFixed(1)}% (Threshold: 60.0%)
            </Text>
          </View>
        )}

        <Text style={styles.statusBadge}>{status}</Text>
      </View>

      <View style={styles.controls}>
        {!previewUri ? (
          <TouchableOpacity
            style={[styles.captureBtn, isProcessing && styles.btnDisabled]}
            onPress={handleVerify}
            disabled={isProcessing}
          >
            {isProcessing ? <ActivityIndicator color="#000" /> : <View style={styles.innerCircle} />}
          </TouchableOpacity>
        ) : (
          <View style={styles.actionRow}>
            <TouchableOpacity style={styles.btnSecondary} onPress={handleReset}>
              <Text style={styles.btnSecondaryText}>Scan Again</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.btnDone} onPress={() => onNavigate?.('FaceMatchingHub')}>
              <Text style={styles.btnDoneText}>Done</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212', justifyContent: 'space-between', paddingVertical: 20, alignItems: 'center' },
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
    borderRadius: 8,
  },
  landmarkPoint: {
    position: 'absolute',
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 1.5,
    borderColor: '#FFF',
  },
  resultCard: {
    marginTop: 14,
    width: SQUARE_SIZE,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#1E1E1E',
    borderWidth: 1.5,
    alignItems: 'center',
  },
  matchBorder: { borderColor: '#00E676' },
  failBorder: { borderColor: '#FF453A' },
  resultTitle: { color: '#FFF', fontSize: 18, fontWeight: '700' },
  resultSubtitle: { color: '#AAA', fontSize: 13, marginTop: 4 },
  statusBadge: {
    color: '#DDD',
    fontSize: 13,
    marginTop: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 12,
    textAlign: 'center',
    maxWidth: SQUARE_SIZE,
  },
  controls: { alignItems: 'center', paddingBottom: 20 },
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
  actionRow: { flexDirection: 'row', width: SQUARE_SIZE, gap: 12 },
  btnSecondary: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  btnSecondaryText: { color: '#FFF', fontSize: 16, fontWeight: '600' },
  btnDone: {
    flex: 1,
    backgroundColor: '#0A84FF',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  btnDoneText: { color: '#FFF', fontSize: 16, fontWeight: '700' },
});

export default VerifyFaceScreen;