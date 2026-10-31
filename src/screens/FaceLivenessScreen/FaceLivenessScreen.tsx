import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  Image,
  ScrollView,
  Alert,
  TextInput,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Default backend URL pointing to your computer on LAN
const DEFAULT_BACKEND_URL = 'http://192.168.0.102:5000';

export interface FaceLivenessScreenProps {
  onNavigate?: (screenName: string) => void;
}

interface PredictionData {
  face_index: number;
  real_probability: number;
  fake_probability: number;
  predicted_label: 'real' | 'fake';
  threshold_used: number;
  is_above_threshold: boolean;
}

interface AnalysisResult {
  status: 'success' | 'no_face_detected' | 'model_unavailable' | 'error';
  model?: {
    name: string;
    version: string;
    checkpoint_loaded: boolean;
  };
  faces_detected?: number;
  predictions?: PredictionData[];
  processing_time_ms?: number;
  message?: string;
}

export const FaceLivenessScreen: React.FC<FaceLivenessScreenProps> = ({ onNavigate }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  const [facing, setFacing] = useState<'front' | 'back'>('front');
  const [backendUrl, setBackendUrl] = useState<string>(DEFAULT_BACKEND_URL);
  const [isEditingUrl, setIsEditingUrl] = useState<boolean>(false);
  const [capturedImageUri, setCapturedImageUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Send image to backend for F3-Net deepfake prediction
  const analyzeImage = async (imageUri: string) => {
    setCapturedImageUri(imageUri);
    setIsAnalyzing(true);
    setErrorMessage(null);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      const filename = imageUri.split('/').pop() || 'selfie.jpg';
      const cleanFilename = filename.includes('?') ? filename.split('?')[0] : filename;
      const finalName = cleanFilename.match(/\.(jpg|jpeg|png)$/i) ? cleanFilename : 'selfie.jpg';
      const mimeType = finalName.endsWith('.png') ? 'image/png' : 'image/jpeg';

      formData.append('file', {
        uri: imageUri,
        name: finalName,
        type: mimeType,
      } as any);

      // Clean and sanitize base URL (strip accidental endpoint paths or trailing slashes)
      let baseUrl = backendUrl.trim().replace(/\/+$/, '');
      baseUrl = baseUrl.replace(/\/api\/deepfake\/predict-image\/?$/i, '');
      baseUrl = baseUrl.replace(/\/api\/ekyc\/verify\/?$/i, '');
      baseUrl = baseUrl.replace(/\/api\/?$/i, '');

      const endpoint = `${baseUrl}/api/deepfake/predict-image`;

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned HTTP ${response.status}: ${errorText}`);
      }

      const data: AnalysisResult = await response.json();
      setAnalysisResult(data);
    } catch (err: any) {
      console.error('Inference error:', err);
      setErrorMessage(
        err.message || 'Failed to connect to backend server. Make sure it is running on the same Wi-Fi.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Capture photo from live camera
  const handleCapture = async () => {
    if (!cameraRef.current || isAnalyzing) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.85,
        skipProcessing: true,
      });

      if (photo?.uri) {
        await analyzeImage(photo.uri);
      }
    } catch (err: any) {
      Alert.alert('Capture Error', err.message || 'Failed to capture photo.');
    }
  };

  // Pick an existing selfie or deepfake from gallery
  const handlePickFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.85,
        allowsEditing: false,
      });

      if (!result.canceled && result.assets && result.assets[0]?.uri) {
        await analyzeImage(result.assets[0].uri);
      }
    } catch (err: any) {
      Alert.alert('Gallery Error', err.message || 'Failed to pick image.');
    }
  };

  const handleReset = () => {
    setCapturedImageUri(null);
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  // Permission handling
  if (!permission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#38BDF8" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permissionTitle}>Camera Permission Required</Text>
        <Text style={styles.permissionDesc}>
          NeoKYC needs camera access to capture your selfie and analyze it with F³-Net.
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Top Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>F³-Net Deepfake Detection</Text>
        <TouchableOpacity
          style={styles.urlPill}
          onPress={() => setIsEditingUrl(!isEditingUrl)}
        >
          <Text style={styles.urlPillText}>
            ⚙ API: {backendUrl.replace('http://', '')}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Optional IP Editor */}
      {isEditingUrl && (
        <View style={styles.urlEditor}>
          <Text style={styles.urlEditorLabel}>Backend API URL:</Text>
          <TextInput
            style={styles.urlInput}
            value={backendUrl}
            onChangeText={setBackendUrl}
            autoCapitalize="none"
            autoCorrect={false}
          />
          <TouchableOpacity
            style={styles.saveUrlButton}
            onPress={() => setIsEditingUrl(false)}
          >
            <Text style={styles.saveUrlButtonText}>Save</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Main Content: Camera OR Analysis Results */}
      {!capturedImageUri ? (
        <View style={styles.cameraWrapper}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
          >
            {/* Oval Face Guide Overlay */}
            <View style={styles.overlay}>
              <View style={styles.ovalGuide} />
              <Text style={styles.guideText}>Align your face inside the oval</Text>
            </View>

            {/* Camera Controls */}
            <View style={styles.bottomControls}>
              <TouchableOpacity
                style={styles.secondaryCircleButton}
                onPress={handlePickFromGallery}
              >
                <Text style={styles.secondaryButtonIcon}>🖼</Text>
                <Text style={styles.secondaryButtonLabel}>Gallery</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.shutterButton}
                onPress={handleCapture}
                disabled={isAnalyzing}
              >
                <View style={styles.shutterInner} />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryCircleButton}
                onPress={() => setFacing(facing === 'front' ? 'back' : 'front')}
              >
                <Text style={styles.secondaryButtonIcon}>🔄</Text>
                <Text style={styles.secondaryButtonLabel}>Flip</Text>
              </TouchableOpacity>
            </View>
          </CameraView>
        </View>
      ) : (
        /* Results View */
        <ScrollView style={styles.resultScrollView} contentContainerStyle={styles.resultContainer}>
          <Image source={{ uri: capturedImageUri }} style={styles.capturedPreview} />

          {isAnalyzing ? (
            <View style={styles.analyzingCard}>
              <ActivityIndicator size="large" color="#38BDF8" style={{ marginBottom: 12 }} />
              <Text style={styles.analyzingTitle}>Running F³-Net Frequency Analysis...</Text>
              <Text style={styles.analyzingSubtitle}>
                Extracting frequency-aware clues (FAD + LFS dual-streams)
              </Text>
            </View>
          ) : errorMessage ? (
            <View style={styles.errorCard}>
              <Text style={styles.errorIcon}>⚠️</Text>
              <Text style={styles.errorTitle}>Analysis Failed</Text>
              <Text style={styles.errorText}>{errorMessage}</Text>
              <TouchableOpacity style={styles.retryButton} onPress={handleReset}>
                <Text style={styles.retryButtonText}>Try Again</Text>
              </TouchableOpacity>
            </View>
          ) : analysisResult ? (
            <View style={styles.resultCard}>
              {analysisResult.status === 'no_face_detected' ? (
                <View style={styles.statusBoxWarning}>
                  <Text style={styles.statusEmoji}>⚠️</Text>
                  <Text style={styles.statusTitle}>No Face Detected</Text>
                  <Text style={styles.statusDesc}>
                    YuNet could not find a clearly visible face. Please ensure good lighting and face the camera directly.
                  </Text>
                </View>
              ) : analysisResult.predictions && analysisResult.predictions.length > 0 ? (
                (() => {
                  const pred = analysisResult.predictions[0];
                  const isReal = pred.predicted_label === 'real';
                  const realPct = (pred.real_probability * 100).toFixed(1);
                  const fakePct = (pred.fake_probability * 100).toFixed(1);

                  return (
                    <>
                      {/* Big Verdict Badge */}
                      <View style={[styles.verdictBadge, isReal ? styles.verdictReal : styles.verdictFake]}>
                        <Text style={styles.verdictIcon}>{isReal ? '🛡️' : '🚨'}</Text>
                        <Text style={styles.verdictTitle}>
                          {isReal ? 'VERIFIED LIVE PERSON' : 'SPOOF / DEEPFAKE DETECTED'}
                        </Text>
                        <Text style={styles.verdictScore}>
                          {isReal ? `${realPct}% Authentic Live` : `${fakePct}% Spoof / Manipulated`}
                        </Text>
                      </View>

                      {/* Probability Bars */}
                      <View style={styles.metricsContainer}>
                        <Text style={styles.metricsHeader}>F³-Net Confidence Breakdown</Text>
                        
                        <View style={styles.metricRow}>
                          <Text style={styles.metricLabel}>Real Probability</Text>
                          <Text style={[styles.metricVal, { color: '#22C55E' }]}>{realPct}%</Text>
                        </View>
                        <View style={styles.progressBarBg}>
                          <View style={[styles.progressBarFill, { width: `${realPct}%` as any, backgroundColor: '#22C55E' }]} />
                        </View>

                        <View style={[styles.metricRow, { marginTop: 12 }]}>
                          <Text style={styles.metricLabel}>Deepfake Probability</Text>
                          <Text style={[styles.metricVal, { color: '#EF4444' }]}>{fakePct}%</Text>
                        </View>
                        <View style={styles.progressBarBg}>
                          <View style={[styles.progressBarFill, { width: `${fakePct}%` as any, backgroundColor: '#EF4444' }]} />
                        </View>
                      </View>

                      {/* Technical Audit Specs */}
                      <View style={styles.specsCard}>
                        <Text style={styles.specsTitle}>Audit Details</Text>
                        <Text style={styles.specLine}>• Model: <Text style={styles.specHighlight}>F³-Net Dual-Stream (ECCV 2020)</Text></Text>
                        <Text style={styles.specLine}>• Streams: <Text style={styles.specHighlight}>FAD + LFS Collaborative Attention</Text></Text>
                        <Text style={styles.specLine}>• Face Localization: <Text style={styles.specHighlight}>OpenCV YuNet DNN (1.3× crop)</Text></Text>
                        <Text style={styles.specLine}>• Latency: <Text style={styles.specHighlight}>{analysisResult.processing_time_ms} ms</Text></Text>
                        <Text style={styles.specLine}>• Threshold: <Text style={styles.specHighlight}>{pred.threshold_used}</Text></Text>
                      </View>
                    </>
                  );
                })()
              ) : (
                <Text style={styles.statusDesc}>{analysisResult.message || 'Completed'}</Text>
              )}

              {/* Action Buttons */}
              <TouchableOpacity style={styles.primaryButton} onPress={handleReset}>
                <Text style={styles.primaryButtonText}>Scan Another Face</Text>
              </TouchableOpacity>
            </View>
          ) : null}
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  centerContainer: {
    flex: 1,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  header: {
    paddingTop: 16,
    paddingBottom: 12,
    paddingHorizontal: 20,
    backgroundColor: '#1E293B',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#38BDF8',
    fontSize: 17,
    fontWeight: '700',
  },
  urlPill: {
    backgroundColor: '#334155',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
  },
  urlPillText: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '600',
  },
  urlEditor: {
    backgroundColor: '#1E293B',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  urlEditorLabel: {
    color: '#94A3B8',
    fontSize: 12,
    marginBottom: 4,
  },
  urlInput: {
    backgroundColor: '#0F172A',
    color: '#F8FAFC',
    padding: 8,
    borderRadius: 6,
    fontSize: 14,
  },
  saveUrlButton: {
    backgroundColor: '#38BDF8',
    padding: 8,
    borderRadius: 6,
    alignItems: 'center',
    marginTop: 8,
  },
  saveUrlButtonText: {
    color: '#0F172A',
    fontWeight: '700',
  },
  cameraWrapper: {
    flex: 1,
  },
  camera: {
    flex: 1,
    justifyContent: 'space-between',
  },
  overlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ovalGuide: {
    width: SCREEN_WIDTH * 0.72,
    height: SCREEN_HEIGHT * 0.44,
    borderRadius: SCREEN_WIDTH * 0.36,
    borderWidth: 3,
    borderColor: '#38BDF8',
    backgroundColor: 'transparent',
    borderStyle: 'dashed',
  },
  guideText: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '600',
    marginTop: 16,
    backgroundColor: 'rgba(15, 23, 42, 0.75)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    overflow: 'hidden',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 36,
    paddingHorizontal: 24,
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
  },
  shutterButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: '#38BDF8',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },
  shutterInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#38BDF8',
  },
  secondaryCircleButton: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 8,
  },
  secondaryButtonIcon: {
    fontSize: 24,
  },
  secondaryButtonLabel: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '600',
    marginTop: 4,
  },
  resultScrollView: {
    flex: 1,
  },
  resultContainer: {
    padding: 20,
    alignItems: 'center',
  },
  capturedPreview: {
    width: SCREEN_WIDTH * 0.5,
    height: SCREEN_WIDTH * 0.62,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#334155',
    marginBottom: 20,
  },
  analyzingCard: {
    backgroundColor: '#1E293B',
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
    width: '100%',
  },
  analyzingTitle: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 4,
  },
  analyzingSubtitle: {
    color: '#94A3B8',
    fontSize: 13,
    textAlign: 'center',
  },
  errorCard: {
    backgroundColor: '#1E293B',
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    width: '100%',
    borderColor: '#EF4444',
    borderWidth: 1,
  },
  errorIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  errorTitle: {
    color: '#EF4444',
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 6,
  },
  errorText: {
    color: '#94A3B8',
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#38BDF8',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#0F172A',
    fontWeight: '700',
  },
  resultCard: {
    width: '100%',
    alignItems: 'center',
  },
  verdictBadge: {
    width: '100%',
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  verdictReal: {
    backgroundColor: 'rgba(34, 197, 94, 0.15)',
    borderColor: '#22C55E',
    borderWidth: 2,
  },
  verdictFake: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderColor: '#EF4444',
    borderWidth: 2,
  },
  verdictIcon: {
    fontSize: 40,
    marginBottom: 8,
  },
  verdictTitle: {
    color: '#F8FAFC',
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  verdictScore: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '700',
  },
  metricsContainer: {
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 14,
    width: '100%',
    marginBottom: 16,
  },
  metricsHeader: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 12,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  metricLabel: {
    color: '#94A3B8',
    fontSize: 13,
  },
  metricVal: {
    fontSize: 14,
    fontWeight: '700',
  },
  progressBarBg: {
    height: 8,
    backgroundColor: '#334155',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  specsCard: {
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 14,
    width: '100%',
    marginBottom: 20,
  },
  specsTitle: {
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  specLine: {
    color: '#94A3B8',
    fontSize: 13,
    marginBottom: 4,
  },
  specHighlight: {
    color: '#F8FAFC',
    fontWeight: '600',
  },
  statusBoxWarning: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    borderColor: '#F59E0B',
    borderWidth: 1,
    padding: 16,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 16,
  },
  statusEmoji: {
    fontSize: 32,
    marginBottom: 8,
  },
  statusTitle: {
    color: '#F59E0B',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 4,
  },
  statusDesc: {
    color: '#94A3B8',
    fontSize: 13,
    textAlign: 'center',
  },
  primaryButton: {
    backgroundColor: '#38BDF8',
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#0F172A',
    fontSize: 16,
    fontWeight: '700',
  },
  permissionTitle: {
    color: '#F8FAFC',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  permissionDesc: {
    color: '#94A3B8',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 24,
  },
});
