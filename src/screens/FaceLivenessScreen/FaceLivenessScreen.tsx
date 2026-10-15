import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  rppgService,
  RppgResult,
  LivenessChallenge,
  ChallengeType,
} from '../../services/rppgService';

type ScanPhase = 'IDLE' | 'PREPARING' | 'RECORDING' | 'ANALYZING' | 'RESULT';

interface FaceLivenessScreenProps {
  onNavigate?: (screenName: string) => void;
  onComplete?: (result: RppgResult) => void;
}

const PREP_COUNTDOWN_SECONDS = 3;
const CHALLENGE_DURATION_SECONDS = 5;
const NUM_CHALLENGES = 6;
const TOTAL_RECORDING_SECONDS = CHALLENGE_DURATION_SECONDS * NUM_CHALLENGES; // 30 seconds

interface ActionDefinition {
  type: ChallengeType;
  instruction: string;
  icon: string;
  tip: string;
}

const AVAILABLE_ACTIONS: ActionDefinition[] = [
  {
    type: 'BLINK',
    instruction: 'Blink twice naturally',
    icon: '👁️',
    tip: 'Close both eyes briefly and open them',
  },
  {
    type: 'SMILE',
    instruction: 'Smile naturally',
    icon: '😊',
    tip: 'Show a gentle natural smile',
  },
  {
    type: 'TURN_LEFT',
    instruction: 'Turn your head slightly left',
    icon: '⬅️',
    tip: 'Rotate head about 15° to your left',
  },
  {
    type: 'TURN_RIGHT',
    instruction: 'Turn your head slightly right',
    icon: '➡️',
    tip: 'Rotate head about 15° to your right',
  },
  {
    type: 'TILT_UP',
    instruction: 'Tilt your head slightly up',
    icon: '⬆️',
    tip: 'Gently raise your chin upward',
  },
];

export const FaceLivenessScreen: React.FC<FaceLivenessScreenProps> = ({
  onNavigate,
  onComplete,
}) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [phase, setPhase] = useState<ScanPhase>('IDLE');
  const [prepCountdown, setPrepCountdown] = useState<number>(PREP_COUNTDOWN_SECONDS);
  const [recordCountdown, setRecordCountdown] = useState<number>(TOTAL_RECORDING_SECONDS);
  const [activeChallengeIndex, setActiveChallengeIndex] = useState<number>(0);
  const [challenges, setChallenges] = useState<LivenessChallenge[]>([]);
  const [result, setResult] = useState<RppgResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analysisStatusText, setAnalysisStatusText] = useState<string>('Initializing analysis...');

  const cameraRef = useRef<CameraView>(null);
  const isRecordingRef = useRef<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const analysisTickerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (analysisTickerRef.current) clearInterval(analysisTickerRef.current);
    };
  }, []);

  /**
   * Generates a randomized sequence of challenges without consecutive duplicates.
   */
  const generateChallenges = (): LivenessChallenge[] => {
    // Shuffle available dynamic actions
    const shuffled = [...AVAILABLE_ACTIONS].sort(() => Math.random() - 0.5);
    const selectedMidActions = shuffled.slice(0, 4);

    const fullSequence: ActionDefinition[] = [
      {
        type: 'LOOK_CENTER',
        instruction: 'Look directly at the camera',
        icon: '🎯',
        tip: 'Center your face in the oval and hold still',
      },
      ...selectedMidActions,
      {
        type: 'LOOK_CENTER',
        instruction: 'Return to center and look at camera',
        icon: '🎯',
        tip: 'Hold still for final pulse measurement',
      },
    ];

    return fullSequence.map((action, idx) => ({
      id: `challenge_${idx + 1}_${action.type.toLowerCase()}`,
      type: action.type,
      instruction: action.instruction,
      startTimeSec: idx * CHALLENGE_DURATION_SECONDS,
      endTimeSec: (idx + 1) * CHALLENGE_DURATION_SECONDS,
    }));
  };

  // Handle permission request
  if (!permission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#38BDF8" />
        <Text style={styles.loadingText}>Initializing camera permissions...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <View style={styles.card}>
          <Text style={styles.warningTitle}>Camera Access Required</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
            <Text style={styles.primaryButtonText}>Enable Camera</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const startScanFlow = () => {
    if (phase !== 'IDLE') return;
    if (!cameraRef.current) {
      setErrorMessage('Camera preview is not ready yet. Please try again.');
      return;
    }

    setErrorMessage(null);
    setResult(null);

    // 1. Generate new randomized challenge sequence
    const newChallenges = generateChallenges();
    setChallenges(newChallenges);
    setActiveChallengeIndex(0);

    // 2. Start preparation countdown (3s)
    setPhase('PREPARING');
    setPrepCountdown(PREP_COUNTDOWN_SECONDS);

    let prepRemaining = PREP_COUNTDOWN_SECONDS;
    timerRef.current = setInterval(() => {
      prepRemaining -= 1;
      setPrepCountdown(prepRemaining);
      if (prepRemaining <= 0) {
        if (timerRef.current) clearInterval(timerRef.current);
        startRecording(newChallenges);
      }
    }, 1000);
  };

  const startRecording = async (activeChallenges: LivenessChallenge[]) => {
    setPhase('RECORDING');
    isRecordingRef.current = true;
    setRecordCountdown(TOTAL_RECORDING_SECONDS);

    let elapsedSeconds = 0;
    timerRef.current = setInterval(() => {
      elapsedSeconds += 1;
      const remaining = TOTAL_RECORDING_SECONDS - elapsedSeconds;
      setRecordCountdown(Math.max(0, remaining));

      // Update current challenge step based on elapsed time
      const curIndex = Math.min(
        NUM_CHALLENGES - 1,
        Math.floor(elapsedSeconds / CHALLENGE_DURATION_SECONDS)
      );
      setActiveChallengeIndex(curIndex);

      if (remaining <= 0) {
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 1000);

    let recordedUri: string | null = null;
    try {
      const recordPromise = cameraRef.current!.recordAsync({
        maxDuration: TOTAL_RECORDING_SECONDS,
      });

      const video = await recordPromise;
      if (video?.uri) {
        recordedUri = video.uri;
      }
    } catch (err: any) {
      setErrorMessage(err?.message || 'Camera recording interrupted.');
      setPhase('IDLE');
      if (timerRef.current) clearInterval(timerRef.current);
      isRecordingRef.current = false;
      return;
    }

    if (timerRef.current) clearInterval(timerRef.current);
    isRecordingRef.current = false;

    if (!recordedUri) {
      setErrorMessage('Video recording produced an empty output. Please retry.');
      setPhase('IDLE');
      return;
    }

    // Phase: Analyzing on-device
    setPhase('ANALYZING');
    setAnalysisStatusText('Decoding video frames at 15 FPS...');

    let tickerStep = 0;
    analysisTickerRef.current = setInterval(() => {
      tickerStep++;
      if (tickerStep === 1) {
        setAnalysisStatusText('Running MediaPipe Face Mesh tracking...');
      } else if (tickerStep === 2) {
        setAnalysisStatusText('Verifying randomized challenge actions...');
      } else if (tickerStep === 3) {
        setAnalysisStatusText('Computing subcutaneous rPPG pulse spectrum...');
      } else if (tickerStep >= 4) {
        setAnalysisStatusText('Finalizing zero-trust multi-factor assessment...');
      }
    }, 2500);

    try {
      const analysisResult = await rppgService.processVideo(recordedUri, activeChallenges);
      if (analysisTickerRef.current) clearInterval(analysisTickerRef.current);

      setResult(analysisResult);
      setPhase('RESULT');

      if (onComplete) {
        onComplete(analysisResult);
      }
    } catch (err: any) {
      if (analysisTickerRef.current) clearInterval(analysisTickerRef.current);
      setErrorMessage('Native rPPG processing failed: ' + (err?.message || 'Unknown error'));
      setPhase('IDLE');
    } finally {
      if (recordedUri) {
        await rppgService.deleteVideoFile(recordedUri);
      }
    }
  };

  const resetScan = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (analysisTickerRef.current) clearInterval(analysisTickerRef.current);
    isRecordingRef.current = false;
    setResult(null);
    setErrorMessage(null);
    setPhase('IDLE');
    setPrepCountdown(PREP_COUNTDOWN_SECONDS);
    setRecordCountdown(TOTAL_RECORDING_SECONDS);
    setActiveChallengeIndex(0);
  };

  const currentChallenge = challenges[activeChallengeIndex] || {
    instruction: 'Keep your face in the oval',
    type: 'LOOK_CENTER',
  };

  const getActionIcon = (type: string) => {
    switch (type) {
      case 'BLINK': return '👁️';
      case 'SMILE': return '😊';
      case 'TURN_LEFT': return '⬅️';
      case 'TURN_RIGHT': return '➡️';
      case 'TILT_UP': return '⬆️';
      case 'LOOK_CENTER': return '🎯';
      default: return '👤';
    }
  };

  return (
    <View style={styles.container}>
      {/* Top Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Liveness Check</Text>
        <Text style={styles.headerSubtitle}>
          Zero-Trust On-Device Action Verification + Subcutaneous Pulse
        </Text>
      </View>

      {/* Main Content Area */}
      {phase !== 'RESULT' ? (
        <View style={styles.cameraWrapper}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="front"
            mode="video"
            mute={true}
          >
            {/* Darkened Mask with Oval Cutout */}
            <View style={styles.overlayContainer}>
              <View
                style={[
                  styles.ovalGuide,
                  phase === 'RECORDING' && styles.ovalGuideRecording,
                  phase === 'PREPARING' && styles.ovalGuidePreparing,
                ]}
              >
                {phase === 'PREPARING' && (
                  <View style={styles.prepOverlay}>
                    <Text style={styles.prepNumber}>{prepCountdown}</Text>
                    <Text style={styles.prepText}>GET READY</Text>
                  </View>
                )}

                {phase === 'RECORDING' && (
                  <View style={styles.recordingIndicator}>
                    <View style={styles.recordingDot} />
                    <Text style={styles.recordingText}>
                      REC {recordCountdown}s
                    </Text>
                  </View>
                )}
              </View>
            </View>
          </CameraView>

          {/* Interactive Instruction / Challenge HUD */}
          <View style={styles.instructionsContainer}>
            {phase === 'IDLE' && (
              <>
                <Text style={styles.instructionHeading}>Randomized Liveness Challenge</Text>
                <Text style={styles.instructionDetail}>
                  You will be prompted with short, guided actions (e.g. smile, turn head, blink)
                  followed by resting pulse capture. Total session duration: 30 seconds.
                </Text>
                {errorMessage && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorText}>{errorMessage}</Text>
                  </View>
                )}
                <TouchableOpacity style={styles.primaryButton} onPress={startScanFlow}>
                  <Text style={styles.primaryButtonText}>Start Guided Liveness Scan</Text>
                </TouchableOpacity>
              </>
            )}

            {phase === 'PREPARING' && (
              <View style={styles.centerHUD}>
                <Text style={styles.instructionHeading}>Position Face in the Oval</Text>
                <Text style={styles.instructionDetail}>
                  Ensure good lighting. Recording starts in {prepCountdown} seconds...
                </Text>
              </View>
            )}

            {phase === 'RECORDING' && (
              <View style={styles.recordingHUD}>
                {/* Step Indicator */}
                <View style={styles.stepRow}>
                  <Text style={styles.stepLabel}>
                    CHALLENGE {activeChallengeIndex + 1} OF {NUM_CHALLENGES}
                  </Text>
                  <Text style={styles.stepTimer}>
                    {CHALLENGE_DURATION_SECONDS -
                      ((TOTAL_RECORDING_SECONDS - recordCountdown) % CHALLENGE_DURATION_SECONDS)}s left
                  </Text>
                </View>

                {/* Challenge Prompt Card */}
                <View style={styles.challengeCard}>
                  <Text style={styles.challengeIcon}>
                    {getActionIcon(currentChallenge.type)}
                  </Text>
                  <Text style={styles.challengeText}>
                    {currentChallenge.instruction}
                  </Text>
                </View>

                {/* Global Progress Bar */}
                <View style={styles.progressBarBackground}>
                  <View
                    style={[
                      styles.progressBarFill,
                      {
                        width: `${Math.round(
                          ((TOTAL_RECORDING_SECONDS - recordCountdown) /
                            TOTAL_RECORDING_SECONDS) *
                          100
                        )}%`,
                      },
                    ]}
                  />
                </View>
              </View>
            )}

            {phase === 'ANALYZING' && (
              <View style={styles.analyzingContainer}>
                <ActivityIndicator size="large" color="#38BDF8" style={styles.spinner} />
                <Text style={styles.analyzingTitle}>Analyzing Guided Liveness</Text>
                <Text style={styles.analyzingStatus}>{analysisStatusText}</Text>
              </View>
            )}
          </View>
        </View>
      ) : (
        /* Results View */
        <ScrollView style={styles.resultScrollView} contentContainerStyle={styles.resultContent}>
          {result && (
            <View style={styles.resultCard}>
              {/* Challenge Verification Section */}
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>1. Challenge-Based Liveness</Text>
                <View
                  style={[
                    styles.badge,
                    result.verdictTier === 'VERIFIED'
                      ? styles.badgeSuccess
                      : styles.badgeWarning,
                  ]}
                >
                  <Text
                    style={[
                      styles.badgeText,
                      result.verdictTier === 'VERIFIED'
                        ? styles.badgeTextSuccess
                        : styles.badgeTextWarning,
                    ]}
                  >
                    {result.verdictTier === 'RETRY_REQUIRED'
                      ? 'RETRY REQUIRED'
                      : result.challengeLivenessPass
                        ? `PASSED (${result.challengesPassed}/${result.challengesTotal})`
                        : `UNCONFIRMED (${result.challengesPassed}/${result.challengesTotal})`}
                  </Text>
                </View>
              </View>

              {/* Challenge Breakdown List */}
              <View style={styles.challengeList}>
                {(result.challenges || []).map((ch, index) => (
                  <View key={ch.id || index} style={styles.challengeRow}>
                    <Text style={styles.challengeItemIcon}>
                      {ch.verified ? '✅' : '⚠️'}
                    </Text>
                    <View style={styles.challengeItemBody}>
                      <Text style={styles.challengeItemTitle}>
                        {ch.instruction}
                      </Text>
                      <Text
                        style={[
                          styles.challengeItemDetail,
                          ch.verified ? styles.textVerified : styles.textUnconfirmed,
                        ]}
                      >
                        {result.verdictTier === 'RETRY_REQUIRED'
                          ? 'Face was not clearly detected to verify movement'
                          : ch.detail}
                      </Text>
                    </View>
                    <Text
                      style={[
                        styles.challengeItemStatus,
                        ch.verified ? styles.textVerified : styles.textUnconfirmed,
                      ]}
                    >
                      {ch.verified ? 'VERIFIED' : 'UNCONFIRMED'}
                    </Text>
                  </View>
                ))}
              </View>

              {/* Physiological Pulse rPPG Section */}
              <View style={[styles.sectionHeader, { marginTop: 20 }]}>
                <Text style={styles.sectionTitle}>2. Physiological Pulse (rPPG)</Text>
                <View
                  style={[
                    styles.badge,
                    result.status === 'CONFIRMED_PULSE'
                      ? styles.badgeSuccess
                      : styles.badgeWarning,
                  ]}
                >
                  <Text
                    style={[
                      styles.badgeText,
                      result.status === 'CONFIRMED_PULSE'
                        ? styles.badgeTextSuccess
                        : styles.badgeTextWarning,
                    ]}
                  >
                    {result.verdictTier === 'RETRY_REQUIRED'
                      ? 'RETRY REQUIRED'
                      : result.status.replace('INCONCLUSIVE_', '')}
                  </Text>
                </View>
              </View>
              <Text style={styles.resultDescription}>{result.message}</Text>

              {/* Metric Tiles Grid */}
              <View style={styles.metricsGrid}>
                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>ESTIMATED PULSE</Text>
                  <Text style={styles.metricValue}>
                    {result.estimatedBpm > 0
                      ? `${result.estimatedBpm.toFixed(1)} BPM`
                      : 'N/A'}
                  </Text>
                </View>

                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>SIGNAL QUALITY</Text>
                  <Text style={styles.metricValue}>{result.signalQuality}</Text>
                </View>

                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>TRACKED FRAMES</Text>
                  <Text style={styles.metricValue}>
                    {result.usableFrames} / {result.totalFrames}
                  </Text>
                </View>

                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>SNR RATIO</Text>
                  <Text style={styles.metricValue}>
                    {result.snrDb.toFixed(1)} dB
                  </Text>
                </View>

                {result.processingMetrics && (
                  <View style={[styles.metricTile, { flexBasis: '100%' }]}>
                    <Text style={styles.metricLabel}>PROCESSING DURATION</Text>
                    <Text style={styles.metricValue}>
                      {(result.processingMetrics.totalDurationMs / 1000).toFixed(2)}s @ {result.processingMetrics.fpsSampled} FPS
                    </Text>
                  </View>
                )}
              </View>

              {/* Multi-Factor Zero-Trust Decision Engine */}
              <View style={styles.riskCard}>
                <Text style={styles.riskCardTitle}>Zero-Trust Multi-Factor Verdict</Text>
                <Text style={styles.riskCardText}>
                  {result.verdictTier === 'RETRY_REQUIRED'
                    ? "We couldn't detect your face clearly. Face the camera, improve lighting, and try again."
                    : result.verdictTier === 'VERIFIED'
                      ? 'Both interactive facial challenge confirmation and physiological blood-volume pulsatile verification passed on-device.'
                      : result.verdictTier === 'INCONCLUSIVE'
                        ? 'Interactive action challenges were completed, but subcutaneous pulse signals did not meet high-confidence threshold.'
                        : 'One or more required interactive challenge actions could not be verified by landmark analysis.'}
                </Text>
                <View style={styles.riskRow}>
                  <Text style={styles.riskLabel}>Overall Assurance Tier:</Text>
                  <Text
                    style={[
                      styles.riskValue,
                      result.verdictTier === 'VERIFIED'
                        ? styles.riskPassed
                        : result.verdictTier === 'FAILED'
                          ? styles.riskFailed
                          : styles.riskElevated,
                    ]}
                  >
                    {result.verdictTier === 'VERIFIED'
                      ? 'VERIFIED (LOW RISK)'
                      : result.verdictTier === 'RETRY_REQUIRED'
                        ? 'RETRY REQUIRED'
                        : result.verdictTier === 'INCONCLUSIVE'
                          ? 'INCONCLUSIVE'
                          : 'FAILED (HIGH RISK)'}
                  </Text>
                </View>
              </View>

              {/* Action Buttons */}
              <View style={styles.buttonRow}>
                <TouchableOpacity
                  style={result.verdictTier === 'RETRY_REQUIRED' ? styles.primaryButton : styles.secondaryButton}
                  onPress={resetScan}
                >
                  <Text style={result.verdictTier === 'RETRY_REQUIRED' ? styles.primaryButtonText : styles.secondaryButtonText}>
                    {result.verdictTier === 'RETRY_REQUIRED' ? 'Try Again' : 'Re-test Liveness'}
                  </Text>
                </TouchableOpacity>

                {onNavigate && (
                  <TouchableOpacity
                    style={result.verdictTier === 'RETRY_REQUIRED' ? styles.secondaryButton : styles.primaryButton}
                    onPress={() => onNavigate('Result')}
                  >
                    <Text style={result.verdictTier === 'RETRY_REQUIRED' ? styles.secondaryButtonText : styles.primaryButtonText}>
                      View Full Results
                    </Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          )}
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
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  header: {
    paddingTop: 85,
    paddingHorizontal: 20,
    paddingBottom: 12,
    backgroundColor: '#0F172A',
  },
  backButton: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E293B',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  backButtonText: {
    color: '#38BDF8',
    fontSize: 13,
    fontWeight: '600',
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: '#F8FAFC',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 2,
    fontWeight: '500',
  },
  cameraWrapper: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  ovalGuide: {
    width: 250,
    height: 340,
    borderRadius: 125,
    borderWidth: 3,
    borderColor: '#38BDF8',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
  },
  ovalGuidePreparing: {
    borderColor: '#F59E0B',
    borderStyle: 'solid',
  },
  ovalGuideRecording: {
    borderColor: '#10B981',
    borderStyle: 'solid',
  },
  prepOverlay: {
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.75)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 16,
  },
  prepNumber: {
    color: '#F59E0B',
    fontSize: 48,
    fontWeight: '900',
  },
  prepText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 1,
    marginTop: 4,
  },
  recordingIndicator: {
    position: 'absolute',
    top: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#EF4444',
    marginRight: 6,
  },
  recordingText: {
    color: '#F8FAFC',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  instructionsContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#1E293B',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 24,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
    borderColor: '#334155',
  },
  centerHUD: {
    alignItems: 'center',
  },
  instructionHeading: {
    fontSize: 17,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 4,
    textAlign: 'center',
  },
  instructionDetail: {
    fontSize: 12,
    color: '#94A3B8',
    lineHeight: 18,
    textAlign: 'center',
    marginBottom: 14,
  },
  recordingHUD: {
    width: '100%',
  },
  stepRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  stepLabel: {
    color: '#38BDF8',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  stepTimer: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '700',
  },
  challengeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0F172A',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#38BDF8',
    marginBottom: 12,
  },
  challengeIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  challengeText: {
    flex: 1,
    color: '#F8FAFC',
    fontSize: 15,
    fontWeight: '700',
  },
  progressBarBackground: {
    height: 6,
    backgroundColor: '#334155',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#38BDF8',
  },
  analyzingContainer: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  spinner: {
    marginBottom: 8,
  },
  analyzingTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 4,
  },
  analyzingStatus: {
    fontSize: 13,
    color: '#38BDF8',
    fontWeight: '600',
    marginBottom: 4,
  },
  analyzingSubtitle: {
    fontSize: 11,
    color: '#64748B',
    textAlign: 'center',
  },
  errorBox: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 12,
  },
  errorText: {
    color: '#F87171',
    fontSize: 12,
    textAlign: 'center',
  },
  primaryButton: {
    backgroundColor: '#0EA5E9',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 14,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#334155',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#F8FAFC',
    fontWeight: '700',
    fontSize: 14,
  },
  resultScrollView: {
    flex: 1,
  },
  resultContent: {
    padding: 16,
    paddingBottom: 40,
  },
  resultCard: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  sectionTitle: {
    color: '#F8FAFC',
    fontSize: 15,
    fontWeight: '700',
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeSuccess: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
  },
  badgeWarning: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  badgeTextSuccess: {
    color: '#10B981',
  },
  badgeTextWarning: {
    color: '#F59E0B',
  },
  challengeList: {
    backgroundColor: '#0F172A',
    borderRadius: 12,
    padding: 10,
    gap: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  challengeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  challengeItemIcon: {
    fontSize: 16,
    marginRight: 10,
  },
  challengeItemBody: {
    flex: 1,
  },
  challengeItemTitle: {
    color: '#F8FAFC',
    fontSize: 13,
    fontWeight: '600',
  },
  challengeItemDetail: {
    fontSize: 11,
    marginTop: 1,
  },
  challengeItemStatus: {
    fontSize: 11,
    fontWeight: '700',
    marginLeft: 8,
  },
  textVerified: {
    color: '#10B981',
  },
  textUnconfirmed: {
    color: '#F59E0B',
  },
  resultDescription: {
    color: '#94A3B8',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 14,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  metricTile: {
    flexBasis: '48%',
    backgroundColor: '#0F172A',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  metricLabel: {
    color: '#64748B',
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.5,
    marginBottom: 3,
  },
  metricValue: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '700',
  },
  riskCard: {
    backgroundColor: '#0F172A',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 16,
  },
  riskCardTitle: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 4,
  },
  riskCardText: {
    color: '#94A3B8',
    fontSize: 12,
    lineHeight: 16,
    marginBottom: 10,
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#334155',
  },
  riskLabel: {
    color: '#94A3B8',
    fontSize: 12,
  },
  riskValue: {
    fontSize: 13,
    fontWeight: '800',
  },
  riskPassed: {
    color: '#10B981',
  },
  riskElevated: {
    color: '#F59E0B',
  },
  riskFailed: {
    color: '#EF4444',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  loadingText: {
    color: '#94A3B8',
    fontSize: 14,
    marginTop: 12,
  },
  card: {
    backgroundColor: '#1E293B',
    padding: 24,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#334155',
    width: '100%',
    maxWidth: 400,
  },
  warningTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 13,
    color: '#94A3B8',
    lineHeight: 20,
    marginBottom: 20,
  },
});