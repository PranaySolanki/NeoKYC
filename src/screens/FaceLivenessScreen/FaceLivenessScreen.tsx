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
const TOTAL_RECORDING_SECONDS = 15;
const CHALLENGE_ACTION_SECONDS = 2;
const INITIAL_CENTER_SECONDS = 3;
const FINAL_STILL_SECONDS = 8;
const NUM_CHALLENGES = 4;

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
  const analysisStatusText = 'Processing the recording on this device.';

  const cameraRef = useRef<CameraView>(null);
  const scanInFlightRef = useRef<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  /**
   * Generates two randomized gestures between a brief setup and the still pulse window.
   */
  const generateChallenges = (): LivenessChallenge[] => {
    // Fisher-Yates gives each action an unbiased position before selecting two.
    const shuffled = [...AVAILABLE_ACTIONS];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    // Two brief gestures leave a continuous eight-second still interval for rPPG.
    const selectedMidActions = shuffled.slice(0, 2);

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

    let startTimeSec = 0;
    return fullSequence.map((action, idx) => {
      const duration = idx === 0
        ? INITIAL_CENTER_SECONDS
        : idx === fullSequence.length - 1
          ? FINAL_STILL_SECONDS
          : CHALLENGE_ACTION_SECONDS;
      const challenge = {
        id: `challenge_${idx + 1}_${action.type.toLowerCase()}`,
        type: action.type,
        instruction: action.instruction,
        startTimeSec,
        endTimeSec: startTimeSec + duration,
      };
      startTimeSec = challenge.endTimeSec;
      return challenge;
    });
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
          <Text style={styles.cardDescription}>
            NeoKYC uses local front-camera video to verify guided liveness actions and
            subcutaneous pulse signals on-device. No biometric data is sent to the cloud.
          </Text>
          <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
            <Text style={styles.primaryButtonText}>Enable Camera</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const startScanFlow = () => {
    // State updates render asynchronously; the ref closes the rapid double-tap race.
    if (phase !== 'IDLE' || scanInFlightRef.current) return;
    if (!cameraRef.current) {
      setErrorMessage('Camera preview is not ready yet. Please try again.');
      return;
    }
    scanInFlightRef.current = true;

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
    setRecordCountdown(TOTAL_RECORDING_SECONDS);

    // Use wall-clock elapsed time so delayed JS timer ticks do not desync prompts.
    const recordingStartedAt = Date.now();
    timerRef.current = setInterval(() => {
      const elapsedSeconds = Math.min(
        TOTAL_RECORDING_SECONDS,
        Math.floor((Date.now() - recordingStartedAt) / 1000),
      );
      const remaining = TOTAL_RECORDING_SECONDS - elapsedSeconds;
      setRecordCountdown(Math.max(0, remaining));

      // Challenge intervals are intentionally uneven: the final interval is still.
      const matchingChallenge = activeChallenges.findIndex(
        (challenge) => elapsedSeconds < challenge.endTimeSec,
      );
      const curIndex = matchingChallenge < 0 ? NUM_CHALLENGES - 1 : matchingChallenge;
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
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : 'Camera recording interrupted.');
      setPhase('IDLE');
      if (timerRef.current) clearInterval(timerRef.current);
      scanInFlightRef.current = false;
      return;
    }

    if (timerRef.current) clearInterval(timerRef.current);
    if (!recordedUri) {
      setErrorMessage('Video recording produced an empty output. Please retry.');
      setPhase('IDLE');
      scanInFlightRef.current = false;
      return;
    }

    // Phase: Analyzing on-device
    setPhase('ANALYZING');
    try {
      const analysisResult = await rppgService.processVideo(recordedUri, activeChallenges);

      setResult(analysisResult);
      setPhase('RESULT');

      if (onComplete) {
        onComplete(analysisResult);
      }
      // The app owns the full result screen, so show it as soon as analysis completes.
      onNavigate?.('Result');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setErrorMessage('Native rPPG processing failed: ' + message);
      setPhase('IDLE');
      scanInFlightRef.current = false;
    } finally {
      if (recordedUri) {
        // Cleanup failure must never replace the visible analysis result.
        try {
          await rppgService.deleteVideoFile(recordedUri);
        } catch {
          // The service reports cleanup failures without exposing the biometric URI.
        }
      }
    }
  };

  const resetScan = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    scanInFlightRef.current = false;
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
        <Text style={styles.headerTitle}>Face Liveness Verification</Text>
        <Text style={styles.headerSubtitle}>
          Active gesture challenges and liveness detection
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
          />

          {/* Expo CameraView is a native surface and does not support React children. */}
          <View pointerEvents="none" style={styles.overlayContainer}>
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

          {/* Interactive Instruction / Challenge HUD */}
          <View style={styles.instructionsContainer}>
            {phase === 'IDLE' && (
              <>
                <Text style={styles.instructionHeading}>Randomized Liveness Challenge</Text>
                <Text style={styles.instructionDetail}>
                  You will be prompted with short, guided actions (e.g. smile, turn head, blink)
                  followed by holding steady. The check takes 15 seconds.
                </Text>
                {errorMessage && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorText}>{errorMessage}</Text>
                  </View>
                )}
                <TouchableOpacity style={styles.primaryButton} onPress={startScanFlow}>
                  <Text style={styles.primaryButtonText}>Start Liveness Scan</Text>
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
                    {Math.max(0, Math.ceil(
                      currentChallenge.endTimeSec - (TOTAL_RECORDING_SECONDS - recordCountdown),
                    ))}s left
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
                <Text style={styles.analyzingTitle}>Verifying Liveness</Text>
                <Text style={styles.analyzingStatus}>{analysisStatusText}</Text>
                <Text style={styles.analyzingSubtitle}>
                  Validating facial gestures and authenticity on-device
                </Text>
              </View>
            )}
          </View>
        </View>
      ) : (
        /* Results View */
        <ScrollView style={styles.resultScrollView} contentContainerStyle={styles.resultContent}>
          {result && (
            <View style={styles.resultCard}>
              {/* 1. Final Verdict Banner */}
              <View
                style={[
                  styles.verdictCard,
                  result.verdict === 'REAL_FACE_VERIFIED'
                    ? styles.verdictReal
                    : result.verdict === 'SPOOF_DETECTED'
                      ? styles.verdictSpoof
                      : result.verdict === 'VERIFICATION_FAILED'
                        ? styles.verdictFailed
                        : styles.verdictInconclusive,
                ]}
              >
                <Text
                  style={[
                    styles.verdictTitle,
                    result.verdict === 'REAL_FACE_VERIFIED'
                      ? styles.textReal
                      : result.verdict === 'SPOOF_DETECTED'
                        ? styles.textSpoof
                        : result.verdict === 'VERIFICATION_FAILED'
                          ? styles.textFailed
                          : styles.textInconclusive,
                  ]}
                >
                  {result.verdictTitle || 'INCONCLUSIVE — TRY AGAIN'}
                </Text>
                <Text style={styles.verdictDescription}>
                  {result.verdictDescription || result.message}
                </Text>
              </View>

              {/* 2. Gesture Checks Card */}
              <View style={styles.gestureCard}>
                <View style={styles.cardHeader}>
                  <Text style={styles.cardTitle}>Gesture Checks</Text>
                  {result.challenges && result.challenges.length > 0 && (
                    <Text style={styles.counterText}>
                      {result.challenges.filter((c) => c.verified).length} / {result.challenges.length} Passed
                    </Text>
                  )}
                </View>

                {result.challenges && result.challenges.length > 0 ? (
                  <View style={styles.gestureList}>
                    {result.challenges.map((ch, index) => (
                      <View key={ch.id || index} style={styles.gestureItem}>
                        <View
                          style={[
                            styles.checkbox,
                            ch.verified ? styles.checkboxChecked : styles.checkboxUnchecked,
                          ]}
                        >
                          {ch.verified && <Text style={styles.checkboxCheckmark}>✓</Text>}
                        </View>
                        <View style={styles.gestureContent}>
                          <Text
                            style={[
                              styles.gestureName,
                              !ch.verified && styles.gestureNameIncomplete,
                            ]}
                          >
                            {ch.instruction}
                          </Text>
                        </View>
                        <Text
                          style={[
                            styles.gestureStatus,
                            ch.verified ? styles.statusCompleted : styles.statusIncomplete,
                          ]}
                        >
                          {ch.verified ? 'Completed' : 'Not completed'}
                        </Text>
                      </View>
                    ))}
                  </View>
                ) : (
                  <Text style={styles.emptyText}>No gesture checks recorded.</Text>
                )}
              </View>

              {/* Action Buttons */}
              <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.primaryButton} onPress={resetScan}>
                  <Text style={styles.primaryButtonText}>Re-test Liveness</Text>
                </TouchableOpacity>

                {onNavigate && (
                  <TouchableOpacity
                    style={styles.secondaryButton}
                    onPress={() => onNavigate('Home')}
                  >
                    <Text style={styles.secondaryButtonText}>Return Home</Text>
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
    zIndex: 1,
    elevation: 1,
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
  verdictCard: {
    borderRadius: 14,
    padding: 16,
    borderWidth: 1.5,
    marginBottom: 16,
  },
  verdictReal: {
    backgroundColor: 'rgba(16, 185, 129, 0.08)',
    borderColor: '#10B981',
  },
  verdictSpoof: {
    backgroundColor: 'rgba(239, 68, 68, 0.08)',
    borderColor: '#EF4444',
  },
  verdictFailed: {
    backgroundColor: 'rgba(245, 158, 11, 0.08)',
    borderColor: '#F59E0B',
  },
  verdictInconclusive: {
    backgroundColor: 'rgba(148, 163, 184, 0.08)',
    borderColor: '#64748B',
  },
  verdictTitle: {
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  textReal: {
    color: '#10B981',
  },
  textSpoof: {
    color: '#EF4444',
  },
  textFailed: {
    color: '#F59E0B',
  },
  textInconclusive: {
    color: '#94A3B8',
  },
  verdictDescription: {
    color: '#E2E8F0',
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '500',
  },
  gestureCard: {
    backgroundColor: '#1E293B',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 20,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardTitle: {
    color: '#F8FAFC',
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  counterText: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '600',
  },
  gestureList: {
    backgroundColor: '#0F172A',
    borderRadius: 10,
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: '#334155',
  },
  gestureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 5,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  checkboxChecked: {
    backgroundColor: '#10B981',
  },
  checkboxUnchecked: {
    borderWidth: 1.5,
    borderColor: '#64748B',
    backgroundColor: 'transparent',
  },
  checkboxCheckmark: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  gestureContent: {
    flex: 1,
    marginRight: 8,
  },
  gestureName: {
    color: '#F8FAFC',
    fontSize: 13,
    fontWeight: '600',
  },
  gestureNameIncomplete: {
    color: '#94A3B8',
  },
  gestureStatus: {
    fontSize: 12,
    fontWeight: '700',
  },
  statusCompleted: {
    color: '#10B981',
  },
  statusIncomplete: {
    color: '#64748B',
  },
  emptyText: {
    color: '#94A3B8',
    fontSize: 13,
    fontStyle: 'italic',
    paddingVertical: 10,
    textAlign: 'center',
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
