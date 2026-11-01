import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Animated, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View, Vibration } from 'react-native';
import { Camera, runAtTargetFps, useCameraDevice, useCameraFormat, useCameraPermission, useFrameProcessor, VisionCameraProxy } from 'react-native-vision-camera';
import { useFaceDetector } from 'react-native-vision-camera-face-detector';
import { useSharedValue, Worklets } from 'react-native-worklets-core';
import { estimatePulse, FrameSample, FrameSampleBuffer, LivenessVerificationResult, makeLivenessVerificationResult, RppgTelemetry } from '../../services/experimentalRppg';
import { FaceIdentityTracker, FacePresenceTracker, GestureChallenge, GestureProgressTracker, isFaceAlignedToGuide, isQuiescentRppgFrame, LivenessFailureReason, pickGestureChallenges, RppgFaceMotion } from '../../services/gestureStateMachine';

type Challenge = GestureChallenge;
type FrameStatus = 'NO_FACE_GRACE' | 'MULTIPLE_FACES_GRACE' | 'OFF_GUIDE' | 'HOLD_STILL' | 'READY';
const COPY: Record<Challenge, string> = {
  BLINK: 'Blink both eyes',
  SMILE: 'Smile',
  TURN_LEFT: 'Turn your head left',
  TURN_RIGHT: 'Turn your head right',
  LOOK_UP: 'Look up',
};
const SESSION_MAX_MS = 30_000;
const CHALLENGE_READ_MS = 2_000;
const CHALLENGE_MAX_MS = 6_000;
const PULSE_CAPTURE_MIN_MS = 4_000;
const PULSE_CAPTURE_MAX_MS = 6_000;
const IS_IOS = Platform.OS === 'ios';

interface FaceObservation {
  centerX: number; centerY: number; width: number; height: number;
  yaw: number; pitch: number; smile: number; leftEyeOpen: number; rightEyeOpen: number;
  trackingId: number | null;
  aligned: boolean;
}
interface DetectionFrame { faceCount: number; face?: FaceObservation; timestampMs: number; }
interface RoiCache { timestampMs: number; regions: Record<string, number>; }
interface FaceLivenessScreenProps {
  onNavigate?: (screenName: string) => void;
  onComplete?: (result: LivenessVerificationResult) => void;
}

export const FaceLivenessScreen: React.FC<FaceLivenessScreenProps> = ({ onNavigate, onComplete }) => {
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [{ videoResolution: { width: 640, height: 480 } }, { fps: 30 }]);
  const { hasPermission, requestPermission } = useCameraPermission();
  const [challenges, setChallenges] = useState(pickGestureChallenges);
  const [hasStarted, setHasStarted] = useState(false);
  const [challengeIndex, setChallengeIndex] = useState(0);
  const gestureTracker = useRef(new GestureProgressTracker(challenges));
  const [sessionResult, setSessionResult] = useState<'running' | 'passed' | 'failed'>('running');
  const [verificationResult, setVerificationResult] = useState<LivenessVerificationResult | null>(null);
  const [frameStatus, setFrameStatus] = useState<FrameStatus>('NO_FACE_GRACE');
  const [elapsedMs, setElapsedMs] = useState(0);
  const guidePulse = useRef(new Animated.Value(0.72)).current;
  const [permissionRequested, setPermissionRequested] = useState(false);
  const samples = useRef(new FrameSampleBuffer(300));
  const faceContinuity = useRef({ maintained: true });
  const presenceTracker = useRef(new FacePresenceTracker());
  const identityTracker = useRef(new FaceIdentityTracker());
  const sessionStart = useRef<number | null>(null);
  const challengeStartedAt = useRef<number | null>(null);
  const pulseCaptureStartedAt = useRef<number | null>(null);
  const [pulseCaptureActive, setPulseCaptureActive] = useState(false);
  const previousCenter = useRef<{ x: number; y: number; width: number; height: number } | null>(null);
  const lastUiUpdate = useRef(0);
  const lastGestureLog = useRef(0);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [gestureFeedback, setGestureFeedback] = useState(false);
  const latestRoi = useSharedValue<RoiCache | null>(null);
  const previousRppgMotion = useSharedValue<RppgFaceMotion | null>(null);
  const pulseCaptureMode = useSharedValue(false);
  const lastRppgSampleTimestamp = useSharedValue(0);
  const resultRef = useRef(sessionResult);
  resultRef.current = sessionResult;

  useEffect(() => {
    const animation = Animated.loop(Animated.sequence([
      Animated.timing(guidePulse, { toValue: 1, duration: 700, useNativeDriver: true }),
      Animated.timing(guidePulse, { toValue: 0.72, duration: 700, useNativeDriver: true }),
    ]));
    animation.start();
    return () => animation.stop();
  }, [guidePulse]);

  useEffect(() => () => {
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
  }, []);

  useEffect(() => {
    if (__DEV__ && hasStarted) {
      console.log(`[GestureEval] UI rendered Challenge ${Math.min(challengeIndex + 1, challenges.length)} of ${challenges.length}: ${challenges[challengeIndex] ?? 'minimum session time'}`);
    }
  }, [challengeIndex, challenges, hasStarted]);

  const options = useMemo(() => ({
    performanceMode: 'fast' as const,
    classificationMode: 'all' as const,
    landmarkMode: 'all' as const,
    contourMode: 'none' as const,
    minFaceSize: 0.15,
    trackingEnabled: true,
  }), []);
  const { detectFaces } = useFaceDetector(options);
  const rppgMeansPlugin = useMemo(() => IS_IOS ? undefined : VisionCameraProxy.initFrameProcessorPlugin('rppgRoiMeans', {}), []);

  const completeVerification = useCallback((completedGestures: number, faceMaintained: boolean, failureReason?: LivenessFailureReason) => {
    if (resultRef.current !== 'running') return;
    pulseCaptureMode.value = false;
    const telemetry = estimatePulse(samples.current.values());
    const result = makeLivenessVerificationResult(completedGestures, challenges.length, telemetry, faceMaintained, challenges, failureReason, gestureTracker.current.outcomes);
    resultRef.current = result.livenessPassed ? 'passed' : 'failed';
    setVerificationResult(result);
    setSessionResult(result.livenessPassed ? 'passed' : 'failed');
    onComplete?.(result);
  }, [challenges, onComplete, pulseCaptureMode]);

  const handleRppgSample = useCallback((sample: FrameSample) => {
    if (resultRef.current !== 'running') return;
    if (pulseCaptureStartedAt.current !== null && sample.timestampMs < pulseCaptureStartedAt.current) return;
    const last = samples.current.latest();
    const luminance = (sample.r + sample.g + sample.b) / 3;
    const previousLuminance = last ? (last.r + last.g + last.b) / 3 : luminance;
    if (last && Math.abs(luminance - previousLuminance) / Math.max(previousLuminance, 1) > 0.2) samples.current.clear();
    samples.current.push(sample);
  }, []);

  const handleFrame = useCallback(({ faceCount, face, timestampMs }: DetectionFrame) => {
    if (resultRef.current !== 'running') return;
    const now = timestampMs;
    // runOnJS is asynchronous: discard frames captured before the latest
    // challenge transition so an old pose cannot satisfy the next gesture.
    if (challengeStartedAt.current !== null && now < challengeStartedAt.current) return;
    if (sessionStart.current == null) sessionStart.current = now;
    const elapsed = now - sessionStart.current;
    if (now - lastUiUpdate.current > 200) {
      lastUiUpdate.current = now;
      setElapsedMs(elapsed);
    }
    const activeIndex = gestureTracker.current.activeGestureIndex;
    const presence = presenceTracker.current.evaluate(faceCount, now);
    let status: FrameStatus = presence === 'NO_FACE_GRACE' ? 'NO_FACE_GRACE' : presence === 'MULTIPLE_FACES_GRACE' ? 'MULTIPLE_FACES_GRACE' : 'READY';
    if (presence === 'NO_FACE_TIMEOUT' || presence === 'MULTIPLE_FACES_TIMEOUT') {
      faceContinuity.current.maintained = false;
      const reason: LivenessFailureReason = presence === 'MULTIPLE_FACES_TIMEOUT' ? 'MULTIPLE_FACES_DETECTED' : 'FACE_OUT_OF_BOUNDS';
      completeVerification(gestureTracker.current.completedCount, false, reason);
      return;
    }
    if (faceCount === 1 && face) {
      if (identityTracker.current.observe(face.trackingId) === 'CHANGED') {
        faceContinuity.current.maintained = false;
        completeVerification(gestureTracker.current.completedCount, false, 'FACE_SWAP_DETECTED');
        return;
      }
      const center = { x: face.centerX, y: face.centerY, width: face.width, height: face.height };
      const moving = Math.abs(face.yaw) > 10 || Math.abs(face.pitch) > 10 ||
        (previousCenter.current && (Math.hypot(center.x - previousCenter.current.x, center.y - previousCenter.current.y) > 0.035 ||
          Math.abs(center.width / previousCenter.current.width - 1) > 0.08 || Math.abs(center.height / previousCenter.current.height - 1) > 0.08));
      status = moving ? 'HOLD_STILL' : face.aligned ? 'READY' : 'OFF_GUIDE';
      previousCenter.current = center;
    } else {
      previousCenter.current = null;
    }
    setFrameStatus(status);
    if (__DEV__ && status !== 'READY' && status !== 'HOLD_STILL' && now - lastGestureLog.current >= 200) {
      lastGestureLog.current = now;
      console.log(`[GestureEval] Guidance: ${status}, faces: ${faceCount}, step: ${Math.min(activeIndex + 1, challenges.length)} of ${challenges.length}`);
    }

    // Resolve this frame's face validity before the timeout/pass decision.
    // Pulse quality and BPM are intentionally absent from liveness scoring.
    if (activeIndex >= challenges.length && presence === 'PRESENT') {
      if (pulseCaptureStartedAt.current === null) {
        pulseCaptureStartedAt.current = now;
        pulseCaptureMode.value = true;
        setPulseCaptureActive(true);
        samples.current.clear();
        gestureTracker.current.resetHold();
      }
      const captureElapsed = now - pulseCaptureStartedAt.current;
      if ((captureElapsed >= PULSE_CAPTURE_MIN_MS && samples.current.length >= 120) || captureElapsed >= PULSE_CAPTURE_MAX_MS) {
        completeVerification(gestureTracker.current.completedCount, faceContinuity.current.maintained);
      }
      return;
    }
    if (elapsed >= SESSION_MAX_MS && presence === 'PRESENT') {
      const completed = gestureTracker.current.completedCount;
      completeVerification(completed, faceContinuity.current.maintained, completed >= 3 ? undefined : 'INCOMPLETE_GESTURES');
      return;
    }

    // rPPG is telemetry only. Invalid face/motion windows clear the local sample
    // buffer and any stale displayed result, but never touch liveness progress.
    // Presence grace frames reset a pending hold but never fail until their timer expires.
    // Alignment and gesture motion are guidance only, not liveness blockers.
    if (!face || faceCount !== 1 || activeIndex >= challenges.length || pulseCaptureStartedAt.current !== null) {
      gestureTracker.current.resetHold();
      return;
    }
    if (challengeStartedAt.current === null) challengeStartedAt.current = now;
    const challengeElapsed = now - challengeStartedAt.current;
    if (challengeElapsed >= CHALLENGE_MAX_MS) {
      const missed = gestureTracker.current.missCurrent();
      const nextIndex = gestureTracker.current.activeGestureIndex;
      challengeStartedAt.current = now;
      setChallengeIndex(nextIndex);
      gestureTracker.current.resetHold();
      if (__DEV__) console.log(`[GestureEval] Timed out ${missed}; advancing to Challenge ${Math.min(nextIndex + 1, challenges.length)} of ${challenges.length}`);
      return;
    }
    if (challengeElapsed < CHALLENGE_READ_MS) {
      gestureTracker.current.resetHold();
      return;
    }
    const evaluation = gestureTracker.current.evaluate({
      timestampMs: now,
      yaw: face.yaw,
      pitch: face.pitch,
      smile: face.smile,
      leftEyeOpen: face.leftEyeOpen,
      rightEyeOpen: face.rightEyeOpen,
    });
    const challenge = evaluation.challenge;
    if (__DEV__ && now - lastGestureLog.current >= 200) {
      lastGestureLog.current = now;
      console.log(`[GestureEval] Current: ${challenge}, Yaw: ${face.yaw.toFixed(1)}, Pitch: ${face.pitch.toFixed(1)}, Smile: ${face.smile.toFixed(2)}, Eyes: [${face.leftEyeOpen.toFixed(2)}, ${face.rightEyeOpen.toFixed(2)}], HoldMs: ${evaluation.holdMs.toFixed(0)}`);
    }
    if (!evaluation.completed) return;

    Vibration.vibrate(35);
    const nextIndex = gestureTracker.current.activeGestureIndex;
    challengeStartedAt.current = now;
    setChallengeIndex(nextIndex);
    setGestureFeedback(true);
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    feedbackTimer.current = setTimeout(() => setGestureFeedback(false), 650);
    if (__DEV__) console.log(`[GestureEval] Completed ${challenge}; advancing to Challenge ${Math.min(nextIndex + 1, challenges.length)} of ${challenges.length}`);
  }, [challenges, completeVerification]);

  const onFrameOnJS = useMemo(() => Worklets.createRunOnJS((frame: DetectionFrame) => handleFrame(frame)), [handleFrame]);
  const onRppgSampleOnJS = useMemo(() => Worklets.createRunOnJS((sample: FrameSample) => handleRppgSample(sample)), [handleRppgSample]);
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    const timestampMs = IS_IOS ? frame.timestamp : frame.timestamp / 1_000_000;
    const detectorWidth = frame.height;
    const detectorHeight = frame.width;

    // ML Kit and gesture evaluation run at 15 FPS. Only compact face metrics
    // cross to JS; never send Face[] or image data over runOnJS.
    runAtTargetFps(12, () => {
      'worklet';
      const faces = detectFaces(frame);
      let faceObservation: FaceObservation | undefined;
      let roiRegions: Record<string, number> | null = null;
      if (faces.length === 1) {
        const face = faces[0];
        const bounds = {
          x: face.bounds.x / detectorWidth,
          y: face.bounds.y / detectorHeight,
          width: face.bounds.width / detectorWidth,
          height: face.bounds.height / detectorHeight,
        };
        const points = face.landmarks;
        const leftEye = points?.LEFT_EYE ? { x: points.LEFT_EYE.x / detectorWidth, y: points.LEFT_EYE.y / detectorHeight } : null;
        const rightEye = points?.RIGHT_EYE ? { x: points.RIGHT_EYE.x / detectorWidth, y: points.RIGHT_EYE.y / detectorHeight } : null;
        const nose = points?.NOSE_BASE ? { x: points.NOSE_BASE.x / detectorWidth, y: points.NOSE_BASE.y / detectorHeight } : null;
        const eyeMidpoint = leftEye && rightEye ? { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 } : null;
        const requiredLandmarks = Boolean(leftEye && rightEye && nose && points?.MOUTH_LEFT && points?.MOUTH_RIGHT);
        const aligned = requiredLandmarks && isFaceAlignedToGuide(bounds, nose, eyeMidpoint);
        const yaw = frame.isMirrored ? -face.yawAngle : face.yawAngle;
        const trackingId = Number.isInteger(face.trackingId) && (face.trackingId as number) >= 0 ? face.trackingId as number : null;
        const centerX = bounds.x + bounds.width / 2, centerY = bounds.y + bounds.height / 2;
        faceObservation = {
          centerX, centerY,
          width: bounds.width, height: bounds.height,
          yaw, pitch: face.pitchAngle, smile: face.smilingProbability,
          leftEyeOpen: face.leftEyeOpenProbability, rightEyeOpen: face.rightEyeOpenProbability,
          trackingId,
          aligned,
        };

        const motion: RppgFaceMotion = { yaw, pitch: face.pitchAngle, centerX, centerY, timestampMs };
        const quiescent = isQuiescentRppgFrame(motion, previousRppgMotion.value);
        previousRppgMotion.value = motion;
        // Alignment is a visual recommendation; pulse ROIs require only
        // forward-facing, low-velocity frames.
        if (quiescent && leftEye && rightEye && eyeMidpoint) {
          const faceX = face.bounds.x / detectorWidth, faceY = face.bounds.y / detectorHeight;
          const faceWidth = bounds.width, faceHeight = bounds.height;
          const eyeLeft = Math.min(leftEye.x, rightEye.x), eyeRight = Math.max(leftEye.x, rightEye.x);
          const regions = [
            // Forehead strip anchored to the eye landmarks, between brows and hairline.
            { left: Math.max(faceX + faceWidth * 0.18, eyeLeft - faceWidth * 0.08), top: faceY + faceHeight * 0.06, right: Math.min(faceX + faceWidth * 0.82, eyeRight + faceWidth * 0.08), bottom: Math.min(eyeMidpoint!.y - faceHeight * 0.025, faceY + faceHeight * 0.36) },
          ];
          roiRegions = {};
          for (let index = 0; index < regions.length; index++) {
            roiRegions[`roi${index}Left`] = regions[index].left;
            roiRegions[`roi${index}Top`] = regions[index].top;
            roiRegions[`roi${index}Right`] = regions[index].right;
            roiRegions[`roi${index}Bottom`] = regions[index].bottom;
          }
        }
      } else previousRppgMotion.value = null;
      latestRoi.value = roiRegions ? { timestampMs, regions: roiRegions } : null;
      onFrameOnJS({ faceCount: faces.length, face: faceObservation, timestampMs });
    });

    // Sample at 15 FPS during gestures and 30 FPS during the four-second
    // pulse phase (120 frames). Extraction stays native/on the worklet.
    const sampleIntervalMs = pulseCaptureMode.value ? (1000 / 30) : (1000 / 15);
    if (timestampMs - lastRppgSampleTimestamp.value >= sampleIntervalMs) {
      lastRppgSampleTimestamp.value = timestampMs;
      const cached = latestRoi.value;
      if (!cached || timestampMs - cached.timestampMs > 100) return;
      let sample: FrameSample | undefined;
      if (IS_IOS && frame.pixelFormat === 'rgb') {
        const bytes = new Uint8Array(frame.toArrayBuffer());
        let red = 0, green = 0, blue = 0, count = 0;
        for (let index = 0; index < 3; index++) {
          const left = Math.max(0, Math.floor(cached.regions[`roi${index}Left`] * detectorWidth));
          const top = Math.max(0, Math.floor(cached.regions[`roi${index}Top`] * detectorHeight));
          const right = Math.min(detectorWidth, Math.floor(cached.regions[`roi${index}Right`] * detectorWidth));
          const bottom = Math.min(detectorHeight, Math.floor(cached.regions[`roi${index}Bottom`] * detectorHeight));
          for (let py = top; py < bottom; py += 6) for (let px = left; px < right; px += 6) {
            const rawX = frame.width - py, rawY = px, offset = rawY * frame.bytesPerRow + rawX * 4;
            if (offset + 2 < bytes.length) { red += bytes[offset + 2]; green += bytes[offset + 1]; blue += bytes[offset]; count++; }
          }
        }
        if (count > 0) sample = { r: red / count, g: green / count, b: blue / count, timestampMs };
      } else if (!IS_IOS && rppgMeansPlugin) {
        const means = rppgMeansPlugin.call(frame, cached.regions) as { r: number; g: number; b: number } | undefined;
        if (means) sample = { ...means, timestampMs };
      }
      if (sample) onRppgSampleOnJS(sample);
    }
  }, [detectFaces, latestRoi, previousRppgMotion, pulseCaptureMode, lastRppgSampleTimestamp, onFrameOnJS, onRppgSampleOnJS, rppgMeansPlugin]);

  const restart = () => {
    const nextChallenges = pickGestureChallenges();
    gestureTracker.current.reset(nextChallenges);
    setChallenges(nextChallenges); setChallengeIndex(0); setElapsedMs(0); setSessionResult('running'); resultRef.current = 'running';
    setHasStarted(false);
    samples.current.clear(); latestRoi.value = null; previousRppgMotion.value = null; pulseCaptureMode.value = false; lastRppgSampleTimestamp.value = 0; sessionStart.current = null; challengeStartedAt.current = null; pulseCaptureStartedAt.current = null; setPulseCaptureActive(false); presenceTracker.current = new FacePresenceTracker(); identityTracker.current.reset(); previousCenter.current = null; lastGestureLog.current = 0; faceContinuity.current = { maintained: true };
  };

  if (!hasPermission) return <View style={styles.center}><Text style={styles.title}>Camera access needed</Text><TouchableOpacity style={styles.primaryButton} onPress={async () => { setPermissionRequested(true); await requestPermission(); }}><Text style={styles.primaryButtonText}>{permissionRequested ? 'Allow camera access in Settings' : 'Allow camera access'}</Text></TouchableOpacity></View>;
  if (!device) return <View style={styles.center}><ActivityIndicator color="#38BDF8" /><Text style={styles.body}>Finding the front camera…</Text></View>;

  const message = pulseCaptureActive ? 'Hold steady while we measure your pulse.' : frameStatus === 'NO_FACE_GRACE' ? 'Face not visible — please return to frame.' : frameStatus === 'MULTIPLE_FACES_GRACE' ? 'Only one face should be visible.' : frameStatus === 'OFF_GUIDE' ? 'Center your face if you can.' : 'Follow the prompt.';
  const completedCount = verificationResult?.gestureBreakdown.filter((item) => item.status === 'Completed').length ?? 0;
  const failureMessages: Record<LivenessFailureReason, string> = {
    MULTIPLE_FACES_DETECTED: 'Multiple faces detected in frame',
    FACE_OUT_OF_BOUNDS: 'Face moved out of camera view',
    FACE_SWAP_DETECTED: 'Person change detected during scan',
    INCOMPLETE_GESTURES: `Incomplete gestures (Completed: ${completedCount}/${challenges.length} | Required: 75%)`,
  };
  const failureMessage = verificationResult?.failureReason ? failureMessages[verificationResult.failureReason] : null;
  const faceAligned = frameStatus === 'READY' || frameStatus === 'HOLD_STILL';
  const guideColor = gestureFeedback ? '#22C55E' : faceAligned ? '#22D3EE' : frameStatus === 'OFF_GUIDE' || frameStatus === 'MULTIPLE_FACES_GRACE' ? '#FBBF24' : '#F8FAFC';
  const currentChallenge = challenges[challengeIndex];
  const gesturesCompleted = gestureTracker.current.completedCount;
  const challengeElapsedUi = challengeStartedAt.current === null ? 0 : Math.max(0, elapsedMs + (sessionStart.current ?? 0) - challengeStartedAt.current);
  const pulseElapsedUi = pulseCaptureStartedAt.current === null ? 0 : Math.max(0, elapsedMs + (sessionStart.current ?? 0) - pulseCaptureStartedAt.current);
  const pulseCountdown = Math.max(0, Math.ceil((PULSE_CAPTURE_MIN_MS - pulseElapsedUi) / 1000));
  const prompt = pulseCaptureActive
    ? pulseCountdown > 0 ? `Hold steady for ${pulseCountdown} seconds` : 'Finishing pulse sample — stay steady'
    : currentChallenge ? COPY[currentChallenge] : 'Hold still while we finish';
  const promptNote = pulseCaptureActive
    ? pulseCountdown > 0 ? 'Keep your face forward.' : 'A moment longer on slower cameras.'
    : challengeElapsedUi < CHALLENGE_READ_MS ? 'Get ready' : 'Hold the action for a moment';
  const directionCue = pulseCaptureActive ? '♥' : currentChallenge === 'TURN_LEFT' ? '←' : currentChallenge === 'TURN_RIGHT' ? '→' : currentChallenge === 'LOOK_UP' ? '↑' : currentChallenge === 'SMILE' ? '☺' : currentChallenge === 'BLINK' ? '◉' : '';
  return <ScrollView style={styles.scroll} contentContainerStyle={styles.container}>
    <View style={styles.headerBar}><TouchableOpacity style={styles.backButton} onPress={() => onNavigate?.('Home')}><Text style={styles.backButtonText}>← </Text></TouchableOpacity><Text style={styles.title}> Liveness check</Text><View style={styles.headerSpacer} /></View>
    {!hasStarted && sessionResult === 'running' ? <View style={styles.challengeCard}>
      <Text style={styles.challengeTitle}>Face Verification</Text>
      <Text style={styles.body}>Center your face and follow the prompts.</Text>
      <View style={styles.actionList}>
        <View style={styles.actionRow}><Text style={styles.actionNumber}>•</Text><Text style={styles.actionText}>Use a bright, even light.</Text></View>
        <View style={styles.actionRow}><Text style={styles.actionNumber}>•</Text><Text style={styles.actionText}>Keep only your face in view.</Text></View>
        <View style={styles.actionRow}><Text style={styles.actionNumber}>•</Text><Text style={styles.actionText}>Follow one prompt at a time.</Text></View>
      </View>
      <Text style={styles.privacyNote}>Pulse is an optional estimate and does not affect verification.</Text>
      <TouchableOpacity style={styles.primaryButton} onPress={() => setHasStarted(true)}><Text style={styles.primaryButtonText}>Start Liveness Scan</Text></TouchableOpacity>
    </View> : <>
      <Text style={styles.subtitle}>{sessionResult === 'running' ? `Follow each prompt · ${gesturesCompleted} of ${challenges.length} passed` : 'Liveness scan finished.'}</Text>
      <View style={styles.cameraFrame}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          format={format}
          fps={30}
          video={true}
          isActive={hasStarted && sessionResult === 'running'}
          frameProcessor={frameProcessor}
          pixelFormat={IS_IOS ? 'rgb' : 'yuv'}
        />
        <Animated.View
          pointerEvents="none"
          style={[styles.faceGuide, { borderColor: guideColor, opacity: faceAligned ? guidePulse : 1 }]}
        />
        {hasStarted && sessionResult === 'running' && <View style={styles.instructionPill}>
          <View style={styles.instructionCopyRow}>
            <View style={styles.instructionCopy}>
              <Text style={styles.instructionStep}>{pulseCaptureActive ? 'PULSE' : `CHALLENGE ${Math.min(challengeIndex + 1, challenges.length)} OF ${challenges.length}`}</Text>
              <Text style={styles.instructionText}>{prompt}</Text>
              <Text style={styles.instructionHelp}>{promptNote}</Text>
            </View>
            <Text style={styles.directionCue}>{directionCue}</Text>
          </View>
          <Text style={[styles.cameraStatusText, frameStatus === 'NO_FACE_GRACE' || frameStatus === 'MULTIPLE_FACES_GRACE' ? styles.warningText : null]}>{message}</Text>
        </View>}
      </View>
      {sessionResult === 'running' ? <View style={styles.progressCard}>
        <View style={styles.timeProgressTrack}><View style={[styles.timeProgressFill, { width: `${Math.max(0, Math.min(1, challengeIndex / challenges.length)) * 100}%` }]} /></View>
        <Text style={styles.timeProgressLabel}>Four actions, then a pulse reading</Text>
      </View> : <View style={styles.challengeCard}>
        <View style={[styles.resultBadge, sessionResult === 'passed' ? styles.resultBadgePass : styles.resultBadgeFail]}><Text style={styles.resultBadgeText}>{sessionResult === 'passed' ? 'PASSED: Real Human' : `FAILED: ${failureMessage ?? 'Unsuccessful'}`}</Text></View>
        {sessionResult === 'failed' && <><Text style={styles.failureReason}>Verdict: Unsuccessful</Text><Text style={styles.failureReason}>Reason: {failureMessage}</Text></>}
        <Text style={styles.challengeTitle}>Gesture score: {Math.round((verificationResult?.livenessScore ?? 0) * 100)}% ({completedCount}/{challenges.length})</Text>
        <Text style={styles.telemetryTitle}>Gesture breakdown</Text>
        <View style={styles.actionList}>{(verificationResult?.gestureBreakdown ?? []).map((item, index) => <View key={`${item.gesture}-${index}`} style={styles.actionRow}><Text style={styles.actionNumber}>{item.status === 'Completed' ? '✓' : '✕'}</Text><Text style={styles.actionText}>{COPY[item.gesture as Challenge] ?? item.gesture}</Text><Text style={[styles.actionState, item.status === 'Completed' ? styles.completedText : styles.missedText]}>{item.status === 'Completed' ? 'Passed' : 'Missed'}</Text></View>)}</View>
        <View style={styles.telemetry}><Text style={styles.telemetryTitle}>Vital telemetry</Text><Text style={styles.telemetryValue}>{verificationResult?.rppgTelemetry.isMeasurementReliable && verificationResult.rppgTelemetry.estimatedBpm !== null ? `Pulse: ${verificationResult.rppgTelemetry.estimatedBpm.toFixed(0)} BPM` : `Pulse: Inconclusive (Confidence: ${Math.round((verificationResult?.rppgTelemetry.signalQualityScore ?? 0) * 100)}%)`}</Text><Text style={styles.telemetryTitle}>Signal Confidence: {Math.round((verificationResult?.rppgTelemetry.signalQualityScore ?? 0) * 100)}%</Text><Text style={styles.privacyNote}>Pulse confidence does not affect the liveness verdict.</Text></View>
        <TouchableOpacity style={styles.primaryButton} onPress={restart}><Text style={styles.primaryButtonText}>Run again</Text></TouchableOpacity>
        {sessionResult === 'passed' && <TouchableOpacity style={styles.secondaryButton} onPress={() => onNavigate?.('RegisterFace')}><Text style={styles.secondaryButtonText}>Continue to next step</Text></TouchableOpacity>}
        <TouchableOpacity style={styles.secondaryButton} onPress={() => onNavigate?.('HomeDash')}><Text style={styles.secondaryButtonText}>Done</Text></TouchableOpacity>
      </View>}
    </>}
  </ScrollView>;
};

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#0F172A' }, container: { flexGrow: 1, backgroundColor: '#0F172A', paddingHorizontal: 20, paddingTop: 68, paddingBottom: 24 }, center: { flex: 1, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center', padding: 28 },
  headerBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '100%', marginBottom: 6 }, backButton: { paddingVertical: 7, paddingHorizontal: 12, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 8 }, backButtonText: { color: '#38BDF8', fontSize: 14, fontWeight: '600' }, headerSpacer: { width: 60 },
  title: { color: '#F8FAFC', fontSize: 21, fontWeight: '800', textAlign: 'center' }, subtitle: { color: '#94A3B8', fontSize: 14, lineHeight: 20, marginBottom: 18 }, body: { color: '#CBD5E1', fontSize: 14, lineHeight: 21 }, cameraFrame: { width: '100%', height: 500, borderRadius: 20, overflow: 'hidden', backgroundColor: '#020617', borderWidth: 1, borderColor: '#334155' }, faceGuide: { position: 'absolute', width: 220, height: 270, alignSelf: 'center', top: '12%', borderWidth: 3, borderColor: '#F8FAFC', borderRadius: 120 }, instructionPill: { position: 'absolute', left: 12, right: 12, bottom: 12, paddingHorizontal: 14, paddingVertical: 12, backgroundColor: 'rgba(15,23,42,0.94)', borderWidth: 1, borderColor: 'rgba(56,189,248,0.6)', borderRadius: 15 }, instructionCopyRow: { flexDirection: 'row', alignItems: 'center', gap: 10 }, instructionCopy: { flex: 1 }, instructionStep: { color: '#7DD3FC', fontSize: 10, fontWeight: '900', letterSpacing: 1 }, instructionText: { color: '#F8FAFC', fontSize: 17, lineHeight: 22, fontWeight: '800', marginTop: 3 }, instructionHelp: { color: '#CBD5E1', fontSize: 11, marginTop: 3 }, cameraStatusText: { color: '#CBD5E1', fontSize: 11, fontWeight: '600', textAlign: 'left', marginTop: 8 }, warningText: { color: '#FBBF24' }, directionCue: { color: '#F8FAFC', fontSize: 30, fontWeight: '900', minWidth: 34, textAlign: 'center' },
  challengeCard: { marginTop: 18, padding: 18, borderRadius: 16, backgroundColor: '#1E293B', borderWidth: 1, borderColor: '#334155' }, progressCard: { marginTop: 14, paddingHorizontal: 12, paddingVertical: 6 }, challengeTitle: { color: '#38BDF8', fontSize: 17, fontWeight: '800', marginBottom: 6 }, timeProgressTrack: { height: 6, backgroundColor: '#334155', borderRadius: 3, overflow: 'hidden' }, timeProgressFill: { height: 6, backgroundColor: '#2DD4BF', borderRadius: 3 }, timeProgressLabel: { color: '#64748B', fontSize: 10, marginTop: 7 }, telemetry: { marginTop: 16, borderTopWidth: 1, borderColor: '#334155', paddingTop: 12 }, telemetryTitle: { color: '#94A3B8', fontSize: 12, fontWeight: '700', marginTop: 6 }, telemetryValue: { color: '#2DD4BF', fontSize: 22, fontWeight: '800', marginTop: 4 }, privacyNote: { color: '#64748B', fontSize: 11, lineHeight: 16, marginTop: 8 }, primaryButton: { alignItems: 'center', backgroundColor: '#0284C7', paddingVertical: 13, paddingHorizontal: 18, borderRadius: 11, marginTop: 16 }, primaryButtonText: { color: '#FFF', fontSize: 14, fontWeight: '800' }, secondaryButton: { padding: 12, marginTop: 2 }, secondaryButtonText: { color: '#94A3B8', fontSize: 13, fontWeight: '700', textAlign: 'center' }, resultBadge: { alignSelf: 'flex-start', maxWidth: '100%', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10, marginBottom: 12 }, resultBadgePass: { backgroundColor: '#064E3B' }, resultBadgeFail: { backgroundColor: '#7F1D1D' }, resultBadgeText: { color: '#F8FAFC', fontSize: 12, lineHeight: 18, fontWeight: '900' }, completedText: { color: '#2DD4BF' }, missedText: { color: '#FCA5A5' },
  failureReason: { color: '#FCA5A5', fontSize: 13, lineHeight: 19, fontWeight: '700', marginBottom: 10 }, actionList: { marginTop: 14, gap: 6 }, actionRow: { flexDirection: 'row', alignItems: 'center', minHeight: 34, paddingHorizontal: 8, paddingVertical: 6, borderRadius: 8, backgroundColor: 'rgba(15,23,42,0.55)' }, currentActionRow: { borderWidth: 1, borderColor: '#38BDF8' }, actionNumber: { width: 25, color: '#2DD4BF', fontSize: 13, fontWeight: '800' }, actionText: { flex: 1, color: '#CBD5E1', fontSize: 12, lineHeight: 17 }, actionState: { color: '#94A3B8', fontSize: 10, fontWeight: '700', marginLeft: 6 },
});

export default FaceLivenessScreen;
