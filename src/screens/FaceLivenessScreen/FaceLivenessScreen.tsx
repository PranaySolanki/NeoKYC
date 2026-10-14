import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import type { Camera as VisionCamera } from 'react-native-vision-camera';
import { Camera, Face } from 'react-native-vision-camera-face-detector';

type Challenge = 'movement' | 'blink' | 'smile';

const CHALLENGE_COPY: Record<Challenge, { title: string; instruction: string }> = {
  movement: {
    title: 'Head movement',
    instruction: 'Slowly turn your head to either side, then face the camera again.',
  },
  blink: {
    title: 'Eye blink',
    instruction: 'Blink once, then open your eyes and look at the camera.',
  },
  smile: {
    title: 'Smile',
    instruction: 'Give the camera a natural smile and hold it briefly.',
  },
};

const shuffledChallenges = (): Challenge[] => {
  const items: Challenge[] = ['movement', 'blink', 'smile'];
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
};

interface FaceLivenessScreenProps {
  onNavigate?: (screenName: string) => void;
}

export const FaceLivenessScreen: React.FC<FaceLivenessScreenProps> = ({ onNavigate }) => {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<VisionCamera>(null);
  const [challenges, setChallenges] = useState<Challenge[]>(shuffledChallenges);
  const [challengeIndex, setChallengeIndex] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [faceMessage, setFaceMessage] = useState('Center your face in the frame');
  const [permissionRequested, setPermissionRequested] = useState(false);
  const blinkState = useRef<'waiting-open' | 'waiting-close' | 'waiting-reopen'>('waiting-open');
  const smileFrames = useRef(0);
  const movementDetected = useRef(false);
  const lastUiUpdate = useRef(0);

  const challenge = challenges[challengeIndex];

  const handleFacesDetection = useCallback((faces: Face[]) => {
    const now = Date.now();
    if (now - lastUiUpdate.current > 250) {
      lastUiUpdate.current = now;
      if (faces.length === 0) setFaceMessage('No face detected. Move into the frame.');
      else if (faces.length > 1) setFaceMessage('Only one person should be visible.');
      else setFaceMessage('Face detected. Follow the prompt below.');
    }

    if (completed || faces.length !== 1 || !challenge) return;
    const face = faces[0];
    let passed = false;

    if (challenge === 'movement') {
      if (Math.abs(face.yawAngle) >= 18) movementDetected.current = true;
      passed = movementDetected.current && Math.abs(face.yawAngle) <= 10;
    } else if (challenge === 'blink') {
      const openProbability = Math.min(face.leftEyeOpenProbability, face.rightEyeOpenProbability);
      if (openProbability >= 0.7 && blinkState.current === 'waiting-open') {
        blinkState.current = 'waiting-close';
      } else if (openProbability <= 0.25 && blinkState.current === 'waiting-close') {
        blinkState.current = 'waiting-reopen';
      } else if (openProbability >= 0.7 && blinkState.current === 'waiting-reopen') {
        passed = true;
        blinkState.current = 'waiting-open';
      }
    } else if (challenge === 'smile') {
      smileFrames.current = face.smilingProbability >= 0.7 ? smileFrames.current + 1 : 0;
      passed = smileFrames.current >= 4;
    }

    if (passed) {
      smileFrames.current = 0;
      if (challengeIndex + 1 >= challenges.length) {
        setCompleted(true);
      } else {
        setChallengeIndex((index) => index + 1);
      }
    }
  }, [challenge, challengeIndex, challenges.length, completed]);

  const restart = () => {
    // A new randomized order prevents the same gesture order on every attempt.
    setChallenges(shuffledChallenges());
    setChallengeIndex(0);
    setCompleted(false);
    setFaceMessage('Center your face in the frame');
    blinkState.current = 'waiting-open';
    smileFrames.current = 0;
    movementDetected.current = false;
  };

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Camera access needed</Text>
        <Text style={styles.body}>NeoKYC uses the front camera to check the requested face movements.</Text>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={async () => {
            setPermissionRequested(true);
            await requestPermission();
          }}
        >
          <Text style={styles.primaryButtonText}>{permissionRequested ? 'Allow camera access in Settings' : 'Allow camera access'}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#38BDF8" />
        <Text style={styles.body}>Finding the front camera…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Liveness check</Text>
      <Text style={styles.subtitle}>Complete all three live actions to continue.</Text>

      <View style={styles.cameraFrame}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={!completed}
          faceDetectionCallback={handleFacesDetection}
          faceDetectionOptions={{
            cameraFacing: 'front',
            performanceMode: 'fast',
            classificationMode: 'all',
            landmarkMode: 'none',
            contourMode: 'none',
            minFaceSize: 0.15,
            trackingEnabled: true,
          }}
        />
        <View pointerEvents="none" style={styles.faceGuide} />
        <View pointerEvents="none" style={styles.cameraStatus}>
          <Text style={styles.cameraStatusText}>{faceMessage}</Text>
        </View>
      </View>

      {completed ? (
        <View style={styles.resultCard}>
          <Text style={styles.successMark}>✓</Text>
          <Text style={styles.resultTitle}>Liveness actions complete</Text>
          <Text style={styles.body}>All three requested actions were detected.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={() => onNavigate?.('Result')}>
            <Text style={styles.primaryButtonText}>Continue to verification</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.secondaryButton} onPress={restart}>
            <Text style={styles.secondaryButtonText}>Run again</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.challengeCard}>
          <View style={styles.progressRow}>
            {challenges.map((item, index) => (
              <View key={item} style={[styles.progressDot, index <= challengeIndex && styles.progressDotActive]} />
            ))}
            <Text style={styles.progressText}>{challengeIndex + 1} of {challenges.length}</Text>
          </View>
          <Text style={styles.challengeTitle}>{CHALLENGE_COPY[challenge].title}</Text>
          <Text style={styles.body}>{CHALLENGE_COPY[challenge].instruction}</Text>
          <Text style={styles.privacyNote}>Face signals are processed on this device and are not saved by this screen.</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', paddingHorizontal: 20, paddingTop: 68, paddingBottom: 24 },
  center: { flex: 1, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center', padding: 28 },
  title: { color: '#F8FAFC', fontSize: 23, fontWeight: '800', marginBottom: 6 },
  subtitle: { color: '#94A3B8', fontSize: 14, lineHeight: 20, marginBottom: 18 },
  body: { color: '#CBD5E1', fontSize: 14, lineHeight: 21 },
  cameraFrame: { width: '100%', height: 350, borderRadius: 20, overflow: 'hidden', backgroundColor: '#020617', borderWidth: 1, borderColor: '#334155' },
  faceGuide: { position: 'absolute', width: 220, height: 290, alignSelf: 'center', top: 30, borderWidth: 2, borderColor: 'rgba(56, 189, 248, 0.9)', borderRadius: 120 },
  cameraStatus: { position: 'absolute', left: 14, right: 14, bottom: 14, paddingVertical: 8, paddingHorizontal: 12, backgroundColor: 'rgba(15,23,42,0.8)', borderRadius: 10, alignItems: 'center' },
  cameraStatusText: { color: '#E2E8F0', fontSize: 12, fontWeight: '600', textAlign: 'center' },
  challengeCard: { marginTop: 18, padding: 18, borderRadius: 16, backgroundColor: '#1E293B', borderWidth: 1, borderColor: '#334155' },
  progressRow: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 16 },
  progressDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#475569' },
  progressDotActive: { backgroundColor: '#38BDF8' },
  progressText: { color: '#94A3B8', fontSize: 12, fontWeight: '700', marginLeft: 'auto' },
  challengeTitle: { color: '#38BDF8', fontSize: 17, fontWeight: '800', marginBottom: 6 },
  privacyNote: { color: '#64748B', fontSize: 11, lineHeight: 16, marginTop: 14 },
  resultCard: { marginTop: 18, padding: 18, borderRadius: 16, backgroundColor: '#1E293B', borderWidth: 1, borderColor: '#0D9488', alignItems: 'center' },
  successMark: { color: '#2DD4BF', fontSize: 32, fontWeight: '900', marginBottom: 4 },
  resultTitle: { color: '#F8FAFC', fontSize: 17, fontWeight: '800', marginBottom: 4 },
  primaryButton: { alignSelf: 'stretch', alignItems: 'center', backgroundColor: '#0284C7', paddingVertical: 13, paddingHorizontal: 18, borderRadius: 11, marginTop: 16 },
  primaryButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '800', textAlign: 'center' },
  secondaryButton: { padding: 12, marginTop: 2 },
  secondaryButtonText: { color: '#94A3B8', fontSize: 13, fontWeight: '700' },
});
