import React, { useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { colors, spacing, radius, typography } from '../theme';

type Props = {
  documentUri: string;
  onCancel: () => void;
  onComplete: () => void;
};

export default function FaceLivenessScreen({ onCancel, onComplete }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [processing, setProcessing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={[styles.container, styles.center]}>
        <Text style={styles.title}>Camera access needed</Text>
        <Pressable style={styles.cta} onPress={requestPermission}>
          <Text style={styles.ctaText}>Grant Access</Text>
        </Pressable>
        <Pressable style={styles.secondary} onPress={onCancel}>
          <Text style={styles.secondaryText}>Cancel</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  const handleCapture = async () => {
    if (!cameraRef.current || processing) return;
    setProcessing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      if (photo?.uri) {
        // TODO: send photo.uri + documentUri to backend for liveness + matching
        onComplete();
      }
    } finally {
      setProcessing(false);
    }
  };

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />

      <SafeAreaView style={StyleSheet.absoluteFill} pointerEvents="box-none">
        <View style={styles.topBar}>
          <Pressable onPress={onCancel} hitSlop={12}>
            <Text style={styles.close}>✕</Text>
          </Pressable>
          <Text style={styles.title}>Face Liveness</Text>
          <View style={{ width: 24 }} />
        </View>

        <View style={styles.guide}>
          <Text style={styles.instruction}>Center your face in the frame</Text>
          <Text style={styles.instructionSub}>Ensure good lighting, remove glasses if possible</Text>
        </View>

        <View style={styles.bottomBar}>
          {processing ? (
            <ActivityIndicator size="large" color={colors.primary} />
          ) : (
            <Pressable onPress={handleCapture} style={styles.shutter}>
              <View style={styles.shutterInner} />
            </Pressable>
          )}
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { alignItems: 'center', justifyContent: 'center', padding: spacing.xl },
  title: { ...typography.h2, color: colors.textPrimary, marginBottom: spacing.sm },
  cta: {
    backgroundColor: colors.primary,
    borderRadius: radius.pill,
    paddingVertical: 14,
    paddingHorizontal: spacing.xl,
    marginBottom: spacing.md,
  },
  ctaText: { ...typography.bodyBold, color: colors.bg },
  secondary: { padding: spacing.sm },
  secondaryText: { ...typography.body, color: colors.textSecondary },
  topBar: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  close: { color: colors.textPrimary, fontSize: 22 },
  guide: {
    position: 'absolute',
    top: '15%',
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  instruction: { ...typography.bodyBold, color: colors.textPrimary, textAlign: 'center' },
  instructionSub: { ...typography.caption, color: colors.textSecondary, marginTop: 4, textAlign: 'center' },
  bottomBar: {
    position: 'absolute',
    bottom: spacing.xxl,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 3,
    borderColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterInner: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.primary,
  },
});