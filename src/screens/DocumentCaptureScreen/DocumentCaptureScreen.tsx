import React, { useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import Svg, { Rect, Defs, Mask } from 'react-native-svg';
import { colors, spacing, radius, typography } from '../theme';

const { width, height } = Dimensions.get('window');
const FRAME_W = width - spacing.lg * 2;
const FRAME_H = FRAME_W * 0.63; // standard ID card aspect ratio

type Props = {
  onCaptured: (uri: string) => void;
  onCancel: () => void;
};

export default function DocumentCaptureScreen({ onCaptured, onCancel }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing] = useState<CameraType>('back');
  const [capturing, setCapturing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={[styles.container, styles.center]}>
        <Text style={styles.permissionTitle}>Camera access needed</Text>
        <Text style={styles.permissionBody}>
          We use your camera only to scan your ID document. Nothing is stored.
        </Text>
        <Pressable style={styles.cta} onPress={requestPermission}>
          <Text style={styles.ctaText}>Grant Camera Access</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  const handleCapture = async () => {
    if (!cameraRef.current || capturing) return;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      if (photo?.uri) onCaptured(photo.uri);
    } finally {
      setCapturing(false);
    }
  };

  const frameTop = (height - FRAME_H) / 2 - 40;

  return (
    <View style={styles.container}>
      {/* IMPORTANT: CameraView must not receive children — every overlay
          below is rendered as a SIBLING using absolute positioning instead.
          Nesting views inside <CameraView> triggers RN warnings/crashes. */}
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing} />

      <Svg width="100%" height="100%" style={StyleSheet.absoluteFill} pointerEvents="none">
        <Defs>
          <Mask id="mask" x="0" y="0" width="100%" height="100%">
            <Rect x="0" y="0" width="100%" height="100%" fill="white" />
            <Rect
              x={spacing.lg}
              y={frameTop}
              width={FRAME_W}
              height={FRAME_H}
              rx={radius.md}
              fill="black"
            />
          </Mask>
        </Defs>
        <Rect x="0" y="0" width="100%" height="100%" fill="rgba(11,15,25,0.78)" mask="url(#mask)" />
        <Rect
          x={spacing.lg}
          y={frameTop}
          width={FRAME_W}
          height={FRAME_H}
          rx={radius.md}
          fill="none"
          stroke={colors.primary}
          strokeWidth={2}
        />
      </Svg>

      <SafeAreaView style={StyleSheet.absoluteFill} pointerEvents="box-none">
        <View style={styles.topBar}>
          <Pressable onPress={onCancel} hitSlop={12}>
            <Text style={styles.close}>✕</Text>
          </Pressable>
          <Text style={styles.topBarTitle}>Scan ID Document</Text>
          <View style={{ width: 24 }} />
        </View>

        <View style={[styles.instructionWrap, { top: frameTop + FRAME_H + spacing.lg }]}>
          <Text style={styles.instruction}>Align the front of your ID within the frame</Text>
          <Text style={styles.instructionSub}>Good lighting, no glare, all four corners visible</Text>
        </View>

        <View style={styles.bottomBar}>
          <Pressable
            onPress={handleCapture}
            disabled={capturing}
            style={({ pressed }) => [
              styles.shutter,
              pressed && { transform: [{ scale: 0.94 }] },
              capturing && { opacity: 0.6 },
            ]}
          >
            <View style={styles.shutterInner} />
          </Pressable>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { alignItems: 'center', justifyContent: 'center', padding: spacing.xl },
  permissionTitle: { ...typography.h2, color: colors.textPrimary, marginBottom: spacing.sm },
  permissionBody: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  topBar: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  close: { color: colors.textPrimary, fontSize: 22 },
  topBarTitle: { ...typography.bodyBold, color: colors.textPrimary },
  instructionWrap: { position: 'absolute', left: 0, right: 0, alignItems: 'center', paddingHorizontal: spacing.lg },
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
  cta: {
    backgroundColor: colors.primary,
    borderRadius: radius.pill,
    paddingVertical: 16,
    paddingHorizontal: spacing.xl,
  },
  ctaText: { ...typography.bodyBold, color: colors.bg },
});