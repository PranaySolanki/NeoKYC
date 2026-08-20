import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, spacing, radius, typography } from '../theme';

type Props = {
  verified: boolean;
  confidence: number;
  reason?: string;
  onRetry: () => void;
  onContinue: () => void;
};

export default function ResultScreen({ verified, confidence, reason, onRetry, onContinue }: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={[styles.icon, verified ? styles.success : styles.fail]}>
          {verified ? '✓' : '✕'}
        </Text>
        <Text style={styles.title}>
          {verified ? 'Identity Verified' : 'Verification Failed'}
        </Text>
        <Text style={styles.subtitle}>
          Confidence: {(confidence * 100).toFixed(1)}%
        </Text>
        {reason && <Text style={styles.reason}>{reason}</Text>}
      </View>

      <View style={styles.actions}>
        {!verified && (
          <Pressable style={styles.secondary} onPress={onRetry}>
            <Text style={styles.secondaryText}>Try Again</Text>
          </Pressable>
        )}
        <Pressable style={styles.cta} onPress={onContinue}>
          <Text style={styles.ctaText}>{verified ? 'Continue' : 'Go Home'}</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: spacing.lg },
  content: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  icon: { fontSize: 64, marginBottom: spacing.md },
  success: { color: '#4ADE80' },
  fail: { color: '#F87171' },
  title: { ...typography.h2, color: colors.textPrimary, marginBottom: spacing.sm, textAlign: 'center' },
  subtitle: { ...typography.body, color: colors.textSecondary, marginBottom: spacing.sm },
  reason: { ...typography.caption, color: colors.textSecondary, textAlign: 'center' },
  actions: { gap: spacing.md },
  cta: {
    backgroundColor: colors.primary,
    borderRadius: radius.pill,
    paddingVertical: 16,
    alignItems: 'center',
  },
  ctaText: { ...typography.bodyBold, color: colors.bg },
  secondary: {
    borderRadius: radius.pill,
    paddingVertical: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  secondaryText: { ...typography.bodyBold, color: colors.primary },
});