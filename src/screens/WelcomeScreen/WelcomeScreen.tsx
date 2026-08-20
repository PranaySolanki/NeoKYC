import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, spacing, radius, typography } from '../theme';

type Props = {
  onGetStarted: () => void;
};

export default function WelcomeScreen({ onGetStarted }: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>NeoKYC</Text>
        <Text style={styles.subtitle}>
          Zero-trust identity verification. No data stored. Ever.
        </Text>
      </View>

      <Pressable style={styles.cta} onPress={onGetStarted}>
        <Text style={styles.ctaText}>Start Verification</Text>
      </Pressable>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: spacing.lg },
  content: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  title: { ...typography.h1, color: colors.textPrimary, marginBottom: spacing.sm },
  subtitle: { ...typography.body, color: colors.textSecondary, textAlign: 'center' },
  cta: {
    backgroundColor: colors.primary,
    borderRadius: radius.pill,
    paddingVertical: 16,
    alignItems: 'center',
  },
  ctaText: { ...typography.bodyBold, color: colors.bg },
});