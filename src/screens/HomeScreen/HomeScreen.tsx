import React from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, spacing, radius, typography } from '../theme';

type ActivityItem = {
  id: string;
  title: string;
  timestamp: string;
  status: 'verified' | 'rejected' | 'pending';
};

type Props = {
  userName: string;
  verifiedAt?: string; // e.g. "Today, 10:42 AM"
  activity: ActivityItem[];
  onStartVerification: () => void;
  onViewCertificate: () => void;
};

const STATUS_META: Record<ActivityItem['status'], { color: string; label: string }> = {
  verified: { color: colors.success, label: 'Verified' },
  rejected: { color: colors.danger, label: 'Rejected' },
  pending: { color: colors.warning, label: 'Pending' },
};

export default function HomeScreen({
  userName,
  verifiedAt,
  activity,
  onStartVerification,
  onViewCertificate,
}: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <Text style={styles.greeting}>Welcome back,</Text>
        <Text style={styles.name}>{userName}</Text>

        <View style={styles.statusCard}>
          <View style={styles.statusIconWrap}>
            <Text style={styles.statusIcon}>{verifiedAt ? '🛡️' : '⏳'}</Text>
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.statusTitle}>
              {verifiedAt ? 'Identity Verified' : 'Not Verified Yet'}
            </Text>
            <Text style={styles.statusSub}>
              {verifiedAt ? `Last verified ${verifiedAt}` : 'Complete your first check to unlock access'}
            </Text>
          </View>
        </View>

        <View style={styles.actionsRow}>
          <Pressable style={styles.primaryAction} onPress={onStartVerification}>
            <Text style={styles.primaryActionIcon}>🪪</Text>
            <Text style={styles.primaryActionText}>New Verification</Text>
          </Pressable>
          <Pressable
            style={[styles.primaryAction, !verifiedAt && styles.disabledAction]}
            onPress={verifiedAt ? onViewCertificate : undefined}
          >
            <Text style={styles.primaryActionIcon}>📄</Text>
            <Text style={styles.primaryActionText}>View Certificate</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>Recent Activity</Text>

        {activity.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>No verification history yet.</Text>
          </View>
        ) : (
          <View style={{ gap: spacing.sm }}>
            {activity.map((item) => {
              const meta = STATUS_META[item.status];
              return (
                <View key={item.id} style={styles.activityRow}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.activityTitle}>{item.title}</Text>
                    <Text style={styles.activityTime}>{item.timestamp}</Text>
                  </View>
                  <View style={[styles.statusPill, { borderColor: meta.color }]}>
                    <Text style={[styles.statusPillText, { color: meta.color }]}>{meta.label}</Text>
                  </View>
                </View>
              );
            })}
          </View>
        )}

        <View style={styles.privacyNote}>
          <Text style={styles.privacyNoteText}>
            🔒 Your verification data is processed in-memory only and is never stored as raw
            images or video.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.lg, paddingBottom: spacing.xxl },
  greeting: { ...typography.body, color: colors.textSecondary },
  name: { ...typography.h1, color: colors.textPrimary, marginBottom: spacing.lg },
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.bgElevated,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  statusIconWrap: {
    width: 48,
    height: 48,
    borderRadius: radius.sm,
    backgroundColor: colors.primaryDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusIcon: { fontSize: 22 },
  statusTitle: { ...typography.bodyBold, color: colors.textPrimary },
  statusSub: { ...typography.caption, color: colors.textSecondary, marginTop: 2 },
  actionsRow: { flexDirection: 'row', gap: spacing.md, marginBottom: spacing.xl },
  primaryAction: {
    flex: 1,
    backgroundColor: colors.bgElevated,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingVertical: spacing.lg,
    alignItems: 'center',
    gap: spacing.xs,
  },
  disabledAction: { opacity: 0.4 },
  primaryActionIcon: { fontSize: 24 },
  primaryActionText: { ...typography.bodyBold, color: colors.textPrimary, fontSize: 13, textAlign: 'center' },
  sectionTitle: { ...typography.h2, color: colors.textPrimary, marginBottom: spacing.md },
  emptyState: {
    backgroundColor: colors.bgElevated,
    borderRadius: radius.md,
    padding: spacing.lg,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  emptyStateText: { ...typography.body, color: colors.textMuted },
  activityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.bgElevated,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  activityTitle: { ...typography.bodyBold, color: colors.textPrimary },
  activityTime: { ...typography.caption, color: colors.textSecondary, marginTop: 2 },
  statusPill: {
    borderWidth: 1,
    borderRadius: radius.pill,
    paddingVertical: 4,
    paddingHorizontal: spacing.sm,
  },
  statusPillText: { ...typography.caption, fontWeight: '700' },
  privacyNote: {
    marginTop: spacing.xl,
    padding: spacing.md,
    borderRadius: radius.md,
    backgroundColor: colors.bgGlass,
  },
  privacyNoteText: { ...typography.caption, color: colors.textSecondary, lineHeight: 18 },
});