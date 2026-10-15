import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView } from 'react-native';
import { RppgResult } from '../../services/rppgService';

interface ResultScreenProps {
  livenessResult?: RppgResult | null;
  onNavigate?: (screenName: string) => void;
}

export const ResultScreen: React.FC<ResultScreenProps> = ({ livenessResult, onNavigate }) => {
  const isPulsePass = livenessResult?.status === 'CONFIRMED_PULSE';
  const isChallengesPass = Boolean(livenessResult?.challengeLivenessPass);
  const isFullyVerified = livenessResult?.verdictTier === 'VERIFIED';
  const isRetryRequired = livenessResult?.verdictTier === 'RETRY_REQUIRED';
  const isFailed = livenessResult?.verdictTier === 'FAILED';
  const isInconclusive = livenessResult?.verdictTier === 'INCONCLUSIVE';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Top Header */}
      <View style={styles.header}>

        <Text style={styles.title}>Verification Result</Text>
        <Text style={styles.subtitle}>
          Zero-Trust Multi-Factor Identity & Liveness Assessment
        </Text>
      </View>

      {/* 1. Challenge-Based Liveness Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>1. Guided Action Challenges</Text>
          <View
            style={[
              styles.badge,
              isFullyVerified || isChallengesPass
                ? styles.badgeSuccess
                : isRetryRequired || (livenessResult?.challenges && livenessResult.challenges.length > 0)
                  ? styles.badgeWarning
                  : styles.badgeNeutral,
            ]}
          >
            <Text
              style={[
                styles.badgeText,
                isFullyVerified || isChallengesPass
                  ? styles.badgeTextSuccess
                  : isRetryRequired || (livenessResult?.challenges && livenessResult.challenges.length > 0)
                    ? styles.badgeTextWarning
                    : styles.badgeTextNeutral,
              ]}
            >
              {isRetryRequired
                ? 'RETRY REQUIRED'
                : isChallengesPass
                ? `PASSED (${livenessResult?.challengesPassed}/${livenessResult?.challengesTotal})`
                : livenessResult?.challenges && livenessResult.challenges.length > 0
                  ? `UNCONFIRMED (${livenessResult?.challengesPassed}/${livenessResult?.challengesTotal})`
                  : 'PENDING'}
            </Text>
          </View>
        </View>

        {livenessResult?.challenges && livenessResult.challenges.length > 0 ? (
          <View style={styles.challengeList}>
            {livenessResult.challenges.map((ch, idx) => (
              <View key={ch.id || idx} style={styles.challengeItem}>
                <Text style={styles.challengeIcon}>{ch.verified ? '✅' : '⚠️'}</Text>
                <View style={styles.challengeBody}>
                  <Text style={styles.challengeName}>{ch.instruction}</Text>
                  <Text
                    style={[
                      styles.challengeDetail,
                      ch.verified ? styles.textVerified : styles.textUnconfirmed,
                    ]}
                  >
                    {isRetryRequired
                      ? 'Face was not clearly detected to track movements'
                      : ch.detail}
                  </Text>
                </View>
                <Text
                  style={[
                    styles.challengeStatusBadge,
                    ch.verified ? styles.textVerified : styles.textUnconfirmed,
                  ]}
                >
                  {ch.verified ? 'VERIFIED' : 'UNCONFIRMED'}
                </Text>
              </View>
            ))}
          </View>
        ) : (
          <Text style={styles.pendingText}>
            No challenge sequence has been completed yet. Complete the guided liveness check to
            verify facial movements.
          </Text>
        )}
      </View>

      {/* 2. Physiological Pulse (rPPG) Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>2. Physiological Pulse (rPPG)</Text>
          <View
            style={[
              styles.badge,
              isPulsePass
                ? styles.badgeSuccess
                : livenessResult
                  ? styles.badgeWarning
                  : styles.badgeNeutral,
            ]}
          >
            <Text
              style={[
                styles.badgeText,
                isPulsePass
                  ? styles.badgeTextSuccess
                  : livenessResult
                    ? styles.badgeTextWarning
                    : styles.badgeTextNeutral,
              ]}
            >
              {isPulsePass
                ? 'PASSED'
                : isRetryRequired
                  ? 'RETRY REQUIRED'
                  : livenessResult
                    ? livenessResult.status.replace('INCONCLUSIVE_', '')
                    : 'PENDING'}
            </Text>
          </View>
        </View>

        {livenessResult ? (
          <>
            <Text style={styles.cardMessage}>{livenessResult.message}</Text>
            <View style={styles.metaGrid}>
              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>ESTIMATED PULSE</Text>
                <Text style={styles.metaValue}>
                  {livenessResult.estimatedBpm > 0
                    ? `${livenessResult.estimatedBpm.toFixed(1)} BPM`
                    : 'N/A'}
                </Text>
              </View>
              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>SIGNAL QUALITY</Text>
                <Text style={styles.metaValue}>{livenessResult.signalQuality}</Text>
              </View>
              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>TRACKED FRAMES</Text>
                <Text style={styles.metaValue}>
                  {livenessResult.usableFrames} / {livenessResult.totalFrames}
                </Text>
              </View>
              <View style={styles.metaItem}>
                <Text style={styles.metaLabel}>SNR RATIO</Text>
                <Text style={styles.metaValue}>{livenessResult.snrDb.toFixed(1)} dB</Text>
              </View>
              {livenessResult.processingMetrics && (
                <View style={[styles.metaItem, { flexBasis: '100%' }]}>
                  <Text style={styles.metaLabel}>PROCESSING DURATION</Text>
                  <Text style={styles.metaValue}>
                    {(livenessResult.processingMetrics.totalDurationMs / 1000).toFixed(2)}s @ {livenessResult.processingMetrics.fpsSampled} FPS
                  </Text>
                </View>
              )}
            </View>
          </>
        ) : (
          <Text style={styles.pendingText}>
            Liveness check has not been completed yet. Run the Liveness scan to measure
            subcutaneous pulse signals.
          </Text>
        )}
      </View>

      {/* 3. Zero-Trust Multi-Factor Decision Engine */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>3. Zero-Trust Risk Engine Verdict</Text>

        <View style={styles.riskRow}>
          <Text style={styles.riskLabel}>Overall Assurance Tier:</Text>
          <Text
            style={[
              styles.riskValue,
              isFullyVerified
                ? styles.riskLow
                : isFailed
                  ? styles.riskHigh
                  : styles.riskMedium,
            ]}
          >
            {isFullyVerified
              ? 'VERIFIED (LOW RISK)'
              : isRetryRequired
                ? 'RETRY REQUIRED'
                : isInconclusive
                  ? 'INCONCLUSIVE'
                  : isFailed
                    ? 'FAILED (HIGH RISK)'
                    : 'PENDING'}
          </Text>
        </View>
      </View>

      {/* Action Buttons */}
      {onNavigate && (
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={isRetryRequired ? styles.primaryButton : styles.secondaryButton}
            onPress={() => onNavigate('FaceLiveness')}
          >
            <Text style={isRetryRequired ? styles.primaryButtonText : styles.secondaryButtonText}>
              {isRetryRequired ? 'Try Again' : 'Re-test Liveness'}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={isRetryRequired ? styles.secondaryButton : styles.primaryButton}
            onPress={() => onNavigate('Home')}
          >
            <Text style={isRetryRequired ? styles.secondaryButtonText : styles.primaryButtonText}>
              Return Home
            </Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  contentContainer: {
    paddingTop: 85,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 20,
  },
  backButton: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E293B',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  backButtonText: {
    color: '#38BDF8',
    fontSize: 13,
    fontWeight: '600',
  },
  title: {
    color: '#F8FAFC',
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 4,
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 13,
  },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  cardTitle: {
    color: '#F8FAFC',
    fontSize: 15,
    fontWeight: '700',
  },
  cardMessage: {
    color: '#94A3B8',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 14,
  },
  pendingText: {
    color: '#64748B',
    fontSize: 13,
    fontStyle: 'italic',
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
  badgeNeutral: {
    backgroundColor: 'rgba(100, 116, 139, 0.15)',
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
  badgeTextNeutral: {
    color: '#94A3B8',
  },
  challengeList: {
    backgroundColor: '#0F172A',
    borderRadius: 10,
    padding: 10,
    gap: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  challengeItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  challengeIcon: {
    fontSize: 15,
    marginRight: 10,
  },
  challengeBody: {
    flex: 1,
  },
  challengeName: {
    color: '#F8FAFC',
    fontSize: 13,
    fontWeight: '600',
  },
  challengeDetail: {
    fontSize: 11,
    marginTop: 1,
  },
  challengeStatusBadge: {
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
  metaGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  metaItem: {
    flexBasis: '48%',
    backgroundColor: '#0F172A',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  metaLabel: {
    color: '#64748B',
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.5,
    marginBottom: 3,
  },
  metaValue: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '700',
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
    paddingTop: 10,
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
  riskLow: {
    color: '#10B981',
  },
  riskMedium: {
    color: '#F59E0B',
  },
  riskHigh: {
    color: '#EF4444',
  },
  explanationText: {
    color: '#64748B',
    fontSize: 12,
    lineHeight: 16,
    marginTop: 8,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
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
    fontSize: 14,
    fontWeight: '700',
  },
  primaryButton: {
    flex: 1,
    backgroundColor: '#0EA5E9',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
});