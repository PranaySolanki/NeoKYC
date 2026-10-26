import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView } from 'react-native';
import { RppgResult, LivenessVerdict } from '../../services/rppgService';

interface ResultScreenProps {
  livenessResult?: RppgResult | null;
  onNavigate?: (screenName: string) => void;
}

interface VerdictConfig {
  title: string;
  description: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
  icon: string;
}

export const ResultScreen: React.FC<ResultScreenProps> = ({ livenessResult, onNavigate }) => {
  // Determine final verdict
  const resolveVerdictConfig = (): VerdictConfig => {
    const verdict = livenessResult?.verdict;

    switch (verdict) {
      case 'REAL_FACE_VERIFIED':
        return {
          title: 'REAL FACE VERIFIED',
          description:
            livenessResult?.verdictDescription ||
            'Sufficient evidence supports a live human face.',
          textColor: '#10B981',
          bgColor: 'rgba(16, 185, 129, 0.12)',
          borderColor: '#10B981',
          icon: '✓',
        };

      case 'SPOOF_DETECTED':
        return {
          title: 'SPOOF / FAKE FACE DETECTED',
          description:
            livenessResult?.verdictDescription ||
            'Reliable detection evidence indicates a potential spoof (phone replay, printed image, or mask).',
          textColor: '#EF4444',
          bgColor: 'rgba(239, 68, 68, 0.12)',
          borderColor: '#EF4444',
          icon: '✕',
        };

      case 'VERIFICATION_FAILED':
        return {
          title: 'VERIFICATION FAILED — TRY AGAIN',
          description:
            livenessResult?.verdictDescription ||
            'The capture was incomplete, unclear, or technically unsuccessful.',
          textColor: '#F59E0B',
          bgColor: 'rgba(245, 158, 11, 0.12)',
          borderColor: '#F59E0B',
          icon: '!',
        };

      case 'INCONCLUSIVE':
      default:
        // Fallback for older results or unhandled states
        if (livenessResult?.success && livenessResult?.isLikelyLiveness) {
          return {
            title: 'REAL FACE VERIFIED',
            description: 'Sufficient evidence supports a live human face.',
            textColor: '#10B981',
            bgColor: 'rgba(16, 185, 129, 0.12)',
            borderColor: '#10B981',
            icon: '✓',
          };
        }
        return {
          title: 'INCONCLUSIVE — TRY AGAIN',
          description:
            livenessResult?.verdictDescription ||
            'Available evidence is insufficient to make a reliable decision.',
          textColor: '#94A3B8',
          bgColor: 'rgba(148, 163, 184, 0.12)',
          borderColor: '#64748B',
          icon: '?',
        };
    }
  };

  const verdictConfig = resolveVerdictConfig();
  const challenges = livenessResult?.challenges || [];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Liveness Verification</Text>
        <Text style={styles.subtitle}>Face authenticity & spoof assessment</Text>
      </View>

      {/* 1. Final Verdict Card */}
      <View
        style={[
          styles.verdictCard,
          {
            backgroundColor: verdictConfig.bgColor,
            borderColor: verdictConfig.borderColor,
          },
        ]}
      >
        <View style={styles.verdictHeader}>
          <View
            style={[
              styles.verdictIconCircle,
              { borderColor: verdictConfig.borderColor, backgroundColor: verdictConfig.bgColor },
            ]}
          >
            <Text style={[styles.verdictIconText, { color: verdictConfig.textColor }]}>
              {verdictConfig.icon}
            </Text>
          </View>
          <Text style={[styles.verdictTitle, { color: verdictConfig.textColor }]}>
            {verdictConfig.title}
          </Text>
        </View>
        <Text style={styles.verdictDescription}>{verdictConfig.description}</Text>
      </View>

      {/* 2. Gesture Checks Card */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>Gesture Checks</Text>
          {challenges.length > 0 && (
            <Text style={styles.counterText}>
              {challenges.filter((c) => c.verified).length} / {challenges.length} Passed
            </Text>
          )}
        </View>

        {challenges.length > 0 ? (
          <View style={styles.gestureList}>
            {challenges.map((ch, idx) => {
              const isChecked = Boolean(ch.verified);
              return (
                <View key={ch.id || idx} style={styles.gestureItem}>
                  {/* Clean Checkbox Indicator */}
                  <View
                    style={[
                      styles.checkbox,
                      isChecked ? styles.checkboxChecked : styles.checkboxUnchecked,
                    ]}
                  >
                    {isChecked && <Text style={styles.checkboxCheckmark}>✓</Text>}
                  </View>

                  {/* Gesture Label */}
                  <View style={styles.gestureContent}>
                    <Text
                      style={[
                        styles.gestureName,
                        !isChecked && styles.gestureNameIncomplete,
                      ]}
                    >
                      {ch.instruction}
                    </Text>
                  </View>

                  {/* Status Indicator */}
                  <Text
                    style={[
                      styles.gestureStatus,
                      isChecked ? styles.statusCompleted : styles.statusIncomplete,
                    ]}
                  >
                    {isChecked ? 'Completed' : 'Not completed'}
                  </Text>
                </View>
              );
            })}
          </View>
        ) : (
          <Text style={styles.emptyText}>
            No gesture checks recorded. Complete the liveness scan to verify facial gestures.
          </Text>
        )}
      </View>

      {/* Action Buttons */}
      {onNavigate && (
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => onNavigate('FaceLiveness')}
            activeOpacity={0.8}
          >
            <Text style={styles.primaryButtonText}>Re-test Liveness</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={() => onNavigate('Home')}
            activeOpacity={0.8}
          >
            <Text style={styles.secondaryButtonText}>Return Home</Text>
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
  title: {
    color: '#F8FAFC',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: 0.3,
    marginBottom: 4,
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 14,
  },
  verdictCard: {
    borderRadius: 16,
    padding: 20,
    borderWidth: 1.5,
    marginBottom: 20,
  },
  verdictHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  verdictIconCircle: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  verdictIconText: {
    fontSize: 18,
    fontWeight: '900',
  },
  verdictTitle: {
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 0.5,
    flex: 1,
  },
  verdictDescription: {
    color: '#E2E8F0',
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '500',
  },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 24,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  cardTitle: {
    color: '#F8FAFC',
    fontSize: 16,
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
    borderRadius: 12,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  gestureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
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
    fontSize: 13,
    fontWeight: '900',
  },
  gestureContent: {
    flex: 1,
    marginRight: 8,
  },
  gestureName: {
    color: '#F8FAFC',
    fontSize: 14,
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
    color: '#64748B',
    fontSize: 13,
    fontStyle: 'italic',
    paddingVertical: 8,
  },
  buttonRow: {
    gap: 12,
  },
  primaryButton: {
    backgroundColor: '#0EA5E9',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#94A3B8',
    fontSize: 14,
    fontWeight: '600',
  },
});
