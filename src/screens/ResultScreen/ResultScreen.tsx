import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import * as Crypto from 'expo-crypto';
import { supabase } from '../../services/supabase';
import { LivenessVerificationResult } from '../../services/experimentalRppg';

export interface ResultScreenProps {
  onNavigate?: (screenName: string, params?: Record<string, any>) => void;
  applicantData?: {
    name: string | null;
    dob: string | null;
    gender: string | null;
    id_number: string | null;
  };
  idSha256?: string;
  liveVector?: number[];
  livenessResult?: LivenessVerificationResult | null;
  pulseBpm?: number | null;
}

// Biometric cosine similarity threshold (0.50 = 50% match)
const MATCH_THRESHOLD = 0.50;

export const ResultScreen: React.FC<ResultScreenProps> = ({
  onNavigate,
  applicantData,
  idSha256: passedIdSha256,
  liveVector,
  livenessResult,
  pulseBpm,
}) => {
  const [loading, setLoading] = useState(true);
  const [dbRecord, setDbRecord] = useState<{
    full_name: string | null;
    date_of_birth: string | null;
    gender: string | null;
  } | null>(null);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [isMatch, setIsMatch] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [resolvedHash, setResolvedHash] = useState<string | null>(null);

  useEffect(() => {
    async function verifyAndCompare() {
      try {
        setLoading(true);
        setErrorMessage(null);

        // 1. Resolve ID SHA-256 hash (fallback to hashing applicantData.id_number)
        let effectiveHash = passedIdSha256;
        if (!effectiveHash && applicantData?.id_number) {
          effectiveHash = await Crypto.digestStringAsync(
            Crypto.CryptoDigestAlgorithm.SHA256,
            applicantData.id_number
          );
        }

        if (!effectiveHash) {
          throw new Error('ID number / SHA-256 hash is missing. Cannot lookup enrolled identity.');
        }

        setResolvedHash(effectiveHash);

        if (!liveVector || liveVector.length !== 512) {
          throw new Error('Live face biometric vector is missing or incomplete (expected 512-D).');
        }

        // 2. Fetch enrolled face coordinates from Supabase
        const { data, error } = await supabase
          .from('person_entries')
          .select('full_name, date_of_birth, gender, face_coordinates')
          .eq('aadhaar_number', effectiveHash)
          .single();

        if (error || !data) {
          throw new Error(
            error?.message || 'No enrolled record found for this ID in database. Please register face first.'
          );
        }

        setDbRecord({
          full_name: data.full_name,
          date_of_birth: data.date_of_birth,
          gender: data.gender,
        });

        const enrolledCoords: number[] = data.face_coordinates;
        if (!enrolledCoords || enrolledCoords.length !== 512) {
          throw new Error('Enrolled biometric coordinates in database are corrupted or invalid.');
        }

        // 3. Compute Cosine Similarity: Dot Product / (||A|| * ||B||)
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < 512; i++) {
          const a = liveVector[i];
          const b = enrolledCoords[i];
          dotProduct += a * b;
          normA += a * a;
          normB += b * b;
        }

        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        const score = denominator > 0 ? dotProduct / denominator : 0;
        const clampedScore = Math.max(0, Math.min(1, score));

        setSimilarity(clampedScore);
        setIsMatch(clampedScore >= MATCH_THRESHOLD);
      } catch (err: any) {
        console.error('[ResultScreen] Verification Error:', err);
        setErrorMessage(err.message || 'An unexpected error occurred during database lookup.');
      } finally {
        setLoading(false);
      }
    }

    verifyAndCompare();
  }, [passedIdSha256, applicantData?.id_number, liveVector]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#38BDF8" />
        <Text style={styles.loadingTitle}>Comparing Facial Signatures</Text>
        <Text style={styles.loadingSubtitle}>
          Querying enrolled record from Supabase & running ArcFace 512-D cosine evaluation...
        </Text>
      </View>
    );
  }

  const similarityPercent = similarity !== null ? Math.round(similarity * 100) : 0;
  const livenessPassed = livenessResult?.livenessPassed ?? false;
  const overallSuccess = livenessPassed && isMatch && !errorMessage;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.container}>
      {/* Header Bar */}
      <View style={styles.headerBar}>
        <TouchableOpacity style={styles.backButton} onPress={() => onNavigate?.('HomeDash')}>
          <Text style={styles.backButtonText}>← Home</Text>
        </TouchableOpacity>
        <Text style={styles.title}>KYC Verification</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Primary Result Banner */}
      <View
        style={[
          styles.verdictCard,
          overallSuccess ? styles.verdictCardPass : styles.verdictCardFail,
        ]}
      >
        <Text style={styles.verdictIcon}>{overallSuccess ? '✓' : '✕'}</Text>
        <Text style={styles.verdictTitle}>
          {overallSuccess ? 'Identity Verified Successfully' : 'Verification Unsuccessful'}
        </Text>
        <Text style={styles.verdictSubtitle}>
          {overallSuccess
            ? 'Live biometric facial signature matches enrolled identity database.'
            : errorMessage ?? 'Biometric facial coordinates do not meet the minimum match threshold.'}
        </Text>
      </View>

      {/* Biometric Comparison Metrics Card */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>ArcFace Biometric Match</Text>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>Similarity Score:</Text>
          <Text
            style={[
              styles.metricValue,
              isMatch ? styles.colorSuccess : styles.colorDanger,
            ]}
          >
            {similarityPercent}% {isMatch ? '(Match)' : '(Mismatch)'}
          </Text>
        </View>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>Required Threshold:</Text>
          <Text style={styles.metricValue}>≥ {Math.round(MATCH_THRESHOLD * 100)}%</Text>
        </View>

        <View style={styles.progressBarBackground}>
          <View
            style={[
              styles.progressBarFill,
              { width: `${similarityPercent}%` },
              isMatch ? styles.progressSuccess : styles.progressDanger,
            ]}
          />
        </View>
      </View>

      {/* Liveness & Vitals Telemetry Card */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Liveness & Vital Telemetry</Text>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>Passive Liveness:</Text>
          <Text style={[styles.metricValue, livenessPassed ? styles.colorSuccess : styles.colorDanger]}>
            {livenessPassed ? 'PASSED (Real Human)' : 'FAILED'}
          </Text>
        </View>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>Challenge Score:</Text>
          <Text style={styles.metricValue}>
            {Math.round((livenessResult?.livenessScore ?? 0) * 100)}%
          </Text>
        </View>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>Estimated Heart Rate:</Text>
          <Text style={styles.metricValue}>
            {pulseBpm ? `${pulseBpm.toFixed(0)} BPM` : 'Inconclusive'}
          </Text>
        </View>

        <View style={styles.metricRow}>
          <Text style={styles.metricLabel}>rPPG Signal Quality:</Text>
          <Text style={styles.metricValue}>
            {Math.round((livenessResult?.rppgTelemetry.signalQualityScore ?? 0) * 100)}%
          </Text>
        </View>
      </View>

      {/* Identity Demographic Details */}
      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Identity Details</Text>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Name:</Text>
          <Text style={styles.infoValue}>
            {applicantData?.name || dbRecord?.full_name || 'N/A'}
          </Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Date of Birth:</Text>
          <Text style={styles.infoValue}>
            {applicantData?.dob || dbRecord?.date_of_birth || 'N/A'}
          </Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Gender:</Text>
          <Text style={styles.infoValue}>
            {applicantData?.gender || dbRecord?.gender || 'N/A'}
          </Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>ID Hash (SHA-256):</Text>
          <Text style={styles.infoValueHash} numberOfLines={1} ellipsizeMode="middle">
            {resolvedHash || 'N/A'}
          </Text>
        </View>
      </View>

      {/* Actions */}
      <View style={styles.actionContainer}>
        {!overallSuccess ? (
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => onNavigate?.('FaceLivenessRPPG', { applicantData, idSha256: resolvedHash ?? undefined })}
          >
            <Text style={styles.primaryButtonText}>Retry Verification</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => onNavigate?.('HomeDash')}
          >
            <Text style={styles.primaryButtonText}>Complete & Proceed</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => onNavigate?.('HomeDash')}
        >
          <Text style={styles.secondaryButtonText}>Return to Home Dashboard</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#0F172A' },
  container: {
    flexGrow: 1,
    backgroundColor: '#0F172A',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 32,
  },
  center: {
    flex: 1,
    backgroundColor: '#0F172A',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loadingTitle: {
    color: '#F8FAFC',
    fontSize: 18,
    fontWeight: '800',
    marginTop: 18,
  },
  loadingSubtitle: {
    color: '#94A3B8',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 18,
    maxWidth: 280,
  },
  headerBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 16,
  },
  backButton: {
    paddingVertical: 7,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 8,
  },
  backButtonText: { color: '#38BDF8', fontSize: 14, fontWeight: '600' },
  headerSpacer: { width: 60 },
  title: { color: '#F8FAFC', fontSize: 20, fontWeight: '800' },
  verdictCard: {
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 18,
    borderWidth: 1,
  },
  verdictCardPass: {
    backgroundColor: 'rgba(6, 78, 59, 0.45)',
    borderColor: '#10B981',
  },
  verdictCardFail: {
    backgroundColor: 'rgba(127, 29, 29, 0.45)',
    borderColor: '#EF4444',
  },
  verdictIcon: { fontSize: 36, color: '#F8FAFC', marginBottom: 4 },
  verdictTitle: {
    color: '#F8FAFC',
    fontSize: 19,
    fontWeight: '800',
    textAlign: 'center',
  },
  verdictSubtitle: {
    color: '#CBD5E1',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 6,
    lineHeight: 18,
  },
  sectionCard: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 14,
  },
  sectionTitle: {
    color: '#38BDF8',
    fontSize: 15,
    fontWeight: '800',
    marginBottom: 14,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  metricLabel: { color: '#94A3B8', fontSize: 13, fontWeight: '600' },
  metricValue: { color: '#F8FAFC', fontSize: 14, fontWeight: '800' },
  colorSuccess: { color: '#2DD4BF' },
  colorDanger: { color: '#FCA5A5' },
  progressBarBackground: {
    height: 8,
    backgroundColor: '#334155',
    borderRadius: 4,
    overflow: 'hidden',
    marginTop: 4,
  },
  progressBarFill: { height: 8, borderRadius: 4 },
  progressSuccess: { backgroundColor: '#2DD4BF' },
  progressDanger: { backgroundColor: '#EF4444' },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderColor: 'rgba(51, 65, 85, 0.4)',
  },
  infoLabel: { color: '#94A3B8', fontSize: 13, fontWeight: '600' },
  infoValue: { color: '#CBD5E1', fontSize: 13, fontWeight: '700' },
  infoValueHash: {
    color: '#CBD5E1',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    maxWidth: 180,
  },
  actionContainer: { marginTop: 10, gap: 8 },
  primaryButton: {
    backgroundColor: '#0284C7',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },
  secondaryButton: { paddingVertical: 12, alignItems: 'center' },
  secondaryButtonText: { color: '#94A3B8', fontSize: 14, fontWeight: '700' },
});

export default ResultScreen;