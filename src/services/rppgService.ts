import { NativeModules, Platform } from 'react-native';

const { RppgModule } = NativeModules;

export type VerdictTier = 'VERIFIED' | 'RETRY_REQUIRED' | 'INCONCLUSIVE' | 'FAILED';

export type RppgStatus =
  | 'CONFIRMED_PULSE'
  | 'INCONCLUSIVE_WEAK_SIGNAL'
  | 'INCONCLUSIVE_MOTION'
  | 'INCONCLUSIVE_NO_FACE'
  | 'INCONCLUSIVE_TOO_SHORT'
  | 'RETRY_REQUIRED'
  | 'FAILED'
  | 'ERROR';

export type SignalQualityTier = 'HIGH' | 'MODERATE' | 'LOW' | 'UNRELIABLE';

export type ChallengeType =
  | 'LOOK_CENTER'
  | 'BLINK'
  | 'SMILE'
  | 'TURN_LEFT'
  | 'TURN_RIGHT'
  | 'TILT_UP';

export interface LivenessChallenge {
  id: string;
  type: ChallengeType;
  instruction: string;
  startTimeSec: number;
  endTimeSec: number;
}

export interface ChallengeVerificationResult {
  id: string;
  type: string;
  instruction: string;
  verified: boolean;
  confidence: number;
  detail: string;
}

export interface ProcessingMetrics {
  totalDurationMs: number;
  frameExtractionMs: number;
  landmarkDetectionMs: number;
  skinExtractionMs: number;
  challengeDetectionMs: number;
  dspMs: number;
  fpsSampled: number;
}

export interface RppgResult {
  success: boolean;
  verdictTier: VerdictTier;
  status: RppgStatus;
  estimatedBpm: number;
  signalQuality: SignalQualityTier;
  usableFrames: number;
  totalFrames: number;
  faceRatio: number;
  durationSeconds: number;
  snrDb: number;
  peakPowerFraction: number;
  isLikelyLiveness: boolean;
  message: string;
  // Guided Challenge Extensions
  challenges?: ChallengeVerificationResult[];
  challengesPassed?: number;
  challengesTotal?: number;
  challengeLivenessPass?: boolean;
  processingMetrics?: ProcessingMetrics;
}

class RppgService {
  /**
   * Checks whether the native Android rPPG module is available in this runtime.
   */
  isAvailable(): boolean {
    return Platform.OS === 'android' && Boolean(RppgModule && (RppgModule.processVideo || RppgModule.processVideoWithChallenges));
  }

  /**
   * Processes a locally recorded video file via on-device MediaPipe & POS rPPG pipeline.
   * Runs completely offline with no network or server dependencies.
   *
   * @param videoUri Local file:// or content:// URI of the recorded video
   * @param challenges Optional list of guided challenges scheduled during recording
   */
  async processVideo(
    videoUri: string,
    challenges?: LivenessChallenge[]
  ): Promise<RppgResult> {
    if (!this.isAvailable()) {
      return {
        success: false,
        verdictTier: 'RETRY_REQUIRED',
        status: 'ERROR',
        estimatedBpm: 0,
        signalQuality: 'UNRELIABLE',
        usableFrames: 0,
        totalFrames: 0,
        faceRatio: 0,
        durationSeconds: 0,
        snrDb: 0,
        peakPowerFraction: 0,
        isLikelyLiveness: false,
        message: 'Native rPPG module is only available on Android native builds.',
      };
    }

    try {
      let result: RppgResult;
      if (challenges && challenges.length > 0 && RppgModule.processVideoWithChallenges) {
        const jsonChallenges = JSON.stringify(challenges);
        result = await RppgModule.processVideoWithChallenges(videoUri, jsonChallenges);
      } else {
        result = await RppgModule.processVideo(videoUri);
      }
      return result;
    } catch (error: any) {
      return {
        success: false,
        verdictTier: 'RETRY_REQUIRED',
        status: 'ERROR',
        estimatedBpm: 0,
        signalQuality: 'UNRELIABLE',
        usableFrames: 0,
        totalFrames: 0,
        faceRatio: 0,
        durationSeconds: 0,
        snrDb: 0,
        peakPowerFraction: 0,
        isLikelyLiveness: false,
        message: error?.message || 'Unknown native processing failure',
      };
    }
  }

  /**
   * Safely deletes the temporary video file from local app cache.
   * Protects user privacy by ensuring biometric video recordings do not persist.
   */
  async deleteVideoFile(videoUri: string): Promise<boolean> {
    if (!this.isAvailable()) return false;
    try {
      return await RppgModule.deleteVideoFile(videoUri);
    } catch {
      return false;
    }
  }
}

export const rppgService = new RppgService();
