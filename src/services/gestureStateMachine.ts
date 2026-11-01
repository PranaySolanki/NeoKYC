export type GestureChallenge = 'BLINK' | 'SMILE' | 'TURN_LEFT' | 'TURN_RIGHT' | 'LOOK_UP';
export type LivenessFailureReason = 'MULTIPLE_FACES_DETECTED' | 'FACE_OUT_OF_BOUNDS' | 'FACE_SWAP_DETECTED' | 'INCOMPLETE_GESTURES';

const GESTURE_POOL: readonly GestureChallenge[] = ['BLINK', 'SMILE', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP'];

export function pickGestureChallenges(random: () => number = Math.random): GestureChallenge[] {
  const pool = [...GESTURE_POOL];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, 4);
}

export interface GestureMetrics {
  timestampMs: number;
  yaw: number;
  pitch: number;
  smile: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
}

export interface GestureHoldState {
  activeSinceMs: number | null;
  lastActiveFrameMs: number | null;
  blinkClosed: boolean;
}

export interface GestureEvaluation {
  completed: boolean;
  holdMs: number;
}

const MAX_CONTIGUOUS_FRAME_GAP_MS = 120;
const GUIDE = { xMin: 0.15, xMax: 0.85, yMin: 0.2, yMax: 0.75 };

export function createGestureHoldState(): GestureHoldState {
  return { activeSinceMs: null, lastActiveFrameMs: null, blinkClosed: false };
}

/** Evaluate one timestamped ML Kit face observation and update its hold accumulator. */
export function evaluateGestureFrame(
  challenge: GestureChallenge,
  metrics: GestureMetrics,
  state: GestureHoldState,
  requiredHoldMs = 300,
): GestureEvaluation {
  const eyesClosed = metrics.leftEyeOpen < 0.25 && metrics.rightEyeOpen < 0.25;
  const eyesOpen = metrics.leftEyeOpen > 0.75 && metrics.rightEyeOpen > 0.75;
  if (eyesClosed) state.blinkClosed = true;

  let conditionMet = false;
  if (challenge === 'BLINK') conditionMet = state.blinkClosed && eyesOpen;
  else if (challenge === 'SMILE') conditionMet = metrics.smile > 0.70;
  else if (challenge === 'TURN_LEFT') conditionMet = metrics.yaw > 20;
  else if (challenge === 'TURN_RIGHT') conditionMet = metrics.yaw < -20;
  else if (challenge === 'LOOK_UP') conditionMet = metrics.pitch > 15;

  if (!conditionMet) {
    state.activeSinceMs = null;
    state.lastActiveFrameMs = null;
    if (challenge === 'BLINK' && eyesOpen) state.blinkClosed = false;
    return { completed: false, holdMs: 0 };
  }

  const timestamp = metrics.timestampMs;
  const frameGap = state.lastActiveFrameMs === null ? 0 : timestamp - state.lastActiveFrameMs;
  if (state.activeSinceMs === null || frameGap <= 0 || frameGap > MAX_CONTIGUOUS_FRAME_GAP_MS) {
    state.activeSinceMs = timestamp;
  }
  state.lastActiveFrameMs = timestamp;
  const holdMs = timestamp - state.activeSinceMs;
  if (holdMs < requiredHoldMs) return { completed: false, holdMs };

  state.activeSinceMs = null;
  state.lastActiveFrameMs = null;
  state.blinkClosed = false;
  return { completed: true, holdMs };
}

export interface GestureProgressEvaluation extends GestureEvaluation {
  challenge: GestureChallenge | undefined;
  activeGestureIndex: number;
}

/** Owns the canonical challenge index and debounce accumulator outside React render state. */
export class GestureProgressTracker {
  private index = 0;
  private holdState = createGestureHoldState();
  private completed: boolean[];

  constructor(private challenges: readonly GestureChallenge[], private readonly requiredHoldMs = 300) {
    this.completed = challenges.map(() => false);
  }

  get activeGestureIndex(): number { return this.index; }
  get completedCount(): number { return this.completed.filter(Boolean).length; }
  get outcomes(): readonly boolean[] { return this.completed; }

  resetHold(): void { this.holdState = createGestureHoldState(); }

  reset(challenges: readonly GestureChallenge[] = this.challenges): void {
    this.challenges = challenges;
    this.index = 0;
    this.completed = challenges.map(() => false);
    this.resetHold();
  }

  evaluate(metrics: GestureMetrics): GestureProgressEvaluation {
    const challenge = this.challenges[this.index];
    if (!challenge) return { challenge, completed: false, holdMs: 0, activeGestureIndex: this.index };
    const result = evaluateGestureFrame(challenge, metrics, this.holdState, this.requiredHoldMs);
    if (result.completed) {
      this.completed[this.index] = true;
      this.index += 1;
      this.resetHold();
    }
    return { ...result, challenge, activeGestureIndex: this.index };
  }

  missCurrent(): GestureChallenge | undefined {
    const challenge = this.challenges[this.index];
    if (!challenge) return undefined;
    this.completed[this.index] = false;
    this.index += 1;
    this.resetHold();
    return challenge;
  }
}

export type FacePresenceState = 'PRESENT' | 'NO_FACE_GRACE' | 'NO_FACE_TIMEOUT' | 'MULTIPLE_FACES_GRACE' | 'MULTIPLE_FACES_TIMEOUT';

/** Tracks continuous absence/extra-face durations using monotonic frame timestamps. */
export class FacePresenceTracker {
  private noFaceSinceMs: number | null = null;
  private multipleFacesSinceMs: number | null = null;

  constructor(private readonly noFaceGraceMs = 2500, private readonly multipleFaceGraceMs = 500) {}

  evaluate(faceCount: number, timestampMs: number): FacePresenceState {
    if (faceCount <= 0) {
      if (this.noFaceSinceMs === null) this.noFaceSinceMs = timestampMs;
      this.multipleFacesSinceMs = null;
      return timestampMs - this.noFaceSinceMs >= this.noFaceGraceMs ? 'NO_FACE_TIMEOUT' : 'NO_FACE_GRACE';
    }
    if (faceCount > 1) {
      if (this.multipleFacesSinceMs === null) this.multipleFacesSinceMs = timestampMs;
      this.noFaceSinceMs = null;
      return timestampMs - this.multipleFacesSinceMs >= this.multipleFaceGraceMs ? 'MULTIPLE_FACES_TIMEOUT' : 'MULTIPLE_FACES_GRACE';
    }
    this.noFaceSinceMs = null;
    this.multipleFacesSinceMs = null;
    return 'PRESENT';
  }
}

/** Keeps the first non-null ML Kit tracking ID as this session's face identity. */
export class FaceIdentityTracker {
  private initialTrackingId: number | null = null;

  observe(trackingId: number | null): 'INITIAL' | 'SAME' | 'CHANGED' {
    if (trackingId === null) return this.initialTrackingId === null ? 'INITIAL' : 'SAME';
    if (this.initialTrackingId === null) {
      this.initialTrackingId = trackingId;
      return 'INITIAL';
    }
    return trackingId === this.initialTrackingId ? 'SAME' : 'CHANGED';
  }

  reset(): void { this.initialTrackingId = null; }
}

export interface RppgFaceMotion { yaw: number; pitch: number; centerX: number; centerY: number; timestampMs: number; }

/** True only for forward-facing observations with low normalized centroid velocity. */
export function isQuiescentRppgFrame(current: RppgFaceMotion, previous: RppgFaceMotion | null, maxVelocity = 0.25): boolean {
  'worklet';
  if (Math.abs(current.yaw) >= 8 || Math.abs(current.pitch) >= 8) return false;
  if (!previous) return true;
  const elapsedSeconds = (current.timestampMs - previous.timestampMs) / 1000;
  if (elapsedSeconds <= 0 || elapsedSeconds > 0.25) return false;
  return Math.hypot(current.centerX - previous.centerX, current.centerY - previous.centerY) / elapsedSeconds <= maxVelocity;
}

export interface NormalizedFaceBounds { x: number; y: number; width: number; height: number; }
export interface NormalizedPoint { x: number; y: number; }

/** Center the face by its stable eye/nose landmarks while tolerating small box overlap at the guide edge. */
export function isFaceAlignedToGuide(bounds: NormalizedFaceBounds, noseBase: NormalizedPoint | null, eyeMidpoint: NormalizedPoint | null): boolean {
  'worklet';
  if (!noseBase || !eyeMidpoint || bounds.width <= 0 || bounds.height <= 0) return false;
  const marginX = bounds.width * 0.15;
  const marginY = bounds.height * 0.15;
  const boundsOverlap = bounds.x >= GUIDE.xMin - marginX && bounds.x + bounds.width <= GUIDE.xMax + marginX &&
    bounds.y >= GUIDE.yMin - marginY && bounds.y + bounds.height <= GUIDE.yMax + marginY;
  const centerX = (GUIDE.xMin + GUIDE.xMax) / 2;
  const centerY = (GUIDE.yMin + GUIDE.yMax) / 2;
  const innerRadiusX = (GUIDE.xMax - GUIDE.xMin) * 0.35;
  const innerRadiusY = (GUIDE.yMax - GUIDE.yMin) * 0.35;
  const noseX = (noseBase.x - centerX) / innerRadiusX;
  const noseY = (noseBase.y - centerY) / innerRadiusY;
  const eyesX = (eyeMidpoint.x - centerX) / innerRadiusX;
  const eyesY = (eyeMidpoint.y - centerY) / innerRadiusY;
  return boundsOverlap && noseX * noseX + noseY * noseY <= 1 && eyesX * eyesX + eyesY * eyesY <= 1;
}

