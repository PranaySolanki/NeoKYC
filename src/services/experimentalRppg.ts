/** Experimental pulse estimation. This module must never be used to decide liveness. */
export interface FrameSample {
  r: number;
  g: number;
  b: number;
  timestampMs: number;
}

export interface RppgTelemetry {
  estimatedBpm: number | null;
  signalQualityScore: number;
  isMeasurementReliable: boolean;
}

/** Result contract keeps experimental vitals separate from the liveness verdict. */
export interface LivenessVerificationResult {
  livenessPassed: boolean;
  livenessScore: number;
  rppgTelemetry: RppgTelemetry;
  faceMaintained: boolean;
  failureReason: 'MULTIPLE_FACES_DETECTED' | 'FACE_OUT_OF_BOUNDS' | 'INCOMPLETE_GESTURES' | null;
  gestureBreakdown: Array<{ gesture: string; status: 'Completed' | 'Missed' }>;
}

/** Bounded in-memory ring for timestamped ROI means; no image data is retained. */
export class FrameSampleBuffer {
  private readonly storage: Array<FrameSample | undefined>;
  private next = 0;
  private count = 0;

  constructor(private readonly capacity = 300) {
    this.storage = new Array<FrameSample | undefined>(Math.max(1, capacity));
  }

  push(sample: FrameSample): void {
    this.storage[this.next] = sample;
    this.next = (this.next + 1) % this.storage.length;
    this.count = Math.min(this.count + 1, this.storage.length);
  }

  clear(): void {
    this.storage.fill(undefined);
    this.next = 0;
    this.count = 0;
  }

  values(): FrameSample[] {
    const start = (this.next - this.count + this.storage.length) % this.storage.length;
    return Array.from({ length: this.count }, (_, index) => this.storage[(start + index) % this.storage.length] as FrameSample);
  }

  latest(): FrameSample | undefined {
    if (this.count === 0) return undefined;
    return this.storage[(this.next - 1 + this.storage.length) % this.storage.length];
  }

  get length(): number { return this.count; }
}

/** Build the combined UI contract without allowing rPPG telemetry into liveness scoring. */
export function makeLivenessVerificationResult(
  completedGestures: number,
  requiredGestures: number,
  rppgTelemetry: RppgTelemetry,
  faceMaintained = true,
  challenges: readonly string[] = [],
  failureReason?: LivenessVerificationResult['failureReason'],
  challengeOutcomes?: readonly boolean[],
): LivenessVerificationResult {
  const denominator = Math.max(1, requiredGestures);
  const completed = Math.max(0, Math.min(requiredGestures, completedGestures));
  const livenessScore = completed / denominator;
  const livenessPassed = completed >= Math.ceil(requiredGestures * 0.75) && faceMaintained;
  return {
    livenessPassed,
    livenessScore,
    rppgTelemetry,
    faceMaintained,
    failureReason: livenessPassed ? null : failureReason ?? (faceMaintained ? 'INCOMPLETE_GESTURES' : 'FACE_OUT_OF_BOUNDS'),
    gestureBreakdown: challenges.map((gesture, index) => ({ gesture, status: (challengeOutcomes?.[index] ?? index < completed) ? 'Completed' : 'Missed' })),
  };
}

const SAMPLE_RATE = 30;
const MIN_SAMPLES = 120;
const MIN_SNR_DB = 1.0;

export function estimatePulse(samples: readonly FrameSample[]): RppgTelemetry {
  const inconclusive: RppgTelemetry = { estimatedBpm: null, signalQualityScore: 0, isMeasurementReliable: false };
  if (samples.length < MIN_SAMPLES) return inconclusive;
  const ordered = samples.slice(-300).reduce<FrameSample[]>((result, sample) => {
    const last = result[result.length - 1];
    if (Number.isFinite(sample.timestampMs) && Number.isFinite(sample.r) && Number.isFinite(sample.g) && Number.isFinite(sample.b) &&
      (!last || sample.timestampMs > last.timestampMs)) result.push(sample);
    return result;
  }, []);
  if (ordered.length < MIN_SAMPLES) return inconclusive;
  if (ordered[ordered.length - 1].timestampMs - ordered[0].timestampMs < 2900) return inconclusive;
  if (ordered.some((sample, i) => i > 0 && sample.timestampMs - ordered[i - 1].timestampMs > 250)) return inconclusive;

  const start = ordered[0].timestampMs;
  const gridCount = Math.floor((ordered[ordered.length - 1].timestampMs - start) * SAMPLE_RATE / 1000) + 1;
  if (gridCount < MIN_SAMPLES) return inconclusive;
  const uniform: FrameSample[] = [];
  let right = 1;
  for (let i = 0; i < gridCount; i++) {
    const t = start + i * 1000 / SAMPLE_RATE;
    while (right < ordered.length - 1 && ordered[right].timestampMs < t) right++;
    const a = ordered[right - 1], b = ordered[right];
    const f = Math.max(0, Math.min(1, (t - a.timestampMs) / (b.timestampMs - a.timestampMs)));
    uniform.push({ r: a.r + f * (b.r - a.r), g: a.g + f * (b.g - a.g), b: a.b + f * (b.b - a.b), timestampMs: t });
  }

  const means = [
    uniform.reduce((sum, s) => sum + s.r, 0) / uniform.length,
    uniform.reduce((sum, s) => sum + s.g, 0) / uniform.length,
    uniform.reduce((sum, s) => sum + s.b, 0) / uniform.length,
  ];
  const rn = uniform.map(s => s.r / (means[0] || 1) - 1);
  const gn = uniform.map(s => s.g / (means[1] || 1) - 1);
  const bn = uniform.map(s => s.b / (means[2] || 1) - 1);
  const s1 = gn.map((g, i) => g - bn[i]);
  const s2 = gn.map((g, i) => g + bn[i] - 2 * rn[i]);
  const sd = (v: number[]) => { const m = v.reduce((x, y) => x + y, 0) / v.length; return Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length); };
  const alpha = sd(s2) > 1e-9 ? sd(s1) / sd(s2) : 0;
  const projected = s1.map((v, i) => v + alpha * s2[i]);
  const butterworthSection = (input: number[], cutoffHz: number, highPass: boolean): number[] => {
    const omega = 2 * Math.PI * cutoffHz / SAMPLE_RATE;
    const cosine = Math.cos(omega), sine = Math.sin(omega), q = Math.SQRT1_2;
    const a = sine / (2 * q), a0 = 1 + a;
    const b0 = highPass ? (1 + cosine) / 2 : (1 - cosine) / 2;
    const b1 = highPass ? -(1 + cosine) : 1 - cosine;
    const b2 = b0;
    const a1 = -2 * cosine, a2 = 1 - a;
    const output = new Array<number>(input.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < input.length; i++) {
      const x0 = input[i];
      const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
      output[i] = y0; x2 = x1; x1 = x0; y2 = y1; y1 = y0;
    }
    return output;
  };
  const pulse = butterworthSection(butterworthSection(projected, 0.75, true), 2.5, false);
  // Remove linear illumination drift, then apply a Hamming window. Frequency
  // power is estimated directly on the uniform grid (0.01 Hz resolution).
  const meanT = (pulse.length - 1) / 2, meanY = pulse.reduce((a, b) => a + b, 0) / pulse.length;
  const slope = pulse.reduce((sum, y, i) => sum + (i - meanT) * (y - meanY), 0) / pulse.reduce((sum, _, i) => sum + (i - meanT) ** 2, 0);
  const signal = pulse.map((v, i) => (v - meanY - slope * (i - meanT)) * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (pulse.length - 1))));
  const powers: { hz: number; power: number }[] = [];
  for (let hz = 0.75; hz <= 2.5001; hz += 0.01) {
    let re = 0, im = 0;
    signal.forEach((v, i) => { const a = 2 * Math.PI * hz * i / SAMPLE_RATE; re += v * Math.cos(a); im -= v * Math.sin(a); });
    powers.push({ hz, power: re * re + im * im });
  }
  const peak = powers.reduce((a, b) => b.power > a.power ? b : a);
  const noise = powers.filter(p => Math.abs(p.hz - peak.hz) > 0.12).map(p => p.power);
  const noiseMean = noise.reduce((a, b) => a + b, 0) / Math.max(1, noise.length);
  const snrDb = 10 * Math.log10((peak.power + 1e-12) / (noiseMean + 1e-12));
  const quality = Math.max(0, Math.min(1, (snrDb - MIN_SNR_DB) / 12));
  const reliable = snrDb >= MIN_SNR_DB && Number.isFinite(peak.hz);
  return { estimatedBpm: reliable ? peak.hz * 60 : null, signalQualityScore: quality, isMeasurementReliable: reliable };
}
