const fs = require('node:fs');
const ts = require('typescript');
const { test } = require('node:test');
const assert = require('node:assert/strict');

// Transpile the project TypeScript modules for Node's built-in test runner.
require.extensions['.ts'] = (module, filename) => {
  const source = fs.readFileSync(filename, 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  module._compile(output, filename);
};

const { createGestureHoldState, evaluateGestureFrame, FacePresenceTracker, GestureProgressTracker, isFaceAlignedToGuide, isQuiescentRppgFrame, pickGestureChallenges } = require('../src/services/gestureStateMachine.ts');
const { estimatePulse, makeLivenessVerificationResult } = require('../src/services/experimentalRppg.ts');

const neutral = (timestampMs, overrides = {}) => ({
  timestampMs, yaw: 0, pitch: 0, smile: 0,
  leftEyeOpen: 1, rightEyeOpen: 1,
  ...overrides,
});

function advanceAfterHold(challenge, holdState, startMs) {
  let result = { completed: false, holdMs: 0 };
  if (challenge === 'BLINK') {
    result = evaluateGestureFrame(challenge, neutral(startMs, { leftEyeOpen: 0.1, rightEyeOpen: 0.1 }), holdState);
    assert.equal(result.completed, false);
    startMs += 33;
  }
  const pose = challenge === 'SMILE' ? { smile: 0.8 }
    : challenge === 'TURN_LEFT' ? { yaw: 21 }
      : challenge === 'TURN_RIGHT' ? { yaw: -21 }
        : challenge === 'LOOK_UP' ? { pitch: 16 } : {};
  for (let elapsed = 0; elapsed <= 330; elapsed += 33) {
    const eyes = challenge === 'BLINK' ? { leftEyeOpen: 0.9, rightEyeOpen: 0.9 } : {};
    result = evaluateGestureFrame(challenge, neutral(startMs + elapsed, { ...pose, ...eyes }), holdState);
    if (result.completed) return result;
  }
  return result;
}

test('random selection returns four unique challenges from the allowed pool', () => {
  const selected = pickGestureChallenges(() => 0.42);
  assert.equal(selected.length, 4);
  assert.equal(new Set(selected).size, 4);
  assert.ok(selected.every((challenge) => ['BLINK', 'SMILE', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP'].includes(challenge)));
});

test('each active challenge advances only after a continuous 300 ms hold', () => {
  const challenges = ['BLINK', 'SMILE', 'TURN_LEFT', 'LOOK_UP'];
  const tracker = new GestureProgressTracker(challenges);
  let timestamp = 1000;
  for (const challenge of challenges) {
    const expectedIndex = tracker.activeGestureIndex;
    const holdState = createGestureHoldState();
    const evaluation = advanceAfterHold(challenge, holdState, timestamp);
    assert.equal(evaluation.completed, true, `${challenge} should complete after the hold`);
    const pose = challenge === 'SMILE' ? { smile: 0.8 }
      : challenge === 'TURN_LEFT' ? { yaw: 21 }
        : challenge === 'TURN_RIGHT' ? { yaw: -21 }
          : challenge === 'LOOK_UP' ? { pitch: 16 } : {};
    if (challenge === 'BLINK') tracker.evaluate(neutral(timestamp, { leftEyeOpen: 0.1, rightEyeOpen: 0.1 }));
    let transition;
    const start = timestamp + (challenge === 'BLINK' ? 33 : 0);
    for (let elapsed = 0; elapsed <= 330; elapsed += 33) {
      transition = tracker.evaluate(neutral(start + elapsed, { ...pose, ...(challenge === 'BLINK' ? { leftEyeOpen: 0.9, rightEyeOpen: 0.9 } : {}) }));
      if (transition.completed) break;
    }
    assert.equal(transition.completed, true, `${challenge} should advance the actual tracker`);
    assert.equal(transition.activeGestureIndex, expectedIndex + 1);
    assert.equal(tracker.activeGestureIndex, expectedIndex + 1);
    timestamp += 1000;
  }
  assert.equal(tracker.activeGestureIndex, 4);
});

test('gesture hold resets if the next true frame is too late to prove continuity', () => {
  const state = createGestureHoldState();
  const pose = { yaw: 21 };
  evaluateGestureFrame('TURN_LEFT', neutral(1000, pose), state);
  evaluateGestureFrame('TURN_LEFT', neutral(1033, pose), state);
  const afterGap = evaluateGestureFrame('TURN_LEFT', neutral(1300, pose), state);
  assert.equal(afterGap.completed, false);
  assert.equal(afterGap.holdMs, 0);
});

test('turn-right uses the specified negative yaw threshold', () => {
  const tracker = new GestureProgressTracker(['TURN_RIGHT']);
  assert.equal(tracker.evaluate(neutral(1000, { yaw: -20 })).holdMs, 0);
  let result;
  for (let elapsed = 0; elapsed <= 330; elapsed += 33) {
    result = tracker.evaluate(neutral(1033 + elapsed, { yaw: -21 }));
    if (result.completed) break;
  }
  assert.equal(result.completed, true);
  assert.equal(tracker.activeGestureIndex, 1);
});

test('landmark-centered oval check tolerates 15% box overlap but rejects points outside the inner ellipse', () => {
  const nose = { x: 0.5, y: 0.52 };
  const eyes = { x: 0.5, y: 0.4 };
  assert.equal(isFaceAlignedToGuide({ x: 0.06, y: 0.1, width: 0.6, height: 0.7 }, nose, eyes), true);
  assert.equal(isFaceAlignedToGuide({ x: 0.05, y: 0.1, width: 0.6, height: 0.7 }, nose, eyes), false);
  assert.equal(isFaceAlignedToGuide({ x: 0.2, y: 0.1, width: 0.6, height: 0.7 }, { x: 0.76, y: 0.52 }, eyes), false);
});

test('face loss under 2.5 seconds is recoverable and continuous absence then times out', () => {
  const tracker = new FacePresenceTracker();
  assert.equal(tracker.evaluate(1, 1000), 'PRESENT');
  assert.equal(tracker.evaluate(0, 1100), 'NO_FACE_GRACE');
  assert.equal(tracker.evaluate(0, 3500), 'NO_FACE_GRACE');
  assert.equal(tracker.evaluate(1, 3550), 'PRESENT', 'reacquisition resets the absence timer');
  assert.equal(tracker.evaluate(0, 4000), 'NO_FACE_GRACE');
  assert.equal(tracker.evaluate(0, 6500), 'NO_FACE_TIMEOUT');
});

test('multiple faces require 500 ms continuous presence before timeout', () => {
  const tracker = new FacePresenceTracker();
  assert.equal(tracker.evaluate(2, 0), 'MULTIPLE_FACES_GRACE');
  assert.equal(tracker.evaluate(2, 499), 'MULTIPLE_FACES_GRACE');
  assert.equal(tracker.evaluate(1, 500), 'PRESENT', 'one face resets the timer');
  assert.equal(tracker.evaluate(2, 800), 'MULTIPLE_FACES_GRACE');
  assert.equal(tracker.evaluate(2, 1300), 'MULTIPLE_FACES_TIMEOUT');
});

test('turn gestures may cross the visual oval without becoming a face-presence failure', () => {
  const presence = new FacePresenceTracker();
  const tracker = new GestureProgressTracker(['TURN_LEFT']);
  const outsideGuide = isFaceAlignedToGuide(
    { x: 0.83, y: 0.2, width: 0.18, height: 0.5 },
    { x: 0.98, y: 0.48 },
    { x: 0.91, y: 0.36 },
  );
  assert.equal(outsideGuide, false, 'the soft oval may report off-guide alignment');
  assert.equal(presence.evaluate(1, 1000), 'PRESENT');
  let result;
  for (let elapsed = 0; elapsed <= 330; elapsed += 33) {
    result = tracker.evaluate(neutral(1000 + elapsed, { yaw: 21 }));
    if (result.completed) break;
  }
  assert.equal(result.completed, true);
  assert.equal(presence.evaluate(1, 1400), 'PRESENT');
});

test('rPPG quiescent filter accepts steady frontal frames and rejects pose or fast motion', () => {
  const first = { yaw: 0, pitch: 0, centerX: 0.5, centerY: 0.5, timestampMs: 0 };
  assert.equal(isQuiescentRppgFrame(first, null), true);
  assert.equal(isQuiescentRppgFrame({ ...first, yaw: 9, timestampMs: 33 }, first), false);
  assert.equal(isQuiescentRppgFrame({ ...first, centerX: 0.55, timestampMs: 100 }, first), false);
  assert.equal(isQuiescentRppgFrame({ ...first, centerX: 0.505, timestampMs: 100 }, first), true);
});

test('POS accepts a clean 72 BPM pulse signal above the relaxed SNR threshold', () => {
  const samples = Array.from({ length: 150 }, (_, index) => {
    const timestampMs = index * 1000 / 30;
    const pulse = Math.sin(2 * Math.PI * 1.2 * timestampMs / 1000);
    return { r: 100 + pulse, g: 120 + 2 * pulse, b: 90 + 0.4 * pulse, timestampMs };
  });
  const telemetry = estimatePulse(samples);
  assert.equal(telemetry.isMeasurementReliable, true);
  assert.ok(Math.abs(telemetry.estimatedBpm - 72) <= 1);
});

test('3 of 4 gestures pass, fewer fail, and pulse telemetry never changes liveness', () => {
  const noPulse = { estimatedBpm: null, signalQualityScore: 0, isMeasurementReliable: false };
  const flatSamples = Array.from({ length: 120 }, (_, index) => ({
    r: 100, g: 120, b: 90, timestampMs: index * 1000 / 30,
  }));
  const lowSnrPulse = estimatePulse(flatSamples);
  assert.equal(lowSnrPulse.isMeasurementReliable, false, 'flat/no-pulse samples should be inconclusive');
  assert.equal(lowSnrPulse.estimatedBpm, null);
  assert.equal(makeLivenessVerificationResult(3, 4, noPulse, true).livenessPassed, true);
  assert.equal(makeLivenessVerificationResult(3, 4, lowSnrPulse, true).livenessPassed, true, 'low SNR must not affect liveness');
  assert.equal(makeLivenessVerificationResult(2, 4, noPulse, true).livenessPassed, false);
  const faceLost = makeLivenessVerificationResult(4, 4, noPulse, false, [], 'FACE_OUT_OF_BOUNDS');
  assert.equal(faceLost.livenessPassed, false);
  assert.equal(faceLost.failureReason, 'FACE_OUT_OF_BOUNDS');
});

test('timed-out challenges remain missed while later challenges can pass', () => {
  const tracker = new GestureProgressTracker(['SMILE', 'TURN_LEFT']);
  tracker.missCurrent();
  for (let elapsed = 0; elapsed <= 330; elapsed += 33) {
    const result = tracker.evaluate(neutral(1000 + elapsed, { yaw: 21 }));
    if (result.completed) break;
  }
  assert.deepEqual(tracker.outcomes, [false, true]);
  const noPulse = { estimatedBpm: null, signalQualityScore: 0, isMeasurementReliable: false };
  const result = makeLivenessVerificationResult(tracker.completedCount, 2, noPulse, true, ['SMILE', 'TURN_LEFT'], undefined, tracker.outcomes);
  assert.deepEqual(result.gestureBreakdown.map((item) => item.status), ['Missed', 'Completed']);
});
