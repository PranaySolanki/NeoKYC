# Liveness check

The Face Liveness screen uses the front camera and on-device ML Kit face classification to request three actions in a randomized order: head movement, an eye blink, and a smile. The screen only proceeds after one visible face performs all three actions. Face data is evaluated in memory and is not saved by this screen.

## Native build requirement

This feature uses Vision Camera and its ML Kit frame processor. A new native app binary is required; Expo Go cannot load these native modules. After installing dependencies, create/rebuild the development client (for example with `npx expo run:android` or an EAS development build), then launch Metro with `npx expo start --dev-client`. On iOS, install CocoaPods as part of the native build.

## Limits and recommended checks

These challenges provide active liveness signals and make a static photo less likely to pass. They are not a presentation-attack detection (PAD) guarantee: a replayed video or sophisticated mask may still pass, and the checks must not be treated as sufficient evidence for high-risk identity approval. For production KYC, add server-side PAD/deepfake analysis on a short, consented capture, bind a fresh server nonce to the session, expire and rate-limit attempts, and retain an auditable result. Also check image quality, face count, lighting, and face match to the document portrait. Keep biometric capture purpose-limited, encrypted, retention-limited, and covered by a clear consent notice.

The movement check deliberately accepts either direction to avoid left/right confusion from mirrored front-camera previews. Blink and smile thresholds are heuristic and should be tuned with representative devices and users before production use.
