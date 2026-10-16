# NeoKYC Repository Audit

> **Audit Date:** October 2026
> **Repository:** `C:\projects\NEO-KYC\NeoKYC`
> **Auditor:** Automated source-code inspection (no code was executed or modified)

---

## 1. Repository Structure Overview

```
NeoKYC/
├── App.tsx                          # React Native entry component (navigation)
├── index.ts                         # Expo registerRootComponent
├── app.json                         # Expo configuration (NeoKYC, dark UI)
├── package.json                     # Node.js dependencies (Expo 54, React 19)
├── tsconfig.json                    # TypeScript strict mode config
├── requirements.txt                 # Python 3.10+ dependencies
├── README.md                        # Project documentation (477 lines)
├── .gitignore                       # Standard Expo + node_modules ignores
│
├── src/
│   └── screens/
│       ├── index.ts                 # Barrel export for all screens
│       ├── HomeScreen/
│       │   └── HomeScreen.tsx       # Navigation hub (98 lines)
│       ├── DocumentScanScreen/
│       │   └── DocumentScanScreen.tsx  # Camera + OCR (745 lines) ★ IMPLEMENTED
│       ├── FaceLivenessScreen/
│       │   └── FaceLivenessScreen.tsx  # Placeholder stub (30 lines)
│       └── ResultScreen/
│           └── ResultScreen.tsx      # Placeholder stub (30 lines)
│
├── backend/
│   ├── train.py                     # Training entry point + smoke test
│   ├── api/
│   │   ├── __init__.py              # Package docstring
│   │   ├── app.py                   # Flask application factory (399 lines) ★
│   │   ├── config.py                # DeepfakeServiceConfig dataclass (100 lines)
│   │   ├── inference_service.py     # F3NetInferenceService (557 lines) ★
│   │   ├── face_detector.py         # YuNet/Haar face detection (208 lines) ★
│   │   └── ekyc_pipeline.py         # eKYC orchestrator (457 lines) ★
│   ├── models/
│   │   └── f3net/
│   │       ├── __init__.py          # Public API exports
│   │       ├── f3net.py             # F3Net main module (172 lines) ★
│   │       ├── dct.py               # DCT-II utilities (104 lines) ★
│   │       ├── fad.py               # FAD_Head (132 lines) ★
│   │       ├── lfs.py               # LFS_Head (123 lines) ★
│   │       ├── mixblock.py          # MixBlock cross-attention (94 lines) ★
│   │       └── xception.py          # XceptionBackbone (321 lines) ★
│   ├── data/
│   │   ├── __init__.py              # Package exports
│   │   ├── faceforensics.py         # FF++ dataset loader (230 lines) ★
│   │   └── transforms.py           # F3NetPreprocessing (104 lines) ★
│   ├── training/
│   │   ├── __init__.py              # Package exports
│   │   ├── config.py                # TrainingConfig dataclass (79 lines)
│   │   ├── trainer.py               # Training loop (230 lines) ★
│   │   └── metrics.py               # ROC-AUC, binary metrics (90 lines) ★
│   ├── evaluation/
│   │   ├── evaluator.py             # F3NetEvaluator (205 lines) ★
│   │   └── metrics.py               # Classification metrics + calibration (158 lines) ★
│   ├── tests/
│   │   ├── test_f3net_components.py  # 11 unit tests ★
│   │   ├── test_f3net_architecture.py # 14 unit tests ★
│   │   ├── test_training_pipeline.py  # 7 unit tests ★
│   │   └── test_api.py               # 42 integration tests ★
│   └── _models/
│       └── face_detection_yunet_2023mar.onnx  # YuNet ONNX model (227 KB)
│
├── checkpoints/
│   ├── f3net/                       # Empty — no production checkpoint
│   └── smoke_test/
│       ├── best_model.pth           # ~339 MB (synthetic random weights)
│       ├── latest_checkpoint.pth    # ~339 MB (synthetic random weights)
│       └── history.json             # 2-epoch smoke test metrics
│
└── android/                         # Generated Expo Android project
```

★ = Implemented and functional based on source inspection

---

## 2. Component-by-Component Audit

### 2.1 Frontend (React Native / Expo)

| Component | File | LOC | Status |
|:---|:---|:---:|:---|
| App Entry | `App.tsx` | 74 | **Implemented** — State-based navigation between 4 screens |
| HomeScreen | `HomeScreen.tsx` | 98 | **Implemented** — Navigation hub with 3 buttons |
| DocumentScanScreen | `DocumentScanScreen.tsx` | 745 | **Implemented** — Full camera + OCR pipeline |
| FaceLivenessScreen | `FaceLivenessScreen.tsx` | 30 | **Stub** — Static text only, no camera or liveness logic |
| ResultScreen | `ResultScreen.tsx` | 30 | **Stub** — Static text only, no API integration |

**DocumentScanScreen Details:**
- Uses `expo-camera` `CameraView` for photo capture
- Uses `@infinitered/react-native-mlkit-text-recognition` for on-device OCR
- Implements spatial-aware field extraction: Name, DOB, Gender, ID Number
- Parses 12-digit Aadhaar-style card numbers via regex
- Uses spatial proximity heuristics (Y-coordinate) to find name above DOB
- Includes torch toggle, alignment guide, confirmation checkbox
- Navigates to FaceLivenessScreen on confirmation

**Key Gaps:**
- No HTTP communication with the backend API
- No face image capture for deepfake detection
- FaceLivenessScreen and ResultScreen are non-functional stubs
- No state management library (uses local `useState`)

### 2.2 Backend API (Flask)

| Module | File | LOC | Status |
|:---|:---|:---:|:---|
| Flask App Factory | `api/app.py` | 399 | **Implemented** — 5 endpoints |
| Service Config | `api/config.py` | 100 | **Implemented** — Typed dataclass |
| Inference Service | `api/inference_service.py` | 557 | **Implemented** — Full lifecycle management |
| Face Detector | `api/face_detector.py` | 208 | **Implemented** — YuNet + Haar fallback |
| eKYC Pipeline | `api/ekyc_pipeline.py` | 457 | **Implemented** — Orchestrator with 4 check stubs |

**API Endpoints:**

| Method | Path | Status | Notes |
|:---|:---|:---|:---|
| GET | `/api/deepfake/health` | Implemented | Returns `degraded` when no checkpoint |
| POST | `/api/deepfake/predict-image` | Implemented | Returns 503 without trained checkpoint |
| POST | `/api/deepfake/predict-video` | Implemented | Temp file + frame sampling + cleanup |
| GET | `/api/ekyc/health` | Implemented | Reports all component statuses |
| POST | `/api/ekyc/verify` | Implemented | Multi-stage with document + face |

**eKYC Pipeline Checks:**

| Check | Implementation Status |
|:---|:---|
| Document OCR validation | **Implemented** — Validates 4 fields from form data |
| Deepfake detection (F³-Net) | **Implemented** — Runs inference if checkpoint present |
| Liveness detection | **Stub** — Returns `UNAVAILABLE` |
| Face identity matching (ArcFace) | **Stub** — Returns `UNAVAILABLE` |
| Graph fraud analysis (Neo4j) | **Stub** — Returns `UNAVAILABLE` |

### 2.3 F³-Net Model Architecture

| Module | File | LOC | Status |
|:---|:---|:---:|:---|
| F3Net (main) | `f3net.py` | 172 | **Implemented** — 4 modes (Mix, Both, FAD, LFS) |
| DCT Utilities | `dct.py` | 104 | **Implemented** — Orthonormal DCT-II matrix, dct2, idct2 |
| FAD Head | `fad.py` | 132 | **Implemented** — 4 frequency bands, learnable filters |
| LFS Head | `lfs.py` | 123 | **Implemented** — Sliding window DCT, 6 bands |
| MixBlock | `mixblock.py` | 94 | **Implemented** — Cross-attention at Block 7 |
| Xception Backbone | `xception.py` | 321 | **Implemented** — Partitioned stages, ImageNet adapter |

**Architecture Verification:**
- ✅ DCT is Type-II (not FFT) — confirmed in `dct.py` line 7–31
- ✅ FAD produces (N, 12, 299, 299) output — 4 bands × 3 RGB channels
- ✅ LFS produces (N, 6, 149, 149) output — 6 spectral bands
- ✅ Both Xception branches produce (N, 728, 19, 19) at Block 7
- ✅ MixBlock operates at 728 channels, 19×19 spatial
- ✅ Final classifier: Linear(4096 → 2) in Mix/Both modes
- ✅ `predict_proba()` uses softmax for 2-class, sigmoid for 1-class

### 2.4 Face Detection

| Feature | Status |
|:---|:---|
| Primary: YuNet (FaceDetectorYN) | **Implemented** — OpenCV 5.x DNN |
| Fallback: Haar Cascade | **Implemented** — OpenCV 4.x |
| YuNet ONNX model present | **Yes** — `_models/face_detection_yunet_2023mar.onnx` (227 KB) |
| Auto-download if missing | **Implemented** — Downloads from OpenCV Zoo |
| Bounding box expansion (1.3×) | **Implemented** — Uses `expand_bounding_box()` |
| Graceful degradation | **Implemented** — Falls back to no-crop analysis |

### 2.5 Data Pipeline

| Feature | Status |
|:---|:---|
| FaceForensicsDataset | **Implemented** — Filesystem scanner with metadata |
| Split isolation (anti-leakage) | **Implemented** — Source-video-level partitioning |
| Official split JSON loading | **Implemented** — Falls back to 720/140/140 |
| F3NetPreprocessing | **Implemented** — Resize 299×299, normalize to [-1, 1] |
| Training augmentation | **Implemented** — Random horizontal flip |
| FaceForensics++ data present | **No** — Dataset directory is empty |

### 2.6 Training Pipeline

| Feature | Status |
|:---|:---|
| Trainer class | **Implemented** — Full training loop with validation |
| SGD + Cosine Annealing | **Implemented** — Matches F³-Net paper defaults |
| Gradient clipping (5.0) | **Implemented** |
| Checkpoint save/load | **Implemented** — best_model.pth + latest_checkpoint.pth |
| Training history JSON | **Implemented** — Saves metrics per epoch |
| Pretrained Xception loading | **Implemented** — Channel adaptation (3→12, 3→6) |
| Smoke test pipeline | **Implemented** — 2-epoch synthetic tensor test |
| Full dataset training | **Not executed** — No FF++ data available |

### 2.7 Evaluation Pipeline

| Feature | Status |
|:---|:---|
| F3NetEvaluator | **Implemented** — Frame + video-level metrics |
| ROC-AUC (trapezoidal) | **Implemented** — Custom implementation |
| Confusion matrix | **Implemented** |
| Threshold calibration | **Implemented** — F1, Youden, EER criteria |
| Video score aggregation | **Implemented** — Mean, median, max |
| Markdown report generation | **Implemented** |
| Actual evaluation results | **None** — No trained model or dataset |

### 2.8 Checkpoint Status

| Checkpoint | Location | Size | Origin |
|:---|:---|:---:|:---|
| `smoke_test/best_model.pth` | `checkpoints/smoke_test/` | ~339 MB | **Synthetic** — 2-epoch random tensor training |
| `smoke_test/latest_checkpoint.pth` | `checkpoints/smoke_test/` | ~339 MB | **Synthetic** — Same run, epoch 2 |
| `smoke_test/history.json` | `checkpoints/smoke_test/` | 306 B | Train loss: 0.637/0.736, Val AUC: 1.0 (meaningless) |
| `f3net/best_model.pth` | `checkpoints/f3net/` | — | **Does not exist** |

> [!CAUTION]
> The smoke test checkpoint was trained on 4 synthetic random tensors for 2 epochs. The reported "Val AUC: 1.0" is an artifact of having only 2 samples (1 per class) in validation. This checkpoint has **zero deepfake detection capability** and must not be used for inference.

### 2.9 Security and Privacy

| Feature | Evidence |
|:---|:---|
| In-memory image processing | ✅ `io.BytesIO` in `inference_service.py` (line 284) |
| Temp video file cleanup | ✅ `finally` block with `os.unlink()` (line 485–489) |
| VideoCapture release before unlink | ✅ `cap.release()` in `finally` (line 478–482) |
| No biometric data in responses | ✅ Predictions return scores only, not image data |
| No hardcoded secrets | ✅ All config via env vars or CLI args |
| No biometric data in logs | ✅ Only dimensions, counts, status codes logged |
| Upload size limit | ✅ 50 MB via `MAX_CONTENT_LENGTH` |
| File extension validation | ✅ Whitelisted extensions for image and video |
| HTTPS/TLS | ❌ Not configured — Flask dev server only |
| Authentication | ❌ No auth middleware on any endpoint |
| Rate limiting | ❌ Not implemented |

### 2.10 Database and Storage Integrations

| Technology | Status |
|:---|:---|
| Neo4j graph database | **Not implemented** — Referenced in README and ResultScreen stub only |
| Vector similarity search | **Not implemented** — No code references |
| ArcFace embeddings | **Not implemented** — Referenced in stubs |
| SQLite/PostgreSQL | **Not used** |
| Redis/caching | **Not used** |

---

## 3. Dependency Audit

### 3.1 Python Dependencies (requirements.txt)

| Package | Version | Purpose | Used In |
|:---|:---|:---|:---|
| torch | ≥ 2.2.0 | Deep learning framework | Model, training, inference |
| Flask | ≥ 3.0.0 | REST API framework | `api/app.py` |
| Werkzeug | ≥ 3.0.0 | Flask WSGI utilities | Implicit |
| numpy | ≥ 1.26.0 | Numerical computing | Metrics, transforms |
| scipy | ≥ 1.11.0 | Scientific computing | Listed but not directly imported |
| scikit-learn | ≥ 1.3.0 | ML evaluation | Listed but not directly imported |
| opencv-python-headless | ≥ 4.8.0 | Face detection, video I/O | `face_detector.py`, `inference_service.py` |
| Pillow | ≥ 10.0.0 | Image processing | Transforms, inference |
| tqdm | ≥ 4.66.0 | Progress bars | Listed for training scripts |

### 3.2 Node.js Dependencies (package.json)

| Package | Version | Purpose |
|:---|:---|:---|
| expo | ~54.0.37 | Mobile framework |
| react | 19.1.0 | UI library |
| react-native | 0.81.5 | Native rendering |
| expo-camera | ~17.0.10 | Camera access |
| @infinitered/react-native-mlkit-text-recognition | ^5.0.1 | On-device OCR |
| expo-image-picker | ~17.0.11 | Image selection |
| @react-navigation/native | ^7.3.17 | Navigation (imported but unused) |
| react-native-webview | 13.15.0 | WebView component (unused) |

---

## 4. Implementation Gap Summary

| Feature | Classification |
|:---|:---|
| F³-Net architecture (all 4 modes) | ✅ Implemented and tested |
| DCT-based frequency processing | ✅ Implemented and tested |
| FAD frequency decomposition | ✅ Implemented and tested |
| LFS local frequency statistics | ✅ Implemented and tested |
| MixBlock cross-attention fusion | ✅ Implemented and tested |
| Xception dual-stream backbone | ✅ Implemented and tested |
| Flask REST API (5 endpoints) | ✅ Implemented and tested |
| Face detection (YuNet + Haar) | ✅ Implemented and tested |
| eKYC pipeline orchestrator | ✅ Implemented and tested |
| Document OCR (mobile) | ✅ Implemented (not backend-tested) |
| Training pipeline + Trainer | ✅ Implemented and smoke-tested |
| Evaluation pipeline | ✅ Implemented (not executed on real data) |
| FaceForensics++ dataset loader | ✅ Implemented (no data populated) |
| Threshold calibration module | ✅ Implemented (not calibrated) |
| **Genuine F³-Net training** | ❌ Not completed — no FF++ dataset |
| **Production-ready checkpoint** | ❌ Not available |
| **FaceLivenessScreen** | ❌ Stub — no camera or liveness logic |
| **ResultScreen** | ❌ Stub — no API integration |
| **Liveness detection backend** | ❌ Stub — returns UNAVAILABLE |
| **ArcFace face matching** | ❌ Not implemented |
| **Neo4j graph fraud detection** | ❌ Not implemented |
| **Vector similarity search** | ❌ Not implemented |
| **Frontend-backend HTTP integration** | ❌ Not implemented |
| **Authentication / authorization** | ❌ Not implemented |
| **HTTPS / TLS** | ❌ Not configured |
| **Rate limiting** | ❌ Not implemented |
