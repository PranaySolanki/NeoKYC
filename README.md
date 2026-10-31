# NeoKYC: Zero-Trust Identity Verification Platform
## Multi-Spectral Deepfake Defense (F³-Net) & Graph Threat Intelligence

---

## 1. F³-Net Architecture Overview
NeoKYC integrates **F³-Net (Frequency in Face Forgery Network)**, reproducing the multi-spectral frequency analysis approach introduced in ECCV 2020:
*Paper*: [*Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues* (Qian et al., ECCV 2020)](https://arxiv.org/abs/2007.09355)  
*Reference*: [https://github.com/yyk-wew/F3Net](https://github.com/yyk-wew/F3Net)

Deepfake generators (GANs, VAEs, Diffusion models, FaceSwap) introduce systematic artifacts into the frequency domain (checkerboard patterns, high-frequency energy anomalies, spectral degradation due to upsampling). F³-Net mines these forgery artifacts through two complementary frequency streams:

```
                          Input RGB Image (299 x 299)
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌──────────────────────┐                          ┌──────────────────────┐
│  FAD Preprocessing   │                          │  LFS Preprocessing   │
│  (Global 2D-DCT,     │                          │  (Sliding 10x10 DCT, │
│  3-band filter + all)│                          │  6 spectral bands)   │
└──────────┬───────────┘                          └──────────┬───────────┘
           │ 12 channels (299x299)                           │ 6 channels (149x149)
           ▼                                                 ▼
┌──────────────────────┐                          ┌──────────────────────┐
│   Xception Stream 1  │                          │   Xception Stream 2  │
│  (Entry -> Block 7)  │                          │  (Entry -> Block 7)  │
└──────────┬───────────┘                          └──────────┬───────────┘
           │ 728 ch (19x19)                                  │ 728 ch (19x19)
           └────────────────────────┬────────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │       MixBlock       │
                         │   Cross-Attention    │
                         │ Collaborative Fusion │
                         └──────────┬───────────┘
           ┌────────────────────────┴────────────────────────┐
           │ 728 ch (19x19)                                  │ 728 ch (19x19)
           ▼                                                 ▼
┌──────────────────────┐                          ┌──────────────────────┐
│   Xception Stream 1  │                          │   Xception Stream 2  │
│  (Blocks 8-12 + Exit)│                          │  (Blocks 8-12 + Exit)│
└──────────┬───────────┘                          └──────────┬───────────┘
           │ 2048-dim pool                                   │ 2048-dim pool
           └────────────────────────┬────────────────────────┘
                                    ▼
                         Concatenation (4096-dim)
                                    │
                             Dropout (p=0.5)
                                    │
                          Linear Classifier (2)
                                    │
                             [Real vs. Fake]
```

### 1.1 Frequency-Aware Decomposition (FAD)
- **Module**: [`backend.models.f3net.fad.FAD_Head`](file:///c:/projects/NEO-KYC/NeoKYC/backend/models/f3net/fad.py)
- **Mechanism**: Computes global 2D Discrete Cosine Transform (DCT-II) across the entire 299×299 image. Decomposes the spectrum into three distinct frequency bands (Low, Mid, High) plus an All-pass band:
  - **Low Frequency**: $[0, \frac{N}{2.82}]$ capturing fundamental facial geometry.
  - **Mid Frequency**: $[\frac{N}{2.82}, \frac{N}{2.0}]$ capturing structural transitions and blending seams.
  - **High Frequency**: $[\frac{N}{2.0}, 2N]$ capturing noise and fine textural artifacts.
  - **All-pass**: $[0, 2N]$ preserving global contextual reference.
- **Learnable Filter Bounding**: Uses $\sigma_{\text{norm}}(w) = 2 \cdot \text{sigmoid}(w) - 1 \in (-1, 1)$ to adaptively emphasize salient artifact frequencies while preventing filter divergence.
- **Output**: 4 bands $\times$ 3 channels = 12 spatial channels reconstructed via differentiable 2D Inverse DCT (IDCT).

### 1.2 Local Frequency Statistics (LFS)
- **Module**: [`backend.models.f3net.lfs.LFS_Head`](file:///c:/projects/NEO-KYC/NeoKYC/backend/models/f3net/lfs.py)
- **Mechanism**: Extracts local patches using sliding-window DCT:
  - Input: Grayscale luminance representation ($0.299R + 0.587G + 0.114B$).
  - Sliding window: size $S = 10$, stride $= 2$, padding $= 4$.
  - Patch-wise 2D-DCT: Computes $10 \times 10$ orthonormal DCT on each patch.
  - Log spectral energy: $\log_{10}(|\text{DCT}| + 10^{-15})$.
  - Bandpass decomposition: Divides spectral domain into $M = 6$ frequency bands.
  - Spectral aggregation: Pools energy over patch dimensions to construct spatial frequency feature maps.
- **Output**: $6 \times 149 \times 149$ tensor capturing spatial distribution of local frequency variations.

### 1.3 Modified Dual-Stream Xception Backbone
- **Module**: [`backend.models.f3net.xception.XceptionBackbone`](file:///c:/projects/NEO-KYC/NeoKYC/backend/models/f3net/xception.py)
- **Adaptation**:
  - **FAD Stream**: Initial conv layer adapted from 3 channels to 12 channels (stride 2, padding 0).
  - **LFS Stream**: Initial conv layer adapted from 3 channels to 6 channels with stride 1, padding 1 to match the $149 \times 149$ input resolution to the Entry Flow.
  - **Intermediate Split**: Partitioned into Entry Flow & Middle Flow Part 1 (Blocks 1–7) delivering $728 \times 19 \times 19$ feature maps, and Middle Flow Part 2 & Exit Flow (Blocks 8–12) producing 2048-dim pooled representations.

### 1.4 Collaborative MixBlock Fusion
- **Module**: [`backend.models.f3net.mixblock.MixBlock`](file:///c:/projects/NEO-KYC/NeoKYC/backend/models/f3net/mixblock.py)
- **Cross-Attention**: Inserted at Middle Flow Block 7 ($C = 728, W = 19, H = 19$):
  - Projects streams into queries ($Q_{\text{FAD}}, Q_{\text{LFS}}$) and keys ($K_{\text{FAD}}, K_{\text{LFS}}$).
  - Calculates channel-wise cross-stream affinity energy:
    $$\text{Energy} = [Q_{\text{FAD}}; Q_{\text{LFS}}] \times [K_{\text{FAD}}; K_{\text{LFS}}]^T \in \mathbb{R}^{(B \cdot C) \times W \times W}$$
  - Cross-modulates features via gated depthwise residual transformations with learnable scalars initialized to 0.

---

## 2. Dependencies and Installation

### 2.1 Prerequisites
- Python 3.10+ (tested and verified on Python 3.13.6 on Windows CPU environment)
- Node.js 18+ and npm 9+ (for React Native mobile application)
- Android SDK / Expo Go (for running on Android devices)

### 2.2 Backend Environment Setup
```bash
# Clone the repository
git clone https://github.com/PranaySolanki/NeoKYC.git
cd NeoKYC

# Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install PyTorch (CPU or CUDA build):
# For CPU-only environments (Windows / Linux):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# For CUDA (if GPU available):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install verified backend dependencies
pip install -r requirements.txt
```

### 2.3 Mobile Frontend Setup
```bash
# Install node packages
npm install

# Start the Expo development server
npx expo start
```

---

## 3. Dataset Setup & Directory Structure

To train or evaluate F³-Net, use **FaceForensics++ (FF++)**, containing pristine source videos and four manipulation variants (Deepfakes, Face2Face, FaceSwap, NeuralTextures) across three compression levels (`c0` raw, `c23` high quality, `c40` low quality).

### 3.1 Directory Layout
Organize the dataset following standard FF++ structure:
```
data/
└── FaceForensics++/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/
    │           └── frames/
    │               ├── 000/
    │               │   ├── 0000.png
    │               │   └── ...
    │               └── 001/
    └── manipulated_sequences/
        ├── Deepfakes/
        │   └── c23/
        │       └── frames/
        │           └── 000_003/
        ├── Face2Face/
        │   └── c23/
        │       └── frames/
        ├── FaceSwap/
        │   └── c23/
        │       └── frames/
        └── NeuralTextures/
            └── c23/
                └── frames/
```

### 3.2 Partition Isolation (Preventing Data Leakage)
[`FaceForensicsDataset`](file:///c:/projects/NEO-KYC/NeoKYC/backend/data/faceforensics.py) uses official video-level partition splits:
- `train`: 720 source sequences (and all derived manipulation videos)
- `val`: 140 source sequences
- `test`: 140 source sequences

> [!IMPORTANT]
> All manipulated variants derived from a source video ID remain within the same split as the original source to prevent train/test identity leakage.

---

## 4. Training Commands

The standalone training script is located at [`backend/train.py`](file:///c:/projects/NEO-KYC/NeoKYC/backend/train.py).

### 4.1 Pretrained Xception Backbone Initialization
The official F³-Net architecture requires ImageNet-pretrained Xception weights for both the FAD and LFS backbones (`xception-b5690688.pth`).

#### Obtaining Pretrained Weights Manually:
1. Download the official PyTorch-compatible Xception checkpoint:
   - **Source**: `https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth`
   - **File size**: ~88 MB (91,674,713 bytes)
2. Place the downloaded checkpoint in your project:
   ```
   checkpoints/
   └── pretrained/
       └── xception-b5690688.pth
   ```
3. Pass `--pretrained_xception` during training. The loader automatically adapts `conv1` weights:
   - **FAD stream**: 3 channels $\to$ 12 channels (repeated $4\times$, scaled by $0.25$)
   - **LFS stream**: 3 channels $\to$ 6 channels (repeated $2\times$, scaled by $0.5$)
   - **Classifier**: ImageNet `fc` layer is stripped to avoid class dimension conflict.

> [!IMPORTANT]
> Pretrained weight loading is explicit and fail-fast. If `--pretrained_xception` points to a missing file or an incompatible checkpoint architecture, the training process will halt immediately with an informative error rather than silently defaulting to random weights.

### 4.2 Training with FaceForensics++
```bash
python backend/train.py \
  --dataset_dir ./data/FaceForensics++ \
  --output_dir ./checkpoints/f3net \
  --pretrained_xception ./checkpoints/pretrained/xception-b5690688.pth \
  --mode Mix \
  --compression c40 \
  --batch_size 16 \
  --epochs 10 \
  --lr 0.002
```

### 4.3 Single-Stream Training
```bash
# Train FAD stream only
python backend/train.py --dataset_dir ./data/FaceForensics++ --output_dir ./checkpoints/fad_only --pretrained_xception ./checkpoints/pretrained/xception-b5690688.pth --mode FAD

# Train LFS stream only
python backend/train.py --dataset_dir ./data/FaceForensics++ --output_dir ./checkpoints/lfs_only --pretrained_xception ./checkpoints/pretrained/xception-b5690688.pth --mode LFS
```

### 4.4 Training Smoke Test (Minimal Verification)
```bash
python backend/train.py \
  --smoke_test \
  --output_dir ./checkpoints/smoke_test \
  --epochs 2 \
  --batch_size 2
```

---

## 5. Evaluation Commands & Metrics

The evaluation module [`backend.evaluation.evaluator`](file:///c:/projects/NEO-KYC/NeoKYC/backend/evaluation/evaluator.py) provides comprehensive performance assessment:

### 5.1 Running Evaluation
```bash
python -m backend.evaluation.evaluator \
  --checkpoint ./checkpoints/f3net/best_model.pth \
  --data_root ./data/FaceForensics++ \
  --split test \
  --compression c40 \
  --calibrate_on val \
  --output ./evaluation_results.json
```

### 5.2 Metrics Computed
- **Accuracy**: Fraction of correct frame classifications.
- **Precision, Recall, F1-Score**: Real vs. Manipulated trade-offs.
- **ROC-AUC**: Area Under Receiver Operating Characteristic curve.
- **Confusion Matrix**: True Negative, False Positive, False Negative, True Positive counts.
- **Video-Level Aggregation**: Average pooling or top-$k$ maximum pooling over sampled frames.
- **Threshold Calibration**: Decision threshold is calibrated exclusively on validation split ($J$-statistic / Youden's Index) and applied unchanged to test evaluation.

---

## 6. Checkpoint Configuration

The service discovers checkpoints through configuration or environment variables:

| Setting | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `checkpoint_path` | `F3NET_CHECKPOINT_PATH` | `checkpoints/f3net/best_model.pth` | Path to trained PyTorch state dict |
| `device` | `F3NET_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `model_mode` | `F3NET_MODE` | `Mix` | `Mix`, `Both`, `FAD`, `LFS` |
| `decision_threshold` | `F3NET_THRESHOLD` | `0.5` | Threshold for classifying as fake |
| `max_video_frames` | `F3NET_MAX_VIDEO_FRAMES`| `32` | Max uniform frames sampled per video |
| `video_agg_method`| `F3NET_VIDEO_AGG` | `mean` | `mean` or `topk_mean` |

---

## 7. API Endpoints & Request/Response Examples

Start the API server:
```bash
python -m backend.api.app --port 5000
```

### 7.1 GET `/api/deepfake/health`
Checks whether the architecture and trained weights are loaded:
```json
{
  "service": "f3net_deepfake_detection",
  "status": "ready",
  "architecture_loaded": true,
  "checkpoint_loaded": true,
  "checkpoint_path": "checkpoints/f3net/best_model.pth",
  "device": "cpu",
  "face_detector": {
    "available": true,
    "type": "yunet_dnn",
    "expansion_scale": 1.3
  },
  "model_details": {
    "mode": "Mix",
    "img_size": 299,
    "decision_threshold": 0.5,
    "threshold_calibrated": false
  }
}
```

### 7.2 POST `/api/deepfake/predict-image`
Submits a biometric image (`multipart/form-data`, field: `file`):
```bash
curl -X POST http://localhost:5000/api/deepfake/predict-image \
  -F "file=@face.jpg"
```
Response:
```json
{
  "status": "success",
  "model": {
    "name": "F3-Net",
    "version": "1.0.0",
    "checkpoint_loaded": true
  },
  "faces_detected": 1,
  "predictions": [
    {
      "face_index": 0,
      "real_probability": 0.9421,
      "fake_probability": 0.0579,
      "predicted_label": "real",
      "threshold_used": 0.5,
      "is_above_threshold": false
    }
  ],
  "processing_time_ms": 142.35
}
```

### 7.3 POST `/api/deepfake/predict-video`
Submits a video selfie (`multipart/form-data`, field: `file`):
```bash
curl -X POST http://localhost:5000/api/deepfake/predict-video \
  -F "file=@liveness_clip.mp4"
```
Response:
```json
{
  "status": "success",
  "model": { "name": "F3-Net", "version": "1.0.0", "checkpoint_loaded": true },
  "video_analysis": {
    "frames_analyzed": 16,
    "frames_with_faces": 16,
    "aggregation_method": "mean",
    "video_score": 0.0812,
    "video_label": "real",
    "threshold_used": 0.5
  },
  "processing_time_ms": 1280.45
}
```

### 7.4 POST `/api/ekyc/verify`
Full multi-stage identity verification:
```bash
curl -X POST http://localhost:5000/api/ekyc/verify \
  -F "face_image=@live_selfie.jpg" \
  -F "document_name=John Doe" \
  -F "document_dob=01/01/1990" \
  -F "document_gender=MALE" \
  -F "document_id_number=123456789012"
```
Response:
```json
{
  "decision": "review_required",
  "checks_completed": [
    {
      "check_name": "document_ocr",
      "status": "passed",
      "message": "Document fields validated."
    },
    {
      "check_name": "deepfake_detection",
      "status": "passed",
      "details": {
        "fake_probability": 0.042,
        "real_probability": 0.958,
        "threshold_used": 0.5,
        "threshold_calibrated": false
      },
      "message": "No manipulation signal detected (score: 0.0420). Note: threshold is uncalibrated."
    },
    {
      "check_name": "liveness_detection",
      "status": "unavailable",
      "message": "Liveness detection is not yet implemented in the backend."
    }
  ],
  "checks_unavailable": [
    "liveness_detection",
    "face_identity_match",
    "graph_fraud_analysis"
  ],
  "message": "Available checks passed but some checks were unavailable: liveness_detection, face_identity_match, graph_fraud_analysis. Consider manual review.",
  "processing_time_ms": 178.60
}
```

---

## 8. eKYC Integration Flow & Zero-Trust Verification

The end-to-end eKYC pipeline implements a defense-in-depth, zero-trust verification hierarchy:

```
[User App: DocumentScanScreen]
      │ Capture card photo + ML Kit on-device OCR
      ▼
[User App: FaceLivenessScreen]
      │ Capture live face frame / short selfie video
      ▼
[POST /api/ekyc/verify]
      │
      ├─► 1. Document OCR Check: Validate Name, DOB, Gender, ID Number
      │
      ├─► 2. Face Detection & Bounding Box Expansion (1.3x)
      │
      ├─► 3. Passive Liveness Check (Anti-Spoofing / Replay prevention)
      │
      ├─► 4. F³-Net Multi-Spectral Deepfake Detection
      │        (Frequency-Aware Decomposition + Local Frequency Statistics)
      │
      ├─► 5. ArcFace 1:1 Biometric Face Match (ID photo vs. Live Selfie)
      │
      ├─► 6. Neo4j Graph Threat Intelligence (Sybil / Ring Fraud Detection)
      │
      ▼
[EKYCDecisionEngine Policy]
      │
      ├─► REJECTED: Any hard failure (invalid ID, fraud graph match)
      ├─► REVIEW_REQUIRED: Suspicious deepfake score or unavailable checks
      ├─► INCOMPLETE: Missing required inputs or unavailable critical checks
      └─► APPROVED: ALL checks present, active, and passed
```

### Policy & Decision Rules:
1. **No Single-Signal Auto-Approval**: Low deepfake score does *not* grant automated approval.
2. **Explicit Fallback on Missing Model**: If the deepfake checkpoint is missing, the service flags `model_unavailable` and routes to `review_required` or `incomplete` (if `deepfake_required=True`).
3. **Auditability**: Every check records its duration, confidence, threshold applied, and status for compliance logs.

---

## 9. Privacy & Zero-Trust Data Protection

Biometric data handling adheres strictly to data minimization and ephemeral processing standards:
- **Zero Disk Biometric Persistence**: Biometric images uploaded to `/api/deepfake/predict-image` and `/api/ekyc/verify` are decoded in-memory via `io.BytesIO` and PIL. No face crops or raw selfies are written to disk.
- **Ephemeral Video Buffer**: Uploaded video files for `/api/deepfake/predict-video` are written to an isolated temporary file descriptor via `tempfile.mkstemp` and **unconditionally unlinked** in a `finally` block immediately after frame extraction.
- **No Biometric Data in Logs**: Log messages never output raw base64 data, biometrics, or personally identifiable strings. Only dimensions, counts, and status codes are logged.
- **Secret Hygiene**: No hardcoded API keys or credentials. All connection parameters (Neo4j, ports, thresholds) are sourced via environment variables.

---

## 10. Known Limitations & Remaining Work

### 10.1 Checkpoint Status
- The architecture, dataset pipeline, inference service, and integration tests are verified and operational.
- The repository currently contains a 2-epoch CPU smoke-test checkpoint (`checkpoints/smoke_test/best_model.pth`).
- **No full-scale FaceForensics++ checkpoint is currently bundled**. A full 30-epoch training run on FF++ (c23/c40) requires an NVIDIA GPU (e.g., RTX 3090/4090 or A100) and ~12–24 hours of compute.

### 10.2 Threshold Calibration
- The current decision threshold (`0.5`) is a heuristic default. Prior to production deployment, run `evaluator.py` with `--calibrate_on val` on a labeled dataset to set the operating point (e.g., FPR $\le 1.0\%$).

### 10.3 Downstream eKYC Modules
- **Liveness Detection**: Face motion/texture passive liveness backend endpoint is pending implementation.
- **ArcFace Face Matching**: 1:1 cosine distance face matching between ID photo and selfie is pending backend model loading.
- **Neo4j Threat Graph**: Graph fraud query integration is stubbed with zero-trust safety fallbacks.
- **Mobile Frontend Hookup**: Connect `FaceLivenessScreen` and `ResultScreen` to consume `/api/ekyc/verify`.
