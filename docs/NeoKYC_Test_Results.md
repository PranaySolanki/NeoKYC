# NeoKYC Test Results and Evidence

> **Audit Date:** October 2026
> **Test Framework:** Python `unittest` (standard library)
> **Test Runner Command:** `python -m unittest discover -s backend/tests -v`

---

## 1. Test Suite Inventory

The repository contains **4 test files** with a combined total of **90 test cases** across **22 test classes**.

| Test File | Classes | Tests | Focus |
|:---|:---:|:---:|:---|
| `test_f3net_components.py` | 3 | 12 | DCT, FAD, LFS unit tests |
| `test_f3net_architecture.py` | 3 | 15 | Xception, MixBlock, F3Net integration |
| `test_training_pipeline.py` | 4 | 7 | Transforms, metrics, dataset splits, trainer |
| `test_api.py` | 12 | 56 | API endpoints, inference service, eKYC pipeline, regression |
| **Total** | **22** | **90** | |

---

## 2. Test File: `test_f3net_components.py` (12 tests)

### 2.1 TestDCTUtilities (4 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_orthogonality` | DCT matrix D @ D.T ≈ I | Max deviation < 1e-5 for sizes 8, 10, 16, 299 |
| `test_reconstruction_accuracy` | idct2(dct2(x)) ≈ x | Max error < 1e-4 for shapes (2,3,16,16), (1,1,32,32), (2,3,299,299) |
| `test_dct_gradient_propagation` | Autograd through DCT/IDCT | Gradient is finite and matches identity (all ones) |
| `test_dct_layer` | DCT2DLayer forward + inverse | Shape preservation and reconstruction accuracy |

### 2.2 TestFADComponent (5 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_filter_mask_generation` | Band mask partitioning | Binary masks, non-overlapping bands |
| `test_norm_sigma_bounds` | Learnable weight bounding | Output strictly in (-1, 1) for moderate inputs |
| `test_fad_output_dimensions_and_finiteness` | FAD_Head shape/finiteness | Output (2, 12, 32, 32), all values finite |
| `test_fad_gradient_flow` | Backprop to input + filters | Non-None finite gradients on input and all learnable params |
| `test_fad_standard_299_dimensions` | Full-size input | Output (1, 12, 299, 299) |

### 2.3 TestLFSComponent (3 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_lfs_synthetic_dimensions_and_finiteness` | Custom LFS dimensions | Output (2, 6, 16, 16) for 32×32 input |
| `test_lfs_standard_299_dimension_149x149x6` | Paper-exact dimensions | Output (1, 6, 149, 149) for 299×299 input |
| `test_lfs_gradient_flow` | Backprop through unfold + DCT + log + filters | Non-None finite gradients |

---

## 3. Test File: `test_f3net_architecture.py` (15 tests)

### 3.1 TestXceptionBackbone (6 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_fad_branch_forward_and_stages` | 12-ch FAD input through stages | Stage 0-7: (2,728,19,19), Stage 8-12: (2,2048,10,10), Pool: (2,2048) |
| `test_lfs_branch_forward_and_stages` | 6-ch LFS 149×149 input | Same stage shapes, confirms FAD/LFS alignment |
| `test_imagenet_conv1_expansion_adaptation` | 3→12 channel weight expansion | conv1 weight shape (32,12,3,3) after loading mock state dict |
| `test_pretrained_weights_file_and_wrapper_loading` | .pth file with module. prefix | Loads from tempfile, unwraps 'state_dict'+'module.' prefixes |
| `test_missing_pretrained_file_raises_filenotfound` | Missing file error handling | Raises FileNotFoundError with filename in message |
| `test_incompatible_checkpoint_raises_value_error` | Bad checkpoint format | Raises ValueError for non-dict, missing conv1, wrong shape |

### 3.2 TestMixBlock (2 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_mixblock_dimensions` | Shape preservation | Both outputs (2, 728, 19, 19), finite values |
| `test_mixblock_gradients` | Cross-attention backprop | Gradients on both inputs and gamma parameters |

### 3.3 TestF3NetIntegration (6 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_full_model_mix_mode` | End-to-end Mix mode | Logits (2,2), features (2,4096), finite gradients |
| `test_predict_proba` | Probability normalization | Probs sum to 1.0, all in [0,1] |
| `test_all_operational_modes` | Both, FAD, LFS modes | Correct logit/feature shapes per mode |
| `test_device_compatibility` | CPU (and CUDA if available) | Output tensor on correct device |
| `test_f3net_pretrained_initialization_and_channel_adaptation` | Channel scaling math | FAD: all 0.25 (3/12), LFS: all 0.5 (3/6) |
| `test_f3net_missing_pretrained_file_raises` | Missing pretrained path | FileNotFoundError raised |
| `test_f3net_incompatible_checkpoint_raises` | Wrong architecture weights | ValueError raised |

---

## 4. Test File: `test_training_pipeline.py` (7 tests)

### 4.1 TestDataTransforms (2 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_expand_bounding_box` | 1.3× expansion + clamping | Exact bbox: (85,85,215,215), edge clamping ≥ 0 |
| `test_preprocessing_pipeline` | Full F3NetPreprocessing | Shape (3,299,299), values in [-1.05, 1.05] |

### 4.2 TestMetrics (2 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_roc_auc_perfect_prediction` | AUC = 1.0 for perfect ranking | Asserts ≈ 1.0 |
| `test_binary_metrics_accuracy_and_auc` | Full metric computation | accuracy=1.0, auc=1.0, f1=1.0 for trivial case |

### 4.3 TestSplitAntiLeakage (1 test)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_anti_leakage_guarantee` | Source '001' + manipulated '001_892' in same split | Train has 2 samples, val has 0 |

### 4.4 TestTrainerPretrainedWeights (2 tests)

| Test | Purpose | Verification Method |
|:---|:---|:---|
| `test_trainer_missing_pretrained_weights_raises_filenotfound` | Fail-fast on missing weights | FileNotFoundError with filename |
| `test_trainer_with_valid_pretrained_weights_file` | Valid pretrained loading | conv1 shapes: (32,12,3,3) FAD, (32,6,3,3) LFS |

---

## 5. Test File: `test_api.py` (56 tests)

### 5.1 TestDeepfakeServiceConfig (3 tests)
- Default values, auto device resolution, explicit device

### 5.2 TestFaceDetector (3 tests)
- Initialization, blank image detection, PIL detect_and_crop

### 5.3 TestHealthEndpoint (3 tests)
- Returns 200, response structure, degraded without checkpoint

### 5.4 TestPredictImageEndpoint (5 tests)
- Missing file (400), empty filename (400), unsupported format (400), empty file (400), model unavailable (503)

### 5.5 TestPredictVideoEndpoint (4 tests)
- Missing file (400), unsupported format (400), empty video (400), model unavailable (503)

### 5.6 TestInferenceServiceWithCheckpoint (4 tests)
- Health ready, no-face detection, corrupted file error, response structure

### 5.7 TestInferenceServiceDirect (4 tests)
- Service without checkpoint, health status structure, image unavailable, video unavailable

### 5.8 TestEKYCPipelineModelUnavailable (5 tests)
- Deepfake unavailable, full pipeline with documents, required deepfake → INCOMPLETE, no inputs, response structure

### 5.9 TestEKYCPipelineWithCheckpoint (5 tests)
- No-face error, valid document no face, failed document → REJECTED, liveness/face_match/graph stubs unavailable

### 5.10 TestEKYCEndpoints (5 tests)
- Health 200, health structure, verify no inputs, verify with documents, verify with face image, unsupported format

### 5.11 TestExistingModulesRegression (7 tests)
- Import tests for F3Net, TrainingConfig, Trainer, evaluation metrics, transforms, FaceForensicsDataset, evaluator

### 5.12 TestRegressionFixes (6 tests)
- Default checkpoint path, repo-root resolution, absolute path preservation, env var override, TrainingConfig paths, video cleanup

---

## 6. Known Test Limitations

> [!WARNING]
> The following limitations apply to all test results:

### 6.1 No Real Data Testing
- All tests use **synthetic tensors** (random `torch.randn`), **solid-color images** (`PIL.Image.new`), or **mock checkpoints** (randomly initialized model weights)
- No test uses actual facial images, real deepfake samples, or FaceForensics++ data
- Successful forward passes through the model do **not** establish deepfake detection accuracy

### 6.2 Smoke Test Checkpoint
- The smoke test checkpoint (`checkpoints/smoke_test/best_model.pth`) was trained on **4 random tensors** for **2 epochs**
- Reported metrics (Val AUC: 1.0) are artifacts of the trivially small dataset (1 positive, 1 negative sample)
- This checkpoint has **zero real-world detection capability**

### 6.3 No Real Evaluation Results
- The `F3NetEvaluator` module is implemented but has **never been executed** on labeled data
- No accuracy, precision, recall, F1, or ROC-AUC figures from real deepfake data exist
- The decision threshold (0.5) is a **heuristic default** that has not been calibrated

### 6.4 Face Detection Coverage
- Tests verify face detector initialization and blank-image behavior
- No tests use images with actual faces — face detection pipeline coverage is limited to negative cases

---

## 7. Metrics Pending Real Evaluation

The following table lists performance metrics that **must be measured** once a genuine trained checkpoint and labeled evaluation dataset are available:

| Metric | Level | Status |
|:---|:---|:---|
| ROC-AUC | Frame-level | **Pending** |
| ROC-AUC | Video-level | **Pending** |
| Accuracy | Frame-level | **Pending** |
| Accuracy | Video-level | **Pending** |
| Precision | Frame-level | **Pending** |
| Recall (Sensitivity) | Frame-level | **Pending** |
| Specificity (TNR) | Frame-level | **Pending** |
| F1-Score | Frame-level | **Pending** |
| Confusion Matrix (TP/FP/TN/FN) | Both levels | **Pending** |
| Calibrated Threshold | Validation-set | **Pending** |
| Inference Latency (ms/image) | CPU and GPU | **Pending** |
| Inference Latency (ms/video) | CPU and GPU | **Pending** |
| Per-manipulation accuracy | Frame-level | **Pending** |
| Cross-compression generalization | c23 → c40 | **Pending** |

---

## 8. Smoke Test Training History

The only existing training record (`checkpoints/smoke_test/history.json`):

| Epoch | Train Loss | Val Loss | Val Accuracy | Val AUC | Val F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.6372 | 0.6931 | 0.500 | 1.000 | 0.667 |
| 2 | 0.7357 | 0.6931 | 0.500 | 1.000 | 0.667 |

> [!CAUTION]
> These metrics are **meaningless** for evaluating deepfake detection capability. The validation set contained exactly 2 samples (1 real, 1 fake), both generated from `torch.randn()`. The AUC of 1.0 reflects random chance separation on a trivial dataset, not model quality.
