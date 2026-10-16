# NeoKYC Architecture Diagrams

All diagrams in this document are rendered using Mermaid and reflect the **actual implementation** found in the repository as of October 2026. Components labelled **[PLANNED]** are referenced in the codebase but have stub or placeholder implementations.

---

## Diagram 1: High-Level System Architecture

```mermaid
graph TB
    subgraph "Mobile Frontend (React Native / Expo)"
        A1["HomeScreen<br/>(Navigation Hub)"]
        A2["DocumentScanScreen<br/>(Camera + ML Kit OCR)"]
        A3["FaceLivenessScreen<br/>[STUB]"]
        A4["ResultScreen<br/>[STUB]"]
    end

    subgraph "Backend REST API (Flask)"
        B1["GET /api/deepfake/health"]
        B2["POST /api/deepfake/predict-image"]
        B3["POST /api/deepfake/predict-video"]
        B4["GET /api/ekyc/health"]
        B5["POST /api/ekyc/verify"]
    end

    subgraph "Core Services"
        C1["F3NetInferenceService"]
        C2["FaceDetector (YuNet/Haar)"]
        C3["EKYCPipeline Orchestrator"]
        C4["F3NetPreprocessing"]
    end

    subgraph "F³-Net Model Architecture"
        D1["FAD_Head (12-ch)"]
        D2["LFS_Head (6-ch)"]
        D3["XceptionBackbone × 2"]
        D4["MixBlock Cross-Attention"]
        D5["Linear Classifier (2)"]
    end

    subgraph "Training & Evaluation Pipeline"
        E1["Trainer (SGD + CosineAnnealing)"]
        E2["FaceForensicsDataset"]
        E3["F3NetEvaluator"]
        E4["Threshold Calibration"]
    end

    subgraph "Planned Components [NOT IMPLEMENTED]"
        F1["Liveness Detection Backend"]
        F2["ArcFace Face Matching"]
        F3["Neo4j Graph Fraud Intelligence"]
        F4["Vector Embedding Search"]
    end

    A1 --> A2
    A1 --> A3
    A1 --> A4
    A2 -->|"OCR data"| B5
    A3 -.->|"face image"| B5
    B2 --> C1
    B3 --> C1
    B5 --> C3
    C3 --> C1
    C3 -.-> F1
    C3 -.-> F2
    C3 -.-> F3
    C1 --> C2
    C1 --> C4
    C1 --> D1
    D1 --> D3
    D2 --> D3
    D3 --> D4
    D4 --> D5
    E1 --> D1
    E2 --> E1
    E3 --> E4
```

**Figure 1.** High-level system architecture of NeoKYC. Solid arrows indicate implemented data flows; dashed arrows indicate planned but not yet implemented connections.

---

## Diagram 2: F³-Net Dual-Stream Architecture

```mermaid
graph TB
    INPUT["Input RGB Image<br/>(N, 3, 299, 299)"]

    subgraph "Stream 1: Frequency-Aware Decomposition"
        FAD_DCT["Global 2D-DCT<br/>(299×299)"]
        FAD_FILT["4 Frequency Filters<br/>(Low, Mid, High, All)"]
        FAD_IDCT["2D-IDCT Reconstruction"]
        FAD_OUT["FAD Output<br/>(N, 12, 299, 299)"]
        FAD_XCEP_ENTRY["Xception Entry Flow<br/>(conv1: 12→32, stride 2)"]
        FAD_XCEP_MID1["Middle Flow Blocks 1–7<br/>(N, 728, 19, 19)"]
    end

    subgraph "Stream 2: Local Frequency Statistics"
        LFS_GRAY["RGB → Grayscale"]
        LFS_UNFOLD["Sliding Window Unfold<br/>(10×10, stride=2, pad=4)"]
        LFS_PATCH_DCT["Patch 2D-DCT (10×10)"]
        LFS_LOG["Log Spectral Energy<br/>log₁₀(|DCT| + ε)"]
        LFS_BAND["6 Bandpass Filters<br/>(Normalized)"]
        LFS_OUT["LFS Output<br/>(N, 6, 149, 149)"]
        LFS_XCEP_ENTRY["Xception Entry Flow<br/>(conv1: 6→32, stride 1)"]
        LFS_XCEP_MID1["Middle Flow Blocks 1–7<br/>(N, 728, 19, 19)"]
    end

    subgraph "Collaborative Fusion"
        MIX["MixBlock<br/>Cross-Attention<br/>(728-ch, 19×19)"]
    end

    subgraph "Exit Flow & Classification"
        FAD_EXIT["FAD Blocks 8–12<br/>+ Exit Flow"]
        LFS_EXIT["LFS Blocks 8–12<br/>+ Exit Flow"]
        FAD_POOL["AdaptiveAvgPool<br/>(N, 2048)"]
        LFS_POOL["AdaptiveAvgPool<br/>(N, 2048)"]
        CONCAT["Concatenation<br/>(N, 4096)"]
        DROP["Dropout (p=0.5)"]
        FC["Linear(4096, 2)"]
        OUTPUT["Logits: [Real, Fake]"]
    end

    INPUT --> FAD_DCT
    INPUT --> LFS_GRAY
    FAD_DCT --> FAD_FILT
    FAD_FILT --> FAD_IDCT
    FAD_IDCT --> FAD_OUT
    FAD_OUT --> FAD_XCEP_ENTRY
    FAD_XCEP_ENTRY --> FAD_XCEP_MID1

    LFS_GRAY --> LFS_UNFOLD
    LFS_UNFOLD --> LFS_PATCH_DCT
    LFS_PATCH_DCT --> LFS_LOG
    LFS_LOG --> LFS_BAND
    LFS_BAND --> LFS_OUT
    LFS_OUT --> LFS_XCEP_ENTRY
    LFS_XCEP_ENTRY --> LFS_XCEP_MID1

    FAD_XCEP_MID1 --> MIX
    LFS_XCEP_MID1 --> MIX

    MIX --> FAD_EXIT
    MIX --> LFS_EXIT
    FAD_EXIT --> FAD_POOL
    LFS_EXIT --> LFS_POOL
    FAD_POOL --> CONCAT
    LFS_POOL --> CONCAT
    CONCAT --> DROP
    DROP --> FC
    FC --> OUTPUT
```

**Figure 2.** Detailed F³-Net dual-stream architecture as implemented in `backend/models/f3net/`. All modules shown are fully implemented and verified via unit tests.

---

## Diagram 3: eKYC Verification Pipeline

```mermaid
sequenceDiagram
    participant User as Mobile App
    participant API as Flask API
    participant EKYC as EKYCPipeline
    participant DF as F3NetInferenceService
    participant FD as FaceDetector
    participant LV as Liveness [STUB]
    participant FM as FaceMatch [STUB]
    participant GF as GraphFraud [STUB]

    User->>API: POST /api/ekyc/verify<br/>(face_image + document fields)
    API->>EKYC: run_verification()

    Note over EKYC: Step 1: Document OCR Check
    EKYC->>EKYC: Validate name, dob, gender, id_number

    Note over EKYC: Step 2: Deepfake Detection
    EKYC->>DF: run_deepfake_check(face_bytes)
    DF->>FD: detect_and_crop(image)
    FD-->>DF: face crops[]
    DF->>DF: preprocess → F3Net inference
    DF-->>EKYC: CheckResult(PASSED/SUSPICIOUS/UNAVAILABLE)

    Note over EKYC: Step 3: Liveness [NOT IMPLEMENTED]
    EKYC->>LV: run_liveness_check()
    LV-->>EKYC: CheckResult(UNAVAILABLE)

    Note over EKYC: Step 4: Face Match [NOT IMPLEMENTED]
    EKYC->>FM: run_face_match_check()
    FM-->>EKYC: CheckResult(UNAVAILABLE)

    Note over EKYC: Step 5: Graph Fraud [NOT IMPLEMENTED]
    EKYC->>GF: run_graph_fraud_check()
    GF-->>EKYC: CheckResult(UNAVAILABLE)

    Note over EKYC: Step 6: Decision Aggregation
    EKYC->>EKYC: _make_decision(checks, unavailable)
    EKYC-->>API: EKYCVerificationResult
    API-->>User: JSON response with decision
```

**Figure 3.** Sequence diagram for the eKYC verification pipeline. Steps 3–5 are currently stubs returning `UNAVAILABLE` status.

---

## Diagram 4: Document Processing Workflow (Mobile App)

```mermaid
flowchart TD
    A["User opens DocumentScanScreen"] --> B["Request camera permission"]
    B -->|Granted| C["Display CameraView<br/>with alignment guide"]
    B -->|Denied| Z["Show permission prompt"]

    C --> D["User taps 'Capture & Extract'"]
    D --> E["takePictureAsync()"]
    E --> F["ML Kit recognizeText()"]
    F --> G["parseMLKitResult()"]

    G --> H{"All fields<br/>extracted?"}
    H -->|Yes| I["Display extracted fields:<br/>Name, DOB, Gender, ID Number"]
    H -->|No| J["Show warning:<br/>Missing fields"]

    I --> K["User confirms checkbox"]
    K --> L["Navigate to FaceLivenessScreen"]

    J --> M["User taps 'Retake Picture'"]
    M --> C
```

**Figure 4.** Document capture and OCR workflow as implemented in `DocumentScanScreen.tsx`. The OCR parsing uses spatial-aware line extraction with ML Kit Text Recognition.

---

## Diagram 5: Deployment Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        PHONE["Android Device<br/>(Expo / React Native)"]
    end

    subgraph "Application Layer"
        FLASK["Flask API Server<br/>(Port 5000)"]
        F3NET["F³-Net Model<br/>(PyTorch CPU/CUDA)"]
        YUNET["YuNet Face Detector<br/>(ONNX via OpenCV DNN)"]
    end

    subgraph "Data Layer"
        CKPT["Checkpoint Files<br/>(checkpoints/f3net/)"]
        ONNX["YuNet ONNX Model<br/>(_models/)"]
    end

    subgraph "Planned Infrastructure [NOT IMPLEMENTED]"
        NEO4J["Neo4j Graph DB"]
        VECTOR["Vector Store"]
        GPU["GPU Training Server"]
    end

    PHONE -->|"HTTP REST"| FLASK
    FLASK --> F3NET
    FLASK --> YUNET
    F3NET --> CKPT
    YUNET --> ONNX
    FLASK -.-> NEO4J
    FLASK -.-> VECTOR
    F3NET -.-> GPU
```

**Figure 5.** Deployment topology. The current system runs as a single Flask server with CPU-based PyTorch inference. GPU training, Neo4j, and vector store are planned components.

---

## Diagram 6: Component Diagram

```mermaid
graph LR
    subgraph "backend/api/"
        APP["app.py<br/>(Flask Routes)"]
        CFG["config.py<br/>(DeepfakeServiceConfig)"]
        INF["inference_service.py<br/>(F3NetInferenceService)"]
        FD["face_detector.py<br/>(FaceDetector)"]
        EKYC["ekyc_pipeline.py<br/>(EKYCPipeline)"]
    end

    subgraph "backend/models/f3net/"
        F3["f3net.py"]
        DCT["dct.py"]
        FAD["fad.py"]
        LFS["lfs.py"]
        MIX["mixblock.py"]
        XCEP["xception.py"]
    end

    subgraph "backend/data/"
        FF["faceforensics.py"]
        TR["transforms.py"]
    end

    subgraph "backend/training/"
        TRAINER["trainer.py"]
        TCFG["config.py"]
        TMET["metrics.py"]
    end

    subgraph "backend/evaluation/"
        EVAL["evaluator.py"]
        EMET["metrics.py"]
    end

    APP --> CFG
    APP --> INF
    APP --> EKYC
    INF --> F3
    INF --> FD
    INF --> TR
    INF --> EMET
    EKYC --> INF
    F3 --> FAD
    F3 --> LFS
    F3 --> XCEP
    F3 --> MIX
    FAD --> DCT
    LFS --> DCT
    TRAINER --> F3
    TRAINER --> TCFG
    TRAINER --> TMET
    FF --> TR
    EVAL --> F3
    EVAL --> EMET
```

**Figure 6.** Component dependency diagram of the backend Python packages. All shown components are implemented.

---

## Diagram 7: Use Case Diagram

```mermaid
graph TB
    subgraph "Actors"
        USER["End User<br/>(KYC Applicant)"]
        ADMIN["System Admin"]
        REVIEWER["Human Reviewer"]
    end

    subgraph "Implemented Use Cases"
        UC1["Capture ID Document"]
        UC2["Extract Document Fields (OCR)"]
        UC3["Confirm Extracted Details"]
        UC4["Check Deepfake Health Status"]
        UC5["Analyze Image for Deepfake"]
        UC6["Analyze Video for Deepfake"]
        UC7["Run eKYC Verification"]
        UC8["Run Training Smoke Test"]
    end

    subgraph "Planned Use Cases"
        UC9["Perform Liveness Check"]
        UC10["Match Face to Document Photo"]
        UC11["Query Neo4j Fraud Graph"]
        UC12["View Verification Result"]
        UC13["Train on FaceForensics++"]
        UC14["Calibrate Decision Threshold"]
    end

    USER --> UC1
    USER --> UC2
    USER --> UC3
    USER --> UC7
    ADMIN --> UC4
    ADMIN --> UC8
    ADMIN --> UC13
    ADMIN --> UC14
    REVIEWER --> UC12
    UC7 --> UC5
    UC7 -.-> UC9
    UC7 -.-> UC10
    UC7 -.-> UC11
```

**Figure 7.** Use case diagram. Solid connections represent implemented use cases; dashed connections represent planned use cases.

---

## Diagram 8: Training Pipeline Data Flow

```mermaid
flowchart TD
    A["FaceForensics++ Dataset<br/>(Not Yet Populated)"] --> B["FaceForensicsDataset<br/>(Split Isolation: 720/140/140)"]
    B --> C["F3NetPreprocessing<br/>(Resize 299×299, Normalize [-1,1])"]
    C --> D["DataLoader<br/>(batch_size=16, shuffle=True)"]
    D --> E["F3Net Model<br/>(Mode: Mix)"]
    E --> F["CrossEntropyLoss"]
    F --> G["SGD Optimizer<br/>(lr=0.002, momentum=0.9)"]
    G --> H["Gradient Clipping<br/>(max_norm=5.0)"]
    H --> I["CosineAnnealingLR<br/>(T_max=epochs)"]
    I --> J["Validation Loop<br/>(compute_binary_metrics)"]
    J --> K{"Best Val AUC?"}
    K -->|Yes| L["Save best_model.pth"]
    K -->|No| M["Save latest_checkpoint.pth"]
    L --> N["Save history.json"]
    M --> N

    style A fill:#ff9999,stroke:#cc0000
    style B fill:#ffcc99
```

**Figure 8.** Training pipeline data flow. The FaceForensics++ dataset (highlighted in red) has not yet been populated; only a synthetic smoke test has been executed.
