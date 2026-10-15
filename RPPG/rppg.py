import argparse
import csv
from datetime import datetime
import os
import sys
import warnings

# Suppress internal TensorFlow / MediaPipe C++ log spam
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

# MediaPipe imports supporting both modern tasks API and legacy solutions API
try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except (ImportError, AttributeError):
    mp_python = None
    mp_vision = None


MIN_BPM = 45.0
MAX_BPM = 180.0
WINDOW_SECONDS = 1.6
RECOMMENDED_MIN_DURATION_SEC = 30.0
ABSOLUTE_MIN_DURATION_SEC = 20.0
DEFAULT_MAX_GAP_SEC = 1.0

# Anatomically defined MediaPipe Face Mesh landmark indices
# Forehead: Glabella to upper forehead between eyebrows and hairline
FOREHEAD_POLYGON = [10, 338, 297, 336, 9, 107, 68, 109]
# Left cheek (subject's right / camera left)
LEFT_CHEEK_POLYGON = [117, 118, 101, 205, 187, 147, 123, 50]
# Right cheek (subject's left / camera right)
RIGHT_CHEEK_POLYGON = [346, 347, 330, 425, 411, 376, 352, 280]


# ---------------- FACE DETECTOR / BACKENDS ----------------

class FaceDetector:
    """
    Face detector supporting:
      1. MediaPipe Tasks FaceLandmarker (MediaPipe >= 1.0, 478 landmarks)
      2. Legacy MediaPipe Solutions FaceMesh (MediaPipe <= 0.10)
      3. OpenCV Haar Cascade (Fast baseline fallback)
    """

    def __init__(self, mode="auto", model_path="face_landmarker.task"):
        self.mode = mode.lower()
        self.model_path = model_path
        self.backend = None
        self.detector = None
        self.haar_cascade = None

        if not os.path.isabs(self.model_path) and not os.path.exists(self.model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, self.model_path)
            if os.path.exists(candidate):
                self.model_path = candidate

        self._init_backend()

    def _init_backend(self):
        # 1. MediaPipe Tasks FaceLandmarker (Preferred landmark tracking)
        if self.mode in ("auto", "mediapipe"):
            if mp_vision is not None and mp_python is not None and os.path.exists(self.model_path):
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        num_faces=1,
                        min_face_detection_confidence=0.2,
                        min_face_presence_confidence=0.2,
                        min_tracking_confidence=0.2,
                    )
                    self.detector = mp_vision.FaceLandmarker.create_from_options(options)
                    self.backend = "mediapipe_tasks"
                    return
                except Exception:
                    pass

            # 2. Legacy MediaPipe Solutions FaceMesh
            if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                try:
                    self.detector = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.backend = "mediapipe_solutions"
                    return
                except Exception:
                    pass

        # 3. OpenCV Haar Cascade (Always available with OpenCV)
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar_cascade = cv2.CascadeClassifier(cascade_file)
        self.backend = "opencv_haar"

    def detect(self, frame, rgb_frame):
        """
        Detects face and returns:
          {"type": "landmarks", "landmarks": lm} or
          {"type": "bbox", "bbox": (x, y, w, h)} or
          None
        """
        # 1. MediaPipe Tasks
        if self.backend == "mediapipe_tasks" and self.detector is not None:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.detector.detect(mp_img)
                if res.face_landmarks and len(res.face_landmarks) > 0:
                    return {"type": "landmarks", "landmarks": res.face_landmarks[0]}
            except Exception:
                pass

        # 2. MediaPipe Solutions
        elif self.backend == "mediapipe_solutions" and self.detector is not None:
            try:
                res = self.detector.process(rgb_frame)
                if res.multi_face_landmarks:
                    return {"type": "landmarks", "landmarks": res.multi_face_landmarks[0]}
            except Exception:
                pass

        # 3. Haar Cascade fallback
        if self.haar_cascade is None:
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_file)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return {"type": "bbox", "bbox": (int(x), int(y), int(w), int(h))}

        return None

    def close(self):
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass


# ---------------- ROI SKIN EXTRACTION ----------------

def extract_roi_patch(frame, landmarks, poly_indices):
    """
    Extracts mean RGB from a fast cropped landmark polygon patch.
    """
    h, w = frame.shape[:2]
    # Extract landmark coordinates
    if hasattr(landmarks, "landmark"):
        pts_norm = [[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in poly_indices]
    else:
        pts_norm = [[landmarks[i].x, landmarks[i].y] for i in poly_indices]

    pts = np.array([[int(px * w), int(py * h)] for px, py in pts_norm], dtype=np.int32)
    x_min, y_min = np.clip(pts.min(axis=0), [0, 0], [w - 1, h - 1])
    x_max, y_max = np.clip(pts.max(axis=0), [0, 0], [w - 1, h - 1])

    if x_max - x_min < 5 or y_max - y_min < 5:
        return None

    crop = frame[y_min:x_max + 1, x_min:x_max + 1]
    if crop.size == 0:
        return None

    local_pts = pts - [x_min, y_min]
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts], 255)

    pixels = crop[mask == 255]
    if len(pixels) < 20:
        return None

    # Filter dark and saturated illumination artifacts
    brightness = pixels.mean(axis=1)
    valid = (brightness > 20) & (brightness < 245)
    if valid.sum() < 20:
        return None

    mean_bgr = pixels[valid].mean(axis=0)
    return mean_bgr[::-1]  # Return RGB


def extract_rgb_from_landmarks(frame, landmarks):
    """
    Extracts mean RGB from anatomically aligned Forehead, Left Cheek, and Right Cheek.
    Averages the three region means with equal spatial weighting.
    """
    region_means = []
    for poly in [FOREHEAD_POLYGON, LEFT_CHEEK_POLYGON, RIGHT_CHEEK_POLYGON]:
        patch_rgb = extract_roi_patch(frame, landmarks, poly)
        if patch_rgb is not None:
            region_means.append(patch_rgb)

    if not region_means:
        return None

    return np.mean(region_means, axis=0)


def extract_rgb_from_box(frame, x, y, face_w, face_h):
    """
    Fallback skin extraction using relative percentage patches when landmarks are unavailable.
    """
    height, width = frame.shape[:2]
    if face_w < 40 or face_h < 40:
        return None

    regions = [
        (0.25, 0.12, 0.75, 0.30),  # Forehead
        (0.12, 0.43, 0.37, 0.63),  # Left cheek
        (0.63, 0.43, 0.88, 0.63),  # Right cheek
    ]

    rgb_values = []
    for x1, y1, x2, y2 in regions:
        left = max(0, x + int(x1 * face_w))
        right = min(width, x + int(x2 * face_w))
        top = max(0, y + int(y1 * face_h))
        bottom = min(height, y + int(y2 * face_h))

        patch = frame[top:bottom, left:right]
        if patch.size == 0:
            continue

        pixels = patch.reshape(-1, 3)
        brightness = pixels.mean(axis=1)
        valid = (brightness > 20) & (brightness < 245)
        if valid.sum() < 20:
            continue

        mean_bgr = pixels[valid].mean(axis=0)
        rgb_values.append(mean_bgr[::-1])

    if not rgb_values:
        return None

    return np.mean(rgb_values, axis=0)


def extract_rgb(frame, face_landmarks):
    """
    Backward compatibility wrapper.
    """
    return extract_rgb_from_landmarks(frame, face_landmarks)


# ---------------- POS ALGORITHM ----------------

def pos_signal(rgb, fps, window_seconds=WINDOW_SECONDS):
    """
    POS (Plane-Orthogonal-to-Skin) rPPG extraction algorithm.
    rgb: (N, 3) array ordered [R, G, B].
    """
    window = max(8, int(window_seconds * fps))
    count = len(rgb)

    output = np.zeros(count, dtype=float)
    weights = np.zeros(count, dtype=float)

    for start in range(0, count - window + 1):
        end = start + window
        segment = rgb[start:end].copy()

        means = segment.mean(axis=0)
        if np.any(means < 1e-6):
            continue

        normalized = segment / means
        r = normalized[:, 0]
        g = normalized[:, 1]
        b = normalized[:, 2]

        s1 = g - b
        s2 = g + b - 2.0 * r

        std1 = np.std(s1)
        std2 = np.std(s2)
        if std2 < 1e-8:
            continue

        alpha = std1 / std2
        h = s1 + alpha * s2
        h -= h.mean()

        output[start:end] += h
        weights[start:end] += 1.0

    valid = weights > 0
    if not np.any(valid):
        raise ValueError("POS could not produce a valid signal.")

    output[valid] /= weights[valid]
    return output


# ---------------- SIGNAL PROCESSING & METRICS ----------------

def bandpass_signal(signal, fps):
    low_hz = MIN_BPM / 60.0
    high_hz = min(MAX_BPM / 60.0, fps * 0.45)

    if high_hz <= low_hz:
        raise ValueError("Video FPS is too low for the selected pulse band.")

    sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    if len(signal) < 30:
        raise ValueError("Video is too short to filter reliably.")

    return sosfiltfilt(sos, signal)


def estimate_bpm_and_sqi(signal, fps):
    """
    Computes pulse frequency spectrum, estimated BPM, peak prominence fraction,
    and Signal-to-Noise Ratio (SNR in dB) within the cardiac band.
    """
    nperseg = min(len(signal), int(8 * fps))
    frequencies, power = welch(signal, fs=fps, nperseg=nperseg)

    valid_mask = (frequencies >= MIN_BPM / 60.0) & (frequencies <= MAX_BPM / 60.0)
    if not np.any(valid_mask):
        raise ValueError("No frequencies in the selected pulse range.")

    candidate_freqs = frequencies[valid_mask]
    candidate_power = power[valid_mask]

    peak_idx = np.argmax(candidate_power)
    peak_hz = candidate_freqs[peak_idx]
    bpm = peak_hz * 60.0

    # Cardiac peak window: +/- 6 BPM (+/- 0.1 Hz)
    half_window_hz = 0.10
    peak_band_mask = np.abs(candidate_freqs - peak_hz) <= half_window_hz
    signal_power = np.sum(candidate_power[peak_band_mask])
    total_inband_power = np.sum(candidate_power)

    peak_fraction = float(signal_power / total_inband_power) if total_inband_power > 0 else 0.0

    noise_power = total_inband_power - signal_power
    if noise_power > 0 and signal_power > 0:
        snr_db = 10.0 * np.log10(signal_power / noise_power)
    else:
        snr_db = 0.0

    return float(bpm), peak_fraction, snr_db, frequencies, power


def evaluate_quality(duration, fps, face_ratio, max_gap_sec, peak_fraction, snr_db):
    """
    Evaluates signal quality rating and compiles actionable warnings.
    """
    warnings_list = []

    # Duration evaluation
    if duration < ABSOLUTE_MIN_DURATION_SEC:
        duration_verdict = f"TOO SHORT ({duration:.1f}s < {ABSOLUTE_MIN_DURATION_SEC:.0f}s)"
        warnings_list.append(
            f"Recording duration ({duration:.1f}s) is critically short. Frequency resolution "
            f"df is coarse (~{60.0/duration:.1f} BPM). Recommended: 30-45s at 25-30 FPS."
        )
    elif duration < RECOMMENDED_MIN_DURATION_SEC:
        duration_verdict = f"MARGINAL ({duration:.1f}s < {RECOMMENDED_MIN_DURATION_SEC:.0f}s)"
        warnings_list.append(
            f"Recording duration ({duration:.1f}s) is below recommended standard (>= 30.0s). "
            f"Spectral resolution is limited."
        )
    else:
        duration_verdict = f"SUFFICIENT ({duration:.1f}s >= {RECOMMENDED_MIN_DURATION_SEC:.0f}s)"

    # Gap and face tracking evaluation
    if max_gap_sec > 1.0:
        warnings_list.append(
            f"Face tracking lost for {max_gap_sec:.2f}s consecutive. Large missing gaps "
            f"distort pulsatile phase information and degrade rPPG accuracy."
        )

    if face_ratio < 0.85:
        warnings_list.append(
            f"Face patch availability is low ({face_ratio:.1%}). Ensure steady lighting "
            f"and centered face position."
        )

    # Spectral prominence / SNR evaluation
    if peak_fraction < 0.20:
        warnings_list.append(
            f"Low spectral peak prominence ({peak_fraction:.1%}). Likely caused by motion "
            f"artifacts, head movement, or uneven lighting."
        )

    # Overall Quality Tier
    if (
        duration >= RECOMMENDED_MIN_DURATION_SEC
        and face_ratio >= 0.95
        and max_gap_sec <= 0.5
        and peak_fraction >= 0.30
        and snr_db >= 2.0
    ):
        quality_rating = "HIGH QUALITY (Clean signal)"
    elif (
        duration >= ABSOLUTE_MIN_DURATION_SEC
        and face_ratio >= 0.85
        and max_gap_sec <= 1.0
        and peak_fraction >= 0.20
    ):
        quality_rating = "MODERATE QUALITY (Usable estimate)"
    elif duration < ABSOLUTE_MIN_DURATION_SEC or max_gap_sec > 1.5 or face_ratio < 0.80:
        quality_rating = "UNRELIABLE (Validation test only, pulse accuracy unconfirmed)"
    else:
        quality_rating = "MARGINAL (Low confidence, observe warnings)"

    return duration_verdict, quality_rating, warnings_list


def compute_gap_statistics(valid_flags, fps):
    """
    Computes total gaps and maximum consecutive dropped frame sequence in seconds.
    """
    gap_count = 0
    max_gap_frames = 0
    curr_gap_frames = 0

    for is_valid in valid_flags:
        if not is_valid:
            curr_gap_frames += 1
            if curr_gap_frames > max_gap_frames:
                max_gap_frames = curr_gap_frames
        else:
            if curr_gap_frames > 0:
                gap_count += 1
                curr_gap_frames = 0

    if curr_gap_frames > 0:
        gap_count += 1

    max_gap_sec = max_gap_frames / fps if fps > 0 else 0.0
    return gap_count, max_gap_frames, max_gap_sec


def validate_against_reference(estimated_bpm, reference_bpm):
    """
    Computes absolute error and clinical agreement metrics against reference.
    """
    abs_error = abs(estimated_bpm - reference_bpm)
    rel_error = (abs_error / reference_bpm) * 100.0 if reference_bpm > 0 else 0.0

    if abs_error <= 5.0:
        agreement = "PASS (Within +/- 5.0 BPM clinical tolerance)"
    elif abs_error <= 10.0:
        agreement = "ACCEPTABLE (Within +/- 10.0 BPM tolerance)"
    else:
        agreement = "FAIL (Absolute error > 10.0 BPM)"

    return abs_error, rel_error, agreement


def log_validation_result(log_path, record):
    """
    Appends execution metrics to a CSV file for tracking across recordings.
    """
    file_exists = os.path.isfile(log_path)
    fieldnames = [
        "Timestamp",
        "Video_File",
        "Duration_s",
        "Nominal_FPS",
        "Actual_Sampling_Rate",
        "Face_Patch_Pct",
        "Max_Gap_s",
        "Estimated_BPM",
        "Reference_BPM",
        "Abs_Error_BPM",
        "Rel_Error_Pct",
        "Peak_Power_Fraction",
        "SNR_dB",
        "Quality_Rating",
        "Detector_Backend",
    ]

    try:
        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        print(f"Validation metrics appended to: {log_path}")
    except Exception as e:
        print(f"Could not write to validation log {log_path}: {e}")


# ---------------- MAIN PIPELINE ----------------

def process_video(
    video_path,
    reference_bpm=None,
    detector_mode="auto",
    save_plot="rppg_plot.png",
    show_plot=True,
    max_gap_threshold=DEFAULT_MAX_GAP_SEC,
    log_file="validation_log.csv",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps < 10:
        cap.release()
        raise ValueError("Invalid or low video FPS. Use a stable video of at least 10 FPS.")

    rgb_samples = []
    timestamps = []
    valid_flags = []
    frames_seen = 0
    face_frames = 0

    detector = FaceDetector(mode=detector_mode)
    backend_name = (
        f"MediaPipe FaceLandmarker (Landmarks)"
        if detector.backend == "mediapipe_tasks"
        else f"MediaPipe Solutions FaceMesh"
        if detector.backend == "mediapipe_solutions"
        else "OpenCV Haar Cascade (Bounding Box Fallback)"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frames_seen / fps
            frames_seen += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = detector.detect(frame, rgb_frame)

            sample = None
            if detection is not None:
                if detection["type"] == "landmarks":
                    sample = extract_rgb_from_landmarks(frame, detection["landmarks"])
                elif detection["type"] == "bbox":
                    x, y, w, h = detection["bbox"]
                    sample = extract_rgb_from_box(frame, x, y, w, h)

            if sample is not None:
                face_frames += 1
                valid_flags.append(True)
                rgb_samples.append(sample)
            else:
                valid_flags.append(False)
                rgb_samples.append([np.nan, np.nan, np.nan])

            timestamps.append(timestamp)
    finally:
        cap.release()
        detector.close()

    if frames_seen == 0:
        raise ValueError("The video contains no readable frames.")

    rgb = np.asarray(rgb_samples, dtype=float)
    timestamps = np.asarray(timestamps)

    duration = frames_seen / fps
    face_ratio = face_frames / frames_seen
    print(f"Duration: {duration:.2f} seconds")
    print(f"Nominal FPS: {fps:.2f}")
    print(f"Frames with usable face patches: {face_ratio:.1%}")

    if duration < 20:
        raise ValueError(
            "Recording too short for this prototype. "
            "Record at least 20 seconds; 30-45 seconds is recommended."
        )

    if fps < 20:
        print(
            "WARNING: FPS is below 20. "
            "Pulse estimation may be less reliable."
        )

    if face_ratio < 0.90:
        raise ValueError(
            "Face detection failed in too many frames. "
            "Use a frontal face video with steady lighting."
        )

    # Actual observed sampling rate
    actual_sampling_rate = (frames_seen - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else fps

    # Identify missing face samples
    missing = ~np.isfinite(rgb).all(axis=1)

    # Find consecutive missing-frame runs
    changes = np.diff(
        np.concatenate(([False], missing, [False])).astype(int)
    )
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    max_gap_seconds = 0.5
    max_gap_frames = max(1, int(max_gap_seconds * fps))

    gap_lengths = ends - starts
    gap_count = len(starts)
    max_gap_len = int(gap_lengths.max()) if len(gap_lengths) > 0 else 0
    max_gap_sec = max_gap_len / fps

    for start, end in zip(starts, ends):
        gap_length = end - start
        # Do not invent a signal across a long detection failure.
        if gap_length > max_gap_frames:
            raise ValueError(
                f"Face detection gap of {gap_length / fps:.2f} seconds. "
                "Use a steadier recording and try again."
            )

    # Interpolate only short missing gaps
    for channel in range(3):
        values = rgb[:, channel]
        valid = np.isfinite(values)
        if valid.sum() < int(0.90 * len(values)):
            raise ValueError("Too many missing RGB samples.")
        rgb[:, channel] = np.interp(
            timestamps,
            timestamps[valid],
            values[valid],
        )

    raw_pos = pos_signal(rgb, fps)
    filtered = bandpass_signal(raw_pos, fps)

    bpm, peak_fraction, snr_db, frequencies, power = estimate_bpm_and_sqi(filtered, fps)

    # Quality and warning assessment
    duration_verdict, quality_rating, quality_warnings = evaluate_quality(
        duration, fps, face_ratio, max_gap_sec, peak_fraction, snr_db
    )

    # Reference validation
    has_ref = reference_bpm is not None and reference_bpm > 0
    if has_ref:
        abs_error, rel_error, agreement = validate_against_reference(bpm, reference_bpm)
    else:
        abs_error, rel_error, agreement = None, None, None

    # Print comprehensive diagnostic report
    print("\n" + "=" * 70)
    print("               rPPG Pulse Estimation & Validation Report")
    print("=" * 70)
    print(f"Video File:              {os.path.basename(video_path)}")
    print(f"Detector Backend:        {backend_name}")

    print("\n--- 1. Recording & Sampling Characteristics ---")
    print(f"Duration:                {duration:.2f} s")
    print(f"Total Frames Processed:  {frames_seen} frames")
    print(f"Nominal Video FPS:       {fps:.2f} Hz")
    print(f"Actual Sampling Rate:    {actual_sampling_rate:.2f} Hz")
    print(f"Duration Status:         {duration_verdict}")

    print("\n--- 2. Face Tracking & ROI Continuity ---")
    print(f"Usable Face Frames:      {face_frames} / {frames_seen} ({face_ratio:.1%})")
    print(f"Tracking Gaps Count:     {gap_count} incident(s)")
    print(f"Max Consecutive Gap:     {max_gap_sec:.2f} s ({max_gap_frames} frames)")
    if max_gap_sec > max_gap_threshold:
        print(f"Gap Warning:             [EXCEEDS {max_gap_threshold:.1f}s THRESHOLD] May distort pulse phase!")
    else:
        print(f"Gap Status:              [PASS] Continuity within {max_gap_threshold:.1f}s threshold")

    print("\n--- 3. Pulse Estimation & Signal Quality Index (SQI) ---")
    print(f"Estimated Pulse:         {bpm:.1f} BPM ({bpm / 60.0:.2f} Hz)")
    print(f"Cardiac Band:            {MIN_BPM:.0f} - {MAX_BPM:.0f} BPM ({MIN_BPM/60:.2f} - {MAX_BPM/60:.2f} Hz)")
    print(f"Peak Power Fraction:     {peak_fraction:.3f} ({peak_fraction * 100:.1f}% in-band energy)")
    print(f"Signal-to-Noise (SNR):   {snr_db:.1f} dB")
    print(f"Signal Quality Rating:   [{quality_rating}]")

    if quality_warnings:
        print("\nSignal Quality Warnings:")
        for w in quality_warnings:
            print(f"  * {w}")

    print("\n--- 4. Reference Validation (Ground Truth) ---")
    if has_ref:
        print(f"Reference BPM:           {reference_bpm:.1f} BPM (from pulse oximeter)")
        print(f"Estimated BPM:           {bpm:.1f} BPM")
        print(f"Absolute Error (AE):     {abs_error:.1f} BPM")
        print(f"Relative Error (MAPE):   {rel_error:.1f} %")
        print(f"Clinical Agreement:      [{agreement}]")
    else:
        print("Reference BPM:           Not provided")
        print("                         (Use --reference-bpm <value> to benchmark against pulse oximeter)")

    print("=" * 70 + "\n")

    # Plot 3-panel visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: RGB intensity over time
    axes[0].plot(timestamps, rgb[:, 0], label="Red Channel", color="#d32f2f", alpha=0.9)
    axes[0].plot(timestamps, rgb[:, 1], label="Green Channel", color="#2e7d32", alpha=0.9)
    axes[0].plot(timestamps, rgb[:, 2], label="Blue Channel", color="#1565c0", alpha=0.9)
    axes[0].set_title(f"Mean ROI Color Intensity Over Time ({backend_name})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Filtered POS rPPG waveform
    axes[1].plot(timestamps, filtered, color="#00838f", linewidth=1.5, label="Filtered POS Signal")
    axes[1].set_title("Filtered POS rPPG Pulsatile Waveform (0.75 - 3.0 Hz)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Pulse frequency spectrum
    in_band_bpm = frequencies * 60.0
    axes[2].plot(in_band_bpm, power, color="#6a1b9a", linewidth=1.8, label="Power Spectrum (Welch PSD)")
    axes[2].axvline(bpm, linestyle="--", color="#c62828", linewidth=2.0, label=f"Estimated: {bpm:.1f} BPM")

    if has_ref:
        axes[2].axvline(
            reference_bpm,
            linestyle="-.",
            color="#2e7d32",
            linewidth=2.0,
            label=f"Reference: {reference_bpm:.1f} BPM (Error: {abs_error:.1f})",
        )

    axes[2].set_xlim(MIN_BPM, MAX_BPM)
    axes[2].set_title(
        f"Pulse Frequency Spectrum | SQI: {quality_rating.split(' ')[0]} (SNR: {snr_db:.1f} dB)",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].set_xlabel("Frequency (BPM)")
    axes[2].set_ylabel("Power")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot, dpi=150)
        print(f"Spectrum plot saved to: {save_plot}")

    # Log to CSV
    if log_file:
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Video_File": os.path.basename(video_path),
            "Duration_s": f"{duration:.2f}",
            "Nominal_FPS": f"{fps:.2f}",
            "Actual_Sampling_Rate": f"{actual_sampling_rate:.2f}",
            "Face_Patch_Pct": f"{face_ratio * 100:.1f}",
            "Max_Gap_s": f"{max_gap_sec:.2f}",
            "Estimated_BPM": f"{bpm:.1f}",
            "Reference_BPM": f"{reference_bpm:.1f}" if has_ref else "N/A",
            "Abs_Error_BPM": f"{abs_error:.1f}" if has_ref else "N/A",
            "Rel_Error_Pct": f"{rel_error:.1f}" if has_ref else "N/A",
            "Peak_Power_Fraction": f"{peak_fraction:.3f}",
            "SNR_dB": f"{snr_db:.1f}",
            "Quality_Rating": quality_rating,
            "Detector_Backend": detector.backend,
        }
        log_validation_result(log_file, record)

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not open GUI plot window: {e}")
    else:
        plt.close(fig)

    return {
        "bpm": bpm,
        "reference_bpm": reference_bpm,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "agreement": agreement,
        "duration": duration,
        "fps": fps,
        "actual_sampling_rate": actual_sampling_rate,
        "face_ratio": face_ratio,
        "max_gap_sec": max_gap_sec,
        "peak_fraction": peak_fraction,
        "snr_db": snr_db,
        "quality_rating": quality_rating,
        "warnings": quality_warnings,
    }


# ---------------- CLI ENTRYPOINT ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and validate an rPPG pulse estimate from a face video."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to the input face video (default: videos/my_face.mp4 if present)",
    )
    parser.add_argument(
        "--reference-bpm",
        "--ref-bpm",
        type=float,
        default=None,
        help="Ground-truth reference pulse rate (from a pulse oximeter or monitor) for validation error calculation",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "mediapipe", "opencv"],
        default="auto",
        help="Face detector backend: 'auto' (preferred MediaPipe landmarks), 'mediapipe', or 'opencv' (default: auto)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="rppg_plot.png",
        help="Path to save output spectrum plot image (default: rppg_plot.png, set to '' to disable)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive GUI plot display (useful for scripts, benchmarks, or headless testing)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_GAP_SEC,
        help=f"Maximum acceptable consecutive tracking gap in seconds before warning (default: {DEFAULT_MAX_GAP_SEC}s)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="validation_log.csv",
        help="CSV log file path to record validation metrics across multiple recordings (default: validation_log.csv)",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=35,
        type=int,
        help="Record a new video using webcam (optional duration in seconds, default 35s)",
    )
    args = parser.parse_args()

    # Handle webcam recording request
    if args.record is not None:
        try:
            import record_video
            recorded_path = record_video.record_webcam(duration_sec=args.record)
            if recorded_path and os.path.exists(recorded_path):
                args.video = recorded_path
            else:
                sys.exit(0)
        except ImportError:
            print("record_video.py module not found. Run recording separately.")
            sys.exit(1)

    video_path = args.video
    if not video_path:
        default_candidate = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "videos", "my_face.mp4"
        )
        if os.path.exists(default_candidate):
            video_path = default_candidate
            print(f"No video argument supplied. Using sample video: {video_path}")
        else:
            parser.error("No input video specified and default video 'videos/my_face.mp4' not found.")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    try:
        save_path = args.save_plot if args.save_plot else None
        log_path = args.log_file if args.log_file else None
        process_video(
            video_path,
            reference_bpm=args.reference_bpm,
            detector_mode=args.detector,
            save_plot=save_path,
            show_plot=not args.no_show,
            max_gap_threshold=args.max_gap,
            log_file=log_path,
        )
    except (ValueError, RuntimeError) as error:
        print(f"\nProcessing failed: {error}")
        sys.exit(1)
