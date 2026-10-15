"""
Webcam Video Recorder for rPPG Pulse Validation.
Records high-quality 30-45s facial video at 25-30 FPS with an on-screen
alignment guide and prompts for ground-truth reference pulse oximeter BPM.
"""

import argparse
from datetime import datetime
import os
import sys
import time

import cv2
import numpy as np


def record_webcam(
    duration_sec=35,
    output_dir="videos",
    camera_idx=0,
    target_fps=30.0,
    width=1280,
    height=720,
):
    """
    Records a webcam video with an on-screen face alignment guide.
    Returns the saved video path or None if cancelled.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"recording_{timestamp_str}.mp4")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}. Ensure webcam is connected and accessible.")
        return None

    # Request high resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or not np.isfinite(actual_fps):
        actual_fps = target_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, actual_fps, (actual_w, actual_h))

    if not out.isOpened():
        print(f"Error: Could not initialize VideoWriter for {output_filename}")
        cap.release()
        return None

    print("\n" + "=" * 65)
    print("               Webcam rPPG Video Recorder")
    print("=" * 65)
    print(f"Target Duration:    {duration_sec} seconds (recommended for pulse validation)")
    print(f"Resolution:         {actual_w}x{actual_h}")
    print(f"Camera FPS:         {actual_fps:.1f} FPS")
    print(f"Saving to:          {output_filename}")
    print("\nGuidelines for best accuracy:")
    print("  * Position your face inside the green oval on screen.")
    print("  * Maintain steady, diffuse lighting (avoid backlighting).")
    print("  * Keep your head still and breathe normally (do not speak).")
    print("  * Press 'q' or 'ESC' to cancel. Press 'SPACE' to stop early.")
    print("=" * 65 + "\n")

    start_time = time.time()
    frames_recorded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from camera.")
                break

            elapsed = time.time() - start_time
            remaining = max(0.0, duration_sec - elapsed)
            frames_recorded += 1

            # Save clean frame to video
            out.write(frame)

            # Draw visual guidance overlay for the display window
            display_frame = frame.copy()
            center_x, center_y = actual_w // 2, actual_h // 2
            axes_x, axes_y = int(actual_w * 0.18), int(actual_h * 0.32)

            # Oval face guide
            cv2.ellipse(
                display_frame,
                (center_x, center_y),
                (axes_x, axes_y),
                0,
                0,
                360,
                (0, 230, 115),
                2,
                cv2.LINE_AA,
            )

            # Top info banner
            cv2.rectangle(display_frame, (0, 0), (actual_w, 75), (20, 20, 20), -1)
            progress = min(1.0, elapsed / duration_sec)

            # Progress bar
            bar_w = int(actual_w * progress)
            cv2.rectangle(display_frame, (0, 70), (bar_w, 75), (0, 220, 100), -1)

            timer_text = f"Recording: {elapsed:.1f}s / {duration_sec:.1f}s  (Remaining: {remaining:.1f}s)"
            cv2.putText(
                display_frame,
                timer_text,
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            fps_text = f"{frames_recorded / max(0.001, elapsed):.1f} FPS"
            cv2.putText(
                display_frame,
                fps_text,
                (actual_w - 140, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 230, 255),
                2,
                cv2.LINE_AA,
            )

            # Bottom instruction text
            inst_text = "Center face in oval | Keep head still | Breathe normally"
            cv2.putText(
                display_frame,
                inst_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("rPPG Video Recording - Press 'q' to Cancel", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("\nRecording cancelled by user.")
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                if os.path.exists(output_filename):
                    os.remove(output_filename)
                return None
            elif key == ord(" "):
                print("\nEarly completion requested by user.")
                break

            if elapsed >= duration_sec:
                break
    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    effective_fps = frames_recorded / total_time if total_time > 0 else actual_fps
    print(f"\nRecording finished!")
    print(f"Total Time:      {total_time:.1f}s")
    print(f"Frames Captured: {frames_recorded}")
    print(f"Effective FPS:   {effective_fps:.1f}")
    print(f"Saved Video:     {output_filename}\n")

    return output_filename


def main():
    parser = argparse.ArgumentParser(
        description="Record a 30-45s facial video from webcam for rPPG pulse validation."
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=35,
        help="Recording duration in seconds (default: 35s, recommended 30-45s)",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="videos",
        help="Output folder to store recorded videos (default: videos)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Automatically run rppg.py on the recorded video after capture",
    )
    args = parser.parse_args()

    video_path = record_webcam(
        duration_sec=args.duration,
        output_dir=args.output_dir,
        camera_idx=args.camera,
    )

    if not video_path:
        sys.exit(0)

    # Prompt for reference pulse oximeter BPM
    try:
        ref_input = input("Enter pulse oximeter reference BPM (press Enter to skip): ").strip()
        ref_bpm = float(ref_input) if ref_input else None
    except ValueError:
        print("Invalid number entered. Skipping reference BPM.")
        ref_bpm = None

    print(f"\nRunning rPPG analysis on {video_path}...")
    import rppg
    rppg.process_video(video_path, reference_bpm=ref_bpm, show_plot=True)


if __name__ == "__main__":
    main()
