"""
collect_data.py — Interactive webcam-based data collection tool.

Controls:
  1–9, 0  → select class (maps to GESTURE_LABELS indices)
  r       → toggle recording ON/OFF
  s       → save dataset to disk immediately
  q       → quit (auto-saves)

Data is saved to:
  data/landmarks_YYYYMMDD_HHMMSS.csv
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add project root to path so `utils` is importable from anywhere
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    GESTURE_LABELS,
    LABEL_TO_KEY,
    draw_overlay,
    landmarks_to_feature_vector,
    FPSCounter,
)

# ── Config ────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0          # Change if your webcam is not index 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.6

# ── MediaPipe setup ───────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


def build_filename() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DATA_DIR / f"landmarks_{ts}.csv"


def save_dataset(rows: list[dict], filepath: Path) -> None:
    if not rows:
        print("[INFO] No data to save.")
        return
    fieldnames = ["label"] + [f"f{i}" for i in range(42)]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {len(rows)} samples → {filepath}")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check macOS Camera permissions.")
        sys.exit(1)

    rows: list[dict] = []
    current_label: str = GESTURE_LABELS[0]
    recording: bool   = False
    fps_counter = FPSCounter()
    save_path   = build_filename()

    # Per-class sample counts (for HUD feedback)
    counts: dict[str, int] = {lbl: 0 for lbl in GESTURE_LABELS}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        print("=" * 52)
        print("  GESTURE DATA COLLECTOR")
        print("=" * 52)
        print("  Keys 1-9,0 → select gesture class")
        print("  r          → toggle recording")
        print("  s          → save now")
        print("  q          → quit (auto-saves)")
        print("=" * 52)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            fps   = fps_counter.tick()

            # ── MediaPipe ────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            feat_vec = None
            hand_detected = False

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                hand_detected = True

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                feat_vec = landmarks_to_feature_vector(hand_lms)

                # Record if toggled on and we have a valid vector
                if recording and feat_vec is not None:
                    row = {"label": current_label}
                    row.update({f"f{i}": round(float(feat_vec[i]), 6) for i in range(42)})
                    rows.append(row)
                    counts[current_label] += 1

            # ── HUD ──────────────────────────────
            # Main overlay (reuse draw_overlay for consistency)
            conf_display = 1.0 if (recording and hand_detected) else 0.0
            draw_overlay(frame, current_label, conf_display, fps, recording)

            # Class selection panel (left side)
            panel_x = 12
            cv2.putText(
                frame, "CLASSES (key→label : count)",
                (panel_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1,
            )
            for idx, lbl in enumerate(GESTURE_LABELS):
                key_char = str((idx + 1) % 10)  # 1–9 then 0
                color = (0, 255, 150) if lbl == current_label else (160, 160, 160)
                marker = "►" if lbl == current_label else " "
                text = f"{marker} [{key_char}] {lbl:<16} {counts[lbl]:>4}"
                cv2.putText(
                    frame, text,
                    (panel_x, 120 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA,
                )

            # Status line
            status = (
                f"{'● REC' if recording else '○ IDLE'}  |  "
                f"Selected: {current_label}  |  "
                f"Total samples: {len(rows)}"
            )
            cv2.putText(
                frame, status,
                (12, frame.shape[0] - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 200, 80) if recording else (180, 180, 180), 1, cv2.LINE_AA,
            )

            cv2.imshow("Gesture Data Collector", frame)

            # ── Keyboard ─────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                state = "ON" if recording else "OFF"
                print(f"[REC] Recording {state} — class: {current_label}")
            elif key == ord("s"):
                save_dataset(rows, save_path)
            elif chr(key) in LABEL_TO_KEY:
                new_label = LABEL_TO_KEY[chr(key)]
                if new_label != current_label:
                    was_recording = recording
                    recording = False  # Pause to avoid accidental samples
                    current_label = new_label
                    print(f"[CLASS] Switched to: {current_label}")
                    if was_recording:
                        time.sleep(0.3)  # Brief pause for hand reposition
                        recording = True

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Auto-save on quit
    save_dataset(rows, save_path)
    print("\n[DONE] Collection complete.")
    print(f"Per-class totals: {counts}")


if __name__ == "__main__":
    main()
