"""
run_live_demo.py — Real-time gesture recognition demo.

Usage:
    python run_live_demo.py
    python run_live_demo.py --model models/model.joblib --smooth on|off

Controls:
    t → toggle temporal smoothing ON/OFF (ablation)
    q → quit

Art Engine Integration:
    After each frame the following module-level variables are updated:
        current_label      : str   — stable predicted gesture (or "Unknown")
        current_confidence : float — average confidence in [0, 1]
    Import this module and poll these variables from your Art Engine.
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FPSCounter,
    TemporalSmoother,
    draw_overlay,
    landmarks_to_feature_vector,
)
from apply_effect import EFFECT_MAP, EFFECT_NAMES

# ── Public Art-Engine interface ──────────────
current_label:      str   = "Unknown"
current_confidence: float = 0.0

# ── Config ───────────────────────────────────
MODEL_PATH  = Path("models/model.joblib")
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

MIN_DETECTION_CONFIDENCE = 0.70
MIN_TRACKING_CONFIDENCE  = 0.60
CONFIDENCE_THRESHOLD     = 0.60   # Below this → "Unknown"
SMOOTH_WINDOW            = 15
MIN_STABLE_FRAMES        = 6

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def load_model(model_path: Path):
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("        Run `python train_model.py` first.")
        sys.exit(1)
    model = joblib.load(model_path)
    print(f"[MODEL] Loaded: {model_path}")
    if hasattr(model, "classes_"):
        print(f"[MODEL] Classes: {list(model.classes_)}")
    return model


THUMB_W   = 320   # width of the effect preview thumbnail
THUMB_H   = 180   # height of the effect preview thumbnail
THUMB_PAD = 10    # gap from frame edge

INPUT_IMAGE_PATH = Path(__file__).parent / "input" / "example.jpg"


def load_input_image() -> np.ndarray | None:
    """Load the static input image once at startup."""
    img = cv2.imread(str(INPUT_IMAGE_PATH))
    if img is None:
        print(f"[WARN] Could not load input image: {INPUT_IMAGE_PATH}")
    else:
        print(f"[INPUT] Loaded {INPUT_IMAGE_PATH.name} ({img.shape[1]}×{img.shape[0]})")
    return img


def draw_effect_thumbnail(frame: np.ndarray, gesture: str, source_img: np.ndarray) -> None:
    """
    Apply the gesture's color effect to *source_img* and composite a resized
    thumbnail into the bottom-right corner of *frame* (in-place).
    Skips silently if gesture is 'Unknown' or source_img is None.
    """
    if gesture == "Unknown" or gesture not in EFFECT_MAP or source_img is None:
        return

    effect_fn = EFFECT_MAP[gesture]
    effected   = effect_fn(source_img.copy())

    h, w = frame.shape[:2]
    thumb = cv2.resize(effected, (THUMB_W, THUMB_H), interpolation=cv2.INTER_LINEAR)

    # Position: bottom-right, above the status-bar text
    x1 = w - THUMB_W - THUMB_PAD
    y1 = h - THUMB_H - THUMB_PAD - 45   # 45 px clearance for bottom text
    x2, y2 = x1 + THUMB_W, y1 + THUMB_H

    # Semi-transparent dark border behind the thumb
    border = 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - border, y1 - border),
                  (x2 + border, y2 + border), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Paste thumbnail
    frame[y1:y2, x1:x2] = thumb

    # Label strip below thumbnail
    label_y = y2 + 14
    cv2.putText(
        frame, EFFECT_NAMES.get(gesture, gesture),
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )


def predict_raw(model, feat_vec: np.ndarray) -> tuple[str, float]:
    """
    Run the sklearn pipeline on a single feature vector.
    Returns (label, confidence).
    """
    x = feat_vec.reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx   = int(np.argmax(proba))
    label = model.classes_[idx]
    confidence = float(proba[idx])
    return label, confidence


def main(model_path: Path = MODEL_PATH, smoothing_on: bool = True):
    global current_label, current_confidence

    model      = load_model(model_path)
    input_img  = load_input_image()
    smoother = TemporalSmoother(
        window_size=SMOOTH_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        min_stable_frames=MIN_STABLE_FRAMES,
    )
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("        On macOS: System Settings → Privacy → Camera → allow Terminal/IDE.")
        sys.exit(1)

    print("[LIVE] Starting real-time gesture recognition.")
    print("       Press 't' to toggle smoothing | 'q' to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                break

            frame = cv2.flip(frame, 1)
            fps   = fps_counter.tick()

            # ── MediaPipe ────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            raw_label      = "Unknown"
            raw_confidence = 0.0

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                feat_vec = landmarks_to_feature_vector(hand_lms)
                if feat_vec is not None:
                    raw_label, raw_confidence = predict_raw(model, feat_vec)

            # ── Smoothing ────────────────────────
            if smoothing_on:
                smooth_label, smooth_conf = smoother.update(raw_label, raw_confidence)
            else:
                # No smoothing — apply only confidence threshold
                smoother.reset()
                if raw_confidence >= CONFIDENCE_THRESHOLD:
                    smooth_label = raw_label
                    smooth_conf  = raw_confidence
                else:
                    smooth_label = "Unknown"
                    smooth_conf  = raw_confidence

            # ── Update public interface ───────────
            current_label      = smooth_label
            current_confidence = smooth_conf

            # ── Draw overlay ─────────────────────
            draw_overlay(frame, smooth_label, smooth_conf, fps)

            # ── Effect thumbnail (bottom-right corner) ───
            draw_effect_thumbnail(frame, smooth_label, input_img)

            # Smoothing status indicator
            mode_text = "SMOOTH: ON  [t]" if smoothing_on else "SMOOTH: OFF [t]"
            cv2.putText(
                frame, mode_text,
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 150) if smoothing_on else (80, 80, 200),
                1, cv2.LINE_AA,
            )

            # Raw prediction (small, for debugging)
            cv2.putText(
                frame,
                f"raw: {raw_label} ({raw_confidence:.2f})",
                (12, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA,
            )

            cv2.imshow("Gesture Recognition — Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                smoothing_on = not smoothing_on
                smoother.reset()
                print(f"[SMOOTH] Temporal smoothing: {'ON' if smoothing_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live gesture recognition demo.")
    parser.add_argument("--model",  type=str, default=str(MODEL_PATH))
    parser.add_argument(
        "--smooth", type=str, default="on", choices=["on", "off"],
        help="Start with smoothing on or off (default: on)",
    )
    args = parser.parse_args()

    main(
        model_path=Path(args.model),
        smoothing_on=(args.smooth == "on"),
    )
