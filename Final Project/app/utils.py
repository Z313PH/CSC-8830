"""
utils.py — Shared utilities for gesture recognition pipeline.
Covers: landmark normalization, feature extraction, temporal smoothing.
"""

import numpy as np
from collections import deque


# ──────────────────────────────────────────────
# 1. LANDMARK NORMALIZATION & FEATURE EXTRACTION
# ──────────────────────────────────────────────

WRIST_ID = 0
MIDDLE_MCP_ID = 9  # Used as scale reference


def landmarks_to_feature_vector(hand_landmarks) -> np.ndarray | None:
    """
    Convert a MediaPipe Hands `hand_landmarks` object into a
    normalized, translation- and scale-invariant feature vector.

    Steps:
      1. Extract raw (x, y, z) for all 21 landmarks.
      2. Center by subtracting the wrist (id=0).
      3. Scale by the wrist→middle-MCP distance.
      4. Return flattened [x1,y1, x2,y2, ..., x21,y21] (z dropped for stability).

    Returns None if landmarks are invalid.
    """
    if hand_landmarks is None:
        return None

    # Raw coords as (21, 3) array
    raw = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )  # shape (21, 3)

    # --- Center at wrist ---
    wrist = raw[WRIST_ID].copy()
    centered = raw - wrist  # (21, 3)

    # --- Scale by wrist→middle_mcp distance ---
    scale = np.linalg.norm(centered[MIDDLE_MCP_ID])
    if scale < 1e-6:
        # Degenerate hand (all landmarks collapsed); skip frame
        return None
    normalized = centered / scale  # (21, 3)

    # --- Use only x, y (drop z — less stable from webcam depth) ---
    xy = normalized[:, :2]  # (21, 2)

    return xy.flatten()  # shape (42,)


def feature_vector_size() -> int:
    """Returns the expected length of the feature vector (42 = 21 landmarks × 2)."""
    return 42


# ──────────────────────────────────────────────
# 2. TEMPORAL SMOOTHING
# ──────────────────────────────────────────────

class TemporalSmoother:
    """
    Rolling-window majority vote with confidence weighting.

    Parameters
    ----------
    window_size : int
        Number of recent frames to consider (default 15).
    confidence_threshold : float
        Minimum smoothed confidence to accept a label (default 0.60).
    min_stable_frames : int
        Minimum frames the winning class must appear in the window (default 6).
    """

    def __init__(
        self,
        window_size: int = 15,
        confidence_threshold: float = 0.60,
        min_stable_frames: int = 6,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_stable_frames = min_stable_frames

        # Each entry: (label, confidence)
        self._window: deque[tuple[str, float]] = deque(maxlen=window_size)

        # Public output
        self.current_label: str = "Unknown"
        self.current_confidence: float = 0.0

    def update(self, label: str, confidence: float) -> tuple[str, float]:
        """
        Feed a new (label, confidence) prediction.
        Returns the smoothed (label, confidence).
        """
        self._window.append((label, confidence))

        if len(self._window) < self.min_stable_frames:
            # Not enough history yet
            self.current_label = "Unknown"
            self.current_confidence = 0.0
            return self.current_label, self.current_confidence

        # --- Confidence-weighted vote per class ---
        class_scores: dict[str, float] = {}
        class_counts: dict[str, int] = {}
        for lbl, conf in self._window:
            class_scores[lbl] = class_scores.get(lbl, 0.0) + conf
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        best_label = max(class_scores, key=class_scores.__getitem__)
        best_count = class_counts[best_label]
        best_avg_conf = class_scores[best_label] / best_count

        # --- Acceptance criteria ---
        if best_count < self.min_stable_frames:
            self.current_label = "Unknown"
            self.current_confidence = 0.0
        elif best_avg_conf < self.confidence_threshold:
            self.current_label = "Unknown"
            self.current_confidence = best_avg_conf
        else:
            self.current_label = best_label
            self.current_confidence = best_avg_conf

        return self.current_label, self.current_confidence

    def reset(self):
        """Clear history (e.g. when switching tasks)."""
        self._window.clear()
        self.current_label = "Unknown"
        self.current_confidence = 0.0


# ──────────────────────────────────────────────
# 3. FPS TRACKER
# ──────────────────────────────────────────────

import time


class FPSCounter:
    """Lightweight exponential-moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps: float = 0.0
        self._last_time: float = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps = (
                inst_fps if self._fps == 0.0
                else self._alpha * inst_fps + (1 - self._alpha) * self._fps
            )
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ──────────────────────────────────────────────
# 4. OVERLAY DRAWING
# ──────────────────────────────────────────────

import cv2


def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    fps: float,
    recording: bool = False,
) -> np.ndarray:
    """
    Draw a HUD overlay on the frame (in-place).

    Parameters
    ----------
    frame      : BGR OpenCV frame
    label      : predicted gesture label (or "Unknown")
    confidence : float in [0, 1]
    fps        : current FPS
    recording  : show red REC indicator if True
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black banner at top
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Gesture label
    label_color = (0, 255, 100) if label != "Unknown" else (80, 80, 80)
    cv2.putText(
        frame, f"Gesture: {label}",
        (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, label_color, 2, cv2.LINE_AA,
    )

    # Confidence bar
    conf_pct = int(confidence * 100)
    bar_w = int((w - 24) * confidence)
    cv2.rectangle(frame, (12, 40), (w - 12, 58), (50, 50, 50), -1)
    bar_color = (0, 200, 80) if confidence >= 0.6 else (0, 140, 220)
    cv2.rectangle(frame, (12, 40), (12 + bar_w, 58), bar_color, -1)
    cv2.putText(
        frame, f"{conf_pct}%",
        (w - 52, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )

    # FPS (bottom-right)
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # REC indicator
    if recording:
        cv2.circle(frame, (w - 22, 22), 9, (0, 0, 220), -1)
        cv2.putText(
            frame, "REC",
            (w - 70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
        )

    return frame


# ──────────────────────────────────────────────
# 5. GESTURE SET REFERENCE
# ──────────────────────────────────────────────

GESTURE_LABELS = [
    "open_palm",      # All 5 fingers extended, spread wide
    "fist",           # All fingers curled into palm
    "peace",          # Index + middle extended (V shape)
    "thumbs_up",      # Thumb up, other fingers curled
    "ok_sign",        # Thumb + index form circle, others extended
    "point_up",       # Only index finger extended upward
    "three_fingers",  # Index + middle + ring extended
    "four_fingers",   # All except thumb extended
    "call_me",        # Thumb + pinky extended (shaka / "call me")
    "rock",           # Index + pinky extended (devil horns)
]

LABEL_TO_KEY = {str(i + 1): label for i, label in enumerate(GESTURE_LABELS)}