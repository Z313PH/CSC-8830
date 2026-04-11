"""
apply_effect.py — Apply a gesture-driven color effect to an input image.

Pipeline:
  1. (Optional) Detect hand in the image with MediaPipe Hands + trained SVM.
  2. Apply a unique color effect for each of the 10 gesture classes.
  3. Save the result to app/output/.

Usage:
    python apply_effect.py                              # all 10 effects on every image in input/
    python apply_effect.py input/example.jpg            # all 10 effects on one file
    python apply_effect.py -g thumbs_up input/example.jpg  # single effect
    python apply_effect.py --detect input/example.jpg   # auto-detect gesture (requires mediapipe)

Note: MediaPipe does not yet support Python 3.14+.
      Auto-detection (--detect) requires a Python ≤3.12 environment.
      Without --detect, all 10 effect variants are saved.
"""

import sys
import argparse
import pathlib

import cv2
import numpy as np

# MediaPipe is optional (not compatible with Python 3.14)
try:
    import joblib
    import mediapipe as mp
    from utils import landmarks_to_feature_vector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

GESTURE_LABELS = [
    "open_palm",
    "fist",
    "peace",
    "thumbs_up",
    "ok_sign",
    "point_up",
    "three_fingers",
    "four_fingers",
    "call_me",
    "rock",
]

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "model.joblib"
INPUT_DIR  = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Color Effect Implementations
# ─────────────────────────────────────────────────────────────────────────────

def effect_thermal(img: np.ndarray) -> np.ndarray:
    """open_palm → Thermal heat-map (COLORMAP_INFERNO)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalise histogram so the full colour range is used
    eq   = cv2.equalizeHist(gray)
    return cv2.applyColorMap(eq, cv2.COLORMAP_INFERNO)


def effect_grayscale(img: np.ndarray) -> np.ndarray:
    """fist → Deep grayscale with boosted contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def effect_sepia(img: np.ndarray) -> np.ndarray:
    """peace → Vintage sepia tone."""
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    sepia = cv2.transform(img.astype(np.float32), kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def effect_golden(img: np.ndarray) -> np.ndarray:
    """thumbs_up → Warm golden / autumn tones (COLORMAP_AUTUMN)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    autumn = cv2.applyColorMap(eq, cv2.COLORMAP_AUTUMN)
    # Blend lightly with original for a warm tint feel
    return cv2.addWeighted(autumn, 0.75, img, 0.25, 0)


def effect_cool_blue(img: np.ndarray) -> np.ndarray:
    """ok_sign → Cool arctic blue (COLORMAP_WINTER)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    winter = cv2.applyColorMap(eq, cv2.COLORMAP_WINTER)
    return cv2.addWeighted(winter, 0.80, img, 0.20, 0)


def effect_neon_glow(img: np.ndarray) -> np.ndarray:
    """point_up → Neon edge glow on dark background."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Thicken + glow the edges
    kernel = np.ones((3, 3), np.uint8)
    thick  = cv2.dilate(edges, kernel, iterations=2)
    glow   = cv2.GaussianBlur(thick, (9, 9), 0)

    # Colour the glow: green channel dominant for classic neon look
    result = np.zeros_like(img)
    result[:, :, 1] = glow          # green neon
    result[:, :, 0] = glow // 3     # slight blue tint

    # Add a faint dark version of the original for context
    dark = (img.astype(np.float32) * 0.15).astype(np.uint8)
    return cv2.add(result, dark)


def effect_posterize(img: np.ndarray) -> np.ndarray:
    """three_fingers → Vivid posterize (4 colour levels per channel)."""
    levels = 4
    shift  = 8 - int(np.log2(levels))          # bits to truncate
    poster = (img >> shift) << shift
    # Boost saturation for vivid pop
    hsv = cv2.cvtColor(poster, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def effect_pencil_sketch(img: np.ndarray) -> np.ndarray:
    """four_fingers → Pencil / ink sketch."""
    gray_sketch, _ = cv2.pencilSketch(
        img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )
    return cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)


def effect_vaporwave(img: np.ndarray) -> np.ndarray:
    """call_me → Vaporwave duotone (pink + cyan)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Dark tones → deep purple/pink; bright tones → cyan
    shadow_color    = np.array([0.80, 0.20, 0.80], dtype=np.float32)   # BGR: magenta
    highlight_color = np.array([0.95, 0.90, 0.10], dtype=np.float32)   # BGR: cyan-ish

    h, w = gray.shape
    result = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        result[:, :, c] = (
            gray * highlight_color[c] + (1.0 - gray) * shadow_color[c]
        )

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Add a subtle scanline overlay for the retro vibe
    scanline = np.ones_like(result)
    scanline[::2, :] = 0          # every other row slightly darkened
    scanline_mask = cv2.merge([scanline[:, :, 0]] * 3)
    result = cv2.addWeighted(result, 0.92, scanline_mask, 0.08, 0)
    return result


def effect_high_contrast_bw(img: np.ndarray) -> np.ndarray:
    """rock → High-contrast B&W with red-tinted edge overlay."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aggressive CLAHE for brutal contrast
    clahe  = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    sharp  = clahe.apply(gray)

    # Threshold to hard B&W
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # Red edge overlay
    edges = cv2.Canny(gray, 60, 160)
    kernel = np.ones((2, 2), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)
    red_overlay = np.zeros_like(img)
    red_overlay[:, :, 2] = edges   # red channel only

    return cv2.addWeighted(bw_bgr, 0.85, red_overlay, 0.80, 0)


# Map each gesture to its effect function
EFFECT_MAP = {
    "open_palm":     effect_thermal,
    "fist":          effect_grayscale,
    "peace":         effect_sepia,
    "thumbs_up":     effect_golden,
    "ok_sign":       effect_cool_blue,
    "point_up":      effect_neon_glow,
    "three_fingers": effect_posterize,
    "four_fingers":  effect_pencil_sketch,
    "call_me":       effect_vaporwave,
    "rock":          effect_high_contrast_bw,
}

EFFECT_NAMES = {
    "open_palm":     "Thermal (Inferno)",
    "fist":          "Grayscale + CLAHE",
    "peace":         "Sepia",
    "thumbs_up":     "Golden Autumn",
    "ok_sign":       "Arctic Blue",
    "point_up":      "Neon Glow",
    "three_fingers": "Vivid Posterize",
    "four_fingers":  "Pencil Sketch",
    "call_me":       "Vaporwave",
    "rock":          "High-Contrast B&W",
}


# ─────────────────────────────────────────────────────────────────────────────
# Gesture Detection (requires MediaPipe + trained model)
# ─────────────────────────────────────────────────────────────────────────────

def detect_gesture(img_bgr: np.ndarray, model) -> tuple[str, float]:
    """
    Run MediaPipe Hands on a single image and return (gesture_label, confidence).
    Returns ("unknown", 0.0) if no hand is detected.
    Raises RuntimeError if MediaPipe is not available.
    """
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "MediaPipe not available in this Python environment.\n"
            "Use --gesture to specify a gesture manually, or run in Python ≤3.12."
        )
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return "unknown", 0.0

        hand_landmarks = results.multi_hand_landmarks[0]
        fv = landmarks_to_feature_vector(hand_landmarks)
        if fv is None:
            return "unknown", 0.0

        proba = model.predict_proba([fv])[0]
        idx   = int(np.argmax(proba))
        label = model.classes_[idx]
        conf  = float(proba[idx])
        return label, conf


def draw_label(img: np.ndarray, gesture: str, confidence: float, effect_name: str) -> np.ndarray:
    """Burn a small info banner onto the result image."""
    h, w = img.shape[:2]
    banner_h = 52
    result = img.copy()

    # Semi-transparent dark bar at bottom
    overlay = result.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, result, 0.45, 0, result)

    cv2.putText(
        result, f"{gesture}  |  {effect_name}",
        (10, h - banner_h + 22),
        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    conf_text = f"conf: {confidence * 100:.1f}%"
    cv2.putText(
        result, conf_text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 230, 160), 1, cv2.LINE_AA,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path: pathlib.Path,
    model,
    forced_gesture: str | None = None,
    use_detect: bool = False,
) -> pathlib.Path:
    """Process one image: detect gesture (if requested), apply effect, save output."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if forced_gesture:
        gesture    = forced_gesture
        confidence = 1.0
    elif use_detect:
        print(f"  Detecting gesture …", end=" ", flush=True)
        gesture, confidence = detect_gesture(img, model)
        if gesture == "unknown":
            print("no hand detected — defaulting to open_palm effect.")
            gesture, confidence = "open_palm", 0.0
        else:
            print(f"{gesture} ({confidence * 100:.1f}%)")
    else:
        raise ValueError("Either forced_gesture or use_detect=True must be provided.")

    effect_fn   = EFFECT_MAP[gesture]
    effect_name = EFFECT_NAMES[gesture]

    result = effect_fn(img)
    result = draw_label(result, gesture, confidence, effect_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem    = image_path.stem
    out_path = OUTPUT_DIR / f"{stem}_{gesture}.jpg"
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved → {out_path.relative_to(SCRIPT_DIR)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Apply gesture-driven color effects to images.")
    parser.add_argument(
        "images", nargs="*",
        help="Image file(s) to process. Defaults to all images in input/.",
    )
    parser.add_argument(
        "--gesture", "-g",
        choices=GESTURE_LABELS,
        default=None,
        help="Apply a specific gesture effect. Without this, all 10 effects are saved.",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Auto-detect gesture via MediaPipe + trained model (requires Python ≤3.12).",
    )
    args = parser.parse_args()

    # Model only needed for auto-detection
    model = None
    if args.detect:
        if not MEDIAPIPE_AVAILABLE:
            sys.exit(
                "ERROR: --detect requires mediapipe, which is not installed.\n"
                "       Use a Python ≤3.12 environment, or use --gesture instead."
            )
        if not MODEL_PATH.exists():
            sys.exit(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        print(f"Loading model from {MODEL_PATH.relative_to(SCRIPT_DIR)} …")
        model = joblib.load(MODEL_PATH)

    # Collect images
    if args.images:
        image_paths = [pathlib.Path(p) for p in args.images]
    else:
        image_paths = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not image_paths:
            sys.exit(f"No images found in {INPUT_DIR}.")

    # Determine which gestures to apply
    if args.gesture:
        gestures_to_apply = [args.gesture]
        print(f"Applying effect: {args.gesture} ({EFFECT_NAMES[args.gesture]})\n")
    elif args.detect:
        gestures_to_apply = None  # determined per image
        print("Auto-detecting gesture …\n")
    else:
        gestures_to_apply = GESTURE_LABELS
        print(f"Generating all 10 effects for {len(image_paths)} image(s) …\n")

    print(f"Processing {len(image_paths)} image(s) …\n")
    for img_path in image_paths:
        print(f"[{img_path.name}]")
        try:
            if args.detect:
                # One output per image, gesture auto-detected
                process_image(img_path, model, forced_gesture=None, use_detect=True)
            else:
                for gesture in gestures_to_apply:
                    process_image(img_path, model, forced_gesture=gesture)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        print()

    print("Done. Results saved to output/")


if __name__ == "__main__":
    main()
