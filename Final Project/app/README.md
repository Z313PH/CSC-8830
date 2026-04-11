# Real-Time Gesture Recognition — Part 1

A complete keypoints-based gesture recognition pipeline for macOS webcam.
Uses MediaPipe Hands + scikit-learn. No image files — pure live-feed landmark classification.

---

## Folder Structure

```
gesture_recognition/
├── collect_data.py       # Step 1: Record landmark data per gesture
├── train_model.py        # Step 2: Train + evaluate scikit-learn classifier
├── run_live_demo.py      # Step 3: Live webcam inference
├── utils.py              # Shared: normalization, smoothing, FPS, overlay
├── requirements.txt
├── data/                 # Created automatically — CSV landmark files go here
└── models/               # Created automatically — model.joblib + plots saved here
```

---

## Quick Start

### 1. Install Dependencies

```bash
# macOS — Python 3.10+ recommended
pip install -r requirements.txt
```

> **Note:** If you use a virtual environment (recommended):
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

---

### 2. Grant Camera Permission (macOS)

The first time you run any script, macOS will prompt for camera access.
If it was previously denied:

```
System Settings → Privacy & Security → Camera
→ Toggle ON for Terminal (or your IDE / Python executable)
```

Then restart Terminal and try again.

---

### 3. Collect Training Data

```bash
python collect_data.py
```

**Controls inside the window:**

| Key     | Action                              |
|---------|-------------------------------------|
| `1`–`9`, `0` | Select gesture class (see list below) |
| `r`     | Toggle recording ON/OFF             |
| `s`     | Save dataset to disk now            |
| `q`     | Quit (auto-saves)                   |

**Recording workflow per class:**
1. Press the number key for that gesture
2. Position your hand in front of the camera
3. Press `r` to start recording — hold the gesture steady
4. Move hand slightly (different angles, distances) while holding gesture
5. Press `r` again to stop
6. Repeat from a **different session** (different lighting, background) for robustness
7. Move to the next class

**Target:** 200–500 samples per class across 2–3 sessions.

---

### 4. Train the Model

```bash
# Default: SVM with RBF kernel (best baseline for this task)
python train_model.py

# Or choose another classifier:
python train_model.py --clf logistic
python train_model.py --clf random_forest

# Or use a specific CSV file:
python train_model.py --data data/landmarks_20240101_120000.csv
```

Outputs saved to `models/`:
- `model.joblib` — trained Pipeline (StandardScaler + classifier)
- `confusion_matrix.png` — per-class confusion visualization
- `training_report.txt` — accuracy + precision/recall/F1

---

### 5. Run Live Demo

```bash
python run_live_demo.py
```

Press `t` to toggle temporal smoothing ON/OFF for the ablation comparison.
Press `q` to quit.

---

## Gesture Set (10 Classes)

Chosen for maximum visual distinctiveness with a single hand and 21 landmarks.

| Key | Label          | Description                                    | Common Confusion    |
|-----|----------------|------------------------------------------------|---------------------|
| `1` | `open_palm`    | All 5 fingers extended, spread wide            | `four_fingers`      |
| `2` | `fist`         | All fingers fully curled into palm             | —                   |
| `3` | `peace`        | Index + middle extended, V shape               | `point_up`          |
| `4` | `thumbs_up`    | Thumb pointing up, other fingers curled        | `fist` at distance  |
| `5` | `ok_sign`      | Thumb + index form circle, others extended     | —                   |
| `6` | `point_up`     | Only index finger extended upward              | `peace`             |
| `7` | `three_fingers`| Index + middle + ring extended                 | `four_fingers`      |
| `8` | `four_fingers` | All fingers except thumb extended              | `open_palm`         |
| `9` | `call_me`      | Thumb + pinky extended (shaka / "hang loose")  | `rock`              |
| `0` | `rock`         | Index + pinky extended (devil horns)           | `call_me`           |

### Recording Tips

- **`open_palm` vs `four_fingers`**: Keep thumb clearly spread outward for open_palm; tuck it for four_fingers.
- **`peace` vs `point_up`**: In peace, both index and middle must be clearly separated. In point_up, middle curls down.
- **`call_me` vs `rock`**: call_me = thumb OUT; rock = thumb folded in. Record exaggerating this difference.
- **All gestures**: Record at 3 distances (close, medium, far) and 2–3 wrist angles (palm toward camera, slightly rotated).
- **Lighting**: Record one session in bright light and one in ambient/dimmer light for robustness.

---

## Pipeline Details

### Feature Vector (42-dim)

```
Input: MediaPipe 21 landmarks × (x, y, z)
→ Center: subtract wrist (landmark 0)
→ Scale:  divide by wrist→middle-MCP (landmark 9) distance
→ Drop z (unstable from webcam depth)
→ Flatten: [x1, y1, x2, y2, ..., x21, y21]  →  shape (42,)
```

### Classifier

Default: **SVM (RBF kernel)** with `probability=True` for `predict_proba` support.
Wrapped in `sklearn.pipeline.Pipeline` with `StandardScaler`.

### Temporal Smoothing

```
Rolling window (N=15 frames)
→ Confidence-weighted vote per class
→ Accept if: winning class count ≥ 6 frames AND avg confidence ≥ 0.60
→ Otherwise: output "Unknown"
```

---

## Art Engine Integration

`run_live_demo.py` exposes two module-level variables you can poll:

```python
# In your Art Engine script:
import run_live_demo as demo
import threading

# Start the demo loop in a background thread
t = threading.Thread(target=demo.main, daemon=True)
t.start()

# Poll in your art loop:
while True:
    label = demo.current_label       # str: e.g. "peace" or "Unknown"
    conf  = demo.current_confidence  # float: 0.0 – 1.0
    # ... drive your art with these values
```

---

## Troubleshooting (macOS)

### "Cannot open webcam" / blank window
```bash
# Check camera access
System Settings → Privacy & Security → Camera → enable your terminal/IDE

# Test if OpenCV can see the camera:
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"
```

### mediapipe ImportError or crash on Apple Silicon
```bash
# Ensure you're using a native arm64 Python (not Rosetta)
python3 -c "import platform; print(platform.machine())"  # Should print 'arm64'
pip install mediapipe  # No special flags needed for M1/M2/M3 on Python 3.10+
```

### Predictions flicker rapidly
- Increase `SMOOTH_WINDOW` in `run_live_demo.py` (try 20–25)
- Increase `MIN_STABLE_FRAMES` (try 8–10)
- Or collect more training data for the confused classes

### Low accuracy on a specific class
- Check the confusion matrix: `models/confusion_matrix.png`
- Collect 100+ more samples for that class
- Try exaggerating the gesture (more pronounced finger extension)

### "No landmark CSVs found"
- Run `collect_data.py` first and save with `s` or `q`
- Ensure you're running `train_model.py` from the project root directory

### OpenCV window doesn't appear / app hangs
```bash
# macOS sometimes requires GUI calls on the main thread
# Always run scripts directly, not inside Jupyter
python collect_data.py  # ✓
# Not: jupyter notebook  # ✗ for OpenCV GUI
```
