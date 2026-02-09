"""
calibration_script.py

Purpose:
- Calibrates the smartphone camera using OpenCV's built-in ArUco GridBoard calibration.

How to run:
    python calibration_script.py

Inputs:
- Calibration photos in: grid_calib/*.jpg

Important settings:
- MARKER_LENGTH is the real-world black marker edge length (in meters), measured on the iPad screen.
  Example: 2.7 cm -> 0.027 m
- MARKER_SEPARATION is set based on how the GridBoard was generated (ratio = 0.25 in this repo).

Outputs:
- outputs/calibration_gridboard.npz (contains K, dist)

Notes:
- Take 20–30 images with different angles and board positions in the frame.
"""



import glob
import cv2
import numpy as np

IMAGE_GLOB = "grid_calib/*.jpg"
OUT_FILE = "calibration_gridboard.npz"

MARKER_LENGTH = 0.027
MARKERS_X = 4
MARKERS_Y = 5
MARKER_SEPARATION = MARKER_LENGTH * 0.25  # 0.00675

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.GridBoard((MARKERS_X, MARKERS_Y), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

images = sorted(glob.glob(IMAGE_GLOB))
if not images:
    raise FileNotFoundError(f"No images found at {IMAGE_GLOB}")

img_size = None

all_corners_flat = []  
all_ids_flat = []      
marker_counter = []    

valid = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print("Could not read:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    corners, ids, _ = detector.detectMarkers(gray)
    n = 0 if ids is None else len(ids)
    print(f"{fname}: markers={n}")

    if ids is None or n < 4:
        continue

    for c in corners:
        all_corners_flat.append(c.astype(np.float32))

    all_ids_flat.extend(ids.flatten().tolist())

    marker_counter.append(n)
    valid += 1

print("Valid frames:", valid)
if valid < 10:
    raise RuntimeError("Too few valid frames.")

all_ids_flat = np.array(all_ids_flat, dtype=np.int32).reshape(-1, 1)
marker_counter = np.array(marker_counter, dtype=np.int32)

# Calibrate from detected markers 
rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
    corners=all_corners_flat,
    ids=all_ids_flat,
    counter=marker_counter,
    board=board,
    imageSize=img_size,
    cameraMatrix=None,
    distCoeffs=None
)

np.savez(
    OUT_FILE,
    K=K,
    dist=dist,
    marker_length=MARKER_LENGTH,
    marker_separation=MARKER_SEPARATION,
    markers_x=MARKERS_X,
    markers_y=MARKERS_Y
)

print("\nSaved:", OUT_FILE)
print("K:\n", K)
print("dist:\n", dist.ravel())
print("RMS reprojection error:", rms)
