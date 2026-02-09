"""
measurement_script.py

Purpose:
- Measures real-world 2D width/height of an object lying on the same plane as the GridBoard,
  using perspective projection equations and a homography.

How to run:
    python measurement_script.py

Inputs:
- calibration_gridboard.npz (from Step 1)
- measurement.jpg  (photo containing GridBoard + object on same plane)

Instructions:
- The script opens a window. Click 4 points:
  1) left endpoint, 2) right endpoint (width)
  3) top endpoint, 4) bottom endpoint (height)

Output:
- Prints estimated width/height in cm.

Assumption:
- Object is coplanar with the GridBoard plane (Z=0). If not coplanar, results will be wrong.
"""



import cv2
import numpy as np

CALIB_FILE = "calibration_gridboard.npz"
IMAGE_FILE = "measurement2.jpg"

# load aclibration
data = np.load(CALIB_FILE)
K = data["K"]
dist = data["dist"]
MARKER_LENGTH = float(data["marker_length"])
MARKER_SEPARATION = float(data["marker_separation"])
MARKERS_X = int(data["markers_x"])
MARKERS_Y = int(data["markers_y"])

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.GridBoard((MARKERS_X, MARKERS_Y), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# load image
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"Could not read {IMAGE_FILE}")

h, w = img.shape[:2]
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
und = cv2.undistort(img, K, dist, None, newK)
gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)


corners, ids, _ = detector.detectMarkers(gray)
if ids is None or len(ids) < 4:
    raise RuntimeError("Not enough ArUco markers detected. Retake the photo.")


cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)


ok, rvec, tvec = cv2.aruco.estimatePoseBoard(
    corners, ids, board, newK, np.zeros((5, 1)), None, None
)
if not ok:
    raise RuntimeError("Pose estimation failed. Retake the photo.")

R, _ = cv2.Rodrigues(rvec)
r1 = R[:, 0].reshape(3, 1)
r2 = R[:, 1].reshape(3, 1)


# H maps (X,Y,1) on the board plane -> (u,v,1) in the image
H = newK @ np.hstack([r1, r2, tvec])
Hinv = np.linalg.inv(H)

def img_to_world(u, v):
    """Map pixel (u,v) -> world plane (X,Y) in meters."""
    p = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    Pw = Hinv @ p
    Pw /= Pw[2, 0]
    return float(Pw[0, 0]), float(Pw[1, 0])

# Click points to measure
clicked = []
show = und.copy()
cv2.aruco.drawDetectedMarkers(show, corners, ids)
cv2.drawFrameAxes(show, newK, np.zeros((5, 1)), rvec, tvec, MARKER_LENGTH * 2)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = img_to_world(x, y)
        clicked.append((x, y, X, Y))
        print(f"Pixel ({x},{y}) -> World (X={X:.4f} m, Y={Y:.4f} m)")
        cv2.circle(show, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(show, str(len(clicked)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print("Click 4 points:")
print("  1) left endpoint   2) right endpoint   (WIDTH)")
print("  3) top endpoint    4) bottom endpoint  (HEIGHT)")
print("Press ESC to quit.")

cv2.namedWindow("Measure")
cv2.setMouseCallback("Measure", on_mouse)

while True:
    cv2.imshow("Measure", show)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    if len(clicked) >= 4:
        break

cv2.destroyAllWindows()

if len(clicked) < 4:
    raise RuntimeError("Not enough points clicked.")

def dist_m(a, b):
    return float(np.sqrt((a[2] - b[2])**2 + (a[3] - b[3])**2))

width_m = dist_m(clicked[0], clicked[1])
height_m = dist_m(clicked[2], clicked[3])

print("\n=== Estimated object dimensions on plane ===")
print(f"Width  = {width_m:.4f} m  ({width_m * 100:.2f} cm)")
print(f"Height = {height_m:.4f} m  ({height_m * 100:.2f} cm)")


# Validation  
W_true_cm = 7.5  
H_true_cm = 7.5  

W_est_cm = width_m * 100
H_est_cm = height_m * 100

abs_err_w = abs(W_est_cm - W_true_cm)
abs_err_h = abs(H_est_cm - H_true_cm)

pct_err_w = abs_err_w / W_true_cm * 100
pct_err_h = abs_err_h / H_true_cm * 100

print("\n=== Validation Results ===")
print(f"True Width  = {W_true_cm:.2f} cm, Estimated Width  = {W_est_cm:.2f} cm")
print(f"True Height = {H_true_cm:.2f} cm, Estimated Height = {H_est_cm:.2f} cm")
print(f"Abs Error Width  = {abs_err_w:.2f} cm  ({pct_err_w:.2f}%)")
print(f"Abs Error Height = {abs_err_h:.2f} cm  ({pct_err_h:.2f}%)")
