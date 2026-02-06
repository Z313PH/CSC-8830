"""
generate_board.py

Purpose:
- Generates an ArUco GridBoard image to display FULL SCREEN on an iPad (used as the calibration target).

How to run:
    python generate_board.py

Output:
- gridboard_ipad.png  (display this on the iPad)

Notes:
- After displaying on the iPad, measure ONE black marker edge length (cm).
- You will use that measurement in Step 2 (calibration script).
"""

import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

MARKERS_X = 4
MARKERS_Y = 5

MARKER_LEN = 200
MARKER_SEP = 50

board_w = MARKERS_X * MARKER_LEN + (MARKERS_X - 1) * MARKER_SEP
board_h = MARKERS_Y * MARKER_LEN + (MARKERS_Y - 1) * MARKER_SEP

MARGIN = 80
W = board_w + 2 * MARGIN
H = board_h + 2 * MARGIN

board = cv2.aruco.GridBoard((MARKERS_X, MARKERS_Y), float(MARKER_LEN), float(MARKER_SEP), aruco_dict)

img = cv2.aruco.drawPlanarBoard(
    board,
    outSize=(int(W), int(H)),
    marginSize=int(MARGIN),
    borderBits=int(1)
)

cv2.imwrite("gridboard_ipad.png", img)
print("Saved gridboard_ipad.png")
print(f"Image size: {W} x {H}")
print("Display FULL SCREEN on your iPad.")
