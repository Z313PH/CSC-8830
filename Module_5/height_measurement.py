"""
Prototype: Estimate a person's height from a single photo using an A4 paper reference.

Reference:
- A4 paper height = 29.7 cm (paper must be vertical and on the SAME wall plane as the person)

How to run:
    python height_measurement.py --image input/photo.jpg

Controls:
- Click 4 points:
  1) top of head
  2) bottom of feet
  3) top of A4 paper
  4) bottom of A4 paper
- Press 'r' to reset
- Press 'q' or ESC to quit
"""

import cv2
import argparse
import numpy as np

A4_HEIGHT_CM = 29.7

clicks = []
labels = [
    "1) Click TOP of HEAD",
    "2) Click BOTTOM of FEET",
    "3) Click TOP of A4 PAPER",
    "4) Click BOTTOM of A4 PAPER"
]

def on_mouse(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))
        print(labels[len(clicks)-1], "->", (x, y))

def main():
    global clicks
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="/input/photo.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    win = "Height Estimator (Photo)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = img.copy()

        # Draw clicked points
        for i, (x, y) in enumerate(clicks):
            cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # On-screen instruction
        if len(clicks) < 4:
            cv2.putText(disp, labels[len(clicks)], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Compute height with 4 points
        if len(clicks) == 4:
            head, feet, a4_top, a4_bot = clicks

            person_px = abs(head[1] - feet[1])
            a4_px = abs(a4_top[1] - a4_bot[1])

            if a4_px > 0:
                height_cm = (person_px / a4_px) * A4_HEIGHT_CM
                cv2.putText(disp, f"Estimated height: {height_cm:.1f} cm",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                cv2.putText(disp, f"person_px={person_px:.1f}, a4_px={a4_px:.1f}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            clicks = []
            print("Reset.")
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    