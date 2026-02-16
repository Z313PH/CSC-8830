"""
THERMAL ANIMAL BOUNDARY DETECTION 

Goal:
- Find the boundary of an animal in a thermal image using only classical OpenCV operations.

How it works :
1) Preprocess: denoise + contrast enhancement (CLAHE)
2) Segment hot region: percentile or Otsu threshold
3) Clean mask: morphological close/open
4) Select animal blob: connected components (largest area by default)
5) Extract boundary: findContours + approxPolyDP
6) Save outputs: binary mask + contour overlay

Inputs:
- A thermal image (grayscale thermal is recommended).
  Example: input/thermal_animal.png

Outputs:
- output/mask_opencv.png     (binary mask of the detected animal)
- output/overlay_opencv.png  (animal boundary drawn on the image)

Run (example):
    python thermal_boundary.py --image input/thermal_animal.png --outdir output --percentile 95

Key parameters to tune:
- --percentile (default 95): lower it if the animal is not fully captured
- morphology kernel size (in code): increase if mask is noisy, decrease if detail is lost

Notes / assumptions:
- Works best when the animal is the hottest object in the scene.
- If the background contains other hot regions, you may need to tune percentile + morphology.
- This is a classical approach only; no deep learning or ML is used here.

(For comparison with SAM2):
- Export SAM2 mask to output/mask_sam2.png
- Compute IoU/Dice against output/mask_opencv.png using a separate comparison script.
"""



import cv2
import numpy as np

def segment_animal_thermal(path, out_prefix="out", percentile=95):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # 1) Preprocess
    den = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(den)

    # 2) Threshold (percentile tends to be stable on thermal)
    t = np.percentile(norm, percentile)
    mask = (norm >= t).astype(np.uint8) * 255

    # 3) Morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # 4) Keep best connected component (largest area)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        raise RuntimeError("No foreground found. Try lowering percentile (e.g., 90).")

    # skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + np.argmax(areas)

    mask2 = (labels == best).astype(np.uint8) * 255

    # 5) Boundary
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found.")

    cnt = max(contours, key=cv2.contourArea)

    eps = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)

    # Output visuals
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)

    cv2.imwrite(f"{out_prefix}_mask.png", mask2)
    cv2.imwrite(f"{out_prefix}_overlay.png", overlay)
    return mask2, approx

if __name__ == "__main__":
    segment_animal_thermal("thermal_animal.jpg", out_prefix="result", percentile=95)
