# Module 2

This repo contains my solution for the Module 2 assignment:

1) **Camera calibration** using OpenCV built-in ArUco tools (smartphone camera + GridBoard displayed on an iPad screen).  
2) **Real-world 2D measurement** of an object on a plane using perspective projection equations (homography).  
3) **Validation** from a distance greater than 2 meters.

---

## Requirements
- Python 3.x
- OpenCV with ArUco module:
  ```bash
  pip install opencv-contrib-python
  ```

---

## Step 0: Generate the GridBoard (iPad target)

Run:
```bash
python generate_board.py
```

This creates `gridboard_ipad.png`. Display it **full-screen** on your iPad.

Measure the **edge length of one black marker** on the iPad with a ruler.
- In my setup: **2.7 cm** (so `MARKER_LENGTH = 0.027 m`)

---

## Step 1: Camera Calibration

1. Take **20–30 photos** of the iPad GridBoard using your smartphone from different angles/positions (board should be sharp and clearly visible).
2. If your phone saves images as `.HEIC`, convert them to `.jpg` (example command for macOS):
   ```bash
   mkdir -p grid_calib_jpg
   for f in grid_calib/*.HEIC; do
     sips -s format jpeg "$f" --out "grid_calib_jpg/$(basename "${f%.*}").jpg"
   done
   ```
3. Put the calibration photos in the folder used by the script.
4. Run:
   ```bash
   python calibration_script.py
   ```

Output:
- `calibration_gridboard.npz` containing the camera intrinsics `K` and distortion coefficients `dist`
- Console prints `K`, `dist`, and RMS reprojection error

---

## Step 2: Real-world 2D measurement (Homography / Perspective Projection)

Take one photo `measurement.jpg` that contains:
- the GridBoard on the iPad
- the object you want to measure on the **same plane** as the iPad screen (coplanar)

Run:
```bash
python measurement_script.py
```

The script will ask you to click 4 points:
1) left endpoint  
2) right endpoint  (WIDTH)  
3) top endpoint  
4) bottom endpoint (HEIGHT)

It prints the estimated width/height in cm.

---

## Step 3: Validation (> 2 meters)

1. Choose a distance **> 2m** (example: **2.50 m**) from the camera to the plane (iPad screen plane).
2. Measure that distance with a tape measure.
3. Take the measurement photo from that distance.
4. Measure the object’s true width/height with a ruler and compare to the script output (absolute + percent error).

---

## Notes / Assumption
This method assumes the object lies on the **same plane** as the GridBoard (planar homography).  
If the object is not coplanar with the board plane, the measured dimensions will be incorrect.
