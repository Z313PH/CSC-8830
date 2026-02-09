"""
blur_compare.py

Purpose:
- Implement image blurring using a spatial filter (convolution in space).
- Implement the Fourier-domain equivalent (multiplication in frequency).
- Show that spatial convolution == frequency multiplication (Convolution Theorem).
- Verify the outputs match (numerical error + saved images).

How to run:
    python blur_compare.py --image input/your_image.jpg --kernel gaussian --ksize 21 --sigma 4.0

Outputs (saved in output/):
- spatial.png: blurred image using spatial convolution
- freq.png: blurred image using FFT multiplication
- diff.png: absolute difference image (scaled for visibility)
- comparison.png: side-by-side summary

Notes:
- To make spatial and frequency results match, we compute LINEAR convolution:
  we pad image and kernel to (H+kh-1, W+kw-1), multiply FFTs, then crop to "same" size.
"""
