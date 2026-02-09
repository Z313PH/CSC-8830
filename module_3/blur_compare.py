"""

Purpose:
- Implement image blurring using a spatial filter (convolution in space).
- Implement the Fourier-domain equivalent (multiplication in frequency).
- Demonstrate that convolution in space equals multiplication in frequency (Convolution Theorem).
- Compare results for BOTH a box (mean) filter and a Gaussian filter.

How to run:
    python blur_compare.py --image input/your_image.jpg --outdir output

Optional args:
    --ksize 21
    --sigma 4.0
    --save_float 0/1

Outputs (saved in outdir/):
- box_spatial.png, box_freq.png, box_diff.png, box_comparison.png, box_metrics.txt
- gauss_spatial.png, gauss_freq.png, gauss_diff.png, gauss_comparison.png, gauss_metrics.txt

Notes (important):
- DFT multiplication corresponds to circular convolution.
- To match spatial linear convolution, we use ZERO-PADDING and compute linear convolution via FFT:
  pad to (H+kh-1, W+kw-1), multiply FFTs, IFFT, then crop to SAME size.
- We use BORDER_CONSTANT (zero outside image) for spatial filtering to match FFT padding.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_box_kernel(ksize: int) -> np.ndarray:
    k = np.ones((ksize, ksize), dtype=np.float64)
    k /= k.sum()
    return k


def make_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k /= k.sum()
    return k


def spatial_same_zeropad(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Spatial-domain blur: SAME output size using zero-padding outside the image.
    Implemented with OpenCV filter2D and BORDER_CONSTANT.
    """
    img_f = img.astype(np.float64)
    k = kernel.astype(np.float64)
    out = cv2.filter2D(img_f, ddepth=cv2.CV_64F, kernel=k, borderType=cv2.BORDER_CONSTANT)
    return out


def freq_same_zeropad(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    kernel = kernel.astype(np.float64)

    H, W = img.shape
    kh, kw = kernel.shape
    P = H + kh - 1
    Q = W + kw - 1

    img_pad = np.zeros((P, Q), dtype=np.float64)
    img_pad[:H, :W] = img

    ker_pad = np.zeros((P, Q), dtype=np.float64)
    ker_pad[:kh, :kw] = kernel

    ker_pad = np.roll(ker_pad, -kh // 2, axis=0)
    ker_pad = np.roll(ker_pad, -kw // 2, axis=1)

    G_full = np.fft.ifft2(np.fft.fft2(img_pad) * np.fft.fft2(ker_pad)).real

    # SAME crop (matches filter2D centered kernel)
    y0, x0 = kh // 2, kw // 2
    return G_full[y0:y0 + H, x0:x0 + W]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / m)


def save_u8(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))


def save_float_npy(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, img.astype(np.float64))


def write_metrics(path: str, label: str, kernel_name: str, ksize: int, sigma: float,
                  max_err: float, mse_all: float, psnr_all: float,
                  max_err_inner: float, mse_inner: float, psnr_inner: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{label}\n")
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"ksize: {ksize}\n")
        if kernel_name == "gaussian":
            f.write(f"sigma: {sigma}\n")
        f.write("\n--- Over full image ---\n")
        f.write(f"Max abs error: {max_err:.6f}\n")
        f.write(f"MSE: {mse_all:.6f}\n")
        f.write(f"PSNR: {psnr_all:.2f} dB\n")
        f.write("\n--- Inner region (excluding border = ksize//2) ---\n")
        f.write(f"Max abs error (inner): {max_err_inner:.6f}\n")
        f.write(f"MSE (inner): {mse_inner:.6f}\n")
        f.write(f"PSNR (inner): {psnr_inner:.2f} dB\n")


def run_one(img: np.ndarray, kernel: np.ndarray, kernel_name: str, ksize: int, sigma: float, outdir: str, save_float: bool):
    spatial = spatial_same_zeropad(img, kernel)
    freq = freq_same_zeropad(img, kernel)

    abs_diff = np.abs(spatial - freq)

    max_err = float(abs_diff.max())
    mse_all = mse(spatial, freq)
    psnr_all = psnr(spatial, freq)

    # Inner region comparison (removes boundary/padding effects)
    b = ksize // 2
    if b > 0 and img.shape[0] > 2*b and img.shape[1] > 2*b:
        inner_spatial = spatial[b:-b, b:-b]
        inner_freq = freq[b:-b, b:-b]
        inner_diff = np.abs(inner_spatial - inner_freq)
        max_err_inner = float(inner_diff.max())
        mse_inner = mse(inner_spatial, inner_freq)
        psnr_inner = psnr(inner_spatial, inner_freq)
    else:
        max_err_inner, mse_inner, psnr_inner = max_err, mse_all, psnr_all

    print(f"\n=== {kernel_name.upper()} (Spatial vs Frequency) ===")
    if kernel_name == "gaussian":
        print(f"ksize={ksize}, sigma={sigma}")
    else:
        print(f"ksize={ksize}")
    print(f"Max abs error (all):   {max_err:.6f}")
    print(f"MSE (all):             {mse_all:.6f}")
    print(f"PSNR (all):            {psnr_all:.2f} dB")
    print(f"Max abs error (inner): {max_err_inner:.6f}")
    print(f"MSE (inner):           {mse_inner:.6f}")
    print(f"PSNR (inner):          {psnr_inner:.2f} dB")

    # Save images
    prefix = "box" if kernel_name == "box" else "gauss"

    save_u8(os.path.join(outdir, f"{prefix}_spatial.png"), spatial)
    save_u8(os.path.join(outdir, f"{prefix}_freq.png"), freq)

    # Difference visualization (scaled)
    diff_vis = abs_diff.copy()
    diff_vis = (diff_vis / (diff_vis.max() + 1e-12)) * 255.0
    save_u8(os.path.join(outdir, f"{prefix}_diff.png"), diff_vis)

    # Side-by-side comparison
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1); plt.title("Original"); plt.imshow(img, cmap="gray"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.title("Spatial"); plt.imshow(np.clip(spatial, 0, 255), cmap="gray"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.title("Frequency"); plt.imshow(np.clip(freq, 0, 255), cmap="gray"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.title("Abs Diff (scaled)"); plt.imshow(diff_vis, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_comparison.png"), dpi=200)
    plt.close()

    # Metrics txt
    write_metrics(
        os.path.join(outdir, f"{prefix}_metrics.txt"),
        label="Spatial vs Frequency Blur Comparison",
        kernel_name=kernel_name,
        ksize=ksize,
        sigma=sigma,
        max_err=max_err,
        mse_all=mse_all,
        psnr_all=psnr_all,
        max_err_inner=max_err_inner,
        mse_inner=mse_inner,
        psnr_inner=psnr_inner
    )

    if save_float:
        save_float_npy(os.path.join(outdir, f"{prefix}_spatial.npy"), spatial)
        save_float_npy(os.path.join(outdir, f"{prefix}_freq.npy"), freq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (jpg/png).")
    parser.add_argument("--ksize", type=int, default=21, help="Odd kernel size.")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian sigma.")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--save_float", type=int, default=0, help="1 to also save .npy float outputs.")
    args = parser.parse_args()

    if args.ksize % 2 == 0:
        raise ValueError("ksize must be odd.")

    os.makedirs(args.outdir, exist_ok=True)

    img_u8 = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    img = img_u8.astype(np.float64)

    box_kernel = make_box_kernel(args.ksize)
    gauss_kernel = make_gaussian_kernel(args.ksize, args.sigma)

    run_one(img, box_kernel, "box", args.ksize, args.sigma, args.outdir, bool(args.save_float))
    run_one(img, gauss_kernel, "gaussian", args.ksize, args.sigma, args.outdir, bool(args.save_float))

    print(f"\nSaved all outputs to: {args.outdir}/")


if __name__ == "__main__":
    main()
