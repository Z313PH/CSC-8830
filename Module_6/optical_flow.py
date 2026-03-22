import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import sys


def create_output_dir(dirname="output_optical_flow"):
    os.makedirs(dirname, exist_ok=True)
    return dirname


def compute_dense_optical_flow(video_path, output_path, label="video"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print(f"[ERROR] Cannot read first frame from: {video_path}")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    frame_count = 0
    first_flow = None
    first_pair = None

    print(f"  Computing dense optical flow for {label}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        if frame_count == 0:
            first_flow = flow.copy()
            first_pair = (first_frame.copy(), frame.copy())

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(flow_vis)

        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    out.release()
    print(f"  Dense flow saved: {output_path} ({frame_count} frames)")

    return first_flow, first_pair


def compute_sparse_optical_flow(video_path, output_path, label="video"):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    h, w = old_frame.shape[:2]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    color = np.random.randint(0, 255, (300, 3))
    mask = np.zeros_like(old_frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    frame_count = 0
    print(f"  Computing sparse optical flow (Lucas-Kanade) for {label}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), color[i % 300].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i % 300].tolist(), -1)

                img = cv2.add(frame, mask)
                out.write(img)

                p0 = good_new.reshape(-1, 1, 2)
            else:
                out.write(frame)
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(old_frame)
        else:
            out.write(frame)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(old_frame)

        old_gray = frame_gray.copy()
        frame_count += 1

        if frame_count % 30 == 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(old_frame)

    cap.release()
    out.release()
    print(f"  Sparse flow saved: {output_path} ({frame_count} frames)")


def validate_tracking(flow, frame_pair, outdir, label="video"):
    frame1, frame2 = frame_pair
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h, w = gray1.shape

    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=20, qualityLevel=0.3, minDistance=30)
    if corners is None:
        print(f"  [WARNING] No corners found for {label}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Frame 1 - Selected Points ({label})", fontsize=12)
    for i, c in enumerate(corners):
        x, y = c.ravel()
        axes[0].plot(x, y, 'ro', markersize=8)
        axes[0].annotate(f'P{i}', (x+5, y-5), color='yellow', fontsize=8,
                        fontweight='bold')

    axes[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Frame 2 - Predicted (green) vs Actual (red)", fontsize=12)

    errors = []
    patch_size = 15

    for i, c in enumerate(corners):
        x, y = int(c[0, 0]), int(c[0, 1])

        fx = min(max(x, 0), w - 1)
        fy = min(max(y, 0), h - 1)

        u = flow[fy, fx, 0]
        v = flow[fy, fx, 1]
        pred_x = x + u
        pred_y = y + v

        y1 = max(0, y - patch_size)
        y2 = min(h, y + patch_size + 1)
        x1 = max(0, x - patch_size)
        x2 = min(w, x + patch_size + 1)
        template = gray1[y1:y2, x1:x2]

        if template.shape[0] < 5 or template.shape[1] < 5:
            continue

        search_margin = 30
        sy1 = max(0, int(pred_y) - search_margin - patch_size)
        sy2 = min(h, int(pred_y) + search_margin + patch_size + 1)
        sx1 = max(0, int(pred_x) - search_margin - patch_size)
        sx2 = min(w, int(pred_x) + search_margin + patch_size + 1)
        search_region = gray2[sy1:sy2, sx1:sx2]

        if search_region.shape[0] < template.shape[0] or \
           search_region.shape[1] < template.shape[1]:
            continue

        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        actual_x = sx1 + max_loc[0] + patch_size
        actual_y = sy1 + max_loc[1] + patch_size

        error = np.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)
        errors.append(error)

        axes[1].plot(pred_x, pred_y, 'go', markersize=10, label='Predicted' if i == 0 else '')
        axes[1].plot(actual_x, actual_y, 'rx', markersize=10, mew=2, label='Actual' if i == 0 else '')
        axes[1].annotate(f'P{i}', (pred_x + 5, pred_y - 5), color='lime', fontsize=8)
        axes[1].annotate('', xy=(actual_x, actual_y), xytext=(pred_x, pred_y),
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))

    if len(errors) > 0:
        axes[1].legend(loc='upper right')

    if len(errors) > 0:
        axes[2].bar(range(len(errors)), errors, color='steelblue', edgecolor='navy')
        axes[2].set_xlabel('Point Index', fontsize=11)
        axes[2].set_ylabel('Euclidean Error (pixels)', fontsize=11)
        axes[2].set_title(f'Tracking Error: Mean={np.mean(errors):.2f}px', fontsize=12)
        axes[2].axhline(y=np.mean(errors), color='red', linestyle='--', label=f'Mean={np.mean(errors):.2f}')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No valid points', ha='center', va='center')

    plt.tight_layout()
    save_path = os.path.join(outdir, f"tracking_validation_{label}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Tracking validation saved: {save_path}")

    print(f"\n  --- Tracking Validation Results ({label}) ---")
    print(f"  {'Point':<8} {'Error (px)':<12}")
    print(f"  {'-'*20}")
    for i, e in enumerate(errors):
        print(f"  P{i:<7} {e:<12.3f}")
    if errors:
        print(f"  {'Mean':<8} {np.mean(errors):<12.3f}")
    print()


def save_flow_magnitude_map(flow, frame, outdir, label="video"):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Frame ({label})")
    axes[0].axis('off')

    im = axes[1].imshow(mag, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f"Flow Magnitude ({label})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='pixels/frame')

    plt.tight_layout()
    save_path = os.path.join(outdir, f"flow_magnitude_{label}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Flow magnitude map saved: {save_path}")


def demonstrate_bilinear_interpolation(outdir):
    print("\n  === Bilinear Interpolation Demonstration ===")

    patch = np.array([
        [120, 150, 170, 180],
        [130, 160, 175, 190],
        [140, 165, 185, 200],
        [145, 170, 195, 210]
    ], dtype=np.float32)

    qx, qy = 1.3, 1.7

    x1, y1 = int(np.floor(qx)), int(np.floor(qy))
    x2, y2 = x1 + 1, y1 + 1

    Q11 = patch[y1, x1]
    Q21 = patch[y1, x2]
    Q12 = patch[y2, x1]
    Q22 = patch[y2, x2]

    a = qx - x1
    b = qy - y1

    f_xy1 = (1 - a) * Q11 + a * Q21
    f_xy2 = (1 - a) * Q12 + a * Q22
    f_xy = (1 - b) * f_xy1 + b * f_xy2

    print(f"  Query point: ({qx}, {qy})")
    print(f"  Neighbors: Q11={Q11}, Q21={Q21}, Q12={Q12}, Q22={Q22}")
    print(f"  Fractional: a={a:.1f}, b={b:.1f}")
    print(f"  Step 1 (x-interp): f(x,y1)={f_xy1:.1f}, f(x,y2)={f_xy2:.1f}")
    print(f"  Step 2 (y-interp): f({qx},{qy}) = {f_xy:.2f}")

    cv_result = cv2.getRectSubPix(patch, (1, 1), (qx, qy))[0, 0]
    print(f"  OpenCV verification: {cv_result:.2f}")
    print(f"  Difference: {abs(f_xy - cv_result):.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(patch, cmap='gray', interpolation='nearest',
                        extent=[-0.5, 3.5, 3.5, -0.5])
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].grid(True, alpha=0.3)
    for iy in range(4):
        for ix in range(4):
            axes[0].text(ix, iy, f'{patch[iy, ix]:.0f}', ha='center', va='center',
                        color='red', fontsize=10, fontweight='bold')
    axes[0].plot(qx, qy, 'g*', markersize=20, label=f'Query ({qx},{qy})')
    axes[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-', lw=2, label='Neighbors')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Bilinear Interpolation: Grid & Query Point')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    upscale = 20
    h_up, w_up = patch.shape[0] * upscale, patch.shape[1] * upscale
    upsampled = cv2.resize(patch, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
    axes[1].imshow(upsampled, cmap='gray', extent=[-0.5, 3.5, 3.5, -0.5])
    axes[1].plot(qx, qy, 'g*', markersize=20)
    axes[1].set_title(f'Bilinear Result: f({qx},{qy}) = {f_xy:.2f}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(outdir, "bilinear_interpolation_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bilinear interpolation plot saved: {save_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Optical Flow Computation and Tracking Validation"
    )
    parser.add_argument('--video1', type=str, required=True,
                        help='Path to first video (30s with motion)')
    parser.add_argument('--video2', type=str, required=True,
                        help='Path to second video (30s with motion)')
    args = parser.parse_args()

    outdir = create_output_dir()

    print("=" * 60)
    print("  OPTICAL FLOW & TRACKING VALIDATION")
    print("=" * 60)

    print(f"\n[Video 1] {args.video1}")
    flow1, pair1 = compute_dense_optical_flow(
        args.video1, os.path.join(outdir, "optical_flow_video1.avi"), "video1"
    )
    compute_sparse_optical_flow(
        args.video1, os.path.join(outdir, "sparse_flow_video1.avi"), "video1"
    )

    if flow1 is not None and pair1 is not None:
        validate_tracking(flow1, pair1, outdir, "video1")
        save_flow_magnitude_map(flow1, pair1[0], outdir, "video1")

    print(f"\n[Video 2] {args.video2}")
    flow2, pair2 = compute_dense_optical_flow(
        args.video2, os.path.join(outdir, "optical_flow_video2.avi"), "video2"
    )
    compute_sparse_optical_flow(
        args.video2, os.path.join(outdir, "sparse_flow_video2.avi"), "video2"
    )

    if flow2 is not None and pair2 is not None:
        validate_tracking(flow2, pair2, outdir, "video2")
        save_flow_magnitude_map(flow2, pair2[0], outdir, "video2")

    demonstrate_bilinear_interpolation(outdir)

    print("=" * 60)
    print("  ALL OUTPUTS SAVED TO:", outdir)
    print("=" * 60)


if __name__ == "__main__":
    main()
