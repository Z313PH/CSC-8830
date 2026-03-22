"""
=============================================================================
CSc 8830: Computer Vision - Assignment 6
Part 2: Structure from Motion (SfM) - 4 Viewpoints of a Planar Object
=============================================================================

README / How to Execute:
-------------------------
1. Install dependencies:
       pip install opencv-python numpy matplotlib scipy

2. Capture 4 images of a FLAT/PLANAR object (e.g., a book, poster, box face)
   from 4 different viewpoints. Make sure:
   - The object is visible in all 4 images
   - Camera positions vary (different angles/distances)
   - Good lighting, minimal blur

3. Usage:
       python structure_from_motion.py --images img1.jpg img2.jpg img3.jpg img4.jpg

   Optional arguments:
       --focal_length 1000    (estimated focal length in pixels; default=1000)
       --output_dir output_sfm

   Example:
       python structure_from_motion.py --images view1.jpg view2.jpg view3.jpg view4.jpg

4. Outputs (saved to ./output_sfm/):
   - feature_matches_X_Y.png    : Feature matches between image pairs
   - epipolar_lines_X_Y.png     : Epipolar geometry visualization
   - reconstructed_3d.png        : 3D point cloud reconstruction
   - boundary_estimation.png     : Object boundary estimation (2D projection)
   - camera_positions.png        : Estimated camera positions in 3D
   - sfm_results.txt             : Numerical results (camera matrices, 3D points)

5. Theory:
   This script implements the SfM pipeline:
   a) Feature detection (SIFT/ORB) and matching across views
   b) Fundamental matrix estimation (8-point algorithm + RANSAC)
   c) Essential matrix computation from F and K
   d) Camera pose recovery (R, t) from E
   e) Triangulation of 3D points
   f) Boundary estimation of the planar object

Author: [Your Name]
Date:   [Date]
=============================================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os
import sys
from scipy.spatial import ConvexHull


def create_output_dir(dirname="output_sfm"):
    os.makedirs(dirname, exist_ok=True)
    return dirname


def load_images(image_paths):
    """Load images and convert to grayscale."""
    images = []
    gray_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Cannot load image: {path}")
            sys.exit(1)
        images.append(img)
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print(f"  Loaded: {path} ({img.shape[1]}x{img.shape[0]})")
    return images, gray_images


def get_camera_matrix(focal_length, image_shape):
    """
    Construct intrinsic camera matrix K.
    
    K = [[f, 0, cx],
         [0, f, cy],
         [0, 0,  1]]
    
    where (cx, cy) is the principal point (image center)
    and f is the focal length in pixels.
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def detect_and_match_features(gray1, gray2, method='SIFT'):
    """
    Detect features and compute matches between two images.
    
    Uses SIFT (Scale-Invariant Feature Transform) for robust feature
    detection and description, followed by FLANN-based matching with
    Lowe's ratio test to filter good matches.
    """
    if method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=2000)
        # FLANN parameters for SIFT (float descriptors)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        # FLANN parameters for ORB (binary descriptors)
        index_params = dict(algorithm=6, table_number=6,
                           key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return [], [], []

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_n in matches_knn:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches


def visualize_matches(img1, img2, pts1, pts2, good_matches, outdir, label):
    """Draw feature matches between two images."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    # Draw a subset of matches
    n_draw = min(50, len(pts1))
    indices = np.random.choice(len(pts1), n_draw, replace=False) if len(pts1) > n_draw else range(len(pts1))

    for i in indices:
        x1, y1 = int(pts1[i][0]), int(pts1[i][1])
        x2, y2 = int(pts2[i][0]) + w1, int(pts2[i][1])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(vis, (x1, y1), 4, color, -1)
        cv2.circle(vis, (x2, y2), 4, color, -1)
        cv2.line(vis, (x1, y1), (x2, y2), color, 1)

    save_path = os.path.join(outdir, f"feature_matches_{label}.png")
    cv2.imwrite(save_path, vis)
    print(f"  Feature matches saved: {save_path} ({len(pts1)} matches)")


def compute_fundamental_matrix(pts1, pts2):
    """
    Compute fundamental matrix using RANSAC.
    
    The fundamental matrix F encodes the epipolar geometry:
        x'^T F x = 0
    
    For corresponding points x <-> x' in two images.
    
    The 8-point algorithm solves:
        [x'x, x'y, x', y'x, y'y, y', x, y, 1] * f = 0
    where f is the vectorized F matrix.
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    inlier_mask = mask.ravel() == 1
    pts1_inlier = pts1[inlier_mask]
    pts2_inlier = pts2[inlier_mask]
    n_inliers = np.sum(inlier_mask)
    print(f"  Fundamental matrix: {n_inliers}/{len(pts1)} inliers")
    return F, pts1_inlier, pts2_inlier


def compute_essential_matrix(F, K):
    """
    Compute essential matrix from fundamental matrix and intrinsics.
    
    E = K'^T F K
    
    The essential matrix relates normalized image coordinates:
        x_hat'^T E x_hat = 0
    where x_hat = K^{-1} x
    """
    E = K.T @ F @ K
    # Enforce rank-2 constraint via SVD
    U, S, Vt = np.linalg.svd(E)
    S = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
    E = U @ np.diag(S) @ Vt
    return E


def recover_pose(E, pts1, pts2, K):
    """
    Recover camera rotation R and translation t from essential matrix.
    
    E = [t]_x R where [t]_x is the skew-symmetric matrix of t.
    
    SVD of E gives 4 possible (R,t) combinations. The correct one
    is chosen by checking that triangulated points are in front of
    both cameras (positive depth / cheirality check).
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    n_good = np.sum(mask > 0)
    print(f"  Pose recovery: {n_good} points with positive depth")
    return R, t, mask


def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """
    Triangulate 3D points from two views using DLT (Direct Linear Transform).
    
    Given projection matrices P1 = K[R1|t1] and P2 = K[R2|t2],
    and corresponding 2D points, solve for 3D point X such that:
        x1 = P1 * X
        x2 = P2 * X
    
    This is solved as a homogeneous linear system using SVD.
    """
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])

    pts1_h = pts1.T.reshape(2, -1).astype(np.float64)
    pts2_h = pts2.T.reshape(2, -1).astype(np.float64)

    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3] / points_4d[3:4]
    return points_3d.T


def filter_3d_points(points_3d, max_dist=50):
    """Remove outlier 3D points that are too far from the median."""
    median = np.median(points_3d, axis=0)
    dists = np.linalg.norm(points_3d - median, axis=1)
    mask = dists < max_dist
    return points_3d[mask]


def estimate_boundary(points_3d, outdir):
    """
    Estimate the object boundary from reconstructed 3D points.
    For a planar object, project to 2D and compute convex hull.
    """
    # Fit a plane to the 3D points using SVD
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid

    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Normal to the best-fit plane

    # Project onto the plane's 2D coordinate system
    # Choose two orthogonal basis vectors in the plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Project 3D points onto 2D plane coordinates
    pts_2d = np.column_stack([
        centered @ u,
        centered @ v
    ])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c='blue', s=15, alpha=0.6, label='Reconstructed Points')

    if len(pts_2d) >= 3:
        try:
            hull = ConvexHull(pts_2d)
            hull_pts = pts_2d[hull.vertices]
            hull_pts_closed = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_pts_closed[:, 0], hull_pts_closed[:, 1], 'r-', lw=2,
                    label='Estimated Boundary (Convex Hull)')
            ax.fill(hull_pts_closed[:, 0], hull_pts_closed[:, 1], alpha=0.15, color='red')
        except Exception as e:
            print(f"  [WARNING] Could not compute convex hull: {e}")

    ax.set_xlabel('U (plane coordinate)', fontsize=11)
    ax.set_ylabel('V (plane coordinate)', fontsize=11)
    ax.set_title('Object Boundary Estimation (Projected onto Best-Fit Plane)', fontsize=13)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(outdir, "boundary_estimation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Boundary estimation saved: {save_path}")

    return pts_2d


def visualize_3d_reconstruction(points_3d, cameras, outdir):
    """
    Visualize the reconstructed 3D points and camera positions.
    """
    fig = plt.figure(figsize=(14, 6))

    # --- 3D Point Cloud ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c='steelblue', s=5, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Reconstructed 3D Points ({len(points_3d)} pts)')

    # --- Camera Positions ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c='lightgray', s=3, alpha=0.3, label='3D Points')

    colors = ['red', 'green', 'blue', 'orange']
    for i, (R, t, label) in enumerate(cameras):
        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t.flatten()
        ax2.scatter(*C, c=colors[i % 4], s=100, marker='^', label=f'Cam {label}')
        # Draw camera direction
        direction = R.T @ np.array([0, 0, 1])
        ax2.quiver(C[0], C[1], C[2], direction[0], direction[1], direction[2],
                   color=colors[i % 4], length=2, arrow_length_ratio=0.3)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Camera Positions and Orientations')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(outdir, "reconstructed_3d.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D reconstruction saved: {save_path}")


def save_results(K, cameras, points_3d, outdir):
    """Save numerical results to a text file."""
    save_path = os.path.join(outdir, "sfm_results.txt")
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STRUCTURE FROM MOTION - NUMERICAL RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Camera Intrinsic Matrix K:\n")
        f.write(f"{K}\n\n")

        for R, t, label in cameras:
            f.write(f"--- Camera {label} ---\n")
            f.write(f"Rotation R:\n{R}\n")
            f.write(f"Translation t:\n{t.flatten()}\n")
            C = -R.T @ t.flatten()
            f.write(f"Camera Center (world): [{C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f}]\n\n")

        f.write(f"Number of reconstructed 3D points: {len(points_3d)}\n")
        f.write(f"Point cloud bounding box:\n")
        f.write(f"  X: [{points_3d[:,0].min():.3f}, {points_3d[:,0].max():.3f}]\n")
        f.write(f"  Y: [{points_3d[:,1].min():.3f}, {points_3d[:,1].max():.3f}]\n")
        f.write(f"  Z: [{points_3d[:,2].min():.3f}, {points_3d[:,2].max():.3f}]\n")

    print(f"  Results saved: {save_path}")


def draw_epipolar_lines(img1, img2, pts1, pts2, F, outdir, label):
    """Visualize epipolar lines for a pair of images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Draw lines on image 2 corresponding to points in image 1
    n_lines = min(15, len(pts1))
    idx = np.random.choice(len(pts1), n_lines, replace=False) if len(pts1) > n_lines else range(len(pts1))

    pts1_sel = pts1[idx]
    pts2_sel = pts2[idx]

    # Compute epilines in image 2 for points in image 1
    lines2 = cv2.computeCorrespondEpilines(pts1_sel.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2_sel.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    h, w = img1.shape[:2]

    ax1.imshow(img1_rgb)
    ax2.imshow(img2_rgb)

    for i in range(n_lines):
        color = plt.cm.hsv(i / n_lines)

        # Epiline in image 1
        a, b, c = lines1[i]
        x0, x1_pt = 0, w
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1_pt = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        ax1.plot([x0, x1_pt], [y0, y1_pt], color=color, linewidth=0.8, alpha=0.7)
        ax1.plot(pts1_sel[i, 0], pts1_sel[i, 1], 'o', color=color, markersize=5)

        # Epiline in image 2
        a, b, c = lines2[i]
        y0 = int(-c / b) if abs(b) > 1e-6 else 0
        y1_pt = int(-(c + a * w) / b) if abs(b) > 1e-6 else h
        ax2.plot([0, w], [y0, y1_pt], color=color, linewidth=0.8, alpha=0.7)
        ax2.plot(pts2_sel[i, 0], pts2_sel[i, 1], 'o', color=color, markersize=5)

    ax1.set_title(f'Image 1 - Epipolar Lines ({label})')
    ax2.set_title(f'Image 2 - Epipolar Lines ({label})')
    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()
    save_path = os.path.join(outdir, f"epipolar_lines_{label}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Epipolar lines saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Structure from Motion - 4 Viewpoints"
    )
    parser.add_argument('--images', nargs=4, required=True,
                        help='Paths to 4 images from different viewpoints')
    parser.add_argument('--focal_length', type=float, default=1000,
                        help='Estimated focal length in pixels (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='output_sfm',
                        help='Output directory')
    args = parser.parse_args()

    outdir = create_output_dir(args.output_dir)

    print("=" * 60)
    print("  STRUCTURE FROM MOTION - 4 VIEWPOINTS")
    print("=" * 60)

    # Load images
    print("\n[1] Loading images...")
    images, grays = load_images(args.images)

    # Camera intrinsic matrix
    K = get_camera_matrix(args.focal_length, images[0].shape)
    print(f"\n[2] Camera matrix K (f={args.focal_length}):")
    print(K)

    # Reference camera (view 0): identity pose
    R_ref = np.eye(3)
    t_ref = np.zeros((3, 1))
    cameras = [(R_ref, t_ref, "View0")]

    all_3d_points = []

    # Process each pair (view 0 with views 1, 2, 3)
    for i in range(1, 4):
        print(f"\n[3.{i}] Processing View 0 <-> View {i}...")

        # Feature detection and matching
        pts1, pts2, matches = detect_and_match_features(grays[0], grays[i])
        if len(pts1) < 8:
            print(f"  [WARNING] Not enough matches ({len(pts1)}), skipping pair")
            continue

        visualize_matches(images[0], images[i], pts1, pts2, matches, outdir, f"0_{i}")

        # Fundamental matrix
        F, pts1_in, pts2_in = compute_fundamental_matrix(pts1, pts2)
        if F is None:
            print(f"  [WARNING] Could not estimate fundamental matrix, skipping")
            continue

        # Epipolar lines
        draw_epipolar_lines(images[0], images[i], pts1_in, pts2_in, F, outdir, f"0_{i}")

        # Essential matrix
        E = compute_essential_matrix(F, K)
        print(f"  Essential matrix computed")

        # Recover pose
        R, t, pose_mask = recover_pose(E, pts1_in, pts2_in, K)
        cameras.append((R, t, f"View{i}"))

        # Triangulate
        points_3d = triangulate_points(K, R_ref, t_ref, R, t, pts1_in, pts2_in)
        points_3d = filter_3d_points(points_3d)
        all_3d_points.append(points_3d)
        print(f"  Triangulated {len(points_3d)} 3D points")

    if len(all_3d_points) == 0:
        print("\n[ERROR] No 3D points reconstructed. Check your images.")
        sys.exit(1)

    # Combine all 3D points
    all_points = np.vstack(all_3d_points)
    all_points = filter_3d_points(all_points)
    print(f"\n[4] Total reconstructed points: {len(all_points)}")

    # Visualize
    print("\n[5] Generating visualizations...")
    visualize_3d_reconstruction(all_points, cameras, outdir)
    estimate_boundary(all_points, outdir)
    save_results(K, cameras, all_points, outdir)

    print("\n" + "=" * 60)
    print("  ALL OUTPUTS SAVED TO:", outdir)
    print("=" * 60)


if __name__ == "__main__":
    main()
