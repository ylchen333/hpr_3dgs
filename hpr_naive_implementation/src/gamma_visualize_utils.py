
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d

import argparse
import pickle
from PIL import Image, ImageDraw
import imageio
import json
from simple_interactive_viewer import visualize_pointcloud_interactive
import visualize
from open3d_visualize import visualize_hpr_result, debug_indexing_with_open3d
# from HPR import HPR
from hpr_naive import hpr, HPR
from scipy.spatial import cKDTree
import pandas as pd
import open3d as o3d
from fast_dipole_sums.point_cloud_util import estimate_areas
from sklearn.neighbors import NearestNeighbors

import gpytoolbox as gpy
dir(gpy)

import numpy as np
import os
import tempfile
from scipy.spatial import cKDTree


DEBUG = False
MESH_PATH = "../data/bunny.obj"

# Set up a benchmark where you can take a mesh, sample random point clouds of different densities (controlled using the number of points) from that mesh, 
# and use the mesh and ray-mesh intersections to compute groundtruth visibility values for each point. 
# Then once this groundtruth is available, use HPR with different param values and inversion kernels to assess accuracy 
# (e.g., number of false negatives and false positives).

# ------------------------ utils --------------------------------------------------------------------------




def match_indices(original_points, subset_points, tolerance=1e-3):
    tree = cKDTree(original_points)
    dists, idx = tree.query(subset_points, k=1)
    valid = dists < tolerance
    if np.any(~valid):
        print(f"[match_indices] Warning: {np.sum(~valid)} points exceed tolerance {tolerance}")
    return idx[valid], valid


def save_points_to_npz(points, colors=None, out_path=None):
    # Ensure points are numpy float32
    points = np.asarray(points, dtype=np.float32)

    # Default gray color if none provided
    if colors is None:
        colors = np.ones_like(points) * 0.5

    # Normalize color range if not in [0,1]
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Create temporary file if not specified
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".npz", prefix="gpy_pts_")
        os.close(fd)

    # Optionally wrap as gpytoolbox point cloud (for validation/export)
    # This ensures your data are in the same convention gpytoolbox expects
    try:
        pc = gpy.PointCloud(points)
        pc.colors = colors
        # You could also export via gpytoolbox if it supports .npz
        # pc.write(out_path)  # (if available)
    except Exception as e:
        # fallback: write directly
        np.savez_compressed(out_path, verts=points, colors=colors)

    return out_path

# ------------------------ mesh sampling --------------------------------------------------------------------------

def sample_points_from_mesh(mesh_path, n_points=10000, with_normals=True):
    V, F = gpy.read_mesh(mesh_path)
    
    sample_result = gpy.random_points_on_mesh(V, F, n_points)
    if isinstance(sample_result, tuple) and len(sample_result) == 2:
        points, face_indices = sample_result
    else:
        points = sample_result
        face_indices = None

    # Compute normals if faces are available
    normals = None
    if face_indices is not None:
        normals = gpy.face_normals(V, F)[face_indices]


    return points, normals

def compute_visibility(mesh_path, points, camera_origin):
    """
    Use gpytool's ray-mesh intersection / visibility module.
    Return a boolean mask of visible (True) vs occluded (False) for each point.
    """
    V, F = gpy.read_mesh(mesh_path)

    # Build rays from camera to each sampled point
    directions = points - camera_origin[None, :]
    distances = np.linalg.norm(directions, axis=1)
    directions /= (distances[:, None] + 1e-8)

    # Repeat the single camera origin for all rays
    origins_array = np.tile(camera_origin, (points.shape[0], 1))

    # Call gpytoolbox.ray_mesh_intersect according to new API
    ts, ids, lambdas= gpy.ray_mesh_intersect(origins_array, directions, V, F)


    ts = np.asarray(ts, dtype=float)
    ids = np.asarray(ids, dtype=int)

    # No hit if id == -1
    no_hit_mask = ids == -1

    # Occluded if the ray hits something before reaching the point
    no_hit_mask = (ids == -1) | (~np.isfinite(ts))
    occluded_mask = (~no_hit_mask) & (ts < distances - 1e-5)
    visibility = ~occluded_mask

    # Visible if not occluded
    # visibility = ~occluded_mask

    return visibility

def evaluate_hpr_with(mesh_path, camera_origin, gamma_values, kernels, num_points=10000):
    V, F = gpy.read_mesh(mesh_path)
    points, normals = sample_points_from_mesh(mesh_path, num_points)
    visibility_gt = compute_visibility(mesh_path, points, camera_origin)

    results = []
    all_hull_pts = {} 
    for kernel in kernels:
        for gamma in gamma_values:
            # Run HPR (returns points in camera-centered coords)
            pts_t = torch.tensor(points.T, dtype=torch.float32)
            cam_t = torch.tensor(camera_origin, dtype=torch.float32)

            visible_pts_t, visible_idx = HPR(pts_t, cam_t, gamma, kernel_type=kernel)

            if len(visible_idx) == 0:
                print(f"[Warn] No visible points for kernel={kernel}, γ={gamma:.4f}")
                continue

            # Convert back to world coordinates
            # visible_pts_t_world = visible_pts_t + cam_t.view(1, 3, 1)
            hull_pts = visible_pts_t.squeeze(0).T.cpu().numpy()
            all_hull_pts[(kernel, gamma)] = hull_pts 

            # Map hull_pts back to indices in `points`
            scene_scale = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1))
            tol = 0.01 * scene_scale  
            visible_idx, _ = match_indices(points, hull_pts, tolerance=tol)
            if len(visible_idx) == 0:
                print(f"[Warn] No matches for kernel={kernel}, γ={gamma:.4f}")
                continue


            # Compute confusion metrics
            tp = np.sum(visibility_gt[visible_idx])
            fp = len(visible_idx) - tp
            fn = np.sum(visibility_gt) - tp

            results.append({
                'kernel': kernel,
                'gamma': float(gamma),
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            })

    return results, points, visibility_gt, all_hull_pts


def visualize_gamma(mesh_path, out_folder, image_size, gamma_min, gamma_max, step, kernel="spherical_flip"):

    os.makedirs(out_folder, exist_ok=True)
    mesh = gpy.read_mesh(mesh_path)
    camera_origin = np.array([0,0,3.0])

    for gamma in tqdm(np.arange(gamma_min, gamma_max + step, step)):
        pts, _ = sample_points_from_mesh(mesh, num_points=10000)
        hull_pts = hpr(
            point_cloud_path=save_points_to_npz(pts),
            inversion_func=kernel,
            gamma=gamma,
            camera_coordinates=camera_origin
        )
        out_path = os.path.join(out_folder, f"{kernel}_{gamma:.2f}.png")
        visualize_hpr_result(hull_pts, out_path, image_size=image_size)


def gamma_sweep(kernel, R):
    """Generate gamma ranges per kernel with extended coverage."""
    if kernel in ("spherical_flip", "mirror"):
        # Sweep γ/R logarithmically over a wider range
        grid = np.logspace(np.log10(0.2), np.log10(5.0), num=51)
        gammas = grid * R
        xvals = np.log10(gammas / R)
        xlabel = "log10(γ / R)"
    elif kernel == "exp_inversion":
        gammas = np.linspace(-10.0, -0.1, 51)
        xvals = gammas
        xlabel = "γ"
    elif kernel == "exp_natural":
        gammas = np.linspace(0.05, 6.0, 51)
        xvals = gammas
        xlabel = "γ"
    else:
        raise ValueError(kernel)
    return gammas, xvals, xlabel


def benchmark_gamma_and_plot(mesh_path, out_folder, gamma_min, gamma_max, step,
                             kernels=("spherical_flip", "mirror", "exp_inversion", "exp_natural"),
                             num_points=10000):
    """
    Run HPR benchmarks across gamma range and kernels.
    Compute FP/FN, save numeric results to CSV, and plot curves.
    Also generates a combined overlay plot of total errors.
    """
    os.makedirs(out_folder, exist_ok=True)
    camera_origin = np.array([0, 0, 3.0])

    # --- Sample points & compute GT visibility ---
    points, _ = sample_points_from_mesh(mesh_path, n_points=num_points)
    visibility_gt = compute_visibility(mesh_path, points, camera_origin)
    print(f"[Debug] Ground truth visible: {np.sum(visibility_gt)} / {len(visibility_gt)}")

    all_results = []

    # --- Loop over kernels ---
    for kernel in kernels:
        print(f"\n[Kernel: {kernel}]")
        kernel_results = []
        R = np.max(np.linalg.norm(points - camera_origin, axis=1))
        gamma_values, xvals, xlabel = gamma_sweep(kernel, R)

        for gamma in tqdm(gamma_values, desc=f"{kernel} sweep"):
            pts_t = torch.tensor(points.T, dtype=torch.float32)
            cam_t = torch.tensor(camera_origin, dtype=torch.float32)

            visible_pts_t, visible_idx = HPR(pts_t, cam_t, gamma, kernel_type=kernel)
            hull_pts = visible_pts_t.squeeze(0).T.cpu().numpy()

            pred_idx, _ = match_indices(points, hull_pts, tolerance=1e-3)
            pred_idx = np.asarray(pred_idx).ravel()

            visibility_pred = np.zeros(points.shape[0], dtype=bool)
            visibility_pred[pred_idx] = True

            fp = np.sum((visibility_pred == 1) & (visibility_gt == 0))
            fn = np.sum((visibility_pred == 0) & (visibility_gt == 1))
            tp = np.sum((visibility_pred == 1) & (visibility_gt == 1))
            tn = np.sum((visibility_pred == 0) & (visibility_gt == 0))

            kernel_results.append({
                "kernel": kernel,
                "gamma": float(gamma),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "FP_rate": fp / (fp + tn + 1e-8),
                "FN_rate": fn / (fn + tp + 1e-8),
                "TOTAL_rate": (fp + fn) / (tp + fn + 1e-8)
            })

        # --- Save per-kernel CSV ---
        df = pd.DataFrame(kernel_results)
        csv_path = os.path.join(out_folder, f"benchmark_{kernel}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Saved CSV] {csv_path}")

        # --- Plot individual FP/FN/TOTAL rates ---
        df["FP_rate"] *= 100
        df["FN_rate"] *= 100
        df["TOTAL_rate"] *= 100

        plt.figure(figsize=(8, 5))
        plt.ylim(0, 120)
        if kernel in ("spherical_flip", "mirror"):
            plt.xscale("log")
        plt.plot(df["gamma"], df["TOTAL_rate"], color="royalblue", marker="o", markersize=3, linewidth=2, label="All falses")
        plt.plot(df["gamma"], df["FP_rate"], color="green", marker="o", markersize=3, linewidth=1.5, label="False positive")
        plt.plot(df["gamma"], df["FN_rate"], color="red", marker="o", markersize=3, linewidth=1.5, label="False negative")
        plt.xlabel(xlabel)
        plt.ylabel("Error rate (%)")
        plt.title(f"HPR Benchmark — {kernel}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        img_path = os.path.join(out_folder, f"benchmark_{kernel}.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"[Saved plot] {img_path}")

        all_results.extend(kernel_results)

    # --- Combined overlay plot ---
    combined_df = pd.DataFrame(all_results)
    combined_csv = os.path.join(out_folder, "benchmark_all_kernels.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n[All kernels combined results saved] {combined_csv}")

    plt.figure(figsize=(9, 6))
    for kernel in kernels:
        subset = combined_df[combined_df["kernel"] == kernel]
        plt.plot(subset["gamma"], subset["TOTAL_rate"] * 100, label=f"{kernel}", linewidth=2)
        min_row = df.loc[df["TOTAL_rate"].idxmin()]
        print(f"[Kernel {kernel}] Min total error {100*min_row['TOTAL_rate']:.2f}% at γ={min_row['gamma']:.4f}")

    plt.xlabel("γ (gamma)")
    plt.ylabel("Total error rate (%)")
    plt.title("HPR Total Error vs γ — All Kernels")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    overlay_path = os.path.join(out_folder, "benchmark_overlay_total.png")
    plt.savefig(overlay_path, dpi=300)
    plt.show()
    print(f"[Saved overlay plot] {overlay_path}")

    return combined_df


def plot_all_kernels_overlay(results_df):
    plt.figure(figsize=(8,6))
    for kernel in results_df["kernel"].unique():
        subset = results_df[results_df["kernel"] == kernel]
        plt.plot(subset["gamma"], subset["FP_rate"], label=f"{kernel} FP", linestyle="--")
        plt.plot(subset["gamma"], subset["FN_rate"], label=f"{kernel} FN")
    plt.xlabel("γ")
    plt.ylabel("Error rate")
    plt.title("FP/FN Comparison Across Kernels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.close()
    return


# ----------------------------------------------------------------------------------------------
def visualize_visibility_comparison(points, visibility_gt, hpr_visible_pts, camera_origin=None):
    gt_idx = np.where(visibility_gt)[0]
    hpr_idx, _ = match_indices(points, hpr_visible_pts, tolerance=5e-3)

    # Overlap (true positives)
    overlap = np.intersect1d(gt_idx, hpr_idx)
    # False negatives (GT visible but not HPR)
    fn = np.setdiff1d(gt_idx, hpr_idx)
    # False positives (HPR visible but not GT)
    fp = np.setdiff1d(hpr_idx, gt_idx)

    pc_all = o3d.geometry.PointCloud()
    pc_all.points = o3d.utility.Vector3dVector(points)
    pc_all.paint_uniform_color([0.4, 0.4, 0.4])

    pc_tp = o3d.geometry.PointCloud()
    pc_tp.points = o3d.utility.Vector3dVector(points[overlap])
    pc_tp.paint_uniform_color([1.0, 1.0, 0.0])  # yellow: correct visible

    pc_fn = o3d.geometry.PointCloud()
    pc_fn.points = o3d.utility.Vector3dVector(points[fn])
    pc_fn.paint_uniform_color([0.0, 1.0, 0.0])  # green: missed by HPR

    pc_fp = o3d.geometry.PointCloud()
    pc_fp.points = o3d.utility.Vector3dVector(points[fp])
    pc_fp.paint_uniform_color([1.0, 0.0, 0.0])  # red: extra by HPR

    geometries = [pc_all, pc_tp, pc_fn, pc_fp]

    # --- Camera arrow ---
    if camera_origin is not None:
        # Compute direction toward object (mean of points)
        center = np.mean(points, axis=0)
        camera_dir = center - camera_origin
        camera_dir /= np.linalg.norm(camera_dir) + 1e-8

        # Create arrow pointing along +Z initially, then rotate to camera_dir
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.04,
            cylinder_height=0.4,
            cone_height=0.1
        )
        arrow.paint_uniform_color([0, 0, 1])  # blue arrow

        # --- Rotation: align +Z → camera_dir ---
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, camera_dir)
        c = np.dot(z_axis, camera_dir)
        if np.linalg.norm(v) > 1e-8:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
            arrow.rotate(R, center=(0, 0, 0))

        # --- Translate arrow to camera origin ---
        arrow.translate(camera_origin)
        geometries.append(arrow)

    # --- Coordinate frames (world + camera) ---
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=camera_origin.tolist()
    )

    geometries.extend([axis_frame, camera_frame])

    # --- Visualize everything ---
    o3d.visualization.draw_geometries(geometries, window_name="HPR vs GT Visibility")



# ------------------------- HPR benchmark --------------------------------------------------------------------------
# For a given mesh and sampled points, run HPR with different gamma values and compute FP/FN against groundtruth visibility.
# use yannis's hpr implementation
def HPR_Param(points, view, param, use_linear=False):
    """
    Hidden Point Removal (HPR) operator.
    Reimplementation of Yannis's Matlab implementation of the HPR operator in Python.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        view (np.ndarray): 3-element camera/viewpoint position.
        param (float): Inversion parameter (0 < param < 1).
        use_linear (bool): If True, use linear inversion; otherwise exponential inversion.

    Returns:
        visible_idx (np.ndarray): Indices of visible points.
    """

    # --- Input validation ---
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T

    # Convert to NumPy for convex hull
    points = points.detach().cpu().numpy()
    view = view.detach().cpu().numpy().reshape(1, 3)

    num_pts, dim = points.shape

    # --- Move points so camera is at the origin ---
    shifted = points - view  # Nx3
    normp = np.linalg.norm(shifted, axis=1)
    directions = shifted / normp[:, None]  # normalized direction vectors

    # --- Inversion step ---
    if use_linear:
        # Linear inversion: R = (1/param)*max(normp)
        R = (1.0 / param) * np.max(normp)
        P = (R - normp)[:, None] * directions
    else:
        # Exponential inversion: points scaled by ||p||^(-param)
        P = (normp ** (-param))[:, None] * directions

    # --- Form augmented set with origin ---
    augmented_points = np.vstack([P, np.zeros((1, dim))])

    # --- Compute convex hull ---
    try:
        hull = ConvexHull(augmented_points)
    except Exception as e:
        raise RuntimeError(f"HPR convex hull failed: {e}")

    visible_idx = np.unique(hull.vertices)
    visible_idx = visible_idx[visible_idx < num_pts]  # remove the origin index
    visible_pts = torch.tensor(points[visible_idx].T, dtype=torch.float32)

    return visible_pts, visible_idx


def benchmark_HPR_visibility(points, camera_origin, kernel, gamma_values, R_pred, R_opt, mesh_path, outdir):
    """
    Run HPR benchmark on given points and mesh for a given kernel.
    Produces total, FP, FN plots and zoomed views.
    """
    visibility_gt = compute_visibility(mesh_path, points, camera_origin)
    results = []

    for gamma in gamma_values:
        pts_t = torch.tensor(points.T, dtype=torch.float32)
        cam_t = torch.tensor(camera_origin, dtype=torch.float32)
        useLinear = False
        if kernel in ("spherical_flip", "mirror"): useLinear = True
        visible_pts_t, visible_idx = HPR_Param(pts_t, cam_t, gamma, use_linear=useLinear)
        hull_pts = visible_pts_t.squeeze(0).T.cpu().numpy()

        pred_idx, _ = match_indices(points, hull_pts, tolerance=1e-3)
        pred_idx = np.asarray(pred_idx).ravel()
        visibility_pred = np.zeros(points.shape[0], dtype=bool)
        visibility_pred[pred_idx] = True

        fp = np.sum((visibility_pred == 1) & (visibility_gt == 0))
        fn = np.sum((visibility_pred == 0) & (visibility_gt == 1))
        total_err = (fp + fn) / len(points) * 100
        results.append((gamma, total_err, fp / len(points) * 100, fn / len(points) * 100))

    
      # --- Save CSV and plot ---
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results, columns=["gamma", "TOTAL_rate", "FP_rate", "FN_rate"])
    csv_path = os.path.join(outdir, f"benchmark_{kernel}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved CSV] {csv_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # Use log scale for gamma axis since sampling is log-spaced
    ax.set_xscale("log")

    ax.plot(df["gamma"], df["TOTAL_rate"], color="royalblue", marker="o", markersize=3, linewidth=2, label="Total error")
    ax.plot(df["gamma"], df["FP_rate"], color="green", marker="o", markersize=3, linewidth=1.5, label="False positive")
    ax.plot(df["gamma"], df["FN_rate"], color="red", marker="o", markersize=3, linewidth=1.5, label="False negative")

    # Highlight minimum total error
    min_row = df.loc[df["TOTAL_rate"].idxmin()]
    ax.scatter(min_row["gamma"], min_row["TOTAL_rate"], color="gold", s=50, edgecolor="black", zorder=5)

    # Choose offset direction dynamically based on location
    x_offset = 25 if min_row["gamma"] < 0.1 else -100

    ax.annotate(
        f"Min error = {min_row['TOTAL_rate']:.2f}%\nγ = {min_row['gamma']:.2e}",
        xy=(min_row["gamma"], min_row["TOTAL_rate"]),
        xycoords="data",
        xytext=(x_offset, 25),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_title(f"HPR Benchmark — {kernel}")
    ax.set_xlabel("γ (log-spaced, 1e-7 → 1)")
    ax.set_ylabel("Error rate (%)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    img_path = os.path.join(outdir, f"benchmark_{kernel}.png")
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"[Saved plot] {img_path}")
    # df = pd.DataFrame(results, columns=["gamma", "TOTAL_rate", "FP_rate", "FN_rate"])
    # df.to_csv(os.path.join(outdir, f"benchmark_{kernel}.csv"), index=False)

    # # Plot
    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.set_xscale("linear")
    # ax.plot(df["gamma"], df["TOTAL_rate"], color="royalblue", marker="o", markersize=3, linewidth=2, label="Total error")
    # ax.plot(df["gamma"], df["FP_rate"], color="green", marker="o", markersize=3, linewidth=1.5, label="False positive")
    # ax.plot(df["gamma"], df["FN_rate"], color="red", marker="o", markersize=3, linewidth=1.5, label="False negative")
    # # ax.axvline(R_pred, color="orange", linestyle="--", linewidth=2, label=f"Predicted R ≈ {R_pred:.2f}")
    # # ax.axvline(R_opt, color="magenta", linestyle="--", linewidth=2, label=f"Optimal R ≈ {R_opt:.2f}")

    # # Annotate minimum
    # min_row = df.loc[df["TOTAL_rate"].idxmin()]
    # ax.scatter(min_row["gamma"], min_row["TOTAL_rate"], color="gold", s=40, edgecolor="black", zorder=5)
    # ax.annotate(f"Min error = {min_row['TOTAL_rate']:.2f}%\nγ = {min_row['gamma']:.4f}",
    #             xy=(min_row["gamma"], min_row["TOTAL_rate"]),
    #             xycoords='data', xytext=(25,25), textcoords='offset points',
    #             arrowprops=dict(arrowstyle="->", color="black"), fontsize=9,
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # ax.set_title(f"HPR Benchmark — {kernel}")
    # ax.set_xlabel("param (log-spaced gamma 1e-7 to 1)")
    # ax.set_ylabel("Error rate (%)")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(outdir, f"benchmark_{kernel}.png"))
    # plt.close(fig)

