
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull

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
    Use gpytool’s ray-mesh intersection / visibility module.
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
    mesh = gpy.read_mesh(mesh_path)
    points, normals = sample_points_from_mesh(mesh, num_points)

    visibility_gt = compute_visibility(mesh, points, camera_origin)

    results = []
    # To call your HPR, you may need to write the sampled points to a temporary .npz file, or allow passing in-memory arrays
    for kernel in kernels:
        for gamma in gamma_values:
            hull_pts = HPR(
                point_cloud_path=save_points_to_npz(points),
                inversion_func=kernel,
                gamma=gamma,
                camera_coordinates=camera_origin
            )

            # map hull_pts back to indices in `points`
            visible_idx = match_indices(points, hull_pts)
            tp = np.sum(visibility_gt[visible_idx])
            fp = len(visible_idx) - tp
            fn = np.sum(visibility_gt) - tp

            results.append({
                'kernel': kernel,
                'gamma': gamma,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
            })

    return results

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
    if kernel in ("spherical_flip", "mirror"):
        # γ behaves like a radius -> sweep as multiples of R
        grid = np.logspace(np.log10(0.5), np.log10(2.0), num=21)
        gammas = grid * R
        xvals = np.log10(gammas / R)  # paper’s log(R̂) with R̂ = γ/R
        xlabel = "log10(R̂)"
    elif kernel == "exp_inversion":
        gammas = np.linspace(-4.0, -0.5, 21)
        xvals = gammas                 # keep linear on γ (or use log10(-γ) if you prefer)
        xlabel = "γ"
    elif kernel == "exp_natural":
        gammas = np.linspace(0.1, 3.0, 21)
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
    """
    os.makedirs(out_folder, exist_ok=True)
    camera_origin = np.array([0, 0, 3.0])

    # Sample points once and compute ground truth visibility
    points, _ = sample_points_from_mesh(mesh_path, n_points=num_points)
    visibility_gt = compute_visibility(mesh_path, points, camera_origin)
    print("[Debug] Ground truth visible:", np.sum(visibility_gt), "/", len(visibility_gt))

    all_results = []

    # Sweep over kernels and gamma
    for kernel in kernels:
        print(f"\n[Kernel: {kernel}]")
        kernel_results = []
        R = np.max(np.linalg.norm(points - camera_origin, axis=1))
        R = np.max(np.linalg.norm(points - camera_origin, axis=1)) 
        gamma_values = np.linspace(-1, 1.0, num=21) * R
        if kernel in ("spherical_flip", "mirror"):
            gamma_values = np.logspace(np.log10(0.5), np.log10(2.0), num=21) * R
        elif kernel == "exp_inversion":
            gamma_values = np.linspace(-4.0, -0.5, 21)
        elif kernel == "exp_natural":
            gamma_values = np.linspace(0.1, 3.0, 21)
        else:
            raise ValueError(kernel)


        # instead of [-1, 1]
        for gamma in gamma_values: # tqdm(np.arange(gamma_min, gamma_max + step, step))
            pts_t = torch.tensor(points.T, dtype=torch.float32)
            cam_t = torch.tensor(camera_origin, dtype=torch.float32)

            visible_pts_t, visible_idx = HPR(pts_t, cam_t, gamma, kernel_type=kernel)
            hull_pts = visible_pts_t.squeeze(0).T.cpu().numpy()

            pred_idx = match_indices(points, hull_pts, tolerance=1e-3)
            pred_idx = np.asarray(pred_idx).ravel()   # ✅ ensure it's 1-D

            visibility_pred = np.zeros(points.shape[0], dtype=bool)
            visibility_pred[pred_idx] = True


            fp = np.sum((visibility_pred == 1) & (visibility_gt == 0))
            fn = np.sum((visibility_pred == 0) & (visibility_gt == 1))
            tp = np.sum((visibility_pred == 1) & (visibility_gt == 1))
            tn = np.sum((visibility_pred == 0) & (visibility_gt == 0))
            N = points.shape[0]

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

        # Save per-kernel results
        df = pd.DataFrame(kernel_results)
        csv_path = os.path.join(out_folder, f"benchmark_{kernel}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")

        # Plot FP/FN rate vs γ
        gammas, xvals, xlabel = gamma_sweep(kernel, R)
        df["FP_rate"]    = 100 * df["FP_rate"]
        df["FN_rate"]    = 100 * df["FN_rate"]
        df["TOTAL_rate"] = 100 * df["TOTAL_rate"]

        if kernel in ("spherical_flip", "mirror"):
            R_hat = df["gamma"] / R
            xvals = np.log10(R_hat)
            xlabel = "log₁₀(γ / R)"

        plt.figure(figsize=(7, 5))
        plt.ylim(0, 120)
        
        plt.plot(df["gamma"], df["TOTAL_rate"], color="royalblue", marker="o",
                linewidth=2.5, label="All falses")
        plt.plot(df["gamma"], df["FP_rate"], color="green", marker="o",
                linewidth=1.8, label="False positive")
        plt.plot(df["gamma"], df["FN_rate"], color="red", marker="o",
         linewidth=1.8, label="False negative")
        plt.xlabel("γ (gamma)")
        plt.ylabel("Error rate (percentage)")
        plt.title(f"HPR Benchmark — {kernel}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        img_path = os.path.join(out_folder, f"benchmark_{kernel}.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"[Saved plot] {img_path}")

        all_results.extend(kernel_results)

    # Save combined results across all kernels
    combined_df = pd.DataFrame(all_results)
    combined_csv = os.path.join(out_folder, "benchmark_all_kernels.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n[All kernels combined results saved] {combined_csv}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default="../data/bunny.obj")
    parser.add_argument("--output_folder", type=str, default="./data/gamma_vis_gpy/")
    parser.add_argument("--gamma_min", type=float, default=-1.0)
    parser.add_argument("--gamma_max", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--kernel", type=str, default="spherical_flip")
    args = parser.parse_args()

    # visualize_gamma(args.mesh_path, args.output_folder, 256, args.gamma_min, args.gamma_max, args.step, kernel=args.kernel)

    import trimesh

    mesh = trimesh.load("../data/bunny.obj", force='mesh')

    # If faces are not triangles, triangulate using built-in utility
    if mesh.faces.shape[1] != 3:
        print("[Info] Triangulating non-triangular mesh...")
        # Use Trimesh's "convex_hull" trick to force triangulation if needed
        # mesh = mesh.convex_hull  # creates a triangulated hull
        # Alternatively, use subdivision to break quads into tris
        mesh = mesh.subdivide_to_size(max_edge=0.01)

    # Verify
    print(f"[Info] Mesh now has {mesh.faces.shape[0]} triangular faces.")

    # Save triangulated copy
    mesh.export("../data/bunny_tri.obj")
    print("[Info] Saved triangulated mesh to ../data/bunny_tri.obj")

    results_df = benchmark_gamma_and_plot(
        mesh_path="../data/bunny_tri.obj",
        out_folder="../data/benchmark_results/",
        gamma_min=-1.0,
        gamma_max=1.0,
        step=0.1,
        num_points=10000
    )
    # plot_all_kernels_overlay(results_df)
