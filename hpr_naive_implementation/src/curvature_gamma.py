
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
import argparse
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

from gamma_visualize_utils import sample_points_from_mesh, compute_visibility, match_indices, visualize_visibility_comparison, evaluate_hpr_with



def R_from_k_r(k, r, safety=1.15):
    """Return a per-point lower bound on R from Lemma 4.3 (β = π/2)."""
    k = np.maximum(k, 1e-9)        # avoid div/0
    return 0.5 * k * (r**2) * safety

def choose_global_R(points, camera_origin, k, strategy="p95", safety=1.15):
    """
    k: curvature magnitude per point (your proxy or estimator)
    strategy: "max" for strict bound, or "p95"/"p99" to ignore outliers
    """
    r = np.linalg.norm(points - camera_origin[None, :], axis=1)
    R_i = R_from_k_r(k, r, safety=safety)

    if strategy == "max":
        R = np.max(R_i)
    elif strategy == "p99":
        R = np.percentile(R_i, 99.0)
    else:   # "p95" default
        R = np.percentile(R_i, 95.0)
    return float(R), R_i
    
def refine_R_via_visibility(points, camera_origin, kernel, step_factor=0.9, iterations=10, samples=40):
    """
    Refine R (gamma) by minimizing the overlap between visibility from two opposite viewpoints.
    """
    areas = estimate_areas(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    normals = np.asarray(pcd.normals)

    k = estimate_curvature(points, normals, areas, n_neighbors=30)
    R_init, _ = choose_global_R(points, camera_origin, k, strategy="p95", safety=1.15)
    print(f"[Refine R] Initial R from curvature: {R_init:.4f}")

    # Define opposite camera position
    cam_center = np.mean(points, axis=0)
    cam_opposite = cam_center - (camera_origin - cam_center)

    best_R = R_init
    best_score = -np.inf

    pts_t = torch.tensor(points.T, dtype=torch.float32)
    cam_t = torch.tensor(camera_origin, dtype=torch.float32)
    cam_t_opp = torch.tensor(cam_opposite, dtype=torch.float32)
    
    # --- Logarithmic search around the bound ---
    R_min = R_init * 1e-2
    R_max = R_init * 2.0
    R_values = np.logspace(np.log10(R_min), np.log10(R_max), samples)

    best_R, best_score = R_init, -np.inf
    scores = []

    for R in R_values:
        vis_front, idx_front = HPR(pts_t, cam_t, R, kernel_type=kernel)
        vis_back,  idx_back  = HPR(pts_t, cam_t_opp, R, kernel_type=kernel)
        score = len(set(idx_front.tolist()) ^ set(idx_back.tolist()))
        scores.append(score)

        if score > best_score:
            best_score = score
            best_R = R

    return best_R


def estimate_curvature(points, normals, areas, n_neighbors=30):
    """
    Estimate mean curvature magnitude using local normal variation,
    normalized by the estimated surface area per point.

    Args:
        points: (N,3) array of point coordinates
        normals: (N,3) array of estimated normals
        areas: (N,1) array of estimated per-point areas (e.g. from Voronoi)
        n_neighbors: number of neighbors for curvature estimation
    Returns:
        curvatures: (N,) array of area-weighted curvature magnitudes
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)

    curvatures = np.zeros(len(points))

    for i in range(len(points)):
        ni = normals[i]
        neighbor_normals = normals[indices[i, 1:]]
        neighbor_points = points[indices[i, 1:]]
        di = np.mean(distances[i, 1:])

        # --- Normal variation curvature ---
        dot_products = np.clip(neighbor_normals @ ni, -1.0, 1.0)
        angle_variation = np.arccos(dot_products)
        weights = 1.0 / (np.linalg.norm(neighbor_points - points[i], axis=1) + 1e-6)
        k_i = np.average(angle_variation, weights=weights)

        # --- Normalize by local spatial and surface area scale ---
        curvatures[i] = (k_i / (di + 1e-6)) / (areas[i] + 1e-6)

    # Smooth for stability
    curvatures = gaussian_filter1d(curvatures, sigma=2)
    return np.maximum(curvatures, 1e-6)


def curvature_guided_gamma(points, camera_origin, kernel, strategy="p95", safety=1.15):
    """
    Generate curvature-informed gamma values for HPR benchmarking.
    
    Args:
        points: Nx3 array of point positions
        camera_origin: 3D camera position
        kernel: kernel type ('spherical_flip', 'mirror', 'exp_inversion', 'exp_natural')
        strategy: how to choose global R ("max", "p95", "p99")
        safety: safety factor for R bounds
    
    Returns:
        gamma_values: array of gamma values to test
        xlabel: string for x-axis label
    """
    # --- Estimate per-point curvature magnitude ---
    areas = estimate_areas(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    normals = np.asarray(pcd.normals)

    k = estimate_curvature(points, normals, areas, n_neighbors=30)

    # --- Compute per-point R bounds from Lemma 4.3 ---
    k_nonzero = np.maximum(k, 1e-6)
    r = np.linalg.norm(points - camera_origin, axis=1)
    
    R_global, R_per_point = choose_global_R(points, camera_origin, k, strategy=strategy, safety=safety)
    
    # Get distribution statistics
    R_min = np.percentile(R_per_point, 10)
    R_max = np.percentile(R_per_point, 90)
    R_median = np.median(R_per_point)
    k_median = np.median(k)
    k_90 = np.percentile(k, 90)
    k_mean = np.mean(k)
    r_median = np.median(r)
    r_max = np.max(r)
    r_mean = np.mean(r)

    # --- Kernel-dependent gamma generation ---
    if kernel == "spherical_flip":
        # p' = (2γ - r) * (p/r)
        # From Lemma 4.3: γ should be at least R = 0.5 * k * r^2
        gamma_min = R_min * 0.01   
        gamma_max = R_max * 1 
        
        gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 60)
        xlabel = "Log(γ)"
        
    elif kernel == "mirror":
        # p' = (γ - r) * (p/r)
        gamma_min = R_min * 0.01
        gamma_max = R_max * 1.0
        
        gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 60)
        xlabel = "Log(γ)"
        
    elif kernel == "exp_inversion":
        # p' = (r^γ) * (p/r)
        # For negative γ: r^γ = 1/r^|γ|
        # The inversion strength should scale with curvature:
        # - Higher curvature → need MORE negative γ (stronger inversion)
        # - The exponent should relate to the curvature-distance relationship
        
        # Compute curvature-based scaling factor
        # If curvature is high relative to distance, need stronger inversion
        curvature_scale = k_mean * r_mean  # dimensionless measure of geometric complexity
        
        # Normalize to a reference scale (typical value ~ 1.0)
        # Higher values → more complex geometry → need more negative gamma
        curvature_factor = np.clip(curvature_scale / 0.1, 0.5, 5.0)  # clamp to reasonable range
        
        # Base range scales with curvature
        # High curvature → more negative (stronger inversion needed)
        gamma_min = -10.0 * curvature_factor    # more negative for high curvature
        gamma_max = -0.5 / curvature_factor      # less negative for low curvature
        
        gamma_values = np.linspace(gamma_min, gamma_max, 60)
        xlabel = "γ (negative exponent)"
        
        print(f"  [exp_inversion] curvature_scale={curvature_scale:.3e}, factor={curvature_factor:.3f}")
        
    elif kernel == "exp_natural":
        # p' = exp(-γ * r) * (p/r)
        gamma_min = k_median / r_max * 0.1       # gentle decay
        gamma_max = k_90 / r_median * 3.0        # aggressive decay
        
        gamma_values = np.linspace(gamma_min, gamma_max, 60)
        xlabel = "γ (decay rate)"
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    print(f"[Curvature-guided γ] {kernel}:")
    print(f"  Curvature k: mean={k_mean:.3e}, median={k_median:.3e}, 90th={k_90:.3e}")
    print(f"  Distance r: mean={r_mean:.3f}, median={r_median:.3f}, max={r_max:.3f}")
    print(f"  R bounds (Lemma 4.3): min={R_min:.3e}, median={R_median:.3e}, max={R_max:.3e}")
    print(f"  Global R ({strategy}): {R_global:.3e}")
    print(f"  γ range: [{gamma_values[0]:.3e}, {gamma_values[-1]:.3e}]")
    if kernel in ["spherical_flip", "mirror"]:
        print(f"  Log(γ) range: [{np.log10(gamma_values[0]):.2f}, {np.log10(gamma_values[-1]):.2f}]")
    
    return gamma_values, xlabel, R_global

def gamma_zoom_curvature(mesh_path, out_folder,
                             kernels=("spherical_flip", "mirror", "exp_inversion", "exp_natural"),
                             safety=1.15,
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

    # --- Estimate curvature to guide gamma selection (Lemma 4.3:  k < 2/R ) ---
    print("[Info] Estimating curvature to guide gamma range ...")


    all_results = []

    # --- Loop over kernels ---
    for kernel in kernels:
        print(f"\n[Kernel: {kernel}]")

        gamma_values, xlabel, R = curvature_guided_gamma( points, camera_origin, kernel, safety=safety)

        kernel_results = []

        # finding optimal gamma
        R_optimal = refine_R_via_visibility(points, camera_origin, kernel, step_factor=0.9, iterations=int(num_points/100))

        # --- Sweep gamma values ---
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
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "FP_rate": fp / (fp + tn + 1e-8),
                "FN_rate": fn / (fn + tp + 1e-8),
                "TOTAL_rate": (fp + fn) / (tp + fn + 1e-8)
            })

        # --- Build DataFrame ---
        df = pd.DataFrame(kernel_results)
        all_results.extend(kernel_results)

        df["FP_rate"] *= 100
        df["FN_rate"] *= 100
        df["TOTAL_rate"] *= 100

        # --- Save CSV ---
        csv_path = os.path.join(out_folder, f"benchmark_{kernel}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Saved CSV] {csv_path}")

        # --- MAIN PLOT ---
        fig, ax = plt.subplots(figsize=(7, 5))
        if kernel in ("spherical_flip", "mirror"):
            ax.set_xscale("log")

        ax.plot(df["gamma"], df["TOTAL_rate"], color="royalblue", marker="o", markersize=3, linewidth=2, label="Total error")
        ax.plot(df["gamma"], df["FP_rate"], color="green", marker="o", markersize=3, linewidth=1.5, label="False positive")
        ax.plot(df["gamma"], df["FN_rate"], color="red", marker="o", markersize=3, linewidth=1.5, label="False negative")
        ax.axvline(R, color="orange", linestyle="--", linewidth=2, label=f"Predicted R (Lemma 4.3)\n≈ {R:.2f}")
        ax.axvline(R_optimal, color="magenta", linestyle="--", linewidth=2, label=f"Optimal R (Lemma 4.3 + etc)\n≈ {R_optimal:.2f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Error rate (%)")
        ax.set_title(f"HPR Benchmark — {kernel}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        img_path_main = os.path.join(out_folder, f"benchmark_{kernel}.png")
        plt.savefig(img_path_main, dpi=300)
        plt.close()
        print(f"[Saved main plot] {img_path_main}")

        # --- ZOOMED PLOT ---
        min_row = df.loc[df["TOTAL_rate"].idxmin()]
        best_gamma = min_row["gamma"]

        # Compute zoom range robustly
        if best_gamma > 0:
            zoom_width = (best_gamma * 0.5, best_gamma * 2.0)
            zoom_df = df[(df["gamma"] >= zoom_width[0]) & (df["gamma"] <= zoom_width[1])]
        else:
            zoom_width = (best_gamma * 2.0, best_gamma * 0.5)
            zoom_df = df[(df["gamma"] <= zoom_width[0]) & (df["gamma"] >= zoom_width[1])]

        if zoom_df.empty:
            gamma_sorted = np.sort(df["gamma"].values)
            idx = np.searchsorted(gamma_sorted, best_gamma)
            idx_min = max(0, idx - 5)
            idx_max = min(len(gamma_sorted) - 1, idx + 5)
            zoom_df = df[df["gamma"].isin(gamma_sorted[idx_min:idx_max])]

        # Create a separate zoom figure
        fig_zoom, ax_zoom = plt.subplots(figsize=(6, 4))
        if kernel in ("spherical_flip", "mirror"):
            ax_zoom.set_xscale("log")

        ax_zoom.plot(zoom_df["gamma"], zoom_df["TOTAL_rate"], color="royalblue", linewidth=2, label="Total error")
        ax_zoom.plot(zoom_df["gamma"], zoom_df["FP_rate"], color="green", linewidth=1.5, label="False positive")
        ax_zoom.plot(zoom_df["gamma"], zoom_df["FN_rate"], color="red", linewidth=1.5, label="False negative")
        if zoom_df["gamma"].min() < R < zoom_df["gamma"].max():
            ax_zoom.axvline(R, color="orange", linestyle="--", linewidth=2, label=f"Predicted R ≈ {R:.2f}")

        ax_zoom.set_xlim(zoom_df["gamma"].min(), zoom_df["gamma"].max())
        ax_zoom.set_ylim(
            min(zoom_df["TOTAL_rate"].min(), zoom_df["FP_rate"].min(), zoom_df["FN_rate"].min()) * 0.9,
            max(zoom_df["TOTAL_rate"].max(), zoom_df["FP_rate"].max(), zoom_df["FN_rate"].max()) * 1.1,
        )

        # --- Annotate minimum total error ---
        min_row = df.loc[df["TOTAL_rate"].idxmin()]
        min_gamma = min_row["gamma"]
        min_error = min_row["TOTAL_rate"]

        # Mark the minimum point
        ax_zoom.scatter(min_gamma, min_error, color="gold", s=50, edgecolor="black", zorder=5)
        ax_zoom.annotate(
            f"Min error = {min_error:.2f}%\nγ = {min_gamma:.4f}",
            xy=(min_gamma, min_error),
            xycoords='data',
            xytext=(30, 30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )


        ax_zoom.set_xlabel(xlabel)
        ax_zoom.set_ylabel("Error rate (%)")
        ax_zoom.set_title(f"HPR Zoom — {kernel} (γ≈{best_gamma:.4f})")
        ax_zoom.legend()
        ax_zoom.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        img_path_zoom = os.path.join(out_folder, f"benchmark_{kernel}_zoom.png")
        plt.savefig(img_path_zoom, dpi=300)
        plt.close()
        print(f"[Saved zoom plot] {img_path_zoom}")

        print(f"→ Kernel {kernel}: min total error {min_row['TOTAL_rate']:.2f}% at γ={min_row['gamma']:.4f}")



    # --- Combined overlay ---
    # --- Combined overlay ---
    if not all_results:
        raise RuntimeError("No kernel results were collected. Did you append kernel_results to all_results?")

    combined_df = pd.DataFrame(all_results)
    if "kernel" not in combined_df.columns:
        print("[Warning] Missing 'kernel' column, adding placeholder...")
        combined_df["kernel"] = "unknown"

    combined_csv = os.path.join(out_folder, "benchmark_all_kernels.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n[All kernels combined results saved] {combined_csv}")


    plt.figure(figsize=(9, 6))
    if "kernel" in combined_df.columns:
        for kernel in combined_df["kernel"].unique():
            subset = combined_df[combined_df["kernel"] == kernel]
            plt.plot(subset["gamma"], subset["TOTAL_rate"], label=kernel, linewidth=2)
    else:
        # fallback if kernel column missing
        plt.plot(combined_df["gamma"], combined_df["TOTAL_rate"], color="royalblue", label="Total error")

    plt.xlabel("γ (gamma)")
    plt.ylabel("Total error rate (%)")
    plt.title("HPR Total Error vs γ — All Kernels (Curvature-Guided)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    overlay_path = os.path.join(out_folder, "benchmark_overlay_total.png")
    plt.savefig(overlay_path, dpi=300)
    plt.show()
    print(f"[Saved overlay plot] {overlay_path}")


    return combined_df
     




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


    # --------------------- visualization test ---------------------
    mesh_path = "../data/bunny_tri.obj"
    camera_origin = np.array([0, 0, 3.0])
    R = None  # will compute below

    # --------------------- visualization test ---------------------
    V, F = gpy.read_mesh(mesh_path)
    points = gpy.random_points_on_mesh(V, F, args.num_points)
    # run gamma based on curvature
    gamma_values, xlabel, R = curvature_guided_gamma( points, camera_origin, args.kernel,)
    kernels = [args.kernel]

    results, points, visibility_gt, all_hull_pts = evaluate_hpr_with(
        mesh_path, camera_origin, gamma_values, kernels, num_points=args.num_points
    )

    print(f"\nCurvature-guided gamma results:")
    print(f"GT visible points: {np.sum(visibility_gt)} / {len(visibility_gt)}")
    print(f"Gamma values: {gamma_values}")

    # Find best result by F1 score
    best_result = max(results, key=lambda r: 2 * r['precision'] * r['recall'] / (r['precision'] + r['recall'] + 1e-10))
    best_gamma = best_result['gamma']
    hpr_visible_pts = all_hull_pts[(args.kernel, best_gamma)]

    print(f"Best gamma: {best_gamma:.4f}")
    print(f"Precision: {best_result['precision']:.3f}, Recall: {best_result['recall']:.3f}")

    visualize_visibility_comparison(points, visibility_gt, hpr_visible_pts, camera_origin)

    # --------------------- sweep visualize ---------------------

    
    gamma_zoom_curvature(
        mesh_path="../data/bunny_tri.obj",
        out_folder="../data/curvature_guided_gamma/",
        kernels=("spherical_flip", "mirror"),
        safety=0.5,
        num_points=10000
    )