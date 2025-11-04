import os
import numpy as np
import open3d as o3d
from gamma_visualize_utils import benchmark_HPR_visibility  # assuming gamma_visualize.py defines this
from curvature_gamma import curvature_guided_gamma, refine_R_via_visibility
import trimesh
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- Parameters ---
meshes = [
    "../../data/bunny.obj",
    "../../data/buddha.obj",
    "../../data/dragon.obj",
    "../../data/erato.obj",
    "../../data/hairball.obj",
]
DENSITY = 10000  # Fixed density for distance benchmark
distances = [0.5, 1.0, 2.0, 5.0]  # e.g. half, equal, 2×, 5× the mesh diagonal
kernels = ["spherical_flip", "exp_inversion"]
out_root = "../data/yannis_hpr/benchmarks_distance_relative"

os.makedirs(out_root, exist_ok=True)


# --- Main loop ---
# --- Storage for R values ---
records = []

# Generate all combinations of mesh, density, kernel
combinations = [(m, d, k) for m in meshes for d in distances for k in kernels]

# Progress bar setup
progress_bar = tqdm(total=len(combinations), desc="Running HPR Benchmarks", ncols=100)

# --- Main loop ---
for mesh_path, distance, kernel in combinations:
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

    # --- Load and triangulate mesh if necessary ---
    mesh = trimesh.load(mesh_path, force='mesh')
    if mesh.faces.shape[1] != 3:
        print(f"[Info] Triangulating {mesh_name} ...")
        mesh = mesh.subdivide_to_size(max_edge=0.01)

    tri_path = os.path.join(os.path.dirname(mesh_path), f"{mesh_name}_tri.obj")
    if not os.path.exists(tri_path):
        mesh.export(tri_path)
    mesh_path = tri_path

    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    mesh_o3d.compute_vertex_normals()

    # --- Sample points ---
    pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=DENSITY)
    points = np.asarray(pcd.points)

    # --- Compute mesh center and scale ---
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    mesh_center = bbox.get_center()
    mesh_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    # mesh_size ≈ diagonal length of bounding box

    # variable distance relative to mesh size
    rel_d = distance
    distance = rel_d * mesh_size
    camera_origin = mesh_center + np.array([0, 0, distance])
    print(f"\n[Info] Camera distance = {rel_d:.2f} × mesh size ({distance:.3f} units).")


    # --- Compute gamma grid ---
    # gamma_values, xlabel, R_pred = curvature_guided_gamma(points, camera_origin, kernel=kernel)

    # --- Normalize gamma_values to [0, 1]
    # gamma_min, gamma_max = np.min(gamma_values), np.max(gamma_values)
    # if gamma_max > gamma_min:
    #     gamma_values = (gamma_values - gamma_min) / (gamma_max - gamma_min)
    # else:
    #     gamma_values = np.zeros_like(gamma_values)
    # gamma_values = np.clip(gamma_values, 1e-3, 0.999)
    gamma_values = np.logspace(-7, 0, 100)

    # --- Refine R ---
    # R_optimal = refine_R_via_visibility(points, camera_origin, kernel)

    # --- Record results ---
    records.append({
        "mesh": mesh_name,
        "distance": distance,
        "kernel": kernel,
        "rel_distance": rel_d,
        # "R_opt": R_optimal
    })

    # --- Run benchmark ---
    outdir = os.path.join(out_root, f"{mesh_name}_{rel_d}_{distance}_{kernel}")
    os.makedirs(outdir, exist_ok=True)
    benchmark_HPR_visibility(
        points=points,
        camera_origin=camera_origin,
        kernel=kernel,
        gamma_values=gamma_values,
        R_pred=0,
        R_opt=0,
        mesh_path=mesh_path,
        outdir=outdir
    )

    progress_bar.set_postfix({"mesh": mesh_name, "rel_distance": rel_d, "distance": distance, "kernel": kernel})
    progress_bar.update(1)

progress_bar.close()

# ====== comparing R_opt ======

# === Run ===
# df_R = pd.DataFrame(records)
# df_R.to_csv(os.path.join(out_root, "R_summary.csv"), index=False)
# print(f"\n✅ Saved all predicted & refined R values to {os.path.join(out_root, 'R_summary.csv')}")

# # --- Now compare to empirical minimum R (from benchmark CSVs) ---
# from tqdm import tqdm

# def collect_min_R_from_csv(results_dir):
#     folders = sorted([f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))])
#     data = []
#     for folder in tqdm(folders, desc="Loading benchmark CSVs", ncols=100):
#         folder_path = os.path.join(results_dir, folder)
#         parts = folder.split("_")
#         if len(parts) < 3:
#             continue
#         kernel = parts[-1]
#         distance = parts[-2]
#         mesh_name = "_".join(parts[:-2])

#         csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
#         if not os.path.exists(csv_path):
#             continue

#         df = pd.read_csv(csv_path)
#         df.columns = [c.strip() for c in df.columns]
#         if "gamma" not in df.columns or "TOTAL_rate" not in df.columns:
#             continue

#         min_gamma = df.loc[df["TOTAL_rate"].idxmin(), "gamma"]
#         data.append({
#             "mesh": mesh_name,
#             "distance": int(distance),
#             "kernel": kernel,
#             "R_min_from_csv": min_gamma
#         })
#     return pd.DataFrame(data)


# df_min = collect_min_R_from_csv(out_root)

# df_R = pd.read_csv(out_root+"/R_summary.csv")
# print("df_R columns:", df_R.columns.tolist())
# print("df_min columns:", df_min.columns.tolist())
# print("df_min sample:\n", df_min.head())

# # --- Merge and compute errors ---
# df_summary = pd.merge(df_R, df_min, on=["mesh", "distance", "kernel"], how="inner")
# print(df_summary[["mesh", "distance", "kernel", "R_opt", "R_min_from_csv"]].head(10))
# df_summary["|R_opt - R_min|"] = abs(df_summary["R_opt"] - df_summary["R_min_from_csv"])
# df_summary["rel_err_opt_%"] = 100 * df_summary["|R_opt - R_min|"] / df_summary["R_min_from_csv"]

# summary_path = os.path.join(out_root, "R_error_summary.csv")
# df_summary.to_csv(summary_path, index=False)
# print(f"✅ Saved comparison results to {summary_path}")

# # --- Plot results ---
# plt.figure(figsize=(8,5))
# for kernel in df_summary["kernel"].unique():
#     subset = df_summary[df_summary["kernel"] == kernel]
#     plt.plot(subset["mesh"] + "_" + subset["distance"].astype(str),
#              subset["|R_opt - R_min|"], "s--", label=f"{kernel} (refined)")

# plt.title("Deviation between Optimal R and True Minimum R")
# plt.ylabel("|R - R_min|")
# plt.xticks(rotation=45, ha="right")
# plt.legend()
# plt.tight_layout()
# plt.show()
