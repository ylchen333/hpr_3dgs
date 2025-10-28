import numpy as np
import trimesh
from gamma_visualize_utils import benchmark_HPR_visibility
from curvature_gamma import curvature_guided_gamma, refine_R_via_visibility
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import open3d as o3d


# --- Parameters ---
meshes = [
    "../../data/bunny.obj",
    "../../data/buddha.obj",
    "../../data/dragon.obj",
    "../../data/erato.obj",
    "../../data/hairball.obj",
]
densities = [1000, 5000, 10000, 20000]
kernels = ["spherical_flip", "mirror"]
out_root = "../data/benchmarks_density"

os.makedirs(out_root, exist_ok=True)


def density_to_gamma(mesh_tri_path, N, camera_origin, alpha=0.5, n_grid=40, span_octaves=2.0):
    """
    Given a triangulated mesh file, number of sampled points N, and camera C,
    return (gamma_grid, gamma_pred, diagnostics_dict).
    """
    # --- load triangulated mesh & surface area ---
    mesh = trimesh.load(mesh_tri_path, force='mesh')
    S = float(mesh.area)  # surface area

    # --- density -> spacing rho ---
    lam = N / S                               # points per unit area
    rho = np.sqrt(1.0 / (np.pi * lam))        # spacing proxy

    # --- thickness d and representative r in camera frame ---
    V = mesh.vertices
    C = np.asarray(camera_origin, dtype=float)

    # Assume camera looks along -z in world. If you have a camera R|t, apply it here.
    z = V[:, 2]                               # world z
    d = float(z.max() - z.min())              # thickness along view

    r_med = float(np.median(np.linalg.norm(V - C, axis=1)))

    # --- bound and predicted gamma ---
    R_max = d * (r_med ** 2) / (rho ** 2)     # = d * r^2 * π * N / S
    gamma_pred = alpha * R_max

    # --- log grid around gamma_pred (span_octaves in log10 space) ---
    lo = gamma_pred / (10 ** span_octaves)
    hi = min(R_max, gamma_pred * (10 ** span_octaves))
    gamma_grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)

    diag = dict(S=S, N=N, lam=lam, rho=rho, d=d, r_med=r_med, R_max=R_max, alpha=alpha)
    return gamma_grid, gamma_pred, diag



# --- Main loop ---
# --- Storage for R values ---
records = []

# Generate all combinations of mesh, density, kernel
combinations = [(m, d, k) for m in meshes for d in densities for k in kernels]

# Progress bar setup
progress_bar = tqdm(total=len(combinations), desc="Running HPR Benchmarks", ncols=100)

# --- Main loop ---
for mesh_path, density, kernel in combinations:
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
    pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=density)
    points = np.asarray(pcd.points)
    camera_origin = np.array([0, 0, 5.0])

    # --- Compute gamma grid ---
    gamma_values, R_pred, info = density_to_gamma(mesh_path, density, camera_origin)

    # --- Refine R ---
    R_optimal = refine_R_via_visibility(points, camera_origin, kernel)

    # --- Record results ---
    records.append({
        "mesh": mesh_name,
        "density": density,
        "kernel": kernel,
        "R_opt": R_optimal
    })

    # --- Run benchmark ---
    outdir = os.path.join(out_root, f"{mesh_name}_{density}_{kernel}")
    os.makedirs(outdir, exist_ok=True)
    benchmark_HPR_visibility(
        points=points,
        camera_origin=camera_origin,
        kernel=kernel,
        gamma_values=gamma_values,
        R_pred=R_pred,
        R_opt=R_optimal,
        mesh_path=mesh_path,
        outdir=outdir
    )

    progress_bar.set_postfix({"mesh": mesh_name, "density": density, "kernel": kernel})
    progress_bar.update(1)

progress_bar.close()
# for mesh_path, density, kernel in combinations:
#     mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    

#     mesh = trimesh.load(mesh_path, force='mesh')

#     # If faces are not triangles, triangulate using built-in utility
#     if mesh.faces.shape[1] != 3:
#         print("[Info] Triangulating non-triangular mesh...")
#         # Use Trimesh's "convex_hull" trick to force triangulation if needed
#         # mesh = mesh.convex_hull  # creates a triangulated hull
#         # Alternatively, use subdivision to break quads into tris
#         mesh = mesh.subdivide_to_size(max_edge=0.01)

#     # Verify
#     print(f"[Info] Mesh now has {mesh.faces.shape[0]} triangular faces.")
#     tri_path = os.path.join(os.path.dirname(mesh_path), f"{mesh_name}_tri.obj")
#     if not os.path.exists(tri_path):
#         mesh.export(tri_path)
#     mesh_path = tri_path  # now benchmark uses the triangulated file


#     # Save triangulated copy
#     mesh.export(tri_path)
#     print(f"[Info] Saved triangulated mesh to {tri_path}")

#     mesh = o3d.io.read_triangle_mesh(mesh_path)
#     mesh.compute_vertex_normals()
#     print(f"\n=== {mesh_name} ===")

    
#     print(f"  -> Sampling {density} points")
#     pcd = mesh.sample_points_poisson_disk(number_of_points=density)
#     points = np.asarray(pcd.points)
#     camera_origin = np.array([0, 0, 5.0])  # or use mesh.get_center() + offset


#     print(f"     Kernel: {kernel}")
#     outdir = os.path.join(out_root, f"{mesh_name}_{density}_{kernel}")
#     os.makedirs(outdir, exist_ok=True)
#     # you already sampled 'points' with len(points) == density
#     N = density
#     gamma_values, R_pred, info = density_to_gamma(mesh_path, N, camera_origin, alpha=0.5,
#                                                 n_grid=40, span_octaves=2.0)

#     # If you still want a refined R via your overlap search:
#     R_optimal = refine_R_via_visibility(points, camera_origin, kernel)

#     records.append({
#         "mesh": mesh_name,
#         "density": density,
#         "kernel": kernel,
#         "R_opt": R_optimal
#     })

#     # Now run the benchmark sweep using the new gamma grid:
#     benchmark_HPR_visibility(
#         points=points,
#         camera_origin=camera_origin,
#         kernel=kernel,
#         gamma_values=gamma_values,
#         R_pred=R_pred,
#         R_opt=R_optimal,
#         mesh_path=mesh_path,
#         outdir=outdir
#     )
#     progress_bar.set_postfix({"mesh": mesh_name, "density": density, "kernel": kernel})
#     progress_bar.update(1)

# progress_bar.close()


# comparing R_opt



# === Run ===
df_R = pd.DataFrame(records)
df_R.to_csv(os.path.join(out_root, "R_summary.csv"), index=False)
print(f"\n✅ Saved all predicted & refined R values to {os.path.join(out_root, 'R_summary.csv')}")

# --- Now compare to empirical minimum R (from benchmark CSVs) ---
from tqdm import tqdm

def collect_min_R_from_csv(results_dir):
    folders = sorted([f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))])
    data = []
    for folder in tqdm(folders, desc="Loading benchmark CSVs", ncols=100):
        folder_path = os.path.join(results_dir, folder)
        parts = folder.split("_")
        if len(parts) < 3:
            continue
        kernel = parts[-1]
        density = parts[-2]
        mesh_name = "_".join(parts[:-2])

        csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "gamma" not in df.columns or "TOTAL_rate" not in df.columns:
            continue

        min_gamma = df.loc[df["TOTAL_rate"].idxmin(), "gamma"]
        data.append({
            "mesh": mesh_name,
            "density": int(density),
            "kernel": kernel,
            "R_min_from_csv": min_gamma
        })
    return pd.DataFrame(data)


df_min = collect_min_R_from_csv(out_root)

df_R = pd.read_csv(out_root+"/R_summary.csv")
print("df_R columns:", df_R.columns.tolist())
print("df_min columns:", df_min.columns.tolist())
print("df_min sample:\n", df_min.head())

# --- Merge and compute errors ---
df_summary = pd.merge(df_R, df_min, on=["mesh", "density", "kernel"], how="inner")
print(df_summary[["mesh", "density", "kernel", "R_opt", "R_min_from_csv"]].head(10))
df_summary["|R_opt - R_min|"] = abs(df_summary["R_opt"] - df_summary["R_min_from_csv"])
df_summary["rel_err_opt_%"] = 100 * df_summary["|R_opt - R_min|"] / df_summary["R_min_from_csv"]

summary_path = os.path.join(out_root, "R_error_summary.csv")
df_summary.to_csv(summary_path, index=False)
print(f"✅ Saved comparison results to {summary_path}")

# --- Plot results ---
plt.figure(figsize=(8,5))
for kernel in df_summary["kernel"].unique():
    subset = df_summary[df_summary["kernel"] == kernel]
    plt.plot(subset["mesh"] + "_" + subset["density"].astype(str),
             subset["|R_opt - R_min|"], "s--", label=f"{kernel} (refined)")

plt.title("Deviation between Optimal R and True Minimum R")
plt.ylabel("|R - R_min|")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
