import os
import numpy as np
import open3d as o3d
from gamma_visualize_utils import benchmark_HPR_visibility  # assuming gamma_visualize.py defines this
from curvature_gamma import curvature_guided_gamma, refine_R_via_visibility
import trimesh
import pandas as pd
import matplotlib.pyplot as plt


# --- Parameters ---
meshes = [
    "../data/bunny.obj",
    # "../data/buddha.obj",
    # "../data/dragon.obj",
    # "../data/erato.obj",
    # "../data/hairball.obj",
]
densities = [1000, 5000, 10000, 20000]
kernels = ["spherical_flip", "mirror"]
out_root = "../data/benchmarks_curvature"

os.makedirs(out_root, exist_ok=True)


def compute_R_error_all(results_dir):
    """
    Automatically scan benchmark folders and compute |R_pred - R_opt|.
    Requires that each folder contains benchmark_{kernel}.csv.
    """
    summary = []

    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse naming convention like 'bunny_1000_mirror'
        try:
            mesh_name, density, kernel = folder.split("_", 2)
        except ValueError:
            print(f"⚠️ Skipping unexpected folder name: {folder}")
            continue

        csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        min_gamma = df.loc[df["TOTAL_rate"].idxmin(), "gamma"]

        # Retrieve R_pred and R_opt from the plotted lines (in filename or pre-stored CSV)
        # If not available, you can load them from metadata or logs you saved separately
        # Example: read from an optional R_summary.csv file or json
        R_pred = None
        R_opt = None

        # Try to extract R_pred/R_opt if you stored them somewhere
        meta_path = os.path.join(folder_path, "meta.txt")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    if "R_pred" in line:
                        R_pred = float(line.split("=")[-1].strip())
                    elif "R_opt" in line:
                        R_opt = float(line.split("=")[-1].strip())

        if R_pred is None or R_opt is None:
            print(f"⚠️ No R info in {folder}")
            continue

        abs_error = abs(R_pred - R_opt)
        rel_error = abs_error / R_opt * 100

        summary.append({
            "mesh": mesh_name,
            "density": int(density),
            "kernel": kernel,
            "R_pred": R_pred,
            "R_opt": R_opt,
            "R_min_from_csv": min_gamma,
            "|R_pred - R_opt|": abs_error,
            "rel_error_%": rel_error
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(results_dir, "R_error_summary.csv"), index=False)
    return df_summary



# --- Storage for R values ---
# records = []
# # --- Main loop ---
# for mesh_path in meshes:
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

    

#     for density in densities:
#         print(f"  -> Sampling {density} points")
#         pcd = mesh.sample_points_poisson_disk(number_of_points=density)
#         points = np.asarray(pcd.points)
#         camera_origin = np.array([0, 0, 5.0])  # or use mesh.get_center() + offset

#         for kernel in kernels:
#             print(f"     Kernel: {kernel}")
#             outdir = os.path.join(out_root, f"{mesh_name}_{density}_{kernel}")
#             os.makedirs(outdir, exist_ok=True)

#             # Compute curvature-guided gamma values
#             gamma_values, xlabel, R = curvature_guided_gamma(points, camera_origin, kernel)

#             # Refine R based on visibility overlap
#             R_optimal = refine_R_via_visibility(points, camera_origin, kernel)

#             records.append({
#                 "mesh": mesh_name,
#                 "density": density,
#                 "kernel": kernel,
#                 "R_opt": R_optimal
#             })

            # Run HPR benchmark and generate plots
            # benchmark_HPR_visibility(
            #     points=points,
            #     camera_origin=camera_origin,
            #     kernel=kernel,
            #     gamma_values=gamma_values,
            #     R_pred=R,
            #     R_opt=R_optimal,
            #     mesh_path=mesh_path,
            #     outdir=outdir
            # )




# comparing R_opt



# === Run ===
# df_R = pd.DataFrame(records)
# df_R.to_csv(os.path.join(out_root, "R_summary.csv"), index=False)
# print(f"\n✅ Saved all predicted & refined R values to {os.path.join(out_root, 'R_summary.csv')}")

# --- Now compare to empirical minimum R (from benchmark CSVs) ---
def collect_min_R_from_csv(results_dir):
    # print("Current working dir:", os.getcwd())
    # print("Results dir:", out_root)
    # print("Folders found:", os.listdir(out_root))
    data = []
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            mesh_name, density, kernel = folder.split("_", 2)
        except ValueError:
            continue

        csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
        print(f"trying to open path {csv_path}")
        if not os.path.exists(csv_path):
            print("this path doesn't exist")
            continue

        df = pd.read_csv(csv_path)
        min_gamma = df.loc[df["TOTAL_rate"].idxmin(), "gamma"]
        print(f" min gamma is {min_gamma} for mesh {mesh_name} {density} {kernel}")
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