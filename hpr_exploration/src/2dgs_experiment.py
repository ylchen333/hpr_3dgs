import numpy as np
from gamma_visualize_utils import benchmark_HPR_visibility, HPR_Param
import os
import open3d as o3d
import trimesh
import torch
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import torch
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- make 2DGS repo importable ---
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../2d-gaussian-splatting"))
sys.path.append(ROOT)

# --- import from 2DGS repo ---
# from gaussian_renderer import render, GaussianModel
# from scene import Scene
# from utils.render_utils import generate_path
# from arguments import ModelParams, PipelineParams


"""
commands used to train the 2dgs model and extract points, run in hpr_3dgs root:

python train.py -s ../data/blender/lego -m output/blender/lego

unbounded mesh extraction!!
python render.py -s ../data/blender/lego -m output/blender/lego --unbounded --skip_test --skip_train --mesh_res 1024

bounded mesh extraction if you focus on foreground
python render.py -s ../data/blender/lego -m output/blender/lego --skip_test --skip_train --mesh_res 1024


after hpr addition:
python train.py -s ../data/blender/lego -m output/hpr/blender/lego
python render.py -s ../data/blender/lego -m output/hpr/blender/lego --unbounded --skip_test --skip_train --mesh_res 1024
python render.py -s ../data/blender/lego -m output/hpr/blender/lego --skip_test --skip_train --mesh_res 1024


"""

#----------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------------



def load_2dgs_points_from_ply(ply_path):
    print("[Loading 2DGS Gaussians from ply]:", ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)



def visible_idx_hpr(points, cam, gamma):
    pts = torch.tensor(points.T, dtype=torch.float32)
    cam = torch.tensor(cam, dtype=torch.float32)

    visible_pts, idx = HPR_Param(pts, cam, gamma, use_linear=False)
    idx = idx.cpu().numpy().ravel()
    return idx

def render_with_mask(cam, mask):
    old_opacity = gaussians.opacity.clone()

    # mask is boolean array: True = visible
    opacity = old_opacity.clone()
    opacity[~mask] = 0.0
    gaussians.opacity = torch.nn.Parameter(opacity)

    # render the frame
    out = render(cam, gaussians, pipe)

    # restore opacities for the next frame
    gaussians.opacity = torch.nn.Parameter(old_opacity)

    return out["render"]


#----------------------------------------------------------------------------------------------------------------------------
# CONFIG
#----------------------------------------------------------------------------------------------------------------------------


KERNEL = "exp_inversion" # ["spherical_flip", "exp_inversion"] fixed kernel for this experiment
DENSITY = 100000  # fixed density for this experiment for now since this is how many points at initialization for 2dgs
gamma_values = np.logspace(-7, 0, 100)
N_FRAMES = 240 # for flyaround video

root = "/home/lorie/Desktop/cmu_coursework/hpr_3dgs"
mesh_path = f"{root}/data/blender/lego/lego.ply"
mesh_2dgs_path = f"{root}/2d-gaussian-splatting/output/blender/lego/train/ours_30000/fuse_unbounded.ply" # path to mesh extracted from 2dgs points
MODEL_PATH = f"{root}/2d-gaussian-splatting/output/blender/lego"

out_root = f"{root}/outputs/2dgs_experiment"
os.makedirs(out_root, exist_ok=True)


#----------------------------------------------------------------------------------------------------------------------------
# Benchmarking
#----------------------------------------------------------------------------------------------------------------------------

# run benchmark on both the original mesh and the mesh extracted from 2DGS points
 # --- Load and triangulate mesh if necessary ---
mesh_gt = trimesh.load(mesh_path, force='mesh')
if mesh_gt.faces.shape[1] != 3:
    print(f"[Info] Triangulating {mesh_path} ...")
    mesh = mesh_gt.subdivide_to_size(max_edge=0.01)

tri_path = os.path.join(os.path.dirname(mesh_path), f"{mesh_path}_tri.obj")
if not os.path.exists(tri_path):
    mesh.export(tri_path)
mesh_path = tri_path

mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
mesh_o3d.compute_vertex_normals()

# --- Sample points ---
pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=DENSITY)
points_gt = np.asarray(pcd.points)
camera_origin = np.array([0, 0, 5.0])


# --- Run benchmark on GT mesh ---
outdir_gt = os.path.join(out_root, f"lego_gt_{KERNEL}_{DENSITY}")
os.makedirs(outdir_gt, exist_ok=True)

print("=== Running HPR benchmark on GT mesh ===")
df_gt = benchmark_HPR_visibility(
    points=points_gt,
    camera_origin=camera_origin,
    kernel=KERNEL,
    gamma_values=gamma_values,
    R_pred=0,
    R_opt=0,
    mesh_path=mesh_path,
    outdir=outdir_gt
)

# --- Load 2DGS points ---
ckpt_dir = f"{root}/2d-gaussian-splatting/output/blender/lego/point_cloud/iteration_30000"

ply_list = [f for f in os.listdir(ckpt_dir) if f.endswith(".ply")]
if not ply_list:
    raise FileNotFoundError("No .ply files found in 2DGS output folder.")

ply_path = os.path.join(ckpt_dir, ply_list[-1])
points_2dgs = load_2dgs_points_from_ply(ply_path)


# --- Run benchmark: 2DGS points vs GT mesh ---
outdir_2dgs_meshgt = os.path.join(out_root, f"lego_2dgs_meshgt{KERNEL}")
os.makedirs(outdir_2dgs_meshgt, exist_ok=True)

print("=== Running HPR benchmark on 2DGS point cloud against GT mesh ===")
df_2dgs_meshgt = benchmark_HPR_visibility(
    points=points_2dgs,
    camera_origin=camera_origin,
    kernel=KERNEL,
    gamma_values=gamma_values,
    R_pred=0,
    R_opt=0,
    mesh_path=mesh_path,
    outdir=outdir_2dgs_meshgt
)

# --- Run benchmark: 2DGS points vs reconstructed mesh ---
print("=== Running HPR benchmark on 2DGS point cloud against reconstructed mesh ===")
outdir_2dgs_mesh = os.path.join(out_root, f"lego_2dgs_mesh{KERNEL}")
os.makedirs(outdir_2dgs_mesh, exist_ok=True)

df_2dgs_mesh = benchmark_HPR_visibility(
    points=points_2dgs,
    camera_origin=camera_origin,
    kernel=KERNEL,
    gamma_values=gamma_values,
    R_pred=0,
    R_opt=0,
    mesh_path=mesh_2dgs_path,
    outdir=outdir_2dgs_mesh
)

print("\n=== DONE RUNNING BENCHMARKS ===")
print(f"GT results saved to:         {outdir_gt}")
print(f"2DGS vs GT mesh saved to:    {outdir_2dgs_meshgt}")
print(f"2DGS vs 2DGS mesh saved to:  {outdir_2dgs_mesh}")


#----------------------------------------------------------------------------------------------------------------------------
# COMBINED COMPARISON: CSV + PLOTS
#----------------------------------------------------------------------------------------------------------------------------

# Label each dataframe and merge
df_gt["source"] = "GT points vs GT mesh"
df_2dgs_meshgt["source"] = "2DGS points vs GT mesh"
df_2dgs_mesh["source"] = "2DGS points vs 2DGS mesh"

df_combined = pd.concat([df_gt, df_2dgs_meshgt, df_2dgs_mesh], ignore_index=True)
combined_csv_path = os.path.join(out_root, f"comparison_{KERNEL}.csv")
df_combined.to_csv(combined_csv_path, index=False)
print(f"[Saved combined CSV] {combined_csv_path}")

# --- Subplot comparison: one panel per config showing FP, FN, Total ---
configs = [
    ("GT points vs GT mesh",     df_gt,          "steelblue"),
    ("2DGS points vs GT mesh",   df_2dgs_meshgt, "darkorange"),
    ("2DGS points vs 2DGS mesh", df_2dgs_mesh,   "forestgreen"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
for ax, (label, df, color) in zip(axes, configs):
    ax.set_xscale("log")
    ax.plot(df["gamma"], df["TOTAL_rate"], color=color,   linewidth=2,   label="Total error")
    ax.plot(df["gamma"], df["FP_rate"],    color="green", linewidth=1.5, linestyle="--", label="FP rate")
    ax.plot(df["gamma"], df["FN_rate"],    color="red",   linewidth=1.5, linestyle=":",  label="FN rate")
    min_row = df.loc[df["TOTAL_rate"].idxmin()]
    ax.scatter(min_row["gamma"], min_row["TOTAL_rate"], color="gold", s=60, edgecolor="black", zorder=5)
    ax.annotate(
        f"Min {min_row['TOTAL_rate']:.1f}%\nγ={min_row['gamma']:.1e}",
        xy=(min_row["gamma"], min_row["TOTAL_rate"]),
        xytext=(0, 20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
    )
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("γ (log scale)")
    ax.set_ylabel("Error rate (%)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.suptitle(f"HPR Visibility Benchmark Comparison — {KERNEL}", fontsize=13, fontweight="bold")
plt.tight_layout()
subplots_path = os.path.join(out_root, f"comparison_subplots_{KERNEL}.png")
plt.savefig(subplots_path, dpi=300)
plt.close(fig)
print(f"[Saved subplot comparison] {subplots_path}")

# --- Single overlay: Total error across all 3 configs ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xscale("log")
for label, df, color in configs:
    ax.plot(df["gamma"], df["TOTAL_rate"], color=color, linewidth=2, label=label)
    min_row = df.loc[df["TOTAL_rate"].idxmin()]
    ax.scatter(min_row["gamma"], min_row["TOTAL_rate"], color=color, s=60, edgecolor="black", zorder=5)

ax.set_title(f"HPR Total Error Rate — All Configurations ({KERNEL})", fontsize=12)
ax.set_xlabel("γ (log scale)")
ax.set_ylabel("Total error rate (%)")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
overlay_path = os.path.join(out_root, f"comparison_overlay_{KERNEL}.png")
plt.savefig(overlay_path, dpi=300)
plt.close(fig)
print(f"[Saved overlay comparison] {overlay_path}")

# --- Summary table: optimal gamma and errors per config ---
summary_rows = []
for label, df, _ in configs:
    min_row = df.loc[df["TOTAL_rate"].idxmin()]
    summary_rows.append({
        "source": label,
        "optimal_gamma": min_row["gamma"],
        "min_total_error_%": round(min_row["TOTAL_rate"], 3),
        "FP_rate_%": round(min_row["FP_rate"], 3),
        "FN_rate_%": round(min_row["FN_rate"], 3),
    })
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(out_root, f"summary_{KERNEL}.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"[Saved summary CSV] {summary_csv_path}")
print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))


# #----------------------------------------------------------------------------------------------------------------------------
# # rendering with HPR visibility mask
# #----------------------------------------------------------------------------------------------------------------------------
# # only render points marked as visible by hpr using optimal gamma
# gamma_values_csv = os.path.join(outdir_gt, "benchmark_exp_inversion.csv")
# optimal_gamma = pd.read_csv(gamma_values_csv).loc[0, "optimal_gamma"]
# print(f"Optimal gamma for GT mesh: {optimal_gamma}")


# device = "cuda"
# parser = ArgumentParser()
# model = ModelParams(parser)
# pipeline = PipelineParams(parser)

# args = parser.parse_args([])
# args.model_path = model_path
# args.iteration = -1  # latest checkpoint

# dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
# gaussians = GaussianModel(dataset.sh_degree)
# scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

# points = gaussians.get_xyz.detach().cpu().numpy()    # shape [N,3]


# # generate flyaround camera path
# out_gif_path = os.path.join(out_root, "lego_hpr.gif")
# frames = []
# for cam in tqdm(cams):
#     cam_pos = cam.camera_center.cpu().numpy()
#     idx = visible_idx_hpr(points, cam_pos, gamma=1e-3)
#     mask = np.zeros(points.shape[0], bool); mask[idx] = True

#     frame = render_with_mask(cam, mask)
#     frame_np = (frame.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
#     frames.append(frame_np)

# imageio.mimsave(out_gif_path, frames, duration=1/24)
