import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import seaborn as sns 

# === CONFIG ===
# Change to your directories
DISTANCE_DIR = "../data/yannis_hpr/benchmarks_distance"
REL_DISTANCE_DIR = "../data/yannis_hpr/benchmarks_distance_relative"
DENSITY_DIR = "../data/yannis_hpr/benchmarks_density"
OUT_ROOT = "../data/yannis_hpr/analysis_results"
os.makedirs(OUT_ROOT, exist_ok=True)



# -----------------------------------------------------------
# Helper: read min TOTAL_rate from all CSVs in a directory
# -----------------------------------------------------------
def parse_folder_name(folder):
    """
    Parse folder names with either format:
      1. {mesh}_{distance}_{kernel}
         e.g. 'bunny_200_exp_inversion'
      2. {mesh}_{relative_distance}_{distance}_{kernel}
         e.g. 'bunny_1.5_200_exp_inversion'

    Returns:
        (mesh_name, rel_distance, abs_distance, kernel)
        - rel_distance will be None if not present.
    """
    parts = folder.split("_")
    numeric_idxs = [i for i, p in enumerate(parts) if re.fullmatch(r"\d+(\.\d+)?", p)]

    # --- sanity check ---
    if not numeric_idxs:
        print(f"[⚠️ Skipped malformed folder name] {folder}")
        return None, None, None, None

    mesh_name = "_".join(parts[:numeric_idxs[0]])

    if len(numeric_idxs) == 1:
        # old format: {mesh}_{distance}_{kernel}
        rel_distance = None
        abs_distance = float(parts[numeric_idxs[0]])
        kernel = "_".join(parts[numeric_idxs[0] + 1:])
    else:
        # new format: {mesh}_{rel}_{dist}_{kernel}
        rel_distance = float(parts[numeric_idxs[0]])
        abs_distance = float(parts[numeric_idxs[1]])
        kernel = "_".join(parts[numeric_idxs[1] + 1:])

    return mesh_name, rel_distance, abs_distance, kernel

def collect_min_error_from_csv(results_dir, variable_name):
    """
    results_dir: path containing folders like bunny_1000_exp_inversion
    variable_name: "distance" or "density"
    """
    print(f"\n🔍 Collecting min TOTAL_rate from: {results_dir}")
    folders = sorted([f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))])
    data = []

    for folder in tqdm(folders, desc=f"Scanning {variable_name}", ncols=100):
        folder_path = os.path.join(results_dir, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and "benchmark" in f]
        if not csv_files:
            continue

        mesh_name, rel_distance, abs_distance, kernel= parse_folder_name(folder)
        if mesh_name is None:
            print(f"[⚠️ Skipped malformed folder name] {folder}")
            continue

        # print(f"[Parsed] mesh={mesh_name}, kernel={kernel}, {variable_name}={variable_value} for folder={folder}")

        if variable_name == "density":
            # density-type benchmark uses only one numeric field
            variable_value = abs_distance or rel_distance  # whichever exists
        elif variable_name == "distance":
            # distance benchmark; prefer abs_distance
            variable_value = abs_distance
        elif variable_name == "relative_distance":
            # relative-distance benchmark; prefer rel_distance
            variable_value = rel_distance
        else:
            print(f"[⚠️ Unknown variable_name: {variable_name}] Skipping {folder}")
            continue

        if variable_value is None:
            print(f"[⚠️ Missing variable value] {folder}")
            continue

        csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "TOTAL_rate" not in df.columns:
            continue

        min_row = df.loc[df["TOTAL_rate"].idxmin()]
        data.append({
            "mesh": mesh_name,
            "kernel": kernel,
            variable_name: variable_value,
            "min_error": min_row["TOTAL_rate"],
            "gamma_at_min": min_row["gamma"]
        })

    df = pd.DataFrame(data)
    print(f"✅ Found {len(df)} records in {results_dir}")
    return df
# -----------------------------------------------------------



# === Collect data ===
df_dist = collect_min_error_from_csv(DISTANCE_DIR, "distance")
df_dens = collect_min_error_from_csv(DENSITY_DIR, "density")
df_rel_dist = collect_min_error_from_csv(REL_DISTANCE_DIR, "relative_distance")

# Save raw summaries
df_dist.to_csv(os.path.join(OUT_ROOT, "min_error_distance.csv"), index=False)
df_dens.to_csv(os.path.join(OUT_ROOT, "min_error_density.csv"), index=False)
df_rel_dist.to_csv(os.path.join(OUT_ROOT, "min_error_rel_distance.csv"), index=False)
print("\n✅ Saved summaries for distance and density.")


# Distinct colors by mesh name (consistent across all plots)
MESHES = sorted(set(df_dist["mesh"].unique())
                | set(df_dens["mesh"].unique())
                | set(df_rel_dist["mesh"].unique()))
MESH_COLORS = dict(zip(MESHES, sns.color_palette("tab10", len(MESHES))))

# Distinct line styles per kernel
LINESTYLES = {
    "exp_inversion": "solid",
    "spherical_flip": "dashed",
    "spherical_mirror": "dotted",
    "linear_flip": "dashdot"
}

# # -----------------------------------------------------------
# # Plot 1 — Minimum HPR Error vs Camera Distance
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_dist["kernel"].unique():
#     subset = df_dist[df_dist["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("distance")
#         plt.plot(msub["distance"], msub["min_error"], marker="o", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("log")
# plt.title("Minimum HPR Error vs Camera Distance")
# plt.xlabel("Camera Distance (× mesh size)")
# plt.ylabel("Minimum Total Error (%)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# dist_plot_path = os.path.join(OUT_ROOT, "summary_min_error_vs_distance.png")
# plt.savefig(dist_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved distance summary plot → {dist_plot_path}")


# # -----------------------------------------------------------
# # Plot 2 — Minimum HPR Error vs Sampling Density
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_dens["kernel"].unique():
#     subset = df_dens[df_dens["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("density")
#         plt.plot(msub["density"], msub["min_error"], marker="s", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("log")
# plt.title("Minimum HPR Error vs Sampling Density")
# plt.xlabel("Number of Points")
# plt.ylabel("Minimum Total Error (%)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# dens_plot_path = os.path.join(OUT_ROOT, "summary_min_error_vs_density.png")
# plt.savefig(dens_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved density summary plot → {dens_plot_path}")

# # -----------------------------------------------------------
# # Plot 3 — Minimum HPR Error vs Relative Camera Distance
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_rel_dist["kernel"].unique():
#     subset = df_rel_dist[df_rel_dist["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("relative_distance")
#         plt.plot(msub["relative_distance"], msub["min_error"], marker="o", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("linear")
# plt.title("Minimum HPR Error vs Relative Camera Distance")
# plt.xlabel("Relative Camera Distance ")
# plt.ylabel("Minimum Total Error (%)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# rel_dist_plot_path = os.path.join(OUT_ROOT, "summary_min_error_vs_rel_distance.png")
# plt.savefig(rel_dist_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved distance summary plot → {rel_dist_plot_path}")


# # -----------------------------------------------------------
# # Plot 1 — Gamma at Minimum HPR Error vs Camera Distance
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_dist["kernel"].unique():
#     subset = df_dist[df_dist["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("distance")
#         plt.plot(msub["distance"], msub["gamma_at_min"], marker="o", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("log")
# plt.yscale("log")
# plt.title("Gamma at Minimum HPR Error vs Camera Distance")
# plt.xlabel("Camera Distance (× mesh size)")
# plt.ylabel("Gamma at Minimum Total Error (0-1)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# dist_plot_path = os.path.join(OUT_ROOT, "summary_gamma_at_min_error_vs_distance.png")
# plt.savefig(dist_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved distance summary plot → {dist_plot_path}")


# # -----------------------------------------------------------
# # Plot 2 — Gamma at Minimum HPR Error vs Sampling Density
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_dens["kernel"].unique():
#     subset = df_dens[df_dens["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("density")
#         plt.plot(msub["density"], msub["gamma_at_min"], marker="s", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("log")
# plt.yscale("log")
# plt.title("Gamma at Minimum HPR Error vs Sampling Density")
# plt.xlabel("Number of Points")
# plt.ylabel("Gamma at Minimum Total Error (0-1)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# dens_plot_path = os.path.join(OUT_ROOT, "summary_gamma_at_min_error_vs_density.png")
# plt.savefig(dens_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved gamma density summary plot → {dens_plot_path}")

# # -----------------------------------------------------------
# # Plot 3 — Gamma at Minimum HPR Error vs Relative Camera Distance
# # -----------------------------------------------------------
# plt.figure(figsize=(8, 6))
# for kernel in df_rel_dist["kernel"].unique():
#     subset = df_rel_dist[df_rel_dist["kernel"] == kernel]
#     for mesh in subset["mesh"].unique():
#         msub = subset[subset["mesh"] == mesh].sort_values("relative_distance")
#         plt.plot(msub["relative_distance"], msub["gamma_at_min"], marker="o", linewidth=2, color=MESH_COLORS.get(mesh, "black"), linestyle=LINESTYLES.get(kernel, "solid"), label=f"{mesh}-{kernel}")

# plt.xscale("linear")
# plt.yscale("log")
# plt.title("Gamma at Minimum HPR Error vs Relative Camera Distance")
# plt.xlabel("Relative Camera Distance ")
# plt.ylabel("Gamma at Minimum Total Error (%)")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()

# rel_dist_plot_path = os.path.join(OUT_ROOT, "summary_gamma_at_min_vs_rel_distance.png")
# plt.savefig(rel_dist_plot_path, dpi=300)
# plt.close()
# print(f"✅ Saved gamma distance summary plot → {rel_dist_plot_path}")


# -----------------------------------------------------------
# Helper: create combined figure (two subplots, one per kernel)
# -----------------------------------------------------------
def plot_combined(df, variable, out_name, value_field="min_error"):
    """
    Creates a figure with two stacked subplots (one per kernel).
    When value_field='gamma_at_min', plots γ at minimum error instead of the error itself.
    """
    kernels = sorted(df["kernel"].unique())
    if len(kernels) == 0:
        print("⚠️ No kernels found, skipping plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for ax, kernel in zip(axes, kernels):
        subset = df[df["kernel"] == kernel]
        for mesh in sorted(subset["mesh"].unique()):
            msub = subset[subset["mesh"] == mesh].sort_values(variable)
            ax.plot(
                msub[variable],
                msub[value_field],
                marker="o",
                linewidth=2,
                label=mesh
            )

        # Decide x-axis scaling
        if variable in ("distance", "density"):
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        # Decide y-axis scaling
        if value_field == "gamma_at_min":
            ax.set_yscale("log")

            ax.set_ylabel("γ at Min Error (0–1)")
        else:
            ax.set_ylabel("Minimum Total Error (%)")

        ax.set_title(f"Kernel: {kernel}")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend()

    # --- Shared X label ---
    if variable == "distance":
        xlabel = "Camera Distance (× mesh size)"
    elif variable == "relative_distance":
        xlabel = "Relative Camera Distance"
    elif variable == "density":
        xlabel = "Number of Points"
    else:
        xlabel = variable.capitalize()

    axes[-1].set_xlabel(xlabel)

    # --- Figure title ---
    label_name = variable.replace("_", " ").capitalize()
    if value_field == "gamma_at_min":
        plt.suptitle(f"γ at Minimum HPR Error vs {label_name}\n(differentiated by kernel type)")
    else:
        plt.suptitle(f"Minimum HPR Error vs {label_name}\n(differentiated by kernel type)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_ROOT, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved combined kernel comparison → {out_path}")


# -----------------------------------------------------------
# Generate the six summary plots (error + γ@min for each variable)
# -----------------------------------------------------------
# if not df_dist.empty:
#     plot_combined(df_dist, "distance", "summary_min_error_vs_distance_combined.png", value_field="min_error")
#     plot_combined(df_dist, "distance", "summary_gamma_at_min_error_vs_distance_combined.png", value_field="gamma_at_min")

# if not df_dens.empty:
#     plot_combined(df_dens, "density", "summary_min_error_vs_density_combined.png", value_field="min_error")
#     plot_combined(df_dens, "density", "summary_gamma_at_min_error_vs_density_combined.png", value_field="gamma_at_min")

# if not df_rel_dist.empty:
#     plot_combined(df_rel_dist, "relative_distance", "summary_min_error_vs_rel_distance_combined.png", value_field="min_error")
#     plot_combined(df_rel_dist, "relative_distance", "summary_gamma_at_min_error_vs_rel_distance_combined.png", value_field="gamma_at_min")


# -----------------------------------------------------------
# Plot error-vs-gamma curves for each density and distance
# -----------------------------------------------------------
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def collect_gamma_curves(results_dir, variable_name, target_meshes=None):
    """
    Walk through the benchmark directory (density or distance)
    and collect gamma vs TOTAL_rate curves.
    Returns DataFrame with columns:
      mesh, kernel, variable (density/distance), gamma, TOTAL_rate, FP_rate, FN_rate
    """
    rows = []
    folders = sorted([f for f in os.listdir(results_dir)
                      if os.path.isdir(os.path.join(results_dir, f))])
    for folder in folders:
        mesh_name, rel_distance, abs_distance, kernel = parse_folder_name(folder)
        if mesh_name is None or kernel is None:
            continue
        if target_meshes and mesh_name.lower() not in target_meshes:
            continue

        if variable_name == "density":
            variable_value = abs_distance or rel_distance
        elif variable_name == "distance":
            variable_value = abs_distance
        else:
            continue

        if variable_value is None:
            continue

        folder_path = os.path.join(results_dir, folder)
        csv_path = os.path.join(folder_path, f"benchmark_{kernel}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if not {"gamma", "TOTAL_rate"} <= set(df.columns):
            continue

        df = df[["gamma", "TOTAL_rate"]].copy()
        df["mesh"] = mesh_name.lower()
        df["kernel"] = kernel
        df[variable_name] = float(variable_value)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["mesh","kernel",variable_name,"gamma","TOTAL_rate"])
    return pd.concat(rows, ignore_index=True)

def plot_error_vs_gamma_grid(df, variable_name, out_name_prefix):
    """
    For each (mesh, kernel), plot a grid of error-vs-gamma curves,
    one per variable value (density or distance).
    """
    if df.empty:
        print(f"⚠️ No data to plot for {variable_name}.")
        return

    for (mesh, kernel), g in df.groupby(["mesh","kernel"]):
        g = g.sort_values(variable_name)
        unique_vals = sorted(g[variable_name].unique())
        nvals = len(unique_vals)

        plt.figure(figsize=(10, 6))
        for val in unique_vals:
            sub = g[g[variable_name] == val].sort_values("gamma")
            label = f"{variable_name}={val:g}"
            plt.plot(
                sub["gamma"],
                sub["TOTAL_rate"],
                linewidth=2,
                marker="o",
                color=MESH_COLORS.get(mesh, "black"),
                linestyle=LINESTYLES.get(kernel, "solid"),
                alpha=0.7,
                label=label
            )

        plt.xscale("log")
        plt.xlabel("Gamma (log scale)")
        plt.ylabel("Total Error Rate")
        plt.title(f"{mesh.title()} — {kernel}\nError vs Gamma for different {variable_name}")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(
            OUT_ROOT, f"{out_name_prefix}_{mesh}_{kernel}.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved {variable_name} gamma curve grid → {out_path}")

# -----------------------------------------------------------
# Run for all meshes and kernels across density & distance
# -----------------------------------------------------------
TARGET_MESHES = ["buddha", "erato", "dragon", "bunny", "hairball"]  # modify if you want others

# 1. Density-based curves
df_dens_curves = collect_gamma_curves(DENSITY_DIR, "density", target_meshes=[m.lower() for m in TARGET_MESHES])
plot_error_vs_gamma_grid(df_dens_curves, "density", "error_vs_gamma_by_density")

# 2. Distance-based curves
df_dist_curves = collect_gamma_curves(DISTANCE_DIR, "distance", target_meshes=[m.lower() for m in TARGET_MESHES])
plot_error_vs_gamma_grid(df_dist_curves, "distance", "error_vs_gamma_by_distance")
